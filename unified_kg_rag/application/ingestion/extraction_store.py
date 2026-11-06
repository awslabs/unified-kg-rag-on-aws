# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extraction store: text-unit records reused across runs and documents.

Each record is stored under ``<fingerprint>:<text hash>``
(``contributions.record_key``) in the vector store's extraction-records index
of the unit's suffix (``VectorIndexer.read_extraction_records``). A unit whose
text and extraction settings are unchanged takes the stored record instead of
being extracted again, so an edited document re-extracts only the chunks that
changed. Reading is a cache: a read that fails is logged and every unit it
read nothing for is extracted. Writing is not: an incremental run derives the
stored items' content from the records of the text units they cite
(``domain.ingestion.derivation``), so :meth:`ExtractionStore.save` returns the
units whose record it did not store and the stage fails them (their documents
are recorded FAILED and extracted again). A text unit is written with the
keys of its stored records (:func:`with_record_keys`), computed from the
run's records rather than set on the unit by the stages, so a run resumed
after the extraction stages writes the same keys.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from unified_kg_rag.domain.ingestion.cache_keys import (
    index_language,
    unit_record_fingerprint,
)
from unified_kg_rag.domain.ingestion.contributions import (
    CHUNKS_RECORD_KIND,
    RECORD_KEYS_ATTRIBUTE,
    record_key,
    record_keys,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    SEED_RECORD_FINGERPRINT,
    is_seed_unit,
)
from unified_kg_rag.domain.models import Config, TextUnit, TextUnitExtraction
from unified_kg_rag.ports.indexer import BaseIndexer
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import (
    NamespaceLeaseError,
    StoreConnectionError,
    StoreRequestRejectedError,
)

if TYPE_CHECKING:
    from unified_kg_rag.ports.indexer import VectorIndexer

logger = get_logger(__name__)
T = TypeVar("T")


def processed_text(config: Config, unit: TextUnit) -> str:
    """The text extraction reads: its translation into the index language
    (``index_language``), if any."""
    if unit.translated_texts:
        return unit.translated_texts.get(index_language(config), unit.text)
    return unit.text


def with_record_keys(
    config: Config,
    units: Iterable[TextUnit],
    records: Iterable[TextUnitExtraction],
    *,
    graph_failed: Collection[str],
    claims: bool,
    claims_failed: Collection[str] = (),
) -> list[TextUnit]:
    """Copies of ``units`` naming the store keys of their stored records.

    A unit names its graph record unless it has none or is in
    ``graph_failed`` (its extraction failed or its record was not stored),
    and, when the run extracted ``claims``, its claims record unless it is
    in ``claims_failed`` too. The keys follow from the record's settings and
    the unit's text (``ExtractionStore.key``), so they are the ones the
    extraction stages stored the records under. A seed graph's unit names
    its graph record only, under the seed fingerprint (it has no claims).
    The record of its pre-chunk's boundary answer, which the chunker named,
    stays named.
    """
    gleaned = {r.text_unit_id: r.gleaned for r in records}
    # The fingerprint depends only on the settings: one per (gleaned, claims).
    fingerprints: dict[tuple[bool, bool], str] = {}

    def fingerprint(was_gleaned: bool, of_claims: bool) -> str:
        key = (was_gleaned, of_claims)
        if key not in fingerprints:
            fingerprints[key] = unit_record_fingerprint(
                config, gleaned=was_gleaned, claims=of_claims
            )
        return fingerprints[key]

    keyed: list[TextUnit] = []
    for unit in units:
        keys: dict[str, str] = {
            kind: key
            for kind, key in record_keys(unit).items()
            if kind == CHUNKS_RECORD_KIND
        }
        if is_seed_unit(unit.id):
            if unit.id in gleaned and unit.id not in graph_failed:
                keys["graph"] = record_key(SEED_RECORD_FINGERPRINT, unit.text)
        elif unit.id in gleaned and unit.id not in graph_failed:
            text = processed_text(config, unit)
            keys["graph"] = record_key(fingerprint(gleaned[unit.id], False), text)
            if claims and unit.id not in claims_failed:
                keys["claims"] = record_key(fingerprint(gleaned[unit.id], True), text)
        attributes = {
            k: v
            for k, v in (unit.attributes or {}).items()
            if k != RECORD_KEYS_ATTRIBUTE
        }
        if keys:
            attributes[RECORD_KEYS_ATTRIBUTE] = keys
        keyed.append(unit.model_copy(update={"attributes": attributes}))
    return keyed


class ExtractionStore:
    """Read and write text units' records through a vector indexer."""

    def __init__(
        self,
        config: Config,
        build_indexer: Callable[[], VectorIndexer],
        *,
        reuse: bool = True,
    ) -> None:
        """``reuse=False`` reads nothing (every unit is extracted) and still
        writes, refreshing the stored records."""
        self.config = config
        self._build_indexer = build_indexer
        self._reuse = reuse
        self._indexer: VectorIndexer | None = None
        self._unavailable = False

    def _store(self) -> VectorIndexer | None:
        if self._indexer is None and not self._unavailable:
            try:
                self._indexer = self._build_indexer()
            except Exception as e:  # noqa: BLE001 - the store is a cache
                logger.warning(
                    "Extraction store unavailable, extracting every text unit: %s", e
                )
                self._unavailable = True
        return self._indexer

    def close(self) -> None:
        if self._indexer is not None:
            self._indexer.close()
            self._indexer = None

    def key(self, unit: TextUnit, fingerprint: str) -> str:
        """The store key of ``unit``'s record under ``fingerprint``."""
        return record_key(fingerprint, processed_text(self.config, unit))

    def _keys(
        self, units: Iterable[TextUnit], fingerprint: str
    ) -> dict[str, dict[str, list[TextUnit]]]:
        """suffix -> key -> the units with that key."""
        grouped: dict[str, dict[str, list[TextUnit]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for unit in units:
            grouped[BaseIndexer.get_suffix(unit)][self.key(unit, fingerprint)].append(
                unit
            )
        return grouped

    def load(
        self,
        units: Iterable[TextUnit],
        fingerprint: str,
        parse: Callable[[dict[str, Any], TextUnit], T],
        legacy_fingerprints: Collection[str] = (),
    ) -> dict[str, T]:
        """The stored records of ``units`` by text unit id, each payload
        ``parse``d for its unit. A unit without one, or whose payload does not
        parse (another release wrote it), is absent: it is extracted.

        A unit without a record under ``fingerprint`` is looked up under each
        of ``legacy_fingerprints`` (the keys an earlier version stored the
        same record under: ``cache_keys.legacy_unit_record_fingerprints``),
        and a record found there is stored under ``fingerprint`` too, so the
        key its text unit names (:func:`with_record_keys`) holds it. One whose
        copy is not stored is absent (extracted and stored again).

        Each suffix's store is created first (``prepare_extraction_records``),
        reused or not: one the backend refuses to create raises
        ``StoreRequestRejectedError`` here, before extraction, rather than
        failing every record write after it; an unreachable store
        (``StoreConnectionError``) and a lost write lease raise too. Any
        other failure leaves the store to its reads and writes (it is a
        cache)."""
        store = self._store()
        if store is None:
            return {}
        grouped = self._keys(units, fingerprint)
        for suffix in grouped:
            try:
                store.prepare_extraction_records(suffix)
            except (
                StoreRequestRejectedError,
                StoreConnectionError,
                NamespaceLeaseError,
            ):
                raise
            except Exception as e:  # noqa: BLE001 - the store is a cache
                logger.warning(
                    "Preparing the extraction store of '%s' failed: %s", suffix, e
                )
        if not self._reuse:
            return {}
        found: dict[str, T] = {}
        for suffix, by_key in grouped.items():
            try:
                payloads = store.read_extraction_records(sorted(by_key), suffix=suffix)
            except Exception as e:  # noqa: BLE001 - the store is a cache
                logger.warning(
                    "Reading the extraction store of '%s' failed; extracting its "
                    "text units: %s",
                    suffix,
                    e,
                )
                continue
            self._parse_into(found, payloads, by_key, parse)
            missing = [
                u for units in by_key.values() for u in units if u.id not in found
            ]
            if missing and legacy_fingerprints:
                self._load_legacy(
                    store,
                    suffix,
                    missing,
                    fingerprint,
                    legacy_fingerprints,
                    parse,
                    found,
                )
        return found

    @staticmethod
    def _parse_into(
        found: dict[str, T],
        payloads: Mapping[str, dict[str, Any]],
        by_key: Mapping[str, list[TextUnit]],
        parse: Callable[[dict[str, Any], TextUnit], T],
    ) -> list[TextUnit]:
        """Parse ``payloads`` into ``found`` for the units of their key;
        returns the units parsed."""
        parsed: list[TextUnit] = []
        for key, payload in payloads.items():
            for unit in by_key.get(key, []):
                try:
                    found[unit.id] = parse(payload, unit)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Ignoring an unreadable extraction record of text "
                        "unit '%s': %s",
                        unit.short_id or unit.id,
                        type(e).__name__,
                    )
                else:
                    parsed.append(unit)
        return parsed

    def _load_legacy(
        self,
        store: VectorIndexer,
        suffix: str,
        units: list[TextUnit],
        fingerprint: str,
        legacy_fingerprints: Collection[str],
        parse: Callable[[dict[str, Any], TextUnit], T],
        found: dict[str, T],
    ) -> None:
        """Read ``units``' records under ``legacy_fingerprints`` into
        ``found`` and store each under ``fingerprint`` (``load``)."""
        copies: dict[str, dict[str, Any]] = {}
        copied: dict[str, list[TextUnit]] = defaultdict(list)
        for legacy in legacy_fingerprints:
            pending = [u for u in units if u.id not in found]
            if not pending:
                break
            by_key: dict[str, list[TextUnit]] = defaultdict(list)
            for unit in pending:
                by_key[self.key(unit, legacy)].append(unit)
            try:
                payloads = store.read_extraction_records(sorted(by_key), suffix=suffix)
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - the store is a cache
                logger.warning(
                    "Reading the extraction store of '%s' failed: %s", suffix, e
                )
                return
            for unit in self._parse_into(found, payloads, by_key, parse):
                key = self.key(unit, fingerprint)
                copies[key] = payloads[self.key(unit, legacy)]
                copied[key].append(unit)
        if not copies:
            return
        try:
            stats = store.write_extraction_records(copies, suffix=suffix)
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - the units are extracted again
            logger.warning(
                "Storing the earlier-keyed extraction records of '%s' under "
                "their current keys failed; extracting their text units: %s",
                suffix,
                e,
            )
            failed = set(copies)
        else:
            failed = (
                set(copies)
                if stats.unattributed_failures
                else set(stats.failed_ids) & copies.keys()
            )
            if failed:
                logger.warning(
                    "Storing %s of %s earlier-keyed extraction records of '%s' "
                    "under their current keys failed; extracting their text "
                    "units: %s",
                    len(failed),
                    len(copies),
                    suffix,
                    "; ".join(stats.errors[:3]),
                )
        for key in failed:
            for unit in copied[key]:
                found.pop(unit.id, None)
        if reused := sum(len(copied[key]) for key in copies.keys() - failed):
            logger.info(
                "Extraction store of '%s': %s text units reused from records "
                "stored under an earlier key, now stored under the current one",
                suffix,
                reused,
            )

    def read_records(
        self, keys: Mapping[str, Collection[str]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """The stored records of ``keys`` by suffix and key, read as a cache:
        none without reuse, and a suffix whose read fails reads none
        (``BoundaryRecords``: the chunker's pre-chunk answers)."""
        store = self._store()
        if store is None or not self._reuse:
            return {}
        found: dict[str, dict[str, dict[str, Any]]] = {}
        for suffix, wanted in keys.items():
            try:
                found[suffix] = store.read_extraction_records(
                    sorted(wanted), suffix=suffix
                )
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - the store is a cache
                logger.warning(
                    "Reading the extraction store of '%s' failed: %s", suffix, e
                )
        return found

    def write_records(
        self, records: Mapping[str, Mapping[str, dict[str, Any]]]
    ) -> None:
        """Store ``records`` (by suffix, then key), best-effort: a record not
        stored is made again by a later run (``BoundaryRecords``)."""
        store = self._store()
        if store is None:
            return
        for suffix, by_key in records.items():
            try:
                stats = store.write_extraction_records(dict(by_key), suffix=suffix)
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - the store is a cache
                logger.warning(
                    "Writing the extraction records of '%s' failed: %s", suffix, e
                )
                continue
            if stats.failed_items:
                logger.warning(
                    "Writing %s of %s extraction records of '%s' failed: %s",
                    stats.failed_items,
                    len(by_key),
                    suffix,
                    "; ".join(stats.errors[:3]),
                )

    def save(
        self, items: Iterable[tuple[TextUnit, dict[str, Any]]], fingerprint: str
    ) -> set[str]:
        """Store payloads for their units; returns the ids of the units whose
        record was not stored (the store is unavailable or the write failed)."""
        items = list(items)
        if not items:
            return set()
        store = self._store()
        if store is None:
            return {unit.id for unit, _ in items}
        payload_by_unit = {unit.id: payload for unit, payload in items}
        unsaved: set[str] = set()
        for suffix, by_key in self._keys(
            [unit for unit, _ in items], fingerprint
        ).items():
            records = {
                key: payload_by_unit[units[0].id] for key, units in by_key.items()
            }
            try:
                stats = store.write_extraction_records(records, suffix=suffix)
            except Exception as e:  # noqa: BLE001 - the caller fails the units
                logger.warning(
                    "Writing the extraction records of '%s' failed: %s", suffix, e
                )
                unsaved.update(u.id for units in by_key.values() for u in units)
                continue
            if stats.failed_items:
                logger.warning(
                    "Writing %s of %s extraction records of '%s' failed: %s",
                    stats.failed_items,
                    len(records),
                    suffix,
                    "; ".join(stats.errors[:3]),
                )
                # A failure without the key it is of fails every unit.
                failed = (
                    set(by_key)
                    if stats.unattributed_failures
                    else set(stats.failed_ids)
                )
                unsaved.update(
                    u.id for key in failed & by_key.keys() for u in by_key[key]
                )
        return unsaved
