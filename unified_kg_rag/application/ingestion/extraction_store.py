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
from collections.abc import Callable, Collection, Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from unified_kg_rag.domain.ingestion.cache_keys import unit_record_fingerprint
from unified_kg_rag.domain.ingestion.contributions import (
    RECORD_KEYS_ATTRIBUTE,
    record_key,
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
    """The text extraction reads: the target-language translation, if any."""
    if unit.translated_texts:
        target = config.processing.translation.target_language.value
        return unit.translated_texts.get(target, unit.text)
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
    extraction stages stored the records under.
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
        keys: dict[str, str] = {}
        if unit.id in gleaned and unit.id not in graph_failed:
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
    ) -> dict[str, T]:
        """The stored records of ``units`` by text unit id, each payload
        ``parse``d for its unit. A unit without one, or whose payload does not
        parse (another release wrote it), is absent: it is extracted.

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
        return found

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
