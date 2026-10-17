# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extraction store: text-unit records reused across runs and documents.

Each record is stored under ``<fingerprint>:<text hash>``
(``contributions.record_key``) in the vector store's extraction-records index
of the unit's suffix (``VectorIndexer.read_extraction_records``). A unit whose
text and extraction settings are unchanged takes the stored record instead of
being extracted again, so an edited document re-extracts only the chunks that
changed. The store is a cache: a read or write that fails is logged and the
run goes on (every unit read nothing is extracted; nothing written is
extracted again next time).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from unified_kg_rag.domain.ingestion.contributions import record_key
from unified_kg_rag.domain.models import Config, TextUnit
from unified_kg_rag.ports.indexer import BaseIndexer
from unified_kg_rag.shared import get_logger

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
        parse (another release wrote it), is absent: it is extracted."""
        store = self._store() if self._reuse else None
        if store is None:
            return {}
        found: dict[str, T] = {}
        for suffix, by_key in self._keys(units, fingerprint).items():
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
    ) -> int:
        """Store payloads for their units; returns the number written."""
        items = list(items)
        store = self._store()
        if store is None or not items:
            return 0
        payload_by_unit = {unit.id: payload for unit, payload in items}
        written = 0
        for suffix, by_key in self._keys(
            [unit for unit, _ in items], fingerprint
        ).items():
            records = {
                key: payload_by_unit[units[0].id] for key, units in by_key.items()
            }
            try:
                stats = store.write_extraction_records(records, suffix=suffix)
            except Exception as e:  # noqa: BLE001 - the store is a cache
                logger.warning(
                    "Writing the extraction records of '%s' failed; their text "
                    "units are extracted again next time: %s",
                    suffix,
                    e,
                )
                continue
            if stats.failed_items:
                logger.warning(
                    "Writing %s of %s extraction records of '%s' failed; their "
                    "text units are extracted again next time: %s",
                    stats.failed_items,
                    len(records),
                    suffix,
                    "; ".join(stats.errors[:3]),
                )
            written += stats.successful_items
        return written
