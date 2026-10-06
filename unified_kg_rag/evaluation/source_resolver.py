# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attribute graph-derived sources to the files they came from.

Only text units store a file name (``attributes.file_name``). Entities,
relationships and community reports carry ``text_unit_ids`` lineage instead
(community reports also carry ``document_ids``, but those are content hashes
that no dataset can reference). Without resolving that lineage, a ``global``
answer built only from community reports could never match a
``reference_sources`` file, so its retrieval metrics were 0 by construction.

``TextUnitFileResolver`` resolves text-unit ids to file names with id-batch
fetches from the document store (the same ``id`` terms-filter lookup local
search uses to fetch chunks), caching each id once per index suffix.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchType,
)
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


class SourceFileResolver(Protocol):
    """Maps text-unit ids to the file name of the document they were cut from."""

    async def aresolve(
        self, text_unit_ids: Iterable[str], suffix: str | None
    ) -> dict[str, str]:
        """Return ``{text_unit_id: file_name}`` for the ids that resolve."""
        ...


class _DocumentRetriever(Protocol):
    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]: ...


def file_name_of(payload: dict[str, Any]) -> str | None:
    """File name from a source/hit payload (top-level or under ``attributes``)."""
    candidates = [payload]
    if isinstance(payload.get("attributes"), dict):
        candidates.append(payload["attributes"])
    for candidate in candidates:
        for key in ("file_name", "file_path"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value.strip()).name
    return None


class TextUnitFileResolver:
    """Resolve text-unit ids to file names via a document retriever."""

    BATCH_SIZE = 500

    def __init__(
        self, config: Config, retriever_provider: Callable[[], _DocumentRetriever]
    ) -> None:
        self._index_prefix = config.indexing.opensearch.text_units_index_prefix
        self._retriever_provider = retriever_provider
        # (suffix, text_unit_id) -> file name, or None when the id is unknown.
        self._cache: dict[tuple[str | None, str], str | None] = {}

    async def aresolve(
        self, text_unit_ids: Iterable[str], suffix: str | None
    ) -> dict[str, str]:
        wanted = list(dict.fromkeys(str(i) for i in text_unit_ids if i))
        missing = [i for i in wanted if (suffix, i) not in self._cache]
        if missing:
            retriever = self._retriever_provider()
            for start in range(0, len(missing), self.BATCH_SIZE):
                batch = missing[start : start + self.BATCH_SIZE]
                found = await self._fetch(retriever, batch, suffix)
                for unit_id in batch:
                    self._cache[(suffix, unit_id)] = found.get(unit_id)
        return {
            unit_id: name
            for unit_id in wanted
            if (name := self._cache.get((suffix, unit_id)))
        }

    async def _fetch(
        self, retriever: _DocumentRetriever, batch: list[str], suffix: str | None
    ) -> dict[str, str]:
        query = SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(batch),
            index_prefixes=[self._index_prefix],
            suffix=suffix,
            filters={"id": batch},
        )
        found: dict[str, str] = {}
        for hit in await retriever.aretrieve(query):
            metadata = hit.metadata or {}
            unit_id = str(metadata.get("id") or hit.source)
            if name := file_name_of(metadata):
                found[unit_id] = name
        return found
