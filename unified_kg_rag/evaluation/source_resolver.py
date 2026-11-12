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
from typing import Protocol

from unified_kg_rag.application.retrieval.text_units import (
    TextUnitReader,
    fetch_text_units,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


class SourceFileResolver(Protocol):
    """Maps text-unit ids to the file name of the document they were cut from."""

    async def aresolve(
        self, text_unit_ids: Iterable[str], suffix: str | None
    ) -> dict[str, str]:
        """Return ``{text_unit_id: file_name}`` for the ids that resolve."""
        ...


class TextUnitFileResolver:
    """Resolve text-unit ids to file names via a document retriever
    (``text_units.fetch_text_units``), caching each settled answer per
    index suffix."""

    def __init__(
        self, config: Config, retriever_provider: Callable[[], TextUnitReader]
    ) -> None:
        self._config = config
        self._retriever_provider = retriever_provider
        # (suffix, text_unit_id) -> file name, or None when the id is unknown.
        self._cache: dict[tuple[str | None, str], str | None] = {}

    async def aresolve(
        self, text_unit_ids: Iterable[str], suffix: str | None
    ) -> dict[str, str]:
        wanted = list(dict.fromkeys(str(i) for i in text_unit_ids if i))
        missing = [i for i in wanted if (suffix, i) not in self._cache]
        if missing:
            fetched = await fetch_text_units(
                self._retriever_provider(),
                self._config,
                suffix,
                missing,
                fields=None,
            )
            # A miss is cached only once settled: from a batch that itself
            # answered with a hit (a failed request degrades to none, as a
            # batch of unknown ids does) and that fit the limits read after
            # the fetch (a reindex lowered them: the hits were cut).
            for unit_id in missing:
                if unit_id in fetched.settled:
                    metadata = fetched.units.get(unit_id)
                    self._cache[(suffix, unit_id)] = (
                        source_fields.file_name(metadata) if metadata else None
                    )
        return {
            unit_id: name
            for unit_id in wanted
            if (name := self._cache.get((suffix, unit_id)))
        }
