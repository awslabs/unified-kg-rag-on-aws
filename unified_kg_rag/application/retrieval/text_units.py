# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stored text units by id: the one read behind the chain's citations,
the MCP server's record documents and the evaluator's file attribution.

Ids are read in batches one search returns in full (``id_batch_size`` of
the text-units index, ``aindex_limits``), at most
``search.max_concurrency`` at once, under the caller's filters the index
declares (a unit they exclude is never named). The limits are read again
after the fetch: a reindex that lowered them meanwhile may have cut a
batch, so only the misses of a batch that answered and still fits are
settled (``TextUnitFetch.settled``) — a cache may keep those.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.retrieval.base import scope_filters
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchType,
)
from unified_kg_rag.domain.retrieval.filters import FilterFields
from unified_kg_rag.shared.utils.concurrency import gather_bounded

# What a caller needs of a stored unit unless it asks for more.
DEFAULT_FIELDS = ("id", "document_ids", "attributes")


class _IndexLimits(Protocol):
    @property
    def id_batch_size(self) -> int: ...


class TextUnitReader(Protocol):
    """The document retriever's reads a text-unit fetch uses."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]: ...

    async def aindex_limits(self, prefix: str, suffix: str | None) -> _IndexLimits: ...

    def filter_fields_for(self, prefixes: Collection[str]) -> FilterFields | None: ...


class TextUnitFetch(BaseModel):
    """The units a fetch found, and the ids whose answer is final."""

    units: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Each found unit's stored metadata, by its id, in the "
        "order asked",
    )
    settled: set[str] = Field(
        default_factory=set,
        description="The ids asked whose answer is final: found, or missing "
        "from a batch that answered and fit the limits read after the fetch",
    )


async def fetch_text_units(
    retriever: TextUnitReader,
    config: Config,
    suffix: str | None,
    ids: Iterable[str],
    *,
    filters: dict[str, Any] | None = None,
    fields: Iterable[str] | None = DEFAULT_FIELDS,
) -> TextUnitFetch:
    """The stored text units of ``ids`` in namespace ``suffix`` that pass
    ``filters`` (the keys the text-units index declares; the others name
    fields it does not hold). ``fields`` are the stored fields returned
    (``None``: all). Raises what the retriever raises."""
    prefix = config.indexing.opensearch.text_units_index_prefix
    wanted = list(dict.fromkeys(str(i) for i in ids if i))
    if not wanted:
        return TextUnitFetch()
    applied: dict[str, Any] = {}
    if filters:
        declared = retriever.filter_fields_for([prefix])
        applied = {
            key: value
            for key, value in filters.items()
            if declared is None or declared.declares(key)
        }
    size = max(1, (await retriever.aindex_limits(prefix, suffix)).id_batch_size)
    batches = [wanted[i : i + size] for i in range(0, len(wanted), size)]
    returned = list(fields) if fields is not None else None
    parts = await gather_bounded(
        config.search.max_concurrency,
        (
            retriever.aretrieve(
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=len(batch),
                    index_prefixes=[prefix],
                    suffix=suffix,
                    filters=scope_filters(applied, id=batch),
                    source_fields=returned,
                )
            )
            for batch in batches
        ),
    )
    asked = set(wanted)
    found: dict[str, dict[str, Any]] = {}
    for hits in parts:
        for hit in hits:
            metadata = hit.metadata or {}
            unit_id = str(metadata.get("id") or hit.source)
            if unit_id in asked:
                found.setdefault(unit_id, metadata)
    after = max(1, (await retriever.aindex_limits(prefix, suffix)).id_batch_size)
    settled = set(found)
    for batch, hits in zip(batches, parts, strict=True):
        if hits and len(batch) <= after:
            settled.update(batch)
    return TextUnitFetch(
        units={u: found[u] for u in wanted if u in found}, settled=settled
    )
