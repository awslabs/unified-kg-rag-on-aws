# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller filters are applied only to the indices / vertex labels that have the field.

Strategies forward ``SearchQuery.filters`` to every sub-query (entities, claims,
relationships, community reports, text units, fetch-by-id hops). A
``term``/``terms`` clause on a field an index does not map matches nothing, and
a Gremlin ``has(key, ...)`` drops every vertex lacking ``key``, so the
retrievers scope each caller key to where it exists. All AWS-free: retrievers
are built via ``__new__`` with fake clients.
"""

from __future__ import annotations

from typing import Any

import pytest

from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config, SearchQuery, SearchType
from unified_kg_rag.shared import AWSServiceError

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# OpenSearch
# --------------------------------------------------------------------------- #


class _FakeOpenSearchClient:
    """Serves a per-alias mapping and records every search body by index."""

    def __init__(
        self,
        mappings: dict[str, dict[str, Any]] | None = None,
        mapping_error: Exception | None = None,
    ) -> None:
        self._mappings = mappings or {}
        self._mapping_error = mapping_error
        self.mapping_calls: list[str] = []
        self.searches: dict[str, dict[str, Any]] = {}

    async def aget_mapping(self, index: str) -> dict[str, Any]:
        self.mapping_calls.append(index)
        if self._mapping_error is not None:
            raise self._mapping_error
        # Every index this package creates declares at least ``id``.
        properties = {"id": {"type": "keyword"}, **self._mappings.get(index, {})}
        return {f"{index}-20260101000000": {"mappings": {"properties": properties}}}

    async def asearch(self, index: str, body: dict[str, Any], **_: Any) -> dict:
        self.searches[index] = body
        return {"hits": {"hits": []}}


def _retriever(config: Config, client: _FakeOpenSearchClient) -> OpenSearchRetriever:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_opensearch_config", config.indexing.opensearch)
    object.__setattr__(inst, "_max_size", config.indexing.opensearch.max_query_size)
    object.__setattr__(inst, "_max_result_window", 10000)
    object.__setattr__(
        inst, "_terms_batch_size", config.indexing.opensearch.terms_batch_size
    )
    object.__setattr__(
        inst, "_max_total_clauses", config.indexing.opensearch.max_total_clauses
    )
    object.__setattr__(
        inst, "_reserved_clauses", config.indexing.opensearch.reserved_clauses
    )
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())
    object.__setattr__(
        inst, "_static_filter_fields", inst._initialize_static_filter_fields()
    )
    object.__setattr__(inst, "_live_filter_fields", {})
    object.__setattr__(inst, "_opensearch_client", client)
    object.__setattr__(inst, "_record_timing", lambda *a, **k: None)
    object.__setattr__(inst, "_record_metric", lambda *a, **k: None)
    return inst


def _prefixes(config: Config) -> dict[str, str]:
    o = config.indexing.opensearch
    return {
        "text_units": o.text_units_index_prefix,
        "entities": o.entities_index_prefix,
        "relationships": o.relationships_index_prefix,
        "claims": o.claims_index_prefix,
        "reports": o.community_reports_index_prefix,
    }


def _alias(retriever: OpenSearchRetriever, prefix: str) -> str:
    return retriever._get_name(prefix, None)


def _filter_clauses(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The filter clauses of a lexical (``bool.filter``) or filter-only body."""
    query = body["query"]
    return query.get("bool", {}).get("filter", [])


def _filtered_fields(body: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    for clause in _filter_clauses(body):
        for kind in ("term", "terms", "range"):
            if kind in clause:
                fields |= set(clause[kind])
    return fields


async def test_key_in_one_index_filters_only_that_index(config: Config) -> None:
    p = _prefixes(config)
    probe = _retriever(config, _FakeOpenSearchClient())
    client = _FakeOpenSearchClient(
        mappings={
            _alias(probe, p["text_units"]): {"attr_category": {"type": "keyword"}},
        }
    )
    retriever = _retriever(config, client)

    await retriever.aretrieve(
        SearchQuery(
            query="supply terms",
            search_type=SearchType.LEXICAL,
            index_prefixes=list(p.values()),
            filters={"attr_category": "research"},
        )
    )

    assert _filtered_fields(client.searches[_alias(retriever, p["text_units"])]) == {
        "attr_category"
    }
    for name in ("entities", "relationships", "claims", "reports"):
        assert _filtered_fields(client.searches[_alias(retriever, p[name])]) == set()


async def test_statically_declared_key_needs_no_live_mapping(config: Config) -> None:
    # ``type`` is declared on entities and claims only; the other indices map
    # nothing dynamic, so it is dropped there.
    p = _prefixes(config)
    client = _FakeOpenSearchClient()
    retriever = _retriever(config, client)

    await retriever.aretrieve(
        SearchQuery(
            query="vendor",
            search_type=SearchType.LEXICAL,
            index_prefixes=list(p.values()),
            filters={"type": "PERSON"},
        )
    )

    for name in ("entities", "claims"):
        assert _filtered_fields(client.searches[_alias(retriever, p[name])]) == {"type"}
    for name in ("text_units", "relationships", "reports"):
        assert _filtered_fields(client.searches[_alias(retriever, p[name])]) == set()


async def test_fetch_by_id_keeps_ids_and_drops_inapplicable_caller_key(
    config: Config,
) -> None:
    p = _prefixes(config)
    client = _FakeOpenSearchClient()
    retriever = _retriever(config, client)

    await retriever.aretrieve(
        SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=2,
            index_prefixes=[p["text_units"]],
            filters={"type": "PERSON", "id": ["tu-1", "tu-2"]},
        )
    )

    clauses = _filter_clauses(client.searches[_alias(retriever, p["text_units"])])
    assert clauses == [{"terms": {"id": ["tu-1", "tu-2"]}}]


async def test_key_present_everywhere_filters_everywhere(config: Config) -> None:
    p = _prefixes(config)
    probe = _retriever(config, _FakeOpenSearchClient())
    client = _FakeOpenSearchClient(
        mappings={
            _alias(probe, prefix): {"attr_region": {"type": "keyword"}}
            for prefix in p.values()
        }
    )
    retriever = _retriever(config, client)

    await retriever.aretrieve(
        SearchQuery(
            query="terms",
            search_type=SearchType.LEXICAL,
            index_prefixes=list(p.values()),
            filters={"attr_region": "north"},
        )
    )

    for prefix in p.values():
        assert _filtered_fields(client.searches[_alias(retriever, prefix)]) == {
            "attr_region"
        }


async def test_nested_and_multi_field_paths_are_recognised(config: Config) -> None:
    p = _prefixes(config)
    probe = _retriever(config, _FakeOpenSearchClient())
    client = _FakeOpenSearchClient(
        mappings={
            _alias(probe, p["entities"]): {
                "attributes": {
                    "properties": {"source": {"type": "keyword"}},
                },
                "label": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            }
        }
    )
    retriever = _retriever(config, client)
    fields = await retriever._live_index_fields(_alias(retriever, p["entities"]))
    assert fields is not None
    assert {"attributes", "attributes.source", "label", "label.raw"} <= fields


async def test_unreadable_mapping_falls_back_to_static_allowlist(
    config: Config,
) -> None:
    p = _prefixes(config)
    client = _FakeOpenSearchClient(mapping_error=AWSServiceError("Read timed out"))
    retriever = _retriever(config, client)

    await retriever.aretrieve(
        SearchQuery(
            query="terms",
            search_type=SearchType.LEXICAL,
            index_prefixes=list(p.values()),
            filters={"attr_category": "research", "type": "PERSON"},
        )
    )

    def fields(name: str) -> set[str]:
        return _filtered_fields(client.searches[_alias(retriever, p[name])])

    # attr_* only where documents carry attribute fields; type where declared.
    assert fields("text_units") == {"attr_category"}
    assert fields("entities") == {"attr_category", "type"}
    assert fields("reports") == {"attr_category"}
    assert fields("claims") == {"type"}
    assert fields("relationships") == set()
    # A transient failure is not cached: the next query retries the read.
    assert retriever._live_filter_fields == {}


async def test_fatal_mapping_error_propagates(config: Config) -> None:
    client = _FakeOpenSearchClient(
        mapping_error=AWSServiceError("The security token included is invalid.")
    )
    retriever = _retriever(config, client)
    with pytest.raises(AWSServiceError, match="security token"):
        await retriever.aretrieve(
            SearchQuery(
                query="q",
                search_type=SearchType.LEXICAL,
                filters={"attr_category": "research"},
            )
        )


async def test_live_mapping_is_cached_per_alias(config: Config) -> None:
    p = _prefixes(config)
    client = _FakeOpenSearchClient()
    retriever = _retriever(config, client)
    query = SearchQuery(
        query="q",
        search_type=SearchType.LEXICAL,
        index_prefixes=[p["entities"]],
        filters={"type": "PERSON"},
    )
    await retriever.aretrieve(query)
    await retriever.aretrieve(query)
    assert client.mapping_calls == [_alias(retriever, p["entities"])]


async def test_unfiltered_query_reads_no_mapping(config: Config) -> None:
    client = _FakeOpenSearchClient()
    retriever = _retriever(config, client)
    await retriever.aretrieve(SearchQuery(query="q", search_type=SearchType.LEXICAL))
    assert client.mapping_calls == []


def test_static_allowlist_matches_indexer_mappings(config: Config) -> None:
    """The fallback allowlist must track the indexer's declared properties."""
    indexer = OpenSearchIndexer.__new__(OpenSearchIndexer)
    object.__setattr__(indexer, "config", config)
    object.__setattr__(indexer, "opensearch_config", config.indexing.opensearch)
    object.__setattr__(indexer, "_embedding_dimension", 8)
    object.__setattr__(
        indexer,
        "target_language",
        config.processing.translation.target_language.value,
    )
    object.__setattr__(indexer, "analyzer", "standard")
    retriever = _retriever(config, _FakeOpenSearchClient())
    p = _prefixes(config)
    mapping_funcs = {
        p["text_units"]: indexer._get_text_units_mapping,
        p["entities"]: indexer._get_entities_mapping,
        p["relationships"]: indexer._get_relationships_mapping,
        p["claims"]: indexer._get_claims_mapping,
        p["reports"]: indexer._get_community_reports_mapping,
    }
    for prefix, func in mapping_funcs.items():
        declared = {
            name
            for name, spec in func()["mappings"]["properties"].items()
            if spec.get("type") not in {"knn_vector", "object"}
        }
        assert declared <= retriever._static_filter_fields[prefix], prefix


# --------------------------------------------------------------------------- #
# Neptune
# --------------------------------------------------------------------------- #


class _RecordingTraversal:
    def __init__(self, calls: list[tuple]) -> None:
        self._calls = calls

    def __getattr__(self, name: str):
        def _step(*args, **kwargs):
            self._calls.append((name, args))
            return self

        return _step


def _neptune(config: Config) -> NeptuneRetriever:
    inst = NeptuneRetriever.__new__(NeptuneRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_neptune_config", config.indexing.neptune)
    object.__setattr__(inst, "_label_property_cache", {})
    return inst


def _with_properties(
    retriever: NeptuneRetriever, present: dict[str, set[str]]
) -> list[tuple[str, str]]:
    """Stub the per-label property probe; returns the probe call log."""
    probes: list[tuple[str, str]] = []

    async def _has(_g, label: str, key: str) -> bool:
        probes.append((label, key))
        return key in present.get(label, set())

    object.__setattr__(retriever, "_label_has_property", _has)
    return probes


async def test_neptune_drops_key_no_target_label_carries(config: Config) -> None:
    retriever = _neptune(config)
    _with_properties(retriever, {"Entity-default": {"type"}})
    filters, exempt = await retriever._scope_filters_to_labels(
        None,
        ["Entity-default"],
        {"id": ["e1"], "type": "PERSON", "attr_category": "research"},
    )
    assert filters == {"id": ["e1"], "type": "PERSON"}
    assert exempt == {}


async def test_neptune_exempts_labels_lacking_a_key(config: Config) -> None:
    retriever = _neptune(config)
    _with_properties(
        retriever, {"Entity-default": {"type"}, "Community-default": set()}
    )
    filters, exempt = await retriever._scope_filters_to_labels(
        None, ["Community-default", "Entity-default"], {"type": "PERSON"}
    )
    assert filters == {"type": "PERSON"}
    assert exempt == {"type": ["Community-default"]}

    calls: list[tuple] = []
    retriever._apply_filters(_RecordingTraversal(calls), filters, exempt)
    # Guarded: vertices of the exempt label pass, the rest must match.
    assert [c[0] for c in calls] == ["or_"]


async def test_neptune_key_on_every_label_is_a_plain_has(config: Config) -> None:
    retriever = _neptune(config)
    _with_properties(
        retriever, {"Entity-default": {"rank"}, "Community-default": {"rank"}}
    )
    filters, exempt = await retriever._scope_filters_to_labels(
        None, ["Community-default", "Entity-default"], {"rank": {"gte": 2}}
    )
    assert exempt == {}
    calls: list[tuple] = []
    retriever._apply_filters(_RecordingTraversal(calls), filters, exempt)
    assert [c[0] for c in calls] == ["has"]


async def test_neptune_property_probe_is_cached(config: Config) -> None:
    retriever = _neptune(config)
    calls: list[tuple] = []
    executed: list[Any] = []

    async def _execute(traversal):
        executed.append(traversal)
        return [1]

    object.__setattr__(retriever, "_execute_traversal", _execute)

    class _G:
        def V(self):  # noqa: N802 - Gremlin step name
            return _RecordingTraversal(calls)

    g = _G()
    assert await retriever._label_has_property(g, "Entity-default", "type")
    assert await retriever._label_has_property(g, "Entity-default", "type")
    assert len(executed) == 1
    assert ("hasLabel", ("Entity-default",)) in calls
    assert ("has", ("type",)) in calls


async def test_neptune_transient_probe_failure_keeps_filter(config: Config) -> None:
    retriever = _neptune(config)

    async def _execute(_traversal):
        raise AWSServiceError("Read timed out")

    object.__setattr__(retriever, "_execute_traversal", _execute)

    class _G:
        def V(self):  # noqa: N802 - Gremlin step name
            return _RecordingTraversal([])

    assert await retriever._label_has_property(_G(), "Entity-default", "type")
    assert retriever._label_property_cache == {}
