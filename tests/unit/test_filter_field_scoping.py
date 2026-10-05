# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller filters apply only to the indices / vertex labels that declare the field.

Strategies forward ``SearchQuery.filters`` to every sub-query (entities, claims,
relationships, community reports, text units, fetch-by-id hops). A
``term``/``terms`` clause on a field an index does not map matches nothing, and
a Gremlin ``has(key, ...)`` drops every vertex lacking ``key``, so the
retrievers scope each key to the stores ``filter_schema`` declares it on, and
the chain rejects a key no store declares. All AWS-free: retrievers are built
via ``__new__`` with fake clients.
"""

from __future__ import annotations

from typing import Any

import pytest

from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.adapters.storage.filter_schema import (
    NEPTUNE_FILTER_FIELDS,
    opensearch_filter_fields,
)
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    SearchQuery,
    SearchType,
    TextUnit,
)
from unified_kg_rag.shared import InvalidFilterError

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# OpenSearch
# --------------------------------------------------------------------------- #


class _FakeOpenSearchClient:
    """Records every search body by index."""

    def __init__(self) -> None:
        self.searches: dict[str, dict[str, Any]] = {}

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
        inst, "_index_filter_fields", inst._initialize_index_filter_fields()
    )
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


async def _search_all(
    config: Config, filters: dict[str, Any]
) -> tuple[OpenSearchRetriever, _FakeOpenSearchClient]:
    client = _FakeOpenSearchClient()
    retriever = _retriever(config, client)
    await retriever.aretrieve(
        SearchQuery(
            query="supply terms",
            search_type=SearchType.LEXICAL,
            index_prefixes=list(_prefixes(config).values()),
            filters=filters,
        )
    )
    return retriever, client


async def test_attribute_key_filters_only_attribute_bearing_indexes(
    config: Config,
) -> None:
    p = _prefixes(config)
    retriever, client = await _search_all(config, {"attr_category": "research"})

    def fields(name: str) -> set[str]:
        return _filtered_fields(client.searches[_alias(retriever, p[name])])

    for name in ("text_units", "entities", "reports"):
        assert fields(name) == {"attr_category"}, name
    for name in ("relationships", "claims"):
        assert fields(name) == set(), name


async def test_declared_key_filters_only_declaring_indexes(config: Config) -> None:
    # ``type`` is declared on entities and claims only.
    p = _prefixes(config)
    retriever, client = await _search_all(config, {"type": "PERSON"})

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


def test_opensearch_filter_fields(config: Config) -> None:
    declared = _retriever(config, _FakeOpenSearchClient()).filter_fields()
    for key in (
        "id",
        "type",
        "status",
        "community_id",
        "name.keyword",
        "attr_anything",
        "attributes.source",
    ):
        assert declared.declares(key), key
    for key in ("entity_type", "category", "attributes"):
        assert not declared.declares(key), key


def _indexer(config: Config) -> OpenSearchIndexer:
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
    return indexer


def test_schema_matches_indexer_mappings(config: Config) -> None:
    """Declared fields == the non-vector fields (incl. multi-fields) mapped."""
    indexer = _indexer(config)
    mapping_funcs = {
        "text_units": indexer._get_text_units_mapping,
        "entities": indexer._get_entities_mapping,
        "relationships": indexer._get_relationships_mapping,
        "claims": indexer._get_claims_mapping,
        "community_reports": indexer._get_community_reports_mapping,
    }
    schema = opensearch_filter_fields(
        config.processing.translation.target_language.value
    )
    assert set(mapping_funcs) == set(schema)
    for kind, func in mapping_funcs.items():
        mapped: set[str] = set()
        for name, spec in func()["mappings"]["properties"].items():
            if spec.get("type") in {"knn_vector", "object"}:
                continue
            mapped.add(name)
            mapped |= {f"{name}.{sub}" for sub in spec.get("fields", {})}
        assert schema[kind].fields == mapped, kind


def test_attribute_prefixes_match_documents_carrying_attributes(
    config: Config,
) -> None:
    """``attr_*`` / ``attributes.*`` are declared exactly where docs carry them."""
    indexer = _indexer(config)
    unit = TextUnit(
        id="tu-1",
        text="Vendor ships goods",
        attributes={"filters": {"category": "research"}},
    )
    doc = indexer._prepare_common_doc_properties(unit)
    assert {"attr_category", "attributes"} <= set(doc)
    schema = opensearch_filter_fields(
        config.processing.translation.target_language.value
    )
    attribute_kinds = {k for k, v in schema.items() if "attr_" in v.prefixes}
    # Documents built via _prepare_common_doc_properties.
    assert attribute_kinds == {"text_units", "entities", "community_reports"}


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
    return inst


def test_neptune_drops_key_no_target_label_declares() -> None:
    filters, exempt = NeptuneRetriever._scope_filters_to_labels(
        {"Entity-default": "entity"},
        {"id": ["e1"], "type": "PERSON", "status": "open"},
    )
    assert filters == {"id": ["e1"], "type": "PERSON"}
    assert exempt == {}


def test_neptune_drops_document_attribute_filters() -> None:
    # Vertices store attributes as raw JSON (``attr_filters``), so a document
    # attribute filter would empty graph expansion; it is scoped out instead.
    filters, exempt = NeptuneRetriever._scope_filters_to_labels(
        {"Community-default": "community", "Entity-default": "entity"},
        {"attr_region": "north", "id": ["e1"]},
    )
    assert filters == {"id": ["e1"]}
    assert exempt == {}


def test_neptune_exempts_labels_lacking_a_key() -> None:
    filters, exempt = NeptuneRetriever._scope_filters_to_labels(
        {"Community-default": "community", "Entity-default": "entity"},
        {"type": "PERSON"},
    )
    assert filters == {"type": "PERSON"}
    assert exempt == {"type": ["Community-default"]}

    calls: list[tuple] = []
    NeptuneRetriever._apply_filters(_RecordingTraversal(calls), filters, exempt)
    # Guarded: vertices of the exempt label pass, the rest must match.
    assert [c[0] for c in calls] == ["or_"]


def test_neptune_key_on_every_label_is_a_plain_has() -> None:
    filters, exempt = NeptuneRetriever._scope_filters_to_labels(
        {"Community-default": "community", "Entity-default": "entity"},
        {"name": "Vendor"},
    )
    assert exempt == {}
    calls: list[tuple] = []
    NeptuneRetriever._apply_filters(_RecordingTraversal(calls), filters, exempt)
    assert [c[0] for c in calls] == ["has"]


def test_neptune_filter_fields(config: Config) -> None:
    declared = _neptune(config).filter_fields()
    for key in ("id", "type", "description", "size"):
        assert declared.declares(key), key
    for key in ("status", "entity_type", "attr_category", "attributes.source"):
        assert not declared.declares(key), key


def test_neptune_schema_covers_indexed_entity_properties(config: Config) -> None:
    """Every non-attribute property ``NeptuneIndexer`` writes is declared."""
    indexer = NeptuneIndexer.__new__(NeptuneIndexer)
    object.__setattr__(indexer, "neptune_config", config.indexing.neptune)
    entity = Entity(
        id="e1",
        name="Vendor",
        type="ORGANIZATION",
        description="A supplier",
        rank=1.0,
        confidence=0.9,
        text_unit_ids=["tu-1"],
        community_ids=["c1"],
        attributes={"region": "north"},
    )
    props = indexer._build_vertex_properties(
        entity,
        {
            "name": entity.name,
            "type": entity.type,
            "description": entity.description,
            "rank": entity.rank,
            "confidence": entity.confidence,
            "text_unit_ids": entity.text_unit_ids,
            "community_ids": entity.community_ids,
        },
    )
    undeclared = {k for k in props if not NEPTUNE_FILTER_FIELDS["entity"].declares(k)}
    assert undeclared == {"attr_region"}


# --------------------------------------------------------------------------- #
# Query-time validation
# --------------------------------------------------------------------------- #


def test_chain_accepts_keys_some_retriever_declares(config: Config) -> None:
    retrievers = [_retriever(config, _FakeOpenSearchClient()), _neptune(config)]
    GraphRAGChain._validate_filter_keys(
        {"type": "PERSON", "attr_category": "research", "status": "open"},
        retrievers,
    )
    GraphRAGChain._validate_filter_keys(None, retrievers)


def test_chain_rejects_key_no_retriever_declares(config: Config) -> None:
    retrievers = [_retriever(config, _FakeOpenSearchClient()), _neptune(config)]
    with pytest.raises(InvalidFilterError) as exc:
        GraphRAGChain._validate_filter_keys(
            {"category": "research", "entity_type": "person", "type": "PERSON"},
            retrievers,
        )
    message = str(exc.value)
    assert "Unknown filter key(s): category, entity_type." in message
    # The error lists what is filterable.
    assert "attr_<key>" in message and "community_id" in message


def test_chain_scopes_validation_to_the_strategy_retrievers(config: Config) -> None:
    # ``status`` exists only on the OpenSearch claims index.
    with pytest.raises(InvalidFilterError, match="status"):
        GraphRAGChain._validate_filter_keys({"status": "open"}, [_neptune(config)])


def test_chain_accepts_any_key_for_a_backend_without_schema(config: Config) -> None:
    class _CustomRetriever:
        def filter_fields(self) -> None:
            return None

    GraphRAGChain._validate_filter_keys(
        {"anything": "goes"},
        [_neptune(config), _CustomRetriever()],  # type: ignore[list-item]
    )
