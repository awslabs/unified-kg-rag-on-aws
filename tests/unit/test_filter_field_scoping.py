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
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import P

from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.adapters.storage.filter_schema import (
    NEPTUNE_FILTER_FIELDS,
    IndexLimits,
    opensearch_filter_fields,
)
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    Entity,
    Relationship,
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
    object.__setattr__(inst, "_settings_denied_logged", False)
    object.__setattr__(inst, "_window_cut_warned", None)
    object.__setattr__(
        inst,
        "_configured_limits",
        IndexLimits.of(config.indexing.opensearch.index_settings),
    )
    object.__setattr__(inst, "_index_limits", {})
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())
    object.__setattr__(
        inst, "_index_filter_fields", inst._initialize_index_filter_fields()
    )
    object.__setattr__(inst, "_opensearch_client", client)
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


class _CountingOpenSearchClient(_FakeOpenSearchClient):
    def __init__(self) -> None:
        super().__init__()
        self.requests = 0

    async def asearch(self, index: str, body: dict[str, Any], **kw: Any) -> dict:
        self.requests += 1
        return await super().asearch(index, body, **kw)


async def test_long_list_filters_are_sent_in_one_request(config: Config) -> None:
    # A terms query is one clause however many values it holds. Batching the
    # lists (and searching the cross product of the batches) sent 400
    # sequential requests for two 3,000-id filters, each ranked on its own.
    client = _CountingOpenSearchClient()
    retriever = _retriever(config, client)
    ids = [f"t{i}" for i in range(3000)]
    communities = [f"c{i}" for i in range(3000)]
    text_units = config.indexing.opensearch.text_units_index_prefix
    await retriever.aretrieve(
        SearchQuery(
            query="supply terms",
            search_type=SearchType.LEXICAL,
            index_prefixes=[text_units],
            filters={"id": ids, "community_ids": communities},
        )
    )
    assert client.requests == 1
    body = client.searches[_alias(retriever, text_units)]
    assert {"terms": {"id": ids}} in _filter_clauses(body)
    assert {"terms": {"community_ids": communities}} in _filter_clauses(body)


async def test_a_list_filter_beyond_max_terms_count_is_rejected(
    config: Config,
) -> None:
    client = _CountingOpenSearchClient()
    retriever = _retriever(config, client)
    with pytest.raises(InvalidFilterError, match="65537 values.*65536"):
        await retriever.aretrieve(
            SearchQuery(
                query="supply terms",
                search_type=SearchType.LEXICAL,
                filters={"id": [f"t{i}" for i in range(65537)]},
            )
        )
    assert client.requests == 0


@pytest.mark.parametrize("key", ["max_terms_count", "index.max_terms_count"])
async def test_max_terms_count_is_read_in_either_spelling(
    config: Config, key: str
) -> None:
    # OpenSearch takes "index.max_terms_count" as well; that spelling fell
    # back to the 65,536 default.
    config.indexing.opensearch.index_settings[key] = 10
    client = _CountingOpenSearchClient()
    retriever = _retriever(config, client)
    with pytest.raises(InvalidFilterError, match="11 values.*at most 10"):
        await retriever.aretrieve(
            SearchQuery(
                query="supply terms",
                search_type=SearchType.LEXICAL,
                filters={"id": [f"t{i}" for i in range(11)]},
            )
        )
    assert client.requests == 0


async def test_attribute_key_filters_only_attribute_bearing_indexes(
    config: Config,
) -> None:
    p = _prefixes(config)
    retriever, client = await _search_all(config, {"attr_category": "research"})

    def fields(name: str) -> set[str]:
        return _filtered_fields(client.searches[_alias(retriever, p[name])])

    # Relationships and claims carry their documents' filters too, so the
    # store applies the filter to them as well (it used to drop the key and
    # search them unfiltered).
    for name in ("text_units", "entities", "reports", "relationships", "claims"):
        assert fields(name) == {"attr_category"}, name


async def test_an_entity_attribute_key_filters_only_the_entities_index(
    config: Config,
) -> None:
    # Regression: every index declared the whole ``attributes.`` prefix, so
    # an entity's own attribute (``attributes.role``) emptied the text
    # units, relationships, claims and reports. Like ``type``, it applies
    # where it exists; document filters apply everywhere.
    p = _prefixes(config)
    retriever, client = await _search_all(config, {"attributes.role": "buyer"})

    def fields(name: str) -> set[str]:
        return _filtered_fields(client.searches[_alias(retriever, p[name])])

    assert fields("entities") == {"attributes.role"}
    for name in ("text_units", "relationships", "claims", "reports"):
        assert fields(name) == set(), name

    retriever, client = await _search_all(
        config, {"attributes.filters.category": "research"}
    )
    for name in ("text_units", "entities", "relationships", "claims", "reports"):
        assert fields(name) == {"attributes.filters.category"}, name


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
    # Documents built via _prepare_common_doc_properties, and relationships
    # and claims, which carry their document filters only.
    assert attribute_kinds == set(schema)
    rel = Relationship(
        id="r1",
        source_id="a",
        target_id="b",
        attributes={
            "filters": {"category": "research"},
            "text_unit_weights": {"tu-1": 1.0},
        },
    )
    claim = Claim(
        id="c1",
        subject_id="a",
        subject_name="A",
        object_name="",
        type="fact",
        attributes={"filters": {"category": "research"}},
    )
    for doc in (
        indexer._prepare_relationship_doc(rel, ([0.1],)),
        indexer._prepare_claim_doc(claim, ([0.1],)),
    ):
        assert doc["attr_category"] == "research"
        # Only the filters: per-unit maps would map one field per unit.
        assert doc["attributes"] == {"filters": {"category": "research"}}


# --------------------------------------------------------------------------- #
# Neptune
# --------------------------------------------------------------------------- #


def _neptune(config: Config) -> NeptuneRetriever:
    inst = NeptuneRetriever.__new__(NeptuneRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_neptune_config", config.indexing.neptune)
    return inst


def test_neptune_drops_key_the_label_does_not_declare() -> None:
    filters = NeptuneRetriever._scope_filters_to_label(
        "Entity-default",
        "entity",
        {"id": ["e1"], "type": "PERSON", "status": "open"},
    )
    assert filters == {"id": ["e1"], "type": "PERSON"}


def _matches(vertex: dict[str, Any], steps: list[list[Any]]) -> bool:
    """Evaluate the filter steps ``_apply_filters`` emits against one vertex.

    Covers the subset used: ``has`` (value or ``P``), ``hasNot``, ``or`` and
    ``where`` over anonymous traversals.
    """

    def _sub(arg: Any) -> bool:
        return _matches(vertex, arg.step_instructions)

    for name, *args in steps:
        if name == "identity":
            continue
        if name == "has":
            key, want = args
            if key not in vertex:
                return False
            value = vertex[key]
            if isinstance(want, P) and want.operator == "within":
                ok = value in want.value
            elif isinstance(want, P):
                assert want.operator == "gte", want.operator
                ok = value >= want.value
            else:
                ok = value == want
            if not ok:
                return False
        elif name == "hasNot":
            if args[0] in vertex:
                return False
        elif name == "or":
            if not any(_sub(a) for a in args):
                return False
        elif name == "where":
            if not _sub(args[0]):
                return False
        else:  # pragma: no cover - guards the evaluator's coverage
            raise AssertionError(f"unsupported step {name}")
    return True


_VERTICES = [
    {"id": "e1", "type": "PERSON", "attr_role": "buyer"},
    {"id": "e2", "type": "PERSON", "attr_role": "vendor"},
    {"id": "e3", "type": "ORGANIZATION"},
]


def _kept(filters: dict[str, Any]) -> list[str]:
    scoped = NeptuneRetriever._scope_filters_to_label(
        "Entity-default", "entity", filters
    )
    traversal = NeptuneRetriever._apply_filters(__.identity(), scoped)
    steps = traversal.bytecode.step_instructions
    return [v["id"] for v in _VERTICES if _matches(v, steps)]


def test_neptune_entity_attribute_filters_vertices_that_have_it() -> None:
    # attr_role is an LLM-extracted entity attribute stored on the vertex;
    # vertices without the property are not filtered by it.
    assert _kept({"attr_role": "buyer"}) == ["e1", "e3"]
    assert _kept({"attr_role": ["buyer", "vendor"]}) == ["e1", "e2", "e3"]


def test_neptune_document_attribute_does_not_empty_expansion() -> None:
    # attr_category is a document attribute: absent on every vertex.
    assert _kept({"attr_category": "research"}) == ["e1", "e2", "e3"]


def test_neptune_strict_key_stays_strict() -> None:
    assert _kept({"type": "PERSON"}) == ["e1", "e2"]
    assert _kept({"type": "PERSON", "attr_role": "vendor"}) == ["e2"]


def test_neptune_filter_fields(config: Config) -> None:
    declared = _neptune(config).filter_fields()
    for key in ("id", "type", "description", "size", "attr_role"):
        assert declared.declares(key), key
    for key in ("status", "entity_type", "attributes.source"):
        assert not declared.declares(key), key


def test_neptune_schema_covers_indexed_entity_properties(config: Config) -> None:
    """Every property ``NeptuneIndexer`` writes on an entity is declared."""
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
    assert "attr_region" in props
    assert all(NEPTUNE_FILTER_FIELDS["entity"].declares(k) for k in props)


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


def test_a_strategy_checks_filter_keys_against_its_own_retrievers(
    config: Config,
) -> None:
    # ``status`` exists only on the OpenSearch claims index: a strategy that
    # reads only the graph rejects it, and so does auto whenever it may route
    # a question to such a strategy (the routed query would raise).
    from types import SimpleNamespace

    from unified_kg_rag.domain.models import SearchStrategy

    reads = {
        SearchStrategy.LOCAL: {"graph": _neptune(config)},
        SearchStrategy.NAIVE: {"document": _retriever(config, _FakeOpenSearchClient())},
    }
    chain = object.__new__(GraphRAGChain)
    chain.config = config
    # Strategies that declare no index set: their retrievers' every index.
    chain._get_strategy_instance = lambda s: SimpleNamespace(  # type: ignore[method-assign]
        retrievers=reads[s], filtered_index_prefixes=lambda _strategy: None
    )
    chain.check_filter_keys({"status": "open"}, SearchStrategy.NAIVE)
    with pytest.raises(InvalidFilterError, match="status"):
        chain.check_filter_keys({"status": "open"}, SearchStrategy.LOCAL)
    config.search.auto_routable_strategies = [SearchStrategy.LOCAL]
    with pytest.raises(InvalidFilterError, match="status"):
        chain.check_filter_keys({"status": "open"}, SearchStrategy.AUTO)
    config.search.auto_routable_strategies = [
        SearchStrategy.NAIVE,
        SearchStrategy.LOCAL,
    ]
    with pytest.raises(InvalidFilterError, match="auto may route to local"):
        chain.check_filter_keys({"status": "open"}, SearchStrategy.AUTO)
    config.search.auto_routable_strategies = [SearchStrategy.NAIVE]
    chain.check_filter_keys({"status": "open"}, SearchStrategy.AUTO)


def _strategy_chain(config: Config) -> GraphRAGChain:
    """A chain over the built-in strategies (built without their models),
    reading through fake OpenSearch and Neptune retrievers."""
    from types import SimpleNamespace

    from unified_kg_rag.adapters.search_strategies.drift_search import (
        DriftSearchStrategy,
    )
    from unified_kg_rag.adapters.search_strategies.global_search import (
        GlobalSearchStrategy,
    )
    from unified_kg_rag.adapters.search_strategies.lightrag_search import (
        LightRAGSearchStrategy,
    )
    from unified_kg_rag.adapters.search_strategies.local_search import (
        LocalSearchStrategy,
    )
    from unified_kg_rag.adapters.search_strategies.simple_search import (
        SimpleSearchStrategy,
    )
    from unified_kg_rag.domain.models import SearchStrategy

    def built(cls: type, **attrs: Any) -> Any:
        inst = object.__new__(cls)
        inst.config = config
        for name, value in attrs.items():
            setattr(inst, name, value)
        return inst

    local = built(LocalSearchStrategy)
    lightrag = built(LightRAGSearchStrategy, _os_config=config.indexing.opensearch)
    strategies = {
        SearchStrategy.LOCAL: local,
        SearchStrategy.DRIFT: built(DriftSearchStrategy, local_search=local),
        SearchStrategy.GLOBAL: built(
            GlobalSearchStrategy, global_search_config=config.search.global_search
        ),
        SearchStrategy.MIX: lightrag,
        SearchStrategy.HYBRID: lightrag,
        SearchStrategy.NAIVE: lightrag,
        SearchStrategy.SIMPLE: built(SimpleSearchStrategy),
    }
    document = {"document": _retriever(config, _FakeOpenSearchClient())}
    both = {**document, "graph": _neptune(config)}
    graph_free = {SearchStrategy.GLOBAL, SearchStrategy.NAIVE, SearchStrategy.SIMPLE}
    chain = object.__new__(GraphRAGChain)
    chain.config = config
    chain._get_strategy_instance = lambda s: SimpleNamespace(  # type: ignore[method-assign]
        retrievers=document if s in graph_free else both,
        filtered_index_prefixes=strategies[s].filtered_index_prefixes,
    )
    return chain


@pytest.mark.parametrize(
    ("strategy", "key", "accepted"),
    [
        # An entity type: no index naive or global reads has it.
        ("naive", "type", False),
        ("global", "type", False),
        ("mix", "type", True),
        ("local", "type", True),
        # A claims field: only strategies that read the claims index.
        ("naive", "status", False),
        ("mix", "status", False),
        ("local", "status", True),
        ("drift", "status", True),
        ("simple", "status", True),
        # A report field, and a document attribute every index carries.
        ("global", "community_id", True),
        ("naive", "community_id", False),
        ("naive", "attr_category", True),
        ("global", "attr_category", True),
        # Text-unit fields reach global only through text_unit_slots.
        ("global", "n_tokens", False),
        ("naive", "n_tokens", True),
    ],
)
def test_filter_keys_are_checked_against_the_indexes_a_strategy_reads(
    config: Config, strategy: str, key: str, accepted: bool
) -> None:
    # Checked against every index of the retriever, so naive or global with
    # type: person ran unfiltered.
    from unified_kg_rag.domain.models import SearchStrategy

    config.processing.claim_extraction.enabled = True
    chain = _strategy_chain(config)
    if accepted:
        chain.check_filter_keys({key: "x"}, SearchStrategy(strategy))
        return
    with pytest.raises(InvalidFilterError, match=f"Unknown filter key\\(s\\): {key}"):
        chain.check_filter_keys({key: "x"}, SearchStrategy(strategy))


def test_global_reads_text_unit_fields_with_text_unit_slots(config: Config) -> None:
    from unified_kg_rag.domain.models import SearchStrategy

    config.search.global_search.text_unit_slots = 2
    _strategy_chain(config).check_filter_keys({"n_tokens": 3}, SearchStrategy.GLOBAL)


def test_a_claims_key_is_refused_when_no_claims_index_is_built(config: Config) -> None:
    from unified_kg_rag.domain.models import SearchStrategy

    config.processing.claim_extraction.enabled = False
    with pytest.raises(InvalidFilterError, match="status"):
        _strategy_chain(config).check_filter_keys(
            {"status": "open"}, SearchStrategy.LOCAL
        )


async def test_a_query_checks_the_indexes_its_strategy_reads(config: Config) -> None:
    # The per-query check (each question, as with --filters) uses the same
    # index set as check_filter_keys.
    from unified_kg_rag.domain.models import SearchStrategy

    chain = _strategy_chain(config)
    state = {
        "resolved_strategy": SearchStrategy.NAIVE,
        "filters": {"type": "person"},
    }
    with pytest.raises(InvalidFilterError, match="type"):
        await chain._search_step(state)


def test_chain_accepts_any_key_for_a_backend_without_schema(config: Config) -> None:
    class _CustomRetriever:
        def filter_fields(self) -> None:
            return None

    GraphRAGChain._validate_filter_keys(
        {"anything": "goes"},
        [_neptune(config), _CustomRetriever()],  # type: ignore[list-item]
    )


@pytest.mark.parametrize("value", [None, ["a", None]], ids=["none", "none-in-list"])
async def test_a_filter_without_a_value_is_rejected(config: Config, value) -> None:
    # OpenSearch answers a terms query naming null with a 400; the embedded
    # engine matched it against null fields (a cleared filter key).
    with pytest.raises(InvalidFilterError, match="has no value"):
        await _search_all(config, {"attr_category": value})


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"gte": None, "lte": 3}, "range bound"),
        ([["a"], "b"], "not a scalar"),
        ([{"x": 1}], "not a scalar"),
        ({"gte": {"x": 1}}, "not a scalar"),
        (("a", "b"), "not a scalar"),
        (float("nan"), "NaN"),
        ({"lte": float("nan")}, "NaN"),
        ({}, "empty range"),
        ({"from": 1}, "range bound 'from'"),
    ],
    ids=[
        "none-bound",
        "nested-list",
        "mapping-item",
        "mapping-bound",
        "tuple",
        "nan",
        "nan-bound",
        "empty-range",
        "unknown-bound",
    ],
)
async def test_a_malformed_filter_value_is_rejected(
    config: Config, value: Any, message: str
) -> None:
    # OpenSearch answers these with a 400; the embedded engine matched them.
    with pytest.raises(InvalidFilterError, match=message):
        await _search_all(config, {"n_tokens": value})


def test_a_malformed_filter_fails_before_any_model_call() -> None:
    # The value check ran at retrieval only, after the query rewrite and
    # entity extraction had been paid for.
    with pytest.raises(InvalidFilterError, match="empty range"):
        GraphRAGChain._prepare_invoke({"query": "q", "filters": {"n_tokens": {}}})
