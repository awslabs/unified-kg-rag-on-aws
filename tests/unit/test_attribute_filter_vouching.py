# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document filters on graph expansion, whose store has no document attribute.

The relationship and claim indexes carry their documents' filters, so the
store applies them. Graph vertices are kept under an ``attr_<key>`` /
``attributes.filters.<key>`` filter only when their entities-index document,
which carries the filters of every document the entity comes from, passes it
(the expansion seeds aside).
"""

from __future__ import annotations

from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.adapters.storage.filter_schema import (
    NEPTUNE_ENTITY_FILTER_FIELDS,
    FilterFields,
    IndexLimits,
    opensearch_filter_fields,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.strategy_registry import get_strategy_spec

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _graph_walk(config: Config) -> None:
    """These tests pin the graph walk's entity check: the path without a
    relationships index (with one, filtered expansion walks that index)."""
    config.indexing.opensearch.build_relationship_vector_index = False


def _hit(
    source: str, kind: str = "entity", units: list[str] | None = None
) -> RetrievalResult:
    return RetrievalResult(
        content=source,
        score=1.0,
        source=source,
        retriever_type=kind,
        metadata={"id": source, "text_unit_ids": units or [f"u-{source}"]},
    )


class _Store:
    """The entities of ``passing`` pass the filter (in ``entities_suffix``);
    every other search returns ``hits``."""

    def __init__(
        self,
        entities_prefix: str,
        hits: list[RetrievalResult],
        passing: frozenset[str] = frozenset({"near", "one"}),
        fail: bool = False,
        entities_suffix: str | None = None,
        passing_units: frozenset[str] = frozenset(),
        id_batch: int = 65536,
        fields: FilterFields | None = None,
    ) -> None:
        self.fields = fields
        self.id_batch = id_batch
        self.passing_units = passing_units
        self.entities_prefix = entities_prefix
        self.hits = hits
        self.passing = passing
        self.fail = fail
        self.entities_suffix = entities_suffix
        self.queries: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits(max_terms_count=self.id_batch)

    def filter_fields(self) -> FilterFields | None:
        return self.fields

    def filter_fields_for(self, prefixes: list[str]) -> FilterFields | None:
        return self.fields

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.queries.append(query)
        units_prefix = self.entities_prefix.replace("entities", "text-units")
        if query.index_prefixes == [units_prefix] and query.query == "":
            # A text unit is one document's: it passes when its id is in
            # ``passing_units``.
            ids = (query.filters or {}).get("id") or []
            return [_hit(i, "text") for i in ids if i in self.passing_units]
        if query.index_prefixes == [self.entities_prefix] and query.query == "":
            if self.fail:
                raise RuntimeError("store hiccup")
            if query.suffix != self.entities_suffix:
                return []
            ids = (query.filters or {}).get("id") or []
            return [_hit(i) for i in ids if i in self.passing]
        return [h.model_copy() for h in self.hits]


def _strategy(config: Config, document: _Store, graph: _Store) -> Any:
    return get_strategy_spec(SearchStrategy.LOCAL).strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: document,
            RetrieverRole.GRAPH.value: graph,
        },
    )


def _graph(prefix: str) -> _Store:
    return _Store(prefix, [_hit(i) for i in ("seed", "near", "one", "far")])


@pytest.mark.parametrize("key", ["attr_category", "attributes.filters.category"])
async def test_graph_expansion_keeps_seeds_and_vouched_vertices(
    config: Config, key: str
) -> None:
    prefix = config.indexing.opensearch.entities_index_prefix
    document = _Store(prefix, [])
    strategy = _strategy(config, document, _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={key: "x", "type": "t"}), ["seed"]
    )

    assert [r.source for r in kept] == ["seed", "near", "one"]
    (check,) = document.queries
    assert check.index_prefixes == [prefix]
    # One request for the non-seed vertices, document filter keys only, and
    # ids only back.
    assert check.filters == {key: "x", "id": ["near", "one", "far"]}
    assert check.source_fields == []


async def test_the_check_searches_the_query_suffix(config: Config) -> None:
    # A query on another tenant/version suffix must check that suffix's
    # entities, not the default one's (which would drop every vertex).
    prefix = config.indexing.opensearch.entities_index_prefix
    document = _Store(prefix, [], entities_suffix="tenant-b")
    strategy = _strategy(config, document, _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", suffix="tenant-b", filters={"attr_category": "x"}),
        ["seed"],
    )

    assert [r.source for r in kept] == ["seed", "near", "one"]
    assert [q.suffix for q in document.queries] == ["tenant-b"]


@pytest.mark.parametrize("key", ["attributes.role", "name.keyword"])
async def test_an_entity_key_the_graph_does_not_store_is_vouched(
    config: Config, key: str
) -> None:
    # Regression: the graph drops a key its entity vertices lack (an
    # entity's own attributes.role, name.keyword), so its walk returned
    # every neighbour, failing the key, as an entity result (59 of 59 under
    # attributes.role with mix). The entities index holds the key.
    prefix = config.indexing.opensearch.entities_index_prefix
    entities = opensearch_filter_fields(
        *config.processing.translation.translated_languages
    )
    document = _Store(
        prefix, [], fields=entities["entities"], passing=frozenset({"near", "one"})
    )
    graph = _graph(prefix)
    graph.fields = NEPTUNE_ENTITY_FILTER_FIELDS
    strategy = _strategy(config, document, graph)

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={key: "x", "type": "t"}), ["seed"]
    )

    # The seed is checked too: LightRAG seeds the walk with relationship
    # endpoints, which no entity key selected (here it fails the key).
    assert [r.source for r in kept] == ["near", "one"]
    # The graph applied type; only the key it dropped is checked.
    assert [q.filters for q in document.queries] == [
        {key: "x", "id": ["near", "one", "far"]},
        {key: "x", "id": ["seed"]},
    ]


async def test_a_key_the_graph_applies_is_not_vouched(config: Config) -> None:
    prefix = config.indexing.opensearch.entities_index_prefix
    entities = opensearch_filter_fields(
        *config.processing.translation.translated_languages
    )
    document = _Store(prefix, [], fields=entities["entities"])
    graph = _graph(prefix)
    graph.fields = NEPTUNE_ENTITY_FILTER_FIELDS
    strategy = _strategy(config, document, graph)

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"type": "t", "name": "n"}), ["seed"]
    )

    assert len(kept) == 4
    assert document.queries == []


async def test_without_schemas_only_document_filters_are_vouched(
    config: Config,
) -> None:
    # A backend that declares no filter keys (filter_fields() is None):
    # which entity keys its graph stores is unknown.
    prefix = config.indexing.opensearch.entities_index_prefix
    document = _Store(prefix, [])
    strategy = _strategy(config, document, _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attributes.role": "buyer"}), ["seed"]
    )

    assert len(kept) == 4
    assert document.queries == []


async def test_the_check_batches_under_one_requests_limits(config: Config) -> None:
    # One fetch must name no more ids than a terms clause and a result
    # window hold, or the store rejects it and the query fails: the
    # retriever reports how many (the limits in force on its index).
    prefix = config.indexing.opensearch.entities_index_prefix
    document = _Store(prefix, [], passing=frozenset({"v3"}), id_batch=2)
    graph = _Store(prefix, [_hit(f"v{i}") for i in range(5)])
    strategy = _strategy(config, document, graph)

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_category": "x"}), ["v0"]
    )

    assert [r.source for r in kept] == ["v0", "v3"]
    assert [q.filters["id"] for q in document.queries] == [["v1", "v2"], ["v3", "v4"]]


async def test_a_failed_check_drops_what_it_could_not_vouch_for(
    config: Config,
) -> None:
    prefix = config.indexing.opensearch.entities_index_prefix
    strategy = _strategy(config, _Store(prefix, [], fail=True), _graph(prefix))

    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_category": "x"}), ["seed"]
    )

    assert [r.source for r in kept] == ["seed"]


async def test_document_store_results_are_filtered_by_the_store(
    config: Config,
) -> None:
    # Relationship and claim indexes apply document filters server-side:
    # nothing post-filters them (an extra fetch per sub-search).
    prefix = config.indexing.opensearch.entities_index_prefix
    store = _Store(prefix, [_hit("r-no", "relationship")])
    strategy = _strategy(config, store, _Store(prefix, []))

    kept = await strategy._safe_aretrieve(
        store, SearchQuery(query="q", filters={"attr_category": "x"}), "test"
    )

    assert [r.source for r in kept] == ["r-no"]
    assert len(store.queries) == 1


async def test_two_keys_need_one_document_passing_both(config: Config) -> None:
    # Regression: an entity's index document unions its documents' values per
    # key, so an entity of a {category: x, grp: a} document and a
    # {category: y, grp: b} one passed category:x + grp:b, which no document
    # does. Under two or more keys its text units (one document each) decide.
    o = config.indexing.opensearch
    assert o.text_units_index_prefix == o.entities_index_prefix.replace(
        "entities", "text-units"
    )
    document = _Store(
        o.entities_index_prefix,
        [],
        passing=frozenset({"near", "split"}),
        passing_units=frozenset({"u-near"}),
    )
    graph = _Store(
        o.entities_index_prefix,
        [
            _hit("seed"),
            _hit("near", units=["u-near", "u-other"]),
            _hit("split", units=["u-split-x", "u-split-b"]),
        ],
    )
    strategy = _strategy(config, document, graph)

    two = {"attr_category": "x", "attr_grp": "b"}
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters=two), ["seed"]
    )
    assert [r.source for r in kept] == ["seed", "near"]
    entity_check, unit_check = document.queries
    assert entity_check.filters == {**two, "id": ["near", "split"]}
    # Only the entity check's survivors' units, ids only.
    assert unit_check.filters == {
        **two,
        "id": ["u-near", "u-other", "u-split-x", "u-split-b"],
    }
    assert unit_check.source_fields == []

    # One key: the entity check alone decides, in one request.
    document.queries.clear()
    kept = await strategy._expand_via_graph(
        SearchQuery(query="q", filters={"attr_category": "x"}), ["seed"]
    )
    assert [r.source for r in kept] == ["seed", "near", "split"]
    assert len(document.queries) == 1
