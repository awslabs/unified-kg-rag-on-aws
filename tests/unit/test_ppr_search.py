# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``ppr`` strategy, ``local``'s ``text_unit_ranking: ppr`` and the
neighbourhood-edge routing, over scripted stores (docs/design.md §6.4)."""

from __future__ import annotations

import functools
from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.adapters.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.domain.models import (
    Config,
    GraphEdge,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.shared.exceptions import StoreConnectionError

pytestmark = pytest.mark.unit


def _config(**search: Any) -> Config:
    config = Config()
    config.providers.rerank.kind = "none"
    for key, value in search.items():
        section, field = key.split("__")
        setattr(getattr(config.search, section), field, value)
    return config


def _hit(kind: str, source: str, score: float, **metadata: Any) -> RetrievalResult:
    return RetrievalResult(
        content=f"{kind} {source}",
        score=score,
        source=source,
        retriever_type=kind,
        metadata={"id": source, **metadata},
    )


class _Documents:
    """A document retriever: per index, the hits of a text search, and every
    stored item by id (a fetch by id returns those the ids name)."""

    def __init__(
        self,
        config: Config,
        *,
        searches: dict[str, list[RetrievalResult]] | None = None,
        stored: dict[str, list[RetrievalResult]] | None = None,
    ) -> None:
        o = config.indexing.opensearch
        self.names = {
            o.text_units_index_prefix: "text_units",
            o.entities_index_prefix: "entities",
            o.relationships_index_prefix: "relationships",
            o.community_reports_index_prefix: "community_reports",
            o.claims_index_prefix: "claims",
        }
        self.searches = searches or {}
        self.stored = stored or {}
        self.calls: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool | None:
        return True

    def filter_fields(self) -> None:
        return None

    def filter_fields_for(self, prefixes: object) -> None:
        return None

    async def aneighbourhood_edges(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        index = self.names[query.index_prefixes[0]]
        filters = query.filters or {}
        if query.query:
            hits = self.searches.get(index, [])
        else:
            hits = self.stored.get(index, [])
        for key in ("id", "source_id", "target_id"):
            if key in filters:
                wanted = filters[key]
                wanted = set(wanted if isinstance(wanted, list) else [wanted])
                hits = [
                    h
                    for h in hits
                    if str((h.metadata or {}).get(key, h.source)) in wanted
                ]
        return hits[: query.top_k * query.retrieval_multiplier]

    def by(self, index: str) -> list[SearchQuery]:
        return [q for q in self.calls if self.names[q.index_prefixes[0]] == index]


class _Graph:
    """A graph retriever: the expansion returns the seeds and their listed
    neighbours; the edges are those between two asked vertices."""

    def __init__(
        self,
        entities: dict[str, list[str]],
        edges: list[tuple[str, str]],
        reach: dict[str, list[str]] | None = None,
        *,
        fail: Exception | None = None,
    ) -> None:
        self.entities = entities  # id -> text units
        self.edges = [
            GraphEdge(id=f"r-{a}-{b}", source_id=a, target_id=b) for a, b in edges
        ]
        self.reach = reach or {}
        self.fail = fail
        self.calls: list[SearchQuery] = []
        self.edge_calls: list[tuple[list[str], SearchQuery, int]] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    def filter_fields(self) -> None:
        return None

    def filter_fields_for(self, prefixes: object) -> None:
        return None

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool | None:
        return None

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        seeds = [str(s) for s in (query.filters or {}).get("id") or []]
        reached = list(
            dict.fromkeys(seeds + [n for s in seeds for n in self.reach.get(s, [])])
        )
        return [
            _hit(
                "entity", e, 1.0 if e in seeds else 0.5, text_unit_ids=self.entities[e]
            )
            for e in reached
            if e in self.entities
        ]

    async def aneighbourhood_edges(
        self, vertex_ids: list[str], query: SearchQuery, *, per_vertex: int
    ) -> list[GraphEdge] | None:
        self.edge_calls.append((list(vertex_ids), query, per_vertex))
        if self.fail is not None:
            raise self.fail
        members = set(vertex_ids)
        return [e for e in self.edges if {e.source_id, e.target_id} <= members]


def _strategy(strategy: SearchStrategy, config: Config, documents, graph):
    return get_strategy_spec(strategy).strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: documents,
            RetrieverRole.GRAPH.value: graph,
        },
    )


# A two-hop question: the facts name Ann (founded Acme) and Bob (born in
# Lund); only the bridge passage cites both. A hub entity, reachable from
# Ann, cites the hub passage; the dense passage search ranks the hub passage
# first.
_UNITS = {
    "ann": ["u-ann", "u-bridge"],
    "bob": ["u-bob", "u-bridge"],
    "hub": ["u-hub"],
    "acme": [],
    "lund": [],
}


def _bridge_stores(config: Config) -> tuple[_Documents, _Graph]:
    units = ["u-ann", "u-bob", "u-bridge", "u-hub"]
    documents = _Documents(
        config,
        searches={
            "relationships": [
                _hit(
                    "relationship", "r-acme-ann", 0.9, source_id="acme", target_id="ann"
                ),
                _hit(
                    "relationship", "r-bob-lund", 0.8, source_id="bob", target_id="lund"
                ),
            ],
            "text_units": [
                _hit("text", "u-hub", 0.9),
                _hit("text", "u-ann", 0.5),
            ],
        },
        stored={
            "text_units": [_hit("text", u, 0.0) for u in units],
            "entities": [
                _hit("entity", e, 0.0, text_unit_ids=u) for e, u in _UNITS.items()
            ],
        },
    )
    graph = _Graph(
        _UNITS,
        [("acme", "ann"), ("bob", "lund"), ("ann", "hub"), ("ann", "bob")],
        reach={"ann": ["hub", "bob"], "bob": ["ann"]},
    )
    return documents, graph


async def test_ppr_ranks_the_bridge_passage_first_and_emits_its_buckets() -> None:
    config = _config()
    documents, graph = _bridge_stores(config)
    strategy = _strategy(SearchStrategy.PPR, config, documents, graph)
    captured: dict[str, Any] = {}
    fuse = strategy._fuse_and_rerank

    async def spy(results: dict[str, list[RetrievalResult]], **kwargs: Any):
        captured.update(results=results, kwargs=kwargs)
        return await fuse(results, **kwargs)

    strategy._fuse_and_rerank = spy  # type: ignore[method-assign]
    result = await strategy.asearch(SearchQuery(query="Where was Acme's founder born?"))

    buckets = captured["results"]
    assert set(buckets) == {"ppr_passages", "ppr_seed_text_units", "ppr_facts"}
    passages = [r.source for r in buckets["ppr_passages"]]
    assert passages[0] == "u-bridge"
    assert passages.index("u-bridge") < passages.index("u-hub")
    assert [r.source for r in buckets["ppr_facts"]] == ["r-acme-ann", "r-bob-lund"]
    assert captured["kwargs"]["rerank_only_types"] == {"text"}
    assert captured["kwargs"]["per_type_quota"] == {"text": 20, "relationship": 5}
    # The facts' endpoints seed the expansion, each with its fact's score,
    # two hops deep, over max_neighbourhood_entities.
    (walk,) = graph.calls
    assert walk.filters == {"id": ["acme", "ann", "bob", "lund"]}
    # Fact scores relative to the best one.
    assert walk.seed_weights == pytest.approx(
        {"acme": 1.0, "ann": 1.0, "bob": 0.8 / 0.9, "lund": 0.8 / 0.9}
    )
    assert (walk.max_hops, walk.top_k, walk.retrieval_multiplier) == (2, 100, 1)
    asked, _, per_vertex = graph.edge_calls[0]
    assert set(asked) == {"acme", "ann", "bob", "lund", "hub"} and per_vertex == 50
    # No query-time LLM: the facts and passages are the question's hits.
    assert {q.query for q in documents.calls if q.query} == {
        "Where was Acme's founder born?"
    }
    assert result.search_strategy == "ppr_search"
    meta = result.metadata
    assert meta["methodology"] == "hipporag"
    assert meta["ppr"]["converged"] and meta["ppr"]["relationships"] == 4
    assert meta["sources"] == {
        "ppr_passages": 4,
        "ppr_seed_text_units": 2,
        "ppr_facts": 2,
    }
    assert {r.retriever_type for r in result.results} == {"text", "relationship"}


async def test_only_entities_in_the_graph_take_a_link_top_k_slot() -> None:
    # A fact naming an entity neither the walk nor the entities index
    # returns (filtered out, or deleted since) must not take a seed slot.
    config = _config(ppr_search__link_top_k=2)
    documents, graph = _bridge_stores(config)
    documents.searches["relationships"] = [
        _hit("relationship", "r-ghost", 0.9, source_id="ghost", target_id="ann"),
        _hit("relationship", "r-bob-lund", 0.8, source_id="bob", target_id="lund"),
    ]
    result = await _strategy(SearchStrategy.PPR, config, documents, graph).asearch(
        SearchQuery(query="q")
    )
    assert result.metadata["ppr"]["entity_seeds"] == 2


async def test_the_seed_balance_does_not_depend_on_the_score_scale() -> None:
    # BM25 scores run to tens, kNN and hybrid to about 1: the facts are read
    # relative to the best, so the passage weight keeps its meaning.
    resets = []
    for scale in (1.0, 12.5):
        config = _config()
        documents, graph = _bridge_stores(config)
        for hit in documents.searches["relationships"]:
            hit.score = (hit.score or 0.0) * scale
        strategy = _strategy(SearchStrategy.PPR, config, documents, graph)

        async def spy(
            query: Any, entities: Any, reset: Any, *args: Any, _rank: Any, **kw: Any
        ):
            resets.append(dict(reset))
            return await _rank(query, entities, reset, *args, **kw)

        strategy._ppr_rank = functools.partial(  # type: ignore[method-assign]
            spy, _rank=strategy._ppr_rank
        )
        await strategy.asearch(SearchQuery(query="q"))
    assert resets[0] == pytest.approx(resets[1])
    assert max(resets[0].values()) <= 1.0


async def test_every_store_request_keeps_the_suffix_and_filters() -> None:
    config = _config()
    documents, graph = _bridge_stores(config)
    strategy = _strategy(SearchStrategy.PPR, config, documents, graph)
    await strategy.asearch(
        SearchQuery(query="q", suffix="tenant-a", filters={"type": "person"})
    )
    assert documents.calls and graph.calls and graph.edge_calls
    for query in [*documents.calls, *graph.calls]:
        assert query.suffix == "tenant-a"
        assert (query.filters or {}).get("type") == "person"
    _, edges_query, _ = graph.edge_calls[0]
    assert edges_query.suffix == "tenant-a"
    assert edges_query.filters == {"type": "person"}


async def test_nothing_matched_returns_no_results_without_a_walk() -> None:
    config = _config()
    documents = _Documents(config)
    graph = _Graph({}, [])
    result = await _strategy(SearchStrategy.PPR, config, documents, graph).asearch(
        SearchQuery(query="q")
    )
    assert result.results == [] and graph.calls == []


async def test_without_the_relationship_index_entity_hits_seed() -> None:
    config = _config()
    config.indexing.opensearch.build_relationship_vector_index = False
    documents, graph = _bridge_stores(config)
    documents.searches["entities"] = [
        _hit("entity", "ann", 0.7, text_unit_ids=_UNITS["ann"])
    ]
    strategy = _strategy(SearchStrategy.PPR, config, documents, graph)
    result = await strategy.asearch(SearchQuery(query="q"))
    assert documents.by("relationships") == []
    (entities,) = [q for q in documents.by("entities") if q.query]
    assert entities.top_k == config.search.ppr_search.fact_top_k
    assert graph.calls[0].filters == {"id": ["ann"]}
    assert "u-bridge" in {r.source for r in result.results}


async def test_rerank_off_reranks_nothing() -> None:
    config = _config(ppr_search__rerank=False)
    documents, graph = _bridge_stores(config)
    strategy = _strategy(SearchStrategy.PPR, config, documents, graph)
    captured: dict[str, Any] = {}

    async def spy(results: dict[str, list[RetrievalResult]], **kwargs: Any):
        captured.update(kwargs)
        return []

    strategy._fuse_and_rerank = spy  # type: ignore[method-assign]
    await strategy.asearch(SearchQuery(query="q"))
    assert captured["rerank_only_types"] == set()


async def test_a_failed_expansion_reads_the_seeds_from_the_entities_index() -> None:
    config = _config()
    documents, graph = _bridge_stores(config)
    graph.entities = {}  # the walk returns nothing
    strategy = _strategy(SearchStrategy.PPR, config, documents, graph)
    result = await strategy.asearch(SearchQuery(query="q"))
    (read,) = [q for q in documents.by("entities") if not q.query]
    assert sorted(read.filters["id"]) == ["acme", "ann", "bob", "lund"]
    # The seeds' own passages still rank: the bridge both seeds cite first.
    assert result.metadata["ppr"]["entities"] == 4
    passages = [r.source for r in result.results if r.retriever_type == "text"]
    assert "u-bridge" in passages


async def test_edges_that_fail_degrade_and_fatal_errors_raise() -> None:
    config = _config()
    documents, graph = _bridge_stores(config)
    graph.fail = RuntimeError("one request failed")
    strategy = _strategy(SearchStrategy.PPR, config, documents, graph)
    result = await strategy.asearch(SearchQuery(query="q"))
    assert result.metadata["ppr"]["relationships"] == 0
    assert result.results  # the entity-passage links still rank
    graph.fail = StoreConnectionError("graph down")
    with pytest.raises(StoreConnectionError):
        await strategy.asearch(SearchQuery(query="q"))


# --- local's text_unit_ranking: ppr -----------------------------------------


def _local(config: Config, documents: _Documents, graph: _Graph):
    strategy = _strategy(SearchStrategy.LOCAL, config, documents, graph)
    captured: dict[str, Any] = {}
    fuse = strategy._fuse_and_rerank

    async def spy(results: dict[str, list[RetrievalResult]], **kwargs: Any):
        captured.update(results)
        return await fuse(results, **kwargs)

    strategy._fuse_and_rerank = spy  # type: ignore[method-assign]
    return strategy, captured


def strategy_support_first(config: Config, nodes: list[RetrievalResult]) -> str:
    """The text unit local's support ranking puts first."""
    local = get_strategy_spec(SearchStrategy.LOCAL).strategy_class
    return local._rank_text_unit_ids(nodes)[0]


def test_local_ranks_by_pagerank_by_default() -> None:
    # Measured against support on 200 MuSiQue questions, two repeats
    # (docs/concepts/defaults-and-evidence.md#local-ppr-ranking).
    assert Config().search.local_search.text_unit_ranking == "ppr"


async def test_local_support_ranking_reads_no_passage_seeds() -> None:
    config = _config(local_search__text_unit_ranking="support")
    documents, graph = _bridge_stores(config)
    documents.searches["entities"] = [_hit("entity", "ann", 0.9, name="Ann")]
    strategy, buckets = _local(config, documents, graph)
    result = await strategy.asearch(SearchQuery(query="q", entity_focus=["Ann"]))
    assert result.metadata["text_unit_ranking"] == "support"
    assert "ppr" not in result.metadata and "ppr_seed_text_units" not in buckets
    # No passage seed search and no edge read.
    assert [q for q in documents.by("text_units") if q.query] == []
    assert graph.edge_calls == []


async def test_local_ppr_ranks_the_text_units_by_pagerank() -> None:
    config = _config(local_search__text_unit_ranking="ppr")
    documents, graph = _bridge_stores(config)
    documents.searches["entities"] = [
        _hit("entity", "ann", 0.9, name="Ann"),
        _hit("entity", "bob", 0.8, name="Bob"),
    ]
    # Three of Ann's neighbours cite one noise passage: by support (citing
    # entities) it outranks the bridge two seeds cite.
    noise = ["x", "y", "z"]
    graph.entities.update(dict.fromkeys(noise, ["u-noise"]))
    documents.stored["text_units"].append(_hit("text", "u-noise", 0.0))
    graph.reach["ann"] = [*graph.reach["ann"], *noise]
    graph.edges += [
        GraphEdge(id=f"r-ann-{n}", source_id="ann", target_id=n) for n in noise
    ]
    expanded = await graph.aretrieve(
        SearchQuery(query="", filters={"id": ["ann", "bob"]})
    )
    assert strategy_support_first(config, expanded) == "u-noise"
    strategy, buckets = _local(config, documents, graph)
    result = await strategy.asearch(SearchQuery(query="q", entity_focus=["Ann", "Bob"]))
    meta = result.metadata
    assert meta["text_unit_ranking"] == "ppr"
    # Ann and Bob, plus the endpoints of the question's relationship hits.
    assert meta["ppr"]["fact_seeds"] == 2 and meta["ppr"]["entity_seeds"] == 4
    units = [r.source for r in buckets["text_units"]]
    assert units[0] == "u-bridge"
    assert units.index("u-bridge") < units.index("u-noise")
    # The passage hits rank inside it, not as a bucket of their own.
    assert "ppr_seed_text_units" not in buckets
    assert "u-hub" in units
    # The passage hits are of the raw question, under the request's search;
    # the facts by kNN, 5 of them (the measured settings).
    (passages,) = [q for q in documents.by("text_units") if q.query]
    assert passages.top_k == config.search.ppr_search.passage_seed_top_k == 20
    assert passages.search_type is SearchType.HYBRID
    (facts,) = [q for q in documents.by("relationships") if q.query == "q"]
    assert (facts.top_k, facts.search_type) == (5, SearchType.VECTOR)


async def test_local_ppr_seeds_relationship_hits_and_reads_their_endpoints() -> None:
    config = _config(
        local_search__text_unit_ranking="ppr", local_search__ppr_seed_relationships=2
    )
    documents, graph = _bridge_stores(config)
    documents.searches["entities"] = [_hit("entity", "ann", 0.9, name="Ann")]
    graph.reach = {}  # the walk reaches no one: Bob comes from the facts
    strategy, buckets = _local(config, documents, graph)
    result = await strategy.asearch(SearchQuery(query="q", entity_focus=["Ann"]))
    assert result.metadata["ppr"]["fact_seeds"] == 2
    (facts,) = [q for q in documents.by("relationships") if q.query == "q"]
    assert facts.top_k == 2
    (endpoints,) = [
        q for q in documents.by("entities") if not q.query and "id" in (q.filters or {})
    ]
    assert sorted(endpoints.filters["id"]) == ["acme", "bob", "lund"]
    assert "u-bob" in {r.source for r in buckets["text_units"]}


# --- neighbourhood edges ------------------------------------------------------


async def test_document_filters_read_the_edges_from_the_relationships_index() -> None:
    config = _config()
    documents = _Documents(
        config,
        stored={
            "relationships": [
                _hit("relationship", "r1", 0.0, source_id="a", target_id="b"),
                _hit("relationship", "r2", 0.0, source_id="b", target_id="c"),
                _hit("relationship", "r3", 0.0, source_id="a", target_id="z"),
            ]
        },
    )
    graph = _Graph({}, [("a", "b")])
    strategy = _strategy(SearchStrategy.PPR, config, documents, graph)
    edges = await strategy._neighbourhood_edges(
        SearchQuery(query="q", filters={"attr_category": "x", "id": ["a"]}),
        ["a", "b", "c"],
        per_vertex=5,
    )
    assert [e.id for e in edges] == ["r1", "r2"]
    assert graph.edge_calls == []
    (asked,) = documents.by("relationships")
    # Both endpoints in the set; the caller's id applied to the vertices.
    assert asked.filters == {
        "attr_category": "x",
        "target_id": ["a", "b", "c"],
        "source_id": ["a", "b", "c"],
    }


async def test_a_graph_that_cannot_tell_falls_back_to_the_index() -> None:
    config = _config()
    documents = _Documents(
        config,
        stored={
            "relationships": [
                _hit("relationship", "r1", 0.0, source_id="a", target_id="b")
            ]
        },
    )
    strategy = _strategy(SearchStrategy.PPR, config, documents, documents)
    edges = await strategy._neighbourhood_edges(SearchQuery(query="q"), ["a", "b"], 5)
    assert [e.id for e in edges] == ["r1"]
    assert await strategy._neighbourhood_edges(SearchQuery(query="q"), ["a"], 5) == []


# --- equivalence with the measured prototype ---------------------------------


def _prototype_ppr(edges: dict[str, dict[str, float]], reset: dict[str, float]):
    """The retrieval screen's PageRank, as run (power iteration, damping 0.5,
    <= 30 iterations, L1 tolerance 1e-6, dangling mass to the reset)."""
    total = sum(reset.values())
    r = {n: v / total for n, v in reset.items() if v > 0}
    nodes = set(edges) | set(r)
    for neighbours in list(edges.values()):
        nodes |= set(neighbours)
    degree = {n: sum(edges.get(n, {}).values()) for n in nodes}
    p = {n: r.get(n, 0.0) for n in nodes}
    for _ in range(30):
        following = {n: 0.5 * r.get(n, 0.0) for n in nodes}
        dangling = 0.0
        for n, mass in p.items():
            if mass == 0.0:
                continue
            if degree[n] <= 0:
                dangling += mass
                continue
            for m, w in edges[n].items():
                following[m] += 0.5 * mass * w / degree[n]
        for n, v in r.items():
            following[n] += 0.5 * dangling * v
        change = sum(abs(following[n] - p[n]) for n in nodes)
        p = following
        if change < 1e-6:
            break
    return p


def _prototype_rank(seeds, nodes, facts, passage_hits, stored_entities, edges, limit):
    """The screen's arm D ranking of local's text units, on the same inputs."""
    units_of = {n.source: list(n.metadata["text_unit_ids"]) for n in nodes}
    weights: dict[str, float] = {}
    for hit in seeds:
        weights[hit.source] = weights.get(hit.source, 0.0) + hit.score
    fact_w: dict[str, list[float]] = {}
    for fact in facts:
        for end in (fact.metadata["source_id"], fact.metadata["target_id"]):
            fact_w.setdefault(end, []).append(fact.score)
    for end in fact_w:
        if end not in units_of and end in stored_entities:
            units_of[end] = list(stored_entities[end])
    for end, scores in fact_w.items():
        weights[end] = weights.get(end, 0.0) + sum(scores) / len(scores)
    graph: dict[str, dict[str, float]] = {}

    def add(a: str, b: str) -> None:
        graph.setdefault(a, {})
        graph.setdefault(b, {})
        graph[a][b] = graph[a].get(b, 0.0) + 1.0
        graph[b][a] = graph[b].get(a, 0.0) + 1.0

    for a, b in edges:
        if a in units_of and b in units_of:
            add("e:" + a, "e:" + b)
    passages: set[str] = set()
    for entity, units in units_of.items():
        if len(units) > limit:
            continue
        for unit in units:
            passages.add(unit)
            add("e:" + entity, "p:" + unit)
    reset: dict[str, float] = {}
    for entity, w in weights.items():
        if w > 0:
            reset["e:" + entity] = w / (len(units_of.get(entity) or []) or 1)
    scores = [h.score for h in passage_hits]
    low, high = min(scores), max(scores)
    for hit in passage_hits:
        passages.add(hit.source)
        graph.setdefault("p:" + hit.source, {})
        norm = 1.0 if high == low else (hit.score - low) / (high - low)
        reset["p:" + hit.source] = reset.get("p:" + hit.source, 0.0) + 0.05 * norm
    p = _prototype_ppr(graph, reset)
    return sorted(passages, key=lambda u: (-p.get("p:" + u, 0.0), u))


@pytest.mark.parametrize("threshold", [20, 2, 1])
async def test_local_ppr_ranks_as_the_measured_prototype(threshold: int) -> None:
    # The retrieval screen's arm D (support@ctx +11.0 on 200 MuSiQue
    # questions): the same ranking on a fixture with a hub over the
    # frequency threshold, a fact endpoint outside the expansion and
    # passage hits no entity cites.
    config = _config(
        local_search__text_unit_ranking="ppr",
        local_search__entity_frequency_threshold=threshold,
    )
    documents, graph = _bridge_stores(config)
    graph.entities["hub"] = ["u-hub", "u-h2", "u-h3"]
    graph.edges.append(GraphEdge(id="r-lund-x", source_id="lund", target_id="x"))
    stored = dict(_UNITS)
    stored["lund"] = ["u-lund"]
    documents.stored["entities"] = [
        _hit("entity", e, 0.0, text_unit_ids=u) for e, u in stored.items()
    ]
    nodes = [
        _hit("entity", "ann", 1.0, text_unit_ids=_UNITS["ann"]),
        _hit("entity", "bob", 1.0, text_unit_ids=_UNITS["bob"]),
        _hit("entity", "hub", 0.5, text_unit_ids=graph.entities["hub"]),
    ]
    seeds = [_hit("entity", "ann", 0.83), _hit("entity", "bob", 0.41)]
    facts = [
        _hit("relationship", "r1", 0.77, source_id="bob", target_id="lund"),
        _hit("relationship", "r2", 0.64, source_id="ann", target_id="bob"),
        _hit("relationship", "r3", 0.52, source_id="acme", target_id="ann"),
    ]
    passages = [
        _hit("text", "u-hub", 0.92),
        _hit("text", "u-free", 0.61),  # cited by no entity
        _hit("text", "u-bob", 0.33),
    ]
    strategy = _strategy(SearchStrategy.LOCAL, config, documents, graph)
    ours, stats = await strategy._ppr_ranked_unit_ids(
        SearchQuery(query="q"), nodes, seeds, passages, facts
    )
    edges = [(e.source_id, e.target_id) for e in graph.edges]
    reference = _prototype_rank(seeds, nodes, facts, passages, stored, edges, threshold)
    assert ours == reference
    assert stats["converged"]
