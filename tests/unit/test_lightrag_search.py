# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LightRAG dual-level search strategy (M3).

Uses fake retrievers (no AWS) and a stubbed HybridScorer to assert that each
mode (naive / hybrid / mix) queries the correct indices with the correct
keyword lists, fusing through the shared scorer.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.adapters.search_strategies.lightrag_search import (
    _pick_by_weighted_polling,
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


class FakeRetriever:
    """Records the index_prefixes/query of each aretrieve call."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        prefix = query.index_prefixes[0] if query.index_prefixes else self.tag
        return [
            RetrievalResult(
                content=f"{prefix} result",
                score=1.0,
                source=f"{prefix}-1",
                retriever_type=self.tag,
                metadata={"id": f"{prefix}-id"},
            )
        ]


def _chunk_selector(config: Config | None = None):
    """A strategy instance for the config-driven chunk-selection helpers.

    ``_collect_linked_chunk_ids`` reads ``related_chunk_number`` off the config, so
    it needs an instance; the retrievers are never touched by these helpers.
    """
    spec = get_strategy_spec(SearchStrategy.MIX)
    return spec.strategy_class(
        config=config or Config(),
        retrievers={
            RetrieverRole.DOCUMENT.value: FakeRetriever("document"),
            RetrieverRole.GRAPH.value: FakeRetriever("graph"),
        },
    )


def _make_strategy(config: Config):
    spec = get_strategy_spec(SearchStrategy.MIX)
    os_r, neptune_r = FakeRetriever("document"), FakeRetriever("graph")
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: os_r,
            RetrieverRole.GRAPH.value: neptune_r,
        },
    )
    # Stub the shared scorer to just flatten the source dict (avoid Bedrock).
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
            r for results in results_dict.values() for r in results
        ]
    )
    return strategy, os_r, neptune_r


def _query(mode: SearchStrategy, **kw) -> SearchQuery:
    return SearchQuery(query="q", metadata={"lightrag_mode": mode.value}, **kw)


async def test_naive_mode_queries_only_text_units(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    result = await strategy.asearch(
        _query(SearchStrategy.NAIVE, ll_keywords=["x"], hl_keywords=["y"])
    )
    # Naive ignores keywords/graph; one text-units retrieval, no Neptune.
    prefixes = [q.index_prefixes[0] for q in os_r.calls]
    assert prefixes == [config.indexing.opensearch.text_units_index_prefix]
    assert neptune_r.calls == []
    assert result.search_strategy == "lightrag_naive"


async def test_hybrid_mode_uses_entities_and_relationships(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    await strategy.asearch(
        _query(SearchStrategy.HYBRID, ll_keywords=["alice"], hl_keywords=["theme"])
    )
    prefixes = {q.index_prefixes[0] for q in os_r.calls}
    assert config.indexing.opensearch.entities_index_prefix in prefixes
    assert config.indexing.opensearch.relationships_index_prefix in prefixes
    # text-units (naive blend) NOT queried in hybrid mode.
    assert config.indexing.opensearch.text_units_index_prefix not in prefixes
    # Entity hits seed a Neptune expansion.
    assert len(neptune_r.calls) == 1


async def test_ll_keywords_go_to_entities_index(config: Config) -> None:
    strategy, os_r, _ = _make_strategy(config)
    await strategy.asearch(_query(SearchStrategy.HYBRID, ll_keywords=["alice", "bob"]))
    entity_calls = [
        q
        for q in os_r.calls
        if q.index_prefixes[0] == config.indexing.opensearch.entities_index_prefix
    ]
    assert entity_calls and "alice, bob" == entity_calls[0].query


async def test_mix_mode_blends_chunks(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    await strategy.asearch(
        _query(SearchStrategy.MIX, ll_keywords=["x"], hl_keywords=["y"])
    )
    prefixes = {q.index_prefixes[0] for q in os_r.calls}
    # mix = hybrid (entities+relationships) PLUS naive chunks.
    assert {
        config.indexing.opensearch.entities_index_prefix,
        config.indexing.opensearch.relationships_index_prefix,
        config.indexing.opensearch.text_units_index_prefix,
    } <= prefixes


def test_collect_linked_chunk_ids_ranks_by_citation_count() -> None:
    cls = _chunk_selector()

    def _hit(unit_ids: list[str]) -> RetrievalResult:
        return RetrievalResult(
            content="x",
            score=1.0,
            source="s",
            retriever_type="document",
            metadata={"id": "i", "text_unit_ids": unit_ids},
        )

    entity_hits = [_hit(["t1", "t2"]), _hit(["t1"])]  # t1 cited twice
    rel_hits = [_hit(["t2", "t3"])]  # t2 cited twice total, t3 once
    ranked = cls._collect_linked_chunk_ids(entity_hits, rel_hits, limit=10)
    # t1 (2) and t2 (2) before t3 (1); ties keep first-seen order (t1 before t2).
    assert ranked[:2] == ["t1", "t2"]
    assert ranked[-1] == "t3"
    # limit caps the list.
    assert cls._collect_linked_chunk_ids(entity_hits, rel_hits, limit=1) == ["t1"]


def test_collect_linked_chunk_ids_ignores_missing_lineage() -> None:
    cls = _chunk_selector()
    no_lineage = RetrievalResult(
        content="x", score=1.0, source="s", retriever_type="document", metadata={}
    )
    assert cls._collect_linked_chunk_ids([no_lineage], [], limit=10) == []


class LineageRetriever:
    """Returns entity/relationship hits carrying text_unit_ids lineage."""

    def __init__(self) -> None:
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        prefix = query.index_prefixes[0] if query.index_prefixes else "x"
        return [
            RetrievalResult(
                content=f"{prefix} result",
                score=1.0,
                source=f"{prefix}-1",
                retriever_type="document",
                metadata={"id": f"{prefix}-id", "text_unit_ids": ["chunk-A"]},
            )
        ]


async def test_mix_mode_fetches_linked_chunks_by_lineage(config: Config) -> None:
    spec = get_strategy_spec(SearchStrategy.MIX)
    os_r = LineageRetriever()
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: os_r,
            RetrieverRole.GRAPH.value: FakeRetriever("graph"),
        },
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
            r for results in results_dict.values() for r in results
        ]
    )
    await strategy.asearch(
        _query(SearchStrategy.MIX, ll_keywords=["x"], hl_keywords=["y"])
    )
    # The linked-chunk fetch queries the text-units index filtered by the chunk
    # ids cited by the matched entities/relationships.
    text_units_prefix = config.indexing.opensearch.text_units_index_prefix
    linked_calls = [
        q for q in os_r.calls if q.index_prefixes == [text_units_prefix] and q.filters
    ]
    assert linked_calls and linked_calls[0].filters.get("id") == ["chunk-A"]


async def test_empty_keywords_fall_back_to_raw_query(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    # Short hybrid query with no extracted keywords -> raw query forced as an
    # ll_keyword (LightRAG behavior), so entities ARE queried (no total miss).
    await strategy.asearch(_query(SearchStrategy.HYBRID))
    entity_calls = [
        q
        for q in os_r.calls
        if q.index_prefixes[0] == config.indexing.opensearch.entities_index_prefix
    ]
    assert entity_calls and entity_calls[0].query == "q"


async def test_long_empty_keyword_query_does_not_fall_back(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    # Exceeds config.search.lightrag_search.raw_query_fallback_max_len (default 50).
    long_query = "x" * 60
    await strategy.asearch(
        SearchQuery(query=long_query, metadata={"lightrag_mode": "hybrid"})
    )
    # No fallback -> no graph retrieval for a long keyword-less query.
    assert os_r.calls == []
    assert neptune_r.calls == []


def test_relationship_endpoint_ids_collects_both_endpoints() -> None:
    cls = _chunk_selector()
    rels = [
        RetrievalResult(
            content="r",
            score=1.0,
            source="rel-1",
            retriever_type="document",
            metadata={"source_id": "e1", "target_id": "e2"},
        ),
        RetrievalResult(
            content="r",
            score=1.0,
            source="rel-2",
            retriever_type="document",
            metadata={"source_id": "e3"},  # missing target_id is tolerated
        ),
    ]
    assert cls._relationship_endpoint_ids(rels) == ["e1", "e2", "e3"]


class RelationshipEndpointRetriever:
    """Returns relationship hits carrying endpoint ids (hl-only thematic query)."""

    def __init__(self) -> None:
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        return [
            RetrievalResult(
                content="rel",
                score=1.0,
                source="rel-1",
                retriever_type="document",
                metadata={"id": "rel-1", "source_id": "e1", "target_id": "e2"},
            )
        ]


async def test_hl_only_query_seeds_graph_via_relationship_endpoints(
    config: Config,
) -> None:
    # The documented fidelity fix: an hl-only (purely thematic) query — matched
    # relationships but no matched entities — must still seed Neptune graph
    # expansion via the relationships' endpoint entity ids.
    spec = get_strategy_spec(SearchStrategy.HYBRID)
    os_r = RelationshipEndpointRetriever()
    neptune_r = FakeRetriever("graph")
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: os_r,
            RetrieverRole.GRAPH.value: neptune_r,
        },
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
            r for results in results_dict.values() for r in results
        ]
    )
    # hl keywords only, no ll keywords -> no entity hits, only relationship hits.
    await strategy.asearch(_query(SearchStrategy.HYBRID, hl_keywords=["theme"]))

    # Neptune expansion was seeded from the relationship endpoints e1, e2.
    assert len(neptune_r.calls) == 1
    seeded = neptune_r.calls[0].filters.get("id")
    assert set(seeded) == {"e1", "e2"}


class FailingRetriever:
    """Always raises — exercises the per-source degradation branches."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        raise RuntimeError(f"{self.tag} backend down")


async def test_per_source_failure_degrades_to_empty_not_crash(config: Config) -> None:
    # If the document retriever raises, each _retrieve_* swallows it and returns
    # {}; the strategy still produces a SearchResult rather than propagating.
    spec = get_strategy_spec(SearchStrategy.HYBRID)
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: FailingRetriever("document"),
            RetrieverRole.GRAPH.value: FailingRetriever("graph"),
        },
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
            r for results in results_dict.values() for r in results
        ]
    )
    result = await strategy.asearch(
        _query(SearchStrategy.HYBRID, ll_keywords=["a"], hl_keywords=["b"])
    )
    # Degrades gracefully: no results, no exception.
    assert result.results == []


def test_collect_linked_chunk_ids_accepts_unwrapped_single_id() -> None:
    # NeptuneRetriever._clean_property_map unwraps a single-element value_map list
    # into a bare scalar, so a KG item whose lineage is exactly ONE chunk arrives
    # with text_unit_ids as a str. The old isinstance(list) guard skipped those
    # entirely, discarding the single-document bridge items multi-hop hinges on.
    uuid = "8c1f2a70-1111-2222-3333-444455556666"
    entity = RetrievalResult(
        content="e",
        score=1.0,
        source="e1",
        retriever_type="entity",
        metadata={"text_unit_ids": uuid},
    )
    rel = RetrievalResult(
        content="r",
        score=1.0,
        source="r1",
        retriever_type="relationship",
        metadata={"text_unit_ids": ["other-chunk"]},
    )
    ranked = _chunk_selector()._collect_linked_chunk_ids([entity], [rel], limit=10)
    assert uuid in ranked
    assert "other-chunk" in ranked


class TestUpstreamTwoRetrievalWidths:
    """Upstream LightRAG has TWO retrieval widths, not one.

    ``QueryParam.top_k`` (DEFAULT_TOP_K = 40) sizes the ENTITY and RELATIONSHIP vector
    queries; ``QueryParam.chunk_top_k`` (DEFAULT_CHUNK_TOP_K = 20) separately sizes the
    chunk stream (lightrag/constants.py; ``search_top_k = query_param.chunk_top_k or
    query_param.top_k`` in operate.py). Collapsing both onto the caller's single
    ``top_k`` starved the KG streams: measured on musique50/n=50 at top_k=10, upstream
    mix assembled a median 71 entities / 86 relations / 20 chunks per query against our
    20 / 10 / 17, and scored token-F1 0.653 vs our 0.477 (McNemar p=0.021).
    """

    async def test_entity_and_relationship_queries_use_the_kg_width(
        self, config: Config
    ) -> None:
        strategy, os_r, _ = _make_strategy(config)
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"], top_k=10)
        )
        # Only the unfiltered VECTOR queries; the id/endpoint-filtered fetches
        # (linked chunks, the incident edges) are not sized by top_k.
        by_prefix = {
            call.index_prefixes[0]: call
            for call in os_r.calls
            if call.index_prefixes and not call.filters
        }
        entities = by_prefix[config.indexing.opensearch.entities_index_prefix]
        relationships = by_prefix[config.indexing.opensearch.relationships_index_prefix]
        # Upstream DEFAULT_TOP_K, not the caller's 10.
        assert entities.top_k == 40
        assert relationships.top_k == 40

    async def test_chunk_query_uses_the_separate_chunk_width(
        self, config: Config
    ) -> None:
        strategy, os_r, _ = _make_strategy(config)
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"], top_k=10)
        )
        text_units = [
            call
            for call in os_r.calls
            if call.index_prefixes
            and call.index_prefixes[0]
            == config.indexing.opensearch.text_units_index_prefix
            # the linked-chunk fetch is an id-filtered lexical lookup, not the
            # naive vector query under test
            and not call.filters
        ]
        assert text_units, "mix must run a naive vector chunk query"
        assert all(call.top_k == 20 for call in text_units)

    async def test_a_caller_top_k_above_the_default_still_widens(
        self, config: Config
    ) -> None:
        # The upstream defaults are FLOORS, not caps: a lab sweep asking for top_k=60
        # must not be narrowed back down to 40/20.
        strategy, os_r, _ = _make_strategy(config)
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"], top_k=60)
        )
        assert all(
            call.top_k == 60
            for call in os_r.calls
            if call.index_prefixes and not call.filters
        )

    async def test_mix_quota_matches_the_two_widths(self, config: Config) -> None:
        strategy, _, _ = _make_strategy(config)
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, retrieval_multiplier=1, query=None, **kw):
            captured.update(kw)
            captured["top_k"] = top_k
            return [r for results in results_dict.values() for r in results]

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"], top_k=10)
        )
        quota = captured["per_type_quota"]
        # The quota -- not the vector-query width -- is what decides how much of each
        # stream survives fusion into the context, so it has to carry the same contract.
        assert quota["entity"] == 40
        assert quota["relationship"] == 40
        assert quota["text"] == 20

    async def test_widths_track_the_configured_values(self, config: Config) -> None:
        # The upstream defaults (40/20) are config, not constants: a deployment or a
        # sweep that rebalances the two widths must move both the vector-query width
        # and the fusion quota, and nothing may pin them back to the defaults.
        config.search.lightrag_search.kg_stream_top_k = 25
        config.search.lightrag_search.chunk_stream_top_k = 12
        strategy, os_r, _ = _make_strategy(config)
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, retrieval_multiplier=1, query=None, **kw):
            captured.update(kw)
            return [r for results in results_dict.values() for r in results]

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"], top_k=5)
        )
        quota = captured["per_type_quota"]
        assert quota["entity"] == 25
        assert quota["relationship"] == 25
        assert quota["text"] == 12
        # And the vector queries themselves: KG streams at 25, chunk stream at 12.
        widths = {
            call.index_prefixes[0]: call.top_k
            for call in os_r.calls
            if call.index_prefixes and not call.filters
        }
        os_config = config.indexing.opensearch
        assert widths[os_config.entities_index_prefix] == 25
        assert widths[os_config.relationships_index_prefix] == 25
        assert widths[os_config.text_units_index_prefix] == 12

    async def test_non_mix_modes_keep_a_flat_width(self, config: Config) -> None:
        # HYBRID has no chunk blend and no quota; only mix is realigned here, so the
        # other modes must be untouched by this fix.
        strategy, _, _ = _make_strategy(config)
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, retrieval_multiplier=1, query=None, **kw):
            captured.update(kw)
            return [r for results in results_dict.values() for r in results]

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(
            _query(
                SearchStrategy.HYBRID, ll_keywords=["a"], hl_keywords=["b"], top_k=10
            )
        )
        assert captured.get("per_type_quota") is None

    def test_linked_chunk_pool_is_bounded_by_the_chunk_width(self) -> None:
        # The lineage pool is a CHUNK stream, so it follows chunk_top_k (20), not the
        # caller's top_k (10) -- with 40 KG hits upstream's pool is much larger than a
        # flat top_k before the final cut.
        entities = [
            RetrievalResult(
                content="e",
                score=1.0,
                source=f"e{i}",
                retriever_type="entity",
                metadata={"text_unit_ids": [f"chunk-{i}"]},
            )
            for i in range(30)
        ]
        ranked = _chunk_selector()._collect_linked_chunk_ids(entities, [], limit=20)
        assert len(ranked) == 20


class TestUpstreamChunkSelection:
    """KG-linked chunk selection must match upstream.

    Upstream LightRAG picks the chunks behind its KG hits in two stages
    (``operate.py`` ``_find_related_text_unit_from_entities`` /
    ``_from_relations`` + ``utils.pick_by_weighted_polling``): a linear-decreasing
    per-hit quota with a ``min_related_chunks=1`` floor, with the relationship
    pass excluding chunks already taken by the entity pass, and a final
    round-robin merge against the naive vector stream under one
    ``seen_chunk_ids`` set (``_merge_chunks``).

    The previous implementation ranked the pooled lineage by global citation
    count and cut at ``chunk_top_k``, blind to the vector stream. Measured on
    musique50/n=50 (m16 mix): 37.2% of the linked stream duplicated chunks the
    vector query already returned, and gold-chain recall was 0.982 on
    full-chain queries vs 0.477 on the rest.
    """

    @staticmethod
    def _entity(idx: int, lineage: list[str]) -> RetrievalResult:
        return RetrievalResult(
            content="e",
            score=1.0,
            source=f"e{idx}",
            retriever_type="entity",
            metadata={"text_unit_ids": lineage},
        )

    def test_every_matched_hit_keeps_at_least_one_chunk(self) -> None:
        # The defect this fix targets: a chunk cited by many entities used to take
        # every slot, so the LAST entity -- which may hold the only chunk carrying
        # a multi-hop question's second hop -- contributed nothing.
        shared = [f"popular-{i}" for i in range(20)]
        entities = [self._entity(i, list(shared)) for i in range(5)]
        entities.append(self._entity(99, ["the-bridge-chunk"]))
        selected = _chunk_selector()._collect_linked_chunk_ids(entities, [], limit=20)
        assert "the-bridge-chunk" in selected

    def test_vector_stream_chunks_are_excluded(self) -> None:
        entities = [self._entity(0, ["dup-1", "dup-2", "fresh-1"])]
        selected = _chunk_selector()._collect_linked_chunk_ids(
            entities, [], limit=20, exclude={"dup-1", "dup-2"}
        )
        assert selected == ["fresh-1"]

    def test_relationship_pass_excludes_entity_chunks(self) -> None:
        entities = [self._entity(0, ["shared-1"])]
        relationships = [
            RetrievalResult(
                content="r",
                score=1.0,
                source="r0",
                retriever_type="relationship",
                metadata={"text_unit_ids": ["shared-1", "rel-only"]},
            )
        ]
        selected = _chunk_selector()._collect_linked_chunk_ids(
            entities, relationships, limit=20
        )
        assert selected.count("shared-1") == 1
        assert "rel-only" in selected

    def test_pool_is_still_bounded_by_the_chunk_width(self) -> None:
        entities = [
            self._entity(i, [f"c-{i}-{j}" for j in range(5)]) for i in range(30)
        ]
        selected = _chunk_selector()._collect_linked_chunk_ids(entities, [], limit=20)
        assert len(selected) == 20

    def test_weighted_polling_matches_the_upstream_gradient(self) -> None:
        # Upstream: expected[i] interpolates max_related_chunks -> min_related_chunks
        # across the hits, so the first hit gets 5 and the last gets 1.
        lineages = [[f"c{i}-{j}" for j in range(5)] for i in range(5)]
        selected = _pick_by_weighted_polling(lineages, 5)
        per_hit = [sum(1 for s in selected if s.startswith(f"c{i}-")) for i in range(5)]
        assert per_hit == [5, 4, 3, 2, 1]

    def test_weighted_polling_redistributes_unfilled_quota(self) -> None:
        # The first hit can only supply 1 chunk; its remaining 4 slots must go to
        # hits that still have unused chunks rather than being dropped.
        lineages = [["only"], [f"b{j}" for j in range(9)]]
        selected = _pick_by_weighted_polling(lineages, 5)
        assert selected[0] == "only"
        assert len(selected) == 1 + 5  # 1 own + 1 first-round + 4 redistributed

    def test_single_hit_takes_the_full_related_chunk_number(self) -> None:
        related = Config().search.lightrag_search.related_chunk_number
        selected = _pick_by_weighted_polling([[f"c{j}" for j in range(9)]], related)
        assert selected == [f"c{j}" for j in range(related)]


class IncidentEdgeRetriever:
    """Serves the entity index, then the incident-edge fetch by endpoint field.

    ``edges`` maps ``(source_id, target_id) -> rank`` so a test can control both
    which endpoint side finds an edge and the ``(rank, weight)`` ordering.
    """

    def __init__(self, config: Config, edges: dict[tuple[str, str], float]) -> None:
        self.os_config = config.indexing.opensearch
        self.edges = edges
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        prefix = query.index_prefixes[0] if query.index_prefixes else ""
        if prefix == self.os_config.entities_index_prefix:
            return [
                RetrievalResult(
                    content="e",
                    score=1.0,
                    source=eid,
                    retriever_type="entity",
                    metadata={"id": eid},
                )
                for eid in ("e1", "e2")
            ]
        if prefix != self.os_config.relationships_index_prefix:
            return []
        # The hl relationship vector query (no filters) returns one already-known edge.
        if not query.filters:
            return [
                RetrievalResult(
                    content="hl edge",
                    score=1.0,
                    source="rel-known",
                    retriever_type="relationship",
                    metadata={
                        "id": "rel-known",
                        "source_id": "e1",
                        "target_id": "e-known",
                    },
                )
            ]
        # The incident fetch: one query per endpoint field.
        field, wanted = next(iter(query.filters.items()))
        hits = []
        for (src, tgt), rank in self.edges.items():
            endpoint = src if field == "source_id" else tgt
            if endpoint in wanted:
                hits.append(
                    RetrievalResult(
                        content=f"{src}->{tgt}",
                        score=1.0,
                        source=f"rel-{src}-{tgt}",
                        retriever_type="relationship",
                        metadata={
                            "id": f"rel-{src}-{tgt}",
                            "source_id": src,
                            "target_id": tgt,
                            "rank": rank,
                            "weight": 1.0,
                        },
                    )
                )
        return hits


def _incident_strategy(
    config: Config, edges: dict[tuple[str, str], float], mode=SearchStrategy.MIX
):
    spec = get_strategy_spec(mode)
    os_r = IncidentEdgeRetriever(config, edges)
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: os_r,
            RetrieverRole.GRAPH.value: FakeRetriever("graph"),
        },
    )
    return strategy, os_r


class TestIncidentRelationshipExpansion:
    """Entity -> incident-edge cross-type expansion.

    Upstream LightRAG's ``_find_most_related_edges_from_entities``
    (``operate.py`` ~5204) pulls EVERY edge incident to the entity vector hits via
    ``get_nodes_edges_batch``, dedups by ``tuple(sorted(e))``, orders by
    ``(rank, weight)`` descending, and applies NO count cap — the per-type token
    limit (``DEFAULT_MAX_RELATION_TOKENS`` = 8000) does the trimming. We had
    only the mirror direction (relation -> endpoint entity), so its relationship
    stream held a median 28 relations per query against upstream's 86 on
    musique50/n=50: the candidates never existed to be truncated.
    """

    async def test_incident_edges_are_fetched_from_both_endpoint_sides(
        self, config: Config
    ) -> None:
        # `_build_filter_clauses` ANDs every filter, so source_id OR target_id
        # needs one query per side or every edge pointing INTO a matched entity is
        # missed.
        strategy, os_r = _incident_strategy(
            config, {("e1", "x"): 3.0, ("y", "e2"): 2.0}
        )
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, **kw):
            captured["sources"] = results_dict
            return []

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"])
        )
        fields = [
            next(iter(call.filters))
            for call in os_r.calls
            if call.filters
            and call.index_prefixes
            == [config.indexing.opensearch.relationships_index_prefix]
        ]
        assert sorted(fields) == ["source_id", "target_id"]
        incident = captured["sources"]["lightrag_incident_relationships"]
        assert {r.source for r in incident} == {"rel-e1-x", "rel-y-e2"}

    async def test_an_edge_between_two_matched_entities_is_carried_once(
        self, config: Config
    ) -> None:
        # e1->e2 is reachable from BOTH endpoint queries; upstream's
        # `tuple(sorted(e))` dedup keeps it once.
        strategy, _ = _incident_strategy(config, {("e1", "e2"): 5.0})
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, **kw):
            captured["sources"] = results_dict
            return []

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"])
        )
        incident = captured["sources"]["lightrag_incident_relationships"]
        assert [r.source for r in incident] == ["rel-e1-e2"]

    async def test_edges_are_ordered_by_rank_then_weight(self, config: Config) -> None:
        strategy, _ = _incident_strategy(
            config, {("e1", "low"): 1.0, ("e1", "hub"): 9.0, ("e1", "mid"): 4.0}
        )
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, **kw):
            captured["sources"] = results_dict
            return []

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"])
        )
        incident = captured["sources"]["lightrag_incident_relationships"]
        assert [r.source for r in incident] == [
            "rel-e1-hub",
            "rel-e1-mid",
            "rel-e1-low",
        ]

    async def test_edges_the_hl_query_already_returned_are_excluded(
        self, config: Config
    ) -> None:
        # `rel-known` (e1 -> e-known) comes back from the hl relationship vector
        # query; re-offering it from the incident side would spend the relationship
        # quota twice on one edge.
        strategy, _ = _incident_strategy(config, {("e1", "e-known"): 7.0})
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, **kw):
            captured["sources"] = results_dict
            return []

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"])
        )
        # The fake indexes the incident hit under a different doc id than the hl
        # hit, so the exclusion has to bite on the endpoint pair, not the id.
        incident = captured["sources"].get("lightrag_incident_relationships", [])
        hl_pairs = {
            (r.metadata["source_id"], r.metadata["target_id"])
            for r in captured["sources"]["lightrag_relationships"]
        }
        assert not [
            r
            for r in incident
            if (r.metadata["source_id"], r.metadata["target_id"]) in hl_pairs
        ]

    async def test_kg_quota_rises_to_the_expanded_candidate_count(
        self, config: Config
    ) -> None:
        # Upstream applies no COUNT cap to the KG lists (only
        # `_apply_token_truncation`), so the quota must be a floor that grows with
        # the expansion — otherwise the per-type cap would clamp the newly found
        # edges straight back to 40 and this fix would be a no-op.
        edges = {("e1", f"x{i}"): float(i) for i in range(60)}
        strategy, _ = _incident_strategy(config, edges)
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, retrieval_multiplier=1, query=None, **kw):
            captured.update(kw)
            return []

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"], top_k=10)
        )
        quota = captured["per_type_quota"]
        # 60 incident edges + 1 hl edge, all retriever_type "relationship".
        assert quota["relationship"] == 61
        # Chunks keep their hard cap: upstream caps the MERGED chunk pool at
        # chunk_top_k (the per-type cap).
        assert quota["text"] == 20

    async def test_no_entity_hits_means_no_incident_fetch(self, config: Config) -> None:
        # An hl-only (thematic) query has no entity hits to expand from; the
        # relation -> endpoint direction already covers that case.
        strategy, os_r = _incident_strategy(config, {("e1", "x"): 1.0})
        strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
            lambda results_dict, top_k, **kw: []
        )
        await strategy.asearch(_query(SearchStrategy.MIX, hl_keywords=["b"]))
        assert not [
            call
            for call in os_r.calls
            if call.filters
            and call.index_prefixes
            == [config.indexing.opensearch.relationships_index_prefix]
        ]


class EndpointEntityRetriever:
    """Serves the hl relationship vector query, then the endpoint entity fetch.

    ``rel_endpoints`` is the ordered list of ``(source_id, target_id)`` pairs the
    hl relationship vector query returns; ``present`` is the set of entity ids the
    entities index actually holds (so a missing node can be exercised).
    """

    def __init__(
        self,
        config: Config,
        rel_endpoints: list[tuple[str, str]],
        present: set[str] | None = None,
        ll_entity_ids: tuple[str, ...] = (),
    ) -> None:
        self.os_config = config.indexing.opensearch
        self.rel_endpoints = rel_endpoints
        self.present = present
        self.ll_entity_ids = ll_entity_ids
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        prefix = query.index_prefixes[0] if query.index_prefixes else ""
        if prefix == self.os_config.entities_index_prefix:
            if not query.filters:
                # the ll (low-level) entity vector query
                return [
                    RetrievalResult(
                        content="ll entity",
                        score=1.0,
                        source=eid,
                        retriever_type="entity",
                        metadata={"id": eid},
                    )
                    for eid in self.ll_entity_ids
                ]
            wanted = list(query.filters["id"])
            present = self.present if self.present is not None else set(wanted)
            # OpenSearch returns terms-filter hits in index/score order, NOT in
            # the order the ids were asked for -- reverse them to prove the
            # strategy restores the endpoint order itself.
            return [
                RetrievalResult(
                    content="endpoint entity",
                    score=1.0,
                    source=eid,
                    retriever_type="entity",
                    metadata={"id": eid},
                )
                for eid in reversed(wanted)
                if eid in present
            ]
        if prefix != self.os_config.relationships_index_prefix:
            return []
        if query.filters:
            return []  # the incident fetch: not under test here
        return [
            RetrievalResult(
                content="hl edge",
                score=1.0,
                source=f"rel-{src}-{tgt}",
                retriever_type="relationship",
                metadata={"id": f"rel-{src}-{tgt}", "source_id": src, "target_id": tgt},
            )
            for src, tgt in self.rel_endpoints
        ]


def _endpoint_strategy(config: Config, os_r: EndpointEntityRetriever):
    spec = get_strategy_spec(SearchStrategy.MIX)
    return spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: os_r,
            RetrieverRole.GRAPH.value: FakeRetriever("graph"),
        },
    )


class TestRelationshipEndpointEntities:
    """Relation endpoints are entity CANDIDATES.

    Upstream's global side runs NO entity vector query — its entity context is
    exactly the matched relationships' endpoints: ``_get_edge_data`` (operate.py
    ~5419) passes its relationship vector hits to
    ``_find_most_related_entities_from_relationships`` (~5478), which collects
    ``src_id``/``tgt_id`` in first-seen order, fetches them with
    ``get_nodes_batch``, and (in hybrid/mix) round-robin merges them with the ll
    entity list in ``_merge_context``. We previously routed those same ids ONLY into
    ``_expand_via_graph`` — a Neptune re-query returning the seeds' neighbourhood,
    not the endpoints — so the endpoints never became context items and the entity
    section sat at a median 48 per query against upstream's 71 on musique50/n=50.
    """

    @staticmethod
    def _captured_sources(strategy) -> dict:
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, retrieval_multiplier=1, query=None, **kw):
            captured["sources"] = results_dict
            captured.update(kw)
            return []

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        return captured

    async def test_endpoints_become_their_own_entity_stream(
        self, config: Config
    ) -> None:
        os_r = EndpointEntityRetriever(config, [("e1", "e2"), ("e3", "e4")])
        strategy = _endpoint_strategy(config, os_r)
        captured = self._captured_sources(strategy)
        await strategy.asearch(_query(SearchStrategy.MIX, hl_keywords=["theme"]))

        endpoints = captured["sources"]["lightrag_endpoint_entities"]
        assert [r.source for r in endpoints] == ["e1", "e2", "e3", "e4"]
        # They are entity-typed, so the `entity` quota floor picks them up.
        assert {r.retriever_type for r in endpoints} == {"entity"}

    async def test_endpoint_order_is_src_then_tgt_per_edge(
        self, config: Config
    ) -> None:
        # Upstream appends src BEFORE tgt for each edge in relevance order, and the
        # KG streams are not reranked -- stream position decides which endpoints
        # survive fusion, so the order is load-bearing, not cosmetic.
        os_r = EndpointEntityRetriever(config, [("a", "b"), ("c", "a"), ("d", "b")])
        strategy = _endpoint_strategy(config, os_r)
        captured = self._captured_sources(strategy)
        await strategy.asearch(_query(SearchStrategy.MIX, hl_keywords=["theme"]))

        endpoints = captured["sources"]["lightrag_endpoint_entities"]
        assert [r.source for r in endpoints] == ["a", "b", "c", "d"]

    async def test_entities_the_ll_query_already_returned_are_excluded(
        self, config: Config
    ) -> None:
        # Upstream merges the two entity lists under one `seen_entities` set
        # (`_merge_context`), so an endpoint that is also an ll vector hit must
        # not spend the entity quota twice.
        os_r = EndpointEntityRetriever(config, [("e1", "e2")], ll_entity_ids=("e1",))
        strategy = _endpoint_strategy(config, os_r)
        captured = self._captured_sources(strategy)
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["theme"])
        )

        fetched = [
            call
            for call in os_r.calls
            if call.filters
            and call.index_prefixes
            == [config.indexing.opensearch.entities_index_prefix]
        ]
        assert [c.filters["id"] for c in fetched] == [["e2"]]
        endpoints = captured["sources"]["lightrag_endpoint_entities"]
        assert [r.source for r in endpoints] == ["e2"]

    async def test_missing_nodes_are_skipped(self, config: Config) -> None:
        # Upstream rebuilds `node_datas` from `get_nodes_batch` and skips ids the
        # graph store has no node for, rather than emitting a placeholder.
        os_r = EndpointEntityRetriever(
            config, [("e1", "gone"), ("e3", "e4")], present={"e1", "e3", "e4"}
        )
        strategy = _endpoint_strategy(config, os_r)
        captured = self._captured_sources(strategy)
        await strategy.asearch(_query(SearchStrategy.MIX, hl_keywords=["theme"]))

        endpoints = captured["sources"]["lightrag_endpoint_entities"]
        assert [r.source for r in endpoints] == ["e1", "e3", "e4"]

    async def test_entity_quota_rises_to_the_endpoint_count(
        self, config: Config
    ) -> None:
        # Same contract as the incident expansion on the relationship side: upstream applies
        # no COUNT cap to the entity list (only `_apply_token_truncation` against
        # DEFAULT_MAX_ENTITY_TOKENS = 6000), so the quota floor has to grow with
        # the expansion or the per-type cap clamps the endpoints straight back out.
        pairs = [(f"s{i}", f"t{i}") for i in range(30)]
        os_r = EndpointEntityRetriever(config, pairs, ll_entity_ids=("e1", "e2"))
        strategy = _endpoint_strategy(config, os_r)
        captured = self._captured_sources(strategy)
        await strategy.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"], top_k=10)
        )
        # 60 endpoints + the 2 ll vector hits.
        assert captured["per_type_quota"]["entity"] == 62

    async def test_no_relationship_hits_means_no_endpoint_fetch(
        self, config: Config
    ) -> None:
        # An ll-only query has no relationship hits; the entity -> incident-edge
        # direction (incident edges) already covers that case.
        os_r = EndpointEntityRetriever(config, [], ll_entity_ids=("e1",))
        strategy = _endpoint_strategy(config, os_r)
        strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
            lambda results_dict, top_k, **kw: []
        )
        await strategy.asearch(_query(SearchStrategy.MIX, ll_keywords=["a"]))
        assert not [
            call
            for call in os_r.calls
            if call.filters
            and call.index_prefixes
            == [config.indexing.opensearch.entities_index_prefix]
        ]


class TestEndpointSeedingIsPreserved:
    """Emitting endpoints as entity candidates must not replace the graph seeding.

    The endpoint ids serve two distinct purposes: they seed Neptune expansion (to
    reach the endpoints' neighborhood) AND they are context items in their own
    right (upstream ``_find_most_related_entities_from_relationships``). Adding
    the second must keep the first.
    """

    async def test_endpoints_both_seed_neptune_and_become_candidates(
        self, config: Config
    ) -> None:
        spec = get_strategy_spec(SearchStrategy.MIX)
        os_r = EndpointEntityRetriever(config, [("e1", "e2")])
        neptune_r = FakeRetriever("graph")
        strategy = spec.strategy_class(
            config=config,
            retrievers={
                RetrieverRole.DOCUMENT.value: os_r,
                RetrieverRole.GRAPH.value: neptune_r,
            },
        )
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, **kw):
            captured["sources"] = results_dict
            return []

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(_query(SearchStrategy.MIX, hl_keywords=["theme"]))
        # Still seeded into the Neptune expansion...
        assert len(neptune_r.calls) == 1
        assert set(neptune_r.calls[0].filters["id"]) == {"e1", "e2"}
        # ...and also emitted as their own entity candidates.
        assert captured["sources"]["lightrag_endpoint_entities"]
