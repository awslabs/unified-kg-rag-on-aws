# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LightRAG dual-level search strategy (M3).

Uses fake retrievers (no AWS) and a stubbed HybridScorer to assert that each
mode (naive / hybrid / mix) queries the correct indices with the correct
keyword lists, fusing through the shared scorer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from tests.fixtures.fakes.retrievers import (
    FakeRetriever,
    incident_ends,
    opensearch_fields_for,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.adapters.search_strategies.lightrag_search import (
    _pick_by_weighted_polling,
)
from unified_kg_rag.adapters.storage.filter_schema import (
    IndexLimits,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.retrieval.filters import (
    FilterFields,
)
from unified_kg_rag.shared.exceptions import InvalidFilterError

pytestmark = pytest.mark.unit


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
    return SearchQuery(query="q", metadata={"search_strategy": mode.value}, **kw)


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
    # LightRAG modes read only OpenSearch: no multi-hop Neptune expansion.
    assert neptune_r.calls == []


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


def test_collect_linked_chunk_ids_reads_each_lineage_in_citation_then_id_order() -> (
    None
):
    cls = _chunk_selector()

    def _hit(unit_ids: list[str]) -> RetrievalResult:
        return RetrievalResult(
            content="x",
            score=1.0,
            source="s",
            retriever_type="document",
            metadata={"id": "i", "text_unit_ids": unit_ids},
        )

    entity_hits = [_hit(["t2", "t1"]), _hit(["t1"])]
    rel_hits = [_hit(["t3", "t2"])]
    entity_side, relation_side = cls._collect_linked_chunk_ids(
        entity_hits, rel_hits, limit=10
    )
    # Each hit's chunks by how many hits of the side cite them, then id,
    # whatever order they are stored in, a chunk kept on its earliest hit
    # only; the relationship side, a list of its own, drops the entity side's.
    assert entity_side == ["t1", "t2"]
    assert relation_side == ["t3"]
    # limit caps each side, not the two together.
    assert cls._collect_linked_chunk_ids(entity_hits, rel_hits, limit=1) == (
        ["t1"],
        ["t2"],
    )


@pytest.mark.parametrize("side", ["entity", "relationship"])
def test_a_chunk_more_matched_hits_cite_comes_first_in_its_lineage(
    side: str,
) -> None:
    # Regression (rv23): each hit's chunks were taken in id (hash) order,
    # so a bridge chunk that hits A, B and C all cite, kept on A only, came
    # after A's ten other chunks and never reached A's five slots. Upstream
    # (operate.py _find_related_text_unit_from_entities / _from_relations)
    # sorts each deduplicated lineage by how many matched hits cite a chunk;
    # chunk id breaks ties, so a full and an incremental build agree.
    def _hit(i: int, units: list[str]) -> RetrievalResult:
        return RetrievalResult(
            content=f"x{i}",
            score=1.0,
            source=f"x{i}",
            retriever_type=side,
            metadata={"id": f"x{i}", "text_unit_ids": units},
        )

    hub = [f"a{j:02d}" for j in range(11)] + ["z_bridge"]
    hits = [_hit(0, hub), _hit(1, ["z_bridge", "b1"]), _hit(2, ["z_bridge", "c1"])]
    entity_hits, rel_hits = (hits, []) if side == "entity" else ([], hits)
    entity_side, relation_side = _chunk_selector()._collect_linked_chunk_ids(
        entity_hits, rel_hits, limit=20
    )
    chosen = entity_side if side == "entity" else relation_side
    assert chosen[0] == "z_bridge"
    assert chosen[:5] == ["z_bridge", "a00", "a01", "a02", "a03"]
    assert {"b1", "c1"} <= set(chosen)


def test_collect_linked_chunk_ids_ignores_missing_lineage() -> None:
    cls = _chunk_selector()
    no_lineage = RetrievalResult(
        content="x", score=1.0, source="s", retriever_type="document", metadata={}
    )
    assert cls._collect_linked_chunk_ids([no_lineage], [], limit=10) == ([], [])


class LineageRetriever:
    """Returns entity/relationship hits carrying text_unit_ids lineage."""

    def __init__(self) -> None:
        self.calls: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    def filter_fields_for(self, prefixes: Collection[str]) -> FilterFields:
        return opensearch_fields_for(Config(), prefixes)

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


async def test_long_empty_keyword_query_does_not_fall_back(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    # Exceeds config.search.lightrag_search.raw_query_fallback_max_len (default 50).
    long_query = "x" * 60
    await strategy.asearch(
        SearchQuery(query=long_query, metadata={"search_strategy": "hybrid"})
    )
    # No fallback -> no graph retrieval for a long keyword-less query.
    assert os_r.calls == []
    assert neptune_r.calls == []
    # ... and says so, as upstream does, instead of abstaining silently.
    assert "too long (60 characters, limit 50" in caplog.text


@pytest.mark.parametrize(
    ("query", "falls_back"),
    [
        # Regression (rv9 query M2): a dense character counted 4, so this
        # Korean question of 19 characters (76) found no entity and
        # abstained. The gate counts characters, as upstream's len(query).
        ("애크미 로보틱스는 누구와 계약했나?", True),
        ("倉庫は誰の部品を保管していますか", True),
        ("仓库为谁保管零件呢请告诉我", True),
        ("창" * 49, True),
        ("창" * 50, False),
        ("x" * 49, True),
        ("x" * 50, False),
    ],
)
async def test_the_fallback_gate_counts_characters_as_upstream(
    config: Config, query: str, falls_back: bool
) -> None:
    strategy, os_r, _ = _make_strategy(config)
    fallback = strategy._apply_keyword_fallback(
        SearchQuery(query=query, metadata={"search_strategy": "hybrid"})
    )
    assert (fallback.ll_keywords == [query]) is falls_back


async def test_a_gate_of_zero_turns_the_fallback_off(config: Config) -> None:
    config.search.lightrag_search.raw_query_fallback_max_len = 0
    strategy, _, _ = _make_strategy(config)
    fallback = strategy._apply_keyword_fallback(
        SearchQuery(query="q", metadata={"search_strategy": "hybrid"})
    )
    assert fallback.ll_keywords == []


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

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

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


async def test_hl_only_query_grounds_entities_via_relationship_endpoints(
    config: Config,
) -> None:
    # An hl-only (purely thematic) query — matched relationships but no matched
    # entities — must still reach entities, via the relationships' endpoint ids
    # (upstream `_find_most_related_entities_from_relationships`).
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

    entity_prefix = config.indexing.opensearch.entities_index_prefix
    endpoint_fetches = [
        q
        for q in os_r.calls
        if q.index_prefixes == [entity_prefix] and q.filters and "id" in q.filters
    ]
    assert len(endpoint_fetches) == 1
    assert set(endpoint_fetches[0].filters["id"]) == {"e1", "e2"}
    assert neptune_r.calls == []


class FailingRetriever:
    """Always raises — exercises the per-source degradation branches."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

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


def test_collect_linked_chunk_ids_follows_a_single_chunk_lineage() -> None:
    # A KG item whose lineage is exactly ONE chunk (the single-document bridge
    # items multi-hop hinges on) is followed like any other.
    uuid = "8c1f2a70-1111-2222-3333-444455556666"
    entity = RetrievalResult(
        content="e",
        score=1.0,
        source="e1",
        retriever_type="entity",
        metadata={"text_unit_ids": [uuid]},
    )
    rel = RetrievalResult(
        content="r",
        score=1.0,
        source="r1",
        retriever_type="relationship",
        metadata={"text_unit_ids": ["other-chunk"]},
    )
    ranked = _chunk_selector()._collect_linked_chunk_ids([entity], [rel], limit=10)
    assert ranked == ([uuid], ["other-chunk"])


class TestUpstreamTwoRetrievalWidths:
    """Upstream LightRAG has TWO retrieval widths, not one.

    ``QueryParam.top_k`` (DEFAULT_TOP_K = 40) sizes the ENTITY and RELATIONSHIP vector
    queries; ``QueryParam.chunk_top_k`` (DEFAULT_CHUNK_TOP_K = 20) separately sizes the
    chunk stream (lightrag/constants.py; ``search_top_k = query_param.chunk_top_k or
    query_param.top_k`` in operate.py). Collapsing both onto the caller's single
    ``top_k`` starves the KG streams: a width sized for chunks is far narrower than
    what the entity/relationship queries need.
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
            if call.index_prefixes and not call.filters and not call.any_of
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
            if call.index_prefixes and not call.filters and not call.any_of
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
            if call.index_prefixes and not call.filters and not call.any_of
        }
        os_config = config.indexing.opensearch
        assert widths[os_config.entities_index_prefix] == 25
        assert widths[os_config.relationships_index_prefix] == 25
        assert widths[os_config.text_units_index_prefix] == 12

    async def test_hybrid_gets_the_same_quota_as_mix(self, config: Config) -> None:
        # HYBRID runs the same three streams as mix (ll->entities, hl->relationships,
        # KG-linked chunks); mix only adds the naive VECTOR chunk blend on top. Gating
        # the quota on mix alone left hybrid taking a flat top_k cut, far narrower
        # than upstream hybrid's context, so it often lacked the evidence to answer.
        # The previous version of this test asserted that no-quota behaviour as
        # intended, which encoded the defect.
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
        quota = captured.get("per_type_quota")
        assert quota is not None
        assert quota["text"] == config.search.lightrag_search.chunk_stream_top_k
        assert quota["entity"] >= config.search.lightrag_search.kg_stream_top_k
        assert quota["relationship"] >= config.search.lightrag_search.kg_stream_top_k
        # And the rerank stays scoped to chunks, as for mix.
        assert captured.get("rerank_only_types") == {"text"}

    async def test_naive_keeps_a_flat_width(self, config: Config) -> None:
        # naive retrieves ONE stream (vector chunks), so there is nothing to reserve
        # slots against and a flat top_k is correct.
        strategy, _, _ = _make_strategy(config)
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, retrieval_multiplier=1, query=None, **kw):
            captured.update(kw, top_k=top_k)
            return [r for results in results_dict.values() for r in results]

        strategy.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        await strategy.asearch(_query(SearchStrategy.NAIVE, top_k=10))
        assert captured.get("per_type_quota") is None
        assert captured.get("rerank_only_types") is None
        # Upstream naive returns `chunk_top_k` chunks, not the KG `top_k`.
        assert captured["top_k"] == config.search.lightrag_search.chunk_stream_top_k

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
        ranked, _ = _chunk_selector()._collect_linked_chunk_ids(entities, [], limit=20)
        assert len(ranked) == 20


class TestUpstreamChunkSelection:
    """KG-linked chunk selection must match upstream.

    Upstream LightRAG picks the chunks behind its KG hits in two stages
    (``operate.py`` ``_find_related_text_unit_from_entities`` /
    ``_from_relations`` + ``utils.pick_by_weighted_polling``): a linear-decreasing
    per-hit quota with a ``min_related_chunks=1`` floor, each pass a list of
    its own, the relationship pass excluding only the chunks the entity pass
    SELECTED (``entity_chunks``), and a final round-robin merge of the vector,
    entity and relation lists under one ``seen_chunk_ids`` set
    (``_merge_chunks``), cut to ``chunk_top_k``.

    The previous implementation ranked the pooled lineage by global citation
    count and cut at ``chunk_top_k``, blind to the vector stream, so part of
    the linked stream's budget went to chunks the vector query had already
    returned, and chunks cited by only one matched item were cut first.
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
        selected, _ = _chunk_selector()._collect_linked_chunk_ids(
            entities, [], limit=20
        )
        assert "the-bridge-chunk" in selected

    def test_vector_stream_chunks_are_excluded(self) -> None:
        entities = [self._entity(0, ["dup-1", "dup-2", "fresh-1"])]
        relationships = [self._relationship(0, ["dup-1", "rel-fresh"])]
        selected = _chunk_selector()._collect_linked_chunk_ids(
            entities, relationships, limit=20, exclude={"dup-1", "dup-2"}
        )
        assert selected == (["fresh-1"], ["rel-fresh"])

    @staticmethod
    def _relationship(idx: int, lineage: list[str]) -> RetrievalResult:
        return RetrievalResult(
            content="r",
            score=1.0,
            source=f"r{idx}",
            retriever_type="relationship",
            metadata={"text_unit_ids": lineage},
        )

    def test_relationship_pass_excludes_entity_chunks(self) -> None:
        entities = [self._entity(0, ["shared-1"])]
        relationships = [self._relationship(0, ["shared-1", "rel-only"])]
        selected = _chunk_selector()._collect_linked_chunk_ids(
            entities, relationships, limit=20
        )
        assert selected == (["shared-1"], ["rel-only"])

    def test_relationship_pass_keeps_entity_chunks_the_entity_pass_did_not_select(
        self,
    ) -> None:
        # Regression: the relationship pass dropped every chunk ANY entity hit
        # cited, selected or not. A hub entity cites most of the corpus, so a
        # relationship's chunks were nearly always excluded and its side came
        # back empty (hybrid one-hop gold-chunk recall 0.30).
        hub = self._entity(0, [f"c{j:02d}" for j in range(30)])
        relationships = [self._relationship(0, ["c00", "c07", "c21"])]
        entity_side, relation_side = _chunk_selector()._collect_linked_chunk_ids(
            [hub], relationships, limit=20
        )
        assert entity_side == ["c00", "c01", "c02", "c03", "c04"]
        assert relation_side == ["c07", "c21"]

    def test_each_pass_is_bounded_by_the_chunk_width(self) -> None:
        # Cut together, the entity pass took the whole width first and the
        # relationship pass got what was left: nothing, with 20 entity hits.
        entities = [
            self._entity(i, [f"c-{i}-{j}" for j in range(5)]) for i in range(30)
        ]
        relationships = [
            self._relationship(i, [f"r-{i}-{j}" for j in range(5)]) for i in range(30)
        ]
        entity_side, relation_side = _chunk_selector()._collect_linked_chunk_ids(
            entities, relationships, limit=20
        )
        assert len(entity_side) == 20
        assert len(relation_side) == 20

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

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

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
        # The hl relationship vector query (no filters) returns one already-known
        # edge: the same relationships-index document the incident fetch finds.
        if not query.filters and not query.any_of:
            return [
                RetrievalResult(
                    content="hl edge",
                    score=1.0,
                    source="rel-e1-e-known",
                    retriever_type="relationship",
                    metadata={
                        "id": "rel-e1-e-known",
                        "source_id": "e1",
                        "target_id": "e-known",
                    },
                )
            ]
        # The incident fetch: both endpoint fields in one query (any_of).
        asked = incident_ends(query)
        hits = []
        for (src, tgt), rank in self.edges.items():
            ends = {"source_id": src, "target_id": tgt}
            if any(ends[end] in ids for end, ids in asked.items()):
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
    only the mirror direction (relation -> endpoint entity), so the relationship
    stream was a fraction of upstream's width: the candidates never existed to be
    truncated.
    """

    async def test_incident_edges_are_fetched_from_both_endpoint_sides(
        self, config: Config
    ) -> None:
        # source_id OR target_id: one query names both sides (any_of), or
        # every edge pointing INTO a matched entity is missed.
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
        (incident_call,) = [
            call
            for call in os_r.calls
            if call.any_of
            and call.index_prefixes
            == [config.indexing.opensearch.relationships_index_prefix]
        ]
        assert sorted(next(iter(g)) for g in incident_call.any_of or []) == [
            "source_id",
            "target_id",
        ]
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
        # `rel-e1-e-known` comes back from the hl relationship vector
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
        incident = captured["sources"].get("lightrag_incident_relationships", [])
        hl_ids = {r.source for r in captured["sources"]["lightrag_relationships"]}
        assert hl_ids == {"rel-e1-e-known"}
        assert not [r for r in incident if r.source in hl_ids]

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

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

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
    entity list in ``_merge_context``. Those ids were once routed ONLY into a
    Neptune neighbourhood expansion, which returns the seeds' neighbours rather
    than the endpoints, so the endpoints never became context items.
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


class TestEndpointsBecomeCandidates:
    """Relationship endpoints are entity context items in their own right.

    Upstream ``_find_most_related_entities_from_relationships`` emits them as
    entities; there is no further multi-hop traversal from them.
    """

    async def test_endpoints_become_candidates_without_graph_traversal(
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
        assert captured["sources"]["lightrag_endpoint_entities"]
        assert neptune_r.calls == []


class TestHybridHasAChunkStream:
    """Upstream `hybrid` builds context from entities + relations + THEIR chunks.

    `_find_related_text_unit_from_entities` / `_from_relations` run for hybrid too.
    What `mix` adds is the separate naive VECTOR chunk query. Gating the lineage
    chunks on mix left hybrid with no chunk stream at all, so it had no passages
    to answer from.
    """

    @staticmethod
    def _lineage_strategy(config: Config, strategy: SearchStrategy):
        spec = get_strategy_spec(strategy)
        st = spec.strategy_class(
            config=config,
            retrievers={
                RetrieverRole.DOCUMENT.value: LineageRetriever(),
                RetrieverRole.GRAPH.value: FakeRetriever("graph"),
            },
        )
        captured: dict = {}

        def _fake_fuse(results_dict, top_k, retrieval_multiplier=1, query=None, **kw):
            captured["sources"] = results_dict
            captured.update(kw)
            return [r for results in results_dict.values() for r in results]

        st.hybrid_scorer.fuse_and_rerank_results = _fake_fuse  # type: ignore[method-assign]
        return st, captured

    async def test_hybrid_retrieves_kg_linked_chunks(self, config: Config) -> None:
        st, captured = self._lineage_strategy(config, SearchStrategy.HYBRID)
        await st.asearch(
            _query(SearchStrategy.HYBRID, ll_keywords=["a"], hl_keywords=["b"])
        )
        sources = captured["sources"]
        assert (
            "lightrag_linked_chunks" in sources
        ), f"hybrid produced no KG-linked chunk stream: {sorted(sources)}"

    async def test_hybrid_does_not_run_the_naive_vector_blend(
        self, config: Config
    ) -> None:
        # That blend is what distinguishes mix from hybrid upstream.
        st, captured = self._lineage_strategy(config, SearchStrategy.HYBRID)
        await st.asearch(
            _query(SearchStrategy.HYBRID, ll_keywords=["a"], hl_keywords=["b"])
        )
        assert "lightrag_chunks" not in captured["sources"]

    async def test_mix_still_runs_both_chunk_streams(self, config: Config) -> None:
        st, captured = self._lineage_strategy(config, SearchStrategy.MIX)
        await st.asearch(
            _query(SearchStrategy.MIX, ll_keywords=["a"], hl_keywords=["b"])
        )
        sources = captured["sources"]
        assert "lightrag_chunks" in sources
        assert any("linked" in key for key in sources)


class TestCallerFiltersReachEverySubQuery:
    """``--filters`` must reach every OpenSearch sub-query LightRAG issues.

    The strategy builds a fresh ``SearchQuery`` per stream; before the fix none
    of them carried ``query.filters``, so a filtered query silently searched the
    whole corpus. Fetch-by-id sub-queries must keep their own id scope too.
    """

    _FILTERS = {"doc_type": "contract", "year": {"gte": 2020}}

    def _strategy(self, config: Config):
        spec = get_strategy_spec(SearchStrategy.MIX)
        os_r, graph_r = LineageRetriever(), FakeRetriever("graph")
        strategy = spec.strategy_class(
            config=config,
            retrievers={
                RetrieverRole.DOCUMENT.value: os_r,
                RetrieverRole.GRAPH.value: graph_r,
            },
        )
        strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
            lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
                r for results in results_dict.values() for r in results
            ]
        )
        return strategy, os_r, graph_r

    async def test_mix_mode_propagates_filters(self, config: Config) -> None:
        strategy, os_r, graph_r = self._strategy(config)
        query = _query(
            SearchStrategy.MIX,
            ll_keywords=["x"],
            hl_keywords=["y"],
            filters=dict(self._FILTERS),
        )

        await strategy.asearch(query)

        os_cfg = config.indexing.opensearch
        queried = {q.index_prefixes[0] for q in os_r.calls}
        assert {
            os_cfg.entities_index_prefix,
            os_cfg.relationships_index_prefix,
            os_cfg.text_units_index_prefix,
        } <= queried
        for call in [*os_r.calls, *graph_r.calls]:
            assert call.filters is not None
            for key, value in self._FILTERS.items():
                assert call.filters[key] == value
        # The linked-chunk fetch keeps its id scope alongside the caller filters.
        linked = [
            q
            for q in os_r.calls
            if q.index_prefixes == [os_cfg.text_units_index_prefix]
            and "id" in (q.filters or {})
        ]
        assert linked and linked[0].filters["id"] == ["chunk-A"]
        # The caller's dict is never mutated by the id scoping.
        assert query.filters == self._FILTERS

    async def test_naive_mode_propagates_filters(self, config: Config) -> None:
        strategy, os_r, _ = _make_strategy(config)

        await strategy.asearch(
            _query(SearchStrategy.NAIVE, filters={"doc_type": "contract"})
        )

        assert [q.filters for q in os_r.calls] == [{"doc_type": "contract"}]

    async def test_no_filters_stays_unfiltered(self, config: Config) -> None:
        strategy, os_r, _ = _make_strategy(config)

        await strategy.asearch(_query(SearchStrategy.NAIVE))

        assert [q.filters for q in os_r.calls] == [None]

    async def test_id_scope_narrows_caller_id_key(self, config: Config) -> None:
        strategy, _, _ = _make_strategy(config)
        query = SearchQuery(query="q", filters={"id": ["b", "caller"], "doc_type": "c"})

        merged = strategy._scoped_filters(query, id=["scoped", "b"])

        # The scope's values the caller allows, in the scope's order.
        assert merged == {"id": ["b"], "doc_type": "c"}
        assert query.filters == {"id": ["b", "caller"], "doc_type": "c"}
        assert strategy._scoped_filters(
            SearchQuery(query="q", filters={"id": "z"}), id=["a"]
        ) == {"id": []}
        assert strategy._scoped_filters(SearchQuery(query="q")) is None
        with pytest.raises(InvalidFilterError, match="range"):
            strategy._scoped_filters(
                SearchQuery(query="q", filters={"id": {"gte": "a"}}), id=["a"]
            )


class _ConcurrencyProbe:
    """Tracks how many aretrieve calls are in flight at once."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return [
            RetrievalResult(
                content="hit",
                score=1.0,
                source="hit-1",
                retriever_type="document",
                metadata={"id": "hit-1", "source_id": "e1", "target_id": "e2"},
            )
        ]


async def test_mix_independent_vector_queries_run_concurrently(
    config: Config,
) -> None:
    # Entities, relationships and the vector chunk query are independent, so
    # they must be in flight together rather than awaited one after another.
    spec = get_strategy_spec(SearchStrategy.MIX)
    probe = _ConcurrencyProbe()
    strategy = spec.strategy_class(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: probe}
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, **_kw: []
    )
    await strategy.asearch(
        _query(SearchStrategy.MIX, ll_keywords=["alice"], hl_keywords=["theme"])
    )
    assert probe.max_in_flight >= 3


async def test_opt_in_graph_expansion_seeds_in_rank_order(config: Config) -> None:
    # enable_graph_expansion restores the Neptune neighbourhood expansion,
    # seeded from the entity hits then the relationship endpoints, in order.
    config.search.lightrag_search.enable_graph_expansion = True
    strategy, os_r, neptune_r = _make_strategy(config)
    result = await strategy.asearch(
        _query(SearchStrategy.HYBRID, ll_keywords=["alice"], hl_keywords=["theme"])
    )
    assert len(neptune_r.calls) == 1
    entity_prefix = config.indexing.opensearch.entities_index_prefix
    assert neptune_r.calls[0].filters["id"][0] == f"{entity_prefix}-id"
    assert result.metadata["sources"]["graph_entities"] == 1


async def test_linked_chunks_keep_their_selection_order(config: Config) -> None:
    # The fetch by id comes back in index order with no score; the weighted
    # polling order (most important KG hit first) must survive it.
    class _Document(FakeRetriever):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            ids = (query.filters or {}).get("id") or []
            return [
                RetrievalResult(
                    content=uid, score=0.0, source=uid, retriever_type="text"
                )
                for uid in sorted(ids)
            ]

    strategy = get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: _Document("document"),
            RetrieverRole.GRAPH.value: FakeRetriever("graph"),
        },
    )
    entities = [
        RetrievalResult(
            content="e",
            score=1.0,
            source=eid,
            retriever_type="entity",
            metadata={"id": eid, "text_unit_ids": units},
        )
        for eid, units in (("e1", ["t-c"]), ("e2", ["t-b", "t-a"]))
    ]

    linked = await strategy._retrieve_linked_chunks(
        SearchQuery(query="q"), entities, []
    )

    chunks = linked["lightrag_linked_chunks"]
    assert [r.source for r in chunks] == ["t-c", "t-a", "t-b"]
    assert [r.score for r in chunks] == sorted((r.score for r in chunks), reverse=True)


def _cites(kind: str, hid: str, units: list[str]) -> RetrievalResult:
    return RetrievalResult(
        content=hid,
        score=1.0,
        source=hid,
        retriever_type=kind,
        metadata={"id": hid, "text_unit_ids": units},
    )


class _ChunksById(FakeRetriever):
    """Chunks by id (in index order, as a store returns them); a text search
    restricted to ids ranks them by the ``ranking`` given."""

    def __init__(self, ranking: list[str] | None = None) -> None:
        super().__init__("document")
        self.ranking = ranking or []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        ids = list((query.filters or {}).get("id") or [])
        if query.query:
            ids = [u for u in self.ranking if u in ids][: query.top_k]
        else:
            ids = sorted(ids)
        return [
            RetrievalResult(content=u, score=0.0, source=u, retriever_type="text")
            for u in ids
        ]


def _linked_strategy(config: Config, document: FakeRetriever):
    return get_strategy_spec(SearchStrategy.MIX).strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: document,
            RetrieverRole.GRAPH.value: FakeRetriever("graph"),
        },
    )


async def test_relationship_chunks_are_a_stream_of_their_own(config: Config) -> None:
    # Both sides' chunks come back in one fetch by id and leave as two
    # streams, so fusion interleaves them instead of the entity side's taking
    # the chunk width first.
    document = _ChunksById()
    strategy = _linked_strategy(config, document)
    hub = _cites("entity", "e1", [f"u{j:02d}" for j in range(30)])
    edge = _cites("relationship", "r1", ["u00", "u12", "u29"])

    linked = await strategy._retrieve_linked_chunks(
        SearchQuery(query="q"), [hub], [edge]
    )

    assert [r.source for r in linked["lightrag_linked_chunks"]] == [
        f"u{j:02d}" for j in range(5)
    ]
    assert [r.source for r in linked["lightrag_relation_chunks"]] == ["u12", "u29"]
    assert len(document.calls) == 1


async def test_fusion_interleaves_the_two_chunk_streams_and_cuts_after(
    config: Config,
) -> None:
    # Through the real scorer, no rerank: 20 entity-side chunks and 2
    # relationship-side ones; the 3-chunk quota keeps a relationship chunk
    # (rank 1 of its stream) beside the entity side's first two.
    config.providers.rerank.kind = "none"
    config.search.fusion.diversity_lambda = 1.0
    config.search.lightrag_search.chunk_stream_top_k = 3
    config.search.lightrag_search.related_chunk_number = 20
    strategy = _linked_strategy(config, _ChunksById())
    hub = _cites("entity", "e1", [f"u{j:02d}" for j in range(20)])
    edge = _cites("relationship", "r1", ["u50", "u51"])
    streams = await strategy._retrieve_linked_chunks(
        SearchQuery(query="q", top_k=3), [hub], [edge]
    )
    fused = strategy.hybrid_scorer.fuse_and_rerank_results(
        streams, top_k=3, per_type_quota={"text": 3}
    )
    assert {r.source for r in fused} == {"u00", "u01", "u50"}


async def test_an_incremental_build_picks_the_chunks_a_full_build_picks(
    config: Config,
) -> None:
    # A full build stores a lineage sorted (the resolver's sorted union); an
    # incremental merge appends the delta's units in arrival order. Picked in
    # stored order, the two builds of one corpus cited different chunks.
    from unified_kg_rag.domain.ingestion.merge.merger import merge_entities
    from unified_kg_rag.domain.models import Entity

    config.search.lightrag_search.related_chunk_number = 2
    stored = Entity(id="ent-vendor", name="Vendor", text_unit_ids=["u7", "u9"])
    delta = Entity(id="ent-vendor", name="Vendor", text_unit_ids=["u1", "u3"])
    (incremental,), _ = merge_entities([stored], [delta])
    full = sorted({*stored.text_unit_ids, *delta.text_unit_ids})
    assert incremental.text_unit_ids != full  # the builds store different orders

    picked = []
    for lineage in (incremental.text_unit_ids, full):
        document = _ChunksById()
        strategy = _linked_strategy(config, document)
        streams = await strategy._retrieve_linked_chunks(
            SearchQuery(query="q"), [_cites("entity", "ent-vendor", lineage)], []
        )
        picked.append([r.source for r in streams["lightrag_linked_chunks"]])
    assert picked[0] == picked[1] == ["u1", "u3"]


class TestVectorChunkPick:
    """``kg_chunk_pick_method: vector``: per side, a search of the cited
    chunks for the query (upstream ``pick_by_vector_similarity``)."""

    @staticmethod
    def _strategy(config: Config, ranking: list[str]):
        config.search.lightrag_search.kg_chunk_pick_method = "vector"
        document = _ChunksById(ranking)
        return _linked_strategy(config, document), document

    async def test_each_side_searches_its_cited_chunks(self, config: Config) -> None:
        strategy, document = self._strategy(config, ["u05", "r2", "u01", "r1", "u09"])
        hits = [_cites("entity", f"e{i}", [f"u{i:02d}"]) for i in range(10)]
        edges = [_cites("relationship", "r", ["r1", "r2", "u05"])]

        streams = await strategy._retrieve_linked_chunks(
            SearchQuery(query="what links them", top_k=5), hits, edges
        )

        # related_chunk_number * 10 hits / 2 = 25, cut to the 10 cited.
        assert [r.source for r in streams["lightrag_linked_chunks"]] == [
            "u05",
            "u01",
            "u09",
        ]
        # The relationship side drops the entity side's pick (u05).
        assert [r.source for r in streams["lightrag_relation_chunks"]] == [
            "r2",
            "r1",
        ]
        assert len(document.calls) == 2
        for call in document.calls:
            assert call.query == "what links them"
            assert call.search_type == SearchType.HYBRID
            assert call.index_prefixes == [
                config.indexing.opensearch.text_units_index_prefix
            ]
        entity_call, relation_call = sorted(
            document.calls, key=lambda q: len(q.filters["id"]), reverse=True
        )
        assert sorted(entity_call.filters["id"]) == [f"u{i:02d}" for i in range(10)]
        assert entity_call.top_k == 10
        # As many more as the entity side may take, for the ones it drops.
        assert sorted(relation_call.filters["id"]) == ["r1", "r2", "u05"]

    async def test_the_vector_stream_and_the_caller_filters_apply(
        self, config: Config
    ) -> None:
        strategy, document = self._strategy(config, ["u1", "u2", "u3"])
        streams = await strategy._retrieve_linked_chunks(
            SearchQuery(query="q", filters={"attr_grp": "a", "id": ["u2", "u3"]}),
            [_cites("entity", "e", ["u1", "u2", "u3"])],
            [],
            exclude={"u3"},
        )
        (call,) = document.calls
        assert call.filters["id"] == ["u2"]
        assert call.filters["attr_grp"] == "a"
        assert [r.source for r in streams["lightrag_linked_chunks"]] == ["u2"]

    async def test_a_side_citing_more_than_one_request_holds_keeps_its_first(
        self, config: Config
    ) -> None:
        strategy, document = self._strategy(config, [])

        async def limits(prefix: str, suffix: str | None) -> IndexLimits:
            return IndexLimits(max_terms_count=3)

        document.aindex_limits = limits  # type: ignore[method-assign]
        hits = [
            _cites("entity", "e0", ["u5", "u6"]),
            _cites("entity", "e1", ["u1", "u2"]),
        ]
        await strategy._retrieve_linked_chunks(SearchQuery(query="q"), hits, [])
        (call,) = document.calls
        assert call.filters["id"] == ["u5", "u6", "u1"]


async def test_a_failed_stream_cancels_the_streams_beside_it(config: Config) -> None:
    # A plain gather raised the first stream's fatal error and left the
    # others (their store requests) running.
    from unified_kg_rag.shared.exceptions import StoreAccessDeniedError

    strategy, _, _ = _make_strategy(config)
    cancelled: list[str] = []

    async def denied(*_args: object) -> dict[str, list[RetrievalResult]]:
        await asyncio.sleep(0.01)
        raise StoreAccessDeniedError("denied")

    async def slow(*_args: object) -> dict[str, list[RetrievalResult]]:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append("chunks")
            raise
        return {}

    strategy._retrieve_entities = denied  # type: ignore[method-assign]
    strategy._retrieve_chunks = slow  # type: ignore[method-assign]
    with pytest.raises(StoreAccessDeniedError):
        await strategy._retrieve_kg_streams(
            _query(SearchStrategy.MIX, ll_keywords=["x"]), SearchStrategy.MIX.value
        )
    await asyncio.sleep(0)
    assert cancelled == ["chunks"]


async def test_a_failed_expansion_cancels_the_one_beside_it(config: Config) -> None:
    # The second round (incident relationships, endpoint entities) too: the
    # other expansion kept going to the store after the query had failed.
    from unified_kg_rag.shared.exceptions import StoreAccessDeniedError

    strategy, _, _ = _make_strategy(config)
    cancelled: list[str] = []

    def hit(kind: str) -> RetrievalResult:
        return RetrievalResult(
            content=kind, score=1.0, source=kind, retriever_type=kind, metadata={}
        )

    async def found(kind: str) -> dict[str, list[RetrievalResult]]:
        return {f"lightrag_{kind}": [hit(kind)]}

    async def denied(*_args: object, **_kw: object) -> dict[str, list[RetrievalResult]]:
        await asyncio.sleep(0.01)
        raise StoreAccessDeniedError("denied")

    async def slow(*_args: object, **_kw: object) -> dict[str, list[RetrievalResult]]:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append("endpoints")
            raise
        return {}

    strategy._retrieve_entities = lambda _q: found("entities")  # type: ignore[method-assign]
    strategy._retrieve_relationships = lambda _q: found("relationships")  # type: ignore[method-assign]
    strategy._retrieve_incident_relationships = denied  # type: ignore[method-assign]
    strategy._retrieve_endpoint_entities = slow  # type: ignore[method-assign]
    with pytest.raises(StoreAccessDeniedError):
        await strategy._retrieve_kg_streams(
            _query(SearchStrategy.HYBRID, ll_keywords=["x"], hl_keywords=["y"]),
            SearchStrategy.HYBRID.value,
        )
    await asyncio.sleep(0)
    assert cancelled == ["endpoints"]
