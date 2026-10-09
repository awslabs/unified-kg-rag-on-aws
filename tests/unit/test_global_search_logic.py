# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for GlobalSearchStrategy pure helpers + orchestration branches.

Complements ``test_global_search_scoring.py`` (which covers the 0-10 -> 0-1
relevance-normalization regression). Here we exercise community selection
(static vs dynamic), map-reduce gating, the map-reduce-applied detector, the
relevance-scorer error path, and the report / context retrieval
guards. The strategy is built via ``__new__`` so its Bedrock-backed ``__init__``
never runs; retriever and chain objects are replaced with fakes / AsyncMocks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unified_kg_rag.adapters.search_strategies.global_search import GlobalSearchStrategy
from unified_kg_rag.domain.models import RetrievalResult, SearchQuery
from unified_kg_rag.shared.utils.langchain import BatchProcessor

pytestmark = pytest.mark.unit


def _community(
    i: int, *, score: float = 0.5, metadata: dict | None = None
) -> RetrievalResult:
    return RetrievalResult(
        content=f"community {i}",
        score=score,
        source=f"c{i}",
        retriever_type="graph",
        metadata=metadata or {},
    )


def _communities(n: int) -> list[RetrievalResult]:
    return [_community(i) for i in range(n)]


class _AScorer:
    """Async runnable returning a fixed string from ainvoke."""

    def __init__(self, value: str) -> None:
        self._value = value
        self.calls: list[dict] = []
        self.configs: list = []

    async def ainvoke(self, inputs: dict, config=None) -> str:
        self.calls.append(inputs)
        self.configs.append(config)
        return self._value


class _RaisingScorer:
    async def ainvoke(self, _inputs: dict, config=None) -> str:
        raise RuntimeError("bedrock down")


class _StubRetriever:
    def __init__(self, results=None, raises: Exception | None = None) -> None:
        self._results = results or []
        self._raises = raises
        self.last_query: SearchQuery | None = None

    async def aretrieve(self, query: SearchQuery):
        self.last_query = query
        if self._raises is not None:
            raise self._raises
        return self._results


def _bare_strategy(
    *,
    threshold: float = 0.5,
    use_dynamic_selection: bool = True,
    max_communities: int = 100,
    ignore_errors: bool = False,
    enable_map_reduce: bool = True,
    map_reduce_min_results: int = 3,
    retrievers: dict | None = None,
) -> GlobalSearchStrategy:
    strat = GlobalSearchStrategy.__new__(GlobalSearchStrategy)
    strat.global_search_config = SimpleNamespace(
        max_communities=max_communities,
        use_dynamic_selection=use_dynamic_selection,
        relevance_threshold=threshold,
        enable_map_reduce=enable_map_reduce,
        map_reduce_min_results=map_reduce_min_results,
        max_text_units=10,
        map_batch_size=2,
        map_relevance_threshold=0,
        max_map_reduce_tokens=8000,
        reduce_with_llm=False,
        reserve_report_slots=True,
        text_unit_slots=None,
    )
    strat.ignore_errors = ignore_errors
    strat.target_language = "en"
    strat.retrievers = retrievers or {}
    strat.config = SimpleNamespace(
        indexing=SimpleNamespace(
            opensearch=SimpleNamespace(
                community_reports_index_prefix="community_reports",
                text_units_index_prefix="text_units",
            )
        )
    )
    return strat


# --------------------------------------------------------------------------- #
# _get_ids with the community_id key
# --------------------------------------------------------------------------- #


def test_get_ids_community_id_key() -> None:
    results = [
        _community(0, metadata={"community_id": "cid-1"}),
        _community(1, metadata={"community_id": "cid-2"}),
        _community(2, metadata={}),  # no community_id, no source fallback for this key
    ]
    assert set(GlobalSearchStrategy._get_ids(results, "community_id")) == {
        "cid-1",
        "cid-2",
    }


# --------------------------------------------------------------------------- #
# _select_relevant_communities — static path
# --------------------------------------------------------------------------- #


async def test_select_static_returns_top_max_communities() -> None:
    strat = _bare_strategy(use_dynamic_selection=False, max_communities=2)
    query = SearchQuery(query="q", retrieval_multiplier=1)
    kept = await strat._select_relevant_communities(_communities(5), query)
    assert [c.source for c in kept] == ["c0", "c1"]


async def test_select_static_honors_retrieval_multiplier() -> None:
    strat = _bare_strategy(use_dynamic_selection=False, max_communities=2)
    query = SearchQuery(query="q", retrieval_multiplier=2)
    kept = await strat._select_relevant_communities(_communities(5), query)
    # max = max_communities (2) * multiplier (2) = 4.
    assert len(kept) == 4


# --------------------------------------------------------------------------- #
# _select_relevant_communities — dynamic path (sorting, blending, error path)
# --------------------------------------------------------------------------- #


async def test_select_dynamic_ties_keep_retrieval_order() -> None:
    # Equal LLM relevance: retrieval rank decides, not the (differently
    # scaled) retrieval score.
    strat = _bare_strategy(threshold=0.0, use_dynamic_selection=True)
    strat.community_relevance_scorer = _AScorer("7")  # 0.7 normalized, passes
    query = SearchQuery(query="q", retrieval_multiplier=1)
    items = [_community(0, score=0.0), _community(1, score=1.0)]
    kept = await strat._select_relevant_communities(items, query)
    assert [c.source for c in kept] == ["c0", "c1"]
    assert all(c.score == pytest.approx(0.7) for c in kept)


async def test_select_dynamic_ranks_by_llm_relevance() -> None:
    strat = _bare_strategy(threshold=0.0, use_dynamic_selection=True)

    class _ByContent:
        async def ainvoke(self, inputs, config=None):
            return "9" if "1" in inputs["community_summary"] else "3"

    strat.community_relevance_scorer = _ByContent()
    query = SearchQuery(query="q", retrieval_multiplier=1)
    items = [_community(0, score=1.0), _community(1, score=0.0)]
    kept = await strat._select_relevant_communities(items, query)
    assert [c.source for c in kept] == ["c1", "c0"]


async def test_select_dynamic_error_with_ignore_errors_keeps_at_retrieval_score() -> (
    None
):
    strat = _bare_strategy(
        threshold=0.1, use_dynamic_selection=True, ignore_errors=True
    )
    strat.community_relevance_scorer = _RaisingScorer()
    query = SearchQuery(query="q", retrieval_multiplier=1)
    kept = await strat._select_relevant_communities(_communities(2), query)
    # A scoring failure must NOT silently drop a community: it is admitted at
    # the threshold relevance (ranked after scored ones) on a copy.
    assert len(kept) == 2
    assert all(c.score == pytest.approx(0.1) for c in kept)


async def test_select_dynamic_does_not_mutate_input_items() -> None:
    # The scored objects must be copies: the input list is shared with the
    # fallback set and downstream fusion, so blending must not mutate it.
    strat = _bare_strategy(threshold=0.0, use_dynamic_selection=True)
    strat.community_relevance_scorer = _AScorer("9")  # 0.9 normalized
    query = SearchQuery(query="q", retrieval_multiplier=1)
    items = [_community(0, score=0.2)]
    kept = await strat._select_relevant_communities(items, query)
    assert items[0].score == 0.2  # original untouched
    assert kept[0].score == pytest.approx(0.9)  # copy carries the relevance


async def test_select_dynamic_error_without_ignore_errors_raises() -> None:
    strat = _bare_strategy(use_dynamic_selection=True, ignore_errors=False)
    strat.community_relevance_scorer = _RaisingScorer()
    query = SearchQuery(query="q", retrieval_multiplier=1)
    with pytest.raises(RuntimeError):
        await strat._select_relevant_communities(_communities(1), query)


# --------------------------------------------------------------------------- #
# _apply_map_reduce
# --------------------------------------------------------------------------- #


async def test_apply_map_reduce_below_min_returns_unchanged() -> None:
    strat = _bare_strategy(map_reduce_min_results=3)
    strat.map_reducer = _AScorer("summary")
    results = _communities(2)  # 2 < 3 -> no synthesis
    query = SearchQuery(query="q")
    out = await strat._apply_map_reduce(results, query)
    assert out is results


async def test_apply_map_reduce_below_min_results_returns_unchanged() -> None:
    # Below map_reduce_min_results, the map-reduce pipeline is skipped and the
    # results pass through untouched. (The full map→filter→rank→reduce pipeline
    # is covered in test_global_search_map_reduce.py.)
    strat = _bare_strategy(map_reduce_min_results=5)
    results = _communities(3)
    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))
    assert out is results


# --------------------------------------------------------------------------- #
# _was_map_reduce_applied
# --------------------------------------------------------------------------- #


def test_was_map_reduce_applied_disabled_is_false() -> None:
    strat = _bare_strategy(enable_map_reduce=False)
    results = [_community(0)]
    results[0].source = "synthesized_summary"
    assert strat._was_map_reduce_applied(results) is False


def test_was_map_reduce_applied_detects_synthesized_source() -> None:
    strat = _bare_strategy(enable_map_reduce=True)
    summary = RetrievalResult(
        content="s", score=1.0, source="synthesized_summary", retriever_type="general"
    )
    assert strat._was_map_reduce_applied([summary, _community(0)]) is True


def test_was_map_reduce_applied_no_synthesized_is_false() -> None:
    strat = _bare_strategy(enable_map_reduce=True)
    assert strat._was_map_reduce_applied(_communities(3)) is False


# --------------------------------------------------------------------------- #
# _retrieve_documents / _retrieve_community_context guards
# --------------------------------------------------------------------------- #


async def test_retrieve_documents_no_retriever_returns_empty() -> None:
    strat = _bare_strategy(retrievers={})
    out = await strat._retrieve_documents(SearchQuery(query="q"), ["community_reports"])
    assert out == []


async def test_retrieve_documents_sets_index_prefixes() -> None:
    retriever = _StubRetriever([_community(0)])
    strat = _bare_strategy(retrievers={"document": retriever})
    query = SearchQuery(query="q")
    out = await strat._retrieve_documents(query, ["community_reports"])
    assert [c.source for c in out] == ["c0"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.index_prefixes == ["community_reports"]
    # Original query untouched (deep copy).
    assert query.index_prefixes is None


async def test_retrieve_documents_swallows_error() -> None:
    strat = _bare_strategy(
        retrievers={"document": _StubRetriever(raises=RuntimeError("os"))}
    )
    out = await strat._retrieve_documents(SearchQuery(query="q"), ["x"])
    assert out == []


async def test_retrieve_community_context_no_community_ids_returns_empty() -> None:
    strat = _bare_strategy(retrievers={"document": _StubRetriever([_community(0)])})
    # Communities lack community_id metadata -> no ids -> empty without a call.
    communities = [_community(0), _community(1)]
    out = await strat._retrieve_community_context(communities, SearchQuery(query="q"))
    assert out == []


async def test_retrieve_community_context_caps_top_k_at_max_text_units() -> None:
    retriever = _StubRetriever([_community(5)])
    strat = _bare_strategy(retrievers={"document": retriever})
    strat.global_search_config.max_text_units = 4
    communities = [_community(0, metadata={"community_id": "cid"})]
    query = SearchQuery(query="q", top_k=100)
    out = await strat._retrieve_community_context(communities, query)
    assert [c.source for c in out] == ["c5"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.filters is not None
    assert sent.top_k == 4  # min(top_k=100, max_text_units=4)
    assert sent.filters["community_ids"] == ["cid"]


# --------------------------------------------------------------------------- #
# _augment_and_rerank_communities — empty-selection fallback
# --------------------------------------------------------------------------- #


async def test_augment_empty_selection_returns_fallback() -> None:
    strat = _bare_strategy()
    fallback = _communities(3)
    out = await strat._augment_and_rerank_communities(
        [], fallback, SearchQuery(query="q")
    )
    assert out is fallback


# --------------------------------------------------------------------------- #
# _retrieve_community_reports
# --------------------------------------------------------------------------- #


async def test_retrieve_community_reports_uses_community_reports_prefix() -> None:
    retriever = _StubRetriever([_community(0)])
    strat = _bare_strategy(retrievers={"document": retriever})
    out = await strat._retrieve_community_reports(SearchQuery(query="q"))
    assert [c.source for c in out] == ["c0"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.index_prefixes == ["community_reports"]


async def test_reports_are_fused_without_a_graph_round_trip() -> None:
    # The Neptune community expansion returned only the candidate communities
    # (its limit equalled their count and they were emitted first), so it was
    # removed: one report bucket is fused, and the graph is never queried.
    candidates = [_community(0, metadata={"community_id": "cid0"}), _community(1)]
    graph = _StubRetriever(raises=AssertionError("graph must not be queried"))
    strat = _bare_strategy(
        retrievers={"document": _StubRetriever(candidates), "graph": graph}
    )
    fused: list[dict] = []

    async def _fuse(results, **kwargs):
        fused.append(results)
        return results["opensearch_candidate_community_reports"]

    strat._fuse_and_rerank = _fuse  # type: ignore[method-assign]
    out = await strat._retrieve_and_fuse_communities(SearchQuery(query="q"))
    assert out == candidates
    assert [list(buckets) for buckets in fused] == [
        ["opensearch_candidate_community_reports"]
    ]
    assert graph.last_query is None


# --- _parse_map_points robustness to non-finite LLM scores (R3 fix) ---


def test_parse_map_points_handles_infinity_and_nan() -> None:
    # parse_llm_json (json.loads) accepts Infinity/-Infinity/NaN, and "1e999"
    # overflows float() to inf; int(inf)/int(nan) would crash the whole query.
    # Non-finite scores must be coerced to 0 (dropped), not raise.
    import json

    raw = json.dumps(
        {"points": [{"description": "a", "score": 5}, {"description": "b", "score": 1}]}
    )
    # Inject the non-standard literals json.dumps won't emit but json.loads reads.
    raw_bad = '{"points": [{"description": "inf", "score": Infinity}, {"description": "nan", "score": NaN}, {"description": "ok", "score": 7}]}'

    pts = GlobalSearchStrategy._parse_map_points(raw_bad)
    scores = {p.description: p.score for p in pts}
    assert scores["inf"] == 0  # coerced, not crashed
    assert scores["nan"] == 0
    assert scores["ok"] == 7
    # And a normal payload still parses.
    assert len(GlobalSearchStrategy._parse_map_points(raw)) == 2


def test_parse_map_points_overflow_string_score() -> None:
    raw = '{"points": [{"description": "big", "score": "1e999"}]}'
    pts = GlobalSearchStrategy._parse_map_points(raw)
    assert pts[0].score == 0  # 1e999 -> inf -> coerced to 0


def test_default_global_config_skips_per_report_llm_scoring() -> None:
    from unified_kg_rag.domain.models import Config

    config = Config()
    gs = config.search.global_search
    # The map step already rates the reports; the per-report pre-filter is opt-in.
    assert gs.enable_map_reduce is True
    assert gs.use_dynamic_selection is False
    assert gs.map_batch_size == 5
    strategy = GlobalSearchStrategy(config=config, retrievers={})
    assert not hasattr(strategy, "community_relevance_scorer")


# --------------------------------------------------------------------------- #
# Reserved community-report slots (reserve_report_slots)
# --------------------------------------------------------------------------- #


def _typed(prefix: str, rtype: str, n: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content=f"{prefix} {i} " + " ".join(f"w{prefix}{i}{j}" for j in range(5)),
            score=1.0 - i * 0.01,
            source=f"{prefix}{i}",
            retriever_type=rtype,
            metadata={"community_id": f"{prefix}{i}"} if rtype == "community" else {},
        )
        for i in range(n)
    ]


def _fusing_strategy(reserve: bool) -> GlobalSearchStrategy:
    from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
    from unified_kg_rag.domain.models import Config

    config = Config()
    config.search.reranking.enabled = False
    strat = _bare_strategy(max_communities=10)
    strat.global_search_config.reserve_report_slots = reserve
    strat.hybrid_scorer = HybridScorer(config)
    return strat


@pytest.mark.parametrize("reserve", [True, False])
async def test_report_slots_keep_every_selected_report(reserve: bool) -> None:
    strat = _fusing_strategy(reserve)
    reports = _typed("r", "community", 10)
    chunks = _typed("t", "text", 30)

    async def _context(_communities, _query):
        return chunks

    strat._retrieve_community_context = _context  # type: ignore[method-assign]

    out = await strat._augment_and_rerank_communities(
        reports, reports, SearchQuery(query="q", top_k=10)
    )

    kept_reports = [r for r in out if r.retriever_type == "community"]
    kept_chunks = [r for r in out if r.retriever_type == "text"]
    if reserve:
        assert len(kept_reports) == 10
        assert len(kept_chunks) == 10  # text_unit_slots defaults to top_k
    else:
        # The previous flat cut: reports and chunks share ten slots.
        assert len(out) == 10
        assert len(kept_reports) < 10


def test_report_quota_honors_text_unit_slots_and_multiplier() -> None:
    strat = _bare_strategy(max_communities=4)
    strat.global_search_config.text_unit_slots = 3
    quota = strat._report_quota(
        SearchQuery(query="q", top_k=10, retrieval_multiplier=2)
    )
    assert quota == {"community": 8, "text": 3}
    strat.global_search_config.reserve_report_slots = False
    assert strat._report_quota(SearchQuery(query="q")) is None


@pytest.mark.parametrize(("reserve", "expected"), [(True, 21), (False, 10)])
async def test_synthesized_item_does_not_take_an_evidence_slot(
    reserve: bool, expected: int
) -> None:
    from unittest.mock import AsyncMock

    strat = _bare_strategy()
    strat.global_search_config.reserve_report_slots = reserve
    evidence = _typed("r", "community", 10) + _typed("t", "text", 10)
    synthesized = RetrievalResult(
        content="points",
        score=1.0,
        source="synthesized_key_points",
        retriever_type="general",
        metadata={"synthesized": True},
    )
    strat._retrieve_and_fuse_communities = AsyncMock(return_value=evidence)  # type: ignore[method-assign]
    strat._select_relevant_communities = AsyncMock(return_value=evidence[:10])  # type: ignore[method-assign]
    strat._augment_and_rerank_communities = AsyncMock(return_value=evidence)  # type: ignore[method-assign]
    strat._apply_map_reduce = AsyncMock(return_value=[synthesized, *evidence])  # type: ignore[method-assign]
    strat._record_search_metrics = lambda *a, **k: None  # type: ignore[method-assign]

    result = await strat.asearch(SearchQuery(query="q", top_k=10))

    assert len(result.results) == expected
    assert result.results[0].source == "synthesized_key_points"


# --------------------------------------------------------------------------- #
# The caller's RunnableConfig reaches every global-search LLM call
# --------------------------------------------------------------------------- #


class _MapRater:
    def __init__(self) -> None:
        self.configs: list = []

    def invoke(self, single_input, config=None):  # noqa: ANN001
        self.configs.append(config)
        return '{"points": [{"description": "Vendor ships", "score": 80}]}'


async def test_select_dynamic_passes_the_callers_config() -> None:
    strat = _bare_strategy(use_dynamic_selection=True, threshold=0.0)
    strat.community_relevance_scorer = _AScorer("8")
    caller_config = {"tags": ["caller"]}
    await strat._select_relevant_communities(
        _communities(2), SearchQuery(query="q"), config=caller_config
    )
    assert strat.community_relevance_scorer.configs == [caller_config] * 2


async def test_map_reduce_passes_the_callers_config_to_map_and_reduce() -> None:
    strat = _bare_strategy(map_reduce_min_results=1)
    strat.global_search_config.reduce_with_llm = True
    strat.map_rater = _MapRater()
    strat.map_reducer = _AScorer("summary")
    strat.batch_processor = BatchProcessor(
        batch_size=1, max_concurrency=4, max_attempts=1
    )
    strat.token_manager = SimpleNamespace(
        count_tokens=len, count_tokens_many=lambda texts: [len(t) for t in texts]
    )
    caller_config = {"tags": ["caller"]}

    await strat._apply_map_reduce(
        _communities(2), SearchQuery(query="q"), config=caller_config
    )

    # The map call gets the caller's config.
    assert strat.map_rater.configs == [caller_config]
    assert strat.map_reducer.configs == [caller_config]
