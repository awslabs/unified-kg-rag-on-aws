# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for GlobalSearchStrategy pure helpers + orchestration branches.

Complements ``test_global_search_scoring.py`` (which covers the 0-10 -> 0-1
relevance-normalization regression). Here we exercise community selection
(static vs dynamic), map-reduce gating, the map-reduce-applied detector, the
relevance-scorer error path, and the retrieve-by-ids / context retrieval
guards. The strategy is built via ``__new__`` so its Bedrock-backed ``__init__``
never runs; retriever and chain objects are replaced with fakes / AsyncMocks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unified_kg_rag.adapters.search_strategies.global_search import GlobalSearchStrategy
from unified_kg_rag.domain.models import RetrievalResult, SearchQuery

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

    async def ainvoke(self, inputs: dict) -> str:
        self.calls.append(inputs)
        return self._value


class _RaisingScorer:
    async def ainvoke(self, _inputs: dict) -> str:
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
        graph_timeout_seconds=5.0,
        map_batch_size=2,
        map_relevance_threshold=0,
        max_map_reduce_tokens=8000,
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


async def test_select_dynamic_sorts_by_blended_score_desc() -> None:
    strat = _bare_strategy(threshold=0.0, use_dynamic_selection=True)
    strat.community_relevance_scorer = _AScorer("7")  # 0.7 normalized, passes
    query = SearchQuery(query="q", retrieval_multiplier=1)
    items = [_community(0, score=0.0), _community(1, score=1.0)]
    kept = await strat._select_relevant_communities(items, query)
    # Both pass threshold 0.0; blended = score*0.4 + 0.7*0.6, so the higher base
    # score sorts first.
    assert [c.source for c in kept] == ["c1", "c0"]
    assert all(0.0 <= c.score <= 1.0 for c in kept)


async def test_select_dynamic_error_with_ignore_errors_drops_item() -> None:
    strat = _bare_strategy(
        threshold=0.1, use_dynamic_selection=True, ignore_errors=True
    )
    strat.community_relevance_scorer = _RaisingScorer()
    query = SearchQuery(query="q", retrieval_multiplier=1)
    kept = await strat._select_relevant_communities(_communities(2), query)
    # relevance_score falls to 0.0 on error; 0.0 < 0.1 threshold -> dropped.
    assert kept == []


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
# _retrieve_documents / _retrieve_reports_by_ids / _retrieve_community_context guards
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


async def test_retrieve_reports_by_ids_empty_ids_returns_empty() -> None:
    strat = _bare_strategy(retrievers={"document": _StubRetriever([_community(0)])})
    assert await strat._retrieve_reports_by_ids([], SearchQuery(query="q")) == []


async def test_retrieve_reports_by_ids_sets_filter_and_top_k() -> None:
    retriever = _StubRetriever([_community(0)])
    strat = _bare_strategy(retrievers={"document": retriever})
    query = SearchQuery(query="original", top_k=99)
    out = await strat._retrieve_reports_by_ids(["a", "b"], query)
    assert [c.source for c in out] == ["c0"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.filters is not None
    assert sent.query == ""  # cleared for an id-filtered lookup
    assert sent.filters["community_id"] == ["a", "b"]
    assert sent.top_k == 2
    assert sent.index_prefixes == ["community_reports"]


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
# _retrieve_community_reports / _retrieve_community_nodes
# --------------------------------------------------------------------------- #


async def test_retrieve_community_reports_uses_community_reports_prefix() -> None:
    retriever = _StubRetriever([_community(0)])
    strat = _bare_strategy(retrievers={"document": retriever})
    out = await strat._retrieve_community_reports(SearchQuery(query="q"))
    assert [c.source for c in out] == ["c0"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.index_prefixes == ["community_reports"]


async def test_retrieve_community_nodes_no_graph_retriever_returns_empty() -> None:
    strat = _bare_strategy(retrievers={})
    out = await strat._retrieve_community_nodes(SearchQuery(query="q"), ["cid"])
    assert out == []


async def test_retrieve_community_nodes_sets_filter_and_community_prefix() -> None:
    retriever = _StubRetriever([_community(0)])
    strat = _bare_strategy(retrievers={"graph": retriever})
    strat.config.indexing.neptune = SimpleNamespace(community_label_prefix="Community")
    out = await strat._retrieve_community_nodes(SearchQuery(query="q"), ["cid-1"])
    assert [c.source for c in out] == ["c0"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.filters is not None
    assert sent.query == ""
    assert sent.filters["id"] == ["cid-1"]
    assert sent.label_prefixes == ["Community"]


async def test_retrieve_community_nodes_swallows_error() -> None:
    strat = _bare_strategy(
        retrievers={"graph": _StubRetriever(raises=RuntimeError("neptune"))}
    )
    strat.config.indexing.neptune = SimpleNamespace(community_label_prefix="Community")
    out = await strat._retrieve_community_nodes(SearchQuery(query="q"), ["cid"])
    assert out == []
