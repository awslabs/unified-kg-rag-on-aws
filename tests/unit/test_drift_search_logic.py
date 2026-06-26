# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for DriftSearchStrategy pure helpers + orchestration branches.

Covers content-hash dedup (``_update_seen_content`` / ``_filter_unique_results``),
result summarization (community vs item formatting + length cap + score sort),
the early-stop heuristics (``_should_stop`` low-gain branch + LLM-convergence
branch with error handling), candidate-entity id resolution, the search-iteration
fan-out (graph + document, exception tolerance) and query evolution (refinement /
keyword expansion, partial failures). The strategy is built via ``__new__`` so its
Bedrock-backed ``__init__`` never runs; chains and retrievers are stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unified_kg_rag.adapters.search_strategies.drift_search import DriftSearchStrategy
from unified_kg_rag.domain.models import RetrievalResult, SearchQuery
from unified_kg_rag.shared.utils import compute_hash

pytestmark = pytest.mark.unit


def _result(
    content: str,
    *,
    score: float = 0.5,
    source: str | None = None,
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        source=source,
        retriever_type="document",
        metadata=metadata or {},
    )


class _AChain:
    """Async chain stub returning a fixed value (or raising) from ainvoke."""

    def __init__(self, value=None, raises: Exception | None = None) -> None:
        self._value = value
        self._raises = raises
        self.calls: list[dict] = []

    async def ainvoke(self, inputs: dict):
        self.calls.append(inputs)
        if self._raises is not None:
            raise self._raises
        return self._value


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
    retrievers: dict | None = None,
    entity_focus_multiplier: int = 2,
    ignore_errors: bool = False,
    enable_query_refinement: bool = True,
    enable_keyword_extraction: bool = True,
    summary_length: int = 5,
    n_entities: int = 5,
    convergence_threshold: float = 0.1,
) -> DriftSearchStrategy:
    strat = DriftSearchStrategy.__new__(DriftSearchStrategy)
    strat.drift_config = SimpleNamespace(
        enable_query_refinement=enable_query_refinement,
        enable_keyword_extraction=enable_keyword_extraction,
        max_iterations=3,
        initial_top_k=5,
        summary_length=summary_length,
        n_entities=n_entities,
        convergence_threshold=convergence_threshold,
        improvement_threshold=0.05,
    )
    strat.entity_focus_multiplier = entity_focus_multiplier
    strat.ignore_errors = ignore_errors
    strat.target_language = "en"
    strat.retrievers = retrievers or {}
    strat.config = SimpleNamespace(
        indexing=SimpleNamespace(
            opensearch=SimpleNamespace(
                community_reports_index_prefix="community_reports",
                entities_index_prefix="entities",
            )
        ),
        search=SimpleNamespace(drift_search=SimpleNamespace(initial_top_k=5)),
    )
    return strat


# --------------------------------------------------------------------------- #
# _update_seen_content + _filter_unique_results (content-hash dedup)
# --------------------------------------------------------------------------- #


def test_update_seen_content_hashes_each_result() -> None:
    seen: set[str] = set()
    results = [_result("alpha"), _result("beta")]
    DriftSearchStrategy._update_seen_content(results, seen)
    assert seen == {compute_hash("alpha", length=16), compute_hash("beta", length=16)}


def test_filter_unique_results_drops_already_seen() -> None:
    seen = {compute_hash("alpha", length=16)}
    results = [_result("alpha"), _result("beta")]
    unique = DriftSearchStrategy._filter_unique_results(results, seen)
    assert [r.content for r in unique] == ["beta"]


def test_filter_unique_results_dedup_roundtrip() -> None:
    # Seeding from a first batch removes those exact contents from a second batch.
    seen: set[str] = set()
    first = [_result("a"), _result("b")]
    DriftSearchStrategy._update_seen_content(first, seen)
    second = [_result("a"), _result("c")]  # "a" repeats, "c" is new
    unique = DriftSearchStrategy._filter_unique_results(second, seen)
    assert [r.content for r in unique] == ["c"]


def test_filter_unique_results_same_content_collapses() -> None:
    # Two results with identical content hash to the same value; both filtered if
    # the content was already seen.
    seen = {compute_hash("dup", length=16)}
    results = [_result("dup"), _result("dup")]
    assert DriftSearchStrategy._filter_unique_results(results, seen) == []


# --------------------------------------------------------------------------- #
# _summarize_results
# --------------------------------------------------------------------------- #


def test_summarize_results_empty_returns_placeholder() -> None:
    strat = _bare_strategy()
    out = strat._summarize_results([])
    assert "No information gathered yet" in out


def test_summarize_results_community_vs_item_formatting() -> None:
    strat = _bare_strategy()
    community = _result(
        "C" * 300, score=0.9, metadata={"_search_index": "community_reports-dev"}
    )
    item = _result("I" * 300, score=0.1, metadata={"_search_index": "text_units-dev"})
    out = strat._summarize_results([item, community])
    lines = out.split("\n")
    # Sorted by score desc -> community (0.9) first; community truncates at 200,
    # item at 150.
    assert lines[0].startswith("Community: ")
    assert lines[1].startswith("Item: ")
    assert len("C" * 200) == 200  # sanity on the truncation length used below
    assert lines[0] == f"Community: {'C' * 200}..."
    assert lines[1] == f"Item: {'I' * 150}..."


def test_summarize_results_caps_at_summary_length() -> None:
    strat = _bare_strategy(summary_length=2)
    results = [_result(f"r{i}", score=float(i)) for i in range(5)]
    out = strat._summarize_results(results)
    assert len(out.split("\n")) == 2


# --------------------------------------------------------------------------- #
# _should_stop (early-convergence heuristics)
# --------------------------------------------------------------------------- #


async def test_should_stop_false_for_early_iterations() -> None:
    strat = _bare_strategy()
    # iteration <= 1 -> never the low-gain branch; metrics empty -> no LLM stop.
    assert await strat._should_stop(0, [], "q") is False
    assert await strat._should_stop(1, [], "q") is False


async def test_should_stop_true_on_consecutive_low_gains() -> None:
    strat = _bare_strategy()
    metrics = [{"unique_new": 0}, {"unique_new": 1}]  # last two both < 2
    assert await strat._should_stop(2, metrics, "q") is True


async def test_should_stop_false_when_gains_above_floor() -> None:
    strat = _bare_strategy()
    metrics = [{"unique_new": 5}, {"unique_new": 5}]
    # iteration 2 -> low-gain branch evaluated but gains high; LLM branch needs
    # iteration > 2 so it is not consulted here.
    assert await strat._should_stop(2, metrics, "q") is False


async def test_should_stop_consults_llm_after_iteration_two() -> None:
    strat = _bare_strategy(convergence_threshold=0.1)
    strat.convergence_assessor = _AChain("0.9")  # >= threshold -> converged
    metrics = [{"unique_new": 5}, {"unique_new": 5}]  # high gains, skip low-gain stop
    assert await strat._should_stop(3, metrics, "q") is True


async def test_should_stop_llm_below_threshold_continues() -> None:
    strat = _bare_strategy(convergence_threshold=0.5)
    strat.convergence_assessor = _AChain("0.1")  # < threshold -> not converged
    metrics = [{"unique_new": 5}, {"unique_new": 5}]
    assert await strat._should_stop(3, metrics, "q") is False


# --------------------------------------------------------------------------- #
# _assess_convergence_with_llm
# --------------------------------------------------------------------------- #


async def test_assess_convergence_empty_metrics_false() -> None:
    strat = _bare_strategy()
    strat.convergence_assessor = _AChain("1.0")
    assert await strat._assess_convergence_with_llm("q", 3, []) is False


async def test_assess_convergence_parses_score_against_threshold() -> None:
    strat = _bare_strategy(convergence_threshold=0.3)
    strat.convergence_assessor = _AChain("0.4")
    metrics = [{"unique_new": 2}]
    assert await strat._assess_convergence_with_llm("q", 3, metrics) is True


async def test_assess_convergence_error_ignored_returns_false() -> None:
    strat = _bare_strategy(ignore_errors=True)
    strat.convergence_assessor = _AChain(raises=RuntimeError("bedrock"))
    metrics = [{"unique_new": 2}]
    assert await strat._assess_convergence_with_llm("q", 3, metrics) is False


async def test_assess_convergence_error_propagates_when_not_ignored() -> None:
    strat = _bare_strategy(ignore_errors=False)
    strat.convergence_assessor = _AChain(raises=RuntimeError("bedrock"))
    metrics = [{"unique_new": 2}]
    with pytest.raises(RuntimeError):
        await strat._assess_convergence_with_llm("q", 3, metrics)


# --------------------------------------------------------------------------- #
# _find_candidate_entities_for_iteration
# --------------------------------------------------------------------------- #


async def test_find_candidate_entities_no_retriever_empty() -> None:
    strat = _bare_strategy(retrievers={})
    assert (
        await strat._find_candidate_entities_for_iteration(SearchQuery(query="q")) == []
    )


async def test_find_candidate_entities_prefers_metadata_id_over_source() -> None:
    retriever = _StubRetriever(
        [
            _result("x", source="src-1", metadata={"id": "meta-1"}),
            _result("y", source="src-2", metadata={}),  # falls back to source
        ]
    )
    strat = _bare_strategy(
        retrievers={"document": retriever}, entity_focus_multiplier=3
    )
    query = SearchQuery(query="q", entity_focus=["Alice", "Bob"])
    out = await strat._find_candidate_entities_for_iteration(query)
    assert out == ["meta-1", "src-2"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.index_prefixes == ["entities"]
    assert sent.top_k == 6  # 2 focus * multiplier 3
    assert sent.retrieval_multiplier == 1


async def test_find_candidate_entities_swallows_error() -> None:
    strat = _bare_strategy(
        retrievers={"document": _StubRetriever(raises=RuntimeError("os"))}
    )
    query = SearchQuery(query="q", entity_focus=["Alice"])
    assert await strat._find_candidate_entities_for_iteration(query) == []


# --------------------------------------------------------------------------- #
# _execute_search_iteration (fan-out, exception tolerance)
# --------------------------------------------------------------------------- #


async def test_execute_search_iteration_no_retrievers_returns_empty() -> None:
    strat = _bare_strategy(retrievers={})
    assert await strat._execute_search_iteration(SearchQuery(query="q")) == []


async def test_execute_search_iteration_merges_graph_and_document(mocker) -> None:
    graph = _StubRetriever([_result("g1")])
    document = _StubRetriever([_result("d1"), _result("d2")])
    strat = _bare_strategy(retrievers={"graph": graph, "document": document})
    # Force a non-empty candidate-entity list so the graph branch runs.
    mocker.patch.object(
        strat,
        "_find_candidate_entities_for_iteration",
        mocker.AsyncMock(return_value=["e1"]),
    )
    out = await strat._execute_search_iteration(SearchQuery(query="q", top_k=7))
    contents = {r.content for r in out}
    assert contents == {"g1", "d1", "d2"}
    # The graph sub-query is id-filtered and focus-cleared.
    graph_sent = graph.last_query
    document_sent = document.last_query
    assert graph_sent is not None and graph_sent.filters is not None
    assert document_sent is not None
    assert graph_sent.filters["id"] == ["e1"]
    assert graph_sent.entity_focus == []
    assert document_sent.top_k == 7


async def test_execute_search_iteration_tolerates_retriever_exception(mocker) -> None:
    graph = _StubRetriever(raises=RuntimeError("neptune"))
    document = _StubRetriever([_result("d1")])
    strat = _bare_strategy(retrievers={"graph": graph, "document": document})
    mocker.patch.object(
        strat,
        "_find_candidate_entities_for_iteration",
        mocker.AsyncMock(return_value=["e1"]),
    )
    out = await strat._execute_search_iteration(SearchQuery(query="q"))
    # gather(return_exceptions=True): the failed graph list is skipped, document
    # results survive.
    assert [r.content for r in out] == ["d1"]


async def test_execute_search_iteration_skips_graph_without_candidates(mocker) -> None:
    document = _StubRetriever([_result("d1")])
    strat = _bare_strategy(
        retrievers={"graph": _StubRetriever([_result("g1")]), "document": document}
    )
    mocker.patch.object(
        strat,
        "_find_candidate_entities_for_iteration",
        mocker.AsyncMock(return_value=[]),
    )
    out = await strat._execute_search_iteration(SearchQuery(query="q"))
    # No candidate entities -> graph branch skipped, only document results.
    assert [r.content for r in out] == ["d1"]


# --------------------------------------------------------------------------- #
# _evolve_query
# --------------------------------------------------------------------------- #


async def test_evolve_query_no_tasks_returns_copy_unchanged() -> None:
    strat = _bare_strategy(
        enable_query_refinement=False, enable_keyword_extraction=False
    )
    query = SearchQuery(query="orig", optional_keywords=["kw"])
    out = await strat._evolve_query(query, "orig", [], 0)
    assert out is not query  # deep copy
    assert out.query == "orig"
    assert out.optional_keywords == ["kw"]


async def test_evolve_query_applies_refinement_and_expansion() -> None:
    strat = _bare_strategy()
    strat.query_refiner = _AChain("  refined query  ")
    strat.keyword_expander = _AChain(["k1", "k2"])
    query = SearchQuery(query="start")
    results = [_result("r", metadata={"name": "Alice"})]
    out = await strat._evolve_query(query, "original", results, 1)
    assert out.query == "refined query"  # stripped
    assert out.optional_keywords == ["k1", "k2"]
    # The keyword expander receives the top-n entity names.
    assert strat.keyword_expander.calls[0]["entities"] == ["Alice"]


async def test_evolve_query_blank_refinement_keeps_original() -> None:
    strat = _bare_strategy(enable_keyword_extraction=False)
    strat.query_refiner = _AChain("   ")  # blank -> ignored
    query = SearchQuery(query="keepme")
    out = await strat._evolve_query(query, "original", [], 0)
    assert out.query == "keepme"


async def test_evolve_query_partial_failure_via_gather(mocker) -> None:
    # Refinement raises, expansion succeeds; return_exceptions keeps the good one.
    strat = _bare_strategy()
    strat.query_refiner = _AChain(raises=RuntimeError("refine boom"))
    strat.keyword_expander = _AChain(["only-kw"])
    query = SearchQuery(query="start")
    out = await strat._evolve_query(query, "original", [], 1)
    assert out.query == "start"  # refinement failed, unchanged
    assert out.optional_keywords == ["only-kw"]


async def test_evolve_query_empty_expansion_list_ignored() -> None:
    strat = _bare_strategy(enable_query_refinement=False)
    strat.keyword_expander = _AChain([])  # empty -> not applied
    query = SearchQuery(query="start", optional_keywords=["prev"])
    out = await strat._evolve_query(query, "original", [], 0)
    assert out.optional_keywords == ["prev"]


# --------------------------------------------------------------------------- #
# _find_candidate_communities
# --------------------------------------------------------------------------- #


async def test_find_candidate_communities_no_retriever_empty() -> None:
    strat = _bare_strategy(retrievers={})
    assert await strat._find_candidate_communities(SearchQuery(query="q")) == []


async def test_find_candidate_communities_sets_prefix_and_top_k() -> None:
    retriever = _StubRetriever([_result("cr1")])
    strat = _bare_strategy(retrievers={"document": retriever})
    out = await strat._find_candidate_communities(SearchQuery(query="q"))
    assert [r.content for r in out] == ["cr1"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.index_prefixes == ["community_reports"]
    assert sent.top_k == 5  # initial_top_k


async def test_find_candidate_communities_swallows_error() -> None:
    strat = _bare_strategy(
        retrievers={"document": _StubRetriever(raises=RuntimeError("os"))}
    )
    assert await strat._find_candidate_communities(SearchQuery(query="q")) == []


# --------------------------------------------------------------------------- #
# _record_search_metrics
# --------------------------------------------------------------------------- #


def test_record_search_metrics_populates_metrics() -> None:
    strat = _bare_strategy()
    strat._metrics = {"timings": {}, "metrics": {}}
    strat._record_search_metrics(2.0, results_count=11, iterations=3)
    assert strat._metrics["timings"]["processing_time"] == 2.0
    assert strat._metrics["metrics"] == {
        "retrieved_count": 11,
        "iterations_completed": 3,
    }
