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
from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery
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
        self.configs: list = []

    async def ainvoke(self, inputs: dict, config=None):
        self.calls.append(inputs)
        self.configs.append(config)
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
    enable_llm_convergence: bool = True,
) -> DriftSearchStrategy:
    strat = DriftSearchStrategy.__new__(DriftSearchStrategy)
    strat.drift_config = SimpleNamespace(
        enable_query_refinement=enable_query_refinement,
        enable_keyword_extraction=enable_keyword_extraction,
        max_iterations=3,
        initial_top_k=5,
        summary_length=summary_length,
        summary_char_limit=200,
        n_entities=n_entities,
        convergence_threshold=convergence_threshold,
        enable_llm_convergence=enable_llm_convergence,
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
        search=SimpleNamespace(
            drift_search=SimpleNamespace(initial_top_k=5),
            local_search=Config().search.local_search,
        ),
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
    # Sorted by score desc -> community (0.9) first. Both community and item are
    # capped at the configurable summary_char_limit (200 here).
    assert lines[0].startswith("Community: ")
    assert lines[1].startswith("Item: ")
    assert lines[0] == f"Community: {'C' * 200}..."
    assert lines[1] == f"Item: {'I' * 200}..."


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
    strat = _bare_strategy(enable_llm_convergence=False)
    metrics = [{"unique_new": 5}, {"unique_new": 5}]
    # Low-gain branch evaluated but gains high; the LLM check is disabled.
    assert await strat._should_stop(2, metrics, "q") is False


async def test_should_stop_consults_llm_once_an_iteration_ran() -> None:
    # Reachable within the default max_iterations=3 (iterations 0, 1, 2): the
    # old `iteration > 2` gate could never fire.
    strat = _bare_strategy(convergence_threshold=0.1)
    strat.convergence_assessor = _AChain("0.9")  # >= threshold -> converged
    metrics = [{"unique_new": 5}]  # high gain, skip low-gain stop
    assert await strat._should_stop(1, metrics, "q") is True


async def test_should_stop_skips_llm_when_disabled() -> None:
    strat = _bare_strategy(enable_llm_convergence=False)
    strat.convergence_assessor = _AChain(raises=AssertionError("must not run"))
    metrics = [{"unique_new": 5}, {"unique_new": 5}]
    assert await strat._should_stop(2, metrics, "q") is False


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


async def test_find_candidate_entities_falls_back_to_query_text() -> None:
    # Regression: with no extracted entity focus, n_candidates was
    # 0 * multiplier = 0, so a DRIFT iteration seeded no graph expansion.
    retriever = _StubRetriever([_result("x", source="e-1", metadata={"id": "e-1"})])
    strat = _bare_strategy(retrievers={"document": retriever})
    query = SearchQuery(query="follow-up about the vendor", top_k=7)
    assert await strat._find_candidate_entities_for_iteration(query) == ["e-1"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.query == "follow-up about the vendor"
    assert sent.top_k == 7
    assert sent.index_prefixes == ["entities"]


async def test_find_candidate_entities_empty_query_and_focus_short_circuits() -> None:
    retriever = _StubRetriever([_result("x", source="e-1")])
    strat = _bare_strategy(retrievers={"document": retriever})
    assert (
        await strat._find_candidate_entities_for_iteration(SearchQuery(query="")) == []
    )
    assert retriever.last_query is None


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
    # A transient failure degrades that branch to []; document results survive.
    assert [r.content for r in out] == ["d1"]


async def test_execute_search_iteration_propagates_fatal_error(mocker) -> None:
    # gather(return_exceptions=True) used to swallow fatal errors too, so an
    # AccessDenied from Neptune read as "no graph results" on every iteration.
    graph = _StubRetriever(raises=RuntimeError("AccessDeniedException: neptune-db"))
    document = _StubRetriever([_result("d1")])
    strat = _bare_strategy(retrievers={"graph": graph, "document": document})
    mocker.patch.object(
        strat,
        "_find_candidate_entities_for_iteration",
        mocker.AsyncMock(return_value=["e1"]),
    )
    with pytest.raises(RuntimeError, match="AccessDenied"):
        await strat._execute_search_iteration(SearchQuery(query="q"))


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


# --------------------------------------------------------------------------- #
# DRIFT primer -> follow-up (enable_primer)
# --------------------------------------------------------------------------- #


def _primer_strategy(*, primer_value=None, primer_raises=None, **kw):
    strat = _bare_strategy(**kw)
    strat.drift_config.enable_primer = True
    strat.drift_config.primer_follow_ups = 3
    if primer_value is not None or primer_raises is not None:
        strat.primer = _AChain(value=primer_value, raises=primer_raises)
    return strat


async def test_run_primer_returns_follow_ups_and_intermediate_answer() -> None:
    strat = _primer_strategy(
        primer_value='{"intermediate_answer": "hypothetical", "score": 0.3, '
        '"follow_up_queries": ["q1", " q2 ", ""]}'
    )
    follow_ups, intermediate = await strat._run_primer(
        SearchQuery(query="orig"), [_result("Community report A")]
    )
    # Whitespace trimmed and empties dropped; HyDE answer returned too.
    assert follow_ups == ["q1", "q2"]
    assert intermediate == "hypothetical"


async def test_run_primer_degrades_on_error_when_ignoring() -> None:
    strat = _primer_strategy(
        primer_raises=RuntimeError("bedrock down"), ignore_errors=True
    )
    result = await strat._run_primer(SearchQuery(query="q"), [_result("report")])
    assert result == ([], "")
    assert len(strat.primer.calls) == 1  # the failing call was attempted


async def test_run_primer_without_reports_makes_no_llm_call() -> None:
    # No community summaries -> nothing to ground a hypothetical answer in, so
    # the primer LLM is not called at all (it could only guess).
    strat = _primer_strategy(
        primer_value='{"intermediate_answer": "guess", "follow_up_queries": ["x"]}'
    )
    assert await strat._run_primer(SearchQuery(query="q"), []) == ([], "")
    assert strat.primer.calls == []


async def test_primer_search_runs_one_iteration_per_follow_up() -> None:
    # Each follow-up sub-query becomes its own search iteration; the graph/doc
    # fan-out is stubbed via _execute_search_iteration.
    strat = _primer_strategy(
        primer_value='{"follow_up_queries": ["fa", "fb"], "score": 0.2}'
    )
    executed: list[str] = []

    async def _fake_iteration(q: SearchQuery):
        executed.append(q.query)
        return [_result(f"hit for {q.query}")]

    strat._execute_search_iteration = _fake_iteration  # type: ignore[method-assign]

    all_results: list = []
    seen: set[str] = set()
    metrics: list[dict] = []
    await strat._primer_search(
        SearchQuery(query="orig"), [_result("seed report")], all_results, seen, metrics
    )

    assert executed == ["fa", "fb"]
    assert {m["source"] for m in metrics} == {"primer_follow_up"}
    assert len(all_results) == 2


async def test_primer_search_keeps_intermediate_answer_out_of_results() -> None:
    # The primer's HyDE intermediate_answer is an LLM guess, not evidence: it
    # must not enter all_results (which become answer context and sources).
    # Only the follow-up queries it drives contribute results.
    strat = _primer_strategy(
        primer_value='{"intermediate_answer": "HYDE seed answer", '
        '"follow_up_queries": ["fa"], "score": 0.2}'
    )

    async def _fake_iteration(q: SearchQuery):
        return [_result(f"hit for {q.query}")]

    strat._execute_search_iteration = _fake_iteration  # type: ignore[method-assign]
    all_results: list = []
    await strat._primer_search(
        SearchQuery(query="orig"), [_result("seed report")], all_results, set(), []
    )

    assert [r.content for r in all_results] == ["hit for fa"]
    assert all(r.source != "drift_primer" for r in all_results)
    assert all("HYDE seed answer" not in r.content for r in all_results)


async def test_asearch_skips_primer_without_candidate_communities() -> None:
    # Primer enabled but community retrieval found nothing: the primer must not
    # run (no reports to ground it); DRIFT falls back to the iterative loop.
    strat = _primer_strategy(
        primer_value='{"intermediate_answer": "guess", "follow_up_queries": ["x"]}',
        retrievers={"document": _StubRetriever(results=[])},
    )
    called = {"iterative": 0}

    async def _fake_iterative(query, all_results, seen, metrics, config=None):
        called["iterative"] += 1

    strat._iterative_search = _fake_iterative  # type: ignore[method-assign]
    strat.hybrid_scorer = SimpleNamespace(
        fuse_and_rerank_results=lambda groups, **kw: groups["results"]
    )
    strat._record_search_metrics = lambda *a, **k: None  # type: ignore[method-assign]

    result = await strat.asearch(SearchQuery(query="q"))

    assert strat.primer.calls == []
    assert called["iterative"] == 1
    assert result.results == []


async def test_primer_search_falls_back_to_iterative_without_follow_ups() -> None:
    strat = _primer_strategy(primer_value='{"follow_up_queries": [], "score": 0.9}')
    called = {"iterative": False}

    async def _fake_iterative(query, all_results, seen, metrics, config=None):
        called["iterative"] = True

    strat._iterative_search = _fake_iterative  # type: ignore[method-assign]
    await strat._primer_search(SearchQuery(query="q"), [], [], set(), [])
    assert called["iterative"]


async def test_iterative_search_uses_original_query_on_first_iteration() -> None:
    # Iteration 0 must search with the user's query; only later iterations are
    # rewritten from what was retrieved.
    strat = _bare_strategy(enable_llm_convergence=False)
    strat.drift_config.max_iterations = 2
    searched: list[str] = []
    evolved_at: list[int] = []

    async def _fake_search(q):
        searched.append(q.query)
        return [_result(f"new-{len(searched)}-{i}") for i in range(5)]

    async def _fake_evolve(q, original, results, iteration, config=None):
        evolved_at.append(iteration)
        out = q.model_copy(deep=True)
        out.query = f"refined-{iteration}"
        return out

    strat._execute_search_iteration = _fake_search  # type: ignore[method-assign]
    strat._evolve_query = _fake_evolve  # type: ignore[method-assign]
    await strat._iterative_search(SearchQuery(query="orig"), [], set(), [])

    assert searched == ["orig", "refined-1"]
    assert evolved_at == [1]


def test_default_config_disables_llm_convergence() -> None:
    drift = Config().search.drift_search
    assert drift.enable_llm_convergence is False
    assert drift.convergence_threshold == 0.8


async def test_asearch_fuses_with_per_type_quota() -> None:
    strat = _bare_strategy(
        enable_llm_convergence=False, retrievers={"document": _StubRetriever([])}
    )
    captured: dict = {}

    async def _fake_iterative(query, all_results, seen, metrics, config=None):
        return None

    def _fuse(groups, **kw):
        captured.update(kw)
        return groups["results"]

    strat._iterative_search = _fake_iterative  # type: ignore[method-assign]
    strat.hybrid_scorer = SimpleNamespace(fuse_and_rerank_results=_fuse)
    strat._record_search_metrics = lambda *a, **k: None  # type: ignore[method-assign]
    strat.drift_config.enable_primer = False

    await strat.asearch(SearchQuery(query="q", top_k=10))

    quota = captured["per_type_quota"]
    assert quota["text"] >= 10 and quota["entity"] >= 1 and quota["community"] >= 1
    assert captured["rerank_only_types"] == {"text"}


# --------------------------------------------------------------------------- #
# The caller's RunnableConfig reaches every DRIFT LLM call
# --------------------------------------------------------------------------- #


async def test_iterative_search_passes_the_callers_config_to_its_llm_calls() -> None:
    strat = _bare_strategy(convergence_threshold=0.99)
    strat.drift_config.max_iterations = 2
    strat.convergence_assessor = _AChain("0.0")  # below threshold: keep going
    strat.query_refiner = _AChain("refined")
    strat.keyword_expander = _AChain(["kw"])

    async def _fake_search(q):
        return [_result(f"{q.query}-{i}") for i in range(5)]

    strat._execute_search_iteration = _fake_search  # type: ignore[method-assign]
    caller_config = {"tags": ["caller"]}
    await strat._iterative_search(
        SearchQuery(query="q"), [], set(), [], config=caller_config
    )

    for chain in (strat.convergence_assessor, strat.query_refiner):
        assert chain.configs and all(c is caller_config for c in chain.configs)
    assert strat.keyword_expander.configs == [caller_config]


async def test_asearch_passes_the_callers_config_to_the_primer() -> None:
    strat = _primer_strategy(
        primer_value='{"follow_up_queries": [], "score": 0.9}',
        retrievers={"document": _StubRetriever(results=[_result("report")])},
    )
    received: list = []

    async def _fake_iterative(query, all_results, seen, metrics, config=None):
        received.append(config)

    strat._iterative_search = _fake_iterative  # type: ignore[method-assign]
    strat.hybrid_scorer = SimpleNamespace(
        fuse_and_rerank_results=lambda groups, **kw: groups["results"]
    )
    strat._record_search_metrics = lambda *a, **k: None  # type: ignore[method-assign]
    caller_config = {"tags": ["caller"]}

    await strat.asearch(SearchQuery(query="q"), config=caller_config)

    assert strat.primer.configs == [caller_config]
    assert received == [caller_config]  # no follow-ups: the loop gets it too
