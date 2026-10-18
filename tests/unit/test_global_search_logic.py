# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free tests of global search's report selection (MS GraphRAG parity).

Global search maps over every report of one cut of the community hierarchy
(``community_depth``), or over a root-down dynamic selection, instead of the
``top_k`` reports most similar to the query. A fake document retriever holds
an indexed hierarchy and applies the filters the strategy sends, so the tests
see which reports reach the map step. The map-reduce pipeline itself is
covered in ``test_global_search_map_reduce.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from botocore.exceptions import ClientError

from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
from unified_kg_rag.adapters.search_strategies.global_search import GlobalSearchStrategy
from unified_kg_rag.domain.ingestion.community_hierarchy import (
    attach_report_hierarchy,
    max_cut_depth,
)
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Config,
    RetrievalResult,
    SearchQuery,
)
from unified_kg_rag.shared import ConfigurationError
from unified_kg_rag.shared.utils.langchain import BatchProcessor

pytestmark = pytest.mark.unit

#            R0                       R1
#          /    \                     |
#        A        B (leaf)            C
#      /   \                        /   \
#    A1     A2                    C1     C2
HIERARCHY: dict[str, tuple[str, list[str]]] = {
    "R0": ("", ["A", "B"]),
    "R1": ("", ["C"]),
    "A": ("R0", ["A1", "A2"]),
    "B": ("R0", []),
    "C": ("R1", ["C1", "C2"]),
    "A1": ("A", []),
    "A2": ("A", []),
    "C1": ("C", []),
    "C2": ("C", []),
}
REPORTS_PREFIX = Config().indexing.opensearch.community_reports_index_prefix
TEXT_PREFIX = Config().indexing.opensearch.text_units_index_prefix


def _documents(*, with_hierarchy: bool = True) -> list[dict[str, Any]]:
    communities = [
        Community(id=cid, name=cid, level="0", parent=parent, children=children)
        for cid, (parent, children) in HIERARCHY.items()
    ]
    reports = [
        CommunityReport(id=f"r-{cid}", name=cid, community_id=cid, rank=i)
        for i, cid in enumerate(HIERARCHY)
    ]
    if with_hierarchy:
        attach_report_hierarchy(reports, communities)
    return [
        {
            "community_id": r.community_id,
            "rank": r.rank,
            "depth": r.depth,
            "max_cut_depth": max_cut_depth(r),
            "parent_community_id": r.parent_community_id,
            "child_community_ids": r.child_community_ids,
        }
        for r in reports
    ]


def _matches(value: Any, condition: Any) -> bool:
    if isinstance(condition, dict):
        if value is None:
            return False
        return all(
            {"lte": value <= b, "gte": value >= b, "lt": value < b, "gt": value > b}[op]
            for op, b in condition.items()
        )
    if isinstance(condition, list):
        return value in condition
    return value == condition


class _ReportStore:
    """A document retriever over the indexed hierarchy that applies filters.

    A text-less query is a browse ordered by rank; a query with text ranks
    by the query word appearing in the report's community id (similarity).
    """

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        if query.index_prefixes != [REPORTS_PREFIX]:
            return [
                RetrievalResult(
                    content="chunk", score=0.5, source="t0", retriever_type="text"
                )
            ]
        filters = query.filters or {}
        hits = [
            d
            for d in self.documents
            if all(_matches(d.get(k), v) for k, v in filters.items())
        ]
        if query.query:
            hits.sort(key=lambda d: d["community_id"] not in query.query)
        else:
            hits.sort(key=lambda d: -d["rank"])
        return [
            RetrievalResult(
                content=f"report {d['community_id']}",
                score=0.5,
                source=f"r-{d['community_id']}",
                retriever_type="community",
                metadata=dict(d),
            )
            for d in hits[: query.top_k * query.retrieval_multiplier]
        ]


class _MapChain:
    """Map chain recording which reports each call rated."""

    def __init__(self) -> None:
        self.reports: list[str] = []
        self.configs: list[Any] = []

    def invoke(self, single_input: dict, config: Any = None) -> str:
        self.configs.append(config)
        self.reports.append(single_input["reports"])
        return json.dumps({"points": [{"description": "a point", "score": 50}]})


class _Rater:
    """Rates a report 9 when its community is in ``relevant``, else 1."""

    def __init__(self, relevant: set[str], fail: Exception | None = None) -> None:
        self.relevant = relevant
        self.fail = fail
        self.rated: list[str] = []
        self.configs: list[Any] = []

    async def ainvoke(self, inputs: dict, config: Any = None) -> str:
        community = inputs["community_summary"].removeprefix("report ")
        self.rated.append(community)
        self.configs.append(config)
        if self.fail is not None:
            raise self.fail
        return "9" if community in self.relevant else "1"


class _Tokens:
    def count_tokens_many(self, texts: list[str]) -> list[int]:
        return [10] * len(texts)


def _strategy(
    store: _ReportStore | None = None, **settings: Any
) -> GlobalSearchStrategy:
    config = Config()
    config.search.reranking.enabled = False
    config.processing.ignore_errors = settings.pop("ignore_errors", False)
    gs = config.search.global_search
    # Map even a cut of one or two reports.
    gs.map_reduce_min_results = 1
    for key, value in settings.items():
        setattr(gs, key, value)
    strat = GlobalSearchStrategy.__new__(GlobalSearchStrategy)
    strat.config = config
    strat.global_search_config = gs
    strat.ignore_errors = config.processing.ignore_errors
    strat.max_concurrency = 4
    strat.target_language = "English"
    strat.retrievers = {"document": store or _ReportStore(_documents())}
    strat.hybrid_scorer = HybridScorer(config)
    strat.token_manager = _Tokens()  # type: ignore[assignment]
    strat.map_rater = _MapChain()
    strat.batch_processor = BatchProcessor(max_concurrency=1, max_attempts=1)
    return strat


def _mapped(strat: GlobalSearchStrategy) -> set[str]:
    text = "\n".join(strat.map_rater.reports)  # type: ignore[attr-defined]
    return {cid for cid in HIERARCHY if f"report {cid}\n" in text + "\n"}


# --------------------------------------------------------------------------- #
# Fixed cut (default)
# --------------------------------------------------------------------------- #


def test_defaults_follow_ms_graphrag() -> None:
    config = Config()
    gs = config.search.global_search
    assert gs.community_depth == 2  # MS GraphRAG's community_level default
    assert gs.use_dynamic_selection is False
    assert gs.map_context_tokens == 12000  # data_max_tokens
    assert gs.text_unit_slots == 0
    strategy = GlobalSearchStrategy(config=config, retrievers={})
    assert not hasattr(strategy, "community_relevance_scorer")


@pytest.mark.parametrize(
    ("depth", "cut"),
    [(0, {"R0", "R1"}), (1, {"A", "B", "C"}), (2, {"A1", "A2", "B", "C1", "C2"})],
)
async def test_every_report_of_the_cut_is_mapped(depth: int, cut: set[str]) -> None:
    # Regression: the map step saw the query.top_k reports most similar to
    # the query, whatever the corpus, rather than a level of the hierarchy.
    strat = _strategy(community_depth=depth, map_context_tokens=20)
    result = await strat.asearch(SearchQuery(query="themes", top_k=1))

    assert _mapped(strat) == cut
    assert result.metadata["global_mapped_report_count"] == len(cut)
    # Two reports of 10 tokens per 20-token map call.
    assert result.metadata["global_map_call_count"] == (len(cut) + 1) // 2
    assert result.metadata["global_level_capped"] is False
    assert {r.metadata["community_id"] for r in result.results[1:]} == cut


async def test_the_cut_is_a_browse_without_query_text() -> None:
    store = _ReportStore(_documents())
    strat = _strategy(store)
    await strat.asearch(SearchQuery(query="themes", filters={"attr_x": "y"}))
    browse = store.calls[0]
    assert browse.query == ""
    assert browse.index_prefixes == [REPORTS_PREFIX]
    # The caller's filters narrow the cut.
    assert browse.filters == {
        "attr_x": "y",
        "depth": {"lte": 2},
        "max_cut_depth": {"gte": 2},
    }


async def test_a_cut_over_the_cap_gives_way_to_the_deepest_cut_that_fits() -> None:
    # Regression: a depth-2 cut of 5 reports over a cap of 3 mapped the 3
    # most similar, leaving entities of the other two out; the depth-1 cut
    # covers them all in 3 broader reports.
    store = _ReportStore(_documents())
    strat = _strategy(store, max_reports_per_level=3)
    result = await strat.asearch(SearchQuery(query="C1 C2 themes"))

    assert _mapped(strat) == {"A", "B", "C"}
    assert result.metadata["global_level_capped"] is False
    assert result.metadata["global_cut_depth"] == 1
    assert [c.filters["depth"] for c in store.calls] == [{"lte": 2}, {"lte": 1}]
    assert all(not c.query for c in store.calls)


async def test_a_cut_no_depth_fits_maps_the_most_similar_reports() -> None:
    store = _ReportStore(_documents())
    strat = _strategy(store, max_reports_per_level=1)
    result = await strat.asearch(SearchQuery(query="C1 themes"))

    assert _mapped(strat) == {"C1"}
    assert result.metadata["global_level_capped"] is True
    assert result.metadata["global_cut_depth"] == 2
    similar = store.calls[-1]
    assert similar.query == "C1 themes"
    assert similar.top_k == 1
    assert similar.filters == store.calls[0].filters


async def test_a_cut_of_exactly_the_cap_is_complete() -> None:
    # Regression: a cut of exactly max_reports_per_level reports was taken
    # for a truncated one and re-fetched by similarity.
    store = _ReportStore(_documents())
    strat = _strategy(store, max_reports_per_level=5)
    result = await strat.asearch(SearchQuery(query="themes"))
    assert result.metadata["global_level_capped"] is False
    assert result.metadata["global_cut_depth"] == 2
    assert _mapped(strat) == {"A1", "A2", "B", "C1", "C2"}
    assert store.calls[0].top_k == 6


def test_the_cap_leaves_the_probe_room_in_the_store_page() -> None:
    strat = _strategy(max_reports_per_level=500)
    strat.config.indexing.opensearch.max_query_size = 100
    assert strat._report_cap() == 99


async def test_a_cut_of_exactly_the_cap_is_complete_at_the_defaults() -> None:
    # Regression: at the defaults (max_reports_per_level == max_query_size ==
    # 100) the probe could ask for no more than the cap, so a cut of exactly
    # the cap counted as truncated.
    strat = _strategy()

    class _Page:
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            # Exactly the cap exists; a store page holds max_query_size hits.
            size = min(query.top_k, strat.config.indexing.opensearch.max_query_size)
            return [
                RetrievalResult(
                    content=str(i), score=1.0, source=str(i), retriever_type="community"
                )
                for i in range(min(size, strat._report_cap()))
            ]

    strat.retrievers = {"document": _Page()}  # type: ignore[dict-item]
    reports, complete = await strat._browse_reports(SearchQuery(query="q"), {}, "cut")
    assert complete and len(reports) == strat._report_cap()


async def test_reports_indexed_without_the_hierarchy_fail_loudly() -> None:
    strat = _strategy(_ReportStore(_documents(with_hierarchy=False)))
    with pytest.raises(ConfigurationError, match="indexing.reset"):
        await strat.asearch(SearchQuery(query="themes"))


async def test_no_reports_at_all_is_an_empty_result() -> None:
    result = await _strategy(_ReportStore([])).asearch(SearchQuery(query="themes"))
    assert result.results == []
    assert result.metadata["global_selected_report_count"] == 0


async def test_without_map_reduce_the_cut_is_the_context() -> None:
    strat = _strategy(enable_map_reduce=False, community_depth=1)
    result = await strat.asearch(SearchQuery(query="themes"))
    assert {r.metadata["community_id"] for r in result.results} == {"A", "B", "C"}
    assert strat.map_rater.reports == []  # type: ignore[attr-defined]


async def test_text_units_are_off_by_default_and_opt_in() -> None:
    store = _ReportStore(_documents())
    await _strategy(store).asearch(SearchQuery(query="themes"))
    assert all(q.index_prefixes == [REPORTS_PREFIX] for q in store.calls)

    store = _ReportStore(_documents())
    result = await _strategy(store, text_unit_slots=3).asearch(
        SearchQuery(query="themes")
    )
    (chunk_query,) = [q for q in store.calls if q.index_prefixes == [TEXT_PREFIX]]
    assert chunk_query.top_k == 3
    assert set(chunk_query.filters["community_ids"]) == {"A1", "A2", "B", "C1", "C2"}
    assert result.results[-1].retriever_type == "text"


# --------------------------------------------------------------------------- #
# Dynamic community selection
# --------------------------------------------------------------------------- #


async def test_dynamic_selection_descends_from_the_roots() -> None:
    # R0 relevant -> A, B rated; A relevant (drops R0) -> A1, A2 rated; A1
    # relevant (drops A). R1 is irrelevant, so its subtree is never rated.
    strat = _strategy(use_dynamic_selection=True)
    rater = _Rater({"R0", "A", "A1"})
    strat.community_relevance_scorer = rater
    result = await strat.asearch(SearchQuery(query="themes"))

    assert sorted(rater.rated) == ["A", "A1", "A2", "B", "R0", "R1"]
    assert _mapped(strat) == {"A1"}
    assert result.metadata["global_rated_report_count"] == 6


async def test_dynamic_selection_follows_relevant_children_past_community_depth() -> (
    None
):
    # Regression: the walk stopped at community_depth, so a relevant A kept
    # its relevant child A1 from being rated. MS GraphRAG follows relevant
    # children to the leaves; its max_level bounds only the fallback.
    strat = _strategy(use_dynamic_selection=True, community_depth=1)
    strat.community_relevance_scorer = _Rater({"R0", "A", "A1"})
    await strat.asearch(SearchQuery(query="themes"))
    assert _mapped(strat) == {"A1"}


async def test_community_depth_bounds_the_whole_depth_fallback() -> None:
    # Nothing relevant at depths 0 and 1: depth 2 is rated only when
    # community_depth reaches it.
    strat = _strategy(use_dynamic_selection=True, community_depth=1)
    rater = _Rater({"C1"})
    strat.community_relevance_scorer = rater
    await strat.asearch(SearchQuery(query="themes"))
    assert sorted(rater.rated) == ["A", "B", "C", "R0", "R1"]

    strat = _strategy(use_dynamic_selection=True, community_depth=2)
    strat.community_relevance_scorer = _Rater({"C1"})
    await strat.asearch(SearchQuery(query="themes"))
    assert _mapped(strat) == {"C1"}


async def test_dynamic_selection_rates_the_next_depth_when_none_is_relevant() -> None:
    strat = _strategy(use_dynamic_selection=True)
    rater = _Rater({"C"})
    strat.community_relevance_scorer = rater
    await strat.asearch(SearchQuery(query="themes"))
    # No root is relevant: all of depth 1 is rated, then C's children.
    assert sorted(rater.rated) == ["A", "B", "C", "C1", "C2", "R0", "R1"]
    assert _mapped(strat) == {"C"}


async def test_dynamic_selection_with_nothing_relevant_maps_nothing() -> None:
    strat = _strategy(use_dynamic_selection=True)
    strat.community_relevance_scorer = _Rater(set())
    result = await strat.asearch(SearchQuery(query="themes"))
    assert result.results == []
    assert strat.map_rater.reports == []  # type: ignore[attr-defined]


async def test_a_failed_rating_keeps_the_report_under_ignore_errors() -> None:
    strat = _strategy(use_dynamic_selection=True, ignore_errors=True)
    strat.community_relevance_scorer = _Rater(set(), fail=TimeoutError("slow"))
    result = await strat.asearch(SearchQuery(query="themes"))
    # Every rating failed, so every report counts as relevant: the leaves.
    assert _mapped(strat) == {"A1", "A2", "B", "C1", "C2"}
    assert result.metadata["degraded_error"] == "TimeoutError"


async def test_a_failed_rating_raises_without_ignore_errors() -> None:
    strat = _strategy(use_dynamic_selection=True)
    strat.community_relevance_scorer = _Rater(set(), fail=TimeoutError("slow"))
    with pytest.raises(TimeoutError):
        await strat.asearch(SearchQuery(query="themes"))


async def test_a_fatal_rating_error_raises_even_under_ignore_errors() -> None:
    denied = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "Converse"
    )
    strat = _strategy(use_dynamic_selection=True, ignore_errors=True)
    strat.community_relevance_scorer = _Rater(set(), fail=denied)
    with pytest.raises(ClientError):
        await strat.asearch(SearchQuery(query="themes"))


async def test_the_callers_config_reaches_rating_and_map_calls() -> None:
    strat = _strategy(use_dynamic_selection=True)
    rater = _Rater({"R0", "R1"})
    strat.community_relevance_scorer = rater
    caller_config = {"tags": ["caller"]}
    await strat.asearch(SearchQuery(query="themes"), config=caller_config)
    assert rater.configs and set(map(id, rater.configs)) == {id(caller_config)}
    assert strat.map_rater.configs == [caller_config]  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# _parse_map_payload robustness to non-finite LLM scores
# --------------------------------------------------------------------------- #


def test_parse_map_payload_handles_infinity_and_nan() -> None:
    # parse_llm_json (json.loads) accepts Infinity/-Infinity/NaN, and "1e999"
    # overflows float() to inf; int(inf)/int(nan) would crash the whole query.
    raw_bad = (
        '{"points": [{"description": "inf", "score": Infinity}, '
        '{"description": "nan", "score": NaN}, {"description": "ok", "score": 7}]}'
    )
    scores = {
        p.description: p.score
        for p in GlobalSearchStrategy._parse_map_payload(raw_bad) or []
    }
    assert scores == {"inf": 0, "nan": 0, "ok": 7}


def test_parse_map_payload_overflow_string_score() -> None:
    raw = '{"points": [{"description": "big", "score": "1e999"}]}'
    pts = GlobalSearchStrategy._parse_map_payload(raw) or []
    assert pts[0].score == 0  # 1e999 -> inf -> coerced to 0


def test_was_map_reduce_applied_detects_synthesized_source() -> None:
    strat = _strategy()
    summary = RetrievalResult(
        content="s", score=1.0, source="synthesized_summary", retriever_type="general"
    )
    assert strat._was_map_reduce_applied([summary]) is True
    strat.global_search_config.enable_map_reduce = False
    assert strat._was_map_reduce_applied([summary]) is False
