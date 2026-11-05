# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local search's community reports by membership
(``search.local_search.report_selection: membership``).

MS GraphRAG local (``mixed_context._build_community_context``) reads the
reports of the communities the selected entities belong to, most matched
entities first, then by rank. Fake retrievers stand in for the stores and the
shared HybridScorer is stubbed to flatten results, so no model is called.
"""

from __future__ import annotations

from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from tests.fixtures.fakes.retrievers import FakeRetriever
from unified_kg_rag.adapters.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.adapters.search_strategies.local_search import LocalSearchStrategy
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)

pytestmark = pytest.mark.unit

# community -> report rank; "c-seed" is a seed graph's community: no report.
_REPORTS = {"c-big": 2.0, "c-small": 9.0, "c-tie": 5.0, "c-other": 7.0}


def _entity(name: str, communities: Any) -> RetrievalResult:
    metadata: dict[str, Any] = {"id": name, "name": name, "text_unit_ids": []}
    if communities is not None:
        metadata["community_ids"] = communities
    return RetrievalResult(
        content=name,
        score=1.0,
        source=name,
        retriever_type="entity",
        metadata=metadata,
    )


class _Graph(FakeRetriever):
    def __init__(self, nodes: list[RetrievalResult]) -> None:
        super().__init__("graph")
        self.nodes = nodes

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        return list(self.nodes)


class _Documents(FakeRetriever):
    """The reports index answers a ``community_id`` fetch from ``_REPORTS``."""

    def __init__(self, reports_prefix: str) -> None:
        super().__init__("document")
        self.reports_prefix = reports_prefix

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        wanted = (query.filters or {}).get("community_id")
        if query.index_prefixes != [self.reports_prefix] or wanted is None:
            return await super().aretrieve(query)
        self.calls.append(query)
        return [
            RetrievalResult(
                content=f"report of {cid}",
                score=0.0,
                source=f"r-{cid}",
                retriever_type="document",
                metadata={"id": f"r-{cid}", "community_id": cid, "rank": rank},
            )
            for cid, rank in _REPORTS.items()
            if cid in wanted
        ]


def _strategy(
    config: Config, nodes: list[RetrievalResult]
) -> tuple[LocalSearchStrategy, _Documents]:
    config.search.local_search.report_selection = "membership"
    reports_prefix = config.indexing.opensearch.community_reports_index_prefix
    documents = _Documents(reports_prefix)
    strategy = get_strategy_spec(SearchStrategy.LOCAL).strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: documents,
            RetrieverRole.GRAPH.value: _Graph(nodes),
        },
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, **_kw: [r for rs in results_dict.values() for r in rs]
    )
    assert isinstance(strategy, LocalSearchStrategy)
    return strategy, documents


def _report_calls(config: Config, documents: _Documents) -> list[SearchQuery]:
    prefix = config.indexing.opensearch.community_reports_index_prefix
    return [q for q in documents.calls if q.index_prefixes == [prefix]]


def _reports(results: list[RetrievalResult]) -> list[str]:
    return [str(r.source) for r in results if str(r.source).startswith("r-")]


_MEMBERS = [
    _entity("a", ["c-big"]),
    _entity("b", ["c-big", "c-tie"]),
    _entity("c", ["c-big"]),  # one community: still a list (graph expansion)
    _entity("d", ["c-small"]),
    _entity("e", ["c-tie", "c-seed"]),
]


async def test_member_reports_rank_by_matches_then_rank(config: Config) -> None:
    strategy, documents = _strategy(config, _MEMBERS)

    result = await strategy.asearch(
        SearchQuery(query="q", entity_focus=["a"], top_k=10)
    )

    # c-big: 3 members; c-tie: 2; c-small: 1. c-other has no member and
    # c-seed no report. Rank only orders equal counts: c-small's 9.0 does
    # not lift it above c-big's 2.0.
    assert _reports(result.results) == ["r-c-big", "r-c-tie", "r-c-small"]
    (fetch,) = _report_calls(config, documents)
    assert fetch.query == ""
    assert fetch.filters is not None
    assert sorted(fetch.filters["community_id"]) == [
        "c-big",
        "c-seed",
        "c-small",
        "c-tie",
    ]
    assert result.metadata["report_selection"] == "membership"


def test_equal_matches_rank_by_report_rank() -> None:
    reports = [
        RetrievalResult(
            content=cid,
            score=0.0,
            source=cid,
            retriever_type="document",
            metadata={"community_id": cid, "rank": rank},
        )
        for cid, rank in (("low", 1.0), ("high", 8.0), ("low", 1.0))
    ]
    ranked = LocalSearchStrategy.rank_member_reports(reports, {"low": 1, "high": 1})
    assert [r.source for r in ranked] == ["high", "low"]
    assert [r.score for r in ranked] == [1.0, 0.5]


async def test_member_reports_are_cut_to_the_report_quota(config: Config) -> None:
    strategy, _ = _strategy(config, _MEMBERS)

    result = await strategy.asearch(SearchQuery(query="q", entity_focus=["a"], top_k=2))

    assert strategy._per_type_quota(2)["community"] == 2
    assert _reports(result.results) == ["r-c-big", "r-c-tie"]


async def test_no_membership_falls_back_to_similar_reports(config: Config) -> None:
    # The entities index holds no community_ids: an expansion over it (a
    # document filter's walk, a failed graph walk) names no community.
    strategy, documents = _strategy(config, [_entity("a", None)])

    result = await strategy.asearch(SearchQuery(query="q", entity_focus=["Acme"]))

    (similar,) = _report_calls(config, documents)
    assert similar.query == "Acme"
    assert result.metadata["report_selection"] == "similarity"


async def test_members_without_a_report_give_no_similar_reports(
    config: Config,
) -> None:
    # A seed graph's community has no report: membership is known, so the
    # section is empty rather than filled with unrelated similar reports.
    strategy, documents = _strategy(config, [_entity("s", ["c-seed"])])

    result = await strategy.asearch(SearchQuery(query="q", entity_focus=["s"]))

    assert _reports(result.results) == []
    assert all(q.query == "" for q in _report_calls(config, documents))
    assert result.metadata["report_selection"] == "membership"


async def test_a_source_kinds_filter_reads_no_member_report(config: Config) -> None:
    strategy, documents = _strategy(config, _MEMBERS)

    await strategy.asearch(
        SearchQuery(query="q", entity_focus=["a"], filters={"source_kinds": "document"})
    )

    assert _report_calls(config, documents) == []


async def test_a_caller_community_filter_narrows_the_members(config: Config) -> None:
    strategy, documents = _strategy(config, _MEMBERS)

    result = await strategy.asearch(
        SearchQuery(
            query="q", entity_focus=["a"], filters={"community_id": ["c-small"]}
        )
    )

    (fetch,) = _report_calls(config, documents)
    assert fetch.filters is not None
    assert fetch.filters["community_id"] == ["c-small"]
    assert _reports(result.results) == ["r-c-small"]


async def test_similarity_is_the_default(config: Config) -> None:
    assert config.search.local_search.report_selection == "similarity"
    strategy, documents = _strategy(config, _MEMBERS)
    config.search.local_search.report_selection = "similarity"

    result = await strategy.asearch(SearchQuery(query="q", entity_focus=["Acme"]))

    (similar,) = _report_calls(config, documents)
    assert similar.query == "Acme"
    assert result.metadata["report_selection"] == "similarity"


@pytest.mark.parametrize(
    ("filters", "selection"),
    [
        ({"community_id": ["c-other"]}, "similarity"),
        ({"community_id": {"gte": "c-a"}}, "similarity"),
        ({"community_id": "c-small"}, "membership"),
        ({"level": 1}, "similarity"),
        ({"level": [1, 2]}, "similarity"),
        ({"level": {"gte": 1}}, "similarity"),
        ({"level": 0}, "membership"),
        ({"level": [0, 1]}, "membership"),
        ({"level": {"gte": 0, "lt": 2}}, "membership"),
    ],
)
async def test_a_caller_filter_without_the_member_communities_reads_similar_ones(
    config: Config, filters: dict[str, Any], selection: str
) -> None:
    # Members are level-0 communities: a filter that admits none of them
    # (it asks for others, or another level) would leave no report at all.
    strategy, documents = _strategy(config, _MEMBERS)

    result = await strategy.asearch(
        SearchQuery(query="q", entity_focus=["Acme"], filters=filters)
    )

    assert result.metadata["report_selection"] == selection
    calls = _report_calls(config, documents)
    if selection == "similarity":
        (similar,) = calls
        assert similar.query == "Acme"
        assert similar.filters == filters
    else:
        assert all(q.query == "" for q in calls)
