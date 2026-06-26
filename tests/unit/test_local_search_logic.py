# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for LocalSearchStrategy pure helpers + orchestration branches.

The strategy is constructed via ``__new__`` so its AWS-client ``__init__`` (which
builds a ``HybridScorer`` / ``TokenManager`` and a boto3 session) never runs. The
``_get_ids`` helper lives on ``BaseSearchStrategy`` and is exercised here through
the concrete subclass. Retriever async methods are replaced with simple fakes /
AsyncMocks to drive the candidate-selection, graph-expansion and document-retrieval
branches (empty-retriever short-circuits, exception fallbacks) without real AWS.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unified_kg_rag.adapters.search_strategies.local_search import LocalSearchStrategy
from unified_kg_rag.domain.models import RetrievalResult, SearchQuery, SearchType

pytestmark = pytest.mark.unit


def _result(
    source: str | None = None,
    *,
    score: float = 0.5,
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        content="c",
        score=score,
        source=source,
        retriever_type="document",
        metadata=metadata or {},
    )


def _bare_strategy(
    *,
    retrievers: dict | None = None,
    entity_focus_multiplier: int = 2,
) -> LocalSearchStrategy:
    """A LocalSearchStrategy whose __init__ never runs (no AWS clients)."""
    strat = LocalSearchStrategy.__new__(LocalSearchStrategy)
    strat.config = SimpleNamespace(
        indexing=SimpleNamespace(
            opensearch=SimpleNamespace(
                entities_index_prefix="entities",
                text_units_index_prefix="text_units",
            ),
            neptune=SimpleNamespace(entity_label_prefix="Entity"),
        ),
        search=SimpleNamespace(
            local_search=SimpleNamespace(entity_frequency_threshold=20)
        ),
    )
    strat.retrievers = retrievers or {}
    strat.entity_focus_multiplier = entity_focus_multiplier
    return strat


class _StubRetriever:
    """Async retriever stub returning a fixed list (or raising) from aretrieve."""

    def __init__(self, results=None, raises: Exception | None = None) -> None:
        self._results = results or []
        self._raises = raises
        self.last_query: SearchQuery | None = None

    async def aretrieve(self, query: SearchQuery):
        self.last_query = query
        if self._raises is not None:
            raise self._raises
        return self._results


# --------------------------------------------------------------------------- #
# _get_ids (inherited from BaseSearchStrategy, exercised via the subclass)
# --------------------------------------------------------------------------- #


def test_get_ids_collects_scalar_and_list_metadata() -> None:
    results = [
        _result("a", metadata={"id": "x1"}),
        _result("b", metadata={"id": ["x2", "x3"]}),
    ]
    ids = set(LocalSearchStrategy._get_ids(results, "id"))
    assert ids == {"x1", "x2", "x3"}


def test_get_ids_falls_back_to_source_for_id_key() -> None:
    # When metadata lacks "id" but a source is present, source is used as the id.
    results = [_result("seed-1", metadata={})]
    assert LocalSearchStrategy._get_ids(results, "id") == ["seed-1"]


def test_get_ids_no_source_fallback_for_non_id_key() -> None:
    # The source fallback only applies to the "id" key, not arbitrary keys.
    results = [_result("seed-1", metadata={})]
    assert LocalSearchStrategy._get_ids(results, "text_unit_ids") == []


def test_get_ids_dedups_across_results() -> None:
    results = [
        _result("a", metadata={"id": "dup"}),
        _result("b", metadata={"id": "dup"}),
    ]
    assert LocalSearchStrategy._get_ids(results, "id") == ["dup"]


def test_get_ids_text_unit_ids_collection() -> None:
    results = [
        _result("a", metadata={"text_unit_ids": ["t1", "t2"]}),
        _result("b", metadata={"text_unit_ids": ["t2", "t3"]}),
    ]
    assert set(LocalSearchStrategy._get_ids(results, "text_unit_ids")) == {
        "t1",
        "t2",
        "t3",
    }


# --------------------------------------------------------------------------- #
# _filter_entities (frequency thresholding)
# --------------------------------------------------------------------------- #


def test_filter_entities_keeps_zero_and_within_threshold() -> None:
    nodes = [
        _result("zero", metadata={"text_unit_ids": []}),  # count 0 -> kept
        _result("low", metadata={"text_unit_ids": ["t1", "t2"]}),  # 2 <= 3 -> kept
    ]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=3)
    assert [n.source for n in kept] == ["zero", "low"]


def test_filter_entities_drops_above_threshold() -> None:
    nodes = [
        _result("hot", metadata={"text_unit_ids": ["t1", "t2", "t3", "t4"]}),  # 4 > 3
        _result("ok", metadata={"text_unit_ids": ["t1"]}),  # 1 <= 3
    ]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=3)
    assert [n.source for n in kept] == ["ok"]


def test_filter_entities_boundary_equal_to_threshold_kept() -> None:
    nodes = [_result("edge", metadata={"text_unit_ids": ["t1", "t2", "t3"]})]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=3)
    assert len(kept) == 1


def test_filter_entities_missing_metadata_treated_as_zero() -> None:
    # No text_unit_ids key -> count 0 -> kept (zero-frequency entities pass).
    nodes = [_result("nometa", metadata={})]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=1)
    assert len(kept) == 1


# --------------------------------------------------------------------------- #
# _find_candidate_entities
# --------------------------------------------------------------------------- #


async def test_find_candidate_entities_no_retriever_returns_empty() -> None:
    strat = _bare_strategy(retrievers={})
    query = SearchQuery(query="q", entity_focus=["Alice"])
    assert await strat._find_candidate_entities(query) == []


async def test_find_candidate_entities_no_entity_focus_returns_empty() -> None:
    strat = _bare_strategy(retrievers={"document": _StubRetriever([_result("e1")])})
    query = SearchQuery(query="q", entity_focus=[])
    assert await strat._find_candidate_entities(query) == []


async def test_find_candidate_entities_returns_sources_and_top_k() -> None:
    retriever = _StubRetriever([_result("e1"), _result(None), _result("e2")])
    strat = _bare_strategy(
        retrievers={"document": retriever}, entity_focus_multiplier=3
    )
    query = SearchQuery(query="q", entity_focus=["Alice", "Bob"])
    out = await strat._find_candidate_entities(query)
    # Drops the result with no source; keeps the rest.
    assert out == ["e1", "e2"]
    # top_k = len(entity_focus) * multiplier = 2 * 3.
    sent = retriever.last_query
    assert sent is not None
    assert sent.top_k == 6
    assert sent.index_prefixes == ["entities"]


async def test_find_candidate_entities_swallows_retriever_error() -> None:
    strat = _bare_strategy(
        retrievers={"document": _StubRetriever(raises=RuntimeError("boom"))}
    )
    query = SearchQuery(query="q", entity_focus=["Alice"])
    assert await strat._find_candidate_entities(query) == []


# --------------------------------------------------------------------------- #
# _expand_via_graph
# --------------------------------------------------------------------------- #


async def test_expand_via_graph_no_graph_retriever_returns_empty() -> None:
    strat = _bare_strategy(retrievers={})
    query = SearchQuery(query="q")
    assert await strat._expand_via_graph(query, ["e1"]) == []


async def test_expand_via_graph_no_seeds_returns_empty() -> None:
    strat = _bare_strategy(retrievers={"graph": _StubRetriever([_result("x")])})
    query = SearchQuery(query="q")
    assert await strat._expand_via_graph(query, []) == []


async def test_expand_via_graph_sets_id_filter_and_label_prefix() -> None:
    retriever = _StubRetriever([_result("x")])
    strat = _bare_strategy(retrievers={"graph": retriever})
    query = SearchQuery(query="q", entity_focus=["Alice"], filters={"existing": 1})
    out = await strat._expand_via_graph(query, ["e1", "e2"])
    assert [r.source for r in out] == ["x"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.filters is not None
    assert sent.filters["id"] == ["e1", "e2"]
    assert sent.filters["existing"] == 1  # pre-existing filters preserved
    assert sent.label_prefixes == ["Entity"]
    assert sent.entity_focus == []  # focus cleared on the graph sub-query
    # The original query must not be mutated (deep copy is used).
    assert query.entity_focus == ["Alice"]
    assert query.filters is not None and "id" not in query.filters


async def test_expand_via_graph_swallows_retriever_error() -> None:
    strat = _bare_strategy(
        retrievers={"graph": _StubRetriever(raises=RuntimeError("neptune down"))}
    )
    query = SearchQuery(query="q")
    assert await strat._expand_via_graph(query, ["e1"]) == []


# --------------------------------------------------------------------------- #
# _retrieve_documents
# --------------------------------------------------------------------------- #


async def test_retrieve_documents_no_retriever_returns_empty_dict() -> None:
    strat = _bare_strategy(retrievers={})
    assert await strat._retrieve_documents(["t1"], None) == {}


async def test_retrieve_documents_no_ids_returns_empty_dict() -> None:
    strat = _bare_strategy(retrievers={"document": _StubRetriever([_result("x")])})
    assert await strat._retrieve_documents([], None) == {}


async def test_retrieve_documents_wraps_results_under_text_units_key() -> None:
    retriever = _StubRetriever([_result("t1"), _result("t2")])
    strat = _bare_strategy(retrievers={"document": retriever})
    out = await strat._retrieve_documents(["t1", "t2"], suffix="dev")
    assert set(out) == {"text_units"}
    assert [r.source for r in out["text_units"]] == ["t1", "t2"]
    sent = retriever.last_query
    assert sent is not None
    assert sent.search_type == SearchType.LEXICAL
    assert sent.top_k == 2
    assert sent.filters == {"id": ["t1", "t2"]}
    assert sent.suffix == "dev"
    assert sent.index_prefixes == ["text_units"]


async def test_retrieve_documents_swallows_retriever_error() -> None:
    strat = _bare_strategy(
        retrievers={"document": _StubRetriever(raises=RuntimeError("os down"))}
    )
    assert await strat._retrieve_documents(["t1"], None) == {}


# --------------------------------------------------------------------------- #
# _record_search_metrics
# --------------------------------------------------------------------------- #


def test_record_search_metrics_populates_metrics() -> None:
    strat = _bare_strategy()
    strat._metrics = {"timings": {}, "metrics": {}}
    strat._record_search_metrics(
        1.5, retrieved_count=7, entity_count=3, text_unit_count=9
    )
    assert strat._metrics["timings"]["processing_time"] == 1.5
    assert strat._metrics["metrics"] == {
        "retrieved_count": 7,
        "entity_count": 3,
        "text_unit_count": 9,
    }
