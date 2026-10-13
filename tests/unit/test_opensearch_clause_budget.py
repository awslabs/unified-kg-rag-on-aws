# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for config-driven OpenSearch clause-budget batching.

The clause-budget knobs moved from hardcoded ClassVars to config; this verifies
``_calculate_safe_batch_size`` reads the instance attributes (set from config in
__init__) without constructing any AWS client.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.domain.models import RetrievalResult, SearchQuery, SearchType

pytestmark = pytest.mark.unit


def _retriever(
    max_size: int = 100,
    terms_batch_size: int = 150,
    max_total_clauses: int = 600,
    reserved_clauses: int = 300,
) -> OpenSearchRetriever:
    # Bypass __init__ (which builds AWS clients) and set only the clause knobs.
    # Use object.__setattr__ because this is a pydantic model with private attrs.
    retriever = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(retriever, "_max_size", max_size)
    object.__setattr__(retriever, "_terms_batch_size", terms_batch_size)
    object.__setattr__(retriever, "_max_total_clauses", max_total_clauses)
    object.__setattr__(retriever, "_reserved_clauses", reserved_clauses)
    return retriever


def test_no_filters_uses_terms_batch_size() -> None:
    assert _retriever(terms_batch_size=150)._calculate_safe_batch_size(None) == 150


def test_no_list_filters_uses_terms_batch_size() -> None:
    retriever = _retriever(terms_batch_size=150)
    assert retriever._calculate_safe_batch_size({"id": "scalar"}) == 150


def test_single_large_list_clamped_to_terms_batch_size() -> None:
    # available = 600-300 = 300; safe_size = 300//1 = 300; clamped to 150.
    retriever = _retriever(terms_batch_size=150)
    assert retriever._calculate_safe_batch_size({"id": list(range(500))}) == 150


def test_multiple_list_filters_split_available_clauses() -> None:
    # 3 list filters: available 300 // 3 = 100 (< terms_batch_size 150) -> 100.
    retriever = _retriever(terms_batch_size=150)
    filters = {
        "a": list(range(200)),
        "b": list(range(200)),
        "c": list(range(200)),
    }
    assert retriever._calculate_safe_batch_size(filters) == 100


def test_config_override_changes_budget() -> None:
    # A larger clause budget raises the computed batch size (vs the default
    # config), proving the value is config-driven rather than a fixed constant.
    small = _retriever(max_total_clauses=600, reserved_clauses=300)
    large = _retriever(max_total_clauses=2000, reserved_clauses=0)
    filters = {key: list(range(999)) for key in "abcd"}
    assert small._calculate_safe_batch_size(filters) == 75
    assert large._calculate_safe_batch_size(filters) == 150


@pytest.mark.parametrize("list_filters", [1, 2])
def test_configured_batch_size_is_never_raised(list_filters: int) -> None:
    # Regression: a floor of half the per-filter clause share (150 with the
    # defaults) overrode a smaller configured terms_batch_size.
    retriever = _retriever(terms_batch_size=50)
    filters = {f"f{i}": list(range(500)) for i in range(list_filters)}
    assert retriever._calculate_safe_batch_size(filters) == 50


def test_exhausted_clause_budget_still_batches_one_term() -> None:
    retriever = _retriever(max_total_clauses=300, reserved_clauses=300)
    assert retriever._calculate_safe_batch_size({"id": list(range(10))}) == 1


async def test_every_batch_of_one_filter_meets_every_batch_of_another() -> None:
    # Regression: batch i of filter A was paired only with batch i of filter B,
    # so a document whose A value is in A's first batch and whose B value is
    # in B's second batch matched no request.
    retriever = _retriever()
    documents = [
        {"id": "d1", "a": 0, "b": 3},
        {"id": "d2", "a": 2, "b": 1},
        {"id": "d3", "a": 1, "b": 0},
    ]
    sent: list[dict] = []

    def _create_search_tasks(query, *_args):
        sent.append(query.filters)

        async def _search() -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    content="", score=1.0, source=doc["id"], retriever_type="entity"
                )
                for doc in documents
                if doc["a"] in query.filters["a"] and doc["b"] in query.filters["b"]
            ]

        return [_search()]

    object.__setattr__(retriever, "_create_search_tasks", _create_search_tasks)
    large = {"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]}
    query = SearchQuery(query="", filters={**large, "type": "ORG"})

    results = await retriever._execute_multi_batched_retrieval(
        query, large, 2, SearchType.LEXICAL, ["entities"], None
    )

    assert sorted(r.source for r in results) == ["d1", "d2", "d3"]
    assert len(sent) == 4
    assert all(filters["type"] == "ORG" for filters in sent)
