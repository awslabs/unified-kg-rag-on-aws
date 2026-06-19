# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for config-driven OpenSearch clause-budget batching.

The clause-budget knobs moved from hardcoded ClassVars to config; this verifies
``_calculate_safe_batch_size`` reads the instance attributes (set from config in
__init__) without constructing any AWS client.
"""

from __future__ import annotations

import pytest

from aws_graphrag.retrieval.retrievers.opensearch_retriever import OpenSearchRetriever

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
    filters = {"a": list(range(999)), "b": list(range(999))}
    assert large._calculate_safe_batch_size(filters) > small._calculate_safe_batch_size(
        filters
    )
