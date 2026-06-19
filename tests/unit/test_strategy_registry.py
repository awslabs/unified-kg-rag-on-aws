# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for the search-strategy registry (M1 dispatch refactor)."""

from __future__ import annotations

import pytest

# Importing the package triggers @register_strategy on every strategy module.
import aws_graphrag.retrieval.search_strategies  # noqa: F401
from aws_graphrag.models import RetrieverRole, SearchStrategy
from aws_graphrag.retrieval.search_strategies import (
    DriftSearchStrategy,
    GlobalSearchStrategy,
    LocalSearchStrategy,
    SimpleSearchStrategy,
)
from aws_graphrag.retrieval.strategy_registry import (
    StrategySpec,
    get_strategy_spec,
    register_strategy,
    registered_strategies,
)

pytestmark = pytest.mark.unit


def test_all_builtin_strategies_registered() -> None:
    registered = set(registered_strategies())
    assert {
        SearchStrategy.SIMPLE,
        SearchStrategy.LOCAL,
        SearchStrategy.GLOBAL,
        SearchStrategy.DRIFT,
    } <= registered


@pytest.mark.parametrize(
    ("strategy", "expected_class"),
    [
        (SearchStrategy.SIMPLE, SimpleSearchStrategy),
        (SearchStrategy.LOCAL, LocalSearchStrategy),
        (SearchStrategy.GLOBAL, GlobalSearchStrategy),
        (SearchStrategy.DRIFT, DriftSearchStrategy),
    ],
)
def test_spec_maps_to_correct_class(
    strategy: SearchStrategy, expected_class: type
) -> None:
    assert get_strategy_spec(strategy).strategy_class is expected_class


def test_simple_requires_only_document_role() -> None:
    spec = get_strategy_spec(SearchStrategy.SIMPLE)
    assert spec.required_roles == (RetrieverRole.DOCUMENT,)


@pytest.mark.parametrize(
    "strategy",
    [SearchStrategy.LOCAL, SearchStrategy.GLOBAL, SearchStrategy.DRIFT],
)
def test_graph_strategies_require_both_roles(strategy: SearchStrategy) -> None:
    spec = get_strategy_spec(strategy)
    assert set(spec.required_roles) == {
        RetrieverRole.DOCUMENT,
        RetrieverRole.GRAPH,
    }


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="No search strategy registered"):
        get_strategy_spec(SearchStrategy.AUTO)


def test_register_is_idempotent_for_same_class() -> None:
    # Re-decorating the same class under the same key must not raise.
    spec = get_strategy_spec(SearchStrategy.SIMPLE)
    register_strategy(SearchStrategy.SIMPLE, required_roles=(RetrieverRole.DOCUMENT,))(
        spec.strategy_class
    )


def test_register_conflicting_class_raises() -> None:
    class _Other:
        pass

    with pytest.raises(ValueError, match="already registered"):
        register_strategy(SearchStrategy.SIMPLE)(_Other)  # type: ignore[arg-type]


def test_strategy_spec_default_roles() -> None:
    class _Dummy:
        pass

    spec = StrategySpec(strategy_class=_Dummy)  # type: ignore[arg-type]
    assert set(spec.required_roles) == {
        RetrieverRole.DOCUMENT,
        RetrieverRole.GRAPH,
    }
