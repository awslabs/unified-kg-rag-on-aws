# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the search-strategy registry (M1 dispatch refactor)."""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs this interpreter on a fixed script
import sys
from typing import Any

import pytest

# Importing the package triggers @register_strategy on every strategy module.
import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    _REGISTRY,
    QueryInput,
    StrategySpec,
    get_strategy_spec,
    register_strategy,
)
from unified_kg_rag.adapters.search_strategies import (
    DriftSearchStrategy,
    GlobalSearchStrategy,
    LocalSearchStrategy,
    SimpleSearchStrategy,
)
from unified_kg_rag.domain.models import Config, RetrieverRole, SearchStrategy

pytestmark = pytest.mark.unit


def test_all_builtin_strategies_registered() -> None:
    for strategy in SearchStrategy:
        if strategy is not SearchStrategy.AUTO:  # AUTO is the router
            assert get_strategy_spec(strategy).strategy_class is not None


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


@pytest.mark.parametrize("strategy", [SearchStrategy.SIMPLE, SearchStrategy.GLOBAL])
def test_document_only_strategies_require_only_the_document_role(
    strategy: SearchStrategy,
) -> None:
    # Global search reads community reports from OpenSearch only.
    spec = get_strategy_spec(strategy)
    assert spec.required_roles == (RetrieverRole.DOCUMENT,)


@pytest.mark.parametrize(
    "strategy",
    [SearchStrategy.LOCAL, SearchStrategy.DRIFT],
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


def _same_arguments(spec: StrategySpec) -> dict[str, Any]:
    return {
        "required_roles": spec.required_roles,
        "query_inputs": spec.query_inputs,
        "wraps": spec.wraps,
        "required_artifacts": spec.required_artifacts,
    }


def test_register_is_idempotent_for_same_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Re-decorating the same class with the same arguments keeps its record.
    spec = get_strategy_spec(SearchStrategy.SIMPLE)
    monkeypatch.setitem(_REGISTRY, SearchStrategy.SIMPLE, spec)
    register_strategy(SearchStrategy.SIMPLE, **_same_arguments(spec))(
        spec.strategy_class
    )
    assert _REGISTRY[SearchStrategy.SIMPLE] is spec


@pytest.mark.parametrize(
    "change",
    [
        {"required_roles": (RetrieverRole.DOCUMENT, RetrieverRole.GRAPH)},
        {"query_inputs": frozenset({QueryInput.ENTITIES})},
        {"required_artifacts": frozenset()},
    ],
)
def test_register_same_class_with_other_arguments_raises(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    # It replaced the record: one test re-registering SIMPLE with the default
    # arguments changed what SIMPLE reads for every later test.
    spec = get_strategy_spec(SearchStrategy.SIMPLE)
    monkeypatch.setitem(_REGISTRY, SearchStrategy.SIMPLE, spec)
    arguments = _same_arguments(spec) | change
    assert arguments != _same_arguments(spec)
    with pytest.raises(ValueError, match="other arguments"):
        register_strategy(SearchStrategy.SIMPLE, **arguments)(spec.strategy_class)
    assert _REGISTRY[SearchStrategy.SIMPLE] is spec


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


def test_a_lookup_in_a_fresh_process_finds_the_built_in_strategies() -> None:
    # The strategies register when their package is imported; an entry point
    # that imported none (run-describe) found no strategy at all.
    script = (
        "import sys\n"
        "from unified_kg_rag.adapters.retrieval import strategy_registry as r\n"
        "from unified_kg_rag.domain.models import Config, SearchStrategy\n"
        "assert 'unified_kg_rag.adapters.search_strategies' not in sys.modules\n"
        "print(r.routable_strategies(Config())[0].value)\n"
        "print(r.get_strategy_spec(SearchStrategy.MIX).strategy_class.__name__)\n"
    )
    found = subprocess.run(  # nosec B603 - this interpreter, a fixed script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={**os.environ, "AWS_EC2_METADATA_DISABLED": "true"},
    )
    assert found.returncode == 0, found.stderr[-2000:]
    assert found.stdout.split() == [
        Config().search.auto_routable_strategies[0].value,
        "LightRAGSearchStrategy",
    ]
