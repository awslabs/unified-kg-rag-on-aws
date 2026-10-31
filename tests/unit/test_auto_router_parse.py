# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AUTO strategy-router response parsing.

Regression: the router parsed the LLM response with SearchStrategy(text.strip()
.lower()), which raised ValueError on anything but a bare enum word ("Local
search.", "I'd use local") and then fell back to DRIFT — the MOST expensive
strategy. Parsing now tolerates surrounding text and defaults to LOCAL.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import SearchStrategy

pytestmark = pytest.mark.unit

parse = GraphRAGChain._parse_routed_strategy
# drift is routable only when configured so.
WITH_DRIFT = (
    SearchStrategy.LOCAL,
    SearchStrategy.MIX,
    SearchStrategy.GLOBAL,
    SearchStrategy.DRIFT,
)


def test_exact_word() -> None:
    assert parse("local") == SearchStrategy.LOCAL
    assert parse("GLOBAL") == SearchStrategy.GLOBAL
    assert parse("  drift  ", WITH_DRIFT) == SearchStrategy.DRIFT
    assert parse("drift") == SearchStrategy.LOCAL  # not routable by default
    assert parse("mix") == SearchStrategy.MIX


def test_tolerates_surrounding_text() -> None:
    assert parse("Local search.") == SearchStrategy.LOCAL
    assert parse("I'd recommend global") == SearchStrategy.GLOBAL


def test_unknown_defaults_to_local_not_drift() -> None:
    # The crux: an unparseable response must NOT land on the costly DRIFT.
    assert parse("banana") == SearchStrategy.LOCAL
    assert parse("") == SearchStrategy.LOCAL


def test_exact_match_precedence_over_substring() -> None:
    # "global" contains no other strategy name; ensure exact wins cleanly.
    assert parse("global") == SearchStrategy.GLOBAL


def test_first_mentioned_strategy_wins_not_list_order() -> None:
    # A substring scan returned whichever name it checked first.
    assert parse("drift, or else local", WITH_DRIFT) == SearchStrategy.DRIFT
    assert parse("local rather than drift", WITH_DRIFT) == SearchStrategy.LOCAL


def test_words_are_matched_whole() -> None:
    # "mixture" / "globally" are not strategy names.
    assert parse("a mixture, globally") == SearchStrategy.LOCAL


def test_simple_is_not_routable_by_default() -> None:
    # simple stays selectable explicitly but is not an AUTO target by default.
    assert parse("simple") == SearchStrategy.LOCAL
    assert parse("simple", (SearchStrategy.SIMPLE,)) == SearchStrategy.SIMPLE


def test_default_routable_set_and_fast_router_model() -> None:
    from unified_kg_rag.domain.models import Config, LanguageModelId, ModelRole

    config = Config()
    search = config.search
    # drift (up to 27 LLM calls a query) is selectable but not routed to.
    assert search.auto_routable_strategies == [
        SearchStrategy.LOCAL,
        SearchStrategy.MIX,
        SearchStrategy.GLOBAL,
    ]
    assert (
        config.model_id(ModelRole.STRATEGY_SELECTION)
        == LanguageModelId.CLAUDE_V5_5_HAIKU
    )


def test_unknown_defaults_to_the_first_routable_when_local_is_excluded() -> None:
    # A configuration without LOCAL (its stores may not be provisioned) must
    # never be routed to it.
    routable = (SearchStrategy.MIX, SearchStrategy.GLOBAL)
    assert parse("banana", routable) == SearchStrategy.MIX


async def test_router_failure_falls_back_to_a_routable_strategy() -> None:
    from unified_kg_rag.domain.models import Config

    config = Config()
    config.search.auto_routable_strategies = [SearchStrategy.GLOBAL, SearchStrategy.MIX]
    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = config
    chain.ignore_errors = True

    def _failing_router(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("router down")

    chain._get_chain_for_prompt = _failing_router  # type: ignore[method-assign]
    state = await chain._resolve_strategy(
        {"search_strategy": SearchStrategy.AUTO, "query": "q"}
    )
    assert state["resolved_strategy"] == SearchStrategy.GLOBAL


async def test_one_routable_strategy_needs_no_router_call() -> None:
    # Regression: the router LLM was called to choose among one strategy.
    from unified_kg_rag.domain.models import Config

    config = Config()
    config.search.auto_routable_strategies = [SearchStrategy.GLOBAL]
    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = config
    chain.ignore_errors = False

    def _no_router(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the router was called")

    chain._get_chain_for_prompt = _no_router  # type: ignore[method-assign]
    state = await chain._resolve_strategy(
        {"search_strategy": SearchStrategy.AUTO, "query": "q"}
    )
    assert state["resolved_strategy"] == SearchStrategy.GLOBAL


async def test_auto_never_routes_to_report_strategies_without_communities() -> None:
    # global and drift answer from community reports; with community detection
    # off there are none, so the router must not be offered them.
    from unified_kg_rag.domain.models import Config

    config = Config()
    config.graph.community_detection.enabled = False
    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = config
    chain.ignore_errors = False
    offered: dict[str, object] = {}

    class _Router:
        async def ainvoke(self, inputs: dict, _config: object = None) -> str:
            offered.update(inputs)
            return "global"

    chain._get_chain_for_prompt = lambda *_a, **_k: _Router()  # type: ignore[method-assign]
    state = await chain._resolve_strategy(
        {"search_strategy": SearchStrategy.AUTO, "query": "q"}
    )
    assert offered["strategies"] == "local, mix"
    # "global" is not routable here, so the answer falls back to local.
    assert state["resolved_strategy"] == SearchStrategy.LOCAL


def test_routable_set_follows_what_the_pipeline_builds() -> None:
    import unified_kg_rag.adapters.search_strategies  # noqa: F401
    from unified_kg_rag.adapters.retrieval.strategy_registry import (
        routable_strategies,
    )
    from unified_kg_rag.domain.models import Config

    config = Config()
    assert routable_strategies(config) == tuple(config.search.auto_routable_strategies)
    config.graph.community_detection.enabled = False
    assert routable_strategies(config) == (SearchStrategy.LOCAL, SearchStrategy.MIX)
    config.search.auto_routable_strategies = [SearchStrategy.GLOBAL]
    assert routable_strategies(config) == (SearchStrategy.LOCAL,)
