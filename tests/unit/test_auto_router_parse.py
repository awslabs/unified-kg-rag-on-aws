# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AUTO strategy-router response parsing.

Regression: the router parsed the LLM response with SearchStrategy(text.strip()
.lower()), which raised ValueError on anything but a bare enum word ("Local
search.", "I'd use local") and then fell back to DRIFT — the MOST expensive
strategy. A word scan that replaced it read "Not global ... local" as global.
The router now answers ``{"strategy": "<name>"}`` (a bare name is read too),
gets one repair when it does not parse, and defaults to LOCAL.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import Config, SearchStrategy

pytestmark = pytest.mark.unit

DEFAULT = tuple(Config().search.auto_routable_strategies)
# drift is routable only when configured so.
WITH_DRIFT = (
    SearchStrategy.LOCAL,
    SearchStrategy.MIX,
    SearchStrategy.GLOBAL,
    SearchStrategy.DRIFT,
)


async def parse(
    raw: str,
    routable: tuple[SearchStrategy, ...] = DEFAULT,
    fixed: str = "still not JSON",
) -> SearchStrategy:
    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = Config()
    chain.factory = ScriptedLLMFactory(lambda _system, _human: fixed)
    strategy, _ = await chain._routed_strategy(raw, routable)
    return strategy


async def test_the_json_answer() -> None:
    assert await parse('{"strategy": "global"}') == SearchStrategy.GLOBAL
    assert await parse("{'strategy': 'MIX'}") == SearchStrategy.MIX
    assert await parse('```json\n{"strategy": "drift"}\n```', WITH_DRIFT) == (
        SearchStrategy.DRIFT
    )
    # A model's reasoning block was dropped before the parser (setup_chain).
    assert await parse('{"strategy": "global",}') == SearchStrategy.GLOBAL


async def test_a_bare_name() -> None:
    assert await parse("local") == SearchStrategy.LOCAL
    assert await parse("GLOBAL") == SearchStrategy.GLOBAL
    assert await parse('  "drift". ', WITH_DRIFT) == SearchStrategy.DRIFT
    assert await parse("drift") == SearchStrategy.LOCAL  # not routable by default
    assert await parse("mix") == SearchStrategy.MIX


async def test_prose_is_repaired_not_guessed_from() -> None:
    # The word scan read the first name: "Not global" routed to global.
    assert await parse("Not global; local fits a named entity") == SearchStrategy.LOCAL
    assert (
        await parse("Not local, I'd use global", fixed='{"strategy": "global"}')
        == SearchStrategy.GLOBAL
    )


async def test_unknown_defaults_to_local_not_drift() -> None:
    # The crux: an unparseable response must NOT land on the costly DRIFT.
    assert await parse("banana") == SearchStrategy.LOCAL
    assert await parse("") == SearchStrategy.LOCAL
    assert await parse('{"strategy": "mixture"}') == SearchStrategy.LOCAL


async def test_simple_is_not_routable_by_default() -> None:
    # simple stays selectable explicitly but is not an AUTO target by default.
    assert await parse('{"strategy": "simple"}') == SearchStrategy.LOCAL
    assert await parse("simple", (SearchStrategy.SIMPLE,)) == SearchStrategy.SIMPLE


def test_default_routable_set_and_fast_router_model() -> None:
    from unified_kg_rag.domain.models import Config, LanguageModelId, ModelRole

    config = Config()
    search = config.search
    # drift (up to 27 LLM calls a query) is selectable but not routed to.
    assert search.auto_routable_strategies == [
        SearchStrategy.LOCAL,
        SearchStrategy.MIX,
        SearchStrategy.GLOBAL,
        SearchStrategy.ITERATIVE,
    ]
    assert (
        config.model_id(ModelRole.STRATEGY_SELECTION)
        == LanguageModelId.CLAUDE_V5_5_HAIKU
    )


async def test_unknown_defaults_to_the_first_routable_when_local_is_excluded() -> None:
    # A configuration without LOCAL (its stores may not be provisioned) must
    # never be routed to it.
    routable = (SearchStrategy.MIX, SearchStrategy.GLOBAL)
    assert await parse("banana", routable) == SearchStrategy.MIX


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
    # off there are none (the index holds lightrag only), so the router is
    # not offered GraphRAG's strategies: mix and iterative (over mix) are left.
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
    assert offered["strategies"] == "mix, iterative"
    # "global" is not offered: the fallback is the first candidate.
    assert state["resolved_strategy"] == SearchStrategy.MIX


async def test_auto_routes_among_the_namespace_methodologies() -> None:
    # What the queried namespace holds (a lightrag-only one) leaves mix and
    # hybrid of the configured strategies; global is never offered.
    from unified_kg_rag.domain.models import Config, Methodology
    from unified_kg_rag.domain.models.methodology import METHODOLOGY_PROFILES

    config = Config()
    config.search.auto_routable_strategies = [
        SearchStrategy.LOCAL,
        SearchStrategy.MIX,
        SearchStrategy.GLOBAL,
        SearchStrategy.HYBRID,
    ]
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
        {
            "search_strategy": SearchStrategy.AUTO,
            "query": "q",
            "held_artifacts": METHODOLOGY_PROFILES[Methodology.LIGHTRAG].required,
        }
    )
    assert offered["strategies"] == "mix, hybrid"
    # "global" is not routable here: the fallback is the first routable one.
    assert state["resolved_strategy"] == SearchStrategy.MIX


def test_routable_set_follows_what_the_pipeline_builds() -> None:
    import unified_kg_rag.adapters.search_strategies  # noqa: F401
    from unified_kg_rag.adapters.retrieval.strategy_registry import (
        routable_strategies,
    )
    from unified_kg_rag.domain.models import Config, IndexArtifact, Methodology
    from unified_kg_rag.domain.models.methodology import METHODOLOGY_PROFILES
    from unified_kg_rag.shared import MethodologyNotIndexedError

    config = Config()
    assert routable_strategies(config) == tuple(config.search.auto_routable_strategies)
    config.graph.community_detection.enabled = False
    assert routable_strategies(config) == (SearchStrategy.MIX, SearchStrategy.ITERATIVE)
    # Nothing configured is servable: LightRAG's general-purpose strategy.
    config.search.auto_routable_strategies = [SearchStrategy.GLOBAL]
    assert routable_strategies(config) == (SearchStrategy.MIX,)
    # GraphRAG's on a namespace holding graphrag alone.
    config.graph.community_detection.enabled = True
    config.search.auto_routable_strategies = [SearchStrategy.MIX]
    graphrag = METHODOLOGY_PROFILES[Methodology.GRAPHRAG].required
    assert routable_strategies(config, graphrag) == (SearchStrategy.LOCAL,)
    # Without the community reports: neither global nor drift.
    config.search.auto_routable_strategies = [
        SearchStrategy.GLOBAL,
        SearchStrategy.DRIFT,
        SearchStrategy.LOCAL,
    ]
    assert routable_strategies(config, graphrag) == (SearchStrategy.LOCAL,)
    assert routable_strategies(
        config, graphrag | {IndexArtifact.COMMUNITY_REPORTS}
    ) == tuple(config.search.auto_routable_strategies)
    # A namespace holding nothing a strategy reads.
    with pytest.raises(MethodologyNotIndexedError, match="holds \\(nothing\\)"):
        routable_strategies(config, frozenset())
