# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-strategy traits (``STRATEGY_TRAITS``): one row per strategy and one
fallback order the router, structured search and the chain share."""

from __future__ import annotations

import pytest

from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    GENERAL_FALLBACKS,
    STRATEGY_TRAITS,
    Config,
    SearchStrategy,
)
from unified_kg_rag.domain.prompts.retrieval import STRATEGY_ROUTING_GUIDE

pytestmark = pytest.mark.unit


def test_every_strategy_has_traits() -> None:
    assert set(STRATEGY_TRAITS) == set(SearchStrategy)


def test_every_routable_strategy_has_a_router_guide() -> None:
    routable = {s for s, t in STRATEGY_TRAITS.items() if t.auto_routable}
    assert routable <= set(STRATEGY_ROUTING_GUIDE)
    assert {s for s in SearchStrategy if not STRATEGY_TRAITS[s].auto_routable} == {
        SearchStrategy.AUTO,
        SearchStrategy.PPR,
        SearchStrategy.TREE,
    }


def test_the_default_routable_strategies_are_allowed() -> None:
    for strategy in Config().search.auto_routable_strategies:
        assert STRATEGY_TRAITS[strategy].auto_routable


def test_one_general_fallback_order() -> None:
    assert GENERAL_FALLBACKS == (SearchStrategy.LOCAL, SearchStrategy.MIX)


@pytest.mark.parametrize(
    ("routable", "chosen"),
    [
        ((SearchStrategy.GLOBAL, SearchStrategy.LOCAL), SearchStrategy.LOCAL),
        # Regression: without local the chain took routable[0], while the
        # router's and structured search's fallbacks took mix.
        ((SearchStrategy.GLOBAL, SearchStrategy.MIX), SearchStrategy.MIX),
        ((SearchStrategy.GLOBAL, SearchStrategy.NAIVE), SearchStrategy.GLOBAL),
        ((), SearchStrategy.LOCAL),
    ],
)
def test_the_chain_falls_back_in_that_order(
    routable: tuple[SearchStrategy, ...], chosen: SearchStrategy
) -> None:
    assert GraphRAGChain._fallback_strategy(routable) is chosen


def test_global_reports_no_sources_from_key_points_alone() -> None:
    config = Config()
    reports = STRATEGY_TRAITS[SearchStrategy.GLOBAL].reports_sources
    assert reports(config)
    config.search.global_search.enable_map_reduce = True
    config.search.global_search.evidence_tokens = 0
    assert not reports(config)
    assert STRATEGY_TRAITS[SearchStrategy.LOCAL].reports_sources(config)
