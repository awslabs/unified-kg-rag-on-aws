# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The AUTO router prompt describes exactly the routable strategies."""

import logging

import pytest
from pydantic import ValidationError

from unified_kg_rag.domain.models import SearchStrategy
from unified_kg_rag.domain.models.config import SearchConfig, _fusion
from unified_kg_rag.domain.prompts import StrategySelectionPrompt
from unified_kg_rag.domain.prompts.retrieval import (
    STRATEGY_ROUTING_GUIDE,
    describe_routable_strategies,
)

pytestmark = pytest.mark.unit

CONFIG_LOGGER = "unified_kg_rag.domain.models.config"


def test_every_selectable_strategy_has_a_guide_entry() -> None:
    assert set(STRATEGY_ROUTING_GUIDE) == set(SearchStrategy) - {SearchStrategy.AUTO}


def test_guide_lists_only_the_given_strategies_in_order() -> None:
    guide = describe_routable_strategies([SearchStrategy.MIX, SearchStrategy.SIMPLE])
    assert guide.startswith("1. MIX SEARCH")
    assert "\n\n2. SIMPLE SEARCH" in guide
    for absent in ("LOCAL", "GLOBAL", "DRIFT", "HYBRID", "NAIVE"):
        assert f"{absent} SEARCH" not in guide


def test_prompt_names_no_strategy_of_its_own() -> None:
    # The strategy list and the decision guidance come only from the config, so
    # the router is never steered toward a strategy it cannot pick.
    template = StrategySelectionPrompt.resolve().system_prompt_template
    assert "{strategy_descriptions}" in template
    for strategy in STRATEGY_ROUTING_GUIDE:
        assert f"{strategy.value.upper()} SEARCH" not in template
        assert f"→ {strategy.value.upper()}" not in template


def test_auto_is_not_routable() -> None:
    with pytest.raises(ValidationError, match="cannot include 'auto'"):
        SearchConfig(auto_routable_strategies=[SearchStrategy.AUTO])


@pytest.mark.parametrize(
    ("strategy", "evidence"),
    [
        (SearchStrategy.PPR, "#the-ppr-strategy"),
        (SearchStrategy.TREE, "#raptor-summary-tree"),
    ],
)
def test_unmeasured_routes_load_with_a_warning_citing_the_evidence(
    strategy: SearchStrategy, evidence: str, caplog: pytest.LogCaptureFixture
) -> None:
    # The operator's call: the config loads and auto may pick it, with a
    # WARNING naming the measurement it was left out on.
    _fusion._warn_unmeasured_route.cache_clear()
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        config = SearchConfig(auto_routable_strategies=[SearchStrategy.LOCAL, strategy])
    assert config.auto_routable_strategies == [SearchStrategy.LOCAL, strategy]
    warned = [r.getMessage() for r in caplog.records]
    assert len(warned) == 1
    assert f"includes '{strategy.value}'" in warned[0]
    assert evidence in warned[0]


def test_default_routes_load_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _fusion._warn_unmeasured_route.cache_clear()
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        SearchConfig()
    assert not caplog.records
