# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The AUTO router prompt describes exactly the routable strategies."""

import pytest
from pydantic import ValidationError

from unified_kg_rag.domain.models import SearchStrategy
from unified_kg_rag.domain.models.config import SearchConfig
from unified_kg_rag.domain.prompts import StrategySelectionPrompt
from unified_kg_rag.domain.prompts.retrieval import (
    STRATEGY_ROUTING_GUIDE,
    describe_routable_strategies,
)

pytestmark = pytest.mark.unit


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
