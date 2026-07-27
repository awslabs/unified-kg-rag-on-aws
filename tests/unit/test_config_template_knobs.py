# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard that config-template.yaml stays in sync with the retrieval-tuning models.

Every tuned retrieval number is a config knob rather than a module constant, which
only helps operators if the template documents it AND the documented value parses
into the model. A knob that drifts from its default (or fails to parse) is a silent
trap: the template reads as authoritative but the running system uses something
else. AWS-free — this only loads YAML and validates Pydantic models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from unified_kg_rag.domain.models.config import SearchConfig

pytestmark = pytest.mark.unit

TEMPLATE = Path(__file__).resolve().parents[2] / "config-template.yaml"


@pytest.fixture(scope="module")
def template_search() -> SearchConfig:
    raw: dict[str, Any] = yaml.safe_load(TEMPLATE.read_text())
    return SearchConfig(**raw["search"])


def test_template_search_block_parses(template_search: SearchConfig) -> None:
    assert template_search.token_manager.type_budgets is not None
    assert template_search.local_search.type_quota is not None


def test_context_type_budgets_match_model_defaults(
    template_search: SearchConfig,
) -> None:
    assert (
        template_search.token_manager.type_budgets.model_dump()
        == SearchConfig().token_manager.type_budgets.model_dump()
    )


def test_local_type_quota_matches_model_defaults(
    template_search: SearchConfig,
) -> None:
    assert (
        template_search.local_search.type_quota.model_dump()
        == SearchConfig().local_search.type_quota.model_dump()
    )


@pytest.mark.parametrize(
    "field",
    ["kg_stream_top_k", "chunk_stream_top_k", "related_chunk_number"],
)
def test_lightrag_width_knobs_match_model_defaults(
    template_search: SearchConfig, field: str
) -> None:
    assert getattr(template_search.lightrag_search, field) == getattr(
        SearchConfig().lightrag_search, field
    )


def test_truncation_floor_matches_model_default(template_search: SearchConfig) -> None:
    assert (
        template_search.token_manager.min_truncated_section_tokens
        == SearchConfig().token_manager.min_truncated_section_tokens
    )
