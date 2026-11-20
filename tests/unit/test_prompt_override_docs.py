# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The domain-adaptation guide names exactly the prompts that have no
``custom_prompts`` key."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.prompt_classes import shipped_prompts
from unified_kg_rag.domain.models.config import CustomPromptConfig

pytestmark = pytest.mark.unit

_DOCS = Path(__file__).resolve().parents[2] / "docs"


def _not_overridable() -> set[str]:
    fields = set(CustomPromptConfig.model_fields)
    return {
        cls.__name__
        for cls in shipped_prompts()
        if f"{getattr(cls, 'prompt_key', None)}_system" not in fields
    }


@pytest.mark.parametrize(
    "guide", ["guides/domain-adaptation.md", "guides/domain-adaptation.ko.md"]
)
def test_the_guide_lists_the_prompts_without_an_override(guide: str) -> None:
    # The guide said every prompt was overridable; two are not (the
    # chunker's and the translator's have keys since).
    text = (_DOCS / guide).read_text(encoding="utf-8")
    section = text[text.index("custom_prompts.*_system") - 600 :][:1500]
    missing = sorted(name for name in _not_overridable() if f"`{name}`" not in section)
    assert _not_overridable() == {"TranslationPrompt", "CommunityRelevancePrompt"}
    assert not missing, f"{guide} does not name {missing} as not overridable"
