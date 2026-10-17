# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The user guide names exactly the prompts that have no ``custom_prompts`` key."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import unified_kg_rag.domain.prompts as prompts
from unified_kg_rag.domain.models.config import CustomPromptConfig
from unified_kg_rag.domain.prompts.base import BasePrompt

pytestmark = pytest.mark.unit

_DOCS = Path(__file__).resolve().parents[2] / "docs"


def _prompt_classes() -> set[type[BasePrompt]]:
    found: set[type[BasePrompt]] = set()
    for info in pkgutil.iter_modules(prompts.__path__):
        module = importlib.import_module(f"{prompts.__name__}.{info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BasePrompt) and cls is not BasePrompt:
                found.add(cls)
    return found


def _not_overridable() -> set[str]:
    fields = set(CustomPromptConfig.model_fields)
    return {
        cls.__name__
        for cls in _prompt_classes()
        if f"{getattr(cls, 'prompt_key', None)}_system" not in fields
    }


@pytest.mark.parametrize("guide", ["user-guide.md", "user-guide.ko.md"])
def test_the_guide_lists_the_prompts_without_an_override(guide: str) -> None:
    # The guide said every prompt was overridable; five are not.
    text = (_DOCS / guide).read_text(encoding="utf-8")
    section = text[text.index("custom_prompts.*_system") - 600 :][:1500]
    missing = sorted(name for name in _not_overridable() if f"`{name}`" not in section)
    assert len(_not_overridable()) == 5
    assert not missing, f"{guide} does not name {missing} as not overridable"
