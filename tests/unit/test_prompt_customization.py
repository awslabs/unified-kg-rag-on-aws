# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt domain-customization ergonomics (A + B refactor).

A: domain entity types are a {entity_types} template slot fed from config, so a
   domain swap needs no prompt copy-paste.
B: per-prompt overrides resolve by a `prompt_key` convention (no per-class
   _get_custom_prompts boilerplate), and partial overrides don't crash on the
   strict missing-variable validation that guards the shipped defaults.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.domain.models.config import CustomPromptConfig
from unified_kg_rag.domain.prompts import (
    AnswerGenerationPrompt,
    CommunityReportPrompt,
    GraphExtractionPrompt,
)
from unified_kg_rag.domain.prompts.base import BasePrompt

pytestmark = pytest.mark.unit


# --- B: prompt_key generalization -------------------------------------------


def test_prompt_keys_map_to_real_config_fields() -> None:
    # Every prompt that declares a prompt_key (i.e. supports customization) must
    # have matching <key>_system / <key>_human fields on CustomPromptConfig, so
    # the by-convention base lookup actually resolves. Prompts WITHOUT a key are
    # intentionally non-customizable (e.g. internal translation/chunking) — fine.
    import unified_kg_rag.domain.prompts as prompts

    cfg = CustomPromptConfig()
    customizable = [
        getattr(prompts, n)
        for n in prompts.__all__
        if isinstance(getattr(prompts, n), type)
        and issubclass(getattr(prompts, n), BasePrompt)
        and getattr(prompts, n) is not BasePrompt
        and getattr(prompts, n).prompt_key
    ]
    assert customizable, "expected some customizable prompts"
    for cls in customizable:
        assert hasattr(cfg, f"{cls.prompt_key}_system"), cls.__name__
        assert hasattr(cfg, f"{cls.prompt_key}_human"), cls.__name__


def test_system_override_by_key() -> None:
    cp = CustomPromptConfig(graph_extraction_system="MY SYSTEM PROMPT")
    assert (
        GraphExtractionPrompt.resolve(cp).system_prompt_template == "MY SYSTEM PROMPT"
    )


def test_partial_human_only_override_does_not_crash() -> None:
    # The crux of the old "all-or-nothing" trap: overriding only the human
    # template (which need not contain every built-in input variable) must not
    # fail the strict missing-variable validation.
    cp = CustomPromptConfig(answer_generation_human="just answer: ")
    resolved = AnswerGenerationPrompt.resolve(cp)
    assert resolved.human_prompt_template == "just answer: "
    # System stays the shipped default.
    assert (
        resolved.system_prompt_template == AnswerGenerationPrompt.system_prompt_template
    )


def test_no_override_returns_defaults() -> None:
    r = CommunityReportPrompt.resolve()
    assert r.system_prompt_template == CommunityReportPrompt.system_prompt_template


def test_shipped_defaults_still_validated() -> None:
    # A default template missing a declared variable must still raise (the
    # validation guard is preserved for shipped prompts).
    with pytest.raises(ValueError, match="not found"):

        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Bad(BasePrompt):
            input_variables = ["missing_var"]
            system_prompt_template = "no vars here"
            human_prompt_template = "none here either"

        Bad.resolve()


# --- A: entity_types template slot ------------------------------------------


def test_extraction_prompt_has_entity_types_slot() -> None:
    r = GraphExtractionPrompt.resolve()
    assert "{entity_types}" in r.system_prompt_template
    assert "entity_types" in r.input_variables
    # The old hardcoded PERSON/ORGANIZATION list is gone from the template body.
    assert "- **PERSON**" not in r.system_prompt_template


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (
            ["PERSON: a person", "GENE: a gene"],
            "- **PERSON**: a person\n- **GENE**: a gene",
        ),
        (["GENE"], "- **GENE**"),
        (["  DRUG  : meds "], "- **DRUG**: meds"),
    ],
)
def test_format_entity_types(types: list[str], expected: str) -> None:
    assert GraphExtractor._format_entity_types(types) == expected


def test_format_entity_types_empty_lets_model_choose() -> None:
    out = GraphExtractor._format_entity_types([])
    assert "any entity types" in out.lower()
