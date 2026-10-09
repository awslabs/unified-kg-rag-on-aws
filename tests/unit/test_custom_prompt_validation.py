# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""custom_prompts overrides are checked against each prompt's variables at load.

A human override without ``{input_text}`` used to run silently on no document
text, and a literal brace raised ``KeyError`` on every chunk at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from unified_kg_rag.application.prompts.tuner import CorpusProfile, PromptTuner
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import CustomPromptConfig
from unified_kg_rag.shared.config import ConfigLoader

pytestmark = pytest.mark.unit


def test_override_missing_a_required_variable_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"graph_extraction.*\{input_text\}"):
        CustomPromptConfig(graph_extraction_human="Extract entities, please.")


def test_unknown_variable_is_rejected_with_the_escaping_hint() -> None:
    with pytest.raises(ValidationError) as exc:
        CustomPromptConfig(
            graph_extraction_human='{input_text}\nReturn {"name": "Vendor"}.'
        )
    message = str(exc.value)
    assert "unknown variable" in message and '"name"' in message
    assert "{{" in message  # tells the user how to write a literal brace


def test_unbalanced_brace_is_rejected() -> None:
    with pytest.raises(ValidationError, match="custom_prompts.answer_generation_human"):
        CustomPromptConfig(answer_generation_human="{query} {context} and a } here")


def test_valid_overrides_pass() -> None:
    CustomPromptConfig(
        # A system-only override keeps the built-in human template's variables.
        graph_extraction_system="You extract entities from contracts.",
        # Optional knobs (entity_types here) may be dropped; escaped braces are
        # literal text.
        graph_extraction_human=(
            "Text: {input_text}\nAt most {max_entities_per_chunk} entities and "
            '{max_relationships_per_chunk} relationships. Format: {{"name": ...}}'
        ),
        answer_generation_human="Q: {query}\nContext: {context}",
    )


def test_config_file_error_names_the_override(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"custom_prompts": {"claim_extraction_human": "no text"}})
    )
    with pytest.raises(ValueError, match="claim_extraction"):
        ConfigLoader(path).load_config()


def test_documented_user_guide_example_is_valid() -> None:
    # The custom_prompts example in docs/user-guide.md §9.B.
    Config(
        custom_prompts=CustomPromptConfig(
            graph_extraction_system="You are a medical knowledge extractor.",
            graph_extraction_human=(
                "Extract medical entities and relationships from this clinical "
                "text:\n{input_text}\nExtraction Limits:\n"
                "- Maximum Entities: {max_entities_per_chunk}\n"
                "- Maximum Relationships: {max_relationships_per_chunk}\n"
            ),
        )
    )


def test_tuned_prompts_escape_literal_braces() -> None:
    # Generated text (persona, few-shot examples) is literal; braces in it
    # must not become template variables.
    profile = CorpusProfile(
        domain="supply {contracts}",
        language="English",
        persona="You read {braces}.",
        entity_types=["VENDOR"],
        few_shot_examples='{"name": "Vendor"}',
    )
    overrides = PromptTuner.build_custom_prompts(profile)
    CustomPromptConfig(**overrides)  # loads without a variable error
    assert "{{braces}}" in overrides["graph_extraction_system"]
