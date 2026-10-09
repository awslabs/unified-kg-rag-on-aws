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

_CONFIG_LOGGER = "unified_kg_rag.domain.models.config"


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


def test_documented_user_guide_example_is_valid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The custom_prompts example in docs/user-guide.md §9.B: a human-only
    # override keeps the built-in system prompt, so no output tag is lost.
    with caplog.at_level("WARNING", logger=_CONFIG_LOGGER):
        Config(
            custom_prompts=CustomPromptConfig(
                graph_extraction_human=(
                    "Extract medical entities and relationships from this "
                    "clinical text:\n{input_text}\nExtraction Limits:\n"
                    "- Maximum Entities: {max_entities_per_chunk}\n"
                    "- Maximum Relationships: {max_relationships_per_chunk}\n"
                ),
                entity_extraction_system=(
                    "You are a financial expert. Extract companies, instruments, "
                    "markets, and metrics from user queries."
                ),
            )
        )
    assert "output tag" not in caplog.text


def test_system_override_without_the_output_format_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Valid (loads), but the parser would find none of its tags: warn by name.
    with caplog.at_level("WARNING", logger=_CONFIG_LOGGER):
        CustomPromptConfig(
            graph_extraction_system="You are a medical knowledge extractor.",
            community_report_system="You are a legal analyst.",
        )
    records = [r.getMessage() for r in caplog.records if "output tag" in r.getMessage()]
    assert len(records) == 2
    extraction = next(m for m in records if "graph_extraction" in m)
    assert "<entities>" in extraction and "<relationships>" in extraction
    # The built-in human template mentions <source_text>, so only the schema
    # tags the system prompt carried are reported missing.
    report = next(m for m in records if "community_report" in m)
    for tag in ("<community_name>", "<rating>", "<findings>"):
        assert tag in report


def test_override_that_keeps_the_format_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unified_kg_rag.domain.prompts import GraphExtractionPrompt

    with caplog.at_level("WARNING", logger=_CONFIG_LOGGER):
        CustomPromptConfig(
            graph_extraction_system=(
                "You extract entities from contracts.\n\n"
                + GraphExtractionPrompt.output_rules
            )
        )
    assert "output tag" not in caplog.text


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
