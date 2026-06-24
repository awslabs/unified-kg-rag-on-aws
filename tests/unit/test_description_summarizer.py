# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for description re-summarization on merge (AWS-free).

Covers the config defaults, the summarization prompt's variable contract, and
the DescriptionSummarizer adapter: it must skip below-threshold descriptions
(no LLM call), summarize over-threshold ones, degrade to the concatenated text
on LLM failure, and never touch non-description fields. The Bedrock/boto wiring
is patched out so __init__ never touches AWS.
"""

from __future__ import annotations

import pytest

import aws_graphrag.adapters.ingestion.description_summarizer as ds_module
from aws_graphrag.adapters.ingestion.description_summarizer import DescriptionSummarizer
from aws_graphrag.domain.models import Config, Entity, Relationship
from aws_graphrag.domain.models.config import DescriptionSummarizationConfig
from aws_graphrag.domain.prompts import DescriptionSummarizationPrompt

pytestmark = pytest.mark.unit


# A description comfortably over the default 600-token threshold. The script-
# aware estimator uses max(word_count, chars // 4), so ~4000 chars -> ~1000.
LONG_DESCRIPTION = "fact " * 1000


@pytest.fixture
def summarizer(config: Config, mocker) -> DescriptionSummarizer:
    """A real DescriptionSummarizer with all AWS/Bedrock wiring stubbed out."""
    mocker.patch.object(ds_module, "boto3")
    mocker.patch.object(ds_module, "BedrockLanguageModelFactory")
    mocker.patch.object(ds_module, "setup_chain")
    return DescriptionSummarizer(config)


# --------------------------------------------------------------------------- #
# DescriptionSummarizationConfig defaults
# --------------------------------------------------------------------------- #
class TestConfigDefaults:
    def test_attached_to_graph_extraction(self) -> None:
        cfg = Config().processing.graph_extraction.description_summarization
        assert isinstance(cfg, DescriptionSummarizationConfig)

    def test_default_values(self) -> None:
        cfg = DescriptionSummarizationConfig()
        assert cfg.enabled is True
        assert cfg.summary_model_id.value == "anthropic.claude-haiku-4-5-20251001-v1:0"
        assert cfg.force_summary_threshold_tokens == 600
        assert cfg.max_summary_tokens == 256

    def test_thresholds_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            DescriptionSummarizationConfig(force_summary_threshold_tokens=0)
        with pytest.raises(ValueError):
            DescriptionSummarizationConfig(max_summary_tokens=0)


# --------------------------------------------------------------------------- #
# DescriptionSummarizationPrompt
# --------------------------------------------------------------------------- #
class TestPrompt:
    def test_required_variables_declared(self) -> None:
        assert set(DescriptionSummarizationPrompt.input_variables) == {
            "entity_name",
            "descriptions",
            "max_summary_tokens",
            "language",
            "target_language",
        }

    def test_resolves_and_renders(self) -> None:
        resolved = DescriptionSummarizationPrompt.resolve()
        rendered = resolved.human_prompt_template.format(
            entity_name="Acme Corp",
            descriptions="A company; A big company",
            max_summary_tokens="256",
            language="en",
            target_language="en",
        )
        assert "Acme Corp" in rendered
        assert "256" in rendered

    def test_custom_override_respected(self) -> None:
        from aws_graphrag.domain.models.config import CustomPromptConfig

        resolved = DescriptionSummarizationPrompt.resolve(
            custom_prompts=CustomPromptConfig(
                description_summarization_human="custom {entity_name}"
            )
        )
        assert resolved.human_prompt_template == "custom {entity_name}"


# --------------------------------------------------------------------------- #
# summarize_entities
# --------------------------------------------------------------------------- #
class TestSummarizeEntities:
    def test_below_threshold_skips_llm(self, summarizer, mocker) -> None:
        spy = mocker.spy(summarizer, "_summarize_many")
        ents = [Entity(id="e1", name="Alice", description="short desc")]
        out = summarizer.summarize_entities(ents)
        assert out[0].description == "short desc"
        spy.assert_not_called()

    def test_above_threshold_summarized(self, summarizer, mocker) -> None:
        summarizer.summarizer.batch = mocker.Mock(return_value=["A concise summary."])
        ents = [Entity(id="e1", name="Alice", description=LONG_DESCRIPTION)]
        out = summarizer.summarize_entities(ents)
        assert out[0].description == "A concise summary."

    def test_disabled_is_noop(self, summarizer, mocker) -> None:
        summarizer.summarization_config.enabled = False
        spy = mocker.spy(summarizer, "_summarize_many")
        ents = [Entity(id="e1", name="Alice", description=LONG_DESCRIPTION)]
        out = summarizer.summarize_entities(ents)
        assert out[0].description == LONG_DESCRIPTION
        spy.assert_not_called()

    def test_llm_failure_keeps_concatenation(self, summarizer, mocker) -> None:
        # The whole batch pass raises -> keep every original description.
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        summarizer.batch_processor = fake_bp
        ents = [Entity(id="e1", name="Alice", description=LONG_DESCRIPTION)]
        out = summarizer.summarize_entities(ents)
        assert out[0].description == LONG_DESCRIPTION

    def test_empty_summary_keeps_concatenation(self, summarizer, mocker) -> None:
        summarizer.summarizer.batch = mocker.Mock(return_value=["   "])
        ents = [Entity(id="e1", name="Alice", description=LONG_DESCRIPTION)]
        out = summarizer.summarize_entities(ents)
        assert out[0].description == LONG_DESCRIPTION

    def test_non_description_fields_preserved(self, summarizer, mocker) -> None:
        summarizer.summarizer.batch = mocker.Mock(return_value=["summary"])
        ent = Entity(
            id="e1",
            name="Alice",
            type="PERSON",
            description=LONG_DESCRIPTION,
            text_unit_ids=["t1", "t2"],
            frequency=2,
            confidence=0.9,
        )
        out = summarizer.summarize_entities([ent])[0]
        assert out.description == "summary"
        assert out.id == "e1"
        assert out.name == "Alice"
        assert out.type == "PERSON"
        assert out.text_unit_ids == ["t1", "t2"]
        assert out.frequency == 2
        assert out.confidence == 0.9

    def test_only_over_threshold_items_sent_to_llm(self, summarizer, mocker) -> None:
        batch = mocker.Mock(return_value=["summary"])
        summarizer.summarizer.batch = batch
        ents = [
            Entity(id="e1", name="Alice", description="short"),
            Entity(id="e2", name="Bob", description=LONG_DESCRIPTION),
        ]
        out = summarizer.summarize_entities(ents)
        # Only the long one was summarized; short one untouched.
        assert out[0].description == "short"
        assert out[1].description == "summary"
        # Exactly one input item was prepared for the LLM.
        sent_inputs = batch.call_args.args[0]
        assert len(sent_inputs) == 1
        assert sent_inputs[0]["entity_name"] == "Bob"

    def test_empty_input(self, summarizer) -> None:
        assert summarizer.summarize_entities([]) == []


# --------------------------------------------------------------------------- #
# summarize_relationships
# --------------------------------------------------------------------------- #
class TestSummarizeRelationships:
    def test_above_threshold_summarized_preserves_weight(
        self, summarizer, mocker
    ) -> None:
        summarizer.summarizer.batch = mocker.Mock(return_value=["rel summary"])
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            source_name="Alice",
            target_name="Acme",
            type="WORKS_AT",
            weight=4.0,
            description=LONG_DESCRIPTION,
            text_unit_ids=["t1"],
        )
        out = summarizer.summarize_relationships([rel])[0]
        assert out.description == "rel summary"
        assert out.weight == 4.0
        assert out.text_unit_ids == ["t1"]
        assert out.type == "WORKS_AT"

    def test_below_threshold_skips_llm(self, summarizer, mocker) -> None:
        spy = mocker.spy(summarizer, "_summarize_many")
        rels = [Relationship(id="r1", source_id="e1", target_id="e2", description="x")]
        out = summarizer.summarize_relationships(rels)
        assert out[0].description == "x"
        spy.assert_not_called()

    def test_empty_input(self, summarizer) -> None:
        assert summarizer.summarize_relationships([]) == []
