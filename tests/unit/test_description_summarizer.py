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

import unified_kg_rag.adapters.ingestion.description_summarizer as ds_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
)
from unified_kg_rag.domain.models import Config, Entity, Relationship
from unified_kg_rag.domain.models.config import DescriptionSummarizationConfig
from unified_kg_rag.domain.prompts import DescriptionSummarizationPrompt
from unified_kg_rag.ports.model_factory import LanguageModelInfo
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED

pytestmark = pytest.mark.unit


# A description comfortably over the default 600-token threshold. The script-
# aware estimator uses max(word_count, chars // 4), so ~4000 chars -> ~1000.
LONG_DESCRIPTION = "fact " * 1000


@pytest.fixture
def summarizer(config: Config, mocker) -> DescriptionSummarizer:
    """A real DescriptionSummarizer with all AWS/Bedrock wiring stubbed out."""
    factory = mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    factory.return_value.get_model_info.return_value = LanguageModelInfo(
        context_window_size=200_000, max_output_tokens=8_192
    )
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
        from unified_kg_rag.domain.models.config import CustomPromptConfig

        resolved = DescriptionSummarizationPrompt.resolve(
            custom_prompts=CustomPromptConfig(
                description_summarization_human="custom {entity_name}: {descriptions}"
            )
        )
        assert resolved.human_prompt_template == "custom {entity_name}: {descriptions}"


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
        summarizer.summarizer.invoke = mocker.Mock(return_value="A concise summary.")
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
        summarizer.summarizer.invoke = mocker.Mock(return_value="   ")
        ents = [Entity(id="e1", name="Alice", description=LONG_DESCRIPTION)]
        out = summarizer.summarize_entities(ents)
        assert out[0].description == LONG_DESCRIPTION

    def test_per_item_failure_sentinel_keeps_concatenation(
        self, summarizer, mocker
    ) -> None:
        # A failed item must NOT become its str() description; the original
        # concatenation is kept.
        summarizer.batch_processor = mocker.Mock()
        summarizer.batch_processor.execute_with_fallback.return_value = [
            BATCH_ITEM_FAILED
        ]
        ents = [Entity(id="e1", name="Alice", description=LONG_DESCRIPTION)]
        out = summarizer.summarize_entities(ents)
        assert out[0].description == LONG_DESCRIPTION

    def test_non_description_fields_preserved(self, summarizer, mocker) -> None:
        summarizer.summarizer.invoke = mocker.Mock(return_value="summary")
        ent = Entity(
            id="e1",
            name="Alice",
            type="PERSON",
            description=LONG_DESCRIPTION,
            text_unit_ids=["t1", "t2"],
            confidence=0.9,
        )
        out = summarizer.summarize_entities([ent])[0]
        assert out.description == "summary"
        assert out.id == "e1"
        assert out.name == "Alice"
        assert out.type == "PERSON"
        assert out.text_unit_ids == ["t1", "t2"]
        assert out.confidence == 0.9

    def test_only_over_threshold_items_sent_to_llm(self, summarizer, mocker) -> None:
        invoke = mocker.Mock(return_value="summary")
        summarizer.summarizer.invoke = invoke
        ents = [
            Entity(id="e1", name="Alice", description="short"),
            Entity(id="e2", name="Bob", description=LONG_DESCRIPTION),
        ]
        out = summarizer.summarize_entities(ents)
        # Only the long one was summarized; short one untouched.
        assert out[0].description == "short"
        assert out[1].description == "summary"
        # Exactly one input item was prepared for the LLM.
        sent_inputs = [call.args[0] for call in invoke.call_args_list]
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
        summarizer.summarizer.invoke = mocker.Mock(return_value="rel summary")
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


# --------------------------------------------------------------------------- #
# Map-reduce over the input budget
# --------------------------------------------------------------------------- #
class TestMapReduce:
    @staticmethod
    def _huge_description() -> str:
        # ~3x the per-call input budget, one fragment per line as merging
        # writes them.
        return "\n".join(
            f"Vendor supplies widgets under purchase order {i} for the Buyer."
            for i in range(3 * ds_module.MAX_INPUT_TOKENS // 10)
        )

    def test_over_budget_description_is_summarized_in_groups(
        self, summarizer, mocker
    ) -> None:
        invoke = mocker.Mock(side_effect=lambda inp: f"summary of {inp['entity_name']}")
        summarizer.summarizer.invoke = invoke
        description = self._huge_description()
        assert (
            summarizer.token_counter.count_tokens(description)
            > 2 * ds_module.MAX_INPUT_TOKENS
        )
        ents = [Entity(id="e1", name="Vendor", description=description)]

        out = summarizer.summarize_entities(ents)

        sent = [call.args[0]["descriptions"] for call in invoke.call_args_list]
        # Several group calls plus the final one, each within the budget
        # (counted line by line, as packing counts).
        assert len(sent) >= 4
        count = summarizer.token_counter.count_tokens
        assert all(
            sum(count(line) for line in text.split("\n")) <= ds_module.MAX_INPUT_TOKENS
            for text in sent
        )
        # The groups (summarized concurrently) cover every line once.
        group_lines = [line for text in sent[:-1] for line in text.split("\n")]
        assert sorted(group_lines) == sorted(description.split("\n"))
        assert out[0].description == "summary of Vendor"

    def test_within_budget_description_is_one_unchanged_call(
        self, summarizer, mocker
    ) -> None:
        invoke = mocker.Mock(return_value="summary")
        summarizer.summarizer.invoke = invoke
        summarizer.summarize_entities(
            [Entity(id="e1", name="Alice", description=LONG_DESCRIPTION)]
        )
        (call,) = invoke.call_args_list
        assert (
            call.args[0] == summarizer._build_inputs(["Alice"], [LONG_DESCRIPTION])[0]
        )

    def test_failed_group_keeps_the_concatenation(self, summarizer, mocker) -> None:
        def run(items_to_process, **kwargs):
            # The first group's call fails every attempt.
            return [
                BATCH_ITEM_FAILED if "order 1 " in item["descriptions"] else "summary"
                for item in items_to_process
            ]

        summarizer.batch_processor = mocker.Mock()
        summarizer.batch_processor.execute_with_fallback.side_effect = run
        description = self._huge_description()
        ents = [
            Entity(id="e1", name="Vendor", description=description),
            Entity(id="e2", name="Alice", description=LONG_DESCRIPTION),
        ]
        out = summarizer.summarize_entities(ents)
        assert out[0].description == description
        assert out[1].description == "summary"


class TestInputBudget:
    @pytest.mark.parametrize(
        ("window", "budget"),
        [
            (1_000_000, ds_module.MAX_INPUT_TOKENS),
            # Half of what an 8K window leaves beside the 256-token summary.
            (8_192, (8_192 - 256) // 2),
            (1_500, 1_000),
        ],
    )
    def test_budget_follows_the_summary_model_window(
        self, config: Config, mocker, window: int, budget: int
    ) -> None:
        # A 12K-token call overflows a small local model's window.
        factory = mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
        factory.return_value.get_model_info.return_value = LanguageModelInfo(
            context_window_size=window, max_output_tokens=1_024
        )
        mocker.patch.object(ds_module, "setup_chain")

        summarizer = DescriptionSummarizer(config)

        assert summarizer.max_input_tokens == budget
        description = "\n".join(f"Vendor supplies part {i}." for i in range(4_000))
        count = summarizer.token_counter.count_tokens
        assert all(
            sum(count(line) for line in group.split("\n")) <= budget
            for group in summarizer._pack(description)
        )
