# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for automatic prompt tuning (M4)."""

from __future__ import annotations

import pytest

from aws_graphrag.domain.models import Config
from aws_graphrag.domain.prompts.tuner import CorpusProfile, PromptTuner

pytestmark = pytest.mark.unit


class TestCorpusProfile:
    def test_from_payload_uppercases_entity_types(self) -> None:
        p = CorpusProfile.from_payload(
            {"domain": "law", "entity_types": ["statute", "court", ""]}
        )
        assert p.domain == "law"
        assert p.entity_types == ["STATUTE", "COURT"]

    def test_from_payload_defaults(self) -> None:
        p = CorpusProfile.from_payload({})
        assert p.domain == "general knowledge"
        assert p.language == "English"
        assert p.entity_types == []


class TestBuildCustomPrompts:
    def test_includes_domain_and_entity_types(self) -> None:
        profile = CorpusProfile(
            domain="clinical oncology",
            language="English",
            persona="You are a medical expert.",
            entity_types=["DRUG", "GENE"],
        )
        cp = PromptTuner.build_custom_prompts(profile)
        system = cp["graph_extraction_system"]
        assert "medical expert" in system
        assert "clinical oncology" in system
        assert "DRUG, GENE" in system

    def test_no_entity_types_omits_guidance(self) -> None:
        profile = CorpusProfile(entity_types=[])
        system = PromptTuner.build_custom_prompts(profile)["graph_extraction_system"]
        assert "Focus on these domain entity types" not in system

    def test_adapts_community_report_persona(self) -> None:
        profile = CorpusProfile(
            domain="clinical oncology", persona="You are a medical expert."
        )
        cp = PromptTuner.build_custom_prompts(profile)
        assert "medical expert" in cp["community_report_system"]
        assert "clinical oncology" in cp["community_report_system"]

    def test_few_shot_examples_embedded_when_present(self) -> None:
        profile = CorpusProfile(few_shot_examples="EXAMPLE TEXT: ...")
        system = PromptTuner.build_custom_prompts(profile)["graph_extraction_system"]
        assert "DOMAIN EXAMPLE" in system
        assert "EXAMPLE TEXT: ..." in system

    def test_no_few_shot_examples_omits_block(self) -> None:
        profile = CorpusProfile(few_shot_examples="")
        system = PromptTuner.build_custom_prompts(profile)["graph_extraction_system"]
        assert "DOMAIN EXAMPLE" not in system


class TestSampleAndParse:
    @pytest.fixture
    def tuner(self, config: Config, mocker) -> PromptTuner:
        mocker.patch("aws_graphrag.domain.prompts.tuner.BedrockLanguageModelFactory")
        return PromptTuner(config)

    def test_sample_respects_budget(self, tuner: PromptTuner) -> None:
        tuner.MAX_SAMPLE_CHARS = 10
        sample = tuner.sample_corpus(["aaaaa", "bbbbb", "ccccc"])
        # Total kept characters cannot exceed the budget.
        assert len(sample.replace("\n\n---\n\n", "")) <= 10

    def test_sample_skips_empty(self, tuner: PromptTuner) -> None:
        assert tuner.sample_corpus(["", ""]) == ""

    def test_parse_json_strips_prose(self, tuner: PromptTuner) -> None:
        assert PromptTuner._parse_json('result: {"domain": "x"} ok') == {"domain": "x"}

    def test_parse_json_no_braces_returns_empty(self, tuner: PromptTuner) -> None:
        assert PromptTuner._parse_json("I cannot help with that") == {}

    def test_parse_json_malformed_returns_empty(self, tuner: PromptTuner) -> None:
        assert PromptTuner._parse_json("{not: valid json}") == {}

    def test_parse_json_non_dict_returns_empty(self, tuner: PromptTuner) -> None:
        assert PromptTuner._parse_json("[1, 2, 3]") == {}

    async def test_profile_corpus_empty_returns_default(
        self, tuner: PromptTuner
    ) -> None:
        profile = await tuner.profile_corpus([])
        assert profile.domain == "general knowledge"

    async def test_tune_returns_profile_and_custom_prompts(
        self, tuner: PromptTuner, mocker
    ) -> None:
        async def _fake_profile(_texts):
            return CorpusProfile(domain="finance", entity_types=["TICKER"])

        mocker.patch.object(tuner, "profile_corpus", side_effect=_fake_profile)

        async def _fake_examples(_profile, _texts):
            return "EXAMPLE TEXT: a trade settled."

        mocker.patch.object(tuner, "generate_examples", side_effect=_fake_examples)
        result = await tuner.tune(["some financial text"])
        assert result["profile"]["domain"] == "finance"
        assert "graph_extraction_system" in result["custom_prompts"]
        assert "DOMAIN EXAMPLE" in result["custom_prompts"]["graph_extraction_system"]

    async def test_generate_examples_empty_corpus_returns_empty(
        self, tuner: PromptTuner
    ) -> None:
        assert await tuner.generate_examples(CorpusProfile(), []) == ""

    async def test_generate_examples_degrades_on_chain_error(
        self, tuner: PromptTuner, mocker
    ) -> None:
        # The documented except branch: a failing chain returns "" rather than
        # propagating, so tuning still completes without a few-shot example.
        chain = mocker.MagicMock()
        chain.ainvoke = mocker.AsyncMock(side_effect=RuntimeError("bedrock down"))
        mocker.patch(
            "aws_graphrag.domain.prompts.tuner.setup_chain", return_value=chain
        )
        result = await tuner.generate_examples(
            CorpusProfile(domain="x"), ["non-empty text"]
        )
        assert result == ""

    async def test_profile_corpus_degrades_on_unparseable_output(
        self, tuner: PromptTuner, mocker
    ) -> None:
        chain = mocker.MagicMock()
        chain.ainvoke = mocker.AsyncMock(return_value="sorry, no JSON here")
        mocker.patch(
            "aws_graphrag.domain.prompts.tuner.setup_chain", return_value=chain
        )
        profile = await tuner.profile_corpus(["some text"])
        assert profile.domain == "general knowledge"
