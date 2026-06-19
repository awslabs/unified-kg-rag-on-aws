# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for automatic prompt tuning (M4)."""
from __future__ import annotations

import pytest

from aws_graphrag.models import Config
from aws_graphrag.prompts.tuner import CorpusProfile, PromptTuner

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


class TestSampleAndParse:
    @pytest.fixture
    def tuner(self, config: Config, mocker) -> PromptTuner:
        mocker.patch("aws_graphrag.prompts.tuner.BedrockLanguageModelFactory")
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
        result = await tuner.tune(["some financial text"])
        assert result["profile"]["domain"] == "finance"
        assert "graph_extraction_system" in result["custom_prompts"]
