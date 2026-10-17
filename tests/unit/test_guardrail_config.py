# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Bedrock Guardrails wiring (WAF security pillar, AWS-free)."""

from __future__ import annotations

import asyncio
import importlib

import pytest

from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.bedrock_retry import is_transient_bedrock_error
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, ModelPurpose

pytestmark = pytest.mark.unit


def test_enabled_property() -> None:
    cfg = Config()
    assert cfg.aws.bedrock.guardrail.enabled is False
    cfg.aws.bedrock.guardrail.identifier = "gr-x"
    assert cfg.aws.bedrock.guardrail.enabled is True


def test_guardrail_identifier_from_env(monkeypatch) -> None:
    # IaC injects the deployed guardrail id via this env var (4-level nested
    # config path); verify the override lands and enables guardrails.
    monkeypatch.setenv("UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER", "gr-from-env")
    from unified_kg_rag.shared.config import ConfigLoader

    cfg = ConfigLoader().load_config()
    assert cfg.aws.bedrock.guardrail.identifier == "gr-from-env"
    assert cfg.aws.bedrock.guardrail.enabled is True


# --- Call sites pass the right purpose ---------------------------------------


class _RecordingFactory(BedrockLanguageModelFactory):
    """Bedrock factory without a boto client that records every get_model call."""

    def __init__(self, config: Config | None = None, **_: object) -> None:
        self.config = config or Config()
        self.calls: list[dict] = []

    def get_model(self, model_id, **kwargs):  # noqa: ANN001, ANN003
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        self.calls.append(kwargs)
        return FakeListChatModel(responses=["{}"])

    def get_model_info(self, model_id):  # noqa: ANN001
        return None


def test_setup_chain_forwards_purpose_and_defaults_to_query() -> None:
    from langchain_core.output_parsers import StrOutputParser

    from unified_kg_rag.adapters.llm.chain_factory import setup_chain
    from unified_kg_rag.domain.models import LanguageModelId
    from unified_kg_rag.domain.prompts import TextTranslationPrompt

    factory = _RecordingFactory()
    common = {
        "factory": factory,
        "model_id": LanguageModelId.CLAUDE_V4_5_HAIKU,
        "prompt_class": TextTranslationPrompt,
        "parser": StrOutputParser(),
    }
    setup_chain(**common)
    setup_chain(**common, model_purpose=ModelPurpose.INGESTION)
    assert [c["model_purpose"] for c in factory.calls] == [
        ModelPurpose.QUERY,
        ModelPurpose.INGESTION,
    ]


def test_output_fixing_llm_inherits_purpose() -> None:
    from unified_kg_rag.adapters.llm.chain_factory import (
        create_robust_xml_output_parser,
    )
    from unified_kg_rag.domain.models import LanguageModelId

    factory = _RecordingFactory()
    create_robust_xml_output_parser(
        factory=factory,
        enable_output_fixing=True,
        output_fixing_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        output_tags=["entities", "relationships"],
        model_purpose=ModelPurpose.INGESTION,
    )
    assert [c["model_purpose"] for c in factory.calls] == [ModelPurpose.INGESTION]


_INGESTION = "unified_kg_rag.adapters.ingestion"
# Every component whose LLM chains process corpus text. Each must request
# INGESTION models so the query retry (which would stack on BatchProcessor's)
# is not attached.
_INGESTION_COMPONENTS = {
    "chunker": (f"{_INGESTION}.chunker", "IntelligentTextChunker"),
    "translator": (f"{_INGESTION}.translator", "TextUnitTranslator"),
    "graph_extractor": (f"{_INGESTION}.graph_extractor", "GraphExtractor"),
    "gleaner": (f"{_INGESTION}.gleaner", "GraphGleaner"),
    "claim_extractor": (f"{_INGESTION}.claim_extractor", "ClaimExtractor"),
    "description_summarizer": (
        f"{_INGESTION}.description_summarizer",
        "DescriptionSummarizer",
    ),
    "community_detector": (f"{_INGESTION}.community_detector", "CommunityDetector"),
    "prompt_tuner": ("unified_kg_rag.application.prompts.tuner", "PromptTuner"),
}


@pytest.mark.parametrize("name", sorted(_INGESTION_COMPONENTS))
def test_ingestion_components_request_ingestion_models(name: str, mocker) -> None:
    from unified_kg_rag.application.prompts.tuner import PromptTuner

    module, cls = _INGESTION_COMPONENTS[name]
    cfg = Config()
    cfg.fixing.enabled = True  # also cover the output-fixing LLMs
    factory = _RecordingFactory(cfg)
    providers = Providers(cfg, boto_session=mocker.MagicMock(), llm_factory=factory)
    component = getattr(importlib.import_module(module), cls)(
        config=cfg, providers=providers
    )
    if isinstance(component, PromptTuner):  # builds its chain lazily
        asyncio.run(component.profile_corpus(["Vendor ships parts to Buyer."]))
    assert factory.calls, f"{name} built no LLM chain"
    assert all(c["model_purpose"] is ModelPurpose.INGESTION for c in factory.calls)
    # The ingestion-side retry must fail fast on permanent Bedrock errors.
    batch_processor = getattr(component, "batch_processor", None)
    if batch_processor is not None:
        assert batch_processor.is_transient_error is is_transient_bedrock_error
