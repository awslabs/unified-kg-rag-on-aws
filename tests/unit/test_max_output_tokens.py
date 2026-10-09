# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default output cap on LLM requests (AWS-free).

Bedrock reserves input + max_tokens against the tokens-per-minute quota when a
request starts, so a request must not ask for the model maximum by default.
The cap is ``aws.bedrock.default_max_output_tokens``, raised to the prompt's
``min_output_tokens`` floor and clamped to the model maximum.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.bedrock_models import (
    effective_max_output_tokens,
    get_language_model_info,
)
from unified_kg_rag.adapters.aws.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.domain.models import Config, LanguageModelId, ModelPurpose
from unified_kg_rag.domain.prompts import (
    AnswerGenerationPrompt,
    GraphExtractionPrompt,
)
from unified_kg_rag.domain.prompts.base import THINKING_HEADROOM_TOKENS
from unified_kg_rag.domain.prompts.data_processing import TextTranslationPrompt
from unified_kg_rag.domain.prompts.graph_extraction import (
    ClaimExtractionPrompt,
    CommunityReportPrompt,
    GraphRefinementPrompt,
)

pytestmark = pytest.mark.unit

SONNET = LanguageModelId.CLAUDE_V5_5_SONNET
HAIKU = LanguageModelId.CLAUDE_V4_5_HAIKU


class _FakeSession:
    profile_name = "default"

    def client(self, service_name: str, **kwargs: Any) -> Any:
        return object()

    def get_credentials(self) -> Any:
        return None


def _factory(config: Config | None = None) -> BedrockLanguageModelFactory:
    return BedrockLanguageModelFactory(
        config or Config(), boto_session=_FakeSession()  # type: ignore[arg-type]
    )


def _max_tokens(cfg: dict[str, Any]) -> int:
    return int(cfg.get("max_tokens") or cfg["model_kwargs"]["max_tokens"])


def test_default_cap_replaces_the_model_maximum() -> None:
    factory = _factory()
    info = factory.get_model_info(SONNET)
    cfg = factory._build_model_config(info, f"us.{SONNET.value}", True)
    assert info.max_output_tokens == 128000
    assert _max_tokens(cfg) == 16384


def test_cap_applies_on_the_invoke_model_path() -> None:
    factory = _factory()
    info = factory.get_model_info(LanguageModelId.CLAUDE_V3_5_SONNET_V2)
    cfg = factory._build_model_config(info, "anthropic.claude-x", False)
    # Claude 3.5 Sonnet v2 tops out below the cap, so it gets its maximum.
    assert cfg["model_kwargs"]["max_tokens"] == 8192


def test_prompt_floor_raises_the_cap_up_to_the_model_maximum() -> None:
    factory = _factory()
    sonnet = factory.get_model_info(SONNET)
    haiku = factory.get_model_info(HAIKU)
    floor = GraphExtractionPrompt.output_floor(Config())
    cfg = factory._build_model_config(
        sonnet, f"us.{SONNET.value}", True, min_output_tokens=floor
    )
    assert _max_tokens(cfg) == floor == 23192
    cfg = factory._build_model_config(
        haiku, f"us.{HAIKU.value}", True, min_output_tokens=100000
    )
    assert _max_tokens(cfg) == haiku.max_output_tokens == 64000


def test_explicit_max_tokens_wins() -> None:
    factory = _factory()
    info = factory.get_model_info(SONNET)
    cfg = factory._build_model_config(
        info, f"us.{SONNET.value}", True, max_tokens=500, min_output_tokens=32768
    )
    assert _max_tokens(cfg) == 500


def test_null_cap_restores_the_model_maximum() -> None:
    config = Config(aws={"bedrock": {"default_max_output_tokens": None}})
    info = get_language_model_info(SONNET)
    assert effective_max_output_tokens(info, config.aws.bedrock) == 128000


def test_configured_cap_above_the_floor_wins() -> None:
    config = Config(aws={"bedrock": {"default_max_output_tokens": 50000}})
    info = get_language_model_info(SONNET)
    assert effective_max_output_tokens(info, config.aws.bedrock, 32768) == 50000


def test_short_output_prompts_take_the_default_cap() -> None:
    assert AnswerGenerationPrompt.min_output_tokens == 0


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_model(self, model_id: str, **kwargs: Any) -> Any:
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        self.calls.append(kwargs)
        return FakeListChatModel(responses=["ok"])

    def get_model_info(self, model_id: str) -> Any:
        return None


def test_setup_chain_forwards_the_prompt_floor() -> None:
    factory = _RecordingFactory()
    setup_chain(factory, SONNET, GraphExtractionPrompt, StrOutputParser())
    assert factory.calls[0]["min_output_tokens"] == 0
    setup_chain(
        factory,
        SONNET,
        GraphExtractionPrompt,
        StrOutputParser(),
        min_output_tokens=23192,
    )
    assert factory.calls[1]["min_output_tokens"] == 23192


def test_output_fixer_gets_the_repaired_prompts_floor() -> None:
    factory = _RecordingFactory()
    create_robust_xml_output_parser(
        factory=factory,
        enable_output_fixing=True,
        output_fixing_model_id=SONNET,
        output_tags=["entities", "relationships"],
        model_purpose=ModelPurpose.INGESTION,
        min_output_tokens=GraphExtractionPrompt.output_floor(Config()),
    )
    assert factory.calls[0]["min_output_tokens"] == 23192


def test_default_floors_follow_the_documented_arithmetic() -> None:
    config = Config()
    # (50 + 50) records x 150 tokens + thinking headroom.
    assert GraphExtractionPrompt.output_floor(config) == 100 * 150 + 8192
    assert GraphRefinementPrompt.output_floor(config) == 100 * 150 + 8192
    # A whole 8000-char chunk at 1.35 tokens/char (twice for claims).
    assert TextTranslationPrompt.output_floor(config) == 10800 + 8192
    assert ClaimExtractionPrompt.output_floor(config) == 21600 + 8192
    # medium report: 10 findings x 300 + 200 header.
    assert CommunityReportPrompt.output_floor(config) == 3200 + 8192


def test_floors_scale_with_the_configured_limits() -> None:
    config = Config(
        processing={
            "chunking": {"max_chunk_size": 20000},
            "graph_extraction": {
                "max_entities_per_chunk": 200,
                "max_relationships_per_chunk": 200,
            },
        },
        graph={
            "community_detection": {"report_generation": {"content_length": "long"}}
        },
    )
    assert GraphExtractionPrompt.output_floor(config) == 400 * 150 + 8192
    assert TextTranslationPrompt.output_floor(config) == 27000 + 8192
    assert CommunityReportPrompt.output_floor(config) == 4700 + 8192


def test_every_floor_leaves_thinking_headroom() -> None:
    config = Config()
    for prompt in (
        GraphExtractionPrompt,
        GraphRefinementPrompt,
        ClaimExtractionPrompt,
        CommunityReportPrompt,
        TextTranslationPrompt,
    ):
        assert prompt.output_floor(config) > THINKING_HEADROOM_TOKENS


def _build(module: str, config: Config, factory: _RecordingFactory) -> None:
    from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractor
    from unified_kg_rag.adapters.ingestion.community_detector import (
        CommunityDetector,
    )
    from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
    from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
    from unified_kg_rag.adapters.ingestion.translator import TextUnitTranslator
    from unified_kg_rag.adapters.providers import Providers

    providers = Providers(config, boto_session=_FakeSession(), llm_factory=factory)  # type: ignore[arg-type]
    builders = {
        "graph_extractor": lambda: GraphExtractor(config, providers=providers),
        "gleaner": lambda: GraphGleaner(config, providers=providers),
        "claim_extractor": lambda: ClaimExtractor(config, providers=providers),
        "community_detector": lambda: CommunityDetector(config, providers=providers),
        "translator": lambda: TextUnitTranslator(config, providers=providers),
    }
    builders[module]()


@pytest.mark.parametrize(
    ("module", "prompt"),
    [
        ("graph_extractor", GraphExtractionPrompt),
        ("gleaner", GraphRefinementPrompt),
        ("claim_extractor", ClaimExtractionPrompt),
        ("community_detector", CommunityReportPrompt),
        ("translator", TextTranslationPrompt),
    ],
)
def test_ingestion_chains_request_the_derived_floor(module: str, prompt: Any) -> None:
    config = Config(fixing={"enabled": True})
    factory = _RecordingFactory()
    _build(module, config, factory)
    floors = {call["min_output_tokens"] for call in factory.calls}
    # The chain and its output fixer both reserve the derived floor.
    assert floors == {prompt.output_floor(config)}
