# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default output cap on LLM requests (AWS-free).

Bedrock reserves input + max_tokens against the tokens-per-minute quota when a
request starts, so a request must not ask for the model maximum by default.
The cap is ``models.default_max_output_tokens``, raised to the prompt's
``min_output_tokens`` floor and clamped to the model maximum.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.output_parsers import StrOutputParser

from tests.fixtures.fakes.boto import FakeBotoSession
from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.bedrock_models import get_language_model_info
from unified_kg_rag.adapters.llm.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.domain.models import (
    Config,
    LanguageModelId,
    ModelPurpose,
    ModelRole,
)
from unified_kg_rag.domain.models.config import EFFORT_THINKING_TOKENS
from unified_kg_rag.domain.prompts import (
    AnswerGenerationPrompt,
    GraphExtractionPrompt,
)
from unified_kg_rag.domain.prompts.data_processing import TextTranslationPrompt
from unified_kg_rag.domain.prompts.graph_extraction import (
    ClaimExtractionPrompt,
    CommunityReportPrompt,
    GraphRefinementPrompt,
)
from unified_kg_rag.ports.model_factory import effective_max_output_tokens

pytestmark = pytest.mark.unit

SONNET = LanguageModelId.CLAUDE_V5_5_SONNET
HAIKU = LanguageModelId.CLAUDE_V4_5_HAIKU


def _factory(config: Config | None = None) -> BedrockLanguageModelFactory:
    return BedrockLanguageModelFactory(
        config or Config(), boto_session=FakeBotoSession()  # type: ignore[arg-type]
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
    config = Config(models={"default_max_output_tokens": None})
    info = get_language_model_info(SONNET)
    assert (
        effective_max_output_tokens(info, config.models.default_max_output_tokens)
        == 128000
    )


def test_configured_cap_above_the_floor_wins() -> None:
    config = Config(models={"default_max_output_tokens": 50000})
    info = get_language_model_info(SONNET)
    assert (
        effective_max_output_tokens(
            info, config.models.default_max_output_tokens, 32768
        )
        == 50000
    )


def test_short_output_prompts_take_the_default_cap() -> None:
    assert AnswerGenerationPrompt.min_output_tokens == 0


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_model(self, model_id: str, **kwargs: Any) -> Any:
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        self.calls.append({**kwargs, "model_id": model_id})
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
    # (50 + 50) records x 150 tokens + the default tier's high-effort budget.
    assert GraphExtractionPrompt.output_floor(config) == 100 * 150 + 8192
    assert GraphRefinementPrompt.output_floor(config) == 100 * 150 + 8192
    # A whole 2000-token chunk, at most 8003 characters, at 1.35 tokens/char
    # (twice for claims); translation runs on the fast tier, at low effort.
    assert TextTranslationPrompt.output_floor(config) == int(8003 * 1.35) + 2048
    assert ClaimExtractionPrompt.output_floor(config) == int(2 * 8003 * 1.35) + 8192
    # medium report: 10 findings x 300 + 200 header.
    assert CommunityReportPrompt.output_floor(config) == 3200 + 8192


def test_floors_scale_with_the_configured_limits() -> None:
    config = Config(
        processing={
            "chunking": {"length_unit": "chars", "max_chunk_size": 20000},
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
    assert TextTranslationPrompt.output_floor(config) == 27000 + 2048
    assert CommunityReportPrompt.output_floor(config) == 4700 + 8192


_FLOOR_PROMPTS = (
    GraphExtractionPrompt,
    GraphRefinementPrompt,
    ClaimExtractionPrompt,
    CommunityReportPrompt,
    TextTranslationPrompt,
)


def _thinking(config: Config, role: ModelRole) -> int:
    tier = config.model_tier(config.model_id(role))
    return EFFORT_THINKING_TOKENS[
        config.aws.bedrock.tier_effort(tier, ModelPurpose.INGESTION)
    ]


@pytest.mark.parametrize("prompt", _FLOOR_PROMPTS)
@pytest.mark.parametrize(
    "bedrock",
    [
        {},
        {"ingestion_effort": "low"},
        {"default_effort": "max", "fast_effort": "medium"},
        {"ingestion_effort": "xhigh", "fast_effort": "high"},
    ],
)
def test_every_floor_leaves_its_models_thinking_budget(
    prompt: Any, bedrock: dict[str, str]
) -> None:
    # The floor adds the budget of the effort the call runs at: the role's
    # tier's ingestion effort, not a fixed high-effort 8192.
    config = Config(aws={"bedrock": bedrock})
    assert prompt.model_role is not None
    base = Config()
    answer = prompt.output_floor(base) - _thinking(base, prompt.model_role)
    assert answer > 0
    assert prompt.output_floor(config) == answer + _thinking(config, prompt.model_role)
    # The output fixer re-emits the answer on its own model.
    assert prompt.output_floor(config, ModelRole.FIXING) == answer + _thinking(
        config, ModelRole.FIXING
    )


def test_a_role_pinned_to_the_fast_model_gets_the_fast_effort() -> None:
    config = Config(
        models={"roles": {"graph_extraction": "fast"}},
        aws={"bedrock": {"default_effort": "high", "fast_effort": "low"}},
    )
    assert GraphExtractionPrompt.output_floor(config) == 100 * 150 + 2048
    assert GraphExtractionPrompt.output_floor(config, ModelRole.FIXING) == (
        100 * 150 + 8192
    )


def _build(module: str, config: Config, factory: _RecordingFactory) -> None:
    from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractor
    from unified_kg_rag.adapters.ingestion.community_detector import (
        CommunityDetector,
    )
    from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
    from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
    from unified_kg_rag.adapters.ingestion.translator import TextUnitTranslator
    from unified_kg_rag.adapters.providers import Providers

    providers = Providers(config, boto_session=FakeBotoSession(), llm_factory=factory)  # type: ignore[arg-type]
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
@pytest.mark.parametrize("fixing", [None, "fast"])
def test_ingestion_chains_request_the_derived_floor(
    module: str, prompt: Any, fixing: str | None
) -> None:
    config = Config(fixing={"enabled": True}, models={"roles": {"fixing": fixing}})
    factory = _RecordingFactory()
    _build(module, config, factory)
    # The chain and its output fixer both reserve the derived floor, each
    # with its own model's thinking budget.
    fixer = config.model_id(ModelRole.FIXING)
    for call in factory.calls:
        role = ModelRole.FIXING if call["model_id"] == fixer else None
        assert call["min_output_tokens"] == prompt.output_floor(config, role)
