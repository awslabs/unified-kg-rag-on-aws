# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every LLM chain must honour ``custom_prompts`` overrides (AWS-free).

Regression: the community-report, DRIFT and conversation-memory chains were
built without ``custom_prompts``, so their documented overrides (e.g.
``community_report_system``) were silently ignored.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from unified_kg_rag.adapters.aws import chain_factory
from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit


class _OfflineFactory(BedrockLanguageModelFactory):
    """Bedrock factory without a boto client, returning a canned chat model."""

    def __init__(self, config: Config | None = None, **_: Any) -> None:
        self.config = config or Config()

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return FakeListChatModel(responses=["{}"])

    def get_model_info(self, model_id: Any) -> Any:
        return None


_INGESTION = "unified_kg_rag.adapters.ingestion"
_STRATEGIES = "unified_kg_rag.adapters.search_strategies"
# (module, class, extra constructor kwargs) for every component that builds
# chains with setup_chain.
_COMPONENTS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "chunker": (f"{_INGESTION}.chunker", "IntelligentTextChunker", {}),
    "translator": (f"{_INGESTION}.translator", "TextUnitTranslator", {}),
    "graph_extractor": (f"{_INGESTION}.graph_extractor", "GraphExtractor", {}),
    "gleaner": (f"{_INGESTION}.gleaner", "GraphGleaner", {}),
    "claim_extractor": (f"{_INGESTION}.claim_extractor", "ClaimExtractor", {}),
    "description_summarizer": (
        f"{_INGESTION}.description_summarizer",
        "DescriptionSummarizer",
        {},
    ),
    "community_detector": (
        f"{_INGESTION}.community_detector",
        "CommunityDetector",
        {},
    ),
    "prompt_tuner": ("unified_kg_rag.application.prompts.tuner", "PromptTuner", {}),
    "memory": (
        "unified_kg_rag.adapters.retrieval.memory_manager",
        "GraphRAGChatMessageHistory",
        {"conversation_id": "c-1"},
    ),
    "drift": (f"{_STRATEGIES}.drift_search", "DriftSearchStrategy", {"retrievers": {}}),
    "global": (
        f"{_STRATEGIES}.global_search",
        "GlobalSearchStrategy",
        {"retrievers": {}},
    ),
}


def _config() -> Config:
    cfg = Config()
    cfg.search.drift_search.enable_primer = True  # build every DRIFT chain
    cfg.search.reranking.enabled = False
    return cfg


@pytest.mark.parametrize("name", sorted(_COMPONENTS))
def test_every_chain_receives_custom_prompts(name: str, mocker) -> None:
    from unified_kg_rag.application.prompts.tuner import PromptTuner

    module_name, cls_name, extra = _COMPONENTS[name]
    module = importlib.import_module(module_name)
    cfg = _config()
    spy = mocker.patch.object(module, "setup_chain", wraps=chain_factory.setup_chain)
    # The offline factory arrives through the shared provider bundle.
    providers = Providers(
        cfg, boto_session=mocker.MagicMock(), llm_factory=_OfflineFactory(cfg)
    )
    component = getattr(module, cls_name)(config=cfg, providers=providers, **extra)
    if isinstance(component, PromptTuner):  # builds its chain lazily
        asyncio.run(component.profile_corpus(["Vendor ships parts to Buyer."]))

    assert spy.call_count, f"{name} built no chain"
    for call in spy.call_args_list:
        prompt = call.kwargs["prompt_class"].__name__
        assert call.kwargs.get("custom_prompts") is cfg.custom_prompts, prompt


def test_community_report_override_reaches_the_chain(mocker) -> None:
    from unified_kg_rag.adapters.ingestion import community_detector

    cfg = _config()
    cfg.custom_prompts.community_report_system = "Synthetic report instructions."
    detector = community_detector.CommunityDetector(
        config=cfg,
        providers=Providers(
            cfg, boto_session=mocker.MagicMock(), llm_factory=_OfflineFactory(cfg)
        ),
    )
    prompt = detector.report_generator.first
    rendered = prompt.format_messages(**dict.fromkeys(prompt.input_variables, "x"))
    assert rendered[0].content == "Synthetic report instructions."
