# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An injected provider bundle reaches every ingestion component (AWS-free).

The ingestion adapters (chunker, translator, graph/claim extraction, gleaning,
description summarization, community reports) and the indexer each built
their own Bedrock factory. The pipeline now builds one ``Providers`` bundle (or
takes one) and passes it to every stage, so with an injected bundle no
Bedrock factory is constructed at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from unified_kg_rag.adapters import providers as providers_module
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import Config, PipelineStageType
from unified_kg_rag.domain.models.config import PipelineConfig

pytestmark = pytest.mark.unit


class _FakeEmbeddingFactory:
    """Structural EmbeddingFactoryPort with a fixed 8-dimension model."""

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return MagicMock()

    def get_model_info(self, model_id: Any) -> Any:
        return MagicMock(dimensions=8)


class _FakeLLMFactory:
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return FakeListChatModel(responses=["ok"])

    def get_model_info(self, model_id: Any) -> Any:
        return None


@pytest.fixture
def no_bedrock(mocker) -> None:
    for name in (
        "BedrockLanguageModelFactory",
        "BedrockEmbeddingModelFactory",
        "BedrockRerankModelFactory",
    ):
        mocker.patch.object(
            providers_module,
            name,
            side_effect=AssertionError(f"{name} must not be constructed"),
        )


def test_injected_providers_reach_every_ingestion_stage(
    tmp_path: Path, no_bedrock
) -> None:
    config = Config()
    config.processing.gleaning.enabled = True
    config.processing.claim_extraction.enabled = True
    config.aws.dynamodb.enabled = False
    llm = _FakeLLMFactory()
    embedding = _FakeEmbeddingFactory()
    providers = Providers(
        config, boto_session=MagicMock(), llm_factory=llm, embedding_factory=embedding
    )

    pipeline = DataIngestionPipeline(
        config=config,
        pipeline_config=PipelineConfig(local_directory=str(tmp_path)),
        source_directory=tmp_path,
        providers=providers,
    )
    stages = {pipeline.name_to_type_map[s.name]: s for s in pipeline.stages}

    assert pipeline.providers is providers
    for stage_type in DataIngestionPipeline.PROVIDER_STAGES:
        assert stages[stage_type].providers is providers, stage_type

    T = PipelineStageType
    assert stages[T.TEXT_CHUNKING].chunker.factory is llm
    assert stages[T.GRAPH_EXTRACTION].extractor.factory is llm
    assert stages[T.GLEANING].gleaner.factory is llm
    assert stages[T.CLAIM_EXTRACTION].extractor.factory is llm
    assert stages[T.GRAPH_RESOLUTION].description_summarizer.factory is llm
    assert stages[T.COMMUNITY_DETECTION].detector.factory is llm
    indexing = stages[T.INDEXING].indexing_manager
    assert indexing.opensearch_indexer.embedding_factory is embedding
    assert indexing.description_summarizer.factory is llm


def test_translator_and_visualization_use_the_bundle(no_bedrock) -> None:
    from unified_kg_rag.adapters.ingestion.translator import TextUnitTranslator
    from unified_kg_rag.visualization.base import GraphVisualizationManager

    config = Config()
    llm, embedding = _FakeLLMFactory(), MagicMock()
    providers = Providers(
        config, boto_session=MagicMock(), llm_factory=llm, embedding_factory=embedding
    )

    translator = TextUnitTranslator(config, providers=providers)
    viz = GraphVisualizationManager(
        config=config,
        graph_analyzer=MagicMock(),
        community_detector=MagicMock(),
        providers=providers,
    )

    assert translator.factory is llm
    assert viz.providers is providers
    assert viz.boto_session is providers.boto_session


def test_pipeline_without_providers_builds_one_default_bundle(
    tmp_path: Path, mocker
) -> None:
    llm_cls = mocker.patch.object(
        providers_module, "BedrockLanguageModelFactory", return_value=_FakeLLMFactory()
    )
    mocker.patch.object(
        providers_module,
        "BedrockEmbeddingModelFactory",
        return_value=_FakeEmbeddingFactory(),
    )
    config = Config()
    config.aws.dynamodb.enabled = False

    pipeline = DataIngestionPipeline(
        config=config,
        pipeline_config=PipelineConfig(local_directory=str(tmp_path)),
        source_directory=tmp_path,
        boto_session=MagicMock(),
    )

    # Every LLM-using stage shares one default factory: built exactly once.
    llm_cls.assert_called_once()
    assert all(
        stage.providers is pipeline.providers
        for stage in pipeline.stages
        if pipeline.name_to_type_map[stage.name] in pipeline.PROVIDER_STAGES
    )
