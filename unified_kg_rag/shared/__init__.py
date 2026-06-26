# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

from .config import ConfigLoader, get_config
from .exceptions import (
    AWSServiceError,
    DataProcessingError,
    EmbeddingModelError,
    EvaluationException,
    GraphError,
    GraphRAGException,
    LanguageModelError,
    PipelineExecutionError,
    PipelineResumeError,
    PipelineStageError,
    PipelineStateError,
    RerankModelError,
)
from .logging import get_logger
from .metrics import CloudWatchEMFSink, MetricsSink, NullMetricsSink
from .pipeline_manager import PipelineResumeManager, PipelineStateManager

if TYPE_CHECKING:
    from .cache_manager import CacheManager


def get_cache_manager() -> "type[CacheManager]":
    from .cache_manager import CacheManager

    return CacheManager


__all__ = [
    "AWSServiceError",
    "CloudWatchEMFSink",
    "ConfigLoader",
    "DataProcessingError",
    "EmbeddingModelError",
    "EvaluationException",
    "GraphError",
    "GraphRAGException",
    "LanguageModelError",
    "MetricsSink",
    "NullMetricsSink",
    "PipelineExecutionError",
    "PipelineResumeError",
    "PipelineResumeManager",
    "PipelineStageError",
    "PipelineStateError",
    "PipelineStateManager",
    "RerankModelError",
    "get_cache_manager",
    "get_config",
    "get_logger",
]
