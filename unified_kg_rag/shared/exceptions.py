# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
class GraphRAGException(Exception):
    pass


class AWSServiceError(GraphRAGException):
    pass


class CacheSyncError(AWSServiceError):
    """A pipeline cache sync with remote storage (S3) failed or was partial.

    Raised instead of returning a partial result so a phased run (one Step
    Functions task per phase) exits non-zero at the phase that lost its
    checkpoint, not later with a misleading "missing stage" error.
    """


class DataProcessingError(GraphRAGException):
    pass


class EvaluationException(GraphRAGException):
    pass


class GraphError(GraphRAGException):
    pass


class ModelError(GraphRAGException):
    pass


class EmbeddingModelError(ModelError):
    pass


class InvalidFilterError(GraphRAGException, ValueError):
    """A caller search filter names a key no target index or label declares."""


class LanguageModelError(ModelError):
    pass


class LLMOutputTruncatedError(LanguageModelError):
    """A model response stopped at its output-token limit, cut mid-answer."""


class PipelineExecutionError(GraphRAGException):
    pass


class PipelineResumeError(GraphRAGException):
    pass


class PipelineStageError(PipelineExecutionError):
    pass


class PipelineStateError(GraphRAGException):
    pass


class RerankModelError(ModelError):
    pass
