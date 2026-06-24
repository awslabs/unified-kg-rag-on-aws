# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
class GraphRAGException(Exception):
    pass


class AWSServiceError(GraphRAGException):
    pass


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


class LanguageModelError(ModelError):
    pass


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
