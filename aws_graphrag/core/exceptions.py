# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
class GraphRAGException(Exception):
    pass


class AWSServiceError(GraphRAGException):
    pass


class ConfigurationError(GraphRAGException):
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


class RetrievalError(GraphRAGException):
    pass


class StorageError(GraphRAGException):
    pass
