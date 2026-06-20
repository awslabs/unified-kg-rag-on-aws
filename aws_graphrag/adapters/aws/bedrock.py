# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Generic, Literal, TypeVar

import boto3
from aws_assume_role_lib.aws_assume_role_lib import assume_role
from botocore.config import Config as BotoConfig
from langchain_aws import BedrockEmbeddings, ChatBedrock, ChatBedrockConverse
from langchain_aws.document_compressors.rerank import BedrockRerank
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.documents import Document
from pydantic import BaseModel, Field, PrivateAttr

from aws_graphrag.adapters.aws.token_counter import BedrockTokenCounter
from aws_graphrag.core import (
    AWSServiceError,
    EmbeddingModelError,
    LanguageModelError,
    RerankModelError,
    get_logger,
)
from aws_graphrag.models import Config, EmbeddingModelId, LanguageModelId, RerankModelId

logger = get_logger(__name__)


DEFAULT_ROLE_SESSION_NAME: str = "aws-graphrag-role-session"


class EmbeddingModelInfo(BaseModel):
    dimensions: int | list[int] | None = Field(
        default=None,
        description="The embedding dimensions. Can be a single value or list of supported dimensions.",
    )
    max_sequence_length: int | None = Field(
        default=None,
        description="Maximum sequence length in characters that the model can process.",
    )
    max_sequence_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens the model can process in a single sequence.",
    )


class LanguageModelInfo(BaseModel):
    context_window_size: int = Field(
        description="Maximum context window size in tokens that the model can handle."
    )
    max_output_tokens: int = Field(
        description="Maximum number of tokens the model can generate in a single response."
    )
    supports_performance_optimization: bool = Field(
        default=False,
        description="Whether the model supports performance optimization features.",
    )
    supports_prompt_caching: bool = Field(
        default=False,
        description="Whether the model supports prompt caching to improve performance.",
    )
    supports_thinking: bool = Field(
        default=False,
        description="Whether the model supports thinking/reasoning capabilities.",
    )
    supports_1m_context_window: bool = Field(
        default=False,
        description="Whether the model supports 1M context window.",
    )


class RerankModelInfo(BaseModel):
    max_documents: int = Field(
        default=1000,
        description="Maximum number of documents that can be reranked in a single request.",
    )
    max_query_length: int | None = Field(
        default=None, description="Maximum query length in characters."
    )
    max_query_tokens: int | None = Field(
        default=None, description="Maximum number of tokens allowed in the query."
    )
    max_document_length: int | None = Field(
        default=None, description="Maximum document length in characters."
    )
    max_document_tokens: int | None = Field(
        default=None, description="Maximum number of tokens allowed per document."
    )


_EMBEDDING_MODEL_INFO: dict[EmbeddingModelId, EmbeddingModelInfo] = {
    EmbeddingModelId.TITAN_EMBED_V1: EmbeddingModelInfo(
        dimensions=1536, max_sequence_length=50000, max_sequence_tokens=8192
    ),
    EmbeddingModelId.TITAN_EMBED_V2: EmbeddingModelInfo(
        dimensions=[256, 512, 1024], max_sequence_length=50000, max_sequence_tokens=8192
    ),
    EmbeddingModelId.EMBED_ENGLISH_V3: EmbeddingModelInfo(
        dimensions=1024, max_sequence_length=2048, max_sequence_tokens=512
    ),
    EmbeddingModelId.EMBED_MULTILINGUAL_V3: EmbeddingModelInfo(
        dimensions=1024, max_sequence_length=2048, max_sequence_tokens=512
    ),
    EmbeddingModelId.EMBED_V4: EmbeddingModelInfo(
        dimensions=1024, max_sequence_length=2048, max_sequence_tokens=512
    ),
    # NOTE: add new models here
}

_LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
    LanguageModelId.CLAUDE_V3_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V3_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_performance_optimization=True,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V4_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=8192
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET_V2: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=8192, supports_prompt_caching=True
    ),
    LanguageModelId.CLAUDE_V3_7_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_5_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_1_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_5_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    # NOTE: add new models here
}

_RERANK_MODEL_INFO: dict[str, RerankModelInfo] = {
    RerankModelId.AMAZON_RERANK_V1: RerankModelInfo(
        max_documents=1000, max_query_tokens=2048, max_document_tokens=4096
    ),
    RerankModelId.COHERE_RERANK_V3_5: RerankModelInfo(
        max_documents=1000, max_query_tokens=512, max_document_tokens=4096
    ),
    # NOTE: add new models here
}


ModelIdT = TypeVar("ModelIdT")
ModelInfoT = TypeVar("ModelInfoT")
WrapperT = TypeVar("WrapperT")


class BaseBedrockWrapper:
    _token_counter: BedrockTokenCounter | None = PrivateAttr(default=None)

    def __init__(
        self, token_counter: BedrockTokenCounter | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._token_counter = token_counter

    @property
    def _buffer_tokens(self) -> int:
        # Concrete subclasses (embeddings/rerank wrappers) declare buffer_tokens
        # as a pydantic Field; read it generically so the shared truncation
        # logic stays here without shadowing the subclass field.
        return int(getattr(self, "buffer_tokens", 0))

    def _truncate_text(
        self, text: str, max_chars: int | None, max_tokens: int | None, text_type: str
    ) -> str:
        if not max_chars and not max_tokens:
            return text

        final_text = text

        if max_tokens and self._token_counter is not None:
            effective_tokens = max_tokens - self._buffer_tokens
            truncated, token_count = self._token_counter.truncate_to_token_limit(
                text, effective_tokens
            )
            if len(truncated) < len(text):
                final_text = truncated
                original_count = self._token_counter.count_tokens(text)
                logger.warning(
                    f"{text_type.capitalize()} token count ({original_count}) exceeds maximum ({max_tokens}). Truncating."
                )

        if max_chars and len(text) > max_chars:
            char_truncated = text[:max_chars]
            if len(char_truncated) < len(final_text):
                final_text = char_truncated
                logger.warning(
                    f"{text_type.capitalize()} character count ({len(text)}) exceeds maximum ({max_chars}). Truncating."
                )

        return final_text


class BaseBedrockModelFactory(Generic[ModelIdT, ModelInfoT, WrapperT], ABC):
    BOTO_READ_TIMEOUT: ClassVar[int] = 900
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 5
    # "adaptive" adds client-side rate limiting on top of retries, which is
    # materially better for throttling-heavy Bedrock workloads than "standard".
    BOTO_RETRY_MODE: ClassVar[Literal["legacy", "standard", "adaptive"]] = "adaptive"

    def _boto_config(self, read_timeout: int | None = None) -> BotoConfig:
        # botocore accepts a plain retries dict at runtime; its stub uses a
        # private _RetryDict that a local dict does not nominally satisfy.
        retries = {"max_attempts": self.BOTO_MAX_ATTEMPTS, "mode": self.BOTO_RETRY_MODE}
        if read_timeout is not None:
            return BotoConfig(read_timeout=read_timeout, retries=retries)  # type: ignore[arg-type]
        return BotoConfig(retries=retries)  # type: ignore[arg-type]

    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        region_name: str | None = None,
    ) -> None:
        self.config = config
        self.boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name
        )
        self.boto_session = get_assumed_role_boto_session(
            self.boto_session, assumed_role_arn=config.aws.bedrock.assumed_role_arn
        )
        self.region_name = region_name or config.aws.bedrock.region_name
        boto_config = self._boto_config(read_timeout=self.BOTO_READ_TIMEOUT)
        # Service name is resolved dynamically per subclass, so it is a plain
        # str and does not match types-boto3's literal-overloaded client().
        self._client = self.boto_session.client(
            self._get_boto_service_name(),  # type: ignore[call-overload]
            region_name=self.region_name,
            config=boto_config,
        )
        logger.debug(
            f"Initialized {self.__class__.__name__} for region: '{self.region_name}'"
        )

    @abstractmethod
    def _get_boto_service_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _get_model_info_dict(self) -> dict[ModelIdT, ModelInfoT]:
        raise NotImplementedError

    @abstractmethod
    def get_model(self, model_id: ModelIdT, **kwargs: Any) -> WrapperT:
        raise NotImplementedError

    def get_model_info(self, model_id: ModelIdT) -> ModelInfoT | None:
        return self._get_model_info_dict().get(model_id)

    def get_supported_models(self) -> list[ModelIdT]:
        return list(self._get_model_info_dict().keys())


class BedrockCrossRegionModelHelper:
    @staticmethod
    def get_cross_region_model_id(
        boto_session: boto3.Session,
        model_id: LanguageModelId,
        region_name: str,
        assumed_role_arn: str | None = None,
        enable_global_profile: bool = False,
    ) -> str:
        try:
            boto_session = get_assumed_role_boto_session(
                boto_session, assumed_role_arn=assumed_role_arn
            )
            bedrock_client = boto_session.client("bedrock", region_name=region_name)

            if enable_global_profile:
                global_model_id = (
                    BedrockCrossRegionModelHelper._build_cross_region_model_id(
                        model_id, region_name, is_global=True
                    )
                )
                if BedrockCrossRegionModelHelper._is_cross_region_model_available(
                    bedrock_client, global_model_id
                ):
                    logger.debug(
                        "Using global cross-region model: '%s'", global_model_id
                    )
                    return global_model_id
            regional_model_id = (
                BedrockCrossRegionModelHelper._build_cross_region_model_id(
                    model_id, region_name, is_global=False
                )
            )
            if BedrockCrossRegionModelHelper._is_cross_region_model_available(
                bedrock_client, regional_model_id
            ):
                logger.debug(
                    "Using regional cross-region model: '%s'", regional_model_id
                )
                return regional_model_id
            logger.debug(
                "Cross-region models not available, using standard model: '%s'",
                model_id.value,
            )
            return model_id.value
        except Exception as e:
            logger.warning(
                "Failed to resolve cross-region model for '%s': %s. Falling back to standard model.",
                model_id.value,
                e,
            )
            return model_id.value

    @staticmethod
    def _build_cross_region_model_id(
        model_id: LanguageModelId, region_name: str, is_global: bool = False
    ) -> str:
        if is_global:
            return f"global.{model_id.value}"
        prefix = "apac" if region_name.startswith("ap-") else region_name[:2]
        return f"{prefix}.{model_id.value}"

    @staticmethod
    def _is_cross_region_model_available(
        bedrock_client: Any, cross_region_id: str
    ) -> bool:
        try:
            response = bedrock_client.list_inference_profiles(
                maxResults=1000, typeEquals="SYSTEM_DEFINED"
            )
            available_profiles = {
                profile["inferenceProfileId"]
                for profile in response.get("inferenceProfileSummaries", [])
            }
            return cross_region_id in available_profiles
        except Exception as e:
            raise AWSServiceError(
                f"Failed to check cross-region model availability: {e}"
            ) from e


class BedrockEmbeddingsWrapper(BaseBedrockWrapper, BedrockEmbeddings):
    buffer_tokens: int = Field(default=512, ge=0)
    max_sequence_length: int | None = Field(default=None)
    max_sequence_tokens: int | None = Field(default=None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        truncated_texts = [
            self._truncate_text(
                text, self.max_sequence_length, self.max_sequence_tokens, "document"
            )
            for text in texts
        ]
        return super().embed_documents(truncated_texts)

    def embed_query(self, text: str) -> list[float]:
        truncated_text = self._truncate_text(
            text, self.max_sequence_length, self.max_sequence_tokens, "query"
        )
        return super().embed_query(truncated_text)


class BedrockEmbeddingModelFactory(
    BaseBedrockModelFactory[
        EmbeddingModelId, EmbeddingModelInfo, BedrockEmbeddingsWrapper
    ]
):
    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def _get_model_info_dict(self) -> dict[EmbeddingModelId, EmbeddingModelInfo]:
        return _EMBEDDING_MODEL_INFO

    def get_model(
        self, model_id: EmbeddingModelId, **kwargs: Any
    ) -> BedrockEmbeddingsWrapper:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise EmbeddingModelError(
                f"Unsupported embedding model ID: '{model_id.value}'"
            )

        model_kwargs = {}
        dimensions = kwargs.pop("dimensions", None)

        if dimensions:
            supported_dims = model_info.dimensions
            is_supported = False
            if isinstance(supported_dims, list):
                is_supported = dimensions in supported_dims
            elif isinstance(supported_dims, int):
                is_supported = dimensions == supported_dims

            if not is_supported:
                raise EmbeddingModelError(
                    f"Dimension {dimensions} is not supported by model '{model_id.value}'. "
                    f"Supported dimensions: {supported_dims}"
                )
            if isinstance(supported_dims, list):
                model_kwargs["dimensions"] = dimensions

        token_counter = BedrockTokenCounter(
            model_id=model_id.value, client=self._client
        )
        model = BedrockEmbeddingsWrapper(
            client=self._client,
            model_id=model_id.value,
            model_kwargs=model_kwargs,
            max_sequence_length=model_info.max_sequence_length,
            max_sequence_tokens=model_info.max_sequence_tokens,
            # Accepted by BaseBedrockWrapper.__init__; pydantic's generated
            # __init__ signature hides it from mypy.
            token_counter=token_counter,  # type: ignore[call-arg]
            **kwargs,
        )
        logger.debug(f"Created embedding model: '{model_id.value}'")
        return model


class BedrockLanguageModelFactory(
    BaseBedrockModelFactory[
        LanguageModelId, LanguageModelInfo, ChatBedrock | ChatBedrockConverse
    ]
):
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0
    DEFAULT_TOP_K: ClassVar[int] = 50
    DEFAULT_THINKING_BUDGET_TOKENS: ClassVar[int] = 2048
    DEFAULT_LATENCY_MODE: ClassVar[str] = "normal"

    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def _get_model_info_dict(self) -> dict[LanguageModelId, LanguageModelInfo]:
        return _LANGUAGE_MODEL_INFO

    def get_model(
        self,
        model_id: LanguageModelId,
        **kwargs: Any,
    ) -> ChatBedrock | ChatBedrockConverse:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise LanguageModelError(
                f"Unsupported language model ID: '{model_id.value}'"
            )
        resolved_model_id = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            self.boto_session,
            model_id,
            self.region_name or "",
            assumed_role_arn=self.config.aws.bedrock.assumed_role_arn,
            enable_global_profile=self.config.aws.bedrock.enable_global_profile,
        )
        is_cross_region = resolved_model_id != model_id.value
        model_config = self._build_model_config(
            model_info, resolved_model_id, is_cross_region, **kwargs
        )
        model_class = ChatBedrockConverse if is_cross_region else ChatBedrock
        model = model_class(**model_config)
        logger.debug(
            "Created language model: '%s' with class %s",
            resolved_model_id,
            model_class.__name__,
        )
        return model

    def _build_model_config(
        self,
        model_info: LanguageModelInfo,
        resolved_model_id: str,
        is_cross_region: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        enable_thinking = kwargs.get("enable_thinking", False)
        supports_1m_context_window = kwargs.get("supports_1m_context_window", False)
        temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        final_temperature = (
            1.0
            if self._should_enable_thinking(enable_thinking, model_info)
            else temperature
        )
        if final_temperature != temperature:
            logger.debug("Adjusting temperature to 1.0 for thinking mode")
        final_max_tokens = self._validate_max_tokens(
            kwargs.get("max_tokens"), model_info
        )
        config = self._build_base_config(resolved_model_id, is_cross_region, **kwargs)
        if is_cross_region:
            config.update(
                {"max_tokens": final_max_tokens, "temperature": final_temperature}
            )
        else:
            config["model_kwargs"].update(
                {"max_tokens": final_max_tokens, "temperature": final_temperature}
            )
        if supports_1m_context_window and model_info.supports_1m_context_window:
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    {"anthropic_beta": ["context-1m-2025-08-07"]}
                )
            else:
                config["model_kwargs"].setdefault(
                    "additionalModelRequestFields", {}
                ).update({"anthropic_beta": ["context-1m-2025-08-07"]})
            logger.debug("Applied 1M context window support")
        self._apply_model_features(config, model_info, is_cross_region, **kwargs)
        return config

    def _build_base_config(
        self, resolved_model_id: str, is_cross_region: bool, **kwargs: Any
    ) -> dict[str, Any]:
        config = {
            "model_id": resolved_model_id,
            "region_name": self.region_name,
            "client": self._client,
            "callbacks": kwargs.get("callbacks"),
        }
        if (
            self.boto_session.profile_name
            and self.boto_session.profile_name != "default"
        ):
            config["credentials_profile_name"] = self.boto_session.profile_name
        common_params = {
            "stop_sequences": ["\n\nHuman:"],
        }
        if is_cross_region:
            config.update(common_params)
        else:
            config["model_kwargs"] = {
                "top_k": kwargs.get("top_k", self.DEFAULT_TOP_K),
                **common_params,
            }
        return config

    def _apply_model_features(
        self,
        config: dict[str, Any],
        model_info: LanguageModelInfo,
        is_cross_region: bool,
        **kwargs: Any,
    ) -> None:
        enable_perf = kwargs.get("enable_performance_optimization", False)
        enable_think = kwargs.get("enable_thinking", False)
        if self._should_enable_performance_optimization(
            enable_perf, model_info, is_cross_region
        ):
            latency = kwargs.get("latency_mode", self.DEFAULT_LATENCY_MODE)
            config.setdefault("performanceConfig", {}).update({"latency": latency})
            logger.debug(
                "Applied performance optimization (latency_mode='%s')", latency
            )
        if self._should_enable_thinking(enable_think, model_info):
            budget = kwargs.get(
                "thinking_budget_tokens", self.DEFAULT_THINKING_BUDGET_TOKENS
            )
            think_config = {"thinking": {"type": "enabled", "budget_tokens": budget}}
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    think_config
                )
            else:
                config.setdefault("model_kwargs", {}).update(think_config)
            logger.debug("Applied thinking mode (budget_tokens=%d)", budget)
        self._apply_guardrail(config, is_cross_region)

    def _apply_guardrail(self, config: dict[str, Any], is_cross_region: bool) -> None:
        """Attach Bedrock Guardrails to the model when configured.

        ChatBedrockConverse exposes ``guardrail_config`` (Converse-API shape);
        ChatBedrock exposes ``guardrails`` (InvokeModel shape). When no guardrail
        identifier is set the model is created without guardrails (no-op).
        """
        guardrail = self.config.aws.bedrock.guardrail
        if not guardrail.enabled:
            return
        if is_cross_region:
            # ChatBedrockConverse passes guardrail_config straight to the
            # Converse API, whose trace field is the literal "enabled"/"disabled".
            config["guardrail_config"] = {
                "guardrailIdentifier": guardrail.identifier,
                "guardrailVersion": guardrail.version,
                "trace": "enabled" if guardrail.trace else "disabled",
            }
        else:
            # ChatBedrock (InvokeModel) treats guardrails["trace"] as a
            # truthiness flag (`if self.guardrails.get("trace")`), so a non-empty
            # string like "disabled" would wrongly enable tracing — pass a bool.
            config["guardrails"] = {
                "guardrailIdentifier": guardrail.identifier,
                "guardrailVersion": guardrail.version,
                "trace": guardrail.trace,
            }
        logger.debug("Applied Bedrock guardrail '%s'", guardrail.identifier)

    @staticmethod
    def _validate_max_tokens(
        max_tokens: int | None, model_info: LanguageModelInfo
    ) -> int:
        final_max_tokens = max_tokens or model_info.max_output_tokens
        if final_max_tokens > model_info.max_output_tokens:
            logger.warning(
                "Requested max_tokens (%d) exceeds model's maximum (%d). Adjusting.",
                final_max_tokens,
                model_info.max_output_tokens,
            )
            return model_info.max_output_tokens
        return final_max_tokens

    @staticmethod
    def _should_enable_performance_optimization(
        enable: bool, model_info: LanguageModelInfo, is_cross_region: bool
    ) -> bool:
        return (
            enable
            and model_info.supports_performance_optimization
            and not is_cross_region
        )

    @staticmethod
    def _should_enable_thinking(enable: bool, model_info: LanguageModelInfo) -> bool:
        return enable and model_info.supports_thinking


class BedrockRerankWrapper(BaseBedrockWrapper, BedrockRerank):
    buffer_tokens: int = Field(default=64, ge=0)
    max_documents: int = Field(default=1000, ge=1)
    max_query_length: int | None = Field(default=None)
    max_query_tokens: int | None = Field(default=None)
    max_document_length: int | None = Field(default=None)
    max_document_tokens: int | None = Field(default=None)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: list[BaseCallbackHandler] | BaseCallbackManager | None = None,
    ) -> list[Document]:
        if len(documents) > self.max_documents:
            logger.warning(
                f"Document count ({len(documents)}) exceeds limit ({self.max_documents}). Using first {self.max_documents} documents."
            )
            documents = documents[: self.max_documents]

        original_top_n = self.top_n
        if self.top_n is not None and len(documents) < self.top_n:
            self.top_n = len(documents)
            logger.info(
                f"Adjusted top_n from {original_top_n} to {self.top_n} to match document count"
            )

        truncated_query = self._truncate_text(
            query, self.max_query_length, self.max_query_tokens, "query"
        )

        for doc in documents:
            doc.page_content = self._truncate_text(
                doc.page_content,
                self.max_document_length,
                self.max_document_tokens,
                "document",
            )

        try:
            result = super().compress_documents(
                documents, truncated_query, callbacks=callbacks
            )
            return list(result)
        except Exception as e:
            raise RerankModelError(f"Reranking failed: {e}") from e
        finally:
            self.top_n = original_top_n


class BedrockRerankModelFactory(
    BaseBedrockModelFactory[str, RerankModelInfo, BedrockRerankWrapper]
):
    DEFAULT_TOP_K: ClassVar[int] = 100

    def _get_boto_service_name(self) -> str:
        return "bedrock-agent-runtime"

    def _get_model_info_dict(self) -> dict[str, RerankModelInfo]:
        return _RERANK_MODEL_INFO

    def get_model(
        self, model_id: RerankModelId | str, **kwargs: Any
    ) -> BedrockRerankWrapper:
        if isinstance(model_id, str):
            try:
                model_id = RerankModelId(model_id)
            except ValueError as e:
                raise RerankModelError(
                    f"Unsupported rerank model ID: '{model_id}'"
                ) from e

        model_info = self.get_model_info(model_id)
        if not model_info:
            raise RerankModelError(f"Unsupported rerank model ID: '{model_id.value}'")

        top_k = kwargs.pop("top_k", self.DEFAULT_TOP_K)
        if top_k > model_info.max_documents:
            logger.warning(
                f"Requested 'top_k' ({top_k}) exceeds model's maximum ({model_info.max_documents}). Adjusting."
            )
            top_k = model_info.max_documents

        model_arn = (
            f"arn:aws:bedrock:{self.region_name}::foundation-model/{model_id.value}"
        )

        bedrock_runtime_client = self.boto_session.client(
            "bedrock-runtime",
            region_name=self.region_name,
            config=self._boto_config(),
        )
        token_counter = BedrockTokenCounter(
            model_id=model_id.value, client=bedrock_runtime_client
        )
        model = BedrockRerankWrapper(
            model_arn=model_arn,
            top_n=top_k,
            max_query_length=model_info.max_query_length,
            max_query_tokens=model_info.max_query_tokens,
            max_document_length=model_info.max_document_length,
            max_document_tokens=model_info.max_document_tokens,
            region_name=self.region_name,
            credentials_profile_name=self.boto_session.profile_name,
            client=self._client,
            # Accepted by BaseBedrockWrapper.__init__; pydantic's generated
            # __init__ signature hides it from mypy.
            token_counter=token_counter,  # type: ignore[call-arg]
            **kwargs,
        )
        logger.debug(f"Created rerank model: '{model_id.value}'")
        return model


def get_assumed_role_boto_session(
    boto_session: boto3.Session,
    assumed_role_arn: str | None = None,
    role_session_name: str = DEFAULT_ROLE_SESSION_NAME,
    duration_seconds: int = 3600,
) -> boto3.Session:
    if assumed_role_arn is None:
        return boto_session

    try:
        credentials = boto_session.get_credentials()
        if credentials and hasattr(credentials, "method"):
            if credentials.method == "assume-role":
                sts_client = boto_session.client("sts")
                try:
                    caller_identity = sts_client.get_caller_identity()
                    current_arn = caller_identity.get("Arn", "")
                    if "assumed-role" in current_arn:
                        current_role_name = (
                            current_arn.split("/")[-2] if "/" in current_arn else ""
                        )
                        target_role_name = (
                            assumed_role_arn.split("/")[-1]
                            if "/" in assumed_role_arn
                            else ""
                        )

                        if current_role_name == target_role_name:
                            logger.debug(
                                "Already using assumed role '%s', skipping duplicate assume",
                                assumed_role_arn,
                            )
                            return boto_session
                except Exception as e:
                    logger.debug(f"Could not verify current role identity: {e}")
    except Exception as e:
        logger.debug(f"Could not check assumed role status: {e}")

    logger.info(
        "Using aws-assume-role-lib to assume role: '%s' with session name: '%s'",
        assumed_role_arn,
        role_session_name,
    )
    return assume_role(
        boto_session,
        assumed_role_arn,
        RoleSessionName=role_session_name,
        DurationSeconds=duration_seconds,
    )
