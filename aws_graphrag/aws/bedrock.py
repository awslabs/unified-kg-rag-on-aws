from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

import boto3
from botocore.config import Config as BotoConfig
from langchain_aws import BedrockEmbeddings, ChatBedrock, ChatBedrockConverse
from langchain_aws.document_compressors.rerank import BedrockRerank
from langchain_core.documents import Document
from pydantic import BaseModel, Field, PrivateAttr
from tiktoken import Encoding, get_encoding

from aws_graphrag.core import (
    AWSServiceError,
    EmbeddingModelError,
    LanguageModelError,
    RerankModelError,
    get_logger,
)
from aws_graphrag.models import Config, EmbeddingModelId, LanguageModelId, RerankModelId

logger = get_logger(__name__)


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
}

_LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
    LanguageModelId.CLAUDE_V3_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
        supports_performance_optimization=True,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=4096
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET_V2: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=4096, supports_prompt_caching=True
    ),
    LanguageModelId.CLAUDE_V3_7_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_1_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
}

_RERANK_MODEL_INFO: dict[str, RerankModelInfo] = {
    RerankModelId.AMAZON_RERANK_V1: RerankModelInfo(
        max_documents=1000, max_query_tokens=2048, max_document_tokens=4096
    ),
    RerankModelId.COHERE_RERANK_V3_5: RerankModelInfo(
        max_documents=1000, max_query_tokens=512, max_document_tokens=4096
    ),
}


ModelIdT = TypeVar("ModelIdT")
ModelInfoT = TypeVar("ModelInfoT")
WrapperT = TypeVar("WrapperT")


class BaseBedrockWrapper:
    buffer_tokens: int = Field(default=128, ge=0)
    _tokenizer: Encoding = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tokenizer = get_encoding("cl100k_base")

    def _truncate_text(
        self, text: str, max_chars: int | None, max_tokens: int | None, text_type: str
    ) -> str:
        if not max_chars and not max_tokens:
            return text

        token_ids = self._tokenizer.encode(text, allowed_special="all")
        final_text = text
        truncated = False

        if max_tokens and len(token_ids) > max_tokens:
            effective_tokens = max_tokens - self.buffer_tokens
            truncated_token_ids = token_ids[:effective_tokens]
            final_text = self._tokenizer.decode(truncated_token_ids)
            logger.warning(
                f"{text_type.capitalize()} token count ({len(token_ids)}) exceeds maximum ({max_tokens}). Truncating."
            )
            truncated = True

        if max_chars and len(text) > max_chars:
            if not truncated or len(text[:max_chars]) < len(final_text):
                final_text = text[:max_chars]
                logger.warning(
                    f"{text_type.capitalize()} character count ({len(text)}) exceeds maximum ({max_chars}). Truncating."
                )

        return final_text


class BaseBedrockModelFactory(Generic[ModelIdT, ModelInfoT, WrapperT], ABC):
    BOTO_READ_TIMEOUT: ClassVar[int] = 300
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 3

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
        self.region_name = region_name or config.aws.bedrock.region_name
        boto_config = BotoConfig(
            read_timeout=self.BOTO_READ_TIMEOUT,
            retries={"max_attempts": self.BOTO_MAX_ATTEMPTS},
        )
        self._client = self.boto_session.client(
            self._get_boto_service_name(),
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
        boto_session: boto3.Session, model_id: LanguageModelId, region_name: str
    ) -> str:
        try:
            bedrock_client = boto_session.client("bedrock", region_name=region_name)
            cross_region_id = (
                BedrockCrossRegionModelHelper._build_cross_region_model_id(
                    model_id, region_name
                )
            )
            if BedrockCrossRegionModelHelper._is_cross_region_model_available(
                bedrock_client, cross_region_id
            ):
                logger.debug(f"Using cross-region model: '{cross_region_id}'")
                return cross_region_id
            logger.debug(
                f"Cross-region model not available, using standard model: '{model_id.value}'"
            )
            return model_id.value
        except Exception as e:
            logger.warning(
                f"Failed to resolve cross-region model for '{model_id.value}': {e}. Falling back to standard model."
            )
            return model_id.value

    @staticmethod
    def _build_cross_region_model_id(
        model_id: LanguageModelId, region_name: str
    ) -> str:
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

        model = BedrockEmbeddingsWrapper(
            client=self._client,
            model_id=model_id.value,
            model_kwargs=model_kwargs,
            max_sequence_length=model_info.max_sequence_length,
            max_sequence_tokens=model_info.max_sequence_tokens,
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
    DEFAULT_TOP_P: ClassVar[float] = 0.95
    DEFAULT_THINKING_BUDGET_TOKENS: ClassVar[int] = 2048
    DEFAULT_LATENCY_MODE: ClassVar[str] = "normal"

    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def _get_model_info_dict(self) -> dict[LanguageModelId, LanguageModelInfo]:
        return _LANGUAGE_MODEL_INFO

    def get_model(
        self, model_id: LanguageModelId, **kwargs: Any
    ) -> ChatBedrock | ChatBedrockConverse:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise LanguageModelError(
                f"Unsupported language model ID: '{model_id.value}'"
            )

        resolved_model_id = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            self.boto_session, model_id, self.region_name
        )
        is_cross_region = resolved_model_id != model_id.value

        model_config = self._build_model_config(
            model_info, resolved_model_id, is_cross_region, **kwargs
        )

        model_class = ChatBedrockConverse if is_cross_region else ChatBedrock
        model = model_class(**model_config)
        logger.debug(
            f"Created language model: '{resolved_model_id}' with class {model_class.__name__}"
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

        self._apply_model_features(config, model_info, is_cross_region, **kwargs)
        return config

    def _build_base_config(
        self, resolved_model_id: str, is_cross_region: bool, **kwargs: Any
    ) -> dict[str, Any]:
        config = {
            "model": resolved_model_id,
            "region_name": self.region_name,
            "credentials_profile_name": self.boto_session.profile_name,
            "client": self._client,
            "callbacks": kwargs.get("callbacks"),
        }

        common_params = {
            "top_p": kwargs.get("top_p", self.DEFAULT_TOP_P),
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
    ):
        enable_perf = kwargs.get("enable_performance_optimization", False)
        enable_think = kwargs.get("enable_thinking", False)

        if self._should_enable_performance_optimization(
            enable_perf, model_info, is_cross_region
        ):
            latency = kwargs.get("latency_mode", self.DEFAULT_LATENCY_MODE)
            config.setdefault("performanceConfig", {}).update({"latency": latency})
            logger.debug(f"Applied performance optimization (latency_mode='{latency}')")

        if self._should_enable_thinking(enable_think, model_info):
            budget = kwargs.get(
                "thinking_budget_tokens", self.DEFAULT_THINKING_BUDGET_TOKENS
            )
            think_config = {"thinking": {"type": "enabled", "budget_tokens": budget}}
            config.setdefault("additional_model_request_fields", {}).update(
                think_config
            )
            logger.debug(f"Applied thinking mode (budget_tokens={budget})")

    @staticmethod
    def _validate_max_tokens(
        max_tokens: int | None, model_info: LanguageModelInfo
    ) -> int:
        final_max_tokens = max_tokens or model_info.max_output_tokens
        if final_max_tokens > model_info.max_output_tokens:
            logger.warning(
                f"Requested max_tokens ({final_max_tokens}) exceeds model's maximum ({model_info.max_output_tokens}). Adjusting."
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
    max_documents: int = Field(default=1000, ge=1)
    max_query_length: int | None = Field(default=None)
    max_query_tokens: int | None = Field(default=None)
    max_document_length: int | None = Field(default=None)
    max_document_tokens: int | None = Field(default=None)

    def compress_documents(
        self, documents: list[Document], query: str, **kwargs: Any
    ) -> list[Document]:
        if len(documents) > self.max_documents:
            logger.warning(
                f"Number of documents ({len(documents)}) exceeds limit ({self.max_documents}). Taking first {self.max_documents} documents."
            )
            documents = documents[: self.max_documents]

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
            result = super().compress_documents(documents, truncated_query, **kwargs)
            return list(result)
        except Exception as e:
            raise RerankModelError(f"Reranking failed: {e}") from e


class BedrockRerankModelFactory(
    BaseBedrockModelFactory[str, RerankModelInfo, BedrockRerankWrapper]
):
    DEFAULT_TOP_K: ClassVar[int] = 100

    def _get_boto_service_name(self) -> str:
        return "bedrock-agent-runtime"

    def _get_model_info_dict(self) -> dict[str, RerankModelInfo]:
        return _RERANK_MODEL_INFO

    def get_model(self, model_id: str, **kwargs: Any) -> BedrockRerankWrapper:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise RerankModelError(f"Unsupported rerank model ID: '{model_id}'")

        top_k = kwargs.pop("top_k", self.DEFAULT_TOP_K)
        if top_k > model_info.max_documents:
            logger.warning(
                f"Requested 'top_k' ({top_k}) exceeds model's maximum ({model_info.max_documents}). Adjusting."
            )
            top_k = model_info.max_documents

        model_arn = f"arn:aws:bedrock:{self.region_name}::foundation-model/{model_id}"

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
            **kwargs,
        )
        logger.debug(f"Created rerank model: '{model_id}'")
        return model
