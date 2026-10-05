# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Generic, Literal, TypeVar
from uuid import UUID

import boto3
from aws_assume_role_lib.aws_assume_role_lib import assume_role
from botocore.config import Config as BotoConfig
from langchain_aws import BedrockEmbeddings, ChatBedrock, ChatBedrockConverse
from langchain_aws.document_compressors.rerank import BedrockRerank
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.documents import Document
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel, Field, PrivateAttr

from unified_kg_rag.adapters.aws.bedrock_retry import (
    DEFAULT_BASE_DELAY_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_DELAY_SECONDS,
    DEFAULT_MAX_TOTAL_SECONDS,
    call_with_transient_retry,
)
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.domain.models import (
    Config,
    EmbeddingModelId,
    LanguageModelId,
    ModelPurpose,
    RerankModelId,
)
from unified_kg_rag.shared import (
    AWSServiceError,
    EmbeddingModelError,
    LanguageModelError,
    RerankModelError,
    get_logger,
)

logger = get_logger(__name__)


DEFAULT_ROLE_SESSION_NAME: str = "unified-kg-rag-on-aws-role-session"


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
    supports_count_tokens: bool = Field(
        default=False,
        description=(
            "Whether BedrockTokenCounter may call CountTokens for this model. The "
            "counter sends a Converse-shaped input, which embedding models do not "
            "accept, so truncation uses the script-aware estimate directly "
            "instead of paying a failing round trip per text."
        ),
    )


ModelProvider = Literal["anthropic", "openai"]

# Effort levels as documented per model family. Anthropic levels are sent as
# ``output_config.effort``; OpenAI levels as ``reasoning_effort``.
_ANTHROPIC_EFFORTS_ALL: frozenset[str] = frozenset(
    {"low", "medium", "high", "xhigh", "max"}
)
_OPENAI_EFFORTS_GPT6: frozenset[str] = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)


class LanguageModelInfo(BaseModel):
    provider: ModelProvider = Field(
        default="anthropic",
        description=(
            "Model provider. Selects the provider-specific request shape: "
            "Anthropic thinking/output_config, anthropic_beta headers and the "
            "legacy stop sequence for 'anthropic'; reasoning_effort for 'openai'."
        ),
    )
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
        description=(
            "Whether the model can reach a 1M context window via the "
            "'context-1m-2025-08-07' beta opt-in. Requires "
            "aws.bedrock.enable_1m_context; without it the effective window is "
            "context_window_size. Use native_1m_context_window for models where "
            "1M needs no opt-in."
        ),
    )
    adaptive_thinking_only: bool = Field(
        default=False,
        description=(
            "Whether the model accepts only adaptive thinking. Claude 4.7+ rejects "
            "the manual {'type': 'enabled', 'budget_tokens': N} shape with a 400; "
            "thinking depth is steered by 'effort' instead of a token budget. "
            "These models also think by default — Sonnet 5 / Fable 5 cannot be "
            "turned off at all — so their requests always carry the adaptive "
            "config, which is what lets 'effort' through."
        ),
    )
    supports_sampling_params: bool = Field(
        default=True,
        description=(
            "Whether temperature/top_p/top_k are accepted. Claude 4.7+ removed "
            "them and returns a 400 for non-default values; the documented "
            "migration path is to omit them and steer behaviour by prompting."
        ),
    )
    native_1m_context_window: bool = Field(
        default=False,
        description=(
            "Whether the 1M context window is the default and needs no beta "
            "opt-in header. True for Claude 5, where 1M is both default and max."
        ),
    )
    requires_inference_profile: bool = Field(
        default=False,
        description=(
            "Whether the model is INFERENCE_PROFILE-only (no ON_DEMAND). Newer "
            "Claude models ship without on-demand throughput, so invoking the "
            "bare model id fails — a cross-region profile must resolve."
        ),
    )
    supports_adaptive_thinking: bool = Field(
        default=False,
        description=(
            "Whether the model accepts Anthropic adaptive thinking "
            "({'type': 'adaptive'} + output_config.effort). Claude 4.6 accepts it "
            "alongside the deprecated budget_tokens shape, so an opt-in thinking "
            "request uses adaptive there; adaptive_thinking_only implies it."
        ),
    )
    supported_efforts: frozenset[str] | None = Field(
        default=None,
        description=(
            "Effort levels the model card documents. A configured effort outside "
            "this set fails fast instead of surfacing as a Bedrock 400. None means "
            "the levels are not documented and the value is passed through."
        ),
    )
    supports_count_tokens: bool = Field(
        default=True,
        description=(
            "Whether the CountTokens API accepts the model on bedrock-runtime. "
            "When False, token counting uses the local estimate directly instead "
            "of issuing a call that is guaranteed to fail."
        ),
    )

    BETA_1M_CONTEXT_WINDOW_SIZE: ClassVar[int] = 1000000

    @property
    def uses_adaptive_thinking(self) -> bool:
        """Whether thinking requests take the adaptive + effort shape."""
        return self.provider == "anthropic" and (
            self.supports_adaptive_thinking or self.adaptive_thinking_only
        )

    @property
    def always_reasons(self) -> bool:
        """Whether every request carries the reasoning config.

        Adaptive-only Claude models think by default (and Sonnet 5 / Fable 5
        cannot turn it off); OpenAI GPT models are reasoning models whose depth
        is set per request. In both cases the reasoning block is what carries
        the configured effort, so it is sent regardless of the caller's flag.
        """
        if not self.supports_thinking:
            return False
        return self.adaptive_thinking_only or self.provider == "openai"

    def effective_context_window(self, enable_1m_context: bool = False) -> int:
        """Context window the model will actually honour for a request.

        ``context_window_size`` is the baseline. Models that reach 1M only
        through the beta opt-in report that baseline until the opt-in is on, so
        a caller sizing a token budget must ask here rather than reading the
        field directly — otherwise the budget ignores an enabled 1M window.
        """
        if enable_1m_context and self.supports_1m_context_window:
            return max(self.context_window_size, self.BETA_1M_CONTEXT_WINDOW_SIZE)
        return self.context_window_size


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
    supports_count_tokens: bool = Field(
        default=False,
        description=(
            "Whether BedrockTokenCounter may call CountTokens for this model. "
            "Rerank models do not support Converse, the input shape the counter "
            "sends, so query/document truncation uses the script-aware estimate "
            "and no bedrock-runtime client is created for counting."
        ),
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
        # Cohere Embed v4 has a 128K-token context (NOT the 512 of Embed v3).
        # max_sequence_length is a character ceiling for the char-based truncation
        # fallback; scaled from the token budget at ~4 chars/token. Under-stating
        # this (the old 512) would clamp every input to 512 tokens — far worse
        # than Titan's 8192 — silently truncating most documents.
        dimensions=1024,
        max_sequence_length=512000,
        max_sequence_tokens=128000,
    ),
    # NOTE: add new models here
}

# Capability sources: the Amazon Bedrock model cards
# (docs.aws.amazon.com/bedrock/latest/userguide/model-card-<provider>-<model>.html)
# and the adaptive-thinking guide (claude-messages-adaptive-thinking.html).
_LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
    # --- Claude 5.5 -----------------------------------------------------
    # 1M context / 128K output; adaptive thinking with effort low..max;
    # CountTokens is not supported on bedrock-runtime.
    LanguageModelId.CLAUDE_V5_5_SONNET: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        supported_efforts=_ANTHROPIC_EFFORTS_ALL,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.CLAUDE_V5_5_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        # Adaptive thinking is always on and cannot be disabled.
        adaptive_thinking_only=True,
        supported_efforts=_ANTHROPIC_EFFORTS_ALL,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    # --- Claude Fable 5.x -----------------------------------------------
    # Adaptive-only; temperature must be 1.0/unset and top_k is unsupported,
    # so sampling params are omitted. Both require the account's data
    # retention mode to be 'aws_review' (set via the Data Retention API), and
    # dual-use classifiers may end a response with stop_reason 'refusal'.
    LanguageModelId.CLAUDE_V5_1_FABLE: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        supported_efforts=_ANTHROPIC_EFFORTS_ALL,
        supports_sampling_params=False,
        requires_inference_profile=True,
        # CountTokens is listed for bedrock-mantle only.
        supports_count_tokens=False,
    ),
    LanguageModelId.CLAUDE_V5_FABLE: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
    ),
    # --- Claude 5 -------------------------------------------------------
    # 1M context is the default (and the maximum), so no beta opt-in header.
    # Both take adaptive thinking only and reject sampling parameters.
    LanguageModelId.CLAUDE_V5_SONNET: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.CLAUDE_V5_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        # Opus 5 also accepts {'type': 'disabled'}, but only at effort <= high.
        # We always send adaptive, so that cap never applies.
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    # --- Claude 4.6 - 4.8 -----------------------------------------------
    # 1M context per the model cards. 4.7 accepts adaptive thinking only and
    # dropped sampling parameters; 4.8 is assumed to keep the 4.7 contract
    # (its card states only "Reasoning: Supported").
    LanguageModelId.CLAUDE_V4_8_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.CLAUDE_V4_7_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    # 4.6 still accepts sampling params and disabled thinking, so thinking stays
    # opt-in; when requested it uses adaptive + effort because budget_tokens is
    # deprecated on these models.
    LanguageModelId.CLAUDE_V4_6_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_adaptive_thinking=True,
        supported_efforts=_ANTHROPIC_EFFORTS_ALL,
        requires_inference_profile=True,
    ),
    LanguageModelId.CLAUDE_V4_6_SONNET: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_adaptive_thinking=True,
        # 'xhigh' is documented for Opus models only.
        supported_efforts=frozenset({"low", "medium", "high", "max"}),
        requires_inference_profile=True,
    ),
    LanguageModelId.CLAUDE_V3_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V3_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
    ),
    LanguageModelId.CLAUDE_V3_OPUS: LanguageModelInfo(
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
    # --- OpenAI GPT (proprietary) ---------------------------------------
    # Served on bedrock-runtime through Converse with US-geo / global inference
    # profiles only. Converse gets implicit prompt caching only (the native
    # cachePoint field is rejected), so no explicit cache markers are sent, and
    # CountTokens is not supported. Sampling params are omitted: these are
    # reasoning models and the cards do not document temperature/top_p.
    LanguageModelId.GPT_V6_1_SOL: LanguageModelInfo(
        provider="openai",
        context_window_size=1000000,
        max_output_tokens=131072,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.GPT_V6_ASTRA: LanguageModelInfo(
        provider="openai",
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.GPT_V6_SOL: LanguageModelInfo(
        provider="openai",
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supported_efforts=_OPENAI_EFFORTS_GPT6,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.GPT_V6_LUNA: LanguageModelInfo(
        provider="openai",
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supported_efforts=_OPENAI_EFFORTS_GPT6,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.GPT_V5_6_SOL: LanguageModelInfo(
        provider="openai",
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.GPT_V5_6_TERRA: LanguageModelInfo(
        provider="openai",
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.GPT_V5_6_LUNA: LanguageModelInfo(
        provider="openai",
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    # The GPT-5.5 / GPT-5.4 cards list bedrock-mantle only, while the
    # bedrock-runtime catalog exposes us./global. profiles for both.
    LanguageModelId.GPT_V5_5: LanguageModelInfo(
        provider="openai",
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.GPT_V5_4: LanguageModelInfo(
        provider="openai",
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
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


def get_language_model_info(model_id: LanguageModelId) -> LanguageModelInfo | None:
    """Capability record for a language model, or None if unregistered.

    Public read accessor for the capability table so callers that need a
    model's limits without constructing a factory (which opens a boto client)
    don't reach into the private dict.
    """
    return _LANGUAGE_MODEL_INFO.get(model_id)


def get_embedding_model_info(model_id: EmbeddingModelId) -> EmbeddingModelInfo | None:
    """Capability record for an embedding model, or None if unregistered."""
    return _EMBEDDING_MODEL_INFO.get(model_id)


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
                    "%s token count (%s) exceeds maximum (%s). Truncating.",
                    text_type.capitalize(),
                    original_count,
                    max_tokens,
                )

        if max_chars and len(text) > max_chars:
            char_truncated = text[:max_chars]
            if len(char_truncated) < len(final_text):
                final_text = char_truncated
                logger.warning(
                    "%s character count (%s) exceeds maximum (%s). Truncating.",
                    text_type.capitalize(),
                    len(text),
                    max_chars,
                )

        return final_text


class BaseBedrockModelFactory(Generic[ModelIdT, ModelInfoT, WrapperT], ABC):
    # Per-request socket read timeout. A single LLM/embedding response completes
    # in well under this; the cap exists so a stalled Converse call (no bytes
    # from the server) fails fast and is retried, instead of blocking a worker
    # for many minutes. Kept generous enough for long generations (community
    # reports, extended thinking) but short enough that a hung socket doesn't
    # freeze a whole ProcessPool-parallel stage (e.g. claim extraction).
    BOTO_READ_TIMEOUT: ClassVar[int] = 300
    # TCP connect timeout. Without this, a client whose endpoint is unreachable
    # (e.g. a private/no-NAT VPC missing the relevant Bedrock interface endpoint,
    # such as bedrock-agent-runtime for the Rerank API) blocks at the socket
    # level effectively forever. A bounded connect timeout turns that hang into a
    # prompt, retryable error — which callers that degrade gracefully (e.g. the
    # hybrid scorer's rerank fallback to RRF-only) can actually catch.
    BOTO_CONNECT_TIMEOUT: ClassVar[int] = 10
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 5
    # "adaptive" adds client-side rate limiting on top of retries, which is
    # materially better for throttling-heavy Bedrock workloads than "standard".
    BOTO_RETRY_MODE: ClassVar[Literal["legacy", "standard", "adaptive"]] = "adaptive"

    def _boto_config(self, read_timeout: int | None = None) -> BotoConfig:
        # botocore accepts a plain retries dict at runtime; its stub uses a
        # private _RetryDict that a local dict does not nominally satisfy.
        retries = {"max_attempts": self.BOTO_MAX_ATTEMPTS, "mode": self.BOTO_RETRY_MODE}
        if read_timeout is not None:
            return BotoConfig(
                connect_timeout=self.BOTO_CONNECT_TIMEOUT,
                read_timeout=read_timeout,
                retries=retries,  # type: ignore[arg-type]
            )
        return BotoConfig(
            connect_timeout=self.BOTO_CONNECT_TIMEOUT,
            retries=retries,  # type: ignore[arg-type]
        )

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
            "Initialized %s for region: '%s'", self.__class__.__name__, self.region_name
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


class BedrockCrossRegionModelHelper:
    # Cache the system-defined inference-profile id set per region. Resolving a
    # model id used to call list_inference_profiles on EVERY get_model() (twice
    # when enable_global_profile) — once per model creation across the whole
    # pipeline. The set is process-stable, so fetch it once per region.
    _profiles_by_region: ClassVar[dict[str, set[str]]] = {}

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
                    bedrock_client, global_model_id, region_name
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
                bedrock_client, regional_model_id, region_name
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

    @classmethod
    def _get_available_profiles(cls, bedrock_client: Any, region_name: str) -> set[str]:
        """System-defined inference-profile ids for a region, fetched once.

        Cached per region on the class so resolving many models in one process
        does not re-issue list_inference_profiles each time.
        """
        cached = cls._profiles_by_region.get(region_name)
        if cached is not None:
            return cached
        try:
            response = bedrock_client.list_inference_profiles(
                maxResults=1000, typeEquals="SYSTEM_DEFINED"
            )
            profiles = {
                profile["inferenceProfileId"]
                for profile in response.get("inferenceProfileSummaries", [])
            }
        except Exception as e:
            raise AWSServiceError(
                f"Failed to check cross-region model availability: {e}"
            ) from e
        cls._profiles_by_region[region_name] = profiles
        return profiles

    @classmethod
    def _is_cross_region_model_available(
        cls, bedrock_client: Any, cross_region_id: str, region_name: str
    ) -> bool:
        return cross_region_id in cls._get_available_profiles(
            bedrock_client, region_name
        )


class BedrockEmbeddingsWrapper(BaseBedrockWrapper, BedrockEmbeddings):
    buffer_tokens: int = Field(default=512, ge=0)
    max_sequence_length: int | None = Field(default=None)
    max_sequence_tokens: int | None = Field(default=None)
    # Application-level retry for transient Bedrock model errors (e.g. HTTP 424
    # ModelErrorException) that botocore's retry modes do not cover. Applied
    # per InvokeModel call, so every sync/async embed path benefits and a
    # failure mid-batch does not restart the already-embedded texts.
    transient_max_attempts: int = Field(default=DEFAULT_MAX_ATTEMPTS, ge=1)
    transient_base_delay: float = Field(default=DEFAULT_BASE_DELAY_SECONDS, ge=0)
    transient_max_delay: float = Field(default=DEFAULT_MAX_DELAY_SECONDS, ge=0)
    transient_max_total_seconds: float = Field(default=DEFAULT_MAX_TOTAL_SECONDS, ge=0)

    def _invoke_model(self, input_body: dict[str, Any] | None = None) -> dict[str, Any]:
        # Single choke point for every embedding request in langchain-aws
        # (embed_documents, embed_query, Cohere batch, and the async variants,
        # which run embed_query in an executor).
        body = input_body or {}
        return call_with_transient_retry(
            lambda: super(BedrockEmbeddingsWrapper, self)._invoke_model(
                input_body=body
            ),
            operation=f"embed:{self.model_id}",
            max_attempts=self.transient_max_attempts,
            base_delay=self.transient_base_delay,
            max_delay=self.transient_max_delay,
            max_total_seconds=self.transient_max_total_seconds,
        )

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
            model_id=model_id.value,
            client=self._client,
            api_supported=model_info.supports_count_tokens,
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
        logger.debug("Created embedding model: '%s'", model_id.value)
        return model


_GUARDRAIL_INTERVENED_STOP_REASON = "guardrail_intervened"
# InvokeModel response-body flag (langchain_aws GUARDRAILS_BODY_KEY).
_GUARDRAIL_ACTION_KEY = "amazon-bedrock-guardrailAction"


def _message_guardrail_intervened(metadata: dict[str, Any]) -> bool:
    """Best-effort detection of a guardrail intervention in response metadata.

    Converse (``ChatBedrockConverse``) reports ``stopReason ==
    "guardrail_intervened"``; InvokeModel surfaces the body flag
    ``amazon-bedrock-guardrailAction == "INTERVENED"`` when langchain_aws
    propagates it. Anything else is treated as "not intervened".
    """
    stop_reason = metadata.get("stopReason") or metadata.get("stop_reason")
    if stop_reason == _GUARDRAIL_INTERVENED_STOP_REASON:
        return True
    return metadata.get(_GUARDRAIL_ACTION_KEY) == "INTERVENED"


class GuardrailInterventionHandler(BaseCallbackHandler):
    """Make guardrail interventions visible instead of silently degrading.

    When a guardrail blocks or masks a call, Bedrock still returns a normal
    response (the configured "blocked" message), so a downstream parser just
    sees text with no entities/answer. This handler logs each intervention at
    WARNING with the model purpose and a process-wide running count.

    It deliberately does NOT raise: on the query path the blocked message *is*
    the intended user-facing response, and raising from a callback would turn
    it into an opaque chain error. Operators alert on the WARNING (or read
    :meth:`intervention_count`).
    """

    _count: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, guardrail_identifier: str, purpose: ModelPurpose) -> None:
        self.guardrail_identifier = guardrail_identifier
        self.purpose = purpose
        # InvokeModel with trace enabled reports through on_llm_error and then
        # still calls on_llm_end; remember those runs to count each call once.
        self._flagged_runs: set[UUID] = set()

    @classmethod
    def intervention_count(cls) -> int:
        with cls._lock:
            return cls._count

    @classmethod
    def reset_count(cls) -> None:
        with cls._lock:
            cls._count = 0

    def _record(self) -> None:
        with self._lock:
            type(self)._count += 1
            total = type(self)._count
        logger.warning(
            "Bedrock guardrail '%s' intervened on a %s model call; the response "
            "was blocked or masked (interventions in this process: %d)",
            self.guardrail_identifier,
            self.purpose.value,
            total,
        )

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        if kwargs.get("reason") == "GUARDRAIL_INTERVENED":
            self._flagged_runs.add(run_id)
            self._record()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        if run_id in self._flagged_runs:
            self._flagged_runs.discard(run_id)
            return
        for generations in response.generations:
            for generation in generations:
                if isinstance(
                    generation, ChatGeneration
                ) and _message_guardrail_intervened(
                    generation.message.response_metadata
                ):
                    self._record()
                    return


class BedrockLanguageModelFactory(
    BaseBedrockModelFactory[
        LanguageModelId, LanguageModelInfo, ChatBedrock | ChatBedrockConverse
    ]
):
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0
    DEFAULT_TOP_K: ClassVar[int] = 50
    DEFAULT_THINKING_BUDGET_TOKENS: ClassVar[int] = 2048
    DEFAULT_LATENCY_MODE: ClassVar[str] = "normal"
    # Effort replaces budget_tokens on adaptive-thinking models. "high" is the
    # Bedrock default; "medium" trades some depth for tokens and latency.
    DEFAULT_EFFORT: ClassVar[str] = "high"
    VALID_EFFORTS: ClassVar[frozenset[str]] = frozenset(
        {"low", "medium", "high", "xhigh", "max"}
    )

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
        if model_info.requires_inference_profile and not is_cross_region:
            # Claude 4.6+ ships INFERENCE_PROFILE-only (no ON_DEMAND throughput),
            # so the bare model id is not invocable. Resolution silently falls
            # back to it, which would surface later as an opaque Bedrock error —
            # fail here with the actual remedy instead.
            raise LanguageModelError(
                f"Model '{model_id.value}' is only available through a "
                f"cross-region inference profile, but none resolved in region "
                f"'{self.region_name}'. Enable aws.bedrock.enable_global_profile, "
                f"grant bedrock:ListInferenceProfiles, or choose a region where "
                f"a profile for this model exists."
            )
        # ChatBedrock speaks the Anthropic InvokeModel body; every other provider
        # goes through the provider-neutral Converse API. Non-Anthropic models are
        # profile-only today (enforced above), so this also guards a future
        # on-demand entry from being sent an Anthropic-shaped body.
        use_converse = is_cross_region or model_info.provider != "anthropic"
        model_config = self._build_model_config(
            model_info, resolved_model_id, use_converse, **kwargs
        )
        model_class = ChatBedrockConverse if use_converse else ChatBedrock
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
        final_max_tokens = self._validate_max_tokens(
            kwargs.get("max_tokens"), model_info
        )
        config = self._build_base_config(
            resolved_model_id, is_cross_region, model_info, **kwargs
        )
        token_params: dict[str, Any] = {"max_tokens": final_max_tokens}
        if model_info.supports_sampling_params:
            temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
            # Thinking models sample at a fixed temperature of 1.0.
            final_temperature = (
                1.0
                if self._should_enable_thinking(enable_thinking, model_info)
                else temperature
            )
            if final_temperature != temperature:
                logger.debug("Adjusting temperature to 1.0 for thinking mode")
            token_params["temperature"] = final_temperature
        else:
            # Claude 4.7+ removed temperature/top_p/top_k; sending any of them
            # returns a 400. Behaviour is steered by prompting and 'effort'.
            logger.debug(
                "Omitting sampling parameters for '%s' (unsupported by model)",
                resolved_model_id,
            )
        if is_cross_region:
            config.update(token_params)
        else:
            config["model_kwargs"].update(token_params)
        if model_info.native_1m_context_window:
            # 1M is the default window on Claude 5 — the beta opt-in header
            # that older models need is unnecessary (and misleading) here.
            logger.debug(
                "Model '%s' has a native 1M context window; skipping beta header",
                resolved_model_id,
            )
        elif (
            self.config.aws.bedrock.enable_1m_context
            and model_info.supports_1m_context_window
            and model_info.provider == "anthropic"
        ):
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
        self,
        resolved_model_id: str,
        is_cross_region: bool,
        model_info: LanguageModelInfo,
        **kwargs: Any,
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
        # "\n\nHuman:" is an Anthropic text-completion turn marker; it means
        # nothing to other providers, so they get no stop sequence at all.
        common_params: dict[str, Any] = (
            {"stop_sequences": ["\n\nHuman:"]}
            if model_info.provider == "anthropic"
            else {}
        )
        if is_cross_region:
            config.update(common_params)
        elif model_info.supports_sampling_params:
            config["model_kwargs"] = {
                "top_k": kwargs.get("top_k", self.DEFAULT_TOP_K),
                **common_params,
            }
        else:
            # top_k is a sampling parameter; Claude 4.7+ rejects it.
            config["model_kwargs"] = dict(common_params)
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
            think_config = self._build_thinking_config(model_info, **kwargs)
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    think_config
                )
            else:
                config.setdefault("model_kwargs", {}).update(think_config)
        self._apply_guardrail(
            config,
            is_cross_region,
            ModelPurpose(kwargs.get("model_purpose", ModelPurpose.QUERY)),
        )

    def _build_thinking_config(
        self, model_info: LanguageModelInfo, **kwargs: Any
    ) -> dict[str, Any]:
        """Assemble the provider-specific reasoning request fields for a model.

        - OpenAI GPT: ``reasoning_effort`` in ``additionalModelRequestFields`` —
          the Chat Completions field, which the Converse mapping for OpenAI
          models passes through unchanged.
        - Anthropic adaptive (Claude 4.6+): ``{"type": "adaptive"}`` plus
          ``effort`` in its own ``output_config`` object — nesting it inside
          ``thinking`` raises a ``ValidationException``. Adaptive-only models
          (Claude 4.7+) reject the manual ``budget_tokens`` shape with a 400.
        - Older Anthropic: ``{"type": "enabled", "budget_tokens": N}``.
        """
        if model_info.provider == "openai":
            effort = self._resolve_effort(model_info, **kwargs)
            logger.debug("Applied OpenAI reasoning (reasoning_effort='%s')", effort)
            return {"reasoning_effort": effort}
        if not model_info.uses_adaptive_thinking:
            budget = kwargs.get(
                "thinking_budget_tokens", self.DEFAULT_THINKING_BUDGET_TOKENS
            )
            logger.debug("Applied extended thinking (budget_tokens=%d)", budget)
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}
        effort = self._resolve_effort(model_info, **kwargs)
        logger.debug("Applied adaptive thinking (effort='%s')", effort)
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }

    def _resolve_effort(self, model_info: LanguageModelInfo, **kwargs: Any) -> str:
        """Pick the effort level (per-call override, else config) and validate it.

        Fails fast with the documented levels rather than letting Bedrock answer
        with a 400 on the first request.
        """
        effort = kwargs.get("effort") or self.config.aws.bedrock.effort
        if effort not in self.VALID_EFFORTS:
            raise LanguageModelError(
                f"Invalid effort level '{effort}'. "
                f"Valid levels: {sorted(self.VALID_EFFORTS)}"
            )
        allowed = model_info.supported_efforts
        if allowed is not None and effort not in allowed:
            raise LanguageModelError(
                f"Effort level '{effort}' is not supported by this model. "
                f"Supported levels: {sorted(allowed)}"
            )
        return str(effort)

    def _apply_guardrail(
        self,
        config: dict[str, Any],
        is_cross_region: bool,
        purpose: ModelPurpose = ModelPurpose.QUERY,
    ) -> None:
        """Attach Bedrock Guardrails to the model when configured.

        ChatBedrockConverse exposes ``guardrail_config`` (Converse-API shape);
        ChatBedrock exposes ``guardrails`` (InvokeModel shape). When no guardrail
        identifier is set the model is created without guardrails (no-op).
        ``purpose`` scopes it via ``guardrail.apply_to``: by default only
        query-path models are guarded. A guarded model also gets a
        :class:`GuardrailInterventionHandler` so interventions are logged.
        """
        guardrail = self.config.aws.bedrock.guardrail
        if guardrail.identifier is None or not guardrail.applies_to(purpose):
            if guardrail.enabled:
                logger.debug(
                    "Skipping Bedrock guardrail for %s model (apply_to='%s')",
                    purpose.value,
                    guardrail.apply_to,
                )
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
        handler = GuardrailInterventionHandler(guardrail.identifier, purpose)
        callbacks = config.get("callbacks")
        if callbacks is None:
            config["callbacks"] = [handler]
        elif isinstance(callbacks, BaseCallbackManager):
            # Copy first: the manager is caller-owned and may be shared by other
            # (unguarded) models, which must not start counting interventions.
            manager = callbacks.copy()
            manager.add_handler(handler, inherit=False)
            config["callbacks"] = manager
        else:
            config["callbacks"] = [*callbacks, handler]
        logger.debug(
            "Applied Bedrock guardrail '%s' to %s model",
            guardrail.identifier,
            purpose.value,
        )

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
        if not model_info.supports_thinking:
            return False
        # Adaptive-only Claude models think by default (and Sonnet 5 / Fable 5
        # reject {'type': 'disabled'} outright) and GPT models always reason, so
        # their requests carry the reasoning config regardless of the caller's
        # flag — that block is also what carries the configured effort level,
        # which would otherwise be silently dropped.
        return enable or model_info.always_reasons


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
                "Document count (%s) exceeds limit (%s). Using first %s documents.",
                len(documents),
                self.max_documents,
                self.max_documents,
            )
            documents = documents[: self.max_documents]

        original_top_n = self.top_n
        if self.top_n is not None and len(documents) < self.top_n:
            self.top_n = len(documents)
            logger.info(
                "Adjusted top_n from %s to %s to match document count",
                original_top_n,
                self.top_n,
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
                "Requested 'top_k' (%s) exceeds model's maximum (%s). Adjusting.",
                top_k,
                model_info.max_documents,
            )
            top_k = model_info.max_documents

        model_arn = (
            f"arn:aws:bedrock:{self.region_name}::foundation-model/{model_id.value}"
        )

        # Only open a bedrock-runtime client when the counter will actually use
        # it; for rerank models CountTokens is unsupported, so counting goes
        # straight to the estimate.
        bedrock_runtime_client = (
            self.boto_session.client(
                "bedrock-runtime",
                region_name=self.region_name,
                config=self._boto_config(),
            )
            if model_info.supports_count_tokens
            else None
        )
        token_counter = BedrockTokenCounter(
            model_id=model_id.value,
            client=bedrock_runtime_client,
            api_supported=model_info.supports_count_tokens,
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
        logger.debug("Created rerank model: '%s'", model_id.value)
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
                    logger.debug("Could not verify current role identity: %s", e)
    except Exception as e:
        logger.debug("Could not check assumed role status: %s", e)

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
