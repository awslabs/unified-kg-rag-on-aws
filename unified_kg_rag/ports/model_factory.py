# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-factory ports — the LLM / Embedding / Rerank provider boundary.

The domain/application layers obtain language, embedding, and rerank models
through a *factory* that resolves a typed model id to a ready-to-use,
LangChain-compatible object. These ``Protocol``s capture that contract so a
non-Bedrock provider can be introduced by writing a conforming factory adapter,
without changing call sites.

The concrete Bedrock factories in ``adapters.aws.bedrock`` already conform
structurally (``get_model`` / ``get_model_info``); these Protocols make the
boundary explicit and give callers a provider-agnostic type to annotate against.
Defined as ``runtime_checkable`` Protocols (structural typing), so no base-class
change to the existing factories is required.

The capability records ``get_model_info`` returns (``LanguageModelInfo``,
``EmbeddingModelInfo``, ``RerankModelInfo``) are part of the contract, so they
live here; the Bedrock catalog in ``adapters.aws.bedrock_models`` and the local
factories build them.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import BedrockConfig


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


# "other" is any provider without provider-specific request shaping: it goes
# through Converse with no reasoning block, sampling params or cache markers.
ModelProvider = Literal["anthropic", "openai", "other"]


class LanguageModelInfo(BaseModel):
    # Forbid unknown keys so a typo in aws.bedrock.model_overrides fails fast.
    model_config = ConfigDict(extra="forbid")

    provider: ModelProvider = Field(
        default="anthropic",
        description=(
            "Model provider. Selects the provider-specific request shape: "
            "Anthropic thinking/output_config, anthropic_beta headers and the "
            "legacy stop sequence for 'anthropic'; reasoning.effort for 'openai'; "
            "a plain Converse request for 'other'."
        ),
    )
    context_window_size: int = Field(
        description="Maximum context window size in tokens that the model can handle."
    )
    max_output_tokens: int = Field(
        description="Maximum number of tokens the model can generate in a single response."
    )
    supports_prompt_caching: bool = Field(
        default=False,
        description="Whether the model supports prompt caching to improve performance.",
    )
    min_cache_tokens: int = Field(
        default=1024,
        ge=0,
        description=(
            "Minimum prompt-prefix tokens for a cache checkpoint. A shorter "
            "system prompt gets no cache marker: Bedrock would accept it but "
            "cache nothing."
        ),
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
            "These models also think by default — Sonnet 5 / 5.5 and Opus 5.5 "
            "cannot be turned off at all — so their requests always carry the "
            "adaptive config, which is what lets 'effort' through."
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
    supports_streaming: bool | None = Field(
        default=None,
        description=(
            "Whether the model streams through ConverseStream. None leaves it "
            "to langchain-aws, which streams only model ids on its own "
            "allowlist; set it for a model the allowlist does not know yet "
            "(otherwise it falls back to non-streaming Converse with a "
            "warning on every model construction)."
        ),
    )
    token_estimate_multiplier: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Factor applied to the local token estimate (~4 characters per "
            "token for Latin text) when CountTokens is not used for this model. "
            "The estimate is calibrated to the older Claude tokenizer; the "
            "tokenizer introduced with Claude Opus 4.7 counts more tokens for "
            "the same text, so those models set 1.3 and context budgets are not "
            "overfilled. Has no effect on counts the API returns."
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

        Adaptive-only Claude models think by default (and Sonnet 5 / 5.5 and
        Opus 5.5 cannot turn it off); OpenAI GPT models are reasoning models
        whose depth is set per request. In both cases the reasoning block is what carries
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


def effective_max_output_tokens(
    model_info: LanguageModelInfo,
    bedrock_config: BedrockConfig,
    min_output_tokens: int = 0,
) -> int:
    """max_tokens for a request that does not set one explicitly.

    The configured default cap, raised to the prompt's output floor, never
    above the model maximum; an unset cap means the model maximum.
    """
    cap = bedrock_config.default_max_output_tokens
    if cap is None:
        return model_info.max_output_tokens
    return min(max(cap, min_output_tokens), model_info.max_output_tokens)


# Generic over the model-id enum and the returned model type so each concrete
# factory keeps its narrow id type (EmbeddingModelId etc.) WITHOUT breaking
# structural conformance. A non-generic union here would be contravariantly
# incompatible with the narrower concrete factories (they each accept only one
# enum). `Any` defaults preserve the prior permissive behaviour for untyped
# call sites while letting a typed adapter parameterize for static safety.
ModelIdT = TypeVar("ModelIdT", contravariant=True)
ModelT = TypeVar("ModelT", covariant=True)
InfoT = TypeVar("InfoT", covariant=True)


@runtime_checkable
class ModelFactoryPort(Protocol[ModelIdT, ModelT, InfoT]):
    """Resolves a typed model id to a provider model object.

    Implemented by ``BedrockLanguageModelFactory`` /
    ``BedrockEmbeddingModelFactory`` / ``BedrockRerankModelFactory``. ``get_model``
    returns a LangChain-compatible model (chat model, embeddings, or reranker);
    ``get_model_info`` returns the capability record for the id: a
    ``LanguageModelInfo`` for every language-model id, an
    ``EmbeddingModelInfo`` / ``RerankModelInfo`` or ``None`` for an unknown
    embedding or rerank id.

    Parameterized over the id enum, the model type and the record type, so a
    custom backend adapter can annotate
    ``ModelFactoryPort[str, ChatModel, LanguageModelInfo]`` and get static
    checking, while the concrete narrow factories still conform.

    A language-model factory may also define ``is_transient_error(exc) ->
    bool``, the classifier of its errors worth a retry (the Bedrock factory
    names botocore's, the local ones network errors and HTTP 408/429/5xx). The
    ingestion stages' batch retries use it, and without it fall back to
    ``adapters.llm.retry.looks_transient``. It may also define
    ``is_item_error(exc) -> bool``: whether an ingestion item's error is the
    item's own (a request the model rejects for its content), which counts
    against its document (``adapters.llm.retry.item_error_predicate``);
    without it ``adapters.llm.retry.looks_like_item_error`` (HTTP
    400/413/422 read from ``status_code`` or ``response.status_code``) and
    unparseable or cut-off output count. A factory that
    also defines ``transient_retry`` (the Bedrock and Ollama ones) gets a
    chain-level retry of query chains from ``setup_chain`` under that policy.
    """

    def get_model(self, model_id: ModelIdT, **kwargs: Any) -> ModelT: ...

    def get_model_info(self, model_id: ModelIdT) -> InfoT: ...


@runtime_checkable
class TokenCounterPort(Protocol):
    """Counts (and truncates by) tokens for one model.

    Implemented by ``BedrockTokenCounter``. A custom backend supplies one
    through a token-counter factory called as
    ``factory(model_id, cache_maxsize=..., api_supported=...)``; like the model
    factories, it must accept and ignore keyword arguments it does not use.

    A counter may also expose a ``counts_via_api`` bool: True when each count is
    a remote round trip, so callers count many texts concurrently. It is
    optional (not part of the structural check); without it the answer model's
    ``LanguageModelInfo.supports_count_tokens`` decides. ``BedrockTokenCounter``
    turns it off once the model rejects CountTokens.
    """

    def count_tokens(self, text: str) -> int: ...

    def truncate_to_token_limit(
        self, text: str, max_tokens: int
    ) -> tuple[str, int]: ...


# One port per role, typed by the capability record its factory returns.


@runtime_checkable
class LLMFactoryPort(ModelFactoryPort[Any, Any, LanguageModelInfo], Protocol):
    """Language-model factory: a record for every model id."""


@runtime_checkable
class EmbeddingFactoryPort(
    ModelFactoryPort[Any, Any, "EmbeddingModelInfo | None"], Protocol
):
    """Embedding-model factory: ``None`` for an id it has no record for."""


@runtime_checkable
class RerankFactoryPort(ModelFactoryPort[Any, Any, "RerankModelInfo | None"], Protocol):
    """Rerank-model factory: ``None`` for an id it has no record for."""
