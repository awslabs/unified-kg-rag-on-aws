# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bedrock model capability catalog.

Capability records for embedding, language and rerank models, the curated
tables behind them and the lookup that resolves any language-model id (curated
row, provider-family default, then ``aws.bedrock.model_overrides``). Pure data
and lookups: no boto3 or LangChain clients are constructed here.
"""

import re
import threading
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from unified_kg_rag.domain.models import (
    EmbeddingModelId,
    LanguageModelId,
    RerankModelId,
)
from unified_kg_rag.domain.models.config import BedrockConfig
from unified_kg_rag.shared import LanguageModelError, get_logger

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


# "other" is any provider without provider-specific request shaping: it goes
# through Converse with no reasoning block, sampling params or cache markers.
ModelProvider = Literal["anthropic", "openai", "other"]

# Effort levels each model family accepts on bedrock-runtime, as reported by
# the service's own validation errors. Anthropic levels are sent as
# ``output_config.effort``; OpenAI levels as ``reasoning.effort``.
_ANTHROPIC_EFFORTS_ALL: frozenset[str] = frozenset(
    {"low", "medium", "high", "xhigh", "max"}
)
# Claude 4.6 rejects 'xhigh' ("Input should be 'low', 'medium', 'high' or 'max'").
_ANTHROPIC_EFFORTS_NO_XHIGH: frozenset[str] = frozenset(
    {"low", "medium", "high", "max"}
)
# GPT also accepts 'none' (reasoning off); it is not exposed because
# the BedrockConfig effort fields and VALID_EFFORTS only carry the shared levels.
_OPENAI_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})


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
# (docs.aws.amazon.com/bedrock/latest/userguide/model-card-<provider>-<model>.html),
# the adaptive-thinking guide (claude-messages-adaptive-thinking.html) and, for
# min_cache_tokens, the prompt-caching guide (prompt-caching.html).
# Request-shape fields (effort levels, thinking/temperature acceptance,
# CountTokens) for the Claude 4.6+ and GPT entries were checked against live
# bedrock-runtime responses.
_LANGUAGE_MODEL_INFO: dict[str, LanguageModelInfo] = {
    # --- Claude 5.5 -----------------------------------------------------
    # 1M context / 128K output; adaptive thinking with effort low..max. Both
    # reject temperature ("deprecated for this model") and thinking types other
    # than adaptive; CountTokens rejects them on bedrock-runtime.
    LanguageModelId.CLAUDE_V5_5_SONNET: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        min_cache_tokens=512,
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
        min_cache_tokens=512,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        supported_efforts=_ANTHROPIC_EFFORTS_ALL,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
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
        min_cache_tokens=512,
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
    # 1M context per the model cards. Opus 4.7 and 4.8 reject the budget_tokens
    # shape and temperature, accept effort low..max, and allow
    # {'type': 'disabled'}; we always send adaptive so 'effort' is honoured.
    # CountTokens rejects both.
    LanguageModelId.CLAUDE_V4_8_OPUS: LanguageModelInfo(
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
    LanguageModelId.CLAUDE_V4_7_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        min_cache_tokens=4096,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        supported_efforts=_ANTHROPIC_EFFORTS_ALL,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    # 4.6 still accepts sampling params, disabled thinking and budget_tokens, so
    # thinking stays opt-in; when requested it uses adaptive + effort because
    # budget_tokens is deprecated there. Neither accepts 'xhigh'. CountTokens
    # accepts the base model id (not the profile id), which is what we send.
    LanguageModelId.CLAUDE_V4_6_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        min_cache_tokens=4096,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_adaptive_thinking=True,
        supported_efforts=_ANTHROPIC_EFFORTS_NO_XHIGH,
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
        supported_efforts=_ANTHROPIC_EFFORTS_NO_XHIGH,
        requires_inference_profile=True,
    ),
    # Claude Fable 5 / 5.1 are deliberately NOT offered: they need a non-default
    # account data-retention mode (Data Retention API only, no console UI), and
    # an account on the default mode gets "data retention mode 'default' is not
    # available for this model" on every call, so a selectable entry would fail
    # for most deployments.
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
        supports_prompt_caching=True,
        min_cache_tokens=2048,
    ),
    LanguageModelId.CLAUDE_V4_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        min_cache_tokens=4096,
        requires_inference_profile=True,
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
        requires_inference_profile=True,
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
        min_cache_tokens=4096,
        supports_thinking=True,
        supports_1m_context_window=True,
        requires_inference_profile=True,
    ),
    # --- OpenAI GPT (proprietary) ---------------------------------------
    # Served on bedrock-runtime through Converse with us./global. inference
    # profiles only. Reasoning depth is the Responses-style
    # ``reasoning.effort`` (the flat ``reasoning_effort`` is rejected as an
    # unknown parameter); every model accepts low..max (and 'none'). Converse gets implicit
    # prompt caching only (the native cachePoint field is rejected), so no
    # explicit cache markers are sent, and CountTokens rejects these models.
    # Sampling params are omitted: these are reasoning models and the cards do
    # not document temperature/top_p.
    LanguageModelId.GPT_V6_1_SOL: LanguageModelInfo(
        provider="openai",
        supported_efforts=_OPENAI_EFFORTS,
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
        supported_efforts=_OPENAI_EFFORTS,
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
        supported_efforts=_OPENAI_EFFORTS,
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.GPT_V6_LUNA: LanguageModelInfo(
        provider="openai",
        supported_efforts=_OPENAI_EFFORTS,
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    LanguageModelId.GPT_V5_6_SOL: LanguageModelInfo(
        provider="openai",
        supported_efforts=_OPENAI_EFFORTS,
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
        supported_efforts=_OPENAI_EFFORTS,
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
        supported_efforts=_OPENAI_EFFORTS,
        context_window_size=1050000,
        max_output_tokens=128000,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    # The GPT-5.5 / GPT-5.4 cards list bedrock-mantle only, but Converse on
    # bedrock-runtime serves both through their us./global. profiles.
    LanguageModelId.GPT_V5_5: LanguageModelInfo(
        provider="openai",
        supported_efforts=_OPENAI_EFFORTS,
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
        supported_efforts=_OPENAI_EFFORTS,
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


# Geography prefixes of system-defined cross-region inference profiles. An id
# that already carries one is a profile id: it is invoked as-is and its
# capabilities are those of the base model id behind the prefix.
_PROFILE_PREFIXES: tuple[str, ...] = (
    "global.",
    "us-gov.",
    "us.",
    "eu.",
    "apac.",
    "jp.",
    "au.",
    "ca.",
)

# Claude generation from either id style: 'claude-3-5-haiku-…' / 'claude-3-haiku-…'
# (version first) or 'claude-sonnet-4-5-…' / 'claude-opus-4-6-v1' (family first).
# The minor part is 1-2 digits so a date suffix ('-20250514') is not read as one.
_CLAUDE_VERSION = re.compile(
    r"^anthropic\.claude-(?:"
    r"(?P<lead_major>\d+)(?:-(?P<lead_minor>\d{1,2}))?-[a-z]"
    r"|[a-z]+-(?P<major>\d+)(?:-(?P<minor>\d{1,2})(?!\d))?"
    r")"
)

# Records for ids without a curated table row. Deliberately conservative: a
# window or output limit that is too small only shrinks budgets, while one that
# is too large fails requests. Raise them via aws.bedrock.model_overrides.
_CLAUDE_ADAPTIVE_ONLY_DEFAULT = LanguageModelInfo(
    # Claude 4.7+ request shape: adaptive thinking only, no sampling params.
    context_window_size=200000,
    max_output_tokens=64000,
    supports_prompt_caching=True,
    supports_thinking=True,
    adaptive_thinking_only=True,
    supports_sampling_params=False,
    supports_count_tokens=False,
)
_CLAUDE_ADAPTIVE_DEFAULT = LanguageModelInfo(
    # Claude 4.6 request shape: opt-in adaptive thinking, sampling allowed.
    context_window_size=200000,
    max_output_tokens=64000,
    supports_prompt_caching=True,
    supports_thinking=True,
    supports_adaptive_thinking=True,
    supports_count_tokens=False,
)
_CLAUDE_BUDGET_THINKING_DEFAULT = LanguageModelInfo(
    # Claude 3.7 - 4.5 request shape: opt-in budget_tokens thinking.
    context_window_size=200000,
    max_output_tokens=64000,
    supports_prompt_caching=True,
    supports_thinking=True,
    supports_count_tokens=False,
)
_CLAUDE_LEGACY_DEFAULT = LanguageModelInfo(
    context_window_size=200000,
    max_output_tokens=4096,
    supports_count_tokens=False,
)
_OPENAI_DEFAULT = LanguageModelInfo(
    # GPT on Bedrock: Converse + reasoning.effort, no sampling params, implicit
    # caching only (explicit cache markers are rejected on Converse).
    provider="openai",
    context_window_size=128000,
    max_output_tokens=32000,
    supports_thinking=True,
    supports_sampling_params=False,
    supports_count_tokens=False,
)
_UNKNOWN_DEFAULT = LanguageModelInfo(
    provider="other",
    context_window_size=32000,
    max_output_tokens=4096,
    supports_sampling_params=False,
    supports_count_tokens=False,
)

_warned_uncurated: set[str] = set()
_warned_lock = threading.Lock()


def base_model_id(model_id: str) -> str:
    """``model_id`` without a cross-region inference-profile prefix."""
    for prefix in _PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id


def _family_default(model_id: str) -> LanguageModelInfo:
    """Provider-family record for an id with no curated table row."""
    if model_id.startswith("openai.gpt-"):
        return _OPENAI_DEFAULT
    if not model_id.startswith("anthropic.claude-"):
        return _UNKNOWN_DEFAULT
    match = _CLAUDE_VERSION.match(model_id)
    if match is None:
        return _CLAUDE_LEGACY_DEFAULT
    major = int(match["lead_major"] or match["major"])
    minor = int(match["lead_minor"] or match["minor"] or 0)
    if (major, minor) >= (4, 7):
        return _CLAUDE_ADAPTIVE_ONLY_DEFAULT
    if (major, minor) >= (4, 6):
        return _CLAUDE_ADAPTIVE_DEFAULT
    if (major, minor) >= (3, 7):
        return _CLAUDE_BUDGET_THINKING_DEFAULT
    return _CLAUDE_LEGACY_DEFAULT


def _warn_uncurated(model_id: str, info: LanguageModelInfo) -> None:
    with _warned_lock:
        if model_id in _warned_uncurated:
            return
        _warned_uncurated.add(model_id)
    if info is _UNKNOWN_DEFAULT:
        logger.warning(
            "Model '%s' has no capability record and no known provider family; "
            "using conservative defaults (Converse, no reasoning or sampling "
            "params, %d-token window, %d-token output). Set "
            "aws.bedrock.model_overrides['%s'] to describe it.",
            model_id,
            info.context_window_size,
            info.max_output_tokens,
            model_id,
        )
    else:
        logger.warning(
            "Model '%s' has no capability record; using %s family defaults "
            "(%d-token window, %d-token output). Set "
            "aws.bedrock.model_overrides['%s'] if they do not fit.",
            model_id,
            info.provider,
            info.context_window_size,
            info.max_output_tokens,
            model_id,
        )


def get_language_model_info(
    model_id: str, overrides: dict[str, dict[str, Any]] | None = None
) -> LanguageModelInfo:
    """Capability record for any Bedrock language-model id.

    Resolution order: the curated table row for the base id (profile prefix
    stripped), else the provider-family default by id prefix (warned once per
    id), then any ``aws.bedrock.model_overrides`` entry for the id on top.
    Public so callers that need a model's limits without constructing a factory
    (which opens a boto client) don't reach into the private table.
    """
    model_id = str(model_id)
    base_id = base_model_id(model_id)
    override = (overrides or {}).get(model_id) or (overrides or {}).get(base_id)
    info = _LANGUAGE_MODEL_INFO.get(base_id)
    if info is None:
        info = _family_default(base_id)
        if not override:
            _warn_uncurated(base_id, info)
    if not override:
        return info
    try:
        return LanguageModelInfo.model_validate({**info.model_dump(), **override})
    except ValidationError as e:
        raise LanguageModelError(
            f"Invalid aws.bedrock.model_overrides entry for '{model_id}': {e}"
        ) from e


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
