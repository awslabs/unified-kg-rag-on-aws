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
from typing import Any

from pydantic import ValidationError

from unified_kg_rag.domain.models import (
    EmbeddingModelId,
    LanguageModelId,
    RerankModelId,
)
from unified_kg_rag.domain.models.config import EFFORT_LEVELS
from unified_kg_rag.ports.model_factory import (
    EmbeddingModelInfo,
    LanguageModelInfo,
    RerankModelInfo,
)
from unified_kg_rag.shared import LanguageModelError, get_logger

logger = get_logger(__name__)


# Effort levels each model family accepts on bedrock-runtime, as reported by
# the service's own validation errors. Anthropic levels are sent as
# ``output_config.effort``; OpenAI levels as ``reasoning.effort``.
_ANTHROPIC_EFFORTS_ALL: frozenset[str] = EFFORT_LEVELS
# Claude 4.6 rejects 'xhigh' ("Input should be 'low', 'medium', 'high' or 'max'").
_ANTHROPIC_EFFORTS_NO_XHIGH: frozenset[str] = EFFORT_LEVELS - {"xhigh"}
# GPT also accepts 'none' (reasoning off); it is not exposed because the
# BedrockConfig effort fields (EffortLevel) only carry the shared levels.
_OPENAI_EFFORTS: frozenset[str] = EFFORT_LEVELS

# Claude models with the tokenizer introduced in Opus 4.7 (Opus 4.7/4.8/5/5.5,
# Sonnet 5/5.5, Haiku 5.5) count more tokens for the same text than older
# Claude: "roughly 1x-1.35x as many tokens" (Opus 4.7 migration notes) and
# "~30% more tokens" (Sonnet 5 vs 4.6; Haiku 5.5 vs 4.5). CountTokens rejects
# all of them on bedrock-runtime, so the local estimate is scaled by this.
_CLAUDE_4_7_TOKENIZER_MULTIPLIER = 1.3


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
    # 1M context / 128K output; adaptive thinking with effort low..max. All
    # reject temperature ("deprecated for this model") and budget_tokens
    # thinking; CountTokens rejects them on bedrock-runtime.
    LanguageModelId.CLAUDE_V5_5_SONNET: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        min_cache_tokens=512,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        token_estimate_multiplier=_CLAUDE_4_7_TOKENIZER_MULTIPLIER,
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
        token_estimate_multiplier=_CLAUDE_4_7_TOKENIZER_MULTIPLIER,
        supported_efforts=_ANTHROPIC_EFFORTS_ALL,
        supports_sampling_params=False,
        requires_inference_profile=True,
        supports_count_tokens=False,
    ),
    # Haiku 5.5 also thinks by default but, unlike Sonnet/Opus 5.5, accepts
    # thinking disabled; it takes no budget_tokens and its tokenizer counts
    # ~30% more tokens than Haiku 4.5 for the same text. It supports Converse
    # (so ConverseStream). langchain-aws 1.9.0 streams "claude-haiku-5"; the
    # 1.8.x the dependency range still allows does not, hence the explicit
    # supports_streaming.
    LanguageModelId.CLAUDE_V5_5_HAIKU: LanguageModelInfo(
        supports_streaming=True,
        context_window_size=1000000,
        max_output_tokens=128000,
        supports_prompt_caching=True,
        min_cache_tokens=512,
        supports_thinking=True,
        supports_1m_context_window=True,
        native_1m_context_window=True,
        adaptive_thinking_only=True,
        token_estimate_multiplier=_CLAUDE_4_7_TOKENIZER_MULTIPLIER,
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
        token_estimate_multiplier=_CLAUDE_4_7_TOKENIZER_MULTIPLIER,
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
        token_estimate_multiplier=_CLAUDE_4_7_TOKENIZER_MULTIPLIER,
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
        token_estimate_multiplier=_CLAUDE_4_7_TOKENIZER_MULTIPLIER,
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
        token_estimate_multiplier=_CLAUDE_4_7_TOKENIZER_MULTIPLIER,
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
    token_estimate_multiplier=_CLAUDE_4_7_TOKENIZER_MULTIPLIER,
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
