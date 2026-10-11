# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token counting without a model API (``TokenCounterPort``).

``estimate`` uses the script-aware estimate the Bedrock counter falls back to
(and uses outright for Claude 5, whose CountTokens is unavailable), so local
and AWS budgets behave alike. ``tiktoken`` counts with a BPE encoding (the
``[local]`` extra); it is closer for GPT-style tokenizers but still not the
served model's own tokenizer, and it falls back to the estimate, with one
warning, when tiktoken or its encoding file cannot be loaded.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.local.embeddings import truncate_to_estimate
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

_encodings: dict[str, Any] = {}
_failed: set[str] = set()
_lock = threading.Lock()


def load_encoding(name: str) -> Any | None:
    """The tiktoken encoding ``name`` (cached), or None when unavailable."""
    with _lock:
        if name in _encodings:
            return _encodings[name]
        if name in _failed:
            return None
        try:
            import tiktoken

            encoding = tiktoken.get_encoding(name)
        except Exception as e:  # noqa: BLE001 - any failure means "estimate"
            _failed.add(name)
            logger.warning(
                "tiktoken encoding '%s' is unavailable (%s); counting tokens "
                "with the script-aware estimate instead.",
                name,
                e,
            )
            return None
        _encodings[name] = encoding
        return encoding


class LocalTokenCounter:
    """Counts and truncates by tokens for one (non-Bedrock) model."""

    def __init__(
        self,
        model_id: str,
        *,
        cache_maxsize: int = 1024,
        tokenizer: str = "estimate",
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.model_id = model_id
        self._encoding = (
            load_encoding(encoding_name) if tokenizer == "tiktoken" else None
        )

        @lru_cache(maxsize=cache_maxsize)
        def _cached(text: str) -> int:
            return self._count(text)

        self._cached = _cached

    def _count(self, text: str) -> int:
        if self._encoding is None:
            return estimate_token_count(text)
        return len(self._encoding.encode(text, disallowed_special=()))

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return int(self._cached(text))

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> tuple[str, int]:
        if not text:
            return text, 0
        count = self.count_tokens(text)
        if count <= max_tokens:
            return text, count
        if self._encoding is None:
            truncated = truncate_to_estimate(text, max_tokens)
        else:
            tokens = self._encoding.encode(text, disallowed_special=())
            truncated = self._encoding.decode(tokens[: max(max_tokens, 0)])
        return truncated, self.count_tokens(truncated)


def local_token_counter_factory(config: Config) -> Any:
    """``Providers`` token-counter factory over ``providers.llm`` settings."""
    settings = config.providers.llm

    def build(
        model_id: str, *, cache_maxsize: int = 1024, **_: Any
    ) -> LocalTokenCounter:
        return LocalTokenCounter(
            str(model_id),
            cache_maxsize=cache_maxsize,
            tokenizer=settings.tokenizer,
            encoding_name=settings.tiktoken_encoding,
        )

    return build
