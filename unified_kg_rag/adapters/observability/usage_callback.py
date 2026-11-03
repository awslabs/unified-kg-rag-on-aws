# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A LangChain callback handler that counts each chat model call's usage.

Attached by the model factories to every chat model, tracing on or off: each
finished call is recorded in the open ``shared.usage`` scopes (see that
module) with the tokens the response reports (parsed as the tracing handler
parses them, ``response_token_usage``). A response the LLM response cache
served counts as a cache hit instead. Outside a scope a call costs one
context-variable read.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from unified_kg_rag.adapters.observability.langchain_callback import (
    response_token_usage,
)
from unified_kg_rag.shared import usage


def _served_from_cache(response: LLMResult) -> bool:
    return any(
        (generation.generation_info or {}).get(usage.CACHED_RESPONSE_KEY)
        for generations in response.generations
        for generation in generations
    )


class UsageCallbackHandler(BaseCallbackHandler):
    """Record each finished call of one model (``model_id``, built for
    ``role``) in the caller's usage scopes."""

    # In the caller's context, where its usage scope is open.
    run_inline = True

    def __init__(self, model_id: str, *, role: str) -> None:
        self.model_id = model_id
        self.role = role

    def _record(self, response: LLMResult) -> None:
        if usage.current_meter() is None:
            return
        if _served_from_cache(response):
            usage.record_cache_hit(self.role, self.model_id)
            return
        tokens = response_token_usage(response)
        usage.record_llm_call(
            self.role,
            self.model_id,
            input_tokens=tokens.input_tokens if tokens else 0,
            output_tokens=tokens.output_tokens if tokens else 0,
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self._record(response)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        # A stream the consumer stopped reading (GeneratorExit) was a call
        # all the same; a failed call returned no usage.
        response = kwargs.get("response")
        if isinstance(error, GeneratorExit) and isinstance(response, LLMResult):
            self._record(response)
