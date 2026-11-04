# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A LangChain callback handler that counts each chat model call's usage.

Attached by the model factories to every chat model, tracing on or off: each
call is counted in the open ``shared.usage`` scopes (see that module) when it
starts, at an estimate of its prompt's tokens, and settled when it ends with
the tokens the response reports (parsed as the tracing handler parses them,
``response_token_usage``). A response the LLM response cache served counts
as a cache hit instead, a failed call not at all; a call still running or
cancelled in flight keeps its estimate (``estimated_calls``). Outside a scope a call costs one
context-variable read.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from typing import Any, ClassVar
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
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

    # Calls kept unsettled per handler: a cancelled call never settles, so
    # past this the oldest are let go (they stay counted as estimated).
    MAX_UNSETTLED: ClassVar[int] = 4096

    def __init__(self, model_id: str, *, role: str) -> None:
        self.model_id = model_id
        self.role = role
        # The calls started and not yet settled, by run id, oldest first.
        self._calls: OrderedDict[UUID, usage.LLMCall] = OrderedDict()
        self._lock = threading.Lock()

    def _start(self, run_id: UUID, texts: list[str]) -> None:
        if usage.current_meter() is None:
            return
        call = usage.start_llm_call(
            self.role,
            self.model_id,
            estimated_input_tokens=sum(map(estimate_token_count, texts)),
        )
        with self._lock:
            self._calls[run_id] = call
            while len(self._calls) > self.MAX_UNSETTLED:
                self._calls.popitem(last=False)

    def _take(self, run_id: UUID) -> usage.LLMCall | None:
        with self._lock:
            return self._calls.pop(run_id, None)

    def _record(self, run_id: UUID, response: LLMResult) -> None:
        call = self._take(run_id)
        if call is None and usage.current_meter() is None:
            return
        if _served_from_cache(response):
            if call is None:
                usage.record_cache_hit(self.role, self.model_id)
            else:
                call.served_from_cache()
            return
        tokens = response_token_usage(response)
        input_tokens = tokens.input_tokens if tokens else 0
        output_tokens = tokens.output_tokens if tokens else 0
        if call is None:
            usage.record_llm_call(
                self.role,
                self.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        else:
            call.finish(input_tokens=input_tokens, output_tokens=output_tokens)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id,
            [m.text for message_list in messages for m in message_list],
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, list(prompts))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self._record(run_id, response)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        # A stream the consumer stopped reading (GeneratorExit) was a call
        # all the same; a cancelled one keeps its estimate (the request was
        # sent); a failed call returned no usage.
        response = kwargs.get("response")
        if isinstance(error, GeneratorExit) and isinstance(response, LLMResult):
            self._record(run_id, response)
            return
        call = self._take(run_id)
        if call is not None and not isinstance(error, asyncio.CancelledError):
            call.fail()
