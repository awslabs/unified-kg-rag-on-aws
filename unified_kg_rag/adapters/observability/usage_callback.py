# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A LangChain callback handler that counts each chat model call's usage.

Attached by the model factories to every chat model, tracing on or off: each
call is counted in the open ``shared.usage`` scopes (see that module) when it
starts, at an estimate of its prompt's tokens, and settled when it ends with
the tokens the response reports (parsed as the tracing handler parses them,
``response_token_usage``: prompt-cache reads and writes apart too) and the
requests botocore resent inside it (``request_retries``). A response the
LLM response cache served counts as a cache hit instead, a failed call as
``failed_calls`` only; a call still running or
cancelled in flight, or one whose response reports no token usage (a
model or server that returns none), keeps its estimate
(``estimated_calls``; the output tokens of such a response are estimated
from its text). LangChain
reports no end or error for a call whose task was cancelled, so such a call
is let go when its task ends. Outside a scope a call costs one
context-variable read.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.observability.langchain_callback import (
    error_retry_attempts,
    response_retry_attempts,
    response_token_usage,
)
from unified_kg_rag.shared import usage


def _served_from_cache(response: LLMResult) -> bool:
    return any(
        (generation.generation_info or {}).get(usage.CACHED_RESPONSE_KEY)
        for generations in response.generations
        for generation in generations
    )


def _estimated_output_tokens(response: LLMResult) -> int:
    """The tokens of the response's text, estimated as a prompt's are."""
    return sum(
        estimate_token_count(generation.text)
        for generations in response.generations
        for generation in generations
    )


def _current_task() -> asyncio.Future[Any] | None:
    """The task running here, if any."""
    try:
        return asyncio.current_task()
    except RuntimeError:  # no running loop in this thread
        return None


class UsageCallbackHandler(BaseCallbackHandler):
    """Record each finished call of one model (``model_id``, built for
    ``role``) in the caller's usage scopes."""

    # In the caller's context, where its usage scope is open.
    run_inline = True

    # Calls kept unsettled per handler: a call whose task never ends, or
    # one started outside a task, may never settle, so past this the oldest
    # are let go (they stay counted as estimated).
    MAX_UNSETTLED: ClassVar[int] = 4096

    def __init__(self, model_id: str, *, role: str) -> None:
        self.model_id = model_id
        self.role = role
        # The calls started and not yet settled, by run id, oldest first.
        self._calls: OrderedDict[UUID, usage.LLMCall] = OrderedDict()
        # The task each call started in, and the callback that lets the call
        # go when that task ends (removed when the call settles).
        self._tasks: dict[UUID, tuple[asyncio.Future[Any], Callable[..., None]]] = {}
        self._lock = threading.Lock()

    def _start(self, run_id: UUID, texts: list[str]) -> None:
        if usage.current_meter() is None:
            return
        call = usage.start_llm_call(
            self.role,
            self.model_id,
            estimated_input_tokens=sum(map(estimate_token_count, texts)),
        )
        task = _current_task()
        with self._lock:
            self._calls[run_id] = call
            while len(self._calls) > self.MAX_UNSETTLED:
                gone, _ = self._calls.popitem(last=False)
                self._tasks.pop(gone, None)
            if task is not None:
                # A cancelled task's call gets no end or error from LangChain
                # (its gather is cancelled): let it go when the task ends.
                ended = partial(self._ended, run_id)
                self._tasks[run_id] = (task, ended)
        if task is not None:
            task.add_done_callback(ended)

    def _ended(self, run_id: UUID, task: asyncio.Future[Any]) -> None:
        with self._lock:
            self._tasks.pop(run_id, None)
            self._calls.pop(run_id, None)

    def _take(self, run_id: UUID) -> usage.LLMCall | None:
        with self._lock:
            call = self._calls.pop(run_id, None)
            started = self._tasks.pop(run_id, None)
        if started is not None:
            task, ended = started
            current = _current_task()
            # A task's callbacks are its own loop's to change; elsewhere the
            # callback stays and finds nothing to let go.
            if current is not None and current.get_loop() is task.get_loop():
                task.remove_done_callback(ended)
        return call

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
        retries = response_retry_attempts(response)
        if tokens is None:
            # No usage reported: settled at 0 tokens the call would read as
            # free, so it keeps its start estimate, its output is estimated
            # from the response text as the input was from the prompt, and
            # it counts as estimated.
            output_tokens = _estimated_output_tokens(response)
            if call is None:
                usage.record_llm_call(
                    self.role,
                    self.model_id,
                    input_tokens=0,
                    output_tokens=output_tokens,
                    request_retries=retries,
                    estimated=True,
                )
            else:
                call.finish_estimated(
                    output_tokens=output_tokens, request_retries=retries
                )
            return
        counts: dict[str, Any] = {
            "input_tokens": tokens.input_tokens,
            "output_tokens": tokens.output_tokens,
            "cache_read_input_tokens": tokens.cache_read,
            "cache_write_input_tokens": tokens.cache_creation,
            "request_retries": retries,
        }
        if call is None:
            usage.record_llm_call(self.role, self.model_id, **counts)
        else:
            call.finish(**counts)

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
            call.fail(request_retries=error_retry_attempts(error))
