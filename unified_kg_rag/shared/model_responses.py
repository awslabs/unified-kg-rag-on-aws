# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model responses a query has open, closed when the query ends.

A streamed model call is a stack of generators (LangChain's steps, the
provider client's own), each reading the next with ``async for``, which
closes nothing it leaves: cancelled while that stack waits between two
chunks (a deadline, a first-chunk window or a reader's stop that lands
there), the stack is left suspended with the provider's HTTP response open,
holding its pooled connection until garbage collection finalizes the
generators, and a reference cycle can keep them. Bedrock's stream is read
on a worker thread, which reads on after the query is cancelled.

So the model clients register every response they open
(:func:`track_response`, from their HTTP clients' hooks) with the scope of
the query that opened it (:func:`closing_responses`), which closes each one
still open when it ends, however it ends. A response opened after its scope
ended (on a worker thread the query left) is closed at once, and a client
whose request such a thread would send refuses it first
(:func:`raise_if_scope_ended`). Outside a scope nothing is tracked.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from unified_kg_rag.shared.logging import get_logger

logger = get_logger(__name__)


class _OpenResponses:
    """The responses one scope has open: each has ``aclose()`` (an
    ``httpx.Response``, opened on the event loop), ``close()`` (closing it
    from a worker thread), or both (a Bedrock event stream)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: list[Any] = []
        self._ended = False

    def add(self, response: Any) -> bool:
        """Track ``response``; False once the scope ended."""
        with self._lock:
            if self._ended:
                return False
            # A response read to its end is closed already (httpx's
            # is_closed): drop those, so a long query keeps few.
            self._open = [r for r in self._open if not _is_closed(r)]
            self._open.append(response)
            return True

    @property
    def ended(self) -> bool:
        return self._ended

    def end(self) -> list[Any]:
        """End the scope: the responses still open."""
        with self._lock:
            self._ended = True
            left, self._open = self._open, []
        return [r for r in left if not _is_closed(r)]


_OPEN_RESPONSES: contextvars.ContextVar[_OpenResponses | None] = contextvars.ContextVar(
    "kg_rag_open_responses", default=None
)

# Closes started outside a scope's await (a response tracked after its scope
# ended, a scope cancelled again while it closed): kept until done.
_PENDING: set[asyncio.Future[None]] = set()


def _is_closed(response: Any) -> bool:
    return bool(getattr(response, "is_closed", False))


async def _close_all(responses: list[Any]) -> None:
    for response in responses:
        try:
            aclose = getattr(response, "aclose", None)
            if aclose is not None:
                await aclose()
            else:
                response.close()
        except Exception as e:  # noqa: BLE001 - closing is best effort
            logger.debug("Closing a model response failed: %s", e)


def _close_later(responses: list[Any]) -> asyncio.Future[None]:
    task = asyncio.ensure_future(_close_all(responses))
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)
    return task


def track_response(response: Any) -> None:
    """Close ``response`` when the scope of the query it is opened in ends
    (:func:`closing_responses`). Called by the model clients' HTTP hooks as
    a response's headers arrive, before its body is read, on the event loop
    or a worker thread; a no-op outside a scope."""
    scope = _OPEN_RESPONSES.get()
    if scope is None or scope.add(response):
        return
    # Its query has ended: no one reads it. Off the event loop (a worker
    # thread's response) it is closed there, on it by a task.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        on_loop = False
    else:
        on_loop = getattr(response, "aclose", None) is not None
    if on_loop:
        _close_later([response])
        return
    try:
        response.close()
    except Exception as e:  # noqa: BLE001 - closing is best effort
        logger.debug("Closing a model response failed: %s", e)


def raise_if_scope_ended() -> None:
    """Raise ``asyncio.CancelledError`` when the scope of the query this
    request is for ended (:func:`closing_responses`): a stop point before a
    model client sends a request, so a worker thread the query left sends
    none (no response anyone reads, and no billed call). A no-op outside a
    scope."""
    scope = _OPEN_RESPONSES.get()
    if scope is not None and scope.ended:
        raise asyncio.CancelledError


@asynccontextmanager
async def closing_responses() -> AsyncIterator[None]:
    """Close every model response opened in the block (and the calls it
    starts, on its tasks and worker threads) that is still open when the
    block ends, by an error or a cancellation too. The closing finishes on
    its own when the block's task is cancelled again while it closes (an
    anyio scope cancels every await)."""
    scope = _OpenResponses()
    token = _OPEN_RESPONSES.set(scope)
    try:
        yield
    finally:
        _OPEN_RESPONSES.reset(token)
        left = scope.end()
        if left:
            await asyncio.shield(_close_later(left))
