# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Work a query starts that may outlive it.

A query can keep some of the work it started running past its end, on
purpose: auto's speculative extractions its route did not read run on, so
their answers fill the extraction cache. Such work must not end with the
query's per-query lifecycle state, so it is started detachable
(:func:`start_detachable`):

- it closes its own model responses (``shared.model_responses``): it runs
  under a scope of its own, which ends when it does, not when its query
  does, so the query closing what it left open closes nothing this work
  still reads;
- it stops with the query (its reader's stop or its deadline,
  ``shared.query_stop``) only until the query keeps it (:func:`detach`):
  until then it is the query's work and sends no model request after the
  query's stop; once kept, that stop is gone from it, and it is bounded by its own retry budget and timeouts
  and by whoever keeps it (a cancel at the query's deadline, or when its
  owner closes).

The rest of its context is the query's (its usage meter, its deadline, its
log and trace ids).
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from unified_kg_rag.shared.model_responses import closing_responses
from unified_kg_rag.shared.query_stop import detachable_context

_T = TypeVar("_T")

# The detach of each detachable task still attached to its query's stop.
_DETACHES: weakref.WeakKeyDictionary[asyncio.Future[Any], Callable[[], None]] = (
    weakref.WeakKeyDictionary()
)


def start_detachable(coroutine: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
    """Run ``coroutine`` as a task of this query that may outlive it: under
    a model-response scope of its own, stopped with the query until
    :func:`detach`."""
    context, release = detachable_context()
    task = asyncio.get_running_loop().create_task(
        _own_responses(coroutine), context=context
    )
    _DETACHES[task] = release
    return task


def detach(task: asyncio.Future[Any]) -> None:
    """``task`` (a :func:`start_detachable` one) is kept past its query: its
    query's stop no longer stops it. A no-op for any other future."""
    release = _DETACHES.pop(task, None)
    if release is not None:
        release()


async def _own_responses(coroutine: Coroutine[Any, Any, _T]) -> _T:
    async with closing_responses():
        return await coroutine
