# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Work a query starts that may outlive it (``shared.query_work``).

Regression: the query's own model-response scope closed the responses of
auto's speculative extraction it kept running past its end, so the
extraction failed and never filled the extraction cache. A detachable task
closes its own responses, and once its query keeps it, its query's reader
stop no longer ends it.
"""

from __future__ import annotations

import asyncio

import pytest

from unified_kg_rag.shared.model_responses import closing_responses, track_response
from unified_kg_rag.shared.query_stop import (
    QueryStop,
    query_stop_scope,
    reader_stopped,
)
from unified_kg_rag.shared.query_work import detach, start_detachable

pytestmark = pytest.mark.unit


class _Response:
    """An HTTP response as the model clients' hooks track it."""

    def __init__(self) -> None:
        self.is_closed = False

    async def aclose(self) -> None:
        self.is_closed = True


async def _reads(response: _Response, go: asyncio.Event) -> bool:
    """Track ``response``, wait for ``go``, then whether it is still open."""
    track_response(response)
    await go.wait()
    return not response.is_closed


async def test_its_responses_outlive_the_query_that_started_it() -> None:
    response, go = _Response(), asyncio.Event()
    async with closing_responses():
        task = start_detachable(_reads(response, go))
        await asyncio.sleep(0)
    # The query's scope ended and closed nothing the task still reads.
    assert not response.is_closed
    go.set()
    assert await task is True
    # Its own scope closes what it left open, as it ends.
    assert response.is_closed


async def test_a_cancelled_one_closes_its_responses() -> None:
    response, go = _Response(), asyncio.Event()
    task = start_detachable(_reads(response, go))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert response.is_closed


async def _stopped_after(go: asyncio.Event) -> bool:
    await go.wait()
    return reader_stopped()


@pytest.mark.parametrize("detached", [False, True])
async def test_the_readers_stop_reaches_it_until_it_is_detached(
    detached: bool,
) -> None:
    stop, go = QueryStop(), asyncio.Event()
    with query_stop_scope(stop):
        task = start_detachable(_stopped_after(go))
    if detached:
        detach(task)
    stop.set()
    go.set()
    assert await task is (not detached)
    assert stop.is_set()


async def test_a_stop_set_before_it_starts_is_its_own() -> None:
    stop, go = QueryStop(), asyncio.Event()
    stop.set()
    with query_stop_scope(stop):
        task = start_detachable(_stopped_after(go))
    go.set()
    assert await task is True


async def test_without_a_stop_it_runs_and_detach_is_a_no_op() -> None:
    go = asyncio.Event()
    task = start_detachable(_stopped_after(go))
    detach(task)
    detach(asyncio.get_running_loop().create_future())
    go.set()
    assert await task is False
