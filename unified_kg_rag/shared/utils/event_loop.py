# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-loop setup for the blocking I/O on the query path."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

_T = TypeVar("_T")


def configure_event_loop(loop: asyncio.AbstractEventLoop, io_workers: int) -> None:
    """Size ``loop``'s default executor for blocking I/O.

    LangChain runs a chat model's ``ainvoke`` in the loop's default executor
    when the model has no native async path, as ``ChatBedrockConverse`` does
    not; Neptune traversals and reranking are offloaded there too. Python sizes
    that executor ``min(32, CPUs + 4)``, six threads on a 2-vCPU task, so no
    more than six Bedrock calls are in flight however many queries run
    concurrently. Call this once per loop before the first query, e.g. in an
    async web server's startup hook.
    """
    loop.set_default_executor(
        ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="graphrag-io")
    )


def run(main: Coroutine[Any, Any, _T], *, io_workers: int) -> _T:
    """``asyncio.run`` with the default executor sized by :func:`configure_event_loop`."""

    async def _main() -> _T:
        configure_event_loop(asyncio.get_running_loop(), io_workers)
        return await main

    return asyncio.run(_main())
