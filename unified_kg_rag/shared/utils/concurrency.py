# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thread pools that keep the submitter's ``contextvars``."""

from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """A ``ThreadPoolExecutor`` whose tasks run in a copy of the caller's context.

    A plain pool thread starts from an empty context, so the log context bound
    with ``structlog.contextvars`` (``pipeline_id``, ``stage``, ``query_id``)
    and any trace context are missing from what the task logs or calls. Each
    ``submit`` (and so ``map``) copies the submitting thread's context, as
    ``asyncio.to_thread`` does.

    Process pools stay plain ``ProcessPoolExecutor``s: a ``Context`` cannot be
    pickled to another process, and a worker process has its own logging and
    contextvars anyway.
    """

    def submit(
        self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_T]:
        context = contextvars.copy_context()
        return super().submit(context.run, functools.partial(fn, *args, **kwargs))
