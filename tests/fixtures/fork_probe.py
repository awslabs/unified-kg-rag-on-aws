# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A lock another thread holds while a process pool starts its workers,
and a task that reports whether a worker finds it free (a forked worker
inherits it held, as a fork copies the memory but not the holding thread)."""

from __future__ import annotations

import threading

LOCK = threading.Lock()


def lock_is_free() -> bool:
    acquired = LOCK.acquire(timeout=2)
    if acquired:
        LOCK.release()
    return acquired
