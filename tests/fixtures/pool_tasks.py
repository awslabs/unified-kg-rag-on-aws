# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slow stand-ins for the tasks the ingestion process pools run.

They live in a module that imports nothing of the package, so a worker
started with ``forkserver`` or ``spawn`` (which unpickles the task by
reference, importing its module) starts in a fraction of a second.
"""

from __future__ import annotations

import time
from typing import Any


def slow_match(name: str) -> list[tuple[str, float]]:
    time.sleep(0.2)
    return []


def slow_minhash(item: tuple[int, str], **_: Any) -> None:
    time.sleep(0.2)
    return None
