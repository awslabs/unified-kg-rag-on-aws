# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A chainable stand-in for a gremlin_python traversal."""

from __future__ import annotations

from typing import Any


class RecordingTraversal:
    """Records every step it receives into ``calls`` and returns itself, so
    any chain of steps builds. A step is recorded as its name, or with
    ``with_args`` as ``(name, args)``."""

    def __init__(self, calls: list[Any], *, with_args: bool = False) -> None:
        self._calls = calls
        self._with_args = with_args

    def __getattr__(self, name: str) -> Any:
        def _step(*args: Any, **kwargs: Any) -> RecordingTraversal:
            self._calls.append((name, args) if self._with_args else name)
            return self

        return _step
