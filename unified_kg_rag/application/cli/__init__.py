# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CLIs. Nothing loads on import, so a console script's entry point
(:mod:`.launch`) runs before the CLI it starts pulls in its dependencies."""

from __future__ import annotations

from typing import Any

__all__ = ["EvaluationRunner"]


def __getattr__(name: str) -> Any:
    if name == "EvaluationRunner":
        from .run_evaluation import EvaluationRunner

        return EvaluationRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
