# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One way to warn of a deprecated API: attributed to the caller's code."""

import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import pydantic

# The code a deprecation is never attributed to: this package's and
# pydantic's (a validator runs under pydantic's own frames), compared by
# real path, so a symlinked checkout or venv matches too.
_INTERNAL = tuple(
    os.path.realpath(path) + os.sep
    for path in (
        Path(__file__).parents[2],
        Path(pydantic.__file__).parent,
    )
)


@lru_cache(maxsize=256)
def _internal(filename: str) -> bool:
    return os.path.realpath(filename).startswith(_INTERNAL)


def warn_deprecated(message: str) -> None:
    """Warn with ``message`` (a ``DeprecationWarning``) at the first frame
    outside this package and pydantic: the caller's line, which Python's
    default filters show when it is ``__main__``'s (a ``stacklevel`` counted
    from a validator named a pydantic frame, which they hide)."""
    frame = sys._getframe(1)
    level = 2
    while frame.f_back is not None and (
        _internal(frame.f_code.co_filename) or _bootstrap(frame)
    ):
        frame = frame.f_back
        # warnings.warn skips the import machinery's frames itself
        # (``from unified_kg_rag import <name>`` reads the name there
        # before the caller's line): only the others count.
        if not _bootstrap(frame):
            level += 1
    warnings.warn(message, DeprecationWarning, stacklevel=level)


def _bootstrap(frame: Any) -> bool:
    """Whether ``frame`` is the import machinery's, as ``warnings`` tells."""
    filename = frame.f_code.co_filename
    return "importlib" in filename and "_bootstrap" in filename
