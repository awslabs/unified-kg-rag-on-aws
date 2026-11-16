# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One way to warn of a deprecated API: attributed to the caller's code."""

import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

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
    # Only this package's or pydantic's frames are passed: the import
    # machinery never runs between them and the caller (the root
    # ``__getattr__`` ignores importlib's read of a name, and no module of
    # this package warns while it is imported), and ``warnings.warn``
    # skips its frames itself.
    while frame.f_back is not None and _internal(frame.f_code.co_filename):
        frame = frame.f_back
        level += 1
    warnings.warn(message, DeprecationWarning, stacklevel=level)
