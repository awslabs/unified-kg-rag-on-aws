# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One way to warn of a deprecated API: attributed to the caller's code."""

import sys
import warnings
from pathlib import Path

import pydantic

# The code a deprecation is never attributed to: this package's and
# pydantic's (a validator runs under pydantic's own frames).
_INTERNAL = tuple(
    f"{path}/"
    for path in (
        Path(__file__).resolve().parents[2],
        Path(pydantic.__file__).resolve().parent,
    )
)


def warn_deprecated(message: str) -> None:
    """Warn with ``message`` (a ``DeprecationWarning``) at the first frame
    outside this package and pydantic: the caller's line, which Python's
    default filters show when it is ``__main__``'s (a ``stacklevel`` counted
    from a validator named a pydantic frame, which they hide)."""
    frame = sys._getframe(1)
    level = 2
    while frame.f_back is not None and frame.f_code.co_filename.startswith(_INTERNAL):
        frame = frame.f_back
        level += 1
    warnings.warn(message, DeprecationWarning, stacklevel=level)
