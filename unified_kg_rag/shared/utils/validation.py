# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a Pydantic validation error is told to a user: each invalid key and
why, without the rejected values."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import ValidationError


def format_validation_errors(
    prefix: str, error: ValidationError, *, root: str | None = "<root>"
) -> str:
    """Each invalid key of ``error`` and why, ``;``-separated, without the
    rejected values (a misplaced secret, a password under the wrong key,
    would be echoed to the terminal and the logs).

    A key is named by ``prefix`` and its location, dot-joined
    (``schema_source.options.colour: Extra inputs are not permitted``). An
    error with no key at all (an empty ``prefix`` and location) is named
    ``root``, or by its message alone when ``root`` is None.
    """
    parts: list[str] = []
    for detail in error.errors(include_url=False, include_input=False):
        names = [str(part) for part in detail["loc"]]
        key = ".".join([prefix, *names] if prefix else names)
        if not key and root is None:
            parts.append(detail["msg"])
        else:
            parts.append(f"{key or root}: {detail['msg']}")
    return "; ".join(parts)
