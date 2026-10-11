# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The description-merge rule shared by extraction, resolution and merge."""

from collections.abc import Iterable


def merge_descriptions(descriptions: Iterable[str | None]) -> str | None:
    """Combine descriptions, dropping duplicate and blank lines (order kept).

    The one description-merge rule of every build path (extraction,
    full-build resolution, cross-run merge), so a full build and an
    incremental build of the same corpus converge on the same text.

    A stored description is the newline join of earlier merges, so the dedupe
    compares lines: comparing whole strings would append a re-applied delta's
    description again on every run.
    """
    lines = [
        line
        for part in descriptions
        if part
        for line in part.split("\n")
        if line.strip()
    ]
    if not lines:
        return None
    return "\n".join(dict.fromkeys(lines))
