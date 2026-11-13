# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summary statistics every report shares one definition of."""

from __future__ import annotations

import math
from collections.abc import Sequence


def quantile(values: Sequence[float], q: float) -> float | None:
    """The ``q`` quantile (0..1) of ``values``, interpolated linearly between
    the closest ranks (numpy's default, as ``run-eval``'s latency quantiles
    read), or None for none."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * min(1.0, max(0.0, q))
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
