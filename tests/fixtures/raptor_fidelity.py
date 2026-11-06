# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic inputs the RAPTOR fidelity tests share with the script that
recorded the original's outputs on them (``tests/fixtures/raptor/README``).

Plain Python (hashlib, math), so the same points come out in the original's
environment and in ours: Gaussian blobs from SHA-256-derived uniforms
(Box-Muller), no random number generator whose stream could change between
numpy versions.
"""

from __future__ import annotations

import hashlib
import math

RECORD = "raptor-fidelity-v1"


def _uniform(*parts: object) -> float:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    # (0, 1): never 0, so the logarithm is finite.
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2.0**64


def normal(*parts: object) -> float:
    """A standard normal draw named by ``parts``."""
    u1, u2 = _uniform(*parts, "u1"), _uniform(*parts, "u2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def integer(low: int, high: int, *parts: object) -> int:
    """An integer in ``[low, high)`` named by ``parts``."""
    return low + int(_uniform(*parts, "int") * (high - low))


def blobs(
    case: str, n: int, centers: int, spread: float, dims: int = 10
) -> list[list[float]]:
    """``n`` points round ``centers`` centers (point i at center i mod
    centers), each coordinate the center's plus ``spread`` times a normal
    draw, rounded to 8 decimals."""
    middles = [
        [3.0 * normal(RECORD, case, "center", c, d) for d in range(dims)]
        for c in range(centers)
    ]
    return [
        [
            round(middles[i % centers][d] + spread * normal(RECORD, case, i, d), 8)
            for d in range(dims)
        ]
        for i in range(n)
    ]
