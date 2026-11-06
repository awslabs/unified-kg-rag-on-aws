# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RAPTOR's collapsed-tree selection, the reference the fidelity tests
compare the ``tree`` strategy with (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in);
not a product mode."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def collapsed_tree_reference(
    query_vector: Sequence[float],
    candidates: Sequence[tuple[str, Sequence[float], int]],
    *,
    top_k: int = 10,
    max_tokens: int = 3500,
) -> list[str]:
    """RAPTOR's collapsed-tree selection, for comparison only (not a product
    mode): ``candidates`` (id, vector, tokens) by cosine distance to the
    query, the first ``top_k`` taken in order until one would pass
    ``max_tokens``."""
    query = np.asarray(query_vector, dtype=np.float64)
    query = query / (np.linalg.norm(query) or 1.0)

    def distance(vector: Sequence[float]) -> float:
        v = np.asarray(vector, dtype=np.float64)
        return 1.0 - float(query @ v) / (float(np.linalg.norm(v)) or 1.0)

    ranked = sorted(range(len(candidates)), key=lambda i: distance(candidates[i][1]))
    selected: list[str] = []
    used = 0
    for index in ranked[:top_k]:
        node, _, tokens = candidates[index]
        if used + tokens > max_tokens:
            break
        selected.append(node)
        used += tokens
    return selected
