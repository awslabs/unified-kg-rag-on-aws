# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic ``visualization_data.json`` exports for the Graph page's tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_export(
    path: Path, nodes: int, *, layout: dict[str, list[float]] | None = None
) -> Path:
    """A star-and-chain graph of ``nodes`` entities: ``n0`` is linked to
    every other, ``n<i>`` to ``n<i+1>`` (so the degrees differ)."""
    ids = [f"n{i}" for i in range(nodes)]
    edges = [(ids[0], ids[i]) for i in range(1, nodes)]
    edges += [(ids[i], ids[i + 1]) for i in range(1, nodes - 1, 2)]
    data: dict[str, Any] = {
        "nodes": [
            {"id": i, "attributes": {"name": f"Entity {i}", "type": "thing"}}
            for i in ids
        ],
        "edges": [
            {"source": a, "target": b, "attributes": {"id": f"{a}-{b}"}}
            for a, b in edges
        ],
        "layout": layout or {},
        "communities": {"hierarchy": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
