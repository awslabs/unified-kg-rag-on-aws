# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic inputs for the GraphRAG and LightRAG fidelity tests, and the
originals' answers on them (``tests/fixtures/graphrag_lightrag/README.md``).

The inputs are plain Python draws (``raptor_fidelity.integer``, SHA-256), so
they come out alike wherever they are made. The originals' functions are
executed from their source (a checkout named by ``GRAPHRAG_SRC`` or
``LIGHTRAG_SRC``), never installed or vendored:

- LightRAG ``utils.pick_by_weighted_polling``, on chunk lineages;
- MS GraphRAG ``graph_context/sort_context.sort_context``, on community
  graphs, read back as the order its context lists the entities and
  relationships in;
- MS GraphRAG ``graph_context/context_builder._prepare_reports_at_level``,
  the relationships it hands ``sort_context`` for each community.

Record the answers again (after bumping a pin in
``.github/workflows/fidelity.yml``) with::

    GRAPHRAG_SRC=<checkout> LIGHTRAG_SRC=<checkout> \\
        uv run --extra eval python -m tests.fixtures.graphrag_lightrag_fidelity
"""

from __future__ import annotations

import ast
import io
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.fixtures.raptor_fidelity import integer

RECORD = "graphrag-lightrag-fidelity-v1"
FIXTURE = Path(__file__).parent / "graphrag_lightrag" / "upstream.json"


def polling_cases() -> list[dict[str, Any]]:
    """60 weighted-polling inputs: 1-12 items, each citing 0-8 chunks (an
    empty lineage included), 1-8 chunks for the first item, 1 or 2 for the
    last."""
    cases = []
    for c in range(60):
        name = f"polling-{c}"
        items = integer(1, 13, RECORD, name, "items")
        lineages = [
            [f"c{i}-{j}" for j in range(integer(0, 9, RECORD, name, "item", i))]
            for i in range(items)
        ]
        cases.append(
            {
                "case": name,
                "max_related_chunks": integer(1, 9, RECORD, name, "max"),
                "min_related_chunks": integer(1, 3, RECORD, name, "min"),
                "lineages": lineages,
            }
        )
    return cases


def report_cases() -> list[dict[str, Any]]:
    """30 communities of 3-14 entities (``E00``...), each pair linked with
    probability 1/3 by one relationship (``R00``... in the order upstream
    numbers them, a random direction); an entity may have none."""
    cases = []
    for c in range(30):
        name = f"report-{c}"
        size = integer(3, 15, RECORD, name, "size")
        edges = []
        for a in range(size):
            for b in range(a + 1, size):
                if integer(0, 3, RECORD, name, "edge", a, b) == 0:
                    flip = integer(0, 2, RECORD, name, "direction", a, b)
                    edges.append([b, a] if flip else [a, b])
        cases.append({"case": name, "size": size, "edges": edges})
    return cases


def degrees(case: dict[str, Any]) -> list[int]:
    """Each entity's degree in the community graph (pairs linked)."""
    counts = [0] * case["size"]
    for source, target in case["edges"]:
        counts[source] += 1
        counts[target] += 1
    return counts


def _functions(path: Path, names: tuple[str, ...], namespace: dict[str, Any]) -> None:
    """Execute the top-level functions ``names`` of the original's ``path``
    in ``namespace`` (the original's own code, MIT, run here only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    picked: list[ast.stmt] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if len(picked) != len(names):
        raise LookupError(f"{path}: the original's functions moved ({names})")
    code = compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec")
    exec(code, namespace)  # noqa: S102  # nosec B102


def lightrag_polling(root: Path) -> Callable[..., list[str]]:
    """LightRAG's ``pick_by_weighted_polling`` from the checkout ``root``."""
    namespace: dict[str, Any] = {}
    _functions(root / "lightrag" / "utils.py", ("pick_by_weighted_polling",), namespace)
    function: Callable[..., list[str]] = namespace["pick_by_weighted_polling"]
    return function


def _graphrag_package(root: Path) -> Path:
    return root / "packages" / "graphrag" / "graphrag"


def _schemas(package: Path) -> SimpleNamespace:
    namespace: dict[str, Any] = {}
    source = (package / "data_model" / "schemas.py").read_text(encoding="utf-8")
    exec(compile(source, "schemas.py", "exec"), namespace)  # noqa: S102  # nosec B102
    return SimpleNamespace(**{k: v for k, v in namespace.items() if k.isupper()})


def graphrag_report_context(root: Path) -> SimpleNamespace:
    """MS GraphRAG's ``sort_context`` and ``_prepare_reports_at_level``
    (its call of ``parallel_sort_context_batch`` captured) from ``root``."""
    import pandas as pd

    package = _graphrag_package(root)
    schemas = _schemas(package)
    folder = (
        package / "index" / "operations" / "summarize_communities" / "graph_context"
    )
    captured: dict[str, Any] = {}

    def parallel_sort_context_batch(community_df: Any, **_: Any) -> Any:
        captured["df"] = community_df
        return community_df

    namespace: dict[str, Any] = {
        "pd": pd,
        "schemas": schemas,
        "Tokenizer": object,
        "logger": logging.getLogger(__name__),
        "parallel_sort_context_batch": parallel_sort_context_batch,
    }
    _functions(folder / "sort_context.py", ("sort_context",), namespace)
    _functions(folder / "context_builder.py", ("_prepare_reports_at_level",), namespace)
    return SimpleNamespace(
        pd=pd,
        schemas=schemas,
        sort_context=namespace["sort_context"],
        prepare=namespace["_prepare_reports_at_level"],
        captured=captured,
    )


def _sorted_order(upstream: SimpleNamespace, case: dict[str, Any]) -> dict[str, Any]:
    """The entity and relationship numbers in the order ``sort_context``
    lists them, given every relationship of the community, no token cap."""
    s, pd = upstream.schemas, upstream.pd
    degree = degrees(case)
    records = []
    for i in range(case["size"]):
        edges = [
            {
                s.SHORT_ID: j,
                s.EDGE_SOURCE: f"E{source:02d}",
                s.EDGE_TARGET: f"E{target:02d}",
                s.DESCRIPTION: f"R{j:02d}",
                s.EDGE_DEGREE: degree[source] + degree[target],
            }
            for j, (source, target) in enumerate(case["edges"])
            if source == i
        ]
        records.append(
            {
                s.TITLE: f"E{i:02d}",
                s.NODE_DETAILS: {s.SHORT_ID: i, s.TITLE: f"E{i:02d}"},
                s.EDGE_DETAILS: edges,
            }
        )
    text = upstream.sort_context(records, tokenizer=None)
    blocks = [block.strip() for block in text.split("\n\n")]
    sections = {
        block.split("\n", 1)[0].strip("-"): block.split("\n", 1)[1]
        for block in blocks
        if block.startswith("-----")
    }

    def numbers(label: str) -> list[int]:
        if label not in sections:
            return []
        frame = pd.read_csv(io.StringIO(sections[label]))
        return [int(v) for v in frame[s.SHORT_ID]]

    return {"entities": numbers("Entities"), "relationships": numbers("Relationships")}


def _handed_edges(upstream: SimpleNamespace, case: dict[str, Any]) -> list[int]:
    """The relationship numbers ``_prepare_reports_at_level`` hands
    ``sort_context`` for the community (all of it one community, level 0)."""
    s, pd = upstream.schemas, upstream.pd
    degree = degrees(case)
    nodes = pd.DataFrame(
        [
            {
                s.TITLE: f"E{i:02d}",
                s.COMMUNITY_ID: 0,
                s.COMMUNITY_LEVEL: 0,
                s.NODE_DEGREE: degree[i],
                s.NODE_DETAILS: {s.SHORT_ID: i, s.TITLE: f"E{i:02d}"},
            }
            for i in range(case["size"])
        ]
    )
    edges = pd.DataFrame(
        [
            {
                s.SHORT_ID: j,
                s.EDGE_SOURCE: f"E{source:02d}",
                s.EDGE_TARGET: f"E{target:02d}",
                s.DESCRIPTION: f"R{j:02d}",
                s.EDGE_DEGREE: degree[source] + degree[target],
            }
            for j, (source, target) in enumerate(case["edges"])
        ],
        columns=[
            s.SHORT_ID,
            s.EDGE_SOURCE,
            s.EDGE_TARGET,
            s.DESCRIPTION,
            s.EDGE_DEGREE,
        ],
    )
    upstream.prepare(nodes, edges, None, None, 0, 16_000)
    (contexts,) = upstream.captured["df"][s.ALL_CONTEXT]
    return sorted(
        {int(e[s.SHORT_ID]) for record in contexts for e in record[s.EDGE_DETAILS]}
    )


def record(graphrag_root: Path, lightrag_root: Path) -> dict[str, Any]:
    """The originals' answers on every case."""
    polling = lightrag_polling(lightrag_root)
    report = graphrag_report_context(graphrag_root)
    return {
        "source": (
            "github.com/HKUDS/LightRAG (MIT) and github.com/microsoft/graphrag "
            "(MIT) at the commits in .github/workflows/fidelity.yml"
        ),
        "polling": [
            {
                "case": case["case"],
                "selected": polling(
                    [{"sorted_chunks": lineage} for lineage in case["lineages"]],
                    case["max_related_chunks"],
                    case["min_related_chunks"],
                ),
            }
            for case in polling_cases()
        ],
        "sort_context": [
            {"case": case["case"], **_sorted_order(report, case)}
            for case in report_cases()
        ],
        "report_edges": [
            {"case": case["case"], "relationships": _handed_edges(report, case)}
            for case in report_cases()
        ],
    }


if __name__ == "__main__":
    answers = record(Path(os.environ["GRAPHRAG_SRC"]), Path(os.environ["LIGHTRAG_SRC"]))
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(answers, separators=(",", ":")) + "\n", "utf-8")
    sys.stdout.write(f"wrote {FIXTURE}\n")
