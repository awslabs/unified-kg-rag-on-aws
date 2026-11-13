# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two ``run-describe --include-text`` JSON Lines exports: the
convergence check after a chaos run (docs/guides/load-testing.md).

Two namespaces converge when they hold the same records of every kind (by
``kind`` and ``id``) with the same content once the keys a rebuild may
change without changing what the index says (:data:`VOLATILE`: build times,
run ids) are removed. Lists keep their order. The export ends with the graph
store's census (kind ``graph``: per-type counts and digests of its vertices
and edges), compared like any record under either tolerance, so a graph
store that lost writes the vector store kept does not converge.

A run resumed after a crash must converge exactly. A clean incremental run
after one that failed some documents re-indexes them as a delta, so it
converges with a full build only up to what incremental indexing documents
(docs/guides/incremental.md, "Rebuild signals"): it clusters only the delta
and appends its communities, a merged description keeps the stored summary
and appends the delta's lines, and lists of ids may come in another order.
``--incremental`` (:data:`INCREMENTAL`) compares within that: communities
and community memberships, descriptions (with their embedding and the
record of the summary they were made from) are left out, and lists of ids
are compared as multisets.

Usage::

    uv run python scripts/loadtest/compare_exports.py full.jsonl resumed.jsonl
    uv run python scripts/loadtest/compare_exports.py full.jsonl recovered.jsonl --incremental
    # exit 0: converged; 1: not (the differences are listed); 2: unreadable
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keys whose value differs between two builds of the same corpus.
VOLATILE: frozenset[str] = frozenset(
    {
        "built_at",
        "created_at",
        "updated_at",
        "indexed_at",
        "last_seen",
        "timestamp",
        "generation_timestamp",
        "pipeline_id",
        "run_id",
    }
)


@dataclass(frozen=True)
class Tolerance:
    """What a comparison leaves out: record kinds, keys (a top-level key, or
    a dotted path into a nested mapping) and list order."""

    unordered: bool = False
    ignore_kinds: frozenset[str] = frozenset()
    ignore_keys: frozenset[str] = frozenset()


EXACT = Tolerance()
# A clean run after a partly failed one (see the module docstring).
INCREMENTAL = Tolerance(
    unordered=True,
    ignore_kinds=frozenset({"community"}),
    ignore_keys=frozenset(
        {
            "community_ids",
            "description",
            "embedding_digests.description_embedding",
            "attributes.summarized_from",
            "attributes.summarized_with",
        }
    ),
)


def scrub(value: Any, tolerance: Tolerance = EXACT, path: str = "") -> Any:
    """``value`` without :data:`VOLATILE` keys at any depth and without the
    tolerance's ignored keys; with ``unordered``, a list of plain values
    sorted."""
    if isinstance(value, Mapping):
        return {
            k: scrub(v, tolerance, f"{path}{k}.")
            for k, v in sorted(value.items())
            if k not in VOLATILE and f"{path}{k}" not in tolerance.ignore_keys
        }
    if isinstance(value, list):
        unordered = tolerance.unordered
        items = [scrub(v, tolerance, path) for v in value]
        if unordered and all(
            isinstance(v, (str, int, float, bool)) or v is None for v in items
        ):
            return sorted(items, key=lambda v: (type(v).__name__, str(v)))
        return items
    return value


def index_records(lines: Iterable[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """``kind -> id -> record`` of an export's lines (blank lines skipped).
    A record without a kind or an id, or an id twice in one kind, is an
    error: the comparison would hide it."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        kind, rid = record.get("kind"), record.get("id")
        if not kind or rid is None:
            raise ValueError(f"line {number}: a record without a kind or an id")
        records = out.setdefault(str(kind), {})
        if str(rid) in records:
            raise ValueError(f"line {number}: {kind} {rid} appears twice")
        records[str(rid)] = record
    return out


@dataclass
class KindDiff:
    """How one kind of record differs between the exports."""

    kind: str
    count_a: int
    count_b: int
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    # id -> the keys whose (scrubbed) values differ
    changed: dict[str, list[str]] = field(default_factory=dict)

    @property
    def same(self) -> bool:
        return not (self.only_a or self.only_b or self.changed)

    def describe(self, limit: int = 3) -> str:
        status = "OK" if self.same else "DIFF"
        text = (
            f"{self.kind}: A={self.count_a} B={self.count_b} "
            f"only_a={len(self.only_a)} only_b={len(self.only_b)} "
            f"changed={len(self.changed)} {status}"
        )
        for rid in self.only_a[:limit]:
            text += f"\n  only in A: {rid}"
        for rid in self.only_b[:limit]:
            text += f"\n  only in B: {rid}"
        for rid, keys in list(self.changed.items())[:limit]:
            text += f"\n  changed: {rid} {keys}"
        return text


def compare(
    a: Mapping[str, Mapping[str, Mapping[str, Any]]],
    b: Mapping[str, Mapping[str, Mapping[str, Any]]],
    tolerance: Tolerance = EXACT,
) -> list[KindDiff]:
    """One :class:`KindDiff` per kind either export holds and the tolerance
    does not leave out, sorted by kind."""
    diffs: list[KindDiff] = []
    for kind in sorted((set(a) | set(b)) - tolerance.ignore_kinds):
        ra, rb = a.get(kind, {}), b.get(kind, {})
        diff = KindDiff(
            kind,
            len(ra),
            len(rb),
            only_a=sorted(set(ra) - set(rb)),
            only_b=sorted(set(rb) - set(ra)),
        )
        for rid in sorted(set(ra) & set(rb)):
            x = scrub(ra[rid], tolerance)
            y = scrub(rb[rid], tolerance)
            keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            if keys:
                diff.changed[rid] = keys
        diffs.append(diff)
    return diffs


def compare_files(a: Path, b: Path, tolerance: Tolerance = EXACT) -> list[KindDiff]:
    with a.open(encoding="utf-8") as fa, b.open(encoding="utf-8") as fb:
        return compare(index_records(fa), index_records(fb), tolerance)


def converged(diffs: list[KindDiff]) -> bool:
    """Whether the comparison found records and no difference (two empty
    exports do not converge: nothing was built)."""
    return bool(diffs) and all(d.same for d in diffs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="B is a clean run after a partly failed one: compare within what "
        "incremental indexing documents (no communities or descriptions, "
        "lists as multisets)",
    )
    args = parser.parse_args(argv)
    try:
        diffs = compare_files(
            args.a, args.b, INCREMENTAL if args.incremental else EXACT
        )
    except (OSError, ValueError) as e:
        print(f"cannot compare: {e}", file=sys.stderr)
        return 2
    for diff in diffs:
        print(diff.describe())
    same = converged(diffs)
    print("CONVERGED" if same else "NOT CONVERGED")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
