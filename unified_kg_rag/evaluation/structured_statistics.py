# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the structured strategy did over a run's answers
(``EvaluationSummary.structured_statistics``; the comparison's per arm)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from unified_kg_rag.domain.models import EvaluationResult

# The fallback reasons of a plan that never reached the graph store.
_BEFORE_STORE = frozenset(
    {"no_schema", "planner_error", "unreadable_plan", "refused_plan", "unanswerable"}
)


# The strategy's RunRecord key.
_KEY = "structured"


def structured_statistics(results: Iterable[EvaluationResult]) -> dict[str, Any]:
    """Over the answers with a structured run record: their count, per
    status and fallback reason, and the rates of a valid plan (one that
    reached the graph store), a repair call, and of the answered ones a
    truncated and an exact answer; ``{}`` when none has a record."""
    records = [r.strategy_records[_KEY] for r in results if _KEY in r.strategy_records]
    if not records:
        return {}
    count = len(records)
    reasons = Counter(
        str(r.get("reason")) for r in records if r.get("status") != "answered"
    )
    valid = sum(
        1
        for r in records
        if r.get("status") == "answered" or r.get("reason") not in _BEFORE_STORE
    )
    answered = [r for r in records if r.get("status") == "answered"]

    def share(found: int, total: int) -> float:
        return found / total if total else 0.0

    return {
        "queries": count,
        "status": dict(sorted(Counter(str(r.get("status")) for r in records).items())),
        "fallback_reasons": dict(sorted(reasons.items())),
        "valid_plan_rate": share(valid, count),
        "repair_rate": share(sum(1 for r in records if r.get("repairs")), count),
        "truncated_rate": share(
            sum(1 for r in answered if r.get("truncated")), len(answered)
        ),
        "exact_rate": share(sum(1 for r in answered if r.get("exact")), len(answered)),
    }
