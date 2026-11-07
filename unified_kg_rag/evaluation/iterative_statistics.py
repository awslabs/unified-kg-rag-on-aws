# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the iterative strategy did over a run's answers
(``EvaluationSummary.iterative_statistics``; the comparison's per arm)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from unified_kg_rag.domain.models import EvaluationResult

# The stop reason of a run that used every follow-up round it was allowed
# (search.iterative.max_rounds): a share of them says the cap, not the
# controller, ended the loop.
_AT_MAX_ROUNDS = "max_rounds"


def iterative_statistics(results: Iterable[EvaluationResult]) -> dict[str, Any]:
    """Over the answers with an iterative run record: their count, the
    follow-up rounds per answer (a histogram and the mean), the stop reasons,
    the share stopped at ``max_rounds`` and the degraded runs (a controller
    reply that did not parse, a failed round); ``{}`` when none has one."""
    records = [r.iterative for r in results if isinstance(r.iterative, dict)]
    if not records:
        return {}
    rounds = [int(r.get("rounds") or 0) for r in records]
    reasons = Counter(str(r.get("stop_reason")) for r in records)
    return {
        "queries": len(records),
        "rounds": {str(k): v for k, v in sorted(Counter(rounds).items())},
        "mean_rounds": sum(rounds) / len(rounds),
        "stop_reasons": dict(sorted(reasons.items())),
        "max_rounds_rate": reasons.get(_AT_MAX_ROUNDS, 0) / len(records),
        "degraded": sum(1 for r in records if r.get("degraded")),
    }
