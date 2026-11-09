# guard: G1
# as: evaluation/report.py
# expect: clean
from typing import Any

from unified_kg_rag.domain.retrieval.degradation import DegradationReport


def _mut(rows: list[dict[str, Any]]) -> int:
    """Count the answers that degraded, read through the owner's report."""
    partial = 0
    for row in rows:
        report = DegradationReport.of(row.get("metadata") or {})
        if report.degraded:
            partial += 1
    return partial
