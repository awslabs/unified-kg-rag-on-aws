# guard: G1
# as: evaluation/report.py
# expect: copy
from typing import Any

from unified_kg_rag.domain.retrieval import degradation as deg

_PARTIAL = deg.UNRESOLVED_KEY


def _mut(rows: list[dict[str, Any]]) -> dict[str, int]:
    partial = sum(1 for row in rows if "degraded" in (row.get("metadata") or {}))
    unresolved = sum(len(row["metadata"].get(_PARTIAL, ())) for row in rows)
    return {"partial": partial, "unresolved": unresolved}
