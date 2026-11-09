# guard: G3
# as: application/cli/console/ask.py
# expect: copy
from typing import Any

from unified_kg_rag.domain.retrieval.degradation import DEGRADED_KEY, UNRESOLVED_KEY


def _mut(result: Any) -> dict[str, Any]:
    md = result.metadata or {}
    return {
        "partial": bool(md.get(DEGRADED_KEY)),
        "unresolved": len(md.get(UNRESOLVED_KEY) or []),
    }
