# guard: G1
# as: adapters/search_strategies/global_search.py
# expect: copy
from typing import Any

from ...domain.retrieval.degradation import DEGRADED_ERROR_KEY, DEGRADED_KEY


def _mut(metadata: dict[str, Any], failures: list[tuple[str, Exception]]) -> None:
    entries = [
        {"step": "global_map", "reason": str(exc), "error_type": type(exc).__name__}
        for _, exc in failures
    ]
    metadata[DEGRADED_KEY] = entries
    if entries:
        metadata[DEGRADED_ERROR_KEY] = entries[0]["error_type"]
