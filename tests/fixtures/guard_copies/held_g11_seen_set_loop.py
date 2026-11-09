# guard: G11
# as: adapters/search_strategies/drift_search.py
# expect: copy
from typing import Any


def _mut(actions: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    lineage: list[str] = []
    for action in actions:
        for tid in action["text_unit_ids"]:
            if tid not in seen:
                seen.add(tid)
                lineage.append(tid)
    return lineage
