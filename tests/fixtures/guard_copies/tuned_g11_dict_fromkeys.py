# guard: G11
# as: adapters/search_strategies/global_search.py
# expect: copy
from typing import Any


def _mut(results: list[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            unit for r in results for unit in r.metadata.get("text_unit_ids", [])
        )
    )
