# guard: G9
# as: adapters/search_strategies/structured_search.py
# expect: clean
from typing import Any

from ...domain.retrieval.filters import RANGE_BOUNDS, bound_holds


def _mut(op: str, left: Any, right: Any) -> bool:
    if op in RANGE_BOUNDS:
        return bound_holds(op, left, right)
    return left == right
