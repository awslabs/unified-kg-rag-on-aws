# guard: G9
# as: adapters/search_strategies/structured_search.py
# expect: copy
from typing import Any


def _mut(op: str, left: Any, right: Any) -> bool:
    match op:
        case "lt":
            return left < right
        case "lte":
            return left <= right
        case "gt":
            return left > right
        case "gte":
            return left >= right
    return left == right
