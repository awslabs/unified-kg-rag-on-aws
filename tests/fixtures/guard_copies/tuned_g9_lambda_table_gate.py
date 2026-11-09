# guard: G9
# as: evaluation/gates.py
# expect: copy
from typing import Any

_CHECKS = {
    "gte": lambda a, b: not a < b,
    "lte": lambda a, b: not a > b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: b > a,
}


def _mut(metric: float, gate: dict[str, Any]) -> list[str]:
    return [
        f"{op} {limit}" for op, limit in gate.items() if not _CHECKS[op](metric, limit)
    ]
