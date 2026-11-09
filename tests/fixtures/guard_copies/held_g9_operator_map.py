# guard: G9
# as: application/storage/inspection.py
# expect: copy
import operator
from typing import Any

_OPS = {"gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le}


def _mut(
    rows: list[dict[str, Any]], field: str, rng: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if field in row and all(_OPS[op](row[field], b) for op, b in rng.items())
    ]
