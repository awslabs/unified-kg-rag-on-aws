# guard: G9
# as: adapters/retrievers/sqlite_graph_retriever.py
# expect: copy
from typing import Any


def _mut(value: Any, bounds: dict[str, Any]) -> bool:
    for op, bound in bounds.items():
        if op == "gt":
            ok = value > bound
        elif op == "gte":
            ok = value >= bound
        elif op == "lt":
            ok = value < bound
        elif op == "lte":
            ok = value <= bound
        else:
            raise ValueError(op)
        if not ok:
            return False
    return True
