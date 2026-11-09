# guard: G11
# as: evaluation/source_resolver.py
# expect: copy
from itertools import chain
from typing import Any


def _mut(items: list[dict[str, Any]]) -> list[str]:
    merged: list[str] = []
    for unit in chain.from_iterable(i.get("text_unit_ids", ()) for i in items):
        if unit not in merged:
            merged.append(unit)
    return merged
