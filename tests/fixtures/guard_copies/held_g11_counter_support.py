# guard: G11
# as: adapters/search_strategies/local_search.py
# expect: copy
from collections import Counter
from typing import Any


def _mut(entities: list[Any], cap: int) -> list[str]:
    support = Counter()
    for entity in entities:
        support.update(set(entity.metadata.get("text_unit_ids") or ()))
    return [unit for unit, _ in support.most_common(cap)]
