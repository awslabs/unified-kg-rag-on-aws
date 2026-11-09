# guard: G6
# as: adapters/retrieval/hybrid_scorer.py
# expect: copy
from typing import Any


def _mut(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("text_unit_ids")
    ids = value if isinstance(value, list) else [value]
    return [i for i in ids if i]
