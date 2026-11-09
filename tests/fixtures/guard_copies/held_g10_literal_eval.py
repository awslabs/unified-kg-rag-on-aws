# guard: G10
# as: application/retrieval/graph_lookup.py
# expect: copy
from ast import literal_eval
from typing import Any


def _mut(props: dict[str, Any]) -> list[str]:
    stored = props.get("relationship_ids") or "[]"
    decoded = literal_eval(stored) if isinstance(stored, str) else stored
    return list(map(str, decoded))
