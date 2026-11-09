# guard: G10
# as: adapters/retrievers/neptune_retriever.py
# expect: copy
import json
from typing import Any


def _mut(edge_props: dict[str, Any]) -> list[str]:
    raw = edge_props.get("text_unit_ids")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if not raw:
        return []
    return [str(x) for x in json.loads(raw)]
