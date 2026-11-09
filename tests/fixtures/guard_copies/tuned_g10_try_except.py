# guard: G10
# as: application/storage/removal.py
# expect: copy
import json
from typing import Any


def _mut(edge: dict[str, Any]) -> set[str]:
    try:
        ids = json.loads(edge["document_ids"])
    except (KeyError, TypeError, ValueError):
        ids = []
    return set(ids) if isinstance(ids, list) else set()
