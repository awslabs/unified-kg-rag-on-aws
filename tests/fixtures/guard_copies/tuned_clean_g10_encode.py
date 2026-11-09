# guard: G10
# as: application/storage/pending_writes.py
# expect: clean
import json


def _mut(text_unit_ids: list[str]) -> dict[str, str]:
    """The edge property as written: one JSON list string."""
    return {"text_unit_ids": json.dumps(sorted(text_unit_ids))}
