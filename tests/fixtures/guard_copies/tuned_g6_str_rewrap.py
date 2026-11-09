# guard: G6
# as: adapters/storage/opensearch_indexer.py
# expect: copy
from typing import Any


def _mut(doc: dict[str, Any]) -> dict[str, Any]:
    entity_ids = doc.get("entity_ids", [])
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]
    doc["entity_ids"] = entity_ids
    return doc
