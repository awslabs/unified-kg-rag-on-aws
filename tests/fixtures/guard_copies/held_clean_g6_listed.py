# guard: G6
# as: adapters/retrievers/graph_expansion.py
# expect: clean
from typing import Any

from ...domain.retrieval.source_fields import id_list, listed


def _mut(metadata: dict[str, Any], raw: Any) -> list[str]:
    return listed(metadata, "entity_ids") + id_list(raw)
