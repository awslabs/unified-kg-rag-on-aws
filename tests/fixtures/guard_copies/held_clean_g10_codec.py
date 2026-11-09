# guard: G10
# as: adapters/retrievers/neptune_retriever.py
# expect: clean
from typing import Any

from ..storage.neptune_codec import decode_edge_list, edge_text_unit_ids


def _mut(edge_props: dict[str, Any]) -> tuple[list[str], list[str]]:
    docs = decode_edge_list(edge_props.get("document_ids"), "document_ids") or []
    return edge_text_unit_ids(edge_props.get("text_unit_ids")), docs
