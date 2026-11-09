# guard: G9
# as: adapters/retrievers/opensearch_retriever.py
# expect: clean
from typing import Any

from ...domain.retrieval.filters import Range


def _mut(clause: Range) -> dict[str, Any]:
    """Translate a range clause into an OpenSearch query; the store compares."""
    return {"range": {clause.field: {op: bound for op, bound in clause.bounds}}}
