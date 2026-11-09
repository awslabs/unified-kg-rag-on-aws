# guard: G5
# as: adapters/retrievers/opensearch_retriever.py
# expect: copy
import os
from typing import Any

_FIELD = "file_path"


def _mut(hit: dict[str, Any]) -> str:
    src = hit["_source"]
    return os.path.basename(src.get(_FIELD, ""))
