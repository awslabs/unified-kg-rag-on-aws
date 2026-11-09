# guard: G5
# as: application/cli/display.py
# expect: clean
from typing import Any

from unified_kg_rag.domain.retrieval.source_fields import file_name, relative_path


def _mut(source: dict[str, Any]) -> str:
    meta = source.get("metadata") or {}
    return relative_path(meta) or file_name(meta) or "unknown"
