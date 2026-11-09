# guard: G5
# as: application/cli/display.py
# expect: copy
from typing import Any


def _mut(source: dict[str, Any]) -> str:
    meta = source.get("metadata") or {}
    name = meta.get("file_name") or meta.get("file_path") or "unknown"
    return name.rsplit("/", 1)[-1]
