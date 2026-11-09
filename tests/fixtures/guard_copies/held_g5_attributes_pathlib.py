# guard: G5
# as: evaluation/source_resolver.py
# expect: copy
from pathlib import PurePath
from typing import Any


def _mut(metadata: dict[str, Any]) -> str | None:
    attrs = metadata.get("attributes", {})
    if "file_path" in attrs:
        return PurePath(attrs["file_path"]).name
    return None
