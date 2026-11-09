# guard: G5
# as: application/cli/console/graph_api.py
# expect: copy
from typing import Any


def _mut(item: dict[str, Any]) -> dict[str, Any]:
    rel = item["attributes"].get("relative_path", "")
    return {"id": item["id"], "path": rel.lstrip("/")}
