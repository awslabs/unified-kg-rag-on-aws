# guard: G3
# as: application/cli/mcp_server.py
# expect: copy
from typing import Any


def _mut(output: Any) -> str | None:
    md = getattr(output, "metadata", None) or {}
    kind = md["error_type"] if "error_type" in md else None
    message = md.get("error_message")
    return f"{kind}: {message}" if kind else None
