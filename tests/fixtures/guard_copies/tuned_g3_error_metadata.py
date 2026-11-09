# guard: G3
# as: application/cli/mcp_tools.py
# expect: copy
from typing import Any


def _mut(output: Any) -> dict[str, Any]:
    if output.metadata.get("error"):
        return {
            "ok": False,
            "error_class": output.metadata.get("error_class"),
            "message": output.search_results.metadata.get("error"),
        }
    return {"ok": True, "answer": output.answer}
