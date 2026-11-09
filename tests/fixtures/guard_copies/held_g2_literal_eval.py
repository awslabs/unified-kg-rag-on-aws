# guard: G2
# as: application/retrieval/query_processing.py
# expect: copy
import ast
from typing import Any


def _mut(message: Any) -> list[str]:
    raw = getattr(message, "content", message)
    try:
        parsed = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError):
        return []
    return (
        list(parsed.get("high_level_keywords", [])) if isinstance(parsed, dict) else []
    )
