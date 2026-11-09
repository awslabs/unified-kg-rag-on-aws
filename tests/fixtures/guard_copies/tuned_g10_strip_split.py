# guard: G10
# as: adapters/retrieval/ppr_ranking.py
# expect: missed: a JSON list parsed by splitting its text, no decoder called
from typing import Any


def _mut(value: Any) -> list[str]:
    text = str(value).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []
    return [part.strip().strip('"') for part in text[1:-1].split(",") if part.strip()]
