# guard: G2
# as: adapters/search_strategies/drift_search.py
# expect: copy
import json
from typing import Any


def _mut(response: Any) -> dict[str, Any]:
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)
