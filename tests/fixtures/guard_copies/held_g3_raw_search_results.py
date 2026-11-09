# guard: G3
# as: application/cli/console/jobs.py
# expect: missed: the raw search results read field by field, no key or sources read
from typing import Any


def _mut(search_output: Any) -> list[dict[str, Any]]:
    rows = []
    for item in search_output.search_results.results:
        rows.append({"id": item.chunk_id, "content": item.content, "score": item.score})
    return rows
