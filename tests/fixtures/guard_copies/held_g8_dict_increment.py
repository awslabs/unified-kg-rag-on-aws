# guard: G8
# as: application/ingestion/source_parsing.py
# expect: missed: the count key is two bindings away from script_of
from collections.abc import Iterable

from unified_kg_rag.shared.utils import scripts


def _mut(texts: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for ch in text:
            if not ch.isalpha():
                continue
            found = scripts.script_of(ch)
            key = found.name if found else "other"
            counts[key] = counts.get(key, 0) + 1
    return counts
