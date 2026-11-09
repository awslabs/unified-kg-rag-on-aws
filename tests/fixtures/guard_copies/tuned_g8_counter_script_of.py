# guard: G8
# as: evaluation/text_matching.py
# expect: copy
from collections import Counter

from unified_kg_rag.shared.utils.scripts import script_of


def _mut(text: str) -> str | None:
    counts = Counter(s.name for s in map(script_of, text) if s is not None)
    return counts.most_common(1)[0][0] if counts else None
