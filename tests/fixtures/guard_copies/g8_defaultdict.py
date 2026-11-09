# guard: G8
# as: evaluation/text_matching.py
# expect: copy
from collections import defaultdict

from unified_kg_rag.shared.utils.scripts import script_of


def _mut(t):
    counts = defaultdict(int)
    for c in t:
        counts[script_of(c)] += 1
    return counts
