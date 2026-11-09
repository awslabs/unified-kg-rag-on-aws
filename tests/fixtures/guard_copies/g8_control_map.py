# guard: G8
# as: evaluation/text_matching.py
# expect: copy
from collections import Counter

from unified_kg_rag.shared.utils.scripts import letter_class


def _mut(t):
    return Counter(map(letter_class, t))
