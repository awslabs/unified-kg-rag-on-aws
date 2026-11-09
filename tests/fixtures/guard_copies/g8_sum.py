# guard: G8
# as: evaluation/text_matching.py
# expect: copy
from unified_kg_rag.shared.utils.scripts import Script, script_of


def _mut(t):
    return sum(1 for c in t if script_of(c) is Script.HANGUL)
