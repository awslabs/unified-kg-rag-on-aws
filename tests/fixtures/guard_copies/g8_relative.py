# guard: G8
# as: evaluation/text_matching.py
# expect: copy
from collections import Counter

from ..shared.utils.scripts import script_of as _so


def _mut(t):
    return Counter(_so(c) for c in t)
