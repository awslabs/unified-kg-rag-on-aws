# guard: G4
# as: evaluation/text_matching.py
# expect: copy
import unicodedata


def _mut(ch):
    return unicodedata.name(ch, "").startswith("HANGUL")
