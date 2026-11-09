# guard: G4
# as: evaluation/text_matching.py
# expect: clean
import unicodedata


def _mut(text: str) -> str:
    """Drop punctuation and control characters; keep every letter and digit."""
    kept = []
    for ch in unicodedata.normalize("NFKC", text):
        category = unicodedata.category(ch)
        if category.startswith("P") or category == "Cc":
            continue
        kept.append(ch.lower() if ch.isascii() else ch)
    return "".join(kept)
