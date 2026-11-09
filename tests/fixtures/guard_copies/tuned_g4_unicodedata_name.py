# guard: G4
# as: adapters/ingestion/translator.py
# expect: copy
import unicodedata

_PREFIXES = {
    "HANGUL": "hangul",
    "HIRAGANA": "kana",
    "KATAKANA": "kana",
    "CJK UNIFIED IDEOGRAPH": "han",
    "ARABIC": "arabic",
    "HEBREW": "hebrew",
}


def _mut(ch: str) -> str | None:
    name = unicodedata.name(ch, "")
    for prefix, script in _PREFIXES.items():
        if name.startswith(prefix):
            return script
    return None
