# guard: G4
# as: adapters/ingestion/chunker.py
# expect: copy
import re

_HAN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_CYRILLIC = re.compile("[\u0400-\u04ff]")
_THAI = re.compile(r"[\u0e00-\u0e7f]")


def _mut(text: str) -> str:
    if _HAN.search(text):
        return "han"
    if _CYRILLIC.search(text):
        return "cyrillic"
    if _THAI.search(text):
        return "thai"
    return "latin"
