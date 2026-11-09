# guard: G8
# as: adapters/ingestion/chunker.py
# expect: missed: letters counted by a findall over a letter_class regex
import re

from ...shared.utils.scripts import letter_class

_HAN = re.compile(f"[{letter_class('han')}]")
_KANA = re.compile(f"[{letter_class('kana')}]")


def _mut(text: str) -> float:
    han = len(_HAN.findall(text))
    kana = len(_KANA.findall(text))
    return kana / (han + kana) if han + kana else 0.0
