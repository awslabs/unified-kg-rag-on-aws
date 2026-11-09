# guard: G4
# as: adapters/ingestion/chunker.py
# expect: clean
from ...shared.utils.scripts import is_unspaced_char, script_of


def _mut(text: str) -> list[str]:
    names = []
    for ch in text:
        script = script_of(ch)
        names.append(script.name if script is not None else "-")
    if any(is_unspaced_char(ch) for ch in text):
        names.append("unspaced")
    return names
