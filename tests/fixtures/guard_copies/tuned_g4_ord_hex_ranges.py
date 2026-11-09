# guard: G4
# as: evaluation/text_matching.py
# expect: copy


def _mut(ch: str) -> str | None:
    cp = ord(ch)
    if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF:
        return "hangul"
    if 0x3040 <= cp <= 0x30FF:
        return "kana"
    if 0x4E00 <= cp <= 0x9FFF:
        return "han"
    return None
