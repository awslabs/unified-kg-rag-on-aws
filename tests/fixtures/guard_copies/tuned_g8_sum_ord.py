# guard: G8
# as: adapters/llm/token_estimate.py
# expect: missed: G8 sees no script reader; G4 reports its code-point ranges


def _mut(text: str) -> int:
    hangul = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7A3)
    cjk = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    other = len(text) - hangul - cjk
    return hangul + cjk + other // 4
