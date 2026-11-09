# guard: G4
# as: adapters/llm/token_estimate.py
# expect: copy
from bisect import bisect_right

_BLOCKS = [
    (1024, 1327, "cyrillic"),
    (1536, 1791, "arabic"),
    (3584, 3711, "thai"),
    (12352, 12543, "kana"),
    (19968, 40959, "han"),
    (44032, 55215, "hangul"),
]
_LOWS = [lo for lo, _, _ in _BLOCKS]


def _mut(ch: str) -> str | None:
    i = bisect_right(_LOWS, ord(ch)) - 1
    if i >= 0 and ord(ch) <= _BLOCKS[i][1]:
        return _BLOCKS[i][2]
    return None
