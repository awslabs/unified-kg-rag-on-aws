# guard: G4
# as: domain/retrieval/keyword_fallback.py
# expect: copy


def _mut(token: str) -> bool:
    """Whether a token is written in Korean or Japanese."""
    return any("가" <= c <= "힣" or "ぁ" <= c <= "ヿ" for c in token)
