# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic texts for chunking tests (English, Korean, mixed).

Seeded, so the same text comes out on every run and Python version; the
chunk-parity tests pin the chunks of these texts.
"""

from __future__ import annotations

import random

_ENGLISH = (
    "vendor buyer shipment delivery invoice payment schedule warranty "
    "agreement parties termination notice period supplier purchaser "
    "inspection acceptance quantity price obligations remedies"
).split()
_HANGUL = (
    "가나다라마바사아자차카타파하한국어문서공급자구매자물품납품대금지급계약해지통지"
)
_KANA_HAN = "契約の当事者は売主と買主であり納品検収支払期日通知解除"


def _sentences(rng: random.Random, words: list[str], count: int, end: str) -> str:
    return " ".join(
        " ".join(rng.choice(words) for _ in range(rng.randint(4, 14))) + end
        for _ in range(count)
    )


def _paragraphs(rng: random.Random, make, chars: int, sep: str) -> str:
    parts: list[str] = []
    while sum(len(p) + len(sep) for p in parts) < chars:
        parts.append(make())
    return sep.join(parts)[:chars]


def english_text(chars: int, seed: int = 1) -> str:
    rng = random.Random(seed)
    return _paragraphs(
        rng, lambda: _sentences(rng, _ENGLISH, rng.randint(2, 9), "."), chars, "\n\n"
    )


def korean_text(chars: int, seed: int = 2) -> str:
    rng = random.Random(seed)

    def word() -> str:
        return "".join(rng.choice(_HANGUL) for _ in range(rng.randint(1, 6)))

    words = [word() for _ in range(300)]
    return _paragraphs(
        rng, lambda: _sentences(rng, words, rng.randint(2, 8), "."), chars, "\n\n"
    )


def japanese_text(chars: int, seed: int = 3) -> str:
    rng = random.Random(seed)

    def sentence() -> str:
        return "".join(rng.choice(_KANA_HAN) for _ in range(rng.randint(8, 40))) + "。"

    return _paragraphs(
        rng, lambda: "".join(sentence() for _ in range(rng.randint(2, 6))), chars, "\n"
    )


def mixed_text(chars: int, seed: int = 4) -> str:
    """English, Korean and Japanese paragraphs, line-broken paragraphs and
    long unbroken runs (no separator but single characters)."""
    rng = random.Random(seed)
    pieces = [
        lambda: english_text(rng.randint(200, 3000), rng.randint(0, 999)),
        lambda: korean_text(rng.randint(200, 3000), rng.randint(0, 999)),
        lambda: japanese_text(rng.randint(200, 2000), rng.randint(0, 999)),
        lambda: "x" * rng.randint(50, 6000),
        lambda: "\n".join(
            english_text(rng.randint(30, 120), rng.randint(0, 999))
            for _ in range(rng.randint(3, 20))
        ),
    ]
    return _paragraphs(rng, lambda: rng.choice(pieces)(), chars, "\n\n")
