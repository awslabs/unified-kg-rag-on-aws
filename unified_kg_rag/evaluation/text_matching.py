# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic text normalization shared by the LLM-free evaluators."""

from __future__ import annotations

import re
import string
import unicodedata

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_ASCII_PUNCTUATION = frozenset(string.punctuation)


def _is_punctuation(ch: str) -> bool:
    # SQuAD removes ASCII string.punctuation (which includes symbols such as
    # "$" and "%"); Unicode punctuation (category P*) covers CJK and fullwidth
    # marks such as 「」 and 。.
    return ch in _ASCII_PUNCTUATION or unicodedata.category(ch).startswith("P")


def normalize_answer(text: str) -> str:
    """SQuAD v1.1 answer normalization, preceded by Unicode NFKC.

    NFKC folds compatibility forms (fullwidth digits/letters, ligatures, the
    "…" ellipsis) to their canonical characters; then, as in the official
    SQuAD script: lowercase, delete punctuation, drop the English articles
    ``a``/``an``/``the``, and collapse whitespace. Punctuation is deleted, not
    replaced by a space, so "1,000" equals "1000" (and "3.5" becomes "35").
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if not _is_punctuation(ch))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())
