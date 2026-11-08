# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language a text's script decides.

One rule tells Korean, Japanese and Chinese apart wherever the package asks
(the corpus check, prompt tuning, the decline's language): Korean writes
Hangul and may mix in Han (Hanja), Japanese writes Kana and Han (Kanji),
Chinese Han alone. A text whose Han, Hangul and Kana letters are a tenth or
more Hangul (and no fewer Hangul than Kana) is Korean, else a tenth or more
Kana Japanese, else Chinese. The shares are of the dense letters only, so
Latin terms in technical text do not move them.
"""

from __future__ import annotations

import re

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.registry import PROFILES
from unified_kg_rag.shared.utils.scripts import ScriptCounts, count_letters

__all__ = ["dense_language", "text_language"]

# The share of a text's dense letters in Hangul (or Kana) that makes it
# Korean (or Japanese). Han carries the stems of both: a Korean statute
# written with Hanja holds about a third Hangul, Japanese prose about half
# Kana, and Chinese none of either.
_SCRIPT_SHARE = 0.1


def dense_language(
    counts: ScriptCounts, fallback: LanguageCode | None = None
) -> LanguageCode | None:
    """The dense-script language ``counts`` read as (see the module), or
    ``None`` without Han, Hangul or Kana letters. Han alone is Chinese,
    unless ``fallback`` (the index language) is Japanese: a Japanese
    index's kanji-only query is Japanese."""
    dense = counts.dense
    if not dense:
        return None
    hangul, kana = counts["hangul"], counts["kana"]
    if hangul >= _SCRIPT_SHARE * dense and hangul >= kana:
        return LanguageCode.KO
    if kana >= _SCRIPT_SHARE * dense:
        return LanguageCode.JA
    if not hangul and not kana and fallback is LanguageCode.JA:
        return LanguageCode.JA
    return LanguageCode.ZH


_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def text_language(text: str, fallback: LanguageCode | str) -> LanguageCode:
    """The language ``text`` (a query) is written in, as far as its script
    and function words tell; ``fallback`` (the index language) when they do
    not and it is written in the text's script (a Latin one for a
    Latin-script text), else English.

    The script most of the text's letters are in decides
    (``ScriptCounts.dominant``: a Korean question quoting an English name is
    Korean, an English one quoting a Korean name English); Han, Hangul and
    Kana read as :func:`dense_language` says, a script one language writes
    as that language, and among the languages of one script (Latin) their
    function words decide."""
    fallback = _code(fallback)
    counts = count_letters([text])
    script = counts.dominant()
    if script == "dense":
        return dense_language(counts, fallback) or fallback
    written = [p.code for p in PROFILES.values() if p.script == script]
    if len(written) == 1:
        return written[0]
    latin = [p for p in PROFILES.values() if p.script == "latin"]
    words = _WORD.findall(text.lower())
    hits = {p.code: sum(word in p.function_words for word in words) for p in latin}
    best = max(hits.values(), default=0)
    if best:
        tied = [code for code, count in hits.items() if count == best]
        if fallback in tied:
            return fallback
        if len(tied) == 1:
            return tied[0]
    return fallback if fallback in hits else LanguageCode.EN


def _code(code: LanguageCode | str) -> LanguageCode:
    try:
        return LanguageCode(code)
    except ValueError:
        return LanguageCode.EN
