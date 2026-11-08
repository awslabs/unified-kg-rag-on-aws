# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language a text's script decides.

One rule (:func:`script_language`) tells the language of a query and of a
corpus sample wherever the package asks (the decline's language, the
corpus check, prompt tuning). Korean, Japanese and Chinese are told apart
by their dense letters: Korean writes
Hangul and may mix in Han (Hanja), Japanese writes Kana and Han (Kanji),
Chinese Han alone. A text whose Han, Hangul and Kana letters are a tenth or
more Hangul (and no fewer Hangul than Kana) is Korean, else a tenth or more
Kana Japanese, else Chinese. The shares are of the dense letters only, so
Latin terms in technical text do not move them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile
from unified_kg_rag.domain.languages.registry import PROFILES
from unified_kg_rag.shared.utils.scripts import ScriptCounts, count_letters

__all__ = ["dense_language", "script_language", "text_language", "words_of"]

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


def words_of(text: str) -> list[str]:
    """The lowercased words of ``text`` function words are counted in."""
    return _WORD.findall(text.lower())


def _candidates(script: str | None) -> list[LanguageProfile]:
    """The languages a text in ``script`` may be in: those written in it,
    else the Latin-script ones (a script no profile writes)."""
    written = [p for p in PROFILES.values() if p.script == script]
    return written or [p for p in PROFILES.values() if p.script == "latin"]


def script_language(
    counts: ScriptCounts,
    words: Iterable[str] = (),
    fallback: LanguageCode | None = None,
) -> LanguageCode | None:
    """The language a text's letters (``counts``) and ``words`` read as,
    or ``None`` when they do not tell: the one rule a query
    (:func:`text_language`) and a corpus sample
    (``domain.languages.corpus``) are judged by.

    The script most of the letters are in decides
    (``ScriptCounts.dominant``): Han, Hangul and Kana read as
    :func:`dense_language` says, a script one profile writes as that
    language, and among the languages of one script (Latin) their function
    words decide: the language with the most, ``fallback`` (the index
    language) among those tied, else none."""
    script = counts.dominant()
    if script is None:
        return None
    if script == "dense":
        return dense_language(counts, fallback)
    candidates = _candidates(script)
    if len(candidates) == 1 and candidates[0].script == script:
        return candidates[0].code
    words = list(words)
    hits = {p.code: sum(word in p.function_words for word in words) for p in candidates}
    best = max(hits.values(), default=0)
    if not best:
        return None
    tied = [code for code, count in hits.items() if count == best]
    if fallback in tied:
        return fallback
    return tied[0] if len(tied) == 1 else None


def text_language(text: str, fallback: LanguageCode | str) -> LanguageCode:
    """The language ``text`` (a query) is written in, as far as its script
    and function words tell (:func:`script_language`: a Korean question
    quoting an English name is Korean, an English one quoting a Korean name
    English); ``fallback`` (the index language) when they do not and it is
    written in the text's script (a Latin one for a Latin-script text),
    else English."""
    fallback = _code(fallback)
    counts = count_letters([text])
    found = script_language(counts, words_of(text), fallback)
    if found is not None:
        return found
    script = counts.dominant()
    if script == "dense":
        return fallback
    return (
        fallback
        if fallback in {p.code for p in _candidates(script)}
        else LanguageCode.EN
    )


def _code(code: LanguageCode | str) -> LanguageCode:
    try:
        return LanguageCode(code)
    except ValueError:
        return LanguageCode.EN
