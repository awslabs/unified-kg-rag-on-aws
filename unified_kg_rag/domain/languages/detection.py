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
from unified_kg_rag.shared.utils.scripts import (
    ScriptCounts,
    count_letters,
    letter_class,
)

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
_LATIN_LETTER = f"[{letter_class('latin')}]"
_LATIN_WORD = re.compile(rf"{_LATIN_LETTER}+(?:'{_LATIN_LETTER}+)*")


def words_of(text: str) -> list[str]:
    """The lowercased words of ``text`` function words are counted in (an
    elided word kept with its apostrophe: "dell'accordo").

    In a text with letters of another script, its Latin words only (the
    function words and elisions are Latin-script languages'), and of them
    none of a Latin name it quotes (:func:`_name_words`): "Bank of
    America和The Walt Disney Company是什么关系？" holds no English
    function word."""
    text = text.replace("\u2019", "'")
    if not any(ch.isalpha() for ch in _LATIN_WORD.sub("", text)):
        return _WORD.findall(text.lower())
    words = list(_LATIN_WORD.finditer(text))
    names = _name_words(text, words)
    return [w.group().lower() for i, w in enumerate(words) if i not in names]


def _name_words(text: str, words: list[re.Match[str]]) -> set[int]:
    """The indexes of ``words`` (Latin words in order) inside a name: a run
    of two or more of them with spaces alone between, opening and closing
    on a capitalised word, with the lowercase words inside it ("Bank of
    America", "Procter and Gamble", "The Walt Disney Company"); a
    capitalised word alone (a question's first word) is no name."""
    names: set[int] = set()
    run: list[int] = []

    def close() -> None:
        while run and not words[run[-1]].group()[0].isupper():
            run.pop()
        if len(run) > 1:
            names.update(run)
        run.clear()

    for i, word in enumerate(words):
        joined = bool(run) and not text[words[run[-1]].end() : word.start()].strip()
        if not joined:
            close()
            if not word.group()[0].isupper():
                continue
        run.append(i)
    close()
    return names


def _signals(
    word: str, language: LanguageProfile, shared: frozenset[str] = frozenset()
) -> bool:
    """Whether ``word`` is one of ``language``'s function words, or opens
    with a word it elides that no other candidate elides (``shared``:
    "dell'accordo" is Italian, "d'unités" either) or closes on a function
    word ("d'où", "c'est", "dov'è")."""
    if word in language.function_words:
        return True
    head, apostrophe, tail = word.partition("'")
    if not apostrophe:
        return False
    return (
        head in language.elided_prefixes and head not in shared
    ) or tail in language.function_words


def _shared_elisions(candidates: list[LanguageProfile]) -> frozenset[str]:
    """The elided words more than one of ``candidates`` writes ("l'", "d'",
    "c'" in French and Italian): a sign of neither."""
    seen: set[str] = set()
    shared: set[str] = set()
    for language in candidates:
        shared |= seen & language.elided_prefixes
        seen |= language.elided_prefixes
    return frozenset(shared)


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
    words and the elisions only one of them writes decide ("dell'" is
    Italian, "qu'" French, "d'" neither): the
    language with the most, ``fallback`` (the index language) among those
    tied, else the one of them whose own articles occur most, else none.
    A script none of whose languages' words occur is names the text quotes
    in another script ("Microsoft Corporation和OpenAI是什么关系？", "ما علاقة
    Microsoft Corporation بـ OpenAI؟"): the next script decides."""
    words = list(words)
    quoted: set[str] = set()
    while (script := counts.dominant(without=quoted)) is not None:
        if script == "dense":
            return dense_language(counts, fallback)
        candidates = _candidates(script)
        if len(candidates) == 1 and candidates[0].script == script:
            return candidates[0].code
        shared = _shared_elisions(candidates)
        hits = {
            p.code: sum(_signals(word, p, shared) for word in words) for p in candidates
        }
        best = max(hits.values(), default=0)
        if not best:
            quoted.add(script)
            continue
        tied = [p for p in candidates if hits[p.code] == best]
        if fallback in {p.code for p in tied}:
            return fallback
        if len(tied) > 1:
            tied = _by_articles(tied, words)
        return tied[0].code if len(tied) == 1 else None
    return None


def _by_articles(
    tied: list[LanguageProfile], words: list[str]
) -> list[LanguageProfile]:
    """Of languages tied on their function words, those whose articles no
    other of them writes occur most in ``words`` ("il" is Italian, not
    Portuguese); all of them when none occurs."""
    counts: dict[LanguageCode, int] = {}
    for language in tied:
        own = language.articles - frozenset().union(
            *(other.articles for other in tied if other is not language)
        )
        counts[language.code] = sum(word in own for word in words)
    best = max(counts.values())
    return [p for p in tied if counts[p.code] == best] if best else tied


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
