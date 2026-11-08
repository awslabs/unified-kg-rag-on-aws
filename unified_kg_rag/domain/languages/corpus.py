# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which language a corpus sample is in, against the configured
``processing.translation.source_language``.

Pure: the sample's letters are counted by script (``shared.utils.scripts``)
and its words kept, and the language is read by the rule a query's is
(:func:`~unified_kg_rag.domain.languages.detection.script_language`): Han,
Hangul and Kana by their shares (Korean written with Hanja is Korean), a
script one language writes as that language (Cyrillic, Arabic, Thai, ...),
and the languages of one script (Latin) by their function words. Wide
margins keep a technical corpus quoting another script from being judged
by its quotes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.detection import script_language, words_of
from unified_kg_rag.domain.languages.registry import language_name, profile
from unified_kg_rag.shared.utils.scripts import (
    ScriptCounts,
    Spacing,
    count_letters,
    script_named,
)

__all__ = [
    "CorpusSample",
    "corpus_language",
    "corpus_sample",
    "corpus_source_language",
    "source_language_mismatch",
]

# What is read of a sample: the first characters of its first documents.
SAMPLE_DOCUMENTS = 20
SAMPLE_CHARS = 2000
# Fewer letters than this decide nothing (a corpus of tables, numbers); a
# two-sentence sample still decides.
_MIN_LETTERS = 40
# A corpus with fewer letters of the configured language's script than
# this share disagrees with the setting, and a language is read only from
# a script holding more than the majority of the letters. Wide margins:
# technical Korean or Japanese text carries Latin terms, and an English
# corpus may quote names in Han or Hangul.
_MINORITY = 0.2
_MAJORITY = 0.5


@dataclass(frozen=True)
class CorpusSample:
    """A corpus sample's letters by script and its words."""

    counts: ScriptCounts
    words: tuple[str, ...]


def corpus_sample(texts: Iterable[str]) -> CorpusSample:
    """The first :data:`SAMPLE_CHARS` characters of each of the first
    :data:`SAMPLE_DOCUMENTS` texts: their letters by script and words."""
    sampled = [text[:SAMPLE_CHARS] for text in islice(texts, SAMPLE_DOCUMENTS)]
    words = tuple(word for text in sampled for word in words_of(text))
    return CorpusSample(count_letters(sampled), words)


def _group(script: str) -> str:
    """``script`` as ``ScriptCounts.dominant`` groups it: the dense ones
    as ``"dense"``."""
    return "dense" if script_named(script).spacing is Spacing.DENSE else script


def _share(counts: ScriptCounts, group: str) -> float:
    """The share of the letters in ``group`` (a script name, or "dense")."""
    held = counts.dense if group == "dense" else counts[group]
    return held / counts.letters if counts.letters else 0.0


def _script_label(group: str) -> str:
    return "Hangul, Kana or Han" if group == "dense" else group.capitalize()


def corpus_language(
    sample: CorpusSample, fallback: LanguageCode | None = None
) -> LanguageCode | None:
    """The language ``sample`` reads as (``script_language``; ``fallback``
    breaks a function-word tie), or ``None`` when its letters are too few or
    no script holds the majority of them."""
    counts = sample.counts
    if counts.letters < _MIN_LETTERS:
        return None
    dominant = counts.dominant()
    if dominant is None or _share(counts, dominant) <= _MAJORITY:
        return None
    return script_language(counts, sample.words, fallback)


def source_language_mismatch(
    sample: CorpusSample, source_language: LanguageCode
) -> str | None:
    """Why ``sample`` disagrees with ``source_language``, or ``None``: few of
    its letters are in that language's script, or it reads as another
    language."""
    counts = sample.counts
    if counts.letters < _MIN_LETTERS:
        return None
    group = _group(profile(source_language).script)
    share = _share(counts, group)
    detected = corpus_language(sample, source_language)
    setting = (
        "processing.translation.source_language is "
        f"'{source_language.value}' ({language_name(source_language.value)})"
    )
    if share < _MINORITY:
        fix = (
            f"set it to '{detected.value}'"
            if detected is not None
            else "set it to the corpus language"
        )
        return (
            f"only {share:.0%} of the sampled letters are {_script_label(group)}, "
            f"but {setting}: {fix}"
        )
    if detected is not None and detected is not source_language:
        return (
            f"the sampled text reads as {language_name(detected.value)} "
            f"({share:.0%} {_script_label(group)} letters), but {setting}: "
            f"set it to '{detected.value}'"
        )
    return None


def corpus_source_language(
    texts: Iterable[str], named: str | None = None
) -> LanguageCode | None:
    """The ``source_language`` a corpus sample reads as
    (:func:`corpus_language`), else the supported language ``named`` (a
    profile's "Korean" or "ko"), else ``None``."""
    if (detected := corpus_language(corpus_sample(texts))) is not None:
        return detected
    wanted = (named or "").strip().lower()
    for code in LanguageCode:
        if wanted in (code.value, language_name(code.value).lower()):
            return code
    return None
