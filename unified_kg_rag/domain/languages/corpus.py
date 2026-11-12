# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which language a corpus sample is in, against the configured
``processing.translation.source_language``.

Pure: the sample's letters are counted by script (``shared.utils.scripts``)
and its words kept, and each document's language is read by the rule a
query's is (:func:`~unified_kg_rag.domain.languages.detection.script_language`):
Han, Hangul and Kana by their shares (Korean written with Hanja is Korean),
a script one language writes as that language (Cyrillic, Arabic, Thai, ...),
and the languages of one script (Latin) by their function words. Each
document votes first for the script group most of its letters are in, and
for a language only when that language is written in the group; the corpus
reads as the language of the group most documents are written in: pooled,
one Korean document's Hangul runs outweighed the function words of several
English ones, and an English document too terse to decide, or quoting a
Korean file name, voted for no language or for Korean. Wide margins keep a
technical corpus quoting another script from being judged by its quotes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import TypeVar

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.detection import (
    TextEvidence,
    language_in_script,
    script_language,
    text_evidence,
)
from unified_kg_rag.domain.languages.registry import language_name, profile
from unified_kg_rag.shared.utils.scripts import (
    ScriptCounts,
    Spacing,
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

_T = TypeVar("_T")


@dataclass(frozen=True)
class CorpusSample:
    """What a corpus sample's texts tell of their language: the evidence of
    each (``detection.text_evidence``) and of all of them."""

    documents: tuple[TextEvidence, ...]
    pooled: TextEvidence

    @property
    def counts(self) -> ScriptCounts:
        """The letters of all the sampled texts, by script."""
        return self.pooled.counts


def corpus_sample(texts: Iterable[str]) -> CorpusSample:
    """The first :data:`SAMPLE_CHARS` characters of each of the first
    :data:`SAMPLE_DOCUMENTS` texts, as the evidence of their language."""
    sampled = [text[:SAMPLE_CHARS] for text in islice(texts, SAMPLE_DOCUMENTS)]
    return CorpusSample(
        tuple(text_evidence([text]) for text in sampled), text_evidence(sampled)
    )


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


def _evidence_vote(
    evidence: TextEvidence, fallback: LanguageCode | None
) -> tuple[str, LanguageCode | None] | None:
    """The script group ``evidence`` is written in (``ScriptCounts.dominant``)
    and the language it reads as (``script_language``; ``fallback`` breaks
    a function-word tie) when that language is written in the group, else
    ``None`` with the group (undecided: a terse English text of few
    function words, or one quoting a Korean file name, is still a Latin
    text); ``None`` when its letters are too few or no group holds the
    majority of them."""
    counts = evidence.counts
    if counts.letters < _MIN_LETTERS:
        return None
    dominant = counts.dominant()
    if dominant is None or _share(counts, dominant) <= _MAJORITY:
        return None
    language = script_language(evidence, fallback)
    if language is not None and _group(profile(language).script) != dominant:
        language = None
    return dominant, language


def _most(votes: Counter[_T], fallback: _T | None) -> tuple[_T | None, bool]:
    """The one of ``votes`` cast most (``fallback`` among those tied, else
    ``None``), and whether any vote was cast."""
    if not votes:
        return None, False
    most = max(votes.values())
    leaders = [vote for vote, count in votes.items() if count == most]
    if len(leaders) == 1:
        return leaders[0], True
    return (fallback if fallback in leaders else None), True


def corpus_language(
    sample: CorpusSample, fallback: LanguageCode | None = None
) -> LanguageCode | None:
    """The language of the script group most of ``sample``'s documents are
    written in (:func:`_evidence_vote`): the language most of that group's
    documents read as, else (none of them decided, a corpus of terse
    technical English) the one their group's letters read as pooled;
    ``fallback`` among those tied, else ``None``. When no document decides
    alone (a corpus of a few short sentences), the language all of them
    read as."""
    fallback_group = None if fallback is None else _group(profile(fallback).script)
    votes = [
        vote
        for document in sample.documents
        if (vote := _evidence_vote(document, fallback)) is not None
    ]
    if not votes:
        pooled = _evidence_vote(sample.pooled, fallback)
        return None if pooled is None else pooled[1]
    group, _ = _most(Counter(group for group, _ in votes), fallback_group)
    if group is None:
        return None
    language, decided = _most(
        Counter(
            language
            for voted, language in votes
            if voted == group and language is not None
        ),
        fallback,
    )
    if decided:
        return language
    return language_in_script(sample.pooled, group, fallback)


def source_language_mismatch(
    sample: CorpusSample, source_language: LanguageCode
) -> str | None:
    """Why ``sample`` disagrees with ``source_language``, or ``None``: few of
    its letters are in that language's script, or it reads as another
    language; never when it reads as ``source_language`` (a Korean text
    that is mostly code)."""
    counts = sample.counts
    if counts.letters < _MIN_LETTERS:
        return None
    group = _group(profile(source_language).script)
    share = _share(counts, group)
    detected = corpus_language(sample, source_language)
    if detected is source_language:
        return None
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
    if detected is not None:
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
