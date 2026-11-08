# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which writing system a corpus sample is in, against the configured
``processing.translation.source_language``.

Pure: letters are counted by Unicode block with the script table of
``shared.utils.scripts``, so the test tells a Korean, Japanese or Chinese
corpus from one in a Latin or other script, and those three apart
(``domain.languages.dense_language``: Korean written with Hanja is
Korean). It cannot tell two languages of one script apart (English from
German), so it judges only where the script decides.
"""

from __future__ import annotations

from collections.abc import Iterable

from unified_kg_rag.domain.languages import dense_language
from unified_kg_rag.domain.models import LanguageCode, language_name
from unified_kg_rag.shared.utils.scripts import ScriptCounts, count_letters

# The languages whose script the dense-script table holds.
DENSE_SCRIPT_LANGUAGES: frozenset[LanguageCode] = frozenset(
    {LanguageCode.KO, LanguageCode.JA, LanguageCode.ZH}
)

# What is read of a sample: the first characters of its first documents.
SAMPLE_DOCUMENTS = 20
SAMPLE_CHARS = 2000
# Fewer letters than this decide nothing (a corpus of tables, numbers); a
# two-sentence sample still decides.
_MIN_LETTERS = 40
# A dense-script language's corpus with fewer of its letters than this
# share, or another language's with more than the majority, disagrees with
# the setting. Wide margins: technical Korean or Japanese text carries Latin
# terms, and an English corpus may quote names in Han or Hangul.
_DENSE_MINORITY = 0.2
_DENSE_MAJORITY = 0.5


def script_share(texts: Iterable[str]) -> ScriptCounts:
    """Count the letters of the first :data:`SAMPLE_CHARS` characters of
    each of the first :data:`SAMPLE_DOCUMENTS` texts by script."""
    return count_letters(texts, max_texts=SAMPLE_DOCUMENTS, max_chars=SAMPLE_CHARS)


def detected_dense_language(share: ScriptCounts) -> LanguageCode | None:
    """The dense-script language a sample is written in, or ``None`` when
    its letters are not mostly Hangul, Kana and Han (or too few)."""
    if share.letters < _MIN_LETTERS or share.dense_share <= _DENSE_MAJORITY:
        return None
    return dense_language(share)


def source_language_mismatch(
    share: ScriptCounts, source_language: LanguageCode
) -> str | None:
    """Why a sample's script disagrees with ``source_language``, or
    ``None``: a mostly Hangul, Kana or Han sample under another language,
    or a dense-script language's corpus with few of its letters."""
    if share.letters < _MIN_LETTERS:
        return None
    percent = f"{share.dense_share:.0%}"
    detected = detected_dense_language(share)
    if source_language not in DENSE_SCRIPT_LANGUAGES:
        if detected is None:
            return None
        return (
            f"{percent} of the sampled letters are Hangul, Kana or Han, which "
            f"reads as {language_name(detected.value)}, but "
            f"processing.translation.source_language is "
            f"'{source_language.value}'"
            f" ({language_name(source_language.value)}): set it to "
            f"'{detected.value}'"
        )
    if share.dense_share < _DENSE_MINORITY:
        return (
            f"only {percent} of the sampled letters are Hangul, Kana or Han, "
            "but processing.translation.source_language is "
            f"'{source_language.value}' ({language_name(source_language.value)})"
            ": set it to the corpus language"
        )
    if detected is not None and detected is not source_language:
        return (
            f"the sampled text reads as {language_name(detected.value)} "
            f"({percent} Hangul, Kana or Han letters), but "
            "processing.translation.source_language is "
            f"'{source_language.value}' ({language_name(source_language.value)})"
            f": set it to '{detected.value}'"
        )
    return None


def corpus_source_language(
    texts: Iterable[str], named: str | None = None
) -> LanguageCode | None:
    """The ``source_language`` a corpus sample reads as: the dense-script
    language its letters are mostly in, else the supported language
    ``named`` (a profile's "Korean" or "ko"), else ``None``."""
    if (detected := detected_dense_language(script_share(texts))) is not None:
        return detected
    wanted = (named or "").strip().lower()
    for code in LanguageCode:
        if wanted in (code.value, language_name(code.value).lower()):
            return code
    return None
