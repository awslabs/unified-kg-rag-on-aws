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

from unified_kg_rag.domain.models import LanguageCode
from unified_kg_rag.shared.utils.scripts import ScriptCounts

__all__ = ["dense_language"]

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
