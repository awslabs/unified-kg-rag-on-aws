# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language profiles, by code.

Adding a language: a module under ``profiles/`` with its ``PROFILE``, its
code in ``LanguageCode``, and an entry below; ``tests/unit/test_languages.py``
lists what each profile holds and which gaps are known.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cache
from types import MappingProxyType

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile
from unified_kg_rag.domain.languages.profiles import (
    ar,
    de,
    en,
    es,
    fr,
    he,
    hi,
    it,
    ja,
    ko,
    pt,
    ru,
    th,
    vi,
    zh,
)

__all__ = ["BUILT_IN_ANALYZERS", "PROFILES", "language_name", "profile", "profiles"]

# OpenSearch's built-in language analyzers of codes without a profile (the
# primary subtag): such a code's text fields get its language's stemming
# and stop words instead of ``standard``. Regions are dropped before
# (``pt-BR`` is ``pt``, a profile), so ``brazilian`` is not reached.
# OpenSearch has one Norwegian analyzer, whose stemmer and stop words are
# Bokmål's: Nynorsk (``nn``) takes it as the closest one OpenSearch has
# (unmeasured against ``standard``; docs/guides/multilingual.md).
BUILT_IN_ANALYZERS: Mapping[str, str] = MappingProxyType(
    {
        "bg": "bulgarian",
        "bn": "bengali",
        "ca": "catalan",
        "ckb": "sorani",
        "cs": "czech",
        "da": "danish",
        "el": "greek",
        "et": "estonian",
        "eu": "basque",
        "fa": "persian",
        "fi": "finnish",
        "ga": "irish",
        "gl": "galician",
        "hu": "hungarian",
        "hy": "armenian",
        "id": "indonesian",
        "lt": "lithuanian",
        "lv": "latvian",
        "nb": "norwegian",
        "nl": "dutch",
        "nn": "norwegian",
        "no": "norwegian",
        "ro": "romanian",
        "sv": "swedish",
        "tr": "turkish",
    }
)

PROFILES: Mapping[LanguageCode, LanguageProfile] = MappingProxyType(
    {
        module.PROFILE.code: module.PROFILE
        for module in (ar, de, en, es, fr, he, hi, it, ja, ko, pt, ru, th, vi, zh)
    }
)


def profiles() -> Iterator[LanguageProfile]:
    """Every profile, in code order."""
    return iter(PROFILES.values())


def profile(code: LanguageCode | str) -> LanguageProfile:
    """The profile of ``code``: for a code without one (another BCP-47 tag,
    ``LanguageCode``), the defaults, named by the code; English's for text
    that is no code."""
    try:
        language = LanguageCode(code)
    except ValueError:
        return PROFILES[LanguageCode.EN]
    return PROFILES.get(language) or _default_profile(language)


@cache
def _default_profile(code: LanguageCode) -> LanguageProfile:
    english = PROFILES[LanguageCode.EN]
    return LanguageProfile(
        code=code,
        name=code.value,
        script="latin",
        analyzer=BUILT_IN_ANALYZERS.get(code.value, "standard"),
        decline=english.decline,
    )


def language_name(language: str) -> str:
    """The language as prompts name it: ``"ko"`` -> ``"Korean"``.

    A ``LanguageCode`` formats as ``LanguageCode.KO`` in a prompt template and
    a bare code is easy to misread, so prompts get the English name. Text that
    is not a supported code (e.g. a caller's ``"Brazilian Portuguese"``) is
    returned as given.
    """
    if isinstance(language, LanguageCode):
        return profile(language).name
    try:
        code = LanguageCode(language.strip().lower())
    except ValueError:
        return language
    # Another BCP-47 tag is named as given.
    return PROFILES[code].name if code in PROFILES else language
