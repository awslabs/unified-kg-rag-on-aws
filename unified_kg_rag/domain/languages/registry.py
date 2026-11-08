# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language profiles, by code.

Adding a language: a module under ``profiles/`` with its ``PROFILE``, its
code in ``LanguageCode``, and an entry below; ``tests/unit/test_languages.py``
lists what each profile holds and which gaps are known.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile
from unified_kg_rag.domain.languages.profiles import (
    de,
    en,
    es,
    fr,
    it,
    ja,
    ko,
    pt,
    ru,
    zh,
)

__all__ = ["PROFILES", "language_name", "profile", "profiles"]

PROFILES: Mapping[LanguageCode, LanguageProfile] = MappingProxyType(
    {
        module.PROFILE.code: module.PROFILE
        for module in (de, en, es, fr, it, ja, ko, pt, ru, zh)
    }
)


def profiles() -> Iterator[LanguageProfile]:
    """Every profile, in code order."""
    return iter(PROFILES.values())


def profile(code: LanguageCode | str) -> LanguageProfile:
    """The profile of ``code``; English's for a code without one."""
    try:
        return PROFILES[LanguageCode(code)]
    except ValueError:
        return PROFILES[LanguageCode.EN]


def language_name(language: str) -> str:
    """The language as prompts name it: ``"ko"`` -> ``"Korean"``.

    A ``LanguageCode`` formats as ``LanguageCode.KO`` in a prompt template and
    a bare code is easy to misread, so prompts get the English name. Text that
    is not a supported code (e.g. a caller's ``"Brazilian Portuguese"``) is
    returned as given.
    """
    if isinstance(language, LanguageCode):
        return PROFILES[language].name
    try:
        return PROFILES[LanguageCode(language.strip().lower())].name
    except ValueError:
        return language
