# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language registry: one :class:`LanguageProfile` per language code.

Everything the package knows of a language lives in its profile under
``profiles/``: its name, script, OpenSearch analyzer, decline reply, and the
words text in it is matched by (function words, claim status words,
placeholders, month names, legal forms, yes/no words, quote pairs). A
consumer reads the profiles instead of a word list of its own, so adding a
language is one module. Which language a text is in, as far as its script
and function words tell, is :mod:`.detection`'s; a corpus sample's, against
the configured source language, :mod:`.corpus`'s (by the same rule).

Pure: imports only ``shared`` (``shared.utils.scripts`` holds the script
table the profiles name).
"""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.detection import dense_language, text_language
from unified_kg_rag.domain.languages.profile import LanguageProfile
from unified_kg_rag.domain.languages.registry import (
    PROFILES,
    language_name,
    profile,
    profiles,
)

__all__ = [
    "PROFILES",
    "LanguageCode",
    "LanguageProfile",
    "dense_language",
    "language_name",
    "profile",
    "profiles",
    "text_language",
]
