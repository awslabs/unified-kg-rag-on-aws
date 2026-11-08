# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language codes the configuration accepts.

A member is a language with a profile in the registry (``domain.languages``).
Another well-formed BCP-47 tag ("sw", "fa", "tl") is accepted too, as a
code of its own (``LanguageCode("sw").value == "sw"``) with one WARNING: it
gets the defaults of a language without a profile (the ``standard``
analyzer, the English decline, prompts naming it by its code). A tag reads
as its language subtag: "pt-BR" is ``LanguageCode.PT``, "zh-Hant"
``LanguageCode.ZH``, "sw-KE" ``LanguageCode("sw")``.
"""

from __future__ import annotations

import re
from enum import Enum

from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.logging import warn_once

__all__ = ["LanguageCode"]

logger = get_logger(__name__)

# A BCP-47 language tag, lowercased: a two- or three-letter primary subtag,
# then script, region or variant subtags.
_TAG = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{1,8})*")


class LanguageCode(str, Enum):
    """A language code: one with a profile, or another BCP-47 tag."""

    AR = "ar"
    DE = "de"
    EN = "en"
    ES = "es"
    FR = "fr"
    HE = "he"
    HI = "hi"
    IT = "it"
    JA = "ja"
    KO = "ko"
    PT = "pt"
    RU = "ru"
    TH = "th"
    VI = "vi"
    ZH = "zh"

    @classmethod
    def _missing_(cls, value: object) -> LanguageCode | None:
        if not isinstance(value, str):
            return None
        tag = value.strip().lower().replace("_", "-")
        if not _TAG.fullmatch(tag):
            return None
        # The language subtag: a profile is of a language, not of a region.
        tag = tag.split("-", 1)[0]
        member = cls._value2member_map_.get(tag)
        if isinstance(member, cls):
            return member
        # A code of its own, created once per tag and kept as the profiled
        # ones are, so ``LanguageCode("sw") is LanguageCode("sw")``. It is
        # not iterated (``list(LanguageCode)`` lists the profiled ones).
        member = str.__new__(cls, tag)
        member._name_ = tag.upper().replace("-", "_")
        member._value_ = tag
        member = cls._value2member_map_.setdefault(tag, member)
        warn_once(
            logger,
            f"language-code:{tag}",
            "Language code '%s' has no profile in unified_kg_rag.domain.languages: "
            "its text fields use indexing.opensearch.default_analyzer unless "
            "language_analyzers maps it, its declines are in English and prompts "
            "name the language '%s'",
            tag,
            tag,
        )
        return member  # type: ignore[return-value]
