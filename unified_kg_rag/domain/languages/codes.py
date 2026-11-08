# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language codes the configuration accepts."""

from __future__ import annotations

from enum import Enum

__all__ = ["LanguageCode"]


class LanguageCode(str, Enum):
    """A language with a profile in the registry (``domain.languages``)."""

    DE = "de"
    EN = "en"
    ES = "es"
    FR = "fr"
    IT = "it"
    JA = "ja"
    KO = "ko"
    PT = "pt"
    RU = "ru"
    ZH = "zh"
