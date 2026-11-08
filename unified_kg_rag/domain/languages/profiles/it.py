# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Italian."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.IT,
    name="Italian",
    script="latin",
    analyzer="italian",
    decline=(
        "Non ho trovato informazioni pertinenti nei dati disponibili per "
        "rispondere a questa domanda."
    ),
    function_words=frozenset(
        "gli che chi quale come perché dove è sono della nel uno".split()
    ),
    no_value_words=("sconosciuto", "nessuno"),
    articles=frozenset("il lo la i gli le l un uno una".split()),
    elided_articles=frozenset({"l", "un"}),
    number_words={
        "zero": 0,
        "due": 2,
        "tre": 3,
        "quattro": 4,
        "cinque": 5,
        "sei": 6,
        "sette": 7,
        "otto": 8,
        "nove": 9,
        "dieci": 10,
        "undici": 11,
        "dodici": 12,
    },
    month_words={
        "January": ("gennaio", "gen"),
        "February": ("febbraio",),
        "March": ("marzo",),
        "April": ("aprile",),
        "May": ("maggio", "mag"),
        "June": ("giugno", "giu"),
        "July": ("luglio", "lug"),
        "August": ("agosto", "ago"),
        "September": ("settembre", "set"),
        "October": ("ottobre", "ott"),
        "November": ("novembre",),
        "December": ("dicembre", "dic"),
    },
    # S.p.A. counts only with its dots (legal_forms), since "spa" is a word.
    legal_forms={"srl": "srl", "spa": "spa"},
)
