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
        "gli che chi quale come perché dove è sono della nel uno quanto quanta "
        "quanti quante cosa per sul alla delle degli dei nella questo "
        "questa".split()
    ),
    no_value_words=("sconosciuto", "nessuno"),
    articles=frozenset("il lo la i gli le l un uno una".split()),
    # The articles and the prepositions joined to them, di, ci, quello,
    # bello, santo, come, anche, dove, quanto, cosa.
    elided_prefixes=frozenset(
        "l un d c dell all dall nell sull coll quell bell sant com anch dov quant "
        "cos".split()
    ),
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
        "tredici": 13,
        "quattordici": 14,
        "quindici": 15,
        "sedici": 16,
        "diciassette": 17,
        "diciotto": 18,
        "diciannove": 19,
        # A tens word drops its vowel before uno and otto ("ventotto"), and
        # tre takes an accent closing one ("ventitré").
        **{
            f"{stem}{ones}": tens + value
            for stem, tens in (
                ("vent", 20),
                ("trent", 30),
                ("quarant", 40),
                ("cinquant", 50),
                ("sessant", 60),
                ("settant", 70),
                ("ottant", 80),
                ("novant", 90),
            )
            for ones, value in (("uno", 1), ("otto", 8))
        },
        "tré": 3,
    },
    # Written as one word ("duecentotrentadue").
    number_tens={
        "venti": 20,
        "trenta": 30,
        "quaranta": 40,
        "cinquanta": 50,
        "sessanta": 60,
        "settanta": 70,
        "ottanta": 80,
        "novanta": 90,
    },
    number_scales={
        "cento": 100,
        "mille": 1000,
        "mila": 1000,
        "milione": 10**6,
        "milioni": 10**6,
        "miliardo": 10**9,
        "miliardi": 10**9,
    },
    compound_ones=("un", "uno", "una"),
    # "e" joins the parts of a written number ("mille e cinquecento").
    number_conjunctions=("e",),
    zero_words=("nessun", "nessuno", "nessuna"),
    absence_phrases=(
        "non ci sono",
        "non c'è",
        "non ha",
        "non hanno",
        "non esiste",
        "non esistono",
    ),
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
