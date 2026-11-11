# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""German."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.DE,
    name="German",
    script="latin",
    analyzer="german",
    decline=(
        "Ich konnte in den verfügbaren Daten keine relevanten Informationen "
        "finden, um diese Frage zu beantworten."
    ),
    function_words=frozenset(
        "der die das was wer wie warum wann wo ist sind war wurde ein eine und "
        "mit von nicht welche welcher gibt kein keine".split()
    ),
    no_value_words=("unbekannt", "keine", "keiner"),
    articles=frozenset(
        "der die das den dem des ein eine einen einem einer eines".split()
    ),
    number_words={
        "null": 0,
        "eins": 1,
        "zwei": 2,
        "drei": 3,
        "vier": 4,
        "fünf": 5,
        "sechs": 6,
        "sieben": 7,
        "acht": 8,
        "neun": 9,
        "zehn": 10,
        "elf": 11,
        "zwölf": 12,
        "dreizehn": 13,
        "vierzehn": 14,
        "fünfzehn": 15,
        "sechzehn": 16,
        "siebzehn": 17,
        "achtzehn": 18,
        "neunzehn": 19,
    },
    # Written as one word, ones before tens ("vierundzwanzig",
    # "zweitausenddreihundert").
    number_tens={
        "zwanzig": 20,
        "dreißig": 30,
        "dreissig": 30,
        "vierzig": 40,
        "fünfzig": 50,
        "sechzig": 60,
        "siebzig": 70,
        "achtzig": 80,
        "neunzig": 90,
    },
    number_scales={
        "hundert": 100,
        "tausend": 1000,
        "million": 10**6,
        "millionen": 10**6,
        "milliarde": 10**9,
        "milliarden": 10**9,
    },
    number_conjunctions=("und",),
    compound_ones=("ein", "eine"),
    zero_words=("kein", "keine", "keinen", "keinem", "keiner", "keines"),
    ones_before_tens=True,
    month_words={
        "January": ("januar", "jänner"),
        "February": ("februar", "feber"),
        "March": ("märz", "maerz", "mär"),
        "May": ("mai",),
        "June": ("juni",),
        "July": ("juli",),
        "October": ("oktober", "okt"),
        "December": ("dezember", "dez"),
    },
    legal_forms={
        "gmbh": "gmbh",
        "ag": "ag",
        "kg": "kg",
        "ohg": "ohg",
        "ug": "ug",
        # Not SE (Societas Europaea): written without dots, it also names a
        # product's edition ("iPhone SE", "Java SE").
        "ev": "ev",
        # A limited partnership whose general partner is a company.
        "gmbh & co kg": "gmbh & co kg",
        "ag & co kg": "ag & co kg",
        "ug & co kg": "ug & co kg",
    },
    quote_pairs=(("„", "“"), ("‚", "‘"), ("»", "«")),
    chars_per_token={CLAUDE_TOKENIZER: 2.05},
)
