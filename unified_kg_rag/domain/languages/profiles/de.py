# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""German."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile

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
        "mit von nicht welche welcher".split()
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
    },
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
    legal_forms={"gmbh": "gmbh", "ag": "ag"},
)
