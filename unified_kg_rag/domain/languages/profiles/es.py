# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spanish."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.ES,
    name="Spanish",
    script="latin",
    analyzer="spanish",
    decline=(
        "No encontré información relevante en los datos disponibles para "
        "responder a esta pregunta."
    ),
    function_words=frozenset(
        "el los las qué quién cuál cómo cuándo dónde es fue y del".split()
    ),
    no_value_words=("desconocido", "ninguno"),
    articles=frozenset("el la los las un una unos unas".split()),
    number_words={
        "cero": 0,
        "dos": 2,
        "tres": 3,
        "cuatro": 4,
        "cinco": 5,
        "seis": 6,
        "siete": 7,
        "ocho": 8,
        "nueve": 9,
        "diez": 10,
        "once": 11,
        "doce": 12,
    },
    month_words={
        "January": ("enero", "ene"),
        "February": ("febrero",),
        "March": ("marzo",),
        "April": ("abril", "abr"),
        "May": ("mayo",),
        "June": ("junio",),
        "July": ("julio",),
        "August": ("agosto", "ago"),
        "September": ("septiembre", "setiembre"),
        "October": ("octubre",),
        "November": ("noviembre",),
        "December": ("diciembre", "dic"),
    },
    # "15 de marzo de 2024".
    date_fillers=("de", "del"),
    legal_forms={"sa": "sa"},
)
