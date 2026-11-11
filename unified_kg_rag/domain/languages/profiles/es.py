# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spanish."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

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
        "el los las qué quién cuál cómo como cuándo dónde es fue y del cuánto "
        "cuántos cuánta cuántas cuanto cuantos cuanta cuantas hay".split()
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
        "trece": 13,
        "catorce": 14,
        "quince": 15,
        "dieciséis": 16,
        "dieciseis": 16,
        "diecisiete": 17,
        "dieciocho": 18,
        "diecinueve": 19,
        "veintiuno": 21,
        "veintiún": 21,
        "veintiuna": 21,
        "veintidós": 22,
        "veintidos": 22,
        "veintitrés": 23,
        "veintitres": 23,
        "veinticuatro": 24,
        "veinticinco": 25,
        "veintiséis": 26,
        "veintiseis": 26,
        "veintisiete": 27,
        "veintiocho": 28,
        "veintinueve": 29,
    },
    number_tens={
        "veinte": 20,
        "treinta": 30,
        "cuarenta": 40,
        "cincuenta": 50,
        "sesenta": 60,
        "setenta": 70,
        "ochenta": 80,
        "noventa": 90,
        "cien": 100,
        "ciento": 100,
        # Masculine and feminine ("doscientos", "doscientas").
        **{
            f"{stem}ient{ending}s": value
            for stem, value in (
                ("dosc", 200),
                ("tresc", 300),
                ("cuatroc", 400),
                ("quin", 500),
                ("seisc", 600),
                ("setec", 700),
                ("ochoc", 800),
                ("novec", 900),
            )
            for ending in "oa"
        },
    },
    number_scales={
        "mil": 1000,
        "millón": 10**6,
        "millon": 10**6,
        "millones": 10**6,
    },
    number_conjunctions=("y",),
    compound_ones=("un", "una", "uno"),
    zero_words=("ningún", "ninguno", "ninguna", "ningun"),
    absence_phrases=("no hay", "no tiene", "no tienen", "no existe", "no existen"),
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
    legal_forms={
        "sa": "sa",
        "sl": "sl",
        "slu": "slu",
        # Mexico's variable-capital forms.
        "sa de cv": "sa de cv",
        "sab de cv": "sab de cv",
        "s de rl": "s de rl",
        "s de rl de cv": "s de rl de cv",
    },
    quote_pairs=(("«", "»"),),
    chars_per_token={CLAUDE_TOKENIZER: 2.84},
)
