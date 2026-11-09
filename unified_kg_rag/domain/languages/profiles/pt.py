# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portuguese."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.PT,
    name="Portuguese",
    script="latin",
    analyzer="portuguese",
    decline=(
        "Não encontrei informações relevantes nos dados disponíveis para "
        "responder a esta pergunta."
    ),
    function_words=frozenset(
        "os quem qual porque onde é são foi com do uma não quantos quantas como "
        "tem há".split()
    ),
    no_value_words=("desconhecido", "nenhum"),
    articles=frozenset("o a os as um uma uns umas".split()),
    number_words={
        "zero": 0,
        "dois": 2,
        "duas": 2,
        "três": 3,
        "quatro": 4,
        "cinco": 5,
        "seis": 6,
        "sete": 7,
        "oito": 8,
        "nove": 9,
        "dez": 10,
        "onze": 11,
        "doze": 12,
        "treze": 13,
        "catorze": 14,
        "quatorze": 14,
        "quinze": 15,
        "dezesseis": 16,
        "dezasseis": 16,
        "dezessete": 17,
        "dezassete": 17,
        "dezoito": 18,
        "dezenove": 19,
        "dezanove": 19,
    },
    number_tens={
        "vinte": 20,
        "trinta": 30,
        "quarenta": 40,
        "cinquenta": 50,
        "sessenta": 60,
        "setenta": 70,
        "oitenta": 80,
        "noventa": 90,
        "cem": 100,
        "cento": 100,
        # Masculine and feminine ("duzentos", "duzentas").
        **{
            f"{stem}ent{ending}s": value
            for stem, value in (
                ("duz", 200),
                ("trez", 300),
                ("quatroc", 400),
                ("quinh", 500),
                ("seisc", 600),
                ("setec", 700),
                ("oitoc", 800),
                ("novec", 900),
            )
            for ending in "oa"
        },
    },
    number_scales={
        "mil": 1000,
        "milhão": 10**6,
        "milhao": 10**6,
        "milhões": 10**6,
        "milhoes": 10**6,
        "bilhão": 10**9,
        "bilhões": 10**9,
    },
    number_conjunctions=("e",),
    compound_ones=("um", "uma"),
    zero_words=("nenhum", "nenhuma"),
    absence_phrases=("não há", "não tem", "não têm", "não existe", "não existem"),
    comparatives=("mais de", "menos de", "mais que", "menos que"),
    year_words=("em", "desde", "até", "antes de", "depois de", "a partir de"),
    partitive_words=("dos", "das", "destes", "destas", "entre os", "entre as"),
    partitive_anaphors=("deles", "delas", "dos quais", "das quais"),
    month_words={
        "January": ("janeiro",),
        "February": ("fevereiro", "fev"),
        "March": ("março", "marco"),
        "April": ("abril", "abr"),
        "May": ("maio",),
        "June": ("junho",),
        "July": ("julho",),
        "August": ("agosto", "ago"),
        "September": ("setembro", "set"),
        "October": ("outubro", "out"),
        "November": ("novembro",),
        "December": ("dezembro", "dez"),
    },
    date_fillers=("de",),
    legal_forms={"ltda": "ltda", "sa": "sa"},
)
