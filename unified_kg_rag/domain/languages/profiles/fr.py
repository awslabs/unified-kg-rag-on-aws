# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""French."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.FR,
    name="French",
    script="latin",
    analyzer="french",
    decline=(
        "Je n'ai trouvé aucune information pertinente dans les données "
        "disponibles pour répondre à cette question."
    ),
    function_words=frozenset(
        "les des qui quel quelle comment pourquoi quand où est sont était et "
        "avec du une dans".split()
    ),
    no_value_words=("inconnu", "inconnue", "aucun", "aucune"),
    articles=frozenset("le la les l un une des".split()),
    # Le, la, de, je, me, ne, se, te, ce, que and its compounds.
    elided_prefixes=frozenset(
        "l d j m n s t c qu jusqu lorsqu puisqu quoiqu presqu".split()
    ),
    number_words={
        "zéro": 0,
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
        "six": 6,
        "sept": 7,
        "huit": 8,
        "neuf": 9,
        "dix": 10,
        "onze": 11,
        "douze": 12,
    },
    month_words={
        "January": ("janvier", "janv"),
        "February": ("février", "févr", "fevrier"),
        "March": ("mars",),
        "April": ("avril", "avr"),
        "May": ("mai",),
        "June": ("juin",),
        "July": ("juillet", "juil"),
        "August": ("août", "aout"),
        "September": ("septembre",),
        "October": ("octobre",),
        "November": ("novembre",),
        "December": ("décembre", "déc", "decembre"),
    },
    legal_forms={"sa": "sa", "sas": "sas", "sarl": "sarl"},
)
