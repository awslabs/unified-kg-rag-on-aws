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
