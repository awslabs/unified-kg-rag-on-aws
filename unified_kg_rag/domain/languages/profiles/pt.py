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
        "os quem qual porque onde é são foi com do uma não".split()
    ),
    no_value_words=("desconhecido", "nenhum"),
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
