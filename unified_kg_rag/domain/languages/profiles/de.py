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
