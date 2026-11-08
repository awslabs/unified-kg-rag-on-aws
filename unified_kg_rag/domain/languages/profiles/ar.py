# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arabic."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.AR,
    name="Arabic",
    script="arabic",
    analyzer="arabic",
    decline=(
        "لم أتمكن من العثور على معلومات ذات صلة في البيانات المتاحة للإجابة عن "
        "هذا السؤال."
    ),
    # "unknown", "none", "not specified" (without spaces, as placeholders
    # are compared).
    no_value_words=("غيرمعروف", "لايوجد", "غيرمحدد"),
    # "Company", written before the name ("شركة النور").
    leading_legal_forms={"شركة": "شركة"},
)
