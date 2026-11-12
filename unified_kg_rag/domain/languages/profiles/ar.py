# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arabic."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

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
    # Limited liability and joint-stock companies (dot-free: "ذ.م.م").
    legal_forms={"ذمم": "ذمم", "شمم": "شمم", "شمع": "شمع"},
    # "Company", written before the name ("شركة النور").
    leading_legal_forms={"شركة": "شركة"},
    # The prefixed particles and the article: "and", "so", "with", "to",
    # "as", "the" ("وبالقاهرة" is "و", "ب", "ال" and "قاهرة").
    proclitics=("و", "ف", "ب", "ل", "ك", "ال"),
    # "لا" is also the negation ("لا يوجد").
    polarity_words={"نعم": "yes", "بلى": "yes", "لا": "no", "كلا": "no"},
    ambiguous_polarity_words={"لا": ""},
    quote_pairs=(("«", "»"),),
    chars_per_token={CLAUDE_TOKENIZER: 1.75},
)
