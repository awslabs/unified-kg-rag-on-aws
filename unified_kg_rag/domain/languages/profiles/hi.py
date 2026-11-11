# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hindi."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.HI,
    name="Hindi",
    script="devanagari",
    analyzer="hindi",
    decline=(
        "उपलब्ध डेटा में इस प्रश्न का उत्तर देने के लिए प्रासंगिक जानकारी नहीं मिली।"
    ),
    # "unknown", "none", "not applicable".
    no_value_words=("अज्ञात", "कोईनहीं", "लागूनहीं"),
    legal_forms={
        "प्राइवेट लिमिटेड": "pvt ltd",
        "लिमिटेड": "ltd",
    },
    quote_pairs=(("“", "”"),),
    chars_per_token={CLAUDE_TOKENIZER: 1.35},
)
