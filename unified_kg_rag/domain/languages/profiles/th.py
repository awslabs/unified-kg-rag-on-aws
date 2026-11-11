# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thai."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.TH,
    name="Thai",
    script="thai",
    # Built in: dictionary word segmentation (Thai writes no spaces between
    # words).
    analyzer="thai",
    decline="ไม่พบข้อมูลที่เกี่ยวข้องในข้อมูลที่มีอยู่เพื่อตอบคำถามนี้",
    # "unknown", "none", "not specified".
    no_value_words=("ไม่ทราบ", "ไม่มี", "ไม่ระบุ"),
    # "Company ... Limited" (public: "จำกัด (มหาชน)").
    legal_forms={"จำกัด": "จำกัด", "จำกัด (มหาชน)": "จำกัด (มหาชน)"},
    leading_legal_forms={"บริษัท": "บริษัท"},
    quote_pairs=(("“", "”"),),
    chars_per_token={CLAUDE_TOKENIZER: 1.09},
)
