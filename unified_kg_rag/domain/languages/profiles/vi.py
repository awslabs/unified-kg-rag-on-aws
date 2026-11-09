# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vietnamese."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.VI,
    name="Vietnamese",
    script="latin",
    # OpenSearch has no built-in Vietnamese analyzer (a plugin's can be
    # named in language_analyzers).
    analyzer="standard",
    decline=(
        "Tôi không tìm thấy thông tin liên quan trong dữ liệu hiện có để trả lời "
        "câu hỏi này."
    ),
    function_words=frozenset(
        "là gì của và những các được không có trong cho với này nào ai đâu khi "
        "bao nhiêu người".split()
    ),
    # "unknown", "none", "not specified".
    no_value_words=("khôngrõ", "khôngcó", "khôngxácđịnh"),
    chars_per_token={CLAUDE_TOKENIZER: 1.75},
)
