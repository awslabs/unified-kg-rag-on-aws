# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hebrew."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.HE,
    name="Hebrew",
    script="hebrew",
    # OpenSearch has no built-in Hebrew analyzer (a plugin's, such as
    # hebmorph, can be named in language_analyzers).
    analyzer="standard",
    decline="לא מצאתי מידע רלוונטי בנתונים הזמינים כדי לענות על שאלה זו.",
    # "unknown", "none", "not specified".
    no_value_words=("לאידוע", "אין", "לאצוין"),
    # Limited company (בע"מ, its quote mark dropped as entity keys drop it).
    legal_forms={"בעמ": "בעמ"},
    # The prefixed particles: "in", "the", "and", "as", "to", "from", "that".
    proclitics=("ב", "ה", "ו", "כ", "ל", "מ", "ש"),
    quote_pairs=(("„", "”"), ('"', '"')),
    chars_per_token={CLAUDE_TOKENIZER: 1.54},
)
