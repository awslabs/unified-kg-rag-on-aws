# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""English."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.EN,
    name="English",
    script="latin",
    analyzer="english",
    decline=(
        "I could not find relevant information in the available data to answer "
        "this question."
    ),
    function_words=frozenset(
        "the what which who whom whose how why when where is are were does did "
        "has have had can could would should of and with from this that".split()
    ),
    # The claim prompt's words, MS GraphRAG's SUSPECTED, and the few words
    # that state the same status plainly.
    claim_statuses={
        "true": ("true", "fact", "factual", "confirmed", "verified"),
        "false": ("false", "untrue", "denied", "refuted"),
        "disputed": ("disputed", "suspected", "uncertain", "unverified"),
        "unknown": ("unknown", "none", "na"),
    },
    no_value_words=(
        "none",
        "unknown",
        "n/a",
        "na",
        "null",
        "nil",
        "unspecified",
        "notapplicable",
    ),
    articles=frozenset({"a", "an", "the"}),
    number_words={
        word: value
        for value, word in enumerate(
            "zero one two three four five six seven eight nine ten eleven "
            "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
            "nineteen".split()
        )
    },
    number_tens=dict(
        zip(
            "twenty thirty forty fifty sixty seventy eighty ninety".split(),
            range(20, 100, 10),
            strict=True,
        )
    ),
    number_scales={
        "hundred": 100,
        "thousand": 1000,
        "million": 10**6,
        "billion": 10**9,
    },
    zero_words=("none", "no"),
    absence_phrases=("not … any", "-n't … any"),
    comparatives=("more than", "fewer than", "less than"),
    number_conjunctions=("and",),
    ordinal_words=frozenset(
        "first second third fourth fifth sixth seventh eighth ninth".split()
    ),
    # Also the words that place a year in its course ("in mid-2024").
    year_words=(
        "in",
        "as of",
        "since",
        "until",
        "till",
        "from",
        "before",
        "after",
        "during",
        "circa",
        "early",
        "mid",
        "late",
    ),
    partitive_words=("of the", "of these", "of those", "of all"),
    partitive_anaphors=(
        "of them",
        "of these",
        "of those",
        "of which",
        "of whom",
        "of us",
        "of its",
        "of their",
    ),
    open_date_words=("tbd", "ongoing", "present"),
    # strptime reads the English month names itself.
    legal_forms={
        "inc": "inc",
        "incorporated": "inc",
        "corp": "corp",
        "corporation": "corp",
        "co": "co",
        "company": "co",
        "ltd": "ltd",
        "limited": "ltd",
        "llc": "llc",
        "llp": "llp",
        "lp": "lp",
        "plc": "plc",
        "pty": "pty",
        "pte": "pte",
        # Dutch and Belgian forms, which English text writes unchanged.
        "bv": "bv",
        "nv": "nv",
    },
    polarity_words={"yes": "yes", "no": "no", "nope": "no"},
    quote_pairs=(('"', '"'), ("“", "”")),
    chars_per_token={CLAUDE_TOKENIZER: 3.25},
)
