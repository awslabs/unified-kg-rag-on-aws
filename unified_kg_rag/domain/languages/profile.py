# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the package knows of one language: one record per language code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from unified_kg_rag.domain.languages.codes import LanguageCode

__all__ = ["CLAUDE_TOKENIZER", "LanguageProfile"]

# The tokenizer of Claude Opus 4.7 and later (Opus 4.8/5/5.5, Sonnet 5/5.5,
# Haiku 5.5), which CountTokens does not count on bedrock-runtime: its
# characters per token by language (``LanguageProfile.chars_per_token``)
# were measured on Bedrock on 2026-11-08 (Sonnet and Haiku 5.5 count alike;
# docs/concepts/defaults-and-evidence.md#token-estimate).
CLAUDE_TOKENIZER = "claude-opus-4.7"


@dataclass(frozen=True)
class LanguageProfile:
    """One language's words and settings, read wherever text in it is
    analyzed, matched or written.

    A word list holds the language's own words only; a consumer reads the
    union over every profile (a model may answer in any of them). Every
    word is as its consumer compares it: casefolded, and for the claim and
    placeholder words without spaces or punctuation.
    """

    code: LanguageCode
    # The English name prompts use ("Write in Korean").
    name: str
    # The script its letters are in, a name of shared.utils.scripts.SCRIPTS.
    script: str
    # The OpenSearch analyzer of its text fields (built in, or a plugin's).
    analyzer: str
    # The fixed reply of a query declined without a model call.
    decline: str
    # Frequent function words no other language of its script shares, which
    # tell a short query's language apart from another of that script.
    function_words: frozenset[str] = frozenset()
    # Words a model writes for a claim status, by the status they state
    # (``claim_resolver.CLAIM_STATUSES``).
    claim_statuses: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Words a model writes for "no value" (a claim object or date).
    no_value_words: tuple[str, ...] = ()
    # Words for an open or undecided date ("ongoing", "TBD").
    open_date_words: tuple[str, ...] = ()
    # Its month names and abbreviations, by the English month name.
    month_words: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Words a written-out date carries around its parts ("15 de marzo").
    date_fillers: tuple[str, ...] = ()
    # Legal-form designators written as words, dot-free, each with the
    # canonical short form its spellings share ("corporation" -> "corp").
    legal_forms: Mapping[str, str] = field(default_factory=dict)
    # Those written as words before the name ("شركة"), which linking a
    # query's name reads (``domain.retrieval.linking``).
    leading_legal_forms: Mapping[str, str] = field(default_factory=dict)
    # Legal-form designators in Han, Hangul or Kana, leading or trailing.
    dense_legal_forms: tuple[str, ...] = ()
    # Its articles, which answer normalization drops ("der Rhein" is
    # "Rhein").
    articles: frozenset[str] = frozenset()
    # The words it writes elided before a vowel, apostrophe dropped
    # ("l'accord", "d'Avignon", "dell'accordo"): split from the next word
    # (an article among them is then dropped), and a sign of the language.
    elided_prefixes: frozenset[str] = frozenset()
    # Its number words with their values, which an answer states a count by
    # ("drei"): the words below twenty a tens word composes with
    # ("twenty-one"), and the ones it writes as one word below a hundred
    # ("veinticuatro", "ventotto"); a form of one that is also the
    # indefinite article ("ein", "un") is left out, as English's "a" is.
    number_words: Mapping[str, int] = field(default_factory=dict)
    # Its tens words ("twenty", Korean "열") with the hundreds it writes as
    # one word ("doscientos", "двести"), and its scale words ("hundred",
    # "Millionen", "тысячи"), by value; the words that state no item at all
    # ("none").
    number_tens: Mapping[str, int] = field(default_factory=dict)
    number_scales: Mapping[str, int] = field(default_factory=dict)
    zero_words: tuple[str, ...] = ()
    # The words that join the parts of a written number ("and" in "three
    # hundred and twelve", "y" in "treinta y dos", "und" in
    # "vierundzwanzig"), and the forms of one left out of ``number_words``
    # as articles, which count inside a written number only ("vingt et
    # un", "einhundert", "un millón").
    number_conjunctions: tuple[str, ...] = ()
    compound_ones: tuple[str, ...] = ()
    # Numerals that state a count only before one of its ``counters``
    # (Korean "세 곳", Han "三社"), by value, and the multipliers ("十" in
    # "十二", "万" in "一万二千", Korean "만"; after digits too: "1만 2천").
    counter_numerals: Mapping[str, int] = field(default_factory=dict)
    numeral_multipliers: Mapping[str, int] = field(default_factory=dict)
    # Digits that state a number only beside one of its
    # ``numeral_multipliers`` (Sino-Korean "이" in "십이", "삼만"): alone
    # they are other words ("이" is "this").
    multiplier_numerals: Mapping[str, int] = field(default_factory=dict)
    counters: tuple[str, ...] = ()
    # Yes/no answer words, each with its polarity ("yes" or "no").
    polarity_words: Mapping[str, str] = field(default_factory=dict)
    # The quote pairs its text wraps a name in (opening, closing).
    quote_pairs: tuple[tuple[str, str], ...] = ()
    # Characters per token of a text in it (spaces and punctuation
    # included) by the tokenizer that counts them (a
    # ``LanguageModelInfo.tokenizer``, such as ``CLAUDE_TOKENIZER``), where
    # measured: a budget for a model of that tokenizer is sized by it
    # (``adapters.llm.token_estimate.budget_token_count``).
    chars_per_token: Mapping[str, float] = field(default_factory=dict)
