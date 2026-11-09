# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script-aware token-count estimate, for models without a counting API.

Each character costs its script's share of a token
(``Script.chars_per_token`` of ``shared.utils.scripts``): a quarter for
Latin, digits, punctuation and every script not measured, one for Han,
Hangul and Kana (and CJK punctuation and full-width forms), and in between
for the scripts a BPE vocabulary splits finer than Latin (Cyrillic, Greek,
Arabic, Hebrew, Devanagari, Thai). Costs are counted in twelfths of a token
(:data:`UNITS_PER_TOKEN`), so the estimate is exact integer arithmetic.
"""

import math
import re
from collections.abc import Iterable
from functools import cache

from unified_kg_rag.domain.languages import LanguageCode, profile
from unified_kg_rag.domain.languages.detection import language_in_script, text_evidence
from unified_kg_rag.shared.utils.scripts import (
    DENSE_SCRIPT_TOKEN_RANGES,
    LATIN_CHARS_PER_TOKEN,
    SCRIPTS,
    Spacing,
    script_named,
)

# What a token is counted in: every rate of the script table is a whole
# number of twelfths of a token per character.
UNITS_PER_TOKEN = 12
# The cost of a Latin character (and of one of no script).
_BASE_UNITS = UNITS_PER_TOKEN // LATIN_CHARS_PER_TOKEN


def _units(chars_per_token: float) -> int:
    units = UNITS_PER_TOKEN / chars_per_token
    if units != int(units):
        raise ValueError(f"{chars_per_token} characters per token is no twelfth")
    return int(units)


def _body(ranges: Iterable[tuple[int, int]]) -> str:
    """``ranges`` as the body of a regex character class."""
    return "".join(f"\\U{lo:08X}-\\U{hi:08X}" for lo, hi in sorted(ranges))


def _class(ranges: list[tuple[int, int]]) -> re.Pattern[str]:
    return re.compile(f"[{_body(ranges)}]")


def _cost_classes() -> tuple[tuple[int, re.Pattern[str]], ...]:
    """One compiled character class per cost above the base, with that
    extra cost: one regex scan per class runs in C (the estimate runs over
    every context section of every query; a per-character Python loop cost
    ~1.5 ms per 1.5 KB section)."""
    by_cost: dict[int, list[tuple[int, int]]] = {
        _units(1.0): list(DENSE_SCRIPT_TOKEN_RANGES)
    }
    for script in SCRIPTS:
        if script.spacing is Spacing.DENSE:
            continue  # in DENSE_SCRIPT_TOKEN_RANGES
        units = _units(script.chars_per_token)
        if units != _BASE_UNITS:
            by_cost.setdefault(units, []).extend(script.ranges)
    return tuple(
        (units - _BASE_UNITS, _class(ranges)) for units, ranges in by_cost.items()
    )


_COST_CLASSES = _cost_classes()


def token_units(text: str) -> int:
    """What ``text`` costs in :data:`UNITS_PER_TOKEN` ths of a token: each
    character its script's share. Additive: a text costs what its pieces
    do."""
    units = _BASE_UNITS * len(text)
    if text.isascii():
        return units
    for extra, pattern in _COST_CLASSES:
        units += extra * len(pattern.findall(text))
    return units


def estimate_token_count(text: str) -> int:
    """Script-aware token-count estimate for the API-unavailable fallback.

    Bedrock's CountTokens API does not support embedding models, so embedding
    truncation always lands on this estimate. Nor does it support every language
    model: Claude 5 (and Opus 4.8) reject it on ``bedrock-runtime``, so context
    budgeting for those models lands here too. A flat ~4-chars-per-token estimate
    under-counts space-less dense scripts (CJK, kana, Hangul) ~4x — a whole
    Korean/Japanese sentence is few "words" and few chars-over-4 but many tokens
    — which let over-limit chunks slip past truncation and fail the embedding
    call — and Hindi, Thai or Hebrew 2-4x. We count each character at its
    script's rate (see the module), then floor with the whitespace word count
    so space-delimited text is never under-counted.
    """
    if not text:
        return 0
    return estimate_from_units(token_units(text), len(text.split()))


def estimate_from_units(units: int, words: int) -> int:
    """The estimate of a non-empty text costing ``units`` (:func:`token_units`)
    of ``words`` whitespace-split words: the whole tokens the units make,
    floored by the word count and by 1. The one formula of
    :func:`estimate_token_count` and of the chunker's running counts."""
    return max(words, units // UNITS_PER_TOKEN, 1)


# A budget sized by a measured rate is sized this much over it: a rate is
# one sentence's (the language matrix), a text's own varies around it.
BUDGET_MARGIN = 1.1
# The characters of a text its language is told from.
_LANGUAGE_SAMPLE_CHARS = 2000


@cache
def _script_letters(script: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """The letters of ``script`` (a ``LanguageProfile.script``), and the
    other letters and digits: a dense script's letters are Han, Hangul and
    Kana alike (Korean and Japanese write Han, and CJK punctuation and
    full-width forms go with them), as its measured rate counted them."""
    named = script_named(script)
    ranges = (
        DENSE_SCRIPT_TOKEN_RANGES if named.spacing is Spacing.DENSE else named.ranges
    )
    body = _body(ranges)
    return re.compile(f"[{body}]"), re.compile(f"[^\\W_{body}]")


def budget_token_count(
    text: str, tokenizer: str | None = None, multiplier: float = 1.0
) -> int:
    """The tokens a budget counts ``text`` as for a model without a
    counting API: the script estimate (:func:`estimate_token_count`) times
    ``multiplier``, and for a model of ``tokenizer`` at least what the
    characters per token measured for the text's language
    (``LanguageProfile.chars_per_token``: the language of the script most
    of the tokens of the text's first characters are in) count it as,
    :data:`BUDGET_MARGIN` over. That rate counts the letters of the
    language's script; other letters and digits
    (the Latin names, ids and numbers of a Korean text) count at English's,
    and spaces and punctuation at each side's share of the letters. The
    script rates are a BPE vocabulary's, which counts German or Vietnamese
    text in about half the tokens Claude's tokenizer does
    (docs/concepts/defaults-and-evidence.md#token-estimate). Chunking keeps
    the script estimate (stored chunks)."""
    estimate = estimate_token_count(text)
    if multiplier != 1.0:
        estimate = math.ceil(estimate * multiplier)
    if tokenizer is None or not text:
        return estimate
    evidence = text_evidence([text[:_LANGUAGE_SAMPLE_CHARS]])
    script = evidence.counts.dominant()
    language = profile(
        (language_in_script(evidence, script, LanguageCode.EN) if script else None)
        or LanguageCode.EN
    )
    rate = language.chars_per_token.get(tokenizer)
    if rate is None:
        return estimate
    other_rate = profile(LanguageCode.EN).chars_per_token.get(tokenizer, rate)
    own_letters, other_letters = _script_letters(language.script)
    own = len(own_letters.findall(text))
    other = len(other_letters.findall(text))
    # Spaces and punctuation, shared as the letters are.
    rest = len(text) - own - other
    share = own / (own + other) if own + other else 1.0
    tokens = (own + rest * share) / rate + (other + rest * (1 - share)) / other_rate
    return max(estimate, math.ceil(tokens * BUDGET_MARGIN))


def truncate_to_estimate(text: str, max_tokens: int) -> str:
    """``text`` cut so its estimated token count is at most ``max_tokens``."""
    if estimate_token_count(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    # Largest prefix that fits; the estimate is monotone in the prefix length.
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_token_count(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]
