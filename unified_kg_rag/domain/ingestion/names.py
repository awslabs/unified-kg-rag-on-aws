# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Name and type keys of entity matching: which spellings are one entity
(``compact_name_key``) and which names a fuzzy match may never link
(``discriminator_tokens``, ``head_character``, ``entity_types_compatible``).

Pure string functions with no matching engine behind them, so the query
path, the stores and the evaluators key names exactly as entity resolution
does without importing the resolver (whose MinHash index loads datasketch
and SciPy).
"""

import re

from unified_kg_rag.domain.languages.legal_forms import strip_legal_forms
from unified_kg_rag.shared.utils import entity_key, normalize_name
from unified_kg_rag.shared.utils.scripts import (
    drop_dense_script_spaces,
    is_dense_script_char,
)

__all__ = [
    "compact_name_key",
    "discriminator_tokens",
    "entity_types_compatible",
    "head_character",
    "normalize_entity_type",
]

# Multi-letter Roman numerals ("ii", "iv", "xii"); single letters ("i", "v",
# "x") are already covered by the single-character rule below. This also matches
# a few real words ("vi", "xi", "iv"), which is accepted: a spurious
# discriminator only *blocks* a fuzzy merge (both names stay separate entities;
# exact-key merges are unaffected), whereas dropping the rule would merge
# "Annex II" into "Annex III" or "Phase II" into "Phase III".
_RE_ROMAN_NUMERAL = re.compile(r"x{0,3}(?:ix|iv|v?i{0,3})")

# Entity types that carry no information and so never block a merge.
_GENERIC_ENTITY_TYPES = frozenset({"", "unknown"})


def compact_name_key(name: str | None) -> str:
    """Return the ``entity_key`` of ``name`` without its spelling variants.

    Legal-form designators are stripped (``domain.languages.legal_forms``) and
    spaces next to Han, Hangul and Kana letters dropped. Names with the same
    compact key are the same entity written differently, and the fuzzy
    matcher links them with score 1.0.
    This is a matching key only: entity ids keep using ``entity_key``.
    """
    return drop_dense_script_spaces(strip_legal_forms(entity_key(name)))


def head_character(name: str | None) -> str:
    """Return the final character of a dense-script name, else ``""``.

    In Han, Hangul and Kana compounds the head noun comes last and is often
    one character: "가나다연구원" (institute) and "가나다연구소" (lab) share
    every character but the head and name different organizations, and so do
    the given names 김철수 and 김철민. Two names may only be fuzzy-linked when
    their head characters are equal. For a name ending in another script the
    result is ``""``, so Latin names are unaffected.
    """
    key = compact_name_key(name)
    return key[-1] if key and is_dense_script_char(key[-1]) else ""


def discriminator_tokens(name: str | None) -> frozenset[str]:
    """Return the tokens of ``name`` that identify *which* instance it denotes.

    Character-shingle similarity cannot tell "purchase order 1001" from
    "purchase order 1002" or "vendor a" from "vendor b": the shared prefix
    dominates the score. The differing part is a short identifier token, so
    two names may only be fuzzy-linked when their discriminator tokens are
    identical. After :func:`normalize_name` tokenization, a token counts as a
    discriminator when it

    * contains a digit of any script (``1001``, ``4``, ``2b``, ``v2``), or
    * is a single letter or digit of any script (``a``, ``b``, ``甲``/``乙``,
      ``갑``/``을``, ``α``/``β``, ``ä``/``ö``), or
    * is a multi-letter ASCII Roman numeral (``ii``, ``iv``, ``xii``).

    Tokens are already NFKC-normalized and casefolded by ``normalize_name``, so
    a precomposed or full-width designator is one character here.

    Requiring *equality* (not overlap) keeps the relation transitive, so a
    cluster built from guarded links can never contain two members whose
    discriminators differ. The rule is deliberately conservative: a variant
    that only one side spells with an initial ("J. Smith" vs "Smith") stays a
    separate entity rather than risking a wrong merge.

    Legal-form designators are removed first (:func:`strip_legal_forms`), so
    the "주" of "(주)가나다" does not count as an identifier.
    """
    tokens = normalize_name(strip_legal_forms(name) if name else name).split()
    return frozenset(
        token
        for token in tokens
        if any(c.isdigit() for c in token)
        or (len(token) == 1 and token.isalnum())
        or (len(token) > 1 and token.isascii() and _RE_ROMAN_NUMERAL.fullmatch(token))
    )


def normalize_entity_type(entity_type: str | None) -> str:
    """Normalize an entity type for compatibility checks ("" = unknown)."""
    normalized = normalize_name(entity_type)
    return "" if normalized in _GENERIC_ENTITY_TYPES else normalized


def entity_types_compatible(type_a: str | None, type_b: str | None) -> bool:
    """Whether two entity types allow a fuzzy merge.

    Types are compatible when they normalize to the same value or either side
    is empty/unknown (the extractor does not always emit a type).
    """
    a, b = normalize_entity_type(type_a), normalize_entity_type(type_b)
    return not a or not b or a == b
