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
from difflib import SequenceMatcher

from unified_kg_rag.domain.languages.legal_forms import strip_legal_forms
from unified_kg_rag.shared.utils import entity_key, normalize_name
from unified_kg_rag.shared.utils.scripts import (
    drop_dense_script_spaces,
    is_dense_script_char,
    join_dense_middle_dots,
)

__all__ = [
    "compact_name_key",
    "discriminator_tokens",
    "display_name_order",
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
    matcher links them with score 1.0: every comparison of "the same entity"
    (resolution, the incremental merge, claim resolution, the stores'
    lookups by name key) is this key's equality, and the fuzzy matcher's
    guards (:func:`discriminator_tokens`, :func:`head_character`) are
    functions of it, so they never keep two spellings of one key apart.
    This is a matching key only: entity ids keep using ``entity_key``.
    """
    # A dot the dropped spaces put between two letters is joined as
    # entity_key joins one, so a compact key is its own compact key.
    return join_dense_middle_dots(
        drop_dense_script_spaces(strip_legal_forms(entity_key(name)))
    )


def display_name_order(name: str) -> tuple[int, str]:
    """The order in which spellings of one entity tie for its display name:
    the fewest characters its ``entity_key`` folds away or reads otherwise
    first (case and spacing aside: "토지" before "「토지」", "Wi-Fi" before
    "Ｗｉ－Ｆｉ"), then the smallest name. A spelling its key writes as it is
    wins; the smallest name alone put a quote mark or a full-width letter
    first whenever it sorted first."""
    written = " ".join(name.casefold().split())
    kept = SequenceMatcher(None, written, entity_key(name), autojunk=False)
    return len(written) - sum(b.size for b in kept.get_matching_blocks()), name


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


def _script_runs(token: str) -> list[str]:
    """``token`` cut where it passes between a dense-script letter (Han,
    Hangul, Kana) and any other character: "제3조" is "제", "3", "조" and
    "lg전자" "lg", "전자"."""
    runs: list[str] = []
    for ch in token:
        if runs and is_dense_script_char(ch) == is_dense_script_char(runs[-1][-1]):
            runs[-1] += ch
        else:
            runs.append(ch)
    return runs


def _is_discriminator(token: str) -> bool:
    return (
        any(c.isnumeric() for c in token)
        or (len(token) == 1 and token.isalnum())
        or (
            len(token) > 1
            and token.isascii()
            and bool(_RE_ROMAN_NUMERAL.fullmatch(token))
        )
    )


def discriminator_tokens(name: str | None) -> frozenset[str]:
    """Return the tokens of ``name`` that identify *which* instance it denotes.

    Character-shingle similarity cannot tell "purchase order 1001" from
    "purchase order 1002" or "vendor a" from "vendor b": the shared prefix
    dominates the score. The differing part is a short identifier token, so
    two names may only be fuzzy-linked when their discriminator tokens are
    identical. The tokens are the words of the name's
    :func:`compact_name_key` (after :func:`normalize_name`), each cut where
    it passes between a dense-script letter and another character
    (:func:`_script_runs`): spellings of one entity have one compact key, so
    they have the same tokens however a dense-script name is spaced ("제3조"
    and "제 3 조" both {제, 3, 조}; legal-form designators are not in the key,
    so the "주" of "(주)가나다" does not count). A token counts as a
    discriminator when it

    * contains a numeral of any script (``1001``, ``4``, ``2b``, ``v2``, and
      the Han numerals: "東京第一工場" and "東京第二工場" differ in one), or
    * is a single letter or digit of any script (``a``, ``b``, ``甲``/``乙``,
      ``갑``/``을``, ``α``/``β``, ``ä``/``ö``; the "a" of "가나다아파트
      a동"), or
    * is a multi-letter ASCII Roman numeral (``ii``, ``iv``, ``xii``).

    Tokens are already NFKC-normalized and casefolded, so a precomposed or
    full-width designator is one character here.

    Requiring *equality* (not overlap) keeps the relation transitive, so a
    cluster built from guarded links can never contain two members whose
    discriminators differ. The rule is deliberately conservative: a variant
    that only one side spells with an initial ("J. Smith" vs "Smith") stays a
    separate entity rather than risking a wrong merge.
    """
    if not name:
        return frozenset()
    return frozenset(
        run
        for word in normalize_name(compact_name_key(name)).split()
        for run in _script_runs(word)
        if _is_discriminator(run)
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
