# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linking a name a query gives to the stored entities a search found.

One rule for every strategy that turns a name into graph entities (local
search's focus seeds, the structured strategy's mention nodes):
:func:`choose_links` keeps the hits that *are* the name (the same
``compact_name_key``), else follows its policy: ``cover`` (structured)
keeps the hits whose name covers every term of it; ``rank`` (local, whose
seeding is measured) keeps the top hit above a score floor unless it is a
one-word lookalike.

A search returns its nearest entities for any text (a kNN search always
does, and BM25 matches one shared word), so a score alone cannot tell
"Acme Supplies" from the absent "Zorblax Supplies", nor 가나다 상사 from the
absent 라마바 상사. Coverage can: each term of the name must be a term of
the entity's name, or one spelled alike (``similarity``: the ``difflib``
ratio entity resolution's ``processing.similarity_threshold`` sets, so
"Bexly" finds "Bexley" while "Supplies" never stands for "Zorblax
Supplies"). Terms are the words of the name's ``compact_name_key``
(NFKC-casefolded, legal-form designators such as "Inc" or "株式会社"
left out); a word in Han, Hangul or Kana is read as its character bigrams,
since those scripts write compounds without spaces ("东芝" is covered by
"东芝电子", "라마바상사" is not by "가나다상사").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Literal

from unified_kg_rag.domain.ingestion.base_resolver import compact_name_key
from unified_kg_rag.domain.models import RetrievalResult
from unified_kg_rag.shared.utils.legal_forms import latin_legal_form
from unified_kg_rag.shared.utils.scripts import DENSE_SCRIPT_LETTERS

__all__ = ["LinkPolicy", "choose_links", "covers", "hit_name", "name_terms"]

_WORD = re.compile(r"\w+")
_DENSE_RUN = re.compile(f"[{DENSE_SCRIPT_LETTERS}]+")


def hit_name(hit: RetrievalResult) -> str:
    """The stored name of an entity hit (``""`` without one)."""
    return str((hit.metadata or {}).get("name") or "")


def name_terms(name: str) -> frozenset[str]:
    """The terms of ``name`` coverage compares (module docstring)."""
    terms: set[str] = set()
    for word in _WORD.findall(compact_name_key(name)):
        if latin_legal_form(word) is not None:
            continue
        for run in _DENSE_RUN.findall(word):
            terms.update(run[i : i + 2] for i in range(max(1, len(run) - 1)))
        terms.update(_DENSE_RUN.sub(" ", word).split())
    return frozenset(terms)


def covers(name: str, mention: str, similarity: float = 1.0) -> bool:
    """Whether the stored ``name`` covers every term of ``mention``: each
    is a term of ``name`` or, below a ``similarity`` of 1, as alike as that
    to one of them (``difflib`` ratio)."""
    wanted = name_terms(mention)
    if not wanted:
        return False
    held = name_terms(name)
    return all(
        term in held
        or (
            similarity < 1.0
            and any(SequenceMatcher(None, term, t).ratio() >= similarity for t in held)
        )
        for term in wanted
    )


LinkPolicy = Literal["cover", "rank"]


def _lookalike(name: str, mention: str) -> bool:
    """Whether ``name`` shares a term with ``mention`` but neither covers
    the other: a lexical hit on one shared word ("Acme Supplies" for
    "Zorblax Supplies"), not a semantic or alias hit (no shared term:
    "New York City" for "NYC") nor a part of the name ("Beatles")."""
    held, wanted = name_terms(name), name_terms(mention)
    if not held & wanted:
        return False
    return not (wanted <= held or held <= wanted)


def choose_links(
    mention: str,
    hits: Sequence[RetrievalResult],
    *,
    policy: LinkPolicy,
    limit: int = 1,
    floor: float = 0.0,
    similarity: float = 1.0,
) -> list[RetrievalResult]:
    """The entities ``mention`` links to among a search's ``hits`` (best
    first): every hit with its ``compact_name_key``; else, by ``policy``:

    - ``cover`` (the structured strategy's mentions, where an absent name
      must link nothing): at most ``limit`` (0: none) hits that score above
      0 and at least ``floor`` and whose name covers the mention
      (:func:`covers` at ``similarity``);
    - ``rank`` (local search's focus seeds, measured as is): the top hit
      when it scores at least ``floor`` (of the best ties, one whose name
      covers the mention), unless it is a one-word lookalike
      (:func:`_lookalike`); a semantic or alias hit still links.

    Hits without a source never link."""
    key = compact_name_key(mention)
    named = [h for h in hits if h.source]
    exact = [h for h in named if key and compact_name_key(hit_name(h)) == key]
    if exact:
        return exact
    if policy == "rank":
        if not named or float(named[0].score or 0.0) < floor:
            return []
        top = float(named[0].score or 0.0)
        tied = [h for h in named if float(h.score or 0.0) == top]
        best = next(
            (h for h in tied if covers(hit_name(h), mention, similarity)), named[0]
        )
        return [] if _lookalike(hit_name(best), mention) else [best]
    chosen: list[RetrievalResult] = []
    for hit in named:
        if len(chosen) >= limit:
            break
        score = float(hit.score or 0.0)
        if (
            score > 0.0
            and score >= floor
            and covers(hit_name(hit), mention, similarity)
        ):
            chosen.append(hit)
    return chosen
