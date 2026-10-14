# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text analyzers of the embedded search engine, by OpenSearch analyzer name.

A text field's mapping names its analyzer (``OpenSearchIndexer`` picks it from
``indexing.opensearch.language_analyzers`` / ``default_analyzer``); the engine
looks the name up here and analyzes both the indexed text and the query text
with it. Approximations of the OpenSearch analyzers of the same names:

- ``standard``: Unicode word split (keeping "don't", "u.s.a" and "1,000.50"
  whole, as OpenSearch's tokenizer does) and lowercase. A run of Hangul, Han
  or Kana letters becomes overlapping character bigrams ("홍길동으로부터" ->
  "홍길", "길동", "동으", ...), and a single such letter a unigram, so a word
  matches inside a longer run without a morphological dictionary.
  (OpenSearch's ``standard`` keeps a Hangul run as one token; the bigrams are
  deliberate.)
- ``nori``, ``kuromoji`` and ``smartcn``: the same analyzer as ``standard``.
  The bigrams stand in for their morphological analysis: recall on a
  word-in-compound query is comparable, ranking is not identical.
- ``english``: OpenSearch's chain — possessive strip, lowercase, Lucene's 33
  English stop words (``_english_``) and Porter stemming (``snowballstemmer``,
  from the ``[local]`` extra; without it the analyzer does not stem, with one
  warning). Dense-script runs become bigrams as in ``standard``.

An unknown name falls back to ``standard`` with one warning per name. Register
another with ``@register_analyzer("name")``: an analyzer is a callable from a
text to its tokens, in order.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from functools import cache, lru_cache

from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.scripts import DENSE_SCRIPT_LETTERS

logger = get_logger(__name__)

Analyzer = Callable[[str], list[str]]

STANDARD = "standard"

_ANALYZERS: dict[str, Analyzer] = {}
_warned: set[str] = set()
_warned_lock = threading.Lock()

# A word: a run of letters, digits, marks or connector punctuation, joined
# across the punctuation Unicode word boundaries (UAX #29, which OpenSearch's
# standard tokenizer follows) keep inside a word: ":", ".", "'" between two
# letters ("u.s.a", "don't"), ",", ".", ";", "'" between two digits ("1,000.50").
_LETTER = f"[^\\W\\d_{DENSE_SCRIPT_LETTERS}]"
_WORD = re.compile(
    rf"\w+(?:(?:(?<={_LETTER})[:.'·‘’＇．](?={_LETTER})"
    rf"|(?<=\d)[,.;'‘’＇．，；](?=\d))\w+)*"
)
# Splits a word into alternating other / dense-script runs (dense at odd indices).
_DENSE_RUN = re.compile(f"([{DENSE_SCRIPT_LETTERS}]+)")
_POSSESSIVE = re.compile(r"['’＇][sS]\b")


def register_analyzer(*names: str) -> Callable[[Analyzer], Analyzer]:
    """Register ``analyzer`` under ``names`` (a later registration replaces)."""

    def decorator(analyzer: Analyzer) -> Analyzer:
        for name in names:
            _ANALYZERS[name] = analyzer
        return analyzer

    return decorator


def _warn_once(key: str, message: str, *args: object) -> None:
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message, *args)


def get_analyzer(name: str | None) -> Analyzer:
    """The analyzer registered as ``name`` (``standard`` when none is)."""
    analyzer = _ANALYZERS.get(name or STANDARD)
    if analyzer is not None:
        return analyzer
    _warn_once(
        f"analyzer:{name}",
        "The embedded search engine has no '%s' analyzer; analyzing that "
        "field with 'standard' instead",
        name,
    )
    return _ANALYZERS[STANDARD]


def _bigrams(run: str) -> list[str]:
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def _tokens(text: str, word_filter: Callable[[str], str | None]) -> list[str]:
    """Words of ``text`` lowercased (each kept by ``word_filter``, which may
    rewrite or drop it), dense-script runs as bigrams."""
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        for i, run in enumerate(_DENSE_RUN.split(word)):
            if i % 2:
                tokens.extend(_bigrams(run))
            elif run and (kept := word_filter(run)) is not None:
                tokens.append(kept)
    return tokens


@register_analyzer(STANDARD, "nori", "kuromoji", "smartcn")
def standard_analyzer(text: str) -> list[str]:
    return _tokens(text, lambda word: word)


# Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET, the ``_english_`` stop words
# of OpenSearch's ``english`` analyzer.
ENGLISH_STOP_WORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such "
    "that the their then there these they this to was will with".split()
)


@cache
def _porter_stem() -> Callable[[str], str] | None:
    try:
        import snowballstemmer
    except ImportError:
        _warn_once(
            "stemmer",
            "snowballstemmer is not installed, so the embedded search engine's "
            "'english' analyzer does not stem (install the [local] extra)",
        )
        return None
    stemmer = snowballstemmer.stemmer("porter")
    lock = threading.Lock()  # a stemmer object is not thread-safe

    # A corpus repeats its words: stem each distinct one once.
    @lru_cache(maxsize=1 << 16)
    def stem(word: str) -> str:
        with lock:
            result: str = stemmer.stemWord(word)
        return result

    return stem


@register_analyzer("english")
def english_analyzer(text: str) -> list[str]:
    stem = _porter_stem()

    def keep(word: str) -> str | None:
        if word in ENGLISH_STOP_WORDS:
            return None
        return stem(word) if stem is not None else word

    return _tokens(_POSSESSIVE.sub("", text), keep)
