# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text analyzers of the embedded search engine, by OpenSearch analyzer name.

A text field's mapping names its analyzer (``OpenSearchIndexer`` picks it from
``indexing.opensearch.language_analyzers`` / ``default_analyzer``); the engine
looks the name up here and analyzes both the indexed text and the query text
with it. Approximations of the OpenSearch analyzers of the same names:

- ``standard``: Unicode word split and lowercase. A run of Hangul, Han or Kana
  letters becomes overlapping character bigrams ("홍길동으로부터" -> "홍길",
  "길동", "동으", ...), and a single such letter a unigram, so a word matches
  inside a longer run without a morphological dictionary. (OpenSearch's
  ``standard`` keeps a Hangul run as one token; the bigrams are deliberate.)
- ``nori``, ``kuromoji`` and ``smartcn``: the same analyzer as ``standard``.
  The bigrams stand in for their morphological analysis: recall on a
  word-in-compound query is comparable, ranking is not identical.
- ``english``: possessive strip, lowercase, scikit-learn's English stop words
  and Porter stemming (``snowballstemmer``, from the ``[local]`` extra; without
  it the analyzer does not stem, with one warning). Dense-script runs become
  bigrams as in ``standard``.

An unknown name falls back to ``standard`` with one warning per name. Register
another with ``@register_analyzer("name")``: an analyzer is a callable from a
text to its tokens, in order.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from functools import cache

from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.scripts import is_dense_script_char

logger = get_logger(__name__)

Analyzer = Callable[[str], list[str]]

STANDARD = "standard"

_ANALYZERS: dict[str, Analyzer] = {}
_warned: set[str] = set()
_warned_lock = threading.Lock()

# A word: a run of letters, digits, marks or connector punctuation.
_WORD = re.compile(r"\w+")
_POSSESSIVE = re.compile(r"['’][sS]\b")


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


def _split_runs(word: str) -> list[tuple[str, bool]]:
    """``word`` cut into runs of dense-script letters and of other characters."""
    runs: list[tuple[str, bool]] = []
    start = 0
    for i in range(1, len(word) + 1):
        if i == len(word) or is_dense_script_char(word[i]) != is_dense_script_char(
            word[start]
        ):
            runs.append((word[start:i], is_dense_script_char(word[start])))
            start = i
    return runs


def _bigrams(run: str) -> list[str]:
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def _tokens(text: str, word_filter: Callable[[str], str | None]) -> list[str]:
    """Words of ``text`` lowercased (each kept by ``word_filter``, which may
    rewrite or drop it), dense-script runs as bigrams."""
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        for run, dense in _split_runs(word):
            if dense:
                tokens.extend(_bigrams(run))
            elif (kept := word_filter(run)) is not None:
                tokens.append(kept)
    return tokens


@register_analyzer(STANDARD, "nori", "kuromoji", "smartcn")
def standard_analyzer(text: str) -> list[str]:
    return _tokens(text, lambda word: word)


@cache
def _english_stop_words() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(ENGLISH_STOP_WORDS)


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

    def stem(word: str) -> str:
        with lock:
            result: str = stemmer.stemWord(word)
        return result

    return stem


@register_analyzer("english")
def english_analyzer(text: str) -> list[str]:
    stop_words = _english_stop_words()
    stem = _porter_stem()

    def keep(word: str) -> str | None:
        if word in stop_words:
            return None
        return stem(word) if stem is not None else word

    return _tokens(_POSSESSIVE.sub("", text), keep)
