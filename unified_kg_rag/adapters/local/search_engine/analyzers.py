# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text analyzers of the embedded search engine, by OpenSearch analyzer name.

A text field's mapping names its analyzer (``OpenSearchIndexer`` picks it from
``indexing.opensearch.language_analyzers`` / ``default_analyzer``); the engine
looks the name up here and analyzes both the indexed text and the query text
with it. Approximations of the OpenSearch analyzers of the same names:

- ``standard``: Unicode word split as OpenSearch's tokenizer does it (UAX
  #29): a word keeps its combining marks ("हिन्दी", a decomposed "café"),
  in-word punctuation joins ("don't", "u.s.a", "1,000.50", "a：b"), and a run
  of Thai, Lao, Myanmar or Khmer stays whole (neither engine segments it).
  Then lowercase. Unlike OpenSearch it drops emoji (OpenSearch keeps one as a
  token), as both drop other symbols ("²", "½"). A run of Hangul, Han
  or Kana letters becomes overlapping character bigrams ("홍길동으로부터" ->
  "홍길", "길동", "동으", ...), and a single such letter a unigram, so a word
  matches inside a longer run without a morphological dictionary.
  (OpenSearch's ``standard`` keeps a Hangul run as one token; the bigrams are
  deliberate.)
- ``nori``, ``kuromoji`` and ``smartcn``: the same analyzer as ``standard``.
  The bigrams stand in for their morphological analysis: recall on a
  word-in-compound query is comparable, ranking is not identical.
- ``english``: OpenSearch's chain — possessive strip ("'s" off a token's
  end), lowercase, Lucene's 33 English stop words (``_english_``) and Porter
  stemming as Lucene's ``PorterStemmer`` (``snowballstemmer``'s "porter" with
  Lucene's departures from it, from the ``[local]`` extra; without it the
  analyzer does not stem, with one warning). Words of one or two letters are
  not stemmed. Dense-script runs become bigrams as in ``standard``.

An unknown name falls back to ``standard`` with one warning per name. Register
another with ``@register_analyzer("name")``: an analyzer is a callable from a
text to its tokens, in order.
"""

from __future__ import annotations

import re
import threading
import unicodedata
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

# Code point ranges by their part in a word under Unicode word boundaries
# (UAX #29, which OpenSearch's standard tokenizer follows), read from
# ``unicodedata`` on first use (about 0.1 s) so they follow the running
# Python's Unicode version.
#
# Scripts written without spaces between words (line break class SA): the
# tokenizer keeps a run of their letters and marks whole, apart from adjacent
# letters of other scripts, without a dictionary.
_SPACELESS_BLOCKS = (
    (0x0E00, 0x0EFF),  # Thai, Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x1950, 0x19DF),  # Tai Le, New Tai Lue
    (0x1A20, 0x1AAF),  # Tai Tham
    (0xA9E0, 0xA9FF),  # Myanmar Extended-B
    (0xAA60, 0xAADF),  # Myanmar Extended-A, Tai Viet
    (0x11700, 0x1174F),  # Ahom
)
# Symbols with the Alphabetic property (circled and squared Latin letters).
_ALPHABETIC_SYMBOLS = (
    (0x24B6, 0x24E9),
    (0x1F130, 0x1F149),
    (0x1F150, 0x1F169),
    (0x1F170, 0x1F189),
)
# Hebrew letters, which a ``"`` joins and a ``'`` may follow ("צה"ל", "ג'").
_HEBREW = (0x05D0, 0x05F2), (0xFB1D, 0xFB4F)
# Every assigned word character lies below U+40000 or in plane 14 (tags and
# variation selectors).
_SCANNED = ((0, 0x3FFFF), (0xE0000, 0xE0FFF))
# Punctuation that joins two letters ("u.s.a", "don't", "a:b") or two digits
# ("1,000.50"); either side may carry marks.
_MID_LETTER = ":\u00b7\u0387\u05f4\u2027\ufe13\ufe55\uff1a"
_MID_NUM = (
    ",;\u037e\u0589\u060c\u060d\u066c\u07f8\u2044\ufe10\ufe14\ufe50\ufe54\uff0c\uff1b"
)
_MID_NUM_LETTER = ".'\u2018\u2019\u2024\ufe52\uff07\uff0e"


def _class(ranges: list[tuple[int, int]]) -> str:
    """``ranges`` as the body of a regex character class."""
    return "".join(
        re.escape(chr(lo)) + ("" if lo == hi else "-" + re.escape(chr(hi)))
        for lo, hi in ranges
    )


@cache
def _word_ranges() -> dict[str, list[tuple[int, int]]]:
    """The code point ranges of each part a character plays in a word."""
    spaceless = {cp for lo, hi in _SPACELESS_BLOCKS for cp in range(lo, hi + 1)}
    ranges: dict[str, list[tuple[int, int]]] = {
        role: [] for role in ("spaceless", "letter", "digit", "extend", "connector")
    }
    category = unicodedata.category
    for lo, hi in _SCANNED:
        for cp in range(lo, hi + 1):
            kind = category(chr(cp))
            if kind[0] in "LM" and cp in spaceless:
                role = "spaceless"
            elif kind[0] == "L" or kind == "Nl":
                role = "letter"
            elif kind == "Nd":
                role = "digit"
            elif kind[0] == "M" or (kind == "Cf" and cp != 0x200B):
                role = "extend"  # a zero width space separates words
            elif kind == "Pc" or cp == 0x202F:
                role = "connector"
            else:
                continue
            spans = ranges[role]
            if spans and spans[-1][1] == cp - 1:
                spans[-1] = (spans[-1][0], cp)
            else:
                spans.append((cp, cp))
    ranges["letter"] = sorted(ranges["letter"] + list(_ALPHABETIC_SYMBOLS))
    return ranges


@cache
def _word_pattern(astral: bool) -> re.Pattern[str]:
    """A word: a whole run of a spaceless script, or a run of letters, digits,
    marks, format characters and connectors (a connector-only run is no word)
    joined across in-word punctuation. No other symbol (emoji, "²", "½") is
    part of a word.

    Without ``astral``, for text of the Basic Multilingual Plane only: the
    same matches, about four times faster (``re`` tries a character class's
    ranges above U+FFFF one by one).
    """
    limit = 0x10FFFF if astral else 0xFFFF

    def part(role: str) -> str:
        return _class(
            [(lo, min(hi, limit)) for lo, hi in _word_ranges()[role] if lo <= limit]
        )

    spaceless, letter, digit, extend, connector = (
        part(role) for role in ("spaceless", "letter", "digit", "extend", "connector")
    )
    hebrew = _class(list(_HEBREW))
    character = f"[{letter}{digit}{extend}{connector}]"
    # A letter on both sides (a mark or format character counts as part of
    # the letter before it); Han, Hangul and Kana letters do not count.
    before_letter = f"(?<=[{letter}{extend}])(?<![{DENSE_SCRIPT_LETTERS}])"
    after_letter = f"(?=[{letter}{extend}])(?![{DENSE_SCRIPT_LETTERS}])"
    joint = (
        f"{before_letter}[{_MID_LETTER}{_MID_NUM_LETTER}]{after_letter}"
        f"|(?<=\\d)[{_MID_NUM}{_MID_NUM_LETTER}](?=\\d)"
        f'|(?<=[{hebrew}])"(?=[{hebrew}])'
    )
    return re.compile(
        f"[{spaceless}][{spaceless}{extend}]*"
        f"|[{connector}]*[{letter}{digit}]{character}*"
        f"(?:(?:{joint}){character}+|(?<=[{hebrew}])'{character}*)*"
    )


# Splits a word into alternating other / dense-script runs (dense at odd indices).
_DENSE_RUN = re.compile(f"([{DENSE_SCRIPT_LETTERS}]+)")


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
    text = text.lower()
    for word in _word_pattern(max(text, default="") > "\uffff").findall(text):
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
            # Lucene's PorterStemmer is Porter's own revision of the published
            # algorithm that snowball's "porter" implements. Where step 1 left
            # "-bli" or "-logi" (and steps 3 to 5 kept it), its step 2 also
            # turns "-bli" into "-ble" (then through steps 3 to 5) and "-logi"
            # into "-log"; its step 1b undoubles any final consonant but l, s
            # and z after dropping "-ed" or "-ing" (snowball: bb dd ff gg mm
            # nn pp rr tt only; steps 2 to 5 keep the doubled ones).
            base = result[:-1]
            if word.startswith(base) and word[len(base) :] in _STEP1_I_ENDINGS:
                if result.endswith("bli") and _has_measure(result[:-3]):
                    result = stemmer.stemWord(base + "e")
                elif result.endswith("logi") and _has_measure(result[:-4]):
                    result = base
            elif (
                word.startswith(result)
                and word[len(result) :] in ("ed", "eds", "ing", "ings")
                and result[-2:] in _UNDOUBLED
            ):
                result = stemmer.stemWord(base)
        return result

    return stem


# What a word ends in where Porter's step 1 leaves an "i" in its place
# ("possibly", "biologies", "bullied", ...).
_STEP1_I_ENDINGS = frozenset({"i", "y", "is", "ys", "ies", "ied", "yed", "ying"})
_UNDOUBLED = frozenset(c * 2 for c in "cghjkqvwx")


def _has_measure(stem: str) -> bool:
    """Porter's m(stem) > 0: a vowel followed by a consonant, where "y" is a
    vowel after a consonant."""
    seen_vowel = previous_consonant = False
    for ch in stem:
        consonant = ch not in "aeiou" and (ch != "y" or not previous_consonant)
        if consonant and seen_vowel:
            return True
        seen_vowel = seen_vowel or not consonant
        previous_consonant = consonant
    return False


# The possessive endings Lucene's EnglishPossessiveFilter strips from a token.
_POSSESSIVES = ("'s", "\u2019s", "\uff07s")


@register_analyzer("english")
def english_analyzer(text: str) -> list[str]:
    stem = _porter_stem()

    def keep(word: str) -> str | None:
        if word.endswith(_POSSESSIVES):
            word = word[:-2]
        if word in ENGLISH_STOP_WORDS:
            return None
        # Like Lucene's PorterStemmer, leave words of one or two letters.
        return stem(word) if stem is not None and len(word) > 2 else word

    return _tokens(text, keep)
