# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one table of Unicode scripts the text heuristics read.

Each :class:`Script` names its letter blocks, how its text is spaced
(:class:`Spacing`), the marks it writes optionally (Arabic short vowels,
Hebrew points), and the characters a BPE token covers in it. Every helper
below, and every caller that asks "is this Hangul?", "does this text need
spaces?" or "how many tokens is this?", reads this table instead of a range
list of its own.

- ``dense``: Han, Hangul and Kana pack a morpheme into one or two
  characters and do not need spaces between words: one character carries
  what a whole Latin word carries. Edit-distance fuzziness, character
  shingles and word-token counts tuned for Latin words need to know when they
  face such text.
- ``spaceless``: Thai, Lao, Myanmar, Khmer and the Tai scripts are alphabets
  written without spaces between words (a space ends a phrase), so a run of
  their letters is many words.
- ``spaced``: every other script separates its words with spaces.

The tests go by Unicode block, not by language: which language a text is in
is the language registry's question (``domain.languages``).
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import cache

__all__ = [
    "APOSTROPHE_STAND_INS",
    "INVISIBLE_FORMAT_CHARS",
    "OTHER_LETTERS",
    "DENSE_SCRIPT_LETTERS",
    "DENSE_SCRIPT_TOKEN_RANGES",
    "LATIN_CHARS_PER_TOKEN",
    "QUOTE_MARKS",
    "SCRIPTS",
    "SPACELESS_SCRIPT_LETTERS",
    "Script",
    "ScriptCounts",
    "Spacing",
    "WORD_DASHES",
    "count_letters",
    "drop_optional_marks",
    "drop_dense_script_spaces",
    "fold_letter_variants",
    "has_dense_script",
    "has_unspaced_script",
    "is_dense_script_char",
    "is_hangul",
    "is_hangul_syllable",
    "is_unspaced_char",
    "join_dense_middle_dots",
    "letter_class",
    "script_named",
    "script_of",
    "without_latin_accents",
    "word_spans",
    "word_terms",
    "words",
]


class Spacing(str, Enum):
    """How a script separates its words."""

    SPACED = "spaced"
    DENSE = "dense"
    SPACELESS = "spaceless"


Ranges = tuple[tuple[int, int], ...]

# Characters per token of Latin text, and of every script not measured, in
# the local token estimate (``adapters.llm.token_estimate``); the chunk
# sizes in tokens bound their characters with it (``domain.ingestion.
# chunk_length``).
LATIN_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Script:
    """A writing system: its letter blocks and how its text behaves.

    ``chars_per_token`` is the local token estimate's rate for the script's
    characters (``adapters.llm.token_estimate``): 4 for Latin and every
    script not measured, 1 for Han, Hangul and Kana, and for the others
    the geometric mean of what the cl100k and o200k BPE vocabularies count
    over a sample sentence (``tests/fixtures/language_matrix.py``), rounded
    to a twelfth of a token per character. ``optional_marks`` are the
    combining marks the script writes or leaves out at will (vowel points),
    so text compared for identity may drop them. ``letter_variants`` are
    letters its writers use for one another (an Arabic alef with or
    without its hamza), each with the letter a search reads it as.
    ``middle_dot_stand_ins`` are letters its writers type for a middle dot,
    each with the letters between which it is one (Korean writes the
    archaic vowel "ㆍ" for "·" between two syllables: "한ㆍ미").
    """

    name: str
    ranges: Ranges
    spacing: Spacing = Spacing.SPACED
    chars_per_token: float = LATIN_CHARS_PER_TOKEN
    optional_marks: Ranges = field(default=())
    letter_variants: Mapping[str, str] = field(default_factory=dict)
    middle_dot_stand_ins: Mapping[str, Ranges] = field(default_factory=dict)


# The precomposed Hangul syllables (가-힣).
_HANGUL_SYLLABLES: Ranges = ((0xAC00, 0xD7A3),)
# Han, Hangul and Kana blocks.
_HANGUL: Ranges = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
)
_KANA: Ranges = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
)
_HAN: Ranges = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x323AF),  # CJK Unified Ideographs Extensions B-H
)

# The script table. Letter blocks only: punctuation and symbol blocks belong
# to no script (CJK punctuation and full-width forms are not letters; NFKC
# folds the latter to ASCII).
SCRIPTS: tuple[Script, ...] = (
    Script(
        "latin",
        (
            (0x0041, 0x005A),
            (0x0061, 0x007A),
            (0x00AA, 0x00AA),
            (0x00BA, 0x00BA),
            (0x00C0, 0x00D6),
            (0x00D8, 0x00F6),
            (0x00F8, 0x02AF),  # Latin-1 to IPA Extensions
            (0x1D00, 0x1DBF),  # Phonetic Extensions
            (0x1E00, 0x1EFF),  # Latin Extended Additional (Vietnamese)
            (0x2C60, 0x2C7F),
            (0xA720, 0xA7FF),
            (0xAB30, 0xAB6F),
            (0xFB00, 0xFB06),  # Latin ligatures
            (0xFF21, 0xFF3A),  # Full-width Latin
            (0xFF41, 0xFF5A),
        ),
    ),
    Script("greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF)), chars_per_token=1.5),
    Script(
        "cyrillic",
        ((0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
        chars_per_token=2.4,
        # Russian writes ё as е at will ("Фёдоров", "Федоров").
        letter_variants={"ё": "е", "Ё": "Е"},
    ),
    Script("armenian", ((0x0530, 0x058F), (0xFB13, 0xFB17))),
    Script(
        "hebrew",
        ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
        chars_per_token=1.5,
        # Cantillation and vowel points (niqqud).
        optional_marks=(
            (0x0591, 0x05BD),
            (0x05BF, 0x05BF),
            (0x05C1, 0x05C2),
            (0x05C4, 0x05C5),
            (0x05C7, 0x05C7),
        ),
    ),
    Script(
        "arabic",
        (
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x0870, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ),
        chars_per_token=2.0,
        # Short vowels (harakat), superscript alef, tatweel and Quranic marks.
        optional_marks=(
            (0x0610, 0x061A),
            (0x0640, 0x0640),
            (0x064B, 0x065F),
            (0x0670, 0x0670),
            (0x06D6, 0x06DC),
            (0x06DF, 0x06E4),
            (0x06E7, 0x06E8),
            (0x06EA, 0x06ED),
        ),
        # Alef with hamza or madda as bare alef, alef maksura as ya.
        letter_variants={"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"},
    ),
    Script("syriac", ((0x0700, 0x074F),)),
    Script("thaana", ((0x0780, 0x07BF),)),
    Script("devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF)), chars_per_token=1.5),
    Script("bengali", ((0x0980, 0x09FF),)),
    Script("gurmukhi", ((0x0A00, 0x0A7F),)),
    Script("gujarati", ((0x0A80, 0x0AFF),)),
    Script("oriya", ((0x0B00, 0x0B7F),)),
    Script("tamil", ((0x0B80, 0x0BFF),)),
    Script("telugu", ((0x0C00, 0x0C7F),)),
    Script("kannada", ((0x0C80, 0x0CFF),)),
    Script("malayalam", ((0x0D00, 0x0D7F),)),
    Script("sinhala", ((0x0D80, 0x0DFF),)),
    Script("thai", ((0x0E00, 0x0E7F),), Spacing.SPACELESS, 1.5),
    Script("lao", ((0x0E80, 0x0EFF),), Spacing.SPACELESS),
    Script("tibetan", ((0x0F00, 0x0FFF),)),
    Script(
        "myanmar",
        ((0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)),
        Spacing.SPACELESS,
    ),
    Script("georgian", ((0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F))),
    Script("ethiopic", ((0x1200, 0x139F), (0x2D80, 0x2DDF))),
    Script("khmer", ((0x1780, 0x17FF),), Spacing.SPACELESS),
    Script(
        "tai",
        (
            (0x1950, 0x19DF),  # Tai Le, New Tai Lue
            (0x1A20, 0x1AAF),  # Tai Tham
            (0xAA80, 0xAADF),  # Tai Viet
            (0x11700, 0x1174F),  # Ahom
        ),
        Spacing.SPACELESS,
    ),
    Script(
        "hangul",
        _HANGUL,
        Spacing.DENSE,
        1.0,
        # The compatibility araea "ㆍ" and the jamo NFKC reads it as.
        middle_dot_stand_ins={"\u318d": _HANGUL_SYLLABLES, "\u119e": _HANGUL_SYLLABLES},
    ),
    Script("kana", _KANA, Spacing.DENSE, 1.0),
    Script("han", _HAN, Spacing.DENSE, 1.0),
)

_BY_NAME = {script.name: script for script in SCRIPTS}
# The table as sorted, disjoint spans for script_of's binary search.
_SPANS = sorted((lo, hi, script) for script in SCRIPTS for lo, hi in script.ranges)
_STARTS = [lo for lo, _, _ in _SPANS]


def script_named(name: str) -> Script:
    """The script of the table named ``name`` (``"hangul"``); ``KeyError``
    for another name."""
    return _BY_NAME[name]


def script_of(ch: str) -> Script | None:
    """The script whose blocks hold ``ch``, ``None`` for a character of no
    script's letter block (digits, punctuation, symbols, spaces)."""
    cp = ord(ch)
    i = bisect_right(_STARTS, cp) - 1
    if i >= 0 and cp <= _SPANS[i][1]:
        return _SPANS[i][2]
    return None


# The name letters of no script's block are counted under.
OTHER_LETTERS = "other"


@dataclass(frozen=True)
class ScriptCounts:
    """How many letters of each script a text holds, by script name
    (:data:`OTHER_LETTERS` for letters of no listed script), and in how
    many runs (words, or stretches of a script written without spaces) by
    the group :meth:`dominant` weighs them in (``"dense"`` for Han, Hangul
    and Kana together), where counted."""

    by_script: Mapping[str, int] = field(default_factory=dict)
    runs: Mapping[str, int] | None = None

    def __getitem__(self, name: str) -> int:
        return self.by_script.get(name, 0)

    @property
    def letters(self) -> int:
        return sum(self.by_script.values())

    @property
    def dense(self) -> int:
        """The Han, Hangul and Kana letters."""
        return sum(
            count
            for name, count in self.by_script.items()
            if name in _BY_NAME and _BY_NAME[name].spacing is Spacing.DENSE
        )

    def dominant(self, without: Collection[str] = ()) -> str | None:
        """The script most of the text is in, weighed by what its letters
        carry (its letters over its ``chars_per_token``, so one Hangul
        syllable outweighs one Latin letter), the dense scripts together
        as ``"dense"``, leaving out the scripts (or ``"dense"``) named in
        ``without``; ``None`` without letters. On a tie, a script other
        than Latin wins: a name in Latin letters is what a question in
        another script quotes."""
        weights: dict[str, float] = {}
        for name, count in self.by_script.items():
            group = _group(name)
            script = _BY_NAME.get(name)
            rate = (
                script.chars_per_token if script is not None else LATIN_CHARS_PER_TOKEN
            )
            if group not in without:
                weights[group] = weights.get(group, 0.0) + count / rate
        if not weights:
            return None
        return max(weights, key=lambda group: (weights[group], group != "latin"))


def _group(name: str) -> str:
    """The group :meth:`ScriptCounts.dominant` weighs the letters of the
    script ``name`` in: ``"dense"`` for Han, Hangul and Kana, else
    ``name``."""
    script = _BY_NAME.get(name)
    return "dense" if script is not None and script.spacing is Spacing.DENSE else name


def count_letters(
    texts: Iterable[str],
    *,
    max_texts: int | None = None,
    max_chars: int | None = None,
) -> ScriptCounts:
    """Count the letters of ``texts`` (the first ``max_chars`` characters of
    each of the first ``max_texts``, when given) by script, and their runs
    (``ScriptCounts.runs``: a combining mark continues a run, any other
    character that is no letter ends it)."""
    counts: Counter[str] = Counter()
    runs: Counter[str] = Counter()
    for i, text in enumerate(texts):
        if max_texts is not None and i >= max_texts:
            break
        current: str | None = None
        for ch in text if max_chars is None else text[:max_chars]:
            if not ch.isalpha():
                if not unicodedata.category(ch).startswith("M"):
                    current = None
                continue
            script = script_of(ch)
            name = script.name if script is not None else OTHER_LETTERS
            counts[name] += 1
            if (group := _group(name)) != current:
                runs[group] += 1
                current = group
    return ScriptCounts(dict(counts), dict(runs))


def _class_body(ranges: Ranges) -> str:
    """``ranges`` as the body of a regex character class (``[...]``)."""
    return "".join(
        re.escape(chr(lo)) + ("" if lo == hi else "-" + re.escape(chr(hi)))
        for lo, hi in sorted(ranges)
    )


def _ranges_of(spacing: Spacing) -> Ranges:
    return tuple(
        sorted(span for s in SCRIPTS if s.spacing is spacing for span in s.ranges)
    )


@cache
def letter_class(*names: str) -> str:
    """The letter blocks of the scripts ``names`` as the body of a regex
    character class: ``f"[{letter_class('hangul')}]"``."""
    return _class_body(tuple(span for n in names for span in _BY_NAME[n].ranges))


# The Han, Hangul and Kana letters, and the spaceless scripts' letters, as
# the body of a regex character class.
DENSE_SCRIPT_LETTERS = _class_body(_ranges_of(Spacing.DENSE))
SPACELESS_SCRIPT_LETTERS = _class_body(_ranges_of(Spacing.SPACELESS))

# What the token estimate (``adapters.llm.token_estimate``) counts at about
# one token per character beside the dense letters: CJK radicals, CJK
# symbols and punctuation, and half/full-width forms.
DENSE_SCRIPT_TOKEN_RANGES: Ranges = tuple(
    sorted(
        (
            *_ranges_of(Spacing.DENSE),
            (0x2E80, 0x2FDF),  # CJK radicals / Kangxi
            (0x3000, 0x303F),  # CJK symbols and punctuation
            (0xFF00, 0xFFEF),  # Half/full-width forms
        )
    )
)


_OPTIONAL_MARK = re.compile(
    f"[{_class_body(tuple(span for s in SCRIPTS for span in s.optional_marks))}]"
)


def drop_optional_marks(text: str) -> str:
    """``text`` without the marks its scripts write at will (Arabic short
    vowels and tatweel, Hebrew points): "مُحَمَّد" is "محمد", "שָׁלוֹם"
    "שלום". Precomposed presentation forms are left as they are: normalize
    the text (NFC or NFKC) first."""
    return _OPTIONAL_MARK.sub("", text)


_LETTER_VARIANTS = str.maketrans(
    {k: v for script in SCRIPTS for k, v in script.letter_variants.items()}
)


def fold_letter_variants(text: str) -> str:
    """``text`` with each letter its script writes for another as that one
    (``Script.letter_variants``): "أحمد" is "احمد", "مستشفى" "مستشفي".
    A search fold only: an identity key keeps the letters as written."""
    return text.translate(_LETTER_VARIANTS)


# The Latin letters with a stroke, a ligature or a dotless form, which no
# Unicode decomposition takes their mark from, by the letters a text without
# accents writes for them ("Đặng" is "Dang", "Łódź" "Lodz", "Yıldız"
# "Yildiz", "Søren" "Soren", "Guðmundsson" "Gudmundsson").
_LATIN_BASE_LETTERS = str.maketrans(
    {
        "đ": "d",
        "Đ": "D",
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "ı": "i",
        "æ": "ae",
        "Æ": "AE",
        "œ": "oe",
        "Œ": "OE",
        "þ": "th",
        "Þ": "TH",
        "ð": "d",
        "Ð": "D",
        "ħ": "h",
        "Ħ": "H",
    }
)


def without_latin_accents(text: str) -> str:
    """``text`` with the accents of its Latin letters dropped ("café" is
    "cafe", "Nguyễn" "Nguyen") and its letters with a stroke, a ligature or
    a dotless form as the base letters (``_LATIN_BASE_LETTERS``: "Đặng" is
    "Dang", "Łódź" "Lodz"); other scripts' marks stay. A comparison fold
    only: names are stored as written."""
    letters: list[str] = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch) == "Mn" and letters:
            script = script_of(letters[-1])
            if script is not None and script.name == "latin":
                continue
        letters.append(ch)
    return unicodedata.normalize("NFC", "".join(letters)).translate(_LATIN_BASE_LETTERS)


# A middle dot (Latin, hyphenation point, katakana and its half-width form)
# between two Han, Hangul or Kana letters, where it separates the words of
# one name ("オリオン・ラボ").
# The middle dots (Latin, hyphenation point, katakana and its half-width
# form) and the letters a script types for one (``Script.middle_dot_stand_ins``).
_MIDDLE_DOTS = "\u00b7\u2027\u30fb\uff65"
_DOTS = re.escape(
    _MIDDLE_DOTS
    + "".join(dot for script in SCRIPTS for dot in script.middle_dot_stand_ins)
)


def _between(letters: str, dots: str) -> str:
    """One of ``dots`` between two of ``letters`` (each a character class
    body) that are no dots: the katakana middle dot is in the Kana block,
    and a dot beside another stays, so a text joined once is joined."""
    return f"(?<=[{letters}])(?<![{_DOTS}])[{dots}](?=[{letters}])(?![{_DOTS}])"


# A middle dot between two Han, Hangul or Kana letters, where it separates
# the words of one name ("オリオン・ラボ"), and a letter a script types for
# one between the letters it stands as one between.
_DENSE_MIDDLE_DOT = re.compile(
    "|".join(
        [
            _between(DENSE_SCRIPT_LETTERS, re.escape(_MIDDLE_DOTS)),
            *(
                _between(_class_body(ranges), re.escape(dot))
                for script in SCRIPTS
                for dot, ranges in script.middle_dot_stand_ins.items()
            ),
        ]
    )
)


def join_dense_middle_dots(text: str) -> str:
    """``text`` without the middle dots between Han, Hangul or Kana letters:
    "オリオン・ラボ" is "オリオンラボ", as those scripts write a name's words
    with a dot, a space or nothing between them, nor the letters a script
    types for one where they are one (``Script.middle_dot_stand_ins``:
    "한ㆍ미" is "한미"); a dot elsewhere stays."""
    return _DENSE_MIDDLE_DOT.sub("", text)


def _punctuation(test: Callable[[str], bool]) -> str:
    """The characters ``test`` takes of the blocks every Unicode quote mark
    and dash is in: up to U+3100 (Latin to CJK punctuation) and from U+FE00
    (vertical, small and full-width forms)."""
    return "".join(
        ch for ch in map(chr, (*range(0x3100), *range(0xFE00, 0x10000))) if test(ch)
    )


def _is_quote_mark(ch: str) -> bool:
    return (
        unicodedata.category(ch) in ("Pi", "Pf")
        or "QUOTATION MARK" in (name := unicodedata.name(ch, ""))
        or "CORNER BRACKET" in name
        or ch in "\u3008\u3009\u300a\u300b\u05f3\u05f4"
    )


# Every script's quote marks: the initial and final quotes (Unicode Pi, Pf:
# « », “ ”, ‘ ’, ‹ ›), the other characters Unicode names a quotation mark
# (the straight one, the low ones „ ‚ of German and Russian, 〝 〞 〟), the
# corner brackets 「 」『 』 Japanese and Chinese quote with, the title marks
# 《 》〈 〉 Chinese and Korean write a title in, and the Hebrew geresh and
# gershayim ׳ ״, written for ' and ".
QUOTE_MARKS = _punctuation(_is_quote_mark)
# The dashes that join words as "-" does: every Unicode dash (Pd: "‐", "–",
# "—", the Hebrew maqaf "־", the double hyphen "゠") and the minus sign "−",
# but the wave dashes "〜" and "〰", which write a range ("第1〜3条").
WORD_DASHES = (
    _punctuation(
        lambda ch: unicodedata.category(ch) == "Pd" and ch not in "\u301c\u3030"
    )
    + "\u2212"
)


# What a writer types for an apostrophe: the acute accent "´", the modifier
# letter apostrophe "ʼ" and the prime "′" ("McDonald´s"). Not the ʻokina
# "ʻ", a Hawaiian letter. NFKC reads "´" as a space and an accent, so a key
# folds these before it.
APOSTROPHE_STAND_INS = "\u00b4\u02bc\u2032"
# The format characters that show nothing and carry no letter: the soft
# hyphen, the zero-width space, the word joiner and the invisible operators,
# the byte-order mark and the direction marks, embeddings, overrides and
# isolates. Not the zero-width joiner and non-joiner, which shape the
# letters of Indic and Arabic scripts and of emoji.
INVISIBLE_FORMAT_CHARS = (
    "\u00ad\u061c\u200b\u200e\u200f"
    + "".join(map(chr, range(0x202A, 0x202F)))
    + "".join(map(chr, range(0x2060, 0x2065)))
    + "".join(map(chr, range(0x2066, 0x206A)))
    + "\ufeff"
)


# One compiled class: the per-character range scan these tests replaced held
# the GIL for about a third of a local search's CPU (MMR's redundancy terms
# test every word of every candidate), which concurrent queries in one
# process wait on.
_DENSE_SCRIPT_LETTER = re.compile(f"[{DENSE_SCRIPT_LETTERS}]")


def is_dense_script_char(ch: str) -> bool:
    """True if ``ch`` is a Han, Hangul or Kana letter."""
    ord(ch)  # one character, as before: TypeError otherwise
    return _DENSE_SCRIPT_LETTER.match(ch) is not None


def has_dense_script(text: str) -> bool:
    """True if ``text`` contains at least one Han, Hangul or Kana letter."""
    return _DENSE_SCRIPT_LETTER.search(text) is not None


def is_hangul(ch: str) -> bool:
    """True if ``ch`` is a Hangul letter: a syllable or a jamo, of any of
    the script table's Hangul blocks."""
    return (script := script_of(ch)) is not None and script.name == "hangul"


def is_hangul_syllable(ch: str) -> bool:
    """True if ``ch`` is a precomposed Hangul syllable (가-힣)."""
    return "가" <= ch <= "힣"


_WHITESPACE = re.compile(r"\s+")


_UNSPACED_LETTER = re.compile(f"[{DENSE_SCRIPT_LETTERS}{SPACELESS_SCRIPT_LETTERS}]")


def is_unspaced_char(ch: str) -> bool:
    """True if ``ch`` is a letter of a script that needs no spaces between
    words: a dense one (Han, Hangul, Kana) or a spaceless one (Thai, Lao,
    Myanmar, Khmer, Tai)."""
    ord(ch)  # one character: TypeError otherwise
    return _UNSPACED_LETTER.match(ch) is not None


def has_unspaced_script(text: str) -> bool:
    """True if ``text`` contains a letter of a dense or spaceless script."""
    return _UNSPACED_LETTER.search(text) is not None


def drop_dense_script_spaces(text: str, *, spaceless: bool = False) -> str:
    """Remove whitespace next to a Han, Hangul or Kana letter (and with
    ``spaceless`` next to a Thai, Lao, Myanmar, Khmer or Tai one).

    Word spacing is optional in these scripts ("가나다 상사" / "가나다상사",
    "3억 원" / "3억원"), so two spellings that differ only there compare
    equal afterwards. Other whitespace runs collapse to one space.
    """
    letter = is_unspaced_char if spaceless else is_dense_script_char

    def replace(match: re.Match[str]) -> str:
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        joined = (bool(before) and letter(before)) or (bool(after) and letter(after))
        return "" if joined else " "

    return _WHITESPACE.sub(replace, text)


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch)[0] == "M"


# A run of letters and numbers (Unicode L and N: "[^\W_]" is exactly them).
_LETTERS_AND_NUMBERS = re.compile(r"[^\W_]+")


def word_spans(text: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` of every word of ``text``: a run of letters,
    numbers and marks (Unicode L, N, M), in every script. "\\w" leaves out
    the vowel signs of Devanagari or Thai ("टाटा", "กรุงเทพ"), which split
    a word into fragments, and takes "_".

    The letters and numbers by one compiled pattern; only a non-ASCII gap
    between them is read a character at a time, for its marks (ASCII holds
    none): a walk over every character with ``unicodedata`` cost 5.6 times
    the old whitespace split on long passages (MMR compares every
    candidate's terms)."""
    if text.isascii():
        return [match.span() for match in _LETTERS_AND_NUMBERS.finditer(text)]
    spans: list[tuple[int, int]] = []

    def add(start: int, end: int) -> None:
        if spans and spans[-1][1] == start:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))

    def marks(low: int, high: int) -> None:
        if low >= high or text[low:high].isascii():
            return
        i = low
        while i < high:
            if not _is_mark(text[i]):
                i += 1
                continue
            j = i + 1
            while j < high and _is_mark(text[j]):
                j += 1
            add(i, j)
            i = j

    position = 0
    for match in _LETTERS_AND_NUMBERS.finditer(text):
        marks(position, match.start())
        add(*match.span())
        position = match.end()
    marks(position, len(text))
    return spans


def words(text: str) -> list[str]:
    """The words of ``text`` (:func:`word_spans`)."""
    if text.isascii():
        return _LETTERS_AND_NUMBERS.findall(text)
    return [text[start:end] for start, end in word_spans(text)]


# A run of letters written without spaces between words: Han, Hangul and
# Kana, and the spaceless alphabets (Thai, Lao, Myanmar, Khmer, Tai).
_UNSPACED_RUN = re.compile(f"[{DENSE_SCRIPT_LETTERS}{SPACELESS_SCRIPT_LETTERS}]+")


def word_terms(text: str) -> frozenset[str]:
    """The terms two texts are compared by: the :func:`words` of ``text``,
    each run of a script written without spaces (Han, Hangul, Kana, Thai,
    Lao, Myanmar, Khmer, Tai) split into its character bigrams (a run of
    one letter kept whole), as such a run is a clause, not a word."""
    found = words(text)
    if _UNSPACED_RUN.search(text) is None:
        return frozenset(found)
    terms: set[str] = set()
    for word in found:
        if _UNSPACED_RUN.search(word) is None:
            terms.add(word)
            continue
        for run in _UNSPACED_RUN.findall(word):
            terms.update(run[i : i + 2] for i in range(max(1, len(run) - 1)))
        terms.update(_UNSPACED_RUN.sub(" ", word).split())
    return frozenset(terms)
