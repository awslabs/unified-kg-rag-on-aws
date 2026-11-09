# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic text normalization shared by the LLM-free evaluators."""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Callable, Iterable, Iterator

from unified_kg_rag.domain.languages import (
    LanguageCode,
    LanguageProfile,
    profile,
    profiles,
)
from unified_kg_rag.shared.utils.scripts import (
    DENSE_SCRIPT_LETTERS,
    drop_dense_script_spaces,
    has_dense_script,
    is_dense_script_char,
    is_hangul,
    is_hangul_syllable,
    letter_class,
)
from unified_kg_rag.shared.utils.sentences import sentence_ends

# A Hangul letter (the script table's blocks): a Korean word goes on.
_HANGUL = f"[{letter_class('hangul')}]"


def _words(words: frozenset[str] | set[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(sorted(words, key=len, reverse=True)) + r")\b")


# The articles of each language whose answers carry them
# (``LanguageProfile.articles``), dropped with the English ones from an
# answer in that language (as MLQA's evaluation script does for de and es).
_ARTICLES = _words(profile(LanguageCode.EN).articles)
_LANGUAGE_ARTICLES: dict[str, re.Pattern[str]] = {
    language.code.value: _words(language.articles)
    for language in profiles()
    if language.articles and language.code is not LanguageCode.EN
}
# An elided word before a vowel ("l'accord", "d'Avignon", "dell'accordo";
# ``LanguageProfile.elided_prefixes``): split from its word before
# punctuation is deleted, instead of gluing to it ("laccord", "davignon");
# an elided article is then dropped as the language's articles are.
_ELIDED_PREFIXES: dict[str, re.Pattern[str]] = {
    language.code.value: re.compile(
        r"(?<!\w)("
        + "|".join(sorted(language.elided_prefixes, key=len, reverse=True))
        + r")['\u2019](?=\w)"
    )
    for language in profiles()
    if language.elided_prefixes
}
_ASCII_PUNCTUATION = frozenset(string.punctuation)
# A comma followed by exactly three digits inside a number ("1,000,000").
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_DASHES = "\u002d\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
# A dash between two word characters separates them: a range ("1990-1995"
# is two numbers) or a compound ("5-year" equals "5 year").
_JOINING_DASH = re.compile(rf"(?<=\w)[{_DASHES}](?=\w)")
# A minus sign starting a number ("-5", "(−5)"); folded to ASCII "-".
_LEADING_MINUS = re.compile(rf"(?<!\w)[{_DASHES}](?=\d)")
# Punctuation that carries meaning and is kept: a leading minus ("-5" is not
# "5"), a decimal point or comma between digits ("1.5" is not "15"), a percent
# sign after a number ("3.5%" is not "3.5") and "+"/"#" ending a word ("C++",
# "C#").
_KEPT_PUNCTUATION = re.compile(
    r"(?<!\w)-(?=\d)|(?<=\d)[.,](?=\d)|(?<=\d)%|(?<=\w)[+#]+"
)
# "3.5 %" is written "3.5%".
_SPACED_PERCENT = re.compile(r"(?<=\d)\s+%")
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*%?")


def _is_punctuation(ch: str) -> bool:
    # ASCII string.punctuation (which includes symbols such as "$" and "%")
    # plus Unicode punctuation (category P*), which covers CJK and fullwidth
    # marks such as 「」 and 。.
    return ch in _ASCII_PUNCTUATION or unicodedata.category(ch).startswith("P")


_WORD_CHAR = re.compile(r"\w")


def _is_lone_letter(text: str, index: int, step: int) -> bool:
    """Whether ``text[index]`` is a letter forming a one-character word.

    ``step`` is the direction away from the dash (-1 left, +1 right); the
    word is one character long when the next character in that direction is
    not a word character (or the text ends).
    """
    if not 0 <= index < len(text) or not text[index].isalpha():
        return False
    beyond = index + step
    return not (0 <= beyond < len(text) and _WORD_CHAR.match(text[beyond]))


def _split_or_join_dash(match: re.Match[str]) -> str:
    """A dash between words is a space, except next to a lone letter.

    "1990-1995" and "5-year" split into separate tokens, but "Type-A" and
    "A-1" stay one token: split, the lone "a" would be dropped as an article
    and "Type-A" would match "Type-B". Only the characters next to the dash
    are inspected, so the pass stays linear in the text length.
    """
    text = match.string
    if _is_lone_letter(text, match.start() - 1, -1) or _is_lone_letter(
        text, match.end(), 1
    ):
        return ""
    return " "


def normalize_answer(text: str, language: str | None = None) -> str:
    """SQuAD-style answer normalization, preceded by Unicode NFKC.

    NFKC folds compatibility forms (fullwidth digits/letters, ligatures, the
    "…" ellipsis) to their canonical characters; then: lowercase, delete
    punctuation, drop the English articles ``a``/``an``/``the``, and collapse
    whitespace. Unlike the official SQuAD v1.1 script, Unicode punctuation is
    deleted too, and punctuation that changes meaning is kept: a leading
    minus ("-5" stays distinct from "5"), a decimal point or comma between
    digits ("1.5" from "15"), a percent sign after a number ("3.5%" from
    "3.5") and a trailing "+"/"#" ("C++", "C#"). A dash between two words or
    numbers becomes a space, so "1990-1995" is two numbers and "5-year"
    equals "5 year"; next to a lone letter of either case it is deleted
    instead, so "Type-A", "type-a" and "TypeA" are one token that does not
    match "Type-B". A space is never deleted: a lone letter after one ("a
    vendor", "Vitamin A", "Yes I agree") is read as an article or a word,
    so "Type A" loses its "A" like any article. Thousands separators are
    deleted, so "1,000" equals "1000". Scores are therefore not comparable with leaderboard numbers
    computed by the official script.

    A text consisting only of articles ("A", "The") keeps them rather than
    normalizing to an empty string.

    ``language`` (the answer's ``LanguageCode`` value) also drops that
    language's articles for German, Spanish, French, Italian and Portuguese
    ("der Rhein" equals "Rhein", "l'accord" "accord"), after splitting a
    French or Italian elided word from the next ("d'Avignon" is "d
    avignon"); other languages, and ``None``, drop the English ones only.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    if (elided := _ELIDED_PREFIXES.get(language or "")) is not None:
        text = elided.sub(r"\1 ", text)
    text = _THOUSANDS_SEPARATOR.sub("", text)
    text = _JOINING_DASH.sub(_split_or_join_dash, text)
    text = _LEADING_MINUS.sub("-", text)
    text = _SPACED_PERCENT.sub("%", text)
    kept = {i for m in _KEPT_PUNCTUATION.finditer(text) for i in range(*m.span())}
    text = "".join(
        ch for i, ch in enumerate(text) if i in kept or not _is_punctuation(ch)
    )
    without_articles = _ARTICLES.sub(" ", text)
    articles = _LANGUAGE_ARTICLES.get(language or "")
    if articles is not None:
        without_articles = articles.sub(" ", without_articles)
    return " ".join(without_articles.split()) or " ".join(text.split())


_SQUAD_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SQUAD_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer_squad(text: str) -> str:
    """The official SQuAD v1.1 ``normalize_answer`` (also MuSiQue's and
    HotpotQA's): lowercase, delete ASCII punctuation, drop ``a``/``an``/``the``
    and collapse whitespace. No Unicode folding."""
    text = "".join(ch for ch in text.lower() if ch not in _SQUAD_PUNCTUATION)
    return " ".join(_SQUAD_ARTICLES.sub(" ", text).split())


# An English possessive ending a word ("Depot's", "Depot’s"). Deleting
# punctuation would leave "depots", a different token from "depot".
_POSSESSIVE = re.compile(r"(?<=\w)['\u2019]s\b", re.IGNORECASE)


def strip_possessives(text: str) -> str:
    """``text`` with the possessive "'s" / "’s" ending each word removed
    (after NFKC, which folds the fullwidth apostrophe), so "Depot's" reads
    "Depot"; a plural ("Depots") is kept."""
    return _POSSESSIVE.sub("", unicodedata.normalize("NFKC", text))


def tokenize(text: str, language: str | None = None) -> list[str]:
    """Whitespace tokens of the normalized text (``normalize_answer``)."""
    return normalize_answer(text, language).split()


def is_spaceless_script(text: str) -> bool:
    """True if the text has CJK characters and no internal whitespace.

    Such a phrase is not whitespace-tokenizable (Chinese/Japanese), or is a
    single Korean word that takes attached particles, so contiguous word-token
    matching would miss it; it is matched as a substring instead.
    """
    if any(ch.isspace() for ch in text.strip()):
        return False
    return has_dense_script(text)


def _continues_word(a: str, b: str) -> bool:
    """True if adjacent characters ``a`` and ``b`` belong to one word or number.

    Digits (of any script, including Han numerals) and non-CJK letters run
    together: "2년" inside "12년" or "二年" inside "十二年" is another number.
    """

    def wordlike(ch: str) -> bool:
        # A "." or "," survives normalization only between digits ("1.5"),
        # a "-" only as the sign of a number ("-2년" is not "2년").
        return (
            ch.isnumeric()
            or ch in ".,-"
            or (ch.isalpha() and not is_dense_script_char(ch))
        )

    return wordlike(a) and wordlike(b)


def _substring_at_boundary(needle: str, haystack: str) -> bool:
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle)
        if not (
            (start and _continues_word(haystack[start - 1], needle[0]))
            or (end < len(haystack) and _continues_word(needle[-1], haystack[end]))
        ):
            return True
        start = haystack.find(needle, start + 1)
    return False


# Korean particles/copula endings attached to a word ("서울은", "서울에서",
# "서울입니다") are at most this many syllables in the common cases.
_MAX_KOREAN_SUFFIX = 3
# Korean magnitude numerals: after a number they scale its value ("3억" is
# 300 million), so a bare gold number never matches a token they extend.
_KOREAN_MAGNITUDES = frozenset("십백천만억조경")
# Korean counters and units that keep a number's value ("2024년", "3명",
# "5,000원"), longest first; a particle may follow them ("2024년에").
_KOREAN_UNITS = (
    "개월",
    "개사",
    "군데",
    "가지",
    "곳",
    "년",
    "월",
    "일",
    "주",
    "시",
    "분",
    "초",
    "명",
    "개",
    "회",
    "건",
    "번",
    "배",
    "원",
)


# Korean particles (josa) and the copula's common endings, the closed word
# class the standard grammar lists (National Institute of Korean Language,
# Standard Korean Language Dictionary, part of speech 조사; the copula
# 이다): case particles, auxiliary particles, the conjunctive ones, and the
# copula's declarative, polite, connective and adnominal endings. Token F1
# drops only a run of these after a reference word (``_KOREAN_JOSA_RUN``),
# so "서울은" counts as "서울" but "서울대학교" does not.
_KOREAN_JOSA = (
    # Case particles.
    "이",
    "가",
    "께서",
    "에서",
    "에게서",
    "한테서",
    "을",
    "를",
    "의",
    "에",
    "에게",
    "한테",
    "께",
    "더러",
    "로",
    "으로",
    "로서",
    "으로서",
    "로써",
    "으로써",
    "와",
    "과",
    "하고",
    "랑",
    "이랑",
    "보다",
    "처럼",
    "만큼",
    "같이",
    # Auxiliary particles.
    "은",
    "는",
    "도",
    "만",
    "까지",
    "부터",
    "마저",
    "조차",
    "밖에",
    "마다",
    "대로",
    "뿐",
    "나",
    "이나",
    "라도",
    "이라도",
    "요",
    "야",
    "이야",
    # The copula and its endings.
    "이다",
    "다",
    "입니다",
    "이에요",
    "예요",
    "이요",
    "였다",
    "이었다",
    "이며",
    "며",
    "이고",
    "고",
    "인",
    "라고",
    "이라고",
    "라는",
    "이라는",
    "란",
    "이란",
)
# A run of at most three of them ("에서는", "으로도", "이었다").
_KOREAN_JOSA_RUN = re.compile(
    "(?:"
    + "|".join(sorted(map(re.escape, _KOREAN_JOSA), key=len, reverse=True))
    + "){1,3}"
)


def _is_particle(suffix: str, *, strict: bool = False) -> bool:
    """True if Hangul ``suffix`` reads as an attached particle: a run of the
    closed class ``_KOREAN_JOSA`` with ``strict`` (token F1), else any one
    to three Hangul syllables (containment and coverage, recall-style)."""
    if strict:
        return _KOREAN_JOSA_RUN.fullmatch(suffix) is not None
    return 0 < len(suffix) <= _MAX_KOREAN_SUFFIX and all(
        is_hangul_syllable(ch) for ch in suffix
    )


def _number_suffix_keeps_value(suffix: str, *, strict: bool = False) -> bool:
    """True if Hangul ``suffix`` after a number is a unit and/or a particle."""
    if not suffix or suffix[0] in _KOREAN_MAGNITUDES:
        return False
    for unit in _KOREAN_UNITS:
        if suffix.startswith(unit):
            rest = suffix[len(unit) :]
            return not rest or _is_particle(rest, strict=strict)
    return _is_particle(suffix, strict=strict)


def _token_matches(expected: str, actual: str, *, strict: bool = False) -> bool:
    # A word may carry an attached particle: "서울은", "AWS는". A bare number
    # may carry a unit and a particle ("2024년에", "3명이"), but not a
    # magnitude numeral ("3억", "5천"), which changes the value. ``strict``
    # (token F1) takes only the closed class of particles as one.
    if expected == actual:
        return True
    if not actual.startswith(expected):
        return False
    suffix = actual[len(expected) :]
    if _NUMBER.fullmatch(expected):
        return _number_suffix_keeps_value(suffix, strict=strict)
    return _is_particle(suffix, strict=strict)


# One Han, Hangul or Kana letter, or a run of anything else.
_SCRIPT_TOKEN = re.compile(f"[{DENSE_SCRIPT_LETTERS}]|[^{DENSE_SCRIPT_LETTERS}]+")


def script_tokens(prediction: str, reference: str) -> tuple[list[str], list[str]]:
    """Token-F1 tokens of a normalized prediction and reference, by script.

    Whitespace words, except that a Han, Hangul or Kana letter is a token of
    its own, as the character F1 of the Chinese (CMRC 2018) and Korean
    (KorQuAD 1.0) evaluation scripts: Chinese and Japanese put no space
    between words, so a whitespace word is a whole clause there, and a
    Korean word carries its particles. Unlike those scripts, which split
    every character, Latin words and numbers stay whole ("2024年" is
    "2024", "年"), and a prediction word that is a reference word plus a
    run of Korean particles from their closed class (``_KOREAN_JOSA``) or a
    number's unit ("서울은" for "서울", "2024년에" for "2024년") first
    counts as that word, so the particle costs no precision (KorQuAD
    charges its characters). Any other Hangul after a reference word stays
    ("서울대학교" is five letters against "서울").
    """
    reference_words = sorted(set(reference.split()), key=len, reverse=True)

    def bare(word: str) -> str:
        return next(
            (
                r
                for r in reference_words
                if r != word and _token_matches(r, word, strict=True)
            ),
            word,
        )

    def split(words: list[str]) -> list[str]:
        return [
            token.strip()
            for word in words
            for token in _SCRIPT_TOKEN.findall(word)
            if token.strip()
        ]

    return split([bare(w) for w in prediction.split()]), split(reference.split())


def _compact_window_matches(expected: str, actual: list[str]) -> bool:
    """True if a run of whole tokens, spacing ignored, equals ``expected``.

    The run may end in an attached particle (``_token_matches``), so the
    gold "가나다 상사" (compacted "가나다상사") matches "가나다상사는" and
    "가나다 상사는", but not "가나다상사관리본부".
    """
    for start in range(len(actual)):
        joined = ""
        for token in actual[start:]:
            joined = drop_dense_script_spaces(f"{joined} {token}" if joined else token)
            if _token_matches(expected, joined):
                return True
            if len(joined) > len(expected) + _MAX_KOREAN_SUFFIX:
                break
    return False


def phrase_in_text(phrase: str, text: str, language: str | None = None) -> bool:
    """True if ``phrase`` appears in ``text`` after normalization
    (``normalize_answer`` in ``language``).

    Space-delimited phrases match a contiguous run of whole word tokens, so
    "AI" does not match inside "airport". Any token except a bare number also
    matches a token that extends it by up to three Hangul syllables, so
    "서울 특별시" matches "서울 특별시는" and "AWS" matches "AWS는" (attached
    particles). A bare number also matches it followed by a Korean unit and
    a particle ("2024" in "2024년에"), but not by a magnitude numeral ("3" is
    not "3억", which changes the value). A phrase
    containing CJK is also compared with the spaces next to CJK letters
    removed, so "가나다 상사" matches "가나다상사이다" and "3억 원" matches
    "3억원". A phrase that is a single CJK run (``is_spaceless_script``) is
    matched as a substring of the normalized text (spacing ignored), but not
    where a digit, decimal point or Latin letter continues it on either side:
    "2년" does not match "12년" or "1.2년", nor "二年" "十二年". In a text
    with CJK, any phrase is also matched as such a substring where no
    Hangul letter is next to it, since Chinese and Japanese put no space
    around a Latin word or a number: "AWS" matches "AWSは" and "2024"
    "2024年", but "AI" does not match "airport附近", nor "2" "12年"
    (``_substring_beside_han_or_kana``).

    LIMITATION (CJK): without a morphological segmenter there is no morpheme
    boundary, so a short expected word can match inside a longer one ("가나"
    within "가나상사", "서울" within "서울대학교"). This over-counts rather
    than under-counts; both uses (coverage, containment) are recall-style.

    Possessives are dropped from both texts first (``strip_possessives``),
    so "Depot" matches "Depot's" (exact match and token F1 do not drop them).
    """
    phrase, text = strip_possessives(phrase), strip_possessives(text)
    if is_spaceless_script(phrase):
        cleaned = normalize_answer(phrase, language)
        return bool(cleaned) and _substring_at_boundary(
            cleaned, drop_dense_script_spaces(normalize_answer(text, language))
        )

    expected = tokenize(phrase, language)
    if not expected:
        return False
    actual = tokenize(text, language)
    window = len(expected)
    if any(
        all(
            _token_matches(e, a)
            for e, a in zip(expected, actual[start : start + window], strict=True)
        )
        for start in range(len(actual) - window + 1)
    ):
        return True
    if has_dense_script(phrase) and _compact_window_matches(
        drop_dense_script_spaces(" ".join(expected)), actual
    ):
        return True
    return has_dense_script(text) and _substring_beside_han_or_kana(
        drop_dense_script_spaces(" ".join(expected)),
        drop_dense_script_spaces(" ".join(actual)),
    )


def _substring_beside_han_or_kana(needle: str, haystack: str) -> bool:
    """True if ``needle`` occurs in ``haystack`` where no word, number or
    Hangul letter continues it on either side.

    Chinese and Japanese write no space between words, so a Latin word or a
    number inside them ("AWSは", "合同于2024年签订") is not a whitespace token
    of its own; a Han or Kana neighbour is a word boundary. A Latin letter or
    a digit is not ("AI" is not in "airport", "2" not in "12年"), nor a Han
    numeral after a digit ("3" is not in "3億"). Korean is left to the token
    rules above: a Hangul neighbour is a particle, a unit or a magnitude
    numeral they already judge ("3억").
    """
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle)
        before = haystack[start - 1] if start else ""
        after = haystack[end] if end < len(haystack) else ""
        if not (
            (before and (is_hangul(before) or _continues_word(before, needle[0])))
            or (after and (is_hangul(after) or _continues_word(needle[-1], after)))
        ):
            return True
        start = haystack.find(needle, start + 1)
    return False


# --- the number an answer states ----------------------------------------------

# The English number words, their tens and scale words and the words for
# none (``LanguageProfile.number_words``, ``number_tens``, ``number_scales``,
# ``zero_words``).
_EN = profile(LanguageCode.EN)
_ENGLISH_UNITS = sorted(_EN.number_words, key=_EN.number_words.__getitem__)
_ENGLISH_TENS = dict(_EN.number_tens)
_ENGLISH_ZERO = _EN.zero_words
# Scale words after a number word: "three hundred", "two thousand five
# hundred".
_ENGLISH_SCALES = dict(_EN.number_scales)
# Korean native numerals before a counter, ones and tens apart (열세 is
# 13), and the counters a Korean count answer takes
# (``LanguageProfile.counter_numerals``, ``number_tens``, ``counters``).
_KO = profile(LanguageCode.KO)
_KOREAN_NATIVE_ONES = dict(_KO.counter_numerals)
_KOREAN_NATIVE_TENS = dict(_KO.number_tens)
_KOREAN_COUNTERS = _KO.counters
# Japanese and Chinese numerals before a counter ("三社", "十二件", "两家"):
# the digits, multipliers and counters of the profiles that write Han.
_HAN_LANGUAGES = [profile(LanguageCode.JA), profile(LanguageCode.ZH)]
_HAN_DIGITS = {
    ch: value
    for language in _HAN_LANGUAGES
    for ch, value in language.counter_numerals.items()
}
# The multipliers of Han ("十", "万") and Sino-Korean ("만"), and the
# Sino-Korean digits, which state a number only beside one ("십이").
_MULTIPLIERS = {
    ch: value
    for language in (*_HAN_LANGUAGES, _KO)
    for ch, value in language.numeral_multipliers.items()
}
_SINO_KOREAN_DIGITS = dict(_KO.multiplier_numerals)
_HAN_COUNTERS = tuple(
    dict.fromkeys(c for language in _HAN_LANGUAGES for c in language.counters)
)


def _magnitude(value: int) -> int:
    """The place a number word fills: ones, teens, tens or hundreds."""
    return 0 if value < 10 else 1 if value < 20 else 2 if value < 100 else 3


def _compose(
    parts: list[tuple[int, bool]], ones_before_tens: bool = False
) -> int | None:
    """The value of a written number's words, each ``(value, scale)``:
    "twenty-one" 21, "three hundred and twelve" 312, "deux mille trois
    cents" 2300, "dos mil millones" 2e9, and with ``ones_before_tens``
    (``LanguageProfile.ones_before_tens``) "vierundzwanzig" (4, 20) 24.
    ``None`` when the words are two numbers, not one ("deux trois", "zwei
    und drei": a place filled twice; "deux vingt": a place after a smaller
    one)."""
    total = current = 0
    largest = 0
    filled: set[int] = set()
    for value, scale in parts:
        if not scale:
            place = _magnitude(value)
            if place in filled:
                return None
            # The places go down, but for ones before tens where written so.
            lowest = min(filled, default=place)
            ones_first = ones_before_tens and place == 2 and lowest == 0
            if place > lowest and not ones_first:
                return None
            filled.add(place)
            current += value
        elif value == 100:
            if 3 in filled:
                return None
            current, filled = (current or 1) * 100, {3}
        else:
            # A larger scale after a smaller one multiplies all before it
            # ("mil millones"); a smaller one adds ("two million three
            # thousand").
            if largest and value > largest:
                total = (total + current) * value
            else:
                total += (current or 1) * value
            largest = max(largest, value)
            current, filled = 0, set()
    return total + current


def _phrases(words: Iterable[str]) -> str:
    """``words`` as a regex alternation, longest first, each written with
    any spaces between its words, a hyphen or a space where it has a hyphen
    ("quatre-vingts", "quatre vingts") and either apostrophe."""

    def phrase(word: str) -> str:
        return (
            re.escape(word)
            .replace(r"\ ", r"\s+")
            .replace(r"\-", r"[-\s]")
            .replace("'", "['\u2019]")
        )

    return "(?:" + "|".join(map(phrase, sorted(words, key=len, reverse=True))) + ")"


class _NumberGrammar:
    """How one language writes a number in words
    (``LanguageProfile.number_words``, ``number_tens``, ``number_scales``,
    ``number_conjunctions``, ``compound_ones``, ``ones_before_tens``): the
    words, as separate words or one ("vingt-quatre", "treinta y dos",
    "vierundzwanzig", "duecentotrentadue"), read by :func:`_compose`, and
    its words for none (``zero_words``: "aucun", "kein", "ни одного")."""

    def __init__(self, language: LanguageProfile) -> None:
        self.words = {**language.number_words, **language.number_tens}
        self.scales = dict(language.number_scales)
        self.ones = frozenset(language.compound_ones)
        self.ones_before_tens = language.ones_before_tens
        word = _phrases({*self.words, *self.scales, *self.ones})
        self.zero = (
            re.compile(rf"(?<!\w){_phrases(language.zero_words)}(?!\w)", re.I)
            if language.zero_words
            else None
        )
        joint = "(?:" + "|".join(map(re.escape, language.number_conjunctions)) + ")"
        # Words joined by a hyphen or spaces (on one line), a conjunction
        # between them, or nothing: a compound written as one word.
        space = r"(?:-|[^\S\n])+"
        if language.number_conjunctions:
            join = rf"(?:{space}(?:{joint}{space})?|{joint})?"
        else:
            join = rf"(?:{space})?"
        self.pattern = re.compile(
            rf"(?<!\w){word}(?:{join}{word})*(?!\w)", re.IGNORECASE
        )
        self._token = re.compile(word, re.IGNORECASE)
        self._join = re.compile(join, re.IGNORECASE)

    def value(self, match: re.Match[str]) -> int | None:
        """The number ``match`` writes, ``None`` when it is none: an
        article alone ("un fournisseur"), or two numbers side by side."""
        text, parts, ones = match.group(), [], 0
        position = 0
        while position < len(text):
            token = self._token.match(text, position)
            if token is None:
                return None
            word = re.sub(r"\s+", "-", token.group().lower())
            if word in self.scales:
                parts.append((self.scales[word], True))
            elif word in self.words:
                parts.append((self.words[word], False))
            else:
                parts.append((1, False))
                ones += 1
            joined = self._join.match(text, token.end())
            position = joined.end() if joined else token.end()
        if ones == len(parts):
            return None
        return _compose(parts, self.ones_before_tens)


# The other languages' written numbers (English's are read by its own
# grammar below, Korean's and the Han ones before a counter), read in an
# answer in that language instead of English's.
_LANGUAGE_NUMBERS: dict[str, _NumberGrammar] = {
    language.code.value: _NumberGrammar(language)
    for language in profiles()
    if language.number_words and language.code is not LanguageCode.EN
}

# Digits, with thousands separators and a decimal part.
_DIGIT_FORM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_DIGITS = re.compile(rf"-?(?:{_DIGIT_FORM})")
# A number word below a hundred ("twenty-one", "seven"), then any scale
# words and the words after them ("three hundred and twelve").
_ENGLISH_SMALL = (
    r"(?:(?:"
    + "|".join(_ENGLISH_TENS)
    + r")(?:[-\s](?:"
    + "|".join(_ENGLISH_UNITS[1:10])
    + r"))?|(?:"
    + "|".join(sorted(_ENGLISH_UNITS, key=len, reverse=True))
    + r"))"
)
_ENGLISH_NUMBER = re.compile(
    rf"\b(?:(?P<count>{_ENGLISH_SMALL}(?:[-\s]+(?:"
    + "|".join(_ENGLISH_SCALES)
    + rf")(?:(?:\s+and)?\s+{_ENGLISH_SMALL}(?!\w))?)*)|(?P<zero>"
    + "|".join(_ENGLISH_ZERO)
    + r"))\b",
    re.IGNORECASE,
)
_KOREAN_NUMBER = re.compile(
    rf"(?<!{_HANGUL})(?P<tens>"
    + "|".join(sorted(_KOREAN_NATIVE_TENS, key=len, reverse=True))
    + r")?(?P<ones>"
    + "|".join(sorted(_KOREAN_NATIVE_ONES, key=len, reverse=True))
    + r")?\s?(?="
    + "|".join(_KOREAN_COUNTERS)
    + r")"
)
# "없습니다" on its own word ("벤더가 없습니다"), not closing a compound
# adjective ("문제없습니다", no problem).
_KOREAN_NONE = re.compile(rf"(?<!{_HANGUL})없(?:습니다|어요|다|음)")
# Zero stated by a Chinese or Japanese verb of absence: "没有供应商",
# "サプライヤーはいません", not "no problem" ("没有问题", "問題ありません")
# nor a negated copula ("Aではありません").
_CJK_NONE = re.compile(
    r"(?:没有|沒有)(?!问题|問題|关系|關係)"
    r"|(?:(?<!で)(?<!問題)[はがも]|^|(?<=[\s、。]))(?:いません|ありません|おりません|ございません)",
    re.MULTILINE,
)
# Han, Sino-Korean and ASCII digits with the multipliers between them
# ("十二", "一万二千", "1만 2천", "만 이천", "三", "1.5万", "3,000万");
# ``_cjk_value`` reads them.
_CJK_PART = (
    f"(?:{_DIGIT_FORM}|["
    + "".join([*_HAN_DIGITS, *_SINO_KOREAN_DIGITS, *_MULTIPLIERS])
    + "])"
)
_CJK_PART_TOKEN = re.compile(rf"{_DIGIT_FORM}|\S")
_CJK_NUMBER = re.compile(
    rf"(?<!{_HANGUL})(?<![\d.,]){_CJK_PART}(?:[^\S\n]?{_CJK_PART})*"
)
_CJK_COUNTER_AFTER = re.compile(
    r"[^\S\n]?(?:"
    + "|".join(sorted({*_KOREAN_COUNTERS, *_HAN_COUNTERS}, key=len, reverse=True))
    + ")"
)


def _cjk_value(match: re.Match[str]) -> float | None:
    """The number of a ``_CJK_NUMBER`` match: Han digits before a counter
    ("三社", "二〇二四年" is no count), or digits with multipliers ("十二台",
    "一万二千", "一千零一家", "1만 2천", "십이 대") before a counter, or
    after ASCII digits alone ("3万", "1.5万", "3,000万", "1만 2천"); ``None``
    for anything else (bare ASCII digits are ``_DIGITS``'). Between
    multipliers a Han zero holds an empty place ("一千零一" is 1001)."""
    parts = _CJK_PART_TOKEN.findall(match.group())
    multiplied = any(part in _MULTIPLIERS for part in parts)
    counted = bool(_CJK_COUNTER_AFTER.match(match.string, match.end()))
    if not multiplied:
        if not counted or not all(part in _HAN_DIGITS for part in parts):
            return None
        return float("".join(str(_HAN_DIGITS[part]) for part in parts))
    if not counted and not parts[0][0].isdigit():
        return None
    total = section = 0.0
    digit: float | None = None
    for part in parts:
        if _HAN_DIGITS.get(part, 1) == 0:
            continue
        if part in _MULTIPLIERS:
            multiplier = _MULTIPLIERS[part]
            if multiplier >= 10**4:
                # 万, 億: they multiply the whole section before them
                # ("十二万" is 120,000).
                total += ((section + (digit or 0)) or 1) * multiplier
                section = 0
            else:
                section += (1 if digit is None else digit) * multiplier
            digit = None
        elif digit is not None:
            return None
        else:
            digit = (
                float(part.replace(",", ""))
                if part[0].isdigit()
                else _HAN_DIGITS.get(part, _SINO_KOREAN_DIGITS.get(part, 0))
            )
    return total + section + (digit or 0)


def _stands_alone(text: str, start: int, end: int) -> bool:
    """Whether ``text[start:end]`` is no part of a Latin word or a longer
    number: a Han, Hangul or Kana letter may touch it ("3곳", "3社"), a
    Latin letter or digit may not ("Zortal13", "3rd")."""

    def joins(ch: str) -> bool:
        return ch.isdigit() or (ch.isalpha() and not is_dense_script_char(ch))

    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""
    if before and (
        joins(before) or (before in ".," and start > 1 and text[start - 2].isdigit())
    ):
        return False
    return not (
        after
        and (joins(after) or (after in ".," and text[end + 1 : end + 2].isdigit()))
    )


# What a number that is no count stands next to. A counter of either
# language (and 番) may sit between the number and an ordinal or negation
# suffix: "두 번째", "三つ目", "一つではなく", "한 곳이 아니라".
_ANY_COUNTER = (
    r"(?:"
    + "|".join(sorted({*_KOREAN_COUNTERS, *_HAN_COUNTERS, "番"}, key=len, reverse=True))
    + r")"
)
# An ordinal: Korean 째 ("두 번째", "2번째"), Japanese 目 ("三つ目", "3番目"),
# Chinese and Japanese 第 before the number ("第三", "第3").
_ORDINAL_AFTER = re.compile(rf"\s?{_ANY_COUNTER}?\s?(?:째|目)")
_ORDINAL_BEFORE = re.compile(r"第\s?$")
# The number itself negated: "not one but three", "不是一个", "一つではなく",
# "한 곳이 아니라".
_NEGATED_BEFORE = re.compile(r"(?:\bnot\s+|(?:不是|并非|並非|不止)\s?)$", re.IGNORECASE)
_NEGATED_AFTER = re.compile(
    rf"\s?{_ANY_COUNTER}?\s?(?:ではな|じゃな|でな|(?:이|가)?\s?아니)"
)
# The whole a partitive takes its part from: "三家(供应商)中只有一家", "세 곳
# 중 한 곳", "三社のうち一社" (and in the answer's language "Of the three
# vendors, only one ...": ``_CountContext``). Bare 中 is a partitive only
# before a comma or what states the part ("三家中国供应商" is three Chinese
# vendors).
_DENSE_WORD = f"[{DENSE_SCRIPT_LETTERS}]"
_PARTITIVE_WHOLE_AFTER = re.compile(
    rf"\s?{_ANY_COUNTER}?(?:の|의)?\s?{_DENSE_WORD}{{0,5}}?\s?(?:之中|当中|當中"
    r"|中(?=[，,、\s只仅僅有的共就\d一二两兩三四五六七八九十])"
    rf"|のうち|の中で|중(?:에서|에)?(?!{_HANGUL})|가운데)"
)
# "Not one vendor" and "not even one" state zero, unless "but" corrects
# them ("not one but three") or a partitive follows ("is not one of them").
_NOT_ONE_BEFORE = re.compile(r"\bnot\s+(?:even\s+)?$", re.IGNORECASE)
_BUT_AFTER = re.compile(r"(?:\s+\w+)?\s*,?\s*but\b|\s+of\b", re.IGNORECASE)
_NOT_A_SINGLE = re.compile(
    r"\bnot\s+(?:even\s+)?a\s+single\b(?!(?:\s+\w+)?\s*,?\s*but\b)", re.IGNORECASE
)
# A list enumerator ("1. Acme", "2) Beta", "(3) Gamma"): at the line start,
# before the item's text.
_ENUMERATOR_BEFORE = re.compile(r"(?:^|\n)[ \t]*[(\[]?$")
_ENUMERATOR_AFTER = re.compile(r"[.)\]][ \t]+\S")
# A citation marker ("[1]", "[2, 3]", "[4-6]").
_CITATION_BEFORE = re.compile(r"\[[\d,\s–-]*$")
_CITATION_AFTER = re.compile(r"[\d,\s–-]*\]")
# A year: four digits before a word that follows one (``year_suffixes``:
# "2019년", "2019以降") or opening an ISO date (or after a word of time in
# the answer's language: ``_CountContext``), or before a particle that
# closes a range (``range_particles``: "2019부터", "从2019到现在") whose
# other end is no four-digit count ("1500から2000社" ranges over counts).
_YEAR_AFTER = re.compile(
    r"\s?"
    + _phrases({w for language in profiles() for w in language.year_suffixes})
    + r"|-\d{1,2}-\d{1,2}"
)
_RANGE_AFTER = re.compile(
    r"\s?"
    + _phrases({w for language in profiles() for w in language.range_particles})
    + rf"(?P<counts>\s?\d{{4}}(?!\d)\s?{_ANY_COUNTER})?"
)
# "no" before these words is no count of zero: "No single vendor
# dominates", "no more than 5", "no longer".
_NO_MODIFIES = re.compile(
    r"\s+(?:single|other|more|fewer|less|longer|later|sooner|earlier|"
    r"matter|doubt)\b",
    re.IGNORECASE,
)
# One with "even" after it is zero: "한 곳도 없습니다", "一社もありません",
# "一家也没有".
_NOT_EVEN_ONE = re.compile(rf"\s?{_ANY_COUNTER}?\s?(?:도|も|也|都)")
# "No," / "No." answers a yes/no question; "one of them" is a partitive.
_NO_POLARITY = re.compile(r"\s*[*_]*\s*[,.;:!?]")

# The English month names (strptime's, which the profiles leave out).
_EN_MONTHS = (
    "january february march april may june july august september october "
    "november december"
).split()


class _CountContext:
    """The words of one language around a number that make it no count
    (``LanguageProfile.year_words``, ``month_words``, ``date_fillers``,
    ``partitive_words``, ``partitive_anaphors``, ``ordinal_words``): a year
    after a word of time ("in March 2024", "seit 2019", "en marzo de
    2019"), the whole of a partitive opening a clause ("Of the three
    vendors, only one", "De los tres proveedores"), a one taken from a set
    counted elsewhere ("one of them", "один из них"; "one of the vendors"
    states one) and the tens of an ordinal ("twenty-first", "vingt-
    troisième")."""

    def __init__(self, language: LanguageProfile) -> None:
        # The English month names are strptime's: a profile lists only its
        # others.
        months = {*_EN_MONTHS, *(w for ws in language.month_words.values() for w in ws)}
        fillers = (
            rf"(?:{_phrases(language.date_fillers)}\s+)?"
            if language.date_fillers
            else ""
        )
        self.year_before = (
            re.compile(
                rf"(?<!\w){_phrases(language.year_words)}[\s-]+"
                rf"(?:{_phrases(months)}\.?[\s-]+(?:\d{{1,2}},?\s+)?{fillers})?$",
                re.IGNORECASE,
            )
            if language.year_words
            else None
        )
        self.partitive_before = (
            re.compile(
                rf"(?:^|[.!?;:]\s+|\n[ \t]*){_phrases(language.partitive_words)}\s+$",
                re.IGNORECASE,
            )
            if language.partitive_words
            else None
        )
        self.one_of = (
            re.compile(
                rf"\s+{_phrases(language.partitive_anaphors)}(?!\w)", re.IGNORECASE
            )
            if language.partitive_anaphors
            else None
        )
        self.ordinal_after = (
            re.compile(
                rf"(?:-|[^\S\n]+){_phrases(language.ordinal_words)}(?!\w)",
                re.IGNORECASE,
            )
            if language.ordinal_words
            else None
        )


def _count_value(
    body: str, start: int, end: int, value: float, context: _CountContext
) -> float | None:
    """``value``, the number at ``body[start:end]``, as the count it states:
    ``None`` when it states none (``_not_a_count``) or is a one taken from
    a set counted elsewhere ("one of them"), 0 for one with "even" after it
    ("한 곳도", "一社も") or "not" before it ("not one vendor")."""
    if (
        value == 1
        and _NOT_ONE_BEFORE.search(body, 0, start)
        and not _BUT_AFTER.match(body, end)
    ):
        return 0.0
    if _not_a_count(body, start, end, context):
        return None
    if value == 1 and context.one_of is not None and context.one_of.match(body, end):
        return None
    if value == 1 and _NOT_EVEN_ONE.match(body, end):
        return 0.0
    return value


def _not_a_count(body: str, start: int, end: int, context: _CountContext) -> bool:
    """Whether the number at ``body[start:end]`` states no count: a list
    enumerator, a citation marker, a year, an ordinal, a negated number
    ("not one but three", "一つではなく") or the whole of a partitive ("of
    the three, one", "三家中一家")."""
    before, after = body[:start], body[end:]
    if _ENUMERATOR_BEFORE.search(before) and _ENUMERATOR_AFTER.match(after):
        return True
    if _CITATION_BEFORE.search(before) and _CITATION_AFTER.match(after):
        return True
    number = body[start:end]
    if re.fullmatch(r"\d{4}", number) and (
        _YEAR_AFTER.match(after)
        or ((ranged := _RANGE_AFTER.match(after)) and not ranged.group("counts"))
        or (context.year_before is not None and context.year_before.search(before))
    ):
        return True
    if _ORDINAL_BEFORE.search(before) or _ORDINAL_AFTER.match(after):
        return True
    if context.ordinal_after is not None and context.ordinal_after.match(after):
        return True
    if _PARTITIVE_WHOLE_AFTER.match(after) or (
        context.partitive_before is not None and context.partitive_before.search(before)
    ):
        return True
    return bool(_NEGATED_BEFORE.search(before) or _NEGATED_AFTER.match(after))


def _english_count(words: str) -> int | None:
    """The value of English number words: "twenty-one" 21, "three hundred
    and twelve" 312, "two thousand five hundred" 2500 (the regex reads
    only well-formed ones)."""
    parts = [
        (
            (_ENGLISH_SCALES[word], True)
            if word in _ENGLISH_SCALES
            else (
                (
                    _ENGLISH_TENS[word]
                    if word in _ENGLISH_TENS
                    else _ENGLISH_UNITS.index(word)
                ),
                False,
            )
        )
        for word in re.split(r"[-\s]+", words.lower())
        if word != "and"
    ]
    return _compose(parts)


def _english_value(match: re.Match[str]) -> int | None:
    """The value of an ``_ENGLISH_NUMBER`` match, ``None`` when the word
    states no count there ("No," answering yes/no, "no single", "one of")."""
    after = match.string[match.end() :]
    if match.group("zero"):
        word = match.group("zero").lower()
        if word == "no" and (_NO_POLARITY.match(after) or _NO_MODIFIES.match(after)):
            return None
        return 0
    return _english_count(match.group("count"))


def _absence_pattern(language: LanguageProfile) -> re.Pattern[str] | None:
    """``language``'s phrases of absence (``absence_phrases``) as one
    pattern: "…" up to three words, a leading "-" a clitic, either
    apostrophe."""

    def phrase(text: str) -> str:
        clitic = text.startswith("-")
        parts = [
            re.escape(part.strip()).replace(r"\ ", r"\s+").replace("'", "['\u2019]")
            for part in text.lstrip("-").split("…")
        ]
        body = r"\s+(?:\w+\s+){0,3}?".join(parts)
        return (
            ("" if clitic else r"(?<!\w)")
            + body
            + ("" if text.endswith("'") else r"(?!\w)")
        )

    if not language.absence_phrases:
        return None
    phrases = sorted(language.absence_phrases, key=len, reverse=True)
    return re.compile("|".join(map(phrase, phrases)), re.IGNORECASE)


# The phrases of absence by language, its comparatives and the words that
# open a clause inside a sentence: English's for an answer in a language
# whose numbers the reader does not know in words.
_ABSENCES = {
    language.code.value: pattern
    for language in profiles()
    if (pattern := _absence_pattern(language)) is not None
}


def _comparative_pattern(words: Iterable[str]) -> re.Pattern[str]:
    """``words`` as whole words: a comparative ends at a word boundary
    ("más de", not "más detalles"; an elided one at its apostrophe:
    "plus d'")."""
    alternatives = "|".join(
        _phrases([word]) + ("" if word.endswith("'") else r"(?!\w)")
        for word in sorted(words, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})", re.IGNORECASE)


_COMPARATIVES = {
    language.code.value: _comparative_pattern(language.comparatives)
    for language in profiles()
    if language.comparatives
}
# A clause break inside a sentence: a comma, a colon or a dash (any
# language's), or one of the language's ``clause_conjunctions``. A
# semicolon is none: the clause after it states a fact of its own ("does
# not have any suppliers; it uses 3 plants").
_CLAUSE_MARK = r"[,:\u2013\u2014\uff0c\uff1a\u3001]|\s-\s"
_CLAUSE_BREAKS = {
    language.code.value: re.compile(
        _CLAUSE_MARK
        + (
            rf"|(?<!\w){_phrases(language.clause_conjunctions)}(?!\w)"
            if language.clause_conjunctions
            else ""
        ),
        re.IGNORECASE,
    )
    for language in profiles()
    if language.absence_phrases
}


def _absence(body: str, language: str) -> int | None:
    """Where ``body`` first states a count of 0 by a phrase of absence of
    ``language`` (``LanguageProfile.absence_phrases``), or ``None``: not
    where a comparative directly follows the phrase ("No hay más de tres"),
    nor where a count follows it after a clause break in its sentence ("No
    hay duda de que son tres", "Нет сомнений, их три"), nor a one-word phrase opening a line
    before a comma or a stop ("Нет."), which answers a yes/no question. A
    number in the phrase's own clause is the absent thing's ("No hay
    proveedores en el lote 7", "with more than 100 employees"): the zero is
    read where the clause opens when a number before the phrase follows a
    comparative ("Поставщиков с более чем 50 сотрудниками нет")."""
    pattern = _ABSENCES.get(language)
    breaks = _CLAUSE_BREAKS.get(language, _CLAUSE_BREAKS["en"])
    for match in pattern.finditer(body) if pattern is not None else ():
        if (
            not re.search(r"\s", match.group())
            and _opens_line(body, match.start())
            and _NO_POLARITY.match(body, match.end())
        ):
            continue
        start, end = _sentence_around(body, match.start())
        if _comparative_at(body, match.end(), end, language) is not None:
            continue
        clause_start = max(
            (m.end() for m in breaks.finditer(body, start, match.start())),
            default=start,
        )
        after = breaks.search(body, match.end(), end)
        if (
            after is not None
            and _leading_number(body[after.start() : end], language, absences=False)
            is not None
        ):
            continue
        return _zero_offset(body, clause_start, match.start(), language)
    return None


def _comparative_at(
    body: str, start: int, end: int, language: str
) -> re.Match[str] | None:
    """The comparative of ``language`` that opens ``body[start:end]`` after
    spaces, or ``None``."""
    comparative = _COMPARATIVES.get(language)
    if comparative is None:
        return None
    rest = body[start:end]
    match = comparative.match(body, start + len(rest) - len(rest.lstrip()), end)
    return (
        match if match is not None and _compares(body, match, end, language) else None
    )


def _compares(body: str, match: re.Match[str], end: int, language: str) -> bool:
    """Whether the comparative ``match`` compares: a one-word one
    ("больше") only before a number ("больше трёх", not "больше
    данных")."""
    if re.search(r"\s", match.group()):
        return True
    word = body[match.end() : end].split(maxsplit=1)[:1]
    return bool(word) and _leading_number(word[0], language, absences=False) is not None


def _zero_offset(body: str, clause_start: int, phrase: int, language: str) -> int:
    """Where the zero of a phrase of absence at ``phrase`` is read: where
    its clause opens when every number before the phrase in it follows a
    comparative (a condition of the absent thing), else at the phrase (a
    count before it is read first)."""
    comparative = _COMPARATIVES.get(language)
    matches = comparative.finditer(body, clause_start, phrase) if comparative else ()
    first = next((m for m in matches if _compares(body, m, phrase, language)), None)
    if (
        first is not None
        and _leading_number(
            body[clause_start : first.start()], language, absences=False
        )
        is None
    ):
        return clause_start
    return phrase


def _sentence_around(body: str, index: int) -> tuple[int, int]:
    """The span of the sentence of ``body`` that holds ``index``."""
    start = 0
    for end in sentence_ends(body):
        if end > index:
            return start, end
        start = end
    return start, len(body)


def _opens_line(body: str, start: int) -> bool:
    """Whether only spaces and markdown emphasis stand before ``start`` on
    its line of ``body``."""
    line_start = body.rfind("\n", 0, start) + 1
    return not body[line_start:start].strip(" \t*_")


# The words around a number in the answer's language: English's for an
# answer in a language whose numbers the reader does not know in words.
_COUNT_CONTEXTS: dict[str, _CountContext] = {
    code: _CountContext(profile(code))
    for code in (LanguageCode.EN.value, *_LANGUAGE_NUMBERS)
}


def leading_number(text: str, language: str | None = None) -> float | None:
    """The first number ``text`` states as a count, or ``None``: digits
    ("4,486", "3곳", "3社"), English number words ("Three", "twenty-one",
    "three hundred and twelve"; "none", "no", "not one" and "not a single"
    for 0), a Korean native numeral before a counter ("세 곳", "열두 개사";
    "없습니다" for 0) or a Han numeral before a counter ("三社", "两家", "三
    社"; "没有" and "いません"/"ありません" for 0). Markdown heading lines are
    skipped; a number inside a word ("Zortal13") is none, and so are numbers
    that count nothing: list enumerators ("1. Acme"), citation markers
    ("[2]"), years ("as of 2024", "2024년", "2019以降"), ordinals ("두 번째", "三つ目",
    "第三"), a negated number ("not one but three", "一つではなく", "不是一个",
    "한 곳이 아니라"), the whole of a partitive ("Of the three vendors, only
    one", "三家中一家", "세 곳 중 한 곳", "三社のうち一社": the part counts),
    "no" answering a yes/no question or before "single", "more" and the like
    ("No single vendor"), and "one of them". A count answer states its
    number first ("Three vendors ... Two of them ..." is 3), so a numeric
    gold is compared with this number, not with every number the answer
    happens to name. With ``language`` German, Spanish, French, Italian,
    Portuguese or Russian, that language's written numbers and words for
    none count instead of English's ("drei", "vingt-quatre", "treinta y
    dos", "Zwei Millionen", "три тысячи"; "aucun", "kein", "ни одного" and
    its phrases of absence, ``absence_phrases``, such as "no hay" and "il
    n'y a pas de", for 0: Spanish "no" is no zero), and its words of time,
    partitives and ordinals tell which state no count ("seit 2019", "De
    los tres", "один из них", "vingt-troisième"); two side by side ("deux
    trois") are none.
    The digits, and Korean and Han numerals, read in every language."""
    return _leading_number(text, language)


def _leading_number(
    text: str, language: str | None, *, absences: bool = True
) -> float | None:
    """:func:`leading_number`, without the phrases of absence unless
    ``absences``."""
    lines = [
        line
        for line in unicodedata.normalize("NFKC", text).splitlines()
        if not re.match(r"\s*#{1,6}(?:\s|$)", line)
    ]
    body = "\n".join(lines)
    found: list[tuple[int, float]] = []
    grammar = _LANGUAGE_NUMBERS.get(language or "")
    context = _COUNT_CONTEXTS.get(language or "", _COUNT_CONTEXTS["en"])

    def first(
        matches: Iterator[re.Match[str]],
        read: Callable[[re.Match[str]], float | None],
    ) -> None:
        # The first match of one form that states a count.
        for match in matches:
            value = read(match)
            if value is not None:
                value = _count_value(body, match.start(), match.end(), value, context)
            if value is not None:
                found.append((match.start(), float(value)))
                return

    first(
        _DIGITS.finditer(body),
        lambda m: (
            float(m.group().replace(",", ""))
            if _stands_alone(body, m.start(), m.end())
            # A multiplier after them scales them ("3万", "1만"): _cjk_value.
            and body[m.end() : m.end() + 1] not in _MULTIPLIERS
            else None
        ),
    )
    zeros = [
        _KOREAN_NONE.search(body),
        _CJK_NONE.search(body),
    ]
    if grammar is None:
        first(_ENGLISH_NUMBER.finditer(body), _english_value)
        zeros.append(_NOT_A_SINGLE.search(body))
    else:
        first(grammar.pattern.finditer(body), grammar.value)
        if grammar.zero is not None:
            first(grammar.zero.finditer(body), lambda _: 0.0)
    first(
        _KOREAN_NUMBER.finditer(body),
        lambda m: (
            _KOREAN_NATIVE_TENS.get(m.group("tens") or "", 0)
            + _KOREAN_NATIVE_ONES.get(m.group("ones") or "", 0)
            if m.group("tens") or m.group("ones")
            else None
        ),
    )
    first(_CJK_NUMBER.finditer(body), _cjk_value)
    for zero in zeros:
        if zero is not None:
            found.append((zero.start(), 0.0))
    if (
        absences
        and (absent := _absence(body, (language or "") if grammar else "en"))
        is not None
    ):
        found.append((absent, 0.0))
    return min(found)[1] if found else None


def number_value(text: str, language: str | None = None) -> float | None:
    """The number a gold answer is ("3", "4,486", "three"), else ``None``."""
    value = leading_number(text, language)
    return value if value is not None and normalize_answer(text) else None
