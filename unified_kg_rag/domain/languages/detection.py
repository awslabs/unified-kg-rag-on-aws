# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language a text's script and words decide.

One scoring decision (:func:`script_language`) tells the language of a
query and of a corpus sample wherever the package asks (the decline's
language, the corpus check, prompt tuning, token budgets), from one record
of evidence (:func:`text_evidence`):

- the text's letters by script, and their runs (words, or stretches of a
  script written without spaces: ``ScriptCounts``);
- its words that may be a language's function words: none that is an
  acronym or a code ("AI", "OS", "iPhone": a capital after its first
  letter), a part of a compound (joined by a hyphen, a digit or another
  script's letter: "to-do", "us-east-1", "ES6", "AIとは"), of a URL, an
  email, a path or a dotted name ("example.com", "admin@example.es",
  "/usr/bin/the", "com.example.Main") or, in a text of two scripts, a
  word of a name it quotes (a run of capitalised words: "Bank of
  America", "The Walt Disney Company"); a shouted text (three or more
  words, every one in capitals: "QUI A SIGNÉ LE CONTRAT ?"; beside
  another script, one of them longer than two letters, so "AS IS / TO BE
  비교" holds acronyms) is read lowercased, its words neither acronyms nor
  names; a colon joins a word only before "//", a digit or another colon
  ("https://", "std::vector"; "Frage:Wer" is a label);
- the texts that open on a capitalised word (a sentence starts there), by
  the script of that word.

The scripts whose languages have function words (Latin) claim a text with
the words that are function words of one of them, and one more when it
opens on a capitalised word of theirs; a script without them is quoted
names there. Every other script claims it with its runs, the heaviest of
them (``ScriptCounts.dominant``) for all of them. The larger claim decides
the script, a tie the one with function words: "Who is the CEO of
한빛전자?" is English (four function words and its opening against one
run), "AI 모델 추천" Korean (no function word: "AI" is an acronym),
"Microsoft Corporation和OpenAI是什么关系？" Chinese (the Latin words are
a name), "with 문 사용법" Korean (one function word against two runs).

The language in the script: Korean, Japanese and Chinese are told apart by
their dense letters (:func:`dense_language`): Korean writes Hangul and may
mix in Han (Hanja), Japanese writes Kana and Han (Kanji), Chinese Han
alone. A text whose Han, Hangul and Kana letters are a tenth or more Hangul
(and no fewer Hangul than Kana) is Korean, else a tenth or more Kana
Japanese, else Chinese; the shares are of the dense letters only, so Latin
terms in technical text do not move them. A script one language writes is
that language, and the languages of one script by their function words and
the elisions only one of them writes.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile
from unified_kg_rag.domain.languages.registry import PROFILES
from unified_kg_rag.shared.utils.scripts import (
    ScriptCounts,
    count_letters,
    letter_class,
    script_of,
)

__all__ = [
    "TextEvidence",
    "decided_language",
    "dense_language",
    "language_in_script",
    "script_language",
    "text_evidence",
    "text_language",
]

# The share of a text's dense letters in Hangul (or Kana) that makes it
# Korean (or Japanese). Han carries the stems of both: a Korean statute
# written with Hanja holds about a third Hangul, Japanese prose about half
# Kana, and Chinese none of either.
_SCRIPT_SHARE = 0.1


def dense_language(
    counts: ScriptCounts, fallback: LanguageCode | None = None
) -> LanguageCode | None:
    """The dense-script language ``counts`` read as (see the module), or
    ``None`` without Han, Hangul or Kana letters. Han alone is Chinese,
    unless ``fallback`` (the index language) is Japanese: a Japanese
    index's kanji-only query is Japanese."""
    dense = counts.dense
    if not dense:
        return None
    hangul, kana = counts["hangul"], counts["kana"]
    if hangul >= _SCRIPT_SHARE * dense and hangul >= kana:
        return LanguageCode.KO
    if kana >= _SCRIPT_SHARE * dense:
        return LanguageCode.JA
    if not hangul and not kana and fallback is LanguageCode.JA:
        return LanguageCode.JA
    return LanguageCode.ZH


# The scripts whose languages have function words (spaced scripts, each
# its own group in ``ScriptCounts``), and those languages by script: their
# words are the evidence the words of a text give.
_WORDED: dict[str, list[LanguageProfile]] = {}
for _profile in PROFILES.values():
    if _profile.function_words:
        _WORDED.setdefault(_profile.script, []).append(_profile)
# Their function words and articles: a capitalised one opens no name
# ("Why", "Wie", "Quién"), but an article directly before a capitalised
# word opens the name it belongs to ("The Walt Disney Company", "the
# United Nations", "der Rhein").
_ARTICLES = frozenset().union(*(p.articles for ps in _WORDED.values() for p in ps))
_FUNCTION_WORDS = _ARTICLES.union(
    *(p.function_words for ps in _WORDED.values() for p in ps)
)
_ELIDED = frozenset().union(*(p.elided_prefixes for ps in _WORDED.values() for p in ps))
# A word of those scripts, an elided one with its apostrophe
# ("dell'accordo"), and any letter or digit (what a compound joins).
_WORD = re.compile(rf"[{letter_class(*_WORDED)}]+(?:'[{letter_class(*_WORDED)}]+)*")
_LETTER_OR_DIGIT = re.compile(r"[^\W_]")
# The characters that join a word to a letter or digit beside them: a
# compound's hyphen, and the punctuation of URLs, emails, paths and dotted
# or snake_case names.
_JOINERS = frozenset("-./@:_\\")
# A colon that joins: before "//", a digit or another colon, or after one.
_COLON_JOINS = re.compile(r"::|:(?://|\d)")
# The fewest words of a shouted text: one or two capitalised words are
# acronyms ("AI ethics", "SUPPLIERS OF").
_SHOUTED_WORDS = 3


@dataclass(frozen=True)
class TextEvidence:
    """What :func:`script_language` reads of one text or a sample of
    several (:func:`text_evidence`): the letters by script with their runs,
    the lowercased words that may be function words, and how many of the
    texts open on a capitalised word, by its script."""

    counts: ScriptCounts
    words: tuple[str, ...] = ()
    openings: Mapping[str, int] = field(default_factory=dict)


def text_evidence(texts: Iterable[str]) -> TextEvidence:
    """The evidence ``texts`` give of their language (module docstring)."""
    texts = list(texts)
    words: list[str] = []
    openings: Counter[str] = Counter()
    for text in texts:
        text = text.replace("\u2019", "'")
        spans = list(_WORD.finditer(text))
        # A shouted text's capitals mark no acronym nor name.
        shouted = len(spans) >= _SHOUTED_WORDS and all(
            span.group().isupper() for span in spans
        )
        # Beside another script, capitals of a function word's length
        # alone are acronyms ("AS IS / TO BE 비교", "IT OT IS の違い").
        if shouted and _mixes_scripts(text):
            shouted = any(len(span.group()) > 2 for span in spans)
        # In a text of one script its capitalised words quote nothing
        # (German writes its nouns so).
        names = (
            _name_words(text, spans) if _mixes_scripts(text) and not shouted else set()
        )
        words.extend(
            span.group().lower()
            for i, span in enumerate(spans)
            if i not in names
            and (shouted or not _is_code(span.group()))
            and not _joined(text, span)
        )
        if (script := _opening_script(text)) is not None:
            openings[script] += 1
    return TextEvidence(count_letters(texts), tuple(words), dict(openings))


def _mixes_scripts(text: str) -> bool:
    """Whether ``text`` holds letters of a script without function words
    beside its words."""
    return any(ch.isalpha() for ch in _WORD.sub("", text))


def _is_code(word: str) -> bool:
    """Whether ``word`` is an acronym or a code: a capital after its first
    letter ("AI", "OS", "iPhone", "OpenAI")."""
    return any(ch.isupper() for ch in word[1:])


def _joined(text: str, word: re.Match[str]) -> bool:
    """Whether ``word`` is a part of a compound: a letter of another script
    or a digit is written onto it ("ES6", "AIとは", "Acme의"), or a hyphen
    or the punctuation of a URL, an email, a path or a dotted name
    (``_JOINERS``) joins it to one ("to-do", "us-east-1", "example.com",
    "admin@example.es", "/usr/bin/the", "my_var")."""

    def joins(i: int, step: int) -> bool:
        if not 0 <= i < len(text):
            return False
        if text[i] == ":" and not _COLON_JOINS.match(text, i - (step < 0)):
            # A colon joins a scheme, a port, a time or "::" ("https://",
            # ":8080", "std::vector"), not a label ("Frage:Wer").
            return False
        # Over the whole joiner run: "::" and "://" join as ":" does.
        while text[i] in _JOINERS:
            i += step
            if not 0 <= i < len(text):
                return False
        return _LETTER_OR_DIGIT.match(text[i]) is not None

    return joins(word.start() - 1, -1) or joins(word.end(), 1)


def _opening_script(text: str) -> str | None:
    """The script of the capitalised word ``text`` opens on (a sentence
    starts there: "Who", "¿Quién"), or ``None``: one that opens on a word
    of another script, on an acronym ("AI") or on a lowercase word."""
    first = next((ch for ch in text if ch.isalpha()), None)
    if first is None or not first.isupper():
        return None
    match = _WORD.match(text, text.index(first))
    if match is None or len(match.group()) < 2 or _is_code(match.group()):
        return None
    script = script_of(first)
    return script.name if script is not None else None


def _name_words(text: str, words: list[re.Match[str]]) -> set[int]:
    """The indexes of ``words`` (in order) inside a name: a run of two or
    more of them with spaces alone between, opening and closing on a
    capitalised word or a code, with the lowercase words inside it ("Bank
    of America", "Procter and Gamble", "The Walt Disney Company", "iPhone
    and iPad"); a capitalised word alone (a question's first word, a German
    noun) is no name, and nor does a function word open one
    (:func:`_opens_name`: "Why did Orion Labs drop" holds the name "Orion
    Labs")."""
    names: set[int] = set()
    run: list[int] = []

    def close() -> None:
        while run and not _is_name_word(words[run[-1]].group()):
            run.pop()
        if len(run) > 1:
            names.update(run)
        run.clear()

    for i, word in enumerate(words):
        joined = bool(run) and _adjacent(text, words[run[-1]], word)
        if not joined:
            close()
            if not _opens_name(text, words, i):
                continue
        run.append(i)
    close()
    return names


def _is_name_word(word: str) -> bool:
    """Whether ``word`` is written as a name's: capitalised, or a code."""
    return word[0].isupper() or _is_code(word)


def _adjacent(text: str, word: re.Match[str], following: re.Match[str]) -> bool:
    """Whether only spaces stand between ``word`` and ``following``."""
    return not text[word.end() : following.start()].strip()


def _opens_name(text: str, words: list[re.Match[str]], i: int) -> bool:
    """Whether ``words[i]`` may open a name: a capitalised word or a code
    that is no function word nor elided one ("Qu'a"), or an article (in
    either case) directly before such a word."""
    word = words[i].group()
    if word.lower() in _ARTICLES and i + 1 < len(words):
        following = words[i + 1]
        if _is_name_word(following.group()) and _adjacent(text, words[i], following):
            return True
    lowered = word.lower()
    head, apostrophe, _ = lowered.partition("'")
    return (
        _is_name_word(word)
        and lowered not in _FUNCTION_WORDS
        and not (apostrophe and head in _ELIDED)
    )


def _signals(
    word: str, language: LanguageProfile, shared: frozenset[str] = frozenset()
) -> bool:
    """Whether ``word`` is one of ``language``'s function words, or opens
    with a word it elides that no other candidate elides (``shared``:
    "dell'accordo" is Italian, "d'unités" either) or closes on a function
    word ("d'où", "c'est", "dov'è")."""
    if word in language.function_words:
        return True
    head, apostrophe, tail = word.partition("'")
    if not apostrophe:
        return False
    return (
        head in language.elided_prefixes and head not in shared
    ) or tail in language.function_words


def _shared_elisions(candidates: list[LanguageProfile]) -> frozenset[str]:
    """The elided words more than one of ``candidates`` writes ("l'", "d'",
    "c'" in French and Italian): a sign of neither."""
    seen: set[str] = set()
    shared: set[str] = set()
    for language in candidates:
        shared |= seen & language.elided_prefixes
        seen |= language.elided_prefixes
    return frozenset(shared)


def _candidates(script: str | None) -> list[LanguageProfile]:
    """The languages a text in ``script`` may be in: those written in it,
    else the Latin-script ones (a script no profile writes)."""
    written = [p for p in PROFILES.values() if p.script == script]
    return written or [p for p in PROFILES.values() if p.script == "latin"]


def script_language(
    evidence: TextEvidence, fallback: LanguageCode | None = None
) -> LanguageCode | None:
    """The language ``evidence`` reads as (module docstring), or ``None``
    when it does not tell: the one rule a query (:func:`text_language`)
    and a corpus sample (``domain.languages.corpus``) are judged by. The
    script whose claim wins, and the language of its letters
    (:func:`language_in_script`)."""
    counts, words = evidence.counts, evidence.words
    claims: dict[str, int] = {}
    for name, written in _WORDED.items():
        shared = _shared_elisions(written)
        if found := sum(any(_signals(w, p, shared) for p in written) for w in words):
            claims[name] = found + evidence.openings.get(name, 0)
    # The heaviest script without function words, and its runs.
    script = counts.dominant(without=set(_WORDED))
    if claims:
        best = max(claims, key=lambda name: claims[name])
        if script is None or claims[best] >= (counts.runs or {}).get(script, 0):
            script = best
    return None if script is None else language_in_script(evidence, script, fallback)


def language_in_script(
    evidence: TextEvidence, script: str, fallback: LanguageCode | None = None
) -> LanguageCode | None:
    """The language ``evidence``'s letters in ``script`` (a group
    ``ScriptCounts`` counts: a script's name, or ``"dense"``) read as, or
    ``None``: Korean, Japanese or Chinese by :func:`dense_language`, the
    language one profile writes the script in, and among the languages of
    one script (Latin) the one whose function words and owned elisions
    ("dell'" is Italian, "qu'" French, "d'" neither) occur most,
    ``fallback`` (the index language) among those tied, else the one of
    them whose own articles occur most, else none."""
    if script == "dense":
        return dense_language(evidence.counts, fallback)
    if script in _WORDED:
        written = _WORDED[script]
        shared = _shared_elisions(written)
        hits = {
            p.code: sum(_signals(word, p, shared) for word in evidence.words)
            for p in written
        }
        return _by_words(written, hits, list(evidence.words), fallback)
    written = [p for p in PROFILES.values() if p.script == script]
    return written[0].code if len(written) == 1 else None


def _by_words(
    written: list[LanguageProfile],
    hits: Mapping[LanguageCode, int],
    words: list[str],
    fallback: LanguageCode | None,
) -> LanguageCode | None:
    """Of the languages ``written`` in one script, the one with the most
    ``hits``, ``fallback`` among those tied, else the one whose own
    articles occur most in ``words``, else none."""
    best = max(hits.values())
    tied = [p for p in written if hits[p.code] == best]
    if fallback in {p.code for p in tied}:
        return fallback
    if len(tied) > 1:
        tied = _by_articles(tied, words)
    return tied[0].code if len(tied) == 1 else None


def _by_articles(
    tied: list[LanguageProfile], words: list[str]
) -> list[LanguageProfile]:
    """Of languages tied on their function words, those whose articles no
    other of them writes occur most in ``words`` ("il" is Italian, not
    Portuguese); all of them when none occurs."""
    counts: dict[LanguageCode, int] = {}
    for language in tied:
        own = language.articles - frozenset().union(
            *(other.articles for other in tied if other is not language)
        )
        counts[language.code] = sum(word in own for word in words)
    best = max(counts.values())
    return [p for p in tied if counts[p.code] == best] if best else tied


def decided_language(text: str, fallback: LanguageCode | str) -> LanguageCode | None:
    """The language ``text`` (a query) is written in when its script and
    function words tell (:func:`script_language`), else ``None``: a name
    or keyword query ("Acme Corp", "GPT-4o") is in no language of its own.
    :func:`text_language` resolves the ``None`` to a language."""
    return script_language(text_evidence([text]), _code(fallback))


def text_language(text: str, fallback: LanguageCode | str) -> LanguageCode:
    """The language ``text`` (a query) is written in, as far as its script
    and function words tell (:func:`script_language`: a Korean question
    quoting an English name is Korean, an English one quoting a Korean name
    English); ``fallback`` (the index language) when they do not and it is
    written in the text's script (a Latin one for a Latin-script text),
    else English."""
    fallback = _code(fallback)
    evidence = text_evidence([text])
    found = script_language(evidence, fallback)
    if found is not None:
        return found
    script = evidence.counts.dominant()
    if script == "dense":
        return fallback
    return (
        fallback
        if fallback in {p.code for p in _candidates(script)}
        else LanguageCode.EN
    )


def _code(code: LanguageCode | str) -> LanguageCode:
    try:
        return LanguageCode(code)
    except ValueError:
        return LanguageCode.EN
