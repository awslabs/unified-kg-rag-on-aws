# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legal-form designators of organization names ("Inc", "GmbH", "株式会社", "(주)").

The designators are the language profiles' (``domain.languages``).
A designator states the form of a company, not which company it is, so
entity resolution ignores it when it compares names ("Acme GmbH" and "Acme"
name one company) and name grounding reads two spellings of one form as the
same word ("Acme Corporation" in a text grounds "Acme Corp"). Both read the
one table here.

Spelling variants are folded rather than listed: text is NFKC-normalized and
casefolded first ("㈜" is "(주)", "Ｉｎｃ" is "inc"), a Latin designator is
compared without its dots ("L.L.C." is "llc", "S.A." is "sa") and its
Latin accents ("Cong ty" is "Công ty"), and a
parenthesized one with any spaces inside the parentheses ("( 주 )").

To avoid merging different names, a designator is stripped only

- at the end of a name for a form written as words, as whole words after
  another word that is neither "&" / "and" ("Smith & Co" is a firm's name,
  not "Smith"), nor only an article ("The Company" is a defined party, not
  "The"), nor a number ("500 kg" is a weight); a designator may span words
  ("GmbH & Co. KG", "S.A. de C.V.", "Private Limited");
- at the start of a name for a form its language writes there ("ООО
  Ромашка", "Công ty TNHH Acme"), before another word;
- at the start or the end for a Han, Hangul or Kana form, with or without a
  space, as written in those scripts;
- never down to nothing: a name that is only designators keeps the first
  ("Inc." stays "inc.", "Pty Ltd" becomes "pty").

``S.p.A.``, ``e.V.`` and ``S.L.`` count only with their dots, since "spa"
and "ev" are words and "SL" a model name.
"""

from __future__ import annotations

import re
import unicodedata

from unified_kg_rag.domain.languages.registry import profiles
from unified_kg_rag.shared.utils.scripts import without_latin_accents

__all__ = ["latin_legal_form", "strip_legal_forms"]


def _folded(form: str) -> str:
    """``form`` as designators are compared: NFKC-casefolded (Thai "ำ" is
    "ํา"), without its Latin accents ("Công ty" is "cong ty", as text
    without accents writes it)."""
    return without_latin_accents(
        unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", form).casefold())
    )


# Designators written as words after the name, dot-free and casefolded, each
# with the canonical short form its spellings share, from every language's
# profile (``LanguageProfile.legal_forms``).
_LATIN_LEGAL_FORMS: dict[str, str] = {
    _folded(form): short
    for language in profiles()
    for form, short in language.legal_forms.items()
}
# Those written before the name (``LanguageProfile.leading_legal_forms``).
_LEADING_LEGAL_FORMS: dict[str, str] = {
    _folded(form): short
    for language in profiles()
    for form, short in language.leading_legal_forms.items()
}
# The most words one designator spans ("gmbh & co kg").
_MAX_WORDS = max(len(form.split()) for form in _LATIN_LEGAL_FORMS)
_MAX_LEADING_WORDS = max(
    (len(form.split()) for form in _LEADING_LEGAL_FORMS), default=0
)
# Designators that are also words or model names ("spa", "ev", Mercedes-Benz
# "SL"): a designator only when spelled with dots.
_DOTTED_ONLY = frozenset({"spa", "ev", "sl"})
# A number: what a unit, not a designator, follows ("500 kg").
_NUMBER = re.compile(r"[\d.,]+")
# What a Latin designator may not follow alone or directly.
_ARTICLES = frozenset({"the", "a", "an"})
_CONJUNCTIONS = frozenset({"&", "and"})

# Han, Hangul and Kana designators (NFKC-casefolded), leading or trailing
# (``LanguageProfile.dense_legal_forms``).
_DENSE_LEGAL_FORMS = tuple(
    dict.fromkeys(
        _folded(form) for language in profiles() for form in language.dense_legal_forms
    )
)


def _dense_form_pattern(form: str) -> str:
    if form.startswith("(") and form.endswith(")"):
        return rf"\(\s*{re.escape(form[1:-1])}\s*\)"
    return re.escape(form)


# Longest first, so "股份有限公司" is one designator, not "股份" + "有限公司".
_DENSE_ALTERNATIVES = "|".join(
    _dense_form_pattern(form)
    for form in sorted(_DENSE_LEGAL_FORMS, key=len, reverse=True)
)
_RE_LEADING_DENSE = re.compile(rf"^(?:{_DENSE_ALTERNATIVES})\s*")
_RE_TRAILING_DENSE = re.compile(rf"\s*(?:{_DENSE_ALTERNATIVES})$")


def latin_legal_form(token: str) -> str | None:
    """The canonical short form of a Latin designator ``token``, else ``None``.

    ``token`` is one casefolded word; its dots are ignored ("l.l.c." is
    "llc"), and spellings of one form share their short form ("corporation"
    and "corp" are "corp").
    """
    bare = without_latin_accents(token.replace(".", ""))
    if bare in _DOTTED_ONLY and "." not in token:
        return None
    return _LATIN_LEGAL_FORMS.get(bare)


def _form_key(words: list[str]) -> str:
    """Words as a designator key: dot-free, without trailing commas and
    Latin accents."""
    return without_latin_accents(
        " ".join(word.rstrip(",").replace(".", "") for word in words)
    )


def _strip_trailing_latin(text: str) -> str:
    words = text.split()
    for n in range(min(_MAX_WORDS, len(words) - 1), 0, -1):
        tail = words[-n:]
        if n == 1:
            found = latin_legal_form(tail[0].rstrip(",")) is not None
        else:
            found = _form_key(tail) in _LATIN_LEGAL_FORMS
        if found:
            break
    else:
        return text
    head = " ".join(words[:-n]).rstrip(" ,").split()
    if (
        not head
        or head[-1] in _CONJUNCTIONS
        or set(head) <= _ARTICLES
        or _NUMBER.fullmatch(head[-1])
    ):
        return text
    return " ".join(head)


def _strip_leading_words(text: str) -> str:
    words = text.split()
    for n in range(min(_MAX_LEADING_WORDS, len(words) - 1), 0, -1):
        if _form_key(words[:n]) in _LEADING_LEGAL_FORMS:
            return " ".join(words[n:]).lstrip(" ,")
    return text


def _strip_leading_dense(text: str) -> str:
    return _RE_LEADING_DENSE.sub("", text)


def _strip_trailing_dense(text: str) -> str:
    return _RE_TRAILING_DENSE.sub("", text)


_STEPS = (
    _strip_leading_dense,
    _strip_trailing_dense,
    _strip_leading_words,
    _strip_trailing_latin,
)


def strip_legal_forms(name: str) -> str:
    """``name`` NFKC-casefolded, without its legal-form designators.

    Designators are removed one at a time until none is left ("Acme Pty
    Ltd", "Acme Co., Ltd." and "(주)가나다" give "acme", "acme" and "가나다"),
    except one whose removal would leave nothing: "주식회사" and "Inc." stay.
    Stripping a result again changes nothing.
    """
    # Normalized again after folding, which can leave a decomposed sequence
    # ("ß" + U+0301 folds to "ss" + U+0301), so a result is its own result.
    folded = unicodedata.normalize(
        "NFKC", unicodedata.normalize("NFKC", name).casefold()
    )
    stripped = " ".join(folded.split())
    changed = True
    while changed:
        changed = False
        for step in _STEPS:
            shorter = step(stripped).strip()
            if shorter and shorter != stripped:
                stripped, changed = shorter, True
    return stripped
