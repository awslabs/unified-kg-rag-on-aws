# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legal-form designators of organization names ("Inc", "GmbH", "株式会社", "(주)").

A designator states the form of a company, not which company it is, so
entity resolution ignores it when it compares names ("Acme GmbH" and "Acme"
name one company) and name grounding reads two spellings of one form as the
same word ("Acme Corporation" in a text grounds "Acme Corp"). Both read the
one table here.

Spelling variants are folded rather than listed: text is NFKC-normalized and
casefolded first ("㈜" is "(주)", "Ｉｎｃ" is "inc"), a Latin designator is
compared without its dots ("L.L.C." is "llc", "S.A." is "sa") and a
parenthesized one with any spaces inside the parentheses ("( 주 )").

To avoid merging different names, a designator is stripped only

- at the end of a name for a Latin form, as a whole word after another word
  that is neither "&" / "and" ("Smith & Co" is a firm's name, not "Smith")
  nor only an article ("The Company" is a defined party, not "The");
- at the start or the end for a Han, Hangul or Kana form, with or without a
  space, as written in those scripts;
- never down to nothing: a name that is only designators keeps the first
  ("Inc." stays "inc.", "Pty Ltd" becomes "pty").

``S.p.A.`` counts only with its dots, since "spa" is a word.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["latin_legal_form", "strip_legal_forms"]

# Latin designators, dot-free and casefolded, each with the canonical short
# form its spellings share.
_LATIN_LEGAL_FORMS: dict[str, str] = {
    "inc": "inc",
    "incorporated": "inc",
    "corp": "corp",
    "corporation": "corp",
    "co": "co",
    "company": "co",
    "ltd": "ltd",
    "limited": "ltd",
    "llc": "llc",
    "llp": "llp",
    "lp": "lp",
    "plc": "plc",
    "pty": "pty",
    "pte": "pte",
    "gmbh": "gmbh",
    "ag": "ag",
    "sa": "sa",
    "sas": "sas",
    "sarl": "sarl",
    "srl": "srl",
    "spa": "spa",
    "ltda": "ltda",
    "bv": "bv",
    "nv": "nv",
    "kk": "kk",
}
# Designators that are also words: a designator only when spelled with dots.
_DOTTED_ONLY = frozenset({"spa"})
# What a Latin designator may not follow alone or directly.
_ARTICLES = frozenset({"the", "a", "an"})
_CONJUNCTIONS = frozenset({"&", "and"})

# Han, Hangul and Kana designators (NFKC-casefolded), leading or trailing.
_DENSE_LEGAL_FORMS = (
    # Korean: stock, limited, limited liability, limited partnership and
    # general partnership companies; incorporated associations, foundations.
    "(주)",
    "주식회사",
    "(유)",
    "유한회사",
    "유한책임회사",
    "합자회사",
    "합명회사",
    "(사)",
    "사단법인",
    "(재)",
    "재단법인",
    # Japanese (with the traditional 會): kabushiki, yugen, godo, goshi and
    # gomei kaisha.
    "(株)",
    "株式会社",
    "株式會社",
    "(有)",
    "有限会社",
    "合同会社",
    "合資会社",
    "合名会社",
    # Chinese (simplified and traditional): company limited by shares,
    # limited liability company, limited company.
    "股份有限公司",
    "有限责任公司",
    "有限責任公司",
    "有限公司",
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
    bare = token.replace(".", "")
    if bare in _DOTTED_ONLY and "." not in token:
        return None
    return _LATIN_LEGAL_FORMS.get(bare)


def _strip_trailing_latin(text: str) -> str:
    head, _, last = text.rpartition(" ")
    if not head or latin_legal_form(last.rstrip(",")) is None:
        return text
    words = head.rstrip(" ,").split()
    if not words or words[-1] in _CONJUNCTIONS or set(words) <= _ARTICLES:
        return text
    return " ".join(words)


def _strip_leading_dense(text: str) -> str:
    return _RE_LEADING_DENSE.sub("", text)


def _strip_trailing_dense(text: str) -> str:
    return _RE_TRAILING_DENSE.sub("", text)


_STEPS = (_strip_leading_dense, _strip_trailing_dense, _strip_trailing_latin)


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
