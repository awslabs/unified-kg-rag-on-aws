# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RDF vocabulary IRIs and the pure helpers that name RDF terms, shared by
every reader of RDF terms (the ``owl`` schema source and the ``rdf`` seed
source): an IRI's namespace and local name, a CURIE of the options as an
IRI, a label in ``UPPER_SNAKE`` or as written, the preferred text of a term
in a language order, and the value types of their options.

Nothing here depends on an RDF library.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Annotated
from urllib.parse import unquote

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
SKOS = "http://www.w3.org/2004/02/skos/core#"
SCHEMA = "https://schema.org/"
DCTERMS = "http://purl.org/dc/terms/"

# The prefixes a CURIE of the options may use without binding them.
WELL_KNOWN_PREFIXES: dict[str, str] = {
    "dcterms": DCTERMS,
    "owl": OWL,
    "rdf": RDF,
    "rdfs": RDFS,
    "schema": SCHEMA,
    "skos": SKOS,
    "xsd": XSD,
}

# The longest label a type may get (a longer one needs a ``labels`` entry).
MAX_LABEL_CHARS = 64

# The value types of the mapping options: a term (IRI or CURIE), a prefix
# name, a language tag (lower-cased; "" = untagged), a label override.
OptionTerm = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PrefixName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][\w.-]*$")]
LanguageTag = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, pattern=r"^(|[A-Za-z]{1,8}(-[A-Za-z0-9]{1,8})*)$"
    ),
    AfterValidator(str.lower),
]
LabelOverride = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_LABEL_CHARS),
]


class PrefixConflictError(ValueError):
    """A CURIE's prefix the well-known prefixes and the files bind to
    different namespaces, which the options do not settle.

    Attributes:
        prefix: The prefix.
        namespaces: Its namespaces, sorted.
    """

    def __init__(self, prefix: str, namespaces: Sequence[str]) -> None:
        self.prefix = prefix
        self.namespaces = tuple(sorted(namespaces))
        super().__init__(
            f"prefix '{prefix}' is bound to {' and '.join(self.namespaces)}"
        )


def resolve_curie(
    term: str, prefixes: Mapping[str, str], file_prefixes: Mapping[str, str]
) -> str | None:
    """``term`` as an IRI: as written when it is one (``://`` or ``urn:``),
    else the CURIE ``prefix:name`` with its prefix's namespace, which
    ``prefixes`` (the options') binds, else the well-known prefixes and
    ``file_prefixes`` (the files' own bindings) bind alike. None when no
    one binds the prefix (or ``term`` has none).

    Raises:
        PrefixConflictError: The well-known prefixes and the files bind the
            prefix to different namespaces and ``prefixes`` does not bind
            it: guessing would read another vocabulary's terms.
    """
    if "://" in term or term.startswith("urn:"):
        return term
    prefix, separator, name = term.partition(":")
    if not separator:
        return None
    if prefix in prefixes:
        return prefixes[prefix] + name
    bound = {
        namespace
        for namespace in (WELL_KNOWN_PREFIXES.get(prefix), file_prefixes.get(prefix))
        if namespace is not None
    }
    if len(bound) > 1:
        raise PrefixConflictError(prefix, sorted(bound))
    return bound.pop() + name if bound else None


class Text(BaseModel):
    """A literal annotation value: ``language`` is its tag, ``""`` untagged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str
    language: str = ""


def local_name(iri: str) -> str:
    """The part of ``iri`` after its last ``#``, else after its last ``/``,
    else after its last ``:``, percent-decoded."""
    return unquote(iri[len(namespace_of(iri)) :])


def namespace_of(iri: str) -> str:
    """``iri`` up to and including the separator before its local name."""
    for separator in ("#", "/", ":"):
        index = iri.rfind(separator)
        if index != -1 and index < len(iri) - 1:
            return iri[: index + 1]
    return ""


def _is_upper(char: str) -> bool:
    return unicodedata.category(char) in ("Lu", "Lt")


def _is_lower(char: str) -> bool:
    return unicodedata.category(char) == "Ll"


def _kept(char: str) -> bool:
    # Letters, digits and combining marks (Hangul, CJK and accents included).
    return unicodedata.category(char)[0] in "LNM"


_UNDERSCORES = re.compile(r"_+")


def upper_snake(text: str) -> str:
    """``text`` as an ``UPPER_SNAKE`` label.

    NFKC-normalized; a word break before an upper-case letter that follows
    a lower-case one (``supplyChain``) or that starts a word after an
    acronym or a number (``HTTPServer``, ``Level2Item``; not ``B2B``); every
    character but letters, digits and marks a break; breaks written ``_``,
    collapsed and trimmed; upper-cased.
    Scripts without case (Hangul, CJK) stay as they are: "공급 업체" →
    ``공급_업체``.
    """
    text = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    for index, char in enumerate(text):
        if not _kept(char):
            out.append("_")
            continue
        if index and _is_upper(char):
            previous = text[index - 1]
            following = text[index + 1] if index + 1 < len(text) else ""
            starts_word = bool(following) and _is_lower(following)
            if _is_lower(previous) or (
                starts_word and (_is_upper(previous) or previous.isdigit())
            ):
                out.append("_")
        out.append(char)
    return _UNDERSCORES.sub("_", "".join(out)).strip("_").upper()


def preserve_case(text: str) -> str:
    """``text`` NFKC-normalized, every character but letters, digits,
    marks, ``_`` and ``-`` a space, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text)
    kept = "".join(char if _kept(char) or char in "_-" else " " for char in text)
    return " ".join(kept.split())


def _language_rank(tag: str, languages: Sequence[str]) -> tuple[int, int, int, str]:
    tag = tag.lower()
    for index, language in enumerate(languages):
        if tag == language:
            return (0, index, 0, "")
        if language and tag.startswith(language + "-"):
            return (0, index, 1, "")
    if not tag:
        return (1, 0, 0, "")
    return (2, 0, 0, tag)


def best_text(
    values: Mapping[str, Sequence[Text]],
    properties: Sequence[str],
    languages: Sequence[str],
) -> str | None:
    """The preferred non-blank value of ``properties`` in ``values``:
    language first (``languages`` in order, an exact tag before its
    subtags, then untagged, then other tags alphabetically), then the order
    of ``properties``, then the smallest NFKC string."""
    ranked = [
        (
            _language_rank(text.language, languages),
            position,
            unicodedata.normalize("NFKC", text.value),
            text.value,
        )
        for position, prop in enumerate(properties)
        for text in values.get(prop, ())
        if text.value.strip()
    ]
    return min(ranked)[3] if ranked else None
