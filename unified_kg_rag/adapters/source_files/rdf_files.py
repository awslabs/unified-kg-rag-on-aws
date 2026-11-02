# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which RDF files a source's path holds, and the checks their bytes pass
before a parser sees them. Needs no rdflib.

:func:`locate` (a local file or directory) and :func:`locate_s3` (an S3
object or prefix) pick the files and the format each is read as: by suffix
(:data:`SUFFIX_FORMATS`), or the one format the source names for every file.
Notation3, OWL/XML, OWL functional and Manchester syntax and OBO are refused
(:data:`REFUSED_SUFFIXES`), with how to convert them. :func:`check_bytes`
refuses what a parser must not see: XML whose entities expand past the XML
parser's amplification limit (external entities are not loaded), OWL/XML,
an XML element with more than ``MAX_XML_TEXT_CHUNKS`` pieces of text (a run
of line breaks or entity references), more than ``MAX_XML_LITERAL_ELEMENTS``
elements inside ``rdf:parseType="Literal"`` values, an N-Triples or N-Quads
line longer than ``MAX_LINE_BYTES`` (rdflib's parsers take quadratic time
over these), and a JSON-LD ``@context`` or ``@import`` naming a document
(the parser would fetch it).

Every error is the caller's :class:`.locations.SourceTerms`: its exception,
and how it names the source, its options and its kind.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal
from xml.parsers import expat

from .locations import SourceFile, SourceReader, SourceTerms, redact_location, split_s3

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_OWL = "http://www.w3.org/2002/07/owl#"

RdfFormat = Literal["auto", "turtle", "xml", "json-ld", "nt", "nquads", "trig"]
# Suffix -> the format it is read as.
SUFFIX_FORMATS: dict[str, str] = {
    ".ttl": "turtle",
    ".nt": "nt",
    ".nq": "nquads",
    ".trig": "trig",
    ".rdf": "xml",
    ".owl": "xml",
    ".xml": "xml",
    ".jsonld": "json-ld",
    ".json-ld": "json-ld",
}
# Refused before parsing: the parsers take quadratic time beyond these.
# rdflib's RDF/XML parser appends each piece of an element's text to a
# string (one piece per line break or entity reference) and re-parses an
# rdf:parseType="Literal" value at every element in it; its N-Triples parser
# copies the rest of a line at every escape.
MAX_XML_TEXT_CHUNKS = 10_000
MAX_XML_LITERAL_ELEMENTS = 1_000
MAX_LINE_BYTES = 256 * 1024
_RDF_PARSE_TYPE = _RDF + " parseType"
_CATALOG_ROOT = "urn:oasis:names:tc:entity:xmlns:xml:catalog catalog"
_CONVERT = (
    "convert it to Turtle or RDF/XML first (ROBOT: robot convert --input "
    "FILE --output FILE.ttl; or Protégé: File > Save as > Turtle)"
)
# Suffix -> what it holds, for the refusal.
REFUSED_SUFFIXES: dict[str, str] = {
    ".n3": "Notation3 (its rules and formulae are not RDF)",
    ".owx": "OWL/XML",
    ".ofn": "OWL functional syntax",
    ".omn": "OWL Manchester syntax",
    ".obo": "OBO",
}


RdfFiles = list[tuple[SourceFile, str]]


def _refuse_suffix(name: str, suffix: str, terms: SourceTerms) -> None:
    what = REFUSED_SUFFIXES.get(suffix)
    if what is not None:
        raise terms(
            f"{terms.noun} file '{name}' is {what}, which kind '{terms.kind}' "
            f"does not read; {_CONVERT}"
        )


def locate(path: Path, *, format_: str, max_bytes: int, terms: SourceTerms) -> RdfFiles:
    """The files at the local ``path`` and the format each is read as:
    ``path`` itself, or the RDF files under the directory (by suffix, or
    every file when ``format_`` is not ``auto``; sorted; names starting with
    ``.`` skipped).

    Raises:
        terms.error: ``path`` is missing, is or holds a refused format, has
            an unknown suffix (with ``format_`` ``auto``), holds no RDF
            file, or the files hold more than ``max_bytes``.
    """
    if path.is_dir():
        return _directory(
            str(path),
            [
                (
                    candidate.relative_to(path).as_posix(),
                    SourceFile(location=str(candidate), size=candidate.stat().st_size),
                )
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file()
            ],
            format_,
            max_bytes,
            terms,
        )
    if path.is_file():
        return _single(
            SourceFile(location=str(path), size=path.stat().st_size),
            format_,
            max_bytes,
            terms,
        )
    raise terms(f"{terms.noun} '{path}' not found")


def locate_s3(
    reader: SourceReader,
    location: str,
    *,
    format_: str,
    max_bytes: int,
    terms: SourceTerms,
) -> RdfFiles:
    """:func:`locate` of an S3 location: the object ``location`` names
    (sized with HeadObject), or, for a prefix (``s3://bucket/prefix/``,
    or the bucket root), the RDF objects under it (every page of the
    listing), filtered, sorted and refused as a local directory is.

    Raises:
        terms.error: What :func:`locate` raises.
        reader.terms.error: The S3 errors of
            :class:`.locations.SourceReader` (access denied, ...).
    """
    _, key = split_s3(location, terms=reader.terms)
    if not key or key.endswith("/"):
        return _directory(
            redact_location(location),
            reader.list_prefix(location),
            format_,
            max_bytes,
            terms,
        )
    return _single(reader.head(location, prefix_hint=True), format_, max_bytes, terms)


def _single(
    file: SourceFile, format_: str, max_bytes: int, terms: SourceTerms
) -> RdfFiles:
    _refuse_suffix(file.name, file.suffix, terms)
    if file.suffix not in SUFFIX_FORMATS and format_ == "auto":
        raise terms(
            f"{terms.noun} file '{file.name}' must end in "
            f"{', '.join(SUFFIX_FORMATS)}, or set {terms.options}.format"
        )
    return _formats(file.name, [file], format_, max_bytes, terms)


def _directory(
    name: str,
    candidates: list[tuple[str, SourceFile]],
    format_: str,
    max_bytes: int,
    terms: SourceTerms,
) -> RdfFiles:
    """The RDF files of ``candidates`` (each with its path relative to the
    directory, sorted): those with an RDF suffix (any, when ``format_`` is
    not ``auto``), outside any name starting with ``.``."""
    found = []
    for relative, candidate in candidates:
        if any(part.startswith(".") for part in relative.split("/")):
            continue
        _refuse_suffix(candidate.name, candidate.suffix, terms)
        if format_ != "auto" or candidate.suffix in SUFFIX_FORMATS:
            found.append(candidate)
    if not found:
        raise terms(
            f"{terms.noun} directory '{name}' holds no {terms.noun.lower()} "
            f"file ({', '.join(SUFFIX_FORMATS)})"
        )
    return _formats(name, found, format_, max_bytes, terms)


def _formats(
    name: str,
    found: list[SourceFile],
    format_: str,
    max_bytes: int,
    terms: SourceTerms,
) -> RdfFiles:
    size = sum(candidate.size for candidate in found)
    if size > max_bytes:
        raise terms(
            f"{terms.noun} '{name}' holds {size} bytes, more than "
            f"{terms.options}.max_bytes ({max_bytes}); point the path at the "
            f"files {terms.purpose} needs, or raise max_bytes"
        )
    return [
        (
            candidate,
            SUFFIX_FORMATS[candidate.suffix] if format_ == "auto" else format_,
        )
        for candidate in found
    ]


def read_local(path: Path, *, terms: SourceTerms) -> bytes:
    """The bytes of the local file ``path``.

    Raises:
        terms.error: The file cannot be read.
    """
    try:
        return path.read_bytes()
    except OSError as e:
        raise terms(
            f"Cannot read {terms.noun.lower()} file '{path}': {e.strerror or e}"
        ) from e


# -- Safety checks before parsing --------------------------------------------


class _Root(Exception):
    """Stops the XML scan at the root element."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def is_xml_catalog(data: bytes) -> bool:
    """True when ``data`` is an OASIS XML catalog (Protégé's
    ``catalog-v001.xml``), which a directory of RDF files may hold."""
    return _xml_root(data) == _CATALOG_ROOT


def _xml_root(data: bytes) -> str | None:
    """The root element of ``data`` as ``namespace local-name``, or None
    when it has none (or is no XML). External entities are not loaded (no
    handler)."""
    parser = expat.ParserCreate(namespace_separator=" ")

    def start(name: str, attributes: Any) -> None:
        raise _Root(name)

    parser.StartElementHandler = start
    try:
        parser.Parse(data, True)
    except _Root as root:
        return root.name
    except expat.ExpatError:
        pass
    return None


class _TooCostly(Exception):
    """Stops the XML scan at a shape the RDF/XML parser is quadratic on."""


class _XmlScan:
    """One pass of the XML parser over the whole of ``data``: its entities
    expanded (within the parser's amplification limit; external ones are
    not loaded), the root element, and the shapes the RDF/XML parser takes
    quadratic time over counted."""

    def __init__(self) -> None:
        self.root: str | None = None
        # The text pieces of each open element, innermost last.
        self.chunks: list[int] = []
        # The depth of the open rdf:parseType="Literal" element, if any.
        self.literal_depth: int | None = None
        self.literal_elements = 0

    def start(self, name: str, attributes: dict[str, str]) -> None:
        if self.root is None:
            self.root = name
        if self.literal_depth is not None:
            self.literal_elements += 1
            if self.literal_elements > MAX_XML_LITERAL_ELEMENTS:
                raise _TooCostly(
                    f"more than {MAX_XML_LITERAL_ELEMENTS} elements inside "
                    'rdf:parseType="Literal" values (MAX_XML_LITERAL_ELEMENTS)'
                )
        elif attributes.get(_RDF_PARSE_TYPE) == "Literal":
            self.literal_depth = len(self.chunks)
        self.chunks.append(0)

    def end(self, name: str) -> None:
        self.chunks.pop()
        if self.literal_depth == len(self.chunks):
            self.literal_depth = None

    def text(self, data: str) -> None:
        if self.chunks:
            self.chunks[-1] += 1
            if self.chunks[-1] > MAX_XML_TEXT_CHUNKS:
                raise _TooCostly(
                    f"an element with more than {MAX_XML_TEXT_CHUNKS} pieces "
                    "of text (line breaks or entity references; "
                    "MAX_XML_TEXT_CHUNKS)"
                )


def _check_xml(path: str, data: bytes, terms: SourceTerms) -> None:
    """Refuse OWL/XML, XML that does not parse (its entities expanding past
    the amplification limit, say) and the shapes :class:`_XmlScan` counts,
    before the RDF/XML parser sees the bytes."""
    scan = _XmlScan()
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.StartElementHandler = scan.start
    parser.EndElementHandler = scan.end
    parser.CharacterDataHandler = scan.text
    try:
        parser.Parse(data, True)
    except _TooCostly as e:
        raise terms(
            f"{terms.noun} file '{path}' holds {e}, which the RDF/XML parser "
            "takes quadratic time over; convert the file to Turtle"
        ) from e
    except expat.ExpatError as e:
        # The message has a position, never the file's text.
        raise terms(f"{terms.noun} file '{path}' does not parse as xml: {e}") from e
    if scan.root == _OWL + " Ontology":
        raise terms(
            f"{terms.noun} file '{path}' is OWL/XML (its root is "
            f"owl:Ontology), which kind '{terms.kind}' does not read; {_CONVERT}"
        )


def _check_lines(path: str, data: bytes, terms: SourceTerms) -> None:
    """Refuse an N-Triples or N-Quads line longer than ``MAX_LINE_BYTES``."""
    longest = max(map(len, data.splitlines()), default=0)
    if longest > MAX_LINE_BYTES:
        raise terms(
            f"{terms.noun} file '{path}' has a line of {longest} bytes, more than "
            f"the {MAX_LINE_BYTES} (MAX_LINE_BYTES) the N-Triples parser reads "
            "in reasonable time; convert the file to Turtle"
        )


def check_bytes(path: str, data: bytes, format_: str, *, terms: SourceTerms) -> None:
    """Refuse what the parser of ``format_`` must not see (the module
    docstring); ``path`` is the file's name as errors show it.

    Raises:
        terms.error: ``data`` is OWL/XML, XML that does not parse or past a
            limit, an N-Triples or N-Quads file with a line past
            ``MAX_LINE_BYTES``, or JSON-LD naming a remote context.
    """
    if format_ == "xml":
        _check_xml(path, data, terms)
    elif format_ in ("nt", "nquads"):
        _check_lines(path, data, terms)
    elif format_ == "json-ld":
        _check_json_ld(path, data, terms)


def _walk(value: Any) -> Iterator[tuple[str, Any]]:
    """Every ``(key, value)`` of the JSON objects in ``value``."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                yield key, child
                stack.append(child)
        elif isinstance(item, list):
            stack.extend(item)


def _check_json_ld(path: str, data: bytes, terms: SourceTerms) -> None:
    """Refuse a JSON-LD context that names a document: the parser would
    fetch (or open) it."""
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        raise terms(
            f"{terms.noun} file '{path}' does not parse as json-ld: "
            f"{' '.join(str(e).split())}"
        ) from e
    for key, value in _walk(document):
        contexts = value if isinstance(value, list) else [value]
        if (key == "@context" and any(isinstance(c, str) for c in contexts)) or (
            key == "@import"
        ):
            raise terms(
                f"{terms.noun} file '{path}' names a remote JSON-LD context "
                f"({key}), which would be fetched; inline the context, or "
                "convert the file to Turtle"
            )
