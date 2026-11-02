# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``schema_source.kind: owl``: an OWL or RDFS ontology as the schema.

``path`` names a local file, or a directory whose ontology files (by
suffix, recursively; every file when ``options.format`` is set) are read
together; or an S3 object
(``s3://bucket/onto.ttl``), or a prefix read as a directory
(``s3://bucket/ontology/``, the trailing ``/`` required)::

    schema_source:
      kind: owl
      path: ontology/procure.ttl
      options: {classes: ["ex:Organization"], languages: [ko, en, ""]}

The files are Turtle (``.ttl``), N-Triples (``.nt``), N-Quads (``.nq``),
TriG (``.trig``), RDF/XML (``.rdf``, ``.owl``, ``.xml``) or JSON-LD
(``.jsonld``, ``.json-ld``); ``options.format`` names one for every file
instead. The reader builds an :class:`.ontology_model.OntologyModel` of the
named classes and properties and :func:`.ontology_mapping.map_ontology`
maps it with the other options (:class:`OwlSourceOptions`).

Nothing else is fetched. The files are parsed from their bytes (never from a
location the parser would open); ``owl:imports`` is not followed (an import
no file of ``path`` declares is logged once: put that ontology in the
directory); a JSON-LD ``@context`` or ``@import`` naming a document is
refused, since the parser would fetch it; XML external entities are not
resolved. OWL/XML, functional and Manchester syntax, OBO and Notation3 are
refused, with how to convert them; an OASIS XML catalog (Protégé's
``catalog-v001.xml``) in a directory is skipped.

The parsers take quadratic time over some shapes, so these are refused
before parsing (:func:`check_bytes`): XML whose entities expand past the
XML parser's amplification limit, an XML element with more than
``MAX_XML_TEXT_CHUNKS`` pieces of text (a run of line breaks or entity
references), more than ``MAX_XML_LITERAL_ELEMENTS`` elements inside
``rdf:parseType="Literal"`` values, and an N-Triples or N-Quads line longer
than ``MAX_LINE_BYTES``.

Needs the ``ontology`` extra (rdflib); without it the loader, and the CLIs
before any model call, say how to install it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from urllib.request import url2pathname
from xml.parsers import expat

from pydantic import Field, ValidationError

from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.domain.models.schema import SchemaDocument
from unified_kg_rag.shared.exceptions import SchemaSourceError

from .locations import SourceFile, SourceReader, is_s3, split_s3
from .ontology_mapping import MappingOptions, map_ontology
from .ontology_model import OWL, RDF, OntologyModel
from .registry import SchemaSourceContext, redact_location, register_schema_source

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "the ontology extra: uv sync --extra ontology (pip install "
    "'unified-kg-rag-on-aws[ontology]')"
)

OntologyFormat = Literal["auto", "turtle", "xml", "json-ld", "nt", "nquads", "trig"]
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
_RDF_PARSE_TYPE = RDF + " parseType"
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


class OwlSourceOptions(MappingOptions):
    """``schema_source.options`` of kind ``owl``: the mapping options and
    how the files are read."""

    format: OntologyFormat = Field(
        default="auto",
        description=(
            "The serialisation of every file (auto: by suffix; set, every "
            "file of a directory is read, whatever its suffix)"
        ),
    )
    max_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1,
        description="Fail when the files hold more bytes together",
    )


@register_schema_source("owl", requires=("rdflib", INSTALL_HINT))
def load_owl_schema(context: SchemaSourceContext) -> SchemaDocument:
    """The schema the ontology at ``source.path`` maps to.

    Raises:
        SchemaSourceError: rdflib is not installed; no path is set, the
            path is a URL other than ``file://`` or ``s3://`` or
            ``secret_arn`` is set; an option is unknown or invalid; the path
            is missing (or access to it is denied), holds no ontology file,
            a refused format, a remote JSON-LD context or OWL/XML, or more
            than ``max_bytes``; a file does not parse; or the mapping fails
            (``map_ontology``).
    """
    source = context.source
    path = _path(source, context.location)
    options = _options(source)
    reader = SourceReader(context)
    files = (
        locate_s3(reader, path, options)
        if isinstance(path, str)
        else locate(path, options)
    )
    model = read_ontology(files, options, reader=reader)
    mapping = map_ontology(model, options)
    # Counts only: no label or description of the ontology.
    counts = mapping.report.summary()
    logger.info(
        "Ontology '%s' (%d %s): %s",
        context.location,
        len(files),
        "file" if len(files) == 1 else "files",
        counts,
    )
    for line in mapping.report.details():
        logger.debug("Ontology '%s': %s", context.location, line)
    return mapping.document


def _path(source: SchemaSourceConfig, location: str) -> Path | str:
    """The local path ``source.path`` names, or the ``s3://`` URL as is."""
    if not source.path:
        raise SchemaSourceError(
            "processing.graph_extraction.schema_source.path is required for "
            "kind 'owl'"
        )
    if source.secret_arn:
        raise SchemaSourceError(
            "kind 'owl' takes no schema_source.secret_arn; leave it unset"
        )
    if "://" not in source.path:
        return Path(source.path)
    if is_s3(source.path):
        split_s3(source.path)
        return source.path
    parts = urlsplit(source.path)
    if parts.scheme.lower() != "file" or parts.netloc not in ("", "localhost"):
        raise SchemaSourceError(
            f"Ontology '{location}' is a URL; kind 'owl' reads a local file "
            "or directory (or a file:// URL) or an s3:// object or prefix. "
            "Copy the ontology next to the config (or into the image, or S3)"
        )
    return Path(url2pathname(parts.path))


def _options(source: SchemaSourceConfig) -> OwlSourceOptions:
    try:
        return OwlSourceOptions.model_validate(source.options)
    except ValidationError as e:
        raise SchemaSourceError(
            "; ".join(
                "schema_source.options"
                + "".join(f".{part}" for part in detail["loc"])
                + f": {detail['msg']}"
                for detail in e.errors(include_url=False, include_input=False)
            )
        ) from e


# -- Files -------------------------------------------------------------------


def _refuse_suffix(name: str, suffix: str) -> None:
    what = REFUSED_SUFFIXES.get(suffix)
    if what is not None:
        raise SchemaSourceError(
            f"Ontology file '{name}' is {what}, which kind 'owl' does not "
            f"read; {_CONVERT}"
        )


OntologyFiles = list[tuple[SourceFile, str]]


def locate(path: Path, options: OwlSourceOptions) -> OntologyFiles:
    """The files at the local ``path`` and the format each is read as:
    ``path`` itself, or the ontology files under the directory (by suffix,
    or every file when ``options.format`` is set; sorted; names starting
    with ``.`` skipped).

    Raises:
        SchemaSourceError: ``path`` is missing, is or holds a refused
            format, has an unknown suffix (without ``options.format``),
            holds no ontology file, or the files hold more than
            ``options.max_bytes``.
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
            options,
        )
    if path.is_file():
        return _single(
            SourceFile(location=str(path), size=path.stat().st_size), options
        )
    raise SchemaSourceError(f"Ontology '{path}' not found")


def locate_s3(
    reader: SourceReader, location: str, options: OwlSourceOptions
) -> OntologyFiles:
    """:func:`locate` of an S3 location: the object ``location`` names
    (sized with HeadObject), or, for a prefix (``s3://bucket/prefix/``,
    or the bucket root), the ontology objects under it (every page of the
    listing), filtered, sorted and refused as a local directory is.

    Raises:
        SchemaSourceError: What :func:`locate` raises, and the S3 errors of
            :class:`.locations.SourceReader` (access denied, ...).
    """
    _, key = split_s3(location)
    if not key or key.endswith("/"):
        return _directory(
            redact_location(location), reader.list_prefix(location), options
        )
    return _single(reader.head(location, prefix_hint=True), options)


def _single(file: SourceFile, options: OwlSourceOptions) -> OntologyFiles:
    _refuse_suffix(file.name, file.suffix)
    if file.suffix not in SUFFIX_FORMATS and options.format == "auto":
        raise SchemaSourceError(
            f"Ontology file '{file.name}' must end in "
            f"{', '.join(SUFFIX_FORMATS)}, or set "
            "schema_source.options.format"
        )
    return _formats(file.name, [file], options)


def _directory(
    name: str, candidates: list[tuple[str, SourceFile]], options: OwlSourceOptions
) -> OntologyFiles:
    """The ontology files of ``candidates`` (each with its path relative
    to the directory, sorted): those with an ontology suffix (any, when
    ``options.format`` is set), outside any name starting with ``.``."""
    found = []
    for relative, candidate in candidates:
        if any(part.startswith(".") for part in relative.split("/")):
            continue
        _refuse_suffix(candidate.name, candidate.suffix)
        if options.format != "auto" or candidate.suffix in SUFFIX_FORMATS:
            found.append(candidate)
    if not found:
        raise SchemaSourceError(
            f"Ontology directory '{name}' holds no ontology file "
            f"({', '.join(SUFFIX_FORMATS)})"
        )
    return _formats(name, found, options)


def _formats(
    name: str, found: list[SourceFile], options: OwlSourceOptions
) -> OntologyFiles:
    size = sum(candidate.size for candidate in found)
    if size > options.max_bytes:
        raise SchemaSourceError(
            f"Ontology '{name}' holds {size} bytes, more than "
            f"schema_source.options.max_bytes ({options.max_bytes}); point "
            "the path at the files the schema needs, or raise max_bytes"
        )
    return [
        (
            candidate,
            (
                SUFFIX_FORMATS[candidate.suffix]
                if options.format == "auto"
                else options.format
            ),
        )
        for candidate in found
    ]


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as e:
        raise SchemaSourceError(
            f"Cannot read ontology file '{path}': {e.strerror or e}"
        ) from e


# -- Safety checks before parsing --------------------------------------------


class _Root(Exception):
    """Stops the XML scan at the root element."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


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


def _check_xml(path: str, data: bytes) -> None:
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
        raise SchemaSourceError(
            f"Ontology file '{path}' holds {e}, which the RDF/XML parser "
            "takes quadratic time over; convert the file to Turtle"
        ) from e
    except expat.ExpatError as e:
        # The message has a position, never the file's text.
        raise SchemaSourceError(
            f"Ontology file '{path}' does not parse as xml: {e}"
        ) from e
    if scan.root == OWL + " Ontology":
        raise SchemaSourceError(
            f"Ontology file '{path}' is OWL/XML (its root is owl:Ontology), "
            f"which kind 'owl' does not read; {_CONVERT}"
        )


def _check_lines(path: str, data: bytes) -> None:
    """Refuse an N-Triples or N-Quads line longer than ``MAX_LINE_BYTES``."""
    longest = max(map(len, data.splitlines()), default=0)
    if longest > MAX_LINE_BYTES:
        raise SchemaSourceError(
            f"Ontology file '{path}' has a line of {longest} bytes, more than "
            f"the {MAX_LINE_BYTES} (MAX_LINE_BYTES) the N-Triples parser reads "
            "in reasonable time; convert the file to Turtle"
        )


def check_bytes(path: str, data: bytes, format_: str) -> None:
    """Refuse what the parser of ``format_`` must not see (the module
    docstring).

    Raises:
        SchemaSourceError: ``data`` is OWL/XML, XML that does not parse or
            past a limit, an N-Triples or N-Quads file with a line past
            ``MAX_LINE_BYTES``, or JSON-LD naming a remote context.
    """
    if format_ == "xml":
        _check_xml(path, data)
    elif format_ in ("nt", "nquads"):
        _check_lines(path, data)
    elif format_ == "json-ld":
        _check_json_ld(path, data)


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


def _check_json_ld(path: str, data: bytes) -> None:
    """Refuse a JSON-LD context that names a document: the parser would
    fetch (or open) it."""
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        raise SchemaSourceError(
            f"Ontology file '{path}' does not parse as json-ld: "
            f"{' '.join(str(e).split())}"
        ) from e
    for key, value in _walk(document):
        contexts = value if isinstance(value, list) else [value]
        if (key == "@context" and any(isinstance(c, str) for c in contexts)) or (
            key == "@import"
        ):
            raise SchemaSourceError(
                f"Ontology file '{path}' names a remote JSON-LD context "
                f"({key}), which would be fetched; inline the context, or "
                "convert the file to Turtle"
            )


# -- Reading -----------------------------------------------------------------


def read_ontology(
    files: OntologyFiles,
    options: OwlSourceOptions,
    *,
    reader: SourceReader | None = None,
) -> OntologyModel:
    """The model of what ``files`` (each with its format, from
    :func:`locate` or :func:`locate_s3`) state together; ``reader`` reads
    the S3 ones.

    Raises:
        SchemaSourceError: rdflib is not installed, or a file cannot be
            read, is refused or does not parse.
    """
    try:
        from . import owl_reader
    except ImportError as e:
        if not (e.name or "").startswith("rdflib"):
            raise
        raise SchemaSourceError(
            "schema_source.kind 'owl' needs rdflib, which is not installed; "
            f"install {INSTALL_HINT}"
        ) from e
    statements = owl_reader.Statements()
    total = 0
    for file, format_ in files:
        if is_s3(file.location):
            if reader is None:
                raise ValueError("read_ontology needs a reader for S3 files")
            data = reader.read(file, max_bytes=options.max_bytes)
        else:
            data = _read(Path(file.location))
        # The files may have grown since they were sized.
        total += len(data)
        if total > options.max_bytes:
            raise SchemaSourceError(
                f"Ontology files hold more than schema_source.options.max_bytes "
                f"({options.max_bytes}) bytes"
            )
        if _xml_root(data) == _CATALOG_ROOT:
            if len(files) == 1:
                raise SchemaSourceError(
                    f"Ontology file '{file.name}' is an OASIS XML catalog "
                    "(Protégé's catalog-v001.xml), not an ontology; point the "
                    "path at the ontology files"
                )
            logger.info("Ontology: skipped the XML catalog '%s'", file.name)
            continue
        check_bytes(file.name, data, format_)
        owl_reader.parse(statements, file.name, file.base, data, format_)
    if unfollowed := owl_reader.unfollowed_imports(statements.graph):
        logger.warning(
            "Ontology: owl:imports not followed: %s; put the imported "
            "ontology's file in the schema_source directory to map its terms",
            ", ".join(unfollowed),
        )
    model = owl_reader.ontology_model(statements.graph, statements.prefixes())
    if unread := owl_reader.unread_properties(statements.graph):
        logger.info(
            "Ontology: %d properties not read (a domain, range or "
            "owl:FunctionalProperty, but no rdf:type saying object, datatype "
            "or annotation, and no sub- or super-property of one): %s; type "
            "them owl:ObjectProperty to map them",
            len(unread),
            ", ".join(unread[:5]) + (", ..." if len(unread) > 5 else ""),
        )
    relative = sorted(
        term.iri
        for term in (*model.classes, *model.properties)
        if term.iri.startswith(("file:", "s3:"))
    )
    if relative:
        logger.warning(
            "Ontology: %d terms have IRIs relative to their file (%s, ...), "
            "so moving the file (or the S3 object) renames them; declare a "
            "base (@base, xml:base) or write absolute IRIs",
            len(relative),
            relative[0],
        )
    return model
