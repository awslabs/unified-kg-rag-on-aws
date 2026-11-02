# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``schema_source.kind: owl``: an OWL or RDFS ontology as the schema.

``path`` names a local file, or a directory whose ontology files (by
suffix, recursively) are read together::

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

Nothing is fetched. The files are parsed from their bytes (never from a
location the parser would open); ``owl:imports`` is not followed (an import
no file of ``path`` declares is logged once: put that ontology in the
directory); a JSON-LD ``@context`` or ``@import`` naming a document is
refused, since the parser would fetch it; XML external entities are not
resolved. OWL/XML, functional and Manchester syntax, OBO and Notation3 are
refused, with how to convert them.

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

from .ontology_mapping import MappingOptions, map_ontology
from .ontology_model import OWL, OntologyModel
from .registry import SchemaSourceContext, register_schema_source

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
        description="The serialisation of every file (auto: by suffix)",
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
            path is a URL other than ``file://`` or ``secret_arn`` is set;
            an option is unknown or invalid; the path is missing, holds no
            ontology file, a refused format, a remote JSON-LD context or
            OWL/XML, or more than ``max_bytes``; a file does not parse; or
            the mapping fails (``map_ontology``).
    """
    source = context.source
    path = _local_path(source, context.location)
    options = _options(source)
    files = locate(path, options)
    model = read_ontology(files, options)
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


def _local_path(source: SchemaSourceConfig, location: str) -> Path:
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
    parts = urlsplit(source.path)
    if parts.scheme.lower() != "file" or parts.netloc not in ("", "localhost"):
        raise SchemaSourceError(
            f"Ontology '{location}' is a URL; kind 'owl' reads a local file "
            "or directory (or a file:// URL). Copy the ontology next to the "
            "config (or into the image)"
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


def _refuse_suffix(path: Path) -> None:
    what = REFUSED_SUFFIXES.get(path.suffix.lower())
    if what is not None:
        raise SchemaSourceError(
            f"Ontology file '{path}' is {what}, which kind 'owl' does not "
            f"read; {_CONVERT}"
        )


def locate(path: Path, options: OwlSourceOptions) -> list[tuple[Path, str]]:
    """The files at ``path`` and the format each is read as: ``path``
    itself, or the ontology files under the directory (by suffix, sorted;
    names starting with ``.`` skipped).

    Raises:
        SchemaSourceError: ``path`` is missing, is or holds a refused
            format, has an unknown suffix (without ``options.format``),
            holds no ontology file, or the files hold more than
            ``options.max_bytes``.
    """
    if path.is_dir():
        found = []
        for candidate in sorted(path.rglob("*")):
            relative = candidate.relative_to(path)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if not candidate.is_file():
                continue
            _refuse_suffix(candidate)
            if candidate.suffix.lower() in SUFFIX_FORMATS:
                found.append(candidate)
        if not found:
            raise SchemaSourceError(
                f"Ontology directory '{path}' holds no ontology file "
                f"({', '.join(SUFFIX_FORMATS)})"
            )
    elif path.is_file():
        _refuse_suffix(path)
        if path.suffix.lower() not in SUFFIX_FORMATS and options.format == "auto":
            raise SchemaSourceError(
                f"Ontology file '{path}' must end in "
                f"{', '.join(SUFFIX_FORMATS)}, or set "
                "schema_source.options.format"
            )
        found = [path]
    else:
        raise SchemaSourceError(f"Ontology '{path}' not found")
    size = sum(candidate.stat().st_size for candidate in found)
    if size > options.max_bytes:
        raise SchemaSourceError(
            f"Ontology '{path}' holds {size} bytes, more than "
            f"schema_source.options.max_bytes ({options.max_bytes}); point "
            "the path at the files the schema needs, or raise max_bytes"
        )
    return [
        (
            candidate,
            (
                SUFFIX_FORMATS[candidate.suffix.lower()]
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
    when it has none. External entities are not loaded (no handler)."""
    parser = expat.ParserCreate(namespace_separator=" ")

    def start(name: str, attributes: Any) -> None:
        raise _Root(name)

    parser.StartElementHandler = start
    try:
        parser.Parse(data, True)
    except _Root as root:
        return root.name
    except expat.ExpatError:
        # The RDF/XML parser reports the syntax error with its context.
        pass
    return None


def _check_xml(path: Path, data: bytes) -> None:
    if _xml_root(data) == OWL + " Ontology":
        raise SchemaSourceError(
            f"Ontology file '{path}' is OWL/XML (its root is owl:Ontology), "
            f"which kind 'owl' does not read; {_CONVERT}"
        )


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


def _check_json_ld(path: Path, data: bytes) -> None:
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
    files: list[tuple[Path, str]], options: OwlSourceOptions
) -> OntologyModel:
    """The model of what ``files`` (each with its format, from
    :func:`locate`) state together.

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
    for path, format_ in files:
        data = _read(path)
        # The files may have grown since they were sized.
        total += len(data)
        if total > options.max_bytes:
            raise SchemaSourceError(
                f"Ontology files hold more than schema_source.options.max_bytes "
                f"({options.max_bytes}) bytes"
            )
        if format_ == "xml":
            _check_xml(path, data)
        elif format_ == "json-ld":
            _check_json_ld(path, data)
        owl_reader.parse(statements, path, data, format_)
    if unfollowed := owl_reader.unfollowed_imports(statements.graph):
        logger.warning(
            "Ontology: owl:imports not followed: %s; put the imported "
            "ontology's file in the schema_source directory to map its terms",
            ", ".join(unfollowed),
        )
    model = owl_reader.ontology_model(statements.graph, statements.prefixes())
    relative = sorted(
        term.iri
        for term in (*model.classes, *model.properties)
        if term.iri.startswith("file:")
    )
    if relative:
        logger.warning(
            "Ontology: %d terms have IRIs relative to their file (%s, ...), "
            "so moving the file renames them; declare a base (@base, "
            "xml:base) or write absolute IRIs",
            len(relative),
            relative[0],
        )
    return model
