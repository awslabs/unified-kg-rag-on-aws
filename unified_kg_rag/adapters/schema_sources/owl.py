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
before parsing (:func:`.source_files.rdf_files.check_bytes`): XML whose entities expand past the
XML parser's amplification limit, an XML element with more than
``MAX_XML_TEXT_CHUNKS`` pieces of text (a run of line breaks or entity
references), more than ``MAX_XML_LITERAL_ELEMENTS`` elements inside
``rdf:parseType="Literal"`` values, and an N-Triples or N-Quads line longer
than ``MAX_LINE_BYTES``.

Needs the ``ontology`` extra (rdflib); without it the loader, and the CLIs
before any model call, say how to install it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, ValidationError

from unified_kg_rag.adapters.source_files import rdf_files
from unified_kg_rag.adapters.source_files.locations import (
    SourceReader,
    SourceTerms,
    resolve_source_path,
)
from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.domain.models.schema import SchemaDocument
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import SchemaSourceError
from unified_kg_rag.shared.extras import extra_hint
from unified_kg_rag.shared.utils.s3_uri import is_s3_uri
from unified_kg_rag.shared.utils.validation import format_validation_errors

from .ontology_mapping import MappingOptions, map_ontology
from .ontology_model import OntologyModel
from .registry import SchemaSourceContext, register_schema_source

logger = get_logger(__name__)

INSTALL_HINT = extra_hint("ontology")

# How the files' errors name the source (the reader's S3 errors keep the
# default "Schema source").
ONTOLOGY_TERMS = SourceTerms(noun="Ontology", kind="owl")


class OwlSourceOptions(MappingOptions):
    """``schema_source.options`` of kind ``owl``: the mapping options and
    how the files are read."""

    format: rdf_files.RdfFormat = Field(
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
    return resolve_source_path(
        source.path,
        refusal=(
            f"Ontology '{location}' is a URL; kind 'owl' reads a local file "
            "or directory (or a file:// URL) or an s3:// object or prefix. "
            "Copy the ontology next to the config (or into the image, or S3)"
        ),
    )


def _options(source: SchemaSourceConfig) -> OwlSourceOptions:
    try:
        return OwlSourceOptions.model_validate(source.options)
    except ValidationError as e:
        raise SchemaSourceError(
            format_validation_errors("schema_source.options", e)
        ) from e


# -- Files -------------------------------------------------------------------


OntologyFiles = rdf_files.RdfFiles


def locate(path: Path, options: OwlSourceOptions) -> OntologyFiles:
    """The files at the local ``path`` and the format each is read as
    (:func:`.source_files.rdf_files.locate` with ``options.format`` and
    ``options.max_bytes``).

    Raises:
        SchemaSourceError: ``path`` is missing, is or holds a refused
            format, has an unknown suffix (without ``options.format``),
            holds no ontology file, or the files hold more than
            ``options.max_bytes``.
    """
    return rdf_files.locate(
        path,
        format_=options.format,
        max_bytes=options.max_bytes,
        terms=ONTOLOGY_TERMS,
    )


def locate_s3(
    reader: SourceReader, location: str, options: OwlSourceOptions
) -> OntologyFiles:
    """:func:`locate` of an S3 object or prefix
    (:func:`.source_files.rdf_files.locate_s3`).

    Raises:
        SchemaSourceError: What :func:`locate` raises, and the S3 errors of
            :class:`.source_files.locations.SourceReader` (access denied,
            ...).
    """
    return rdf_files.locate_s3(
        reader,
        location,
        format_=options.format,
        max_bytes=options.max_bytes,
        terms=ONTOLOGY_TERMS,
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
        # The package's own module, which imports rdflib: one that fails to
        # import for another reason fails as itself, rdflib installed or not.
        from . import owl_reader
    except ImportError as e:
        if not (e.name or "").startswith("rdflib"):
            raise
        raise SchemaSourceError(
            "schema_source.kind 'owl' needs rdflib, which is not installed; "
            f"install {INSTALL_HINT}"
        ) from e
    # rdflib is installed: owl_reader imported it.
    from unified_kg_rag.adapters.source_files import rdf_parse

    statements = rdf_parse.Statements()
    total = 0
    for file, format_ in files:
        if is_s3_uri(file.location):
            if reader is None:
                raise SchemaSourceError("read_ontology needs a reader for S3 files")
            data = reader.read(file, max_bytes=options.max_bytes)
        else:
            data = rdf_files.read_local(Path(file.location), terms=ONTOLOGY_TERMS)
        # The files may have grown since they were sized.
        total += len(data)
        if total > options.max_bytes:
            raise SchemaSourceError(
                f"Ontology files hold more than schema_source.options.max_bytes "
                f"({options.max_bytes}) bytes"
            )
        if rdf_files.is_xml_catalog(data):
            if len(files) == 1:
                raise SchemaSourceError(
                    f"Ontology file '{file.name}' is an OASIS XML catalog "
                    "(Protégé's catalog-v001.xml), not an ontology; point the "
                    "path at the ontology files"
                )
            logger.info("Ontology: skipped the XML catalog '%s'", file.name)
            continue
        rdf_files.check_bytes(file.name, data, format_, terms=ONTOLOGY_TERMS)
        rdf_parse.parse(
            statements, file.name, file.base, data, format_, terms=ONTOLOGY_TERMS
        )
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
