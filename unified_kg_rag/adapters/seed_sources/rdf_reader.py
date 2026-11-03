# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kind: rdf``: an RDF A-Box (the instances of an ontology) as a seed graph.

``path`` names a local file, or a directory whose RDF files (by suffix,
recursively; every file when ``options.format`` is set) are read together;
or an S3 object, or a prefix read as a directory (``s3://bucket/seed/``)::

    processing:
      graph_extraction:
        schema_source: {kind: owl, path: ontology/procure.ttl}
      seed_graphs:
        - name: procure
          kind: rdf
          path: seed/procure-abox.ttl
          options: {languages: [ko, en, ""]}

The files are Turtle, N-Triples, N-Quads, TriG, RDF/XML or JSON-LD, read
as the ``owl`` schema source reads an ontology, with the same checks
(:mod:`.source_files.rdf_files`): parsed from their bytes, never from a
location the parser would open; ``owl:imports`` not followed (one WARNING
names them); a JSON-LD ``@context`` or ``@import`` naming a document
refused; XML external entities not resolved; OWL/XML, functional and
Manchester syntax, OBO and Notation3 refused; an OASIS XML catalog in a
directory skipped; the shapes the parsers take quadratic time over refused
before parsing. The files together may hold at most the seed's
``max_bytes``, checked before any byte is read and again while reading.

:func:`.abox_mapping.map_abox` turns the statements into nodes and edges
against the run's resolved extraction schema (the options:
:class:`RdfSeedOptions`). Instance IRIs relative to their file (no
``@base``) are warned about: moving the file renames them.

Needs the ``ontology`` extra (rdflib); without it the loader, and the CLIs
before any model call, say how to install it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from unified_kg_rag.adapters.schema_sources import INSTALL_HINT
from unified_kg_rag.adapters.source_files import rdf_files
from unified_kg_rag.domain.models.seed_graph import SeedGraph
from unified_kg_rag.shared import get_logger

from .abox_mapping import AboxMappingOptions, RdfTerm, Triple, map_abox
from .property_graph import FileReader, check_path
from .registry import SeedSourceContext, register_seed_source

if TYPE_CHECKING:
    from unified_kg_rag.adapters.source_files.rdf_parse import Statements

logger = get_logger(__name__)


class RdfSeedOptions(AboxMappingOptions):
    """``processing.seed_graphs[].options`` of kind ``rdf``: the mapping
    options and how the files are read."""

    format: rdf_files.RdfFormat = Field(
        default="auto",
        description="The serialisation of every file (auto: by suffix; set, "
        "every file of a directory is read, whatever its suffix)",
    )


@register_seed_source("rdf", requires=("rdflib", INSTALL_HINT))
def load_rdf_seed(context: SeedSourceContext) -> SeedGraph:
    """The seed graph the RDF file(s) at ``source.path`` state.

    Raises:
        SeedGraphError: rdflib is not installed; an option is unknown or
            invalid; ``secret_arn`` is set; the path is a URL other than
            ``s3://``, is missing (or access is denied), holds no RDF file,
            a refused format, a remote JSON-LD context or OWL/XML, or more
            than ``max_bytes``; a file does not parse; a term of the options
            is not bound.
        ConfigurationError: The extraction schema source is not resolved.
    """
    options = context.options(RdfSeedOptions)
    path = check_path(context)
    # Before any file is read: a seed of an unresolved schema cannot map.
    schema = context.config.processing.graph_extraction.extraction_schema()
    reader = FileReader(context)
    max_bytes = context.source.max_bytes
    files = (
        rdf_files.locate_s3(
            reader.s3,
            path,
            format_=options.format,
            max_bytes=max_bytes,
            terms=context.terms,
        )
        if isinstance(path, str)
        else rdf_files.locate(
            path, format_=options.format, max_bytes=max_bytes, terms=context.terms
        )
    )
    statements = read_statements(context, reader, files)
    triples = rdf_triples(statements)
    report = reader.report()
    mapping = map_abox(
        triples,
        options,
        schema,
        prefixes=statements.prefixes(),
        terms=context.terms,
        files=report.files,
        bytes_read=report.bytes_read,
    )
    name = context.source.name
    if mapping.unfollowed_imports:
        logger.warning(
            "Seed graph '%s': owl:imports not followed: %s; a seed states the "
            "triples of its own files only",
            name,
            ", ".join(mapping.unfollowed_imports),
        )
    if mapping.relative_iris:
        logger.warning(
            "Seed graph '%s': %d instances have IRIs relative to their file "
            "(%s, ...), so moving the file (or the S3 object) renames them; "
            "declare a base (@base, xml:base) or write absolute IRIs",
            name,
            len(mapping.relative_iris),
            mapping.relative_iris[0],
        )
    graph = mapping.graph
    # Counts only: no name, description or property of the seed.
    logger.info(
        "Seed graph '%s' (rdf, %d triples): %d instances, %d edges; skipped %s",
        name,
        len(triples),
        len(graph.nodes),
        len(graph.edges),
        ", ".join(f"{k} {v}" for k, v in graph.report.skipped.items()) or "nothing",
    )
    return graph


def read_statements(
    context: SeedSourceContext, reader: FileReader, files: rdf_files.RdfFiles
) -> Statements:
    """What ``files`` (each with its format) state together, each checked
    before it is parsed; an XML catalog among several files is skipped.

    Raises:
        SeedGraphError: rdflib is not installed, or a file cannot be read,
            grew past ``max_bytes``, is refused or does not parse.
    """
    terms = context.terms
    try:
        from unified_kg_rag.adapters.source_files import rdf_parse
    except ImportError as e:
        # Only a missing rdflib is the missing extra: another import error
        # fails as itself.
        if not (e.name or "").startswith("rdflib"):
            raise
        raise terms(
            f"kind 'rdf' needs rdflib, which is not installed; install {INSTALL_HINT}"
        ) from e
    statements = rdf_parse.Statements()
    for file, format_ in files:
        data = reader.read(file)
        if rdf_files.is_xml_catalog(data):
            if len(files) == 1:
                raise terms(
                    f"file '{file.name}' is an OASIS XML catalog (Protégé's "
                    "catalog-v001.xml), not RDF; point the path at the seed's "
                    "files"
                )
            logger.info(
                "Seed graph '%s': skipped the XML catalog '%s'",
                context.source.name,
                file.name,
            )
            continue
        rdf_files.check_bytes(file.name, data, format_, terms=terms)
        rdf_parse.parse(statements, file.name, file.base, data, format_, terms=terms)
    return statements


def rdf_triples(statements: Statements) -> list[Triple]:
    """The triples of ``statements`` as :class:`.abox_mapping.RdfTerm`
    tuples (rdflib is installed: ``statements`` holds an rdflib graph)."""
    from rdflib import BNode, Literal, URIRef

    def term(node: Any) -> RdfTerm:
        if isinstance(node, URIRef):
            return RdfTerm("iri", str(node))
        if isinstance(node, BNode):
            return RdfTerm("blank", str(node))
        if isinstance(node, Literal):
            return RdfTerm(
                "literal",
                str(node),
                str(node.datatype) if node.datatype is not None else None,
                node.language or "",
            )
        # A triple term or formula (RDF-star, Notation3): counted, skipped.
        return RdfTerm("triple")

    return [(term(s), term(p), term(o)) for s, p, o in statements.graph]
