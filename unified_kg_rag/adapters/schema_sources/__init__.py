# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extraction-schema sources and their registry.

Importing this package registers the built-in sources (``file``, and
``owl``, whose loader imports the ``ontology`` extra only when it runs).
It also exports the ontology mapping's pure naming helpers (``local_name``,
``upper_snake``, ``best_text``, ...) and vocabulary IRIs for another reader
of RDF terms (the ``rdf`` seed source).
"""

from unified_kg_rag.adapters.source_files.locations import redact_location

# Importing a source module registers it.
from . import file, owl  # noqa: F401
from .ontology_mapping import (
    WELL_KNOWN_PREFIXES,
    best_text,
    local_name,
    namespace_of,
    upper_snake,
)
from .ontology_model import OWL, RDF, RDFS, SCHEMA, SKOS, XSD, Text
from .owl import INSTALL_HINT
from .registry import (
    SchemaLoader,
    SchemaSourceContext,
    get_schema_source,
    register_schema_source,
    registered_schema_sources,
    schema_source_requirement,
)

__all__ = [
    "INSTALL_HINT",
    "OWL",
    "RDF",
    "RDFS",
    "SCHEMA",
    "SKOS",
    "WELL_KNOWN_PREFIXES",
    "XSD",
    "SchemaLoader",
    "SchemaSourceContext",
    "Text",
    "best_text",
    "get_schema_source",
    "local_name",
    "namespace_of",
    "redact_location",
    "register_schema_source",
    "registered_schema_sources",
    "schema_source_requirement",
    "upper_snake",
]
