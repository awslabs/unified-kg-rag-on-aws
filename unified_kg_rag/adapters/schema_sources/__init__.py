# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extraction-schema sources and their registry.

Importing this package registers the built-in sources (``file``, and
``owl``, whose loader imports the ``ontology`` extra only when it runs).
The RDF vocabulary IRIs and naming helpers the ontology mapping shares
with the ``rdf`` seed source live in
:mod:`unified_kg_rag.adapters.source_files.rdf_terms`.
"""

from unified_kg_rag.adapters.source_files.locations import redact_location

# Importing a source module registers it.
from . import file, owl  # noqa: F401
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
    "SchemaLoader",
    "SchemaSourceContext",
    "get_schema_source",
    "redact_location",
    "register_schema_source",
    "registered_schema_sources",
    "schema_source_requirement",
]
