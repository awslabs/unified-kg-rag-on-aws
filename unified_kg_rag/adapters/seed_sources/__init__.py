# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed graph sources and their registry.

Importing this package registers the built-in sources: ``jsonl`` and
``csv`` (:mod:`.property_graph` holds what they share), and ``rdf``
(:mod:`.rdf_reader`, whose loader imports the ``ontology`` extra only when
it runs; :mod:`.abox_mapping` holds its mapping, free of rdflib). A source of
another format registers its kind with :func:`register_seed_source`; one
that needs an optional dependency declares it (``requires=``), so the
CLIs check it before any model call. An RDF source reads its files with
:mod:`unified_kg_rag.adapters.source_files` (``rdf_files``, ``rdf_parse``),
passing :attr:`SeedSourceContext.terms` for its errors.
"""

# Importing a source module registers it.
from . import csv_source, jsonl, rdf_reader  # noqa: F401
from .registry import (
    COMMON_OPTIONS,
    SEED_TERMS,
    SeedLoader,
    SeedSourceContext,
    SeedSourceOptions,
    SeedTerms,
    get_seed_source,
    register_seed_source,
    registered_seed_sources,
    seed_options,
    seed_source_requirement,
)

__all__ = [
    "COMMON_OPTIONS",
    "SEED_TERMS",
    "SeedLoader",
    "SeedSourceContext",
    "SeedSourceOptions",
    "SeedTerms",
    "get_seed_source",
    "register_seed_source",
    "registered_seed_sources",
    "seed_options",
    "seed_source_requirement",
]
