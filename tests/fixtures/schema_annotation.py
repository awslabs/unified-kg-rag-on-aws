# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema annotation digests the tests compare registry records with."""

from __future__ import annotations

from unified_kg_rag.domain.ingestion.schema_annotation import SchemaAnnotator
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.schema import ExtractionSchema

# The digest of an annotator that annotates nothing: what every item written
# without schema annotations holds (no schema, or one that gives none).
UNANNOTATED_DIGEST = SchemaAnnotator(ExtractionSchema()).digest


def schema_annotation_digest(config: Config) -> str:
    """The digest of the schema annotations indexing writes with ``config``
    (``IndexingManager.annotation_digest``); raises ``ConfigurationError``
    while a schema source is not loaded."""
    return SchemaAnnotator(
        config.processing.graph_extraction.extraction_schema()
    ).digest
