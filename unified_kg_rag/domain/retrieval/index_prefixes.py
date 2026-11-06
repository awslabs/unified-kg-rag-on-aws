# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which document-index prefixes a configured pipeline builds.

Config-only helpers shared by the document retriever's default sweep and the
strategies that pin a sweep explicitly, so any document backend honours the
same ingestion gates.
"""

from collections.abc import Collection, Mapping
from typing import Any

from unified_kg_rag.domain.ingestion.methodologies import artifacts_to_build
from unified_kg_rag.domain.models import Config, IndexArtifact

# ``SearchQuery.metadata`` key of the index artifacts the queried namespace
# holds (``IndexArtifact`` values), which the chain sets.
HELD_ARTIFACTS_KEY = "held_artifacts"


def all_index_prefixes(config: Config) -> list[str]:
    """Every document index prefix, in sweep order."""
    opensearch = config.indexing.opensearch
    return [
        opensearch.text_units_index_prefix,
        opensearch.entities_index_prefix,
        opensearch.relationships_index_prefix,
        opensearch.claims_index_prefix,
        opensearch.community_reports_index_prefix,
    ]


def configured_index_prefixes(config: Config) -> list[str]:
    """Index prefixes the configured pipeline actually builds.

    This is the default sweep when a query pins no ``index_prefixes``. Optional
    indices are skipped when their producing stage is disabled, so the sweep
    never queries an alias that was never created: claims need
    ``processing.claim_extraction.enabled``, the relationship vector index needs
    ``indexing.opensearch.build_relationship_vector_index``, and community
    reports need ``graph.community_detection.enabled`` and ``graphrag`` among
    the effective ``indexing.methodologies``. Explicit ``index_prefixes``
    still reach any mapped index.
    """
    opensearch = config.indexing.opensearch
    optional_enabled = {
        opensearch.claims_index_prefix: config.processing.claim_extraction.enabled,
        opensearch.relationships_index_prefix: (
            opensearch.build_relationship_vector_index
        ),
        opensearch.community_reports_index_prefix: (
            IndexArtifact.COMMUNITY_REPORTS in artifacts_to_build(config)
        ),
    }
    return [
        prefix
        for prefix in all_index_prefixes(config)
        if optional_enabled.get(prefix, True)
    ]


def held_artifacts_of(
    metadata: Mapping[str, Any] | None,
) -> frozenset[IndexArtifact] | None:
    """The artifacts ``SearchQuery.metadata`` says the namespace holds
    (``HELD_ARTIFACTS_KEY``), or ``None`` when it does not say (a caller
    that drives a strategy directly)."""
    values = (metadata or {}).get(HELD_ARTIFACTS_KEY)
    if not isinstance(values, list):
        return None
    known = {a.value: a for a in IndexArtifact}
    return frozenset(known[v] for v in values if isinstance(v, str) and v in known)


def held_index_prefixes(config: Config, held: Collection[IndexArtifact]) -> list[str]:
    """Of ``configured_index_prefixes``, those whose artifact the namespace
    holds (``held``), so a sweep never searches an index it lacks."""
    opensearch = config.indexing.opensearch
    artifact = {
        opensearch.text_units_index_prefix: IndexArtifact.TEXT_UNITS,
        opensearch.entities_index_prefix: IndexArtifact.ENTITIES,
        opensearch.relationships_index_prefix: IndexArtifact.RELATIONSHIP_VECTORS,
        opensearch.claims_index_prefix: IndexArtifact.CLAIMS,
        opensearch.community_reports_index_prefix: IndexArtifact.COMMUNITY_REPORTS,
    }
    return [
        prefix
        for prefix in configured_index_prefixes(config)
        if artifact.get(prefix) is None or artifact[prefix] in held
    ]
