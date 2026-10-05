# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which document-index prefixes a configured pipeline builds.

Config-only helpers shared by the document retriever's default sweep and the
strategies that pin a sweep explicitly, so any document backend honours the
same ingestion gates.
"""

from unified_kg_rag.domain.models import Config


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
    reports need ``graph.community_detection.enabled``. Explicit
    ``index_prefixes`` still reach any mapped index.
    """
    opensearch = config.indexing.opensearch
    optional_enabled = {
        opensearch.claims_index_prefix: config.processing.claim_extraction.enabled,
        opensearch.relationships_index_prefix: (
            opensearch.build_relationship_vector_index
        ),
        opensearch.community_reports_index_prefix: (
            config.graph.community_detection.enabled
        ),
    }
    return [
        prefix
        for prefix in all_index_prefixes(config)
        if optional_enabled.get(prefix, True)
    ]
