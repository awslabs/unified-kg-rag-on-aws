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
from unified_kg_rag.domain.models.methodology import parse_artifacts

# ``SearchQuery.metadata`` key of the index artifacts the queried namespace
# holds (``IndexArtifact`` values), which the chain sets.
HELD_ARTIFACTS_KEY = "held_artifacts"


def _prefix_artifacts(config: Config) -> list[tuple[str, IndexArtifact]]:
    """Every document index prefix, in sweep order, with the artifact it holds."""
    opensearch = config.indexing.opensearch
    return [
        (opensearch.text_units_index_prefix, IndexArtifact.TEXT_UNITS),
        (opensearch.entities_index_prefix, IndexArtifact.ENTITIES),
        (opensearch.relationships_index_prefix, IndexArtifact.RELATIONSHIP_VECTORS),
        (opensearch.claims_index_prefix, IndexArtifact.CLAIMS),
        (opensearch.community_reports_index_prefix, IndexArtifact.COMMUNITY_REPORTS),
    ]


def index_artifacts(config: Config) -> dict[str, IndexArtifact]:
    """Every vector-store index prefix the indexer writes (the swept ones and
    the summary tree's) with the artifact it holds."""
    return {
        **dict(_prefix_artifacts(config)),
        config.indexing.opensearch.summary_nodes_index_prefix: (
            IndexArtifact.SUMMARY_TREE
        ),
    }


def all_index_prefixes(config: Config) -> list[str]:
    """Every document index prefix, in sweep order."""
    return [prefix for prefix, _ in _prefix_artifacts(config)]


def configured_index_prefixes(config: Config) -> list[str]:
    """Index prefixes the configured pipeline actually builds.

    This is the default sweep when a query pins no ``index_prefixes``. An
    index whose artifact a run of the config does not build
    (``artifacts_to_build``: a switch of ``ARTIFACT_SWITCHES`` turns it off,
    or no effective methodology reads it, as the entities under ``raptor``
    alone) is skipped, so the sweep never queries an alias that was never
    created. Explicit ``index_prefixes`` still reach any mapped index.
    """
    return [prefix for prefix, _ in _configured(config)]


def _configured(config: Config) -> list[tuple[str, IndexArtifact]]:
    built = artifacts_to_build(config)
    return [
        (prefix, artifact)
        for prefix, artifact in _prefix_artifacts(config)
        if artifact in built
    ]


def held_artifacts_of(
    metadata: Mapping[str, Any] | None,
) -> frozenset[IndexArtifact] | None:
    """The artifacts ``SearchQuery.metadata`` says the namespace holds
    (``HELD_ARTIFACTS_KEY``), or ``None`` when it does not say (a caller
    that drives a strategy directly)."""
    values = (metadata or {}).get(HELD_ARTIFACTS_KEY)
    return parse_artifacts(values) if isinstance(values, list) else None


def held_index_prefixes(config: Config, held: Collection[IndexArtifact]) -> list[str]:
    """Of ``configured_index_prefixes``, those whose artifact the namespace
    holds (``held``), so a sweep never searches an index it lacks."""
    return [prefix for prefix, artifact in _configured(config) if artifact in held]


def readable_artifacts(
    config: Config, held: Collection[IndexArtifact] | None
) -> frozenset[IndexArtifact]:
    """The artifacts a query reads: those a run of the config builds
    (``artifacts_to_build``) that the queried namespace holds (``held``,
    ``held_artifacts_of``; all of them when ``None``). A strategy reads an
    optional index (claims, the relationship vectors, the reports) only
    when it is among these, as the default sweep does."""
    built = artifacts_to_build(config)
    return built if held is None else built & frozenset(held)


def readable_index_prefixes(
    config: Config, prefixes: list[str] | None, held: Collection[IndexArtifact]
) -> list[str]:
    """Of the index ``prefixes`` a strategy reads (every configured one when
    ``None``), those whose artifact the namespace holds (``held``): the
    indexes a filter key must be declared on. A prefix of no known
    artifact is kept."""
    if prefixes is None:
        return held_index_prefixes(config, held)
    o = config.indexing.opensearch
    artifacts = dict(_prefix_artifacts(config))
    artifacts[o.summary_nodes_index_prefix] = IndexArtifact.SUMMARY_TREE
    return [
        prefix
        for prefix in prefixes
        if prefix not in artifacts or artifacts[prefix] in held
    ]
