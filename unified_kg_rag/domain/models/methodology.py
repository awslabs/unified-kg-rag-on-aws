# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The retrieval methodologies an index namespace can hold, and the index
artifacts each one reads (docs/design.md#46-methodology-profiles).

``indexing.methodologies`` selects which methodologies a run builds; the
stages that build each artifact are mapped in
``domain.ingestion.methodologies``. Every search strategy declares the
methodology it reads (``BaseSearchStrategy.methodology``), and a query whose
strategy's methodology the namespace does not hold fails fast. Pure: no
config, stage or store import.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Methodology(str, Enum):
    """A retrieval methodology: the index artifacts its strategies read."""

    GRAPHRAG = "graphrag"
    LIGHTRAG = "lightrag"
    HIPPORAG = "hipporag"
    RAPTOR = "raptor"


class IndexArtifact(str, Enum):
    """What an ingestion run stores for a methodology to read."""

    TEXT_UNITS = "text_units"
    ENTITIES = "entities"
    RELATIONSHIPS = "relationships"
    # The relationships' vector-store documents (embedded descriptions).
    RELATIONSHIP_VECTORS = "relationship_vectors"
    # The graph store's entity vertices and relationship edges.
    GRAPH = "graph"
    COMMUNITIES = "communities"
    COMMUNITY_REPORTS = "community_reports"
    CLAIMS = "claims"
    # RAPTOR's recursive summaries over the text units.
    SUMMARY_TREE = "summary_tree"


@dataclass(frozen=True)
class MethodologyProfile:
    """The artifacts a methodology needs (``required``: its strategies are
    refused without them) and those it reads when built (``optional``)."""

    required: frozenset[IndexArtifact]
    optional: frozenset[IndexArtifact] = frozenset()

    @property
    def artifacts(self) -> frozenset[IndexArtifact]:
        return self.required | self.optional


_A = IndexArtifact
_GRAPH_CORE = frozenset({_A.TEXT_UNITS, _A.ENTITIES, _A.RELATIONSHIPS, _A.GRAPH})

METHODOLOGY_PROFILES: dict[Methodology, MethodologyProfile] = {
    # Local reads the relationship vectors and claims when they are built
    # (its relationship section, bridge relationships); global and drift
    # answer from the community reports.
    Methodology.GRAPHRAG: MethodologyProfile(
        required=_GRAPH_CORE | {_A.COMMUNITIES, _A.COMMUNITY_REPORTS},
        optional=frozenset({_A.RELATIONSHIP_VECTORS, _A.CLAIMS}),
    ),
    # High-level keywords search the relationship vectors.
    Methodology.LIGHTRAG: MethodologyProfile(
        required=_GRAPH_CORE | {_A.RELATIONSHIP_VECTORS}
    ),
    # Personalized PageRank seeded from relationship and passage hits.
    Methodology.HIPPORAG: MethodologyProfile(
        required=_GRAPH_CORE | {_A.RELATIONSHIP_VECTORS}
    ),
    Methodology.RAPTOR: MethodologyProfile(
        required=frozenset({_A.TEXT_UNITS, _A.SUMMARY_TREE})
    ),
}

# What indexing.methodologies selects by default: what one ingestion built
# before methodologies were selectable.
DEFAULT_METHODOLOGIES: tuple[Methodology, ...] = (
    Methodology.GRAPHRAG,
    Methodology.LIGHTRAG,
)

# Artifacts no ingestion stage of this version builds: a methodology that
# requires one cannot be selected yet.
UNBUILT_ARTIFACTS: frozenset[IndexArtifact] = frozenset({IndexArtifact.SUMMARY_TREE})


def available_methodologies() -> tuple[Methodology, ...]:
    """The methodologies this version can build, in declaration order."""
    return tuple(
        m
        for m in Methodology
        if not METHODOLOGY_PROFILES[m].required & UNBUILT_ARTIFACTS
    )


def artifacts_of(methodologies: Iterable[Methodology]) -> frozenset[IndexArtifact]:
    """Every artifact (required or optional) the methodologies read."""
    return frozenset().union(
        *(METHODOLOGY_PROFILES[m].artifacts for m in methodologies)
    )


def methodologies_holding(
    artifacts: Iterable[IndexArtifact],
) -> tuple[Methodology, ...]:
    """The methodologies whose required artifacts are all in ``artifacts``."""
    held = frozenset(artifacts)
    return tuple(m for m in Methodology if METHODOLOGY_PROFILES[m].required <= held)


def parse_artifacts(values: Iterable[object]) -> frozenset[IndexArtifact]:
    """The known artifacts among ``values`` (a stored record's names): one a
    later version added is skipped, not an error."""
    known = {a.value: a for a in IndexArtifact}
    return frozenset(known[v] for v in values if isinstance(v, str) and v in known)


def switched_off_artifacts(
    *, communities: bool, relationship_vectors: bool
) -> frozenset[IndexArtifact]:
    """The artifacts a configuration switch turns off:
    ``graph.community_detection.enabled`` (``communities``) and
    ``indexing.opensearch.build_relationship_vector_index``
    (``relationship_vectors``)."""
    off: set[IndexArtifact] = set()
    if not communities:
        off |= {IndexArtifact.COMMUNITIES, IndexArtifact.COMMUNITY_REPORTS}
    if not relationship_vectors:
        off.add(IndexArtifact.RELATIONSHIP_VECTORS)
    return frozenset(off)


def buildable(
    selected: Iterable[Methodology], switched_off: frozenset[IndexArtifact]
) -> tuple[Methodology, ...]:
    """The ``selected`` methodologies none of whose required artifacts is in
    ``switched_off``, in their order."""
    return tuple(
        m for m in selected if not METHODOLOGY_PROFILES[m].required & switched_off
    )
