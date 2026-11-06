# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which ingestion stages build what the selected methodologies read, and
what an index namespace holds (docs/design.md#46-methodology-profiles).

Pure: the stage map and the derivations from a config, a run's completed
stages and a stored build record.
"""

from collections.abc import Collection, Iterable, Mapping
from typing import Any

from unified_kg_rag.domain.models import Config, PipelineStageType
from unified_kg_rag.domain.models.methodology import (
    METHODOLOGY_PROFILES,
    UNBUILT_ARTIFACTS,
    IndexArtifact,
    Methodology,
    artifacts_of,
    methodologies_holding,
    parse_artifacts,
)

_S = PipelineStageType
_A = IndexArtifact

# The stages each artifact needs run, beyond those they read
# (STAGE_DEPENDENCIES). Translation and gleaning ride with extraction; their
# config sections switch them.
ARTIFACT_STAGES: dict[IndexArtifact, frozenset[PipelineStageType]] = {
    _A.TEXT_UNITS: frozenset({_S.TEXT_CHUNKING, _S.INDEXING}),
    _A.ENTITIES: frozenset({_S.GRAPH_RESOLUTION, _S.INDEXING}),
    _A.RELATIONSHIPS: frozenset({_S.GRAPH_RESOLUTION, _S.INDEXING}),
    _A.RELATIONSHIP_VECTORS: frozenset({_S.GRAPH_RESOLUTION, _S.INDEXING}),
    _A.GRAPH: frozenset({_S.GRAPH_RESOLUTION, _S.GRAPH_ANALYSIS, _S.INDEXING}),
    _A.COMMUNITIES: frozenset({_S.COMMUNITY_DETECTION, _S.INDEXING}),
    _A.COMMUNITY_REPORTS: frozenset({_S.COMMUNITY_DETECTION, _S.INDEXING}),
    _A.CLAIMS: frozenset({_S.CLAIM_RESOLUTION, _S.INDEXING}),
    # Built from the stored text units after indexing (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).
    _A.SUMMARY_TREE: frozenset({_S.SUMMARY_TREE}),
}

# The stage whose completion shows a run built the artifact (indexing, which
# writes them all, is the stage asking).
ARTIFACT_PRODUCERS: dict[IndexArtifact, PipelineStageType] = {
    _A.TEXT_UNITS: _S.TEXT_CHUNKING,
    _A.ENTITIES: _S.GRAPH_RESOLUTION,
    _A.RELATIONSHIPS: _S.GRAPH_RESOLUTION,
    _A.RELATIONSHIP_VECTORS: _S.GRAPH_RESOLUTION,
    _A.GRAPH: _S.GRAPH_RESOLUTION,
    _A.COMMUNITIES: _S.COMMUNITY_DETECTION,
    _A.COMMUNITY_REPORTS: _S.COMMUNITY_DETECTION,
    _A.CLAIMS: _S.CLAIM_RESOLUTION,
    _A.SUMMARY_TREE: _S.SUMMARY_TREE,
}

# The stages whose output each stage reads (optional ones included: their
# config sections switch them off).
STAGE_DEPENDENCIES: dict[PipelineStageType, frozenset[PipelineStageType]] = {
    _S.DOCUMENT_PARSING: frozenset(),
    _S.DOCUMENT_LOADING: frozenset({_S.DOCUMENT_PARSING}),
    _S.TEXT_CHUNKING: frozenset({_S.DOCUMENT_LOADING}),
    _S.TRANSLATION: frozenset({_S.TEXT_CHUNKING}),
    _S.GRAPH_EXTRACTION: frozenset({_S.TEXT_CHUNKING, _S.TRANSLATION}),
    _S.GLEANING: frozenset({_S.GRAPH_EXTRACTION}),
    _S.GRAPH_RESOLUTION: frozenset({_S.GRAPH_EXTRACTION, _S.GLEANING}),
    _S.CLAIM_EXTRACTION: frozenset({_S.GRAPH_RESOLUTION}),
    _S.CLAIM_RESOLUTION: frozenset({_S.CLAIM_EXTRACTION}),
    _S.GRAPH_ANALYSIS: frozenset({_S.GRAPH_RESOLUTION}),
    _S.COMMUNITY_DETECTION: frozenset({_S.GRAPH_RESOLUTION, _S.GRAPH_ANALYSIS}),
    _S.INDEXING: frozenset({_S.TEXT_CHUNKING}),
    # Reads the text units indexing stored (not the run's context),
    # translated when translation is on: the tree is written in the index
    # language.
    _S.SUMMARY_TREE: frozenset({_S.INDEXING, _S.TRANSLATION}),
}

# Stages the artifacts are built without: pre-parsed documents need no
# parsing, and translation and gleaning refine what extraction reads and
# states.
OPTIONAL_STAGES: frozenset[PipelineStageType] = frozenset(
    {_S.DOCUMENT_PARSING, _S.TRANSLATION, _S.GLEANING}
)

# Artifacts an incremental run cannot add to a namespace that lacks them: it
# builds them for its delta only (communities cluster the run's graph, the
# relationship vectors embed the run's relationships). Adding one rebuilds
# the namespace from its stored extractions instead. Not the summary tree:
# its stage builds it over every stored text unit on any run.
CORPUS_WIDE_ARTIFACTS: frozenset[IndexArtifact] = frozenset(
    {
        _A.RELATIONSHIP_VECTORS,
        _A.COMMUNITIES,
        _A.COMMUNITY_REPORTS,
    }
)


# The artifacts the graph store holds (entity vertices and relationship
# edges, community vertices); the others are the vector store's.
GRAPH_STORE_ARTIFACTS: frozenset[IndexArtifact] = frozenset({_A.GRAPH, _A.COMMUNITIES})


def stages_for(artifacts: Iterable[IndexArtifact]) -> frozenset[PipelineStageType]:
    """The stages that build ``artifacts``, with every stage they read."""
    pending = [
        stage
        for artifact in artifacts
        if artifact not in UNBUILT_ARTIFACTS
        for stage in ARTIFACT_STAGES[artifact]
    ]
    stages: set[PipelineStageType] = set()
    while pending:
        stage = pending.pop()
        if stage not in stages:
            stages.add(stage)
            pending.extend(STAGE_DEPENDENCIES[stage])
    return frozenset(stages)


def artifacts_to_build(config: Config) -> frozenset[IndexArtifact]:
    """What a run of ``config`` builds: every artifact its effective
    methodologies read whose switch is on, and claims when
    ``processing.claim_extraction`` is on (per-document extraction output,
    which its switch alone governs). Community reports are switched off by
    ``report_generation.enabled`` too: a run never builds them then, so an
    incremental run does not take them for missing (and rebuild)."""
    artifacts = set(artifacts_of(config.effective_methodologies))
    if config.processing.claim_extraction.enabled:
        artifacts.add(_A.CLAIMS)
    else:
        artifacts.discard(_A.CLAIMS)
    if not config.indexing.opensearch.build_relationship_vector_index:
        artifacts.discard(_A.RELATIONSHIP_VECTORS)
    if not config.graph.community_detection.enabled:
        artifacts -= {_A.COMMUNITIES, _A.COMMUNITY_REPORTS}
    if not config.graph.community_detection.report_generation.enabled:
        artifacts.discard(_A.COMMUNITY_REPORTS)
    return frozenset(artifacts)


def writes_graph_store(config: Config) -> bool:
    """Whether a run of ``config`` builds an artifact the graph store holds
    (not with ``raptor`` alone): without one, it needs no graph store."""
    return bool(artifacts_to_build(config) & GRAPH_STORE_ARTIFACTS)


def selected_stages(config: Config) -> frozenset[PipelineStageType]:
    """The stages a run of ``config`` needs (``stages_for`` what it builds).
    The default methodologies need every stage."""
    return stages_for(artifacts_to_build(config))


def stage_methodologies(
    config: Config, stage: PipelineStageType
) -> tuple[Methodology, ...]:
    """The effective methodologies whose required artifacts need ``stage``
    (none for a stage they are built without, ``OPTIONAL_STAGES``)."""
    if stage in OPTIONAL_STAGES:
        return ()
    return tuple(
        m
        for m in config.effective_methodologies
        if stage in stages_for(METHODOLOGY_PROFILES[m].required)
    )


def built_artifacts(
    config: Config, completed: Collection[PipelineStageType]
) -> frozenset[IndexArtifact]:
    """The artifacts a run built: those of ``artifacts_to_build`` whose
    producing stage is in ``completed`` (the run's or its earlier phases')."""
    return frozenset(
        artifact
        for artifact in artifacts_to_build(config)
        if artifact in ARTIFACT_PRODUCERS and ARTIFACT_PRODUCERS[artifact] in completed
    )


def stored_artifacts(
    record: Mapping[str, Any] | None,
) -> frozenset[IndexArtifact] | None:
    """The artifacts a stored build record (an ``IndexBuildRecord`` dump)
    names, or ``None`` when there is none or it predates them."""
    if not isinstance(record, Mapping):
        return None
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    return parse_artifacts(artifacts)


def held_artifacts(
    record: Mapping[str, Any] | None, config: Config
) -> frozenset[IndexArtifact]:
    """The artifacts an index namespace holds: those its build record names;
    without a record that names them (none written, a backend without index
    metadata, a record from before methodology profiles) those a run of
    ``config`` builds (``artifacts_to_build``)."""
    artifacts = stored_artifacts(record)
    return artifacts_to_build(config) if artifacts is None else artifacts


def held_methodologies(
    record: Mapping[str, Any] | None, config: Config
) -> tuple[Methodology, ...]:
    """The methodologies an index namespace holds: those whose required
    artifacts it holds (``held_artifacts``): a store the default graphrag +
    lightrag run built before build records holds hipporag too, which reads
    what lightrag does."""
    return methodologies_holding(held_artifacts(record, config))


def missing_artifacts(
    record: Mapping[str, Any] | None, config: Config
) -> frozenset[IndexArtifact]:
    """The corpus-wide artifacts a run of ``config`` builds that the
    namespace's build record does not name (none without a record that names
    its artifacts): an incremental run adding them rebuilds the namespace."""
    stored = stored_artifacts(record)
    if stored is None:
        return frozenset()
    return (artifacts_to_build(config) & CORPUS_WIDE_ARTIFACTS) - stored
