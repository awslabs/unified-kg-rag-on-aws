# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .community import Community
from .community_report import CommunityReport
from .config import PipelineConfig
from .covariate import Claim
from .document import Document, DocumentDelta, ExtractionSettings
from .entity import Entity
from .extraction import TextUnitExtraction
from .relationship import Relationship
from .text_unit import TextUnit


class PipelineStageStatus(str, Enum):
    CACHED = "cached"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    RUNNING = "running"
    SKIPPED = "skipped"


class IndexBuildRecord(BaseModel):
    """What built an index suffix: recorded by ingestion, read by evaluation.

    The vector store keeps the record of the latest run that wrote a suffix
    (its index mappings' ``_meta``), so a run manifest can name the index it
    was scored against. An incremental record carries the full build it was
    applied on top of (``full_build``), so the corpus size of that build
    survives the incremental runs after it.
    """

    pipeline_id: str = Field(description="The ingestion run that wrote the index.")
    mode: str = Field(
        description="'full' (the suffix was rebuilt) or 'incremental' (a delta)."
    )
    built_at: datetime = Field(description="When the run finished indexing (UTC).")
    documents: int = Field(
        description=(
            "Distinct documents whose text units the run indexed (an "
            "incremental run counts only its new and changed documents)."
        )
    )
    extraction_fingerprint: str = Field(
        description="Fingerprint of the extraction settings (cache_keys)."
    )
    extraction_settings: ExtractionSettings | None = Field(
        default=None,
        description="The extraction settings themselves, field by field.",
    )
    package_version: str | None = Field(
        default=None, description="Version of the package that ran ingestion."
    )
    full_build: "IndexBuildRecord | None" = Field(
        default=None,
        description=(
            "For an incremental record, the full build of the suffix it was "
            "applied on top of (without its extraction settings); None for a "
            "full build or when no full build was recorded."
        ),
    )

    def over(self, previous: Mapping[str, Any] | None) -> "IndexBuildRecord":
        """This record as it replaces ``previous`` (the stored record's dump):
        an incremental record keeps the full build ``previous`` is or keeps."""
        if self.mode != "incremental" or not previous:
            return self
        try:
            stored = IndexBuildRecord.model_validate(previous)
        except ValidationError:
            return self
        base = stored if stored.mode == "full" else stored.full_build
        if base is None:
            return self
        return self.model_copy(
            update={
                "full_build": base.model_copy(
                    update={"extraction_settings": None, "full_build": None}
                )
            }
        )


class PipelineMetrics(BaseModel):
    pipeline_id: str = Field(description="Unique identifier for the pipeline run")
    total_duration_seconds: float = Field(
        description="Total pipeline execution time in seconds"
    )
    total_documents_processed: int = Field(
        description="Total number of documents processed"
    )
    total_text_units_created: int = Field(
        description="Total number of text units created"
    )
    total_translated_units: int = Field(
        description="Total number of translated text units"
    )
    total_entities_extracted: int = Field(
        description="Total number of entities extracted"
    )
    total_relationships_extracted: int = Field(
        description="Total number of relationships extracted"
    )
    total_claims_extracted: int = Field(description="Total number of claims extracted")
    total_communities_detected: int = Field(
        description="Total number of communities detected"
    )
    total_community_reports_generated: int = Field(
        description="Total number of community reports generated"
    )
    total_items_indexed: int = Field(
        default=0,
        description="Total artifacts successfully written to the stores (indexing stage)",
    )
    total_items_index_failed: int = Field(
        default=0,
        description="Total artifacts that failed to index across all backends",
    )
    relationships_indexed: int = Field(
        default=0,
        description="Relationships successfully indexed (0 while extracted>0 signals "
        "the silent-drop failure mode)",
    )
    community_drift_overlap_share: float = Field(
        default=0.0,
        description="Incremental runs: share of the entities the run clustered "
        "that also belong to an earlier level-0 community",
    )
    community_rebuild_recommended: int = Field(
        default=0,
        description="1 when community_drift_overlap_share reached "
        "indexing.community_drift_threshold (rebuild with indexing.reset)",
    )
    total_extraction_failures: int = Field(
        default=0,
        description="Text units whose graph extraction failed (LLM error or "
        "unparseable output); the run still succeeds without their entities",
    )
    total_gleaning_failures: int = Field(
        default=0,
        description="Text-unit refinements that failed across all gleaning rounds",
    )
    total_claim_extraction_failures: int = Field(
        default=0, description="Text units whose claim extraction failed"
    )
    total_documents_quarantined: int = Field(
        default=0,
        description="Incremental runs: documents quarantined this run, their "
        "processing having killed the run indexing.max_document_attempts "
        "consecutive runs, the last one processing it alone",
    )
    total_documents_deferred: int = Field(
        default=0,
        description="Incremental runs: new or changed documents left for a "
        "later run while the run isolates the document that keeps killing "
        "runs (it processes half of the suspects)",
    )
    total_records_not_stored: int = Field(
        default=0,
        description="Extraction records graph resolution and claim extraction "
        "could not store; their documents are recorded FAILED and re-extracted",
    )
    underived_items: int = Field(
        default=0,
        description="Incremental runs: stored items whose derived attributes "
        "the indexing stage could not recompute (they keep their stored ones)",
    )
    gleaning_improvement_rate: float = Field(
        default=0.0,
        description="Entities plus relationships gleaning added, per entity "
        "and relationship extracted before it",
    )
    entity_resolution_merge_rate: float = Field(
        default=0.0, description="Rate of entity merging during resolution"
    )
    relationship_resolution_merge_rate: float = Field(
        default=0.0, description="Rate of relationship merging during resolution"
    )
    claim_resolution_merge_rate: float = Field(
        default=0.0, description="Rate of claim merging during resolution"
    )
    community_modularity_score: float = Field(
        default=0.0, description="Modularity score of detected communities"
    )
    stage_durations: dict[str, float] = Field(
        default_factory=dict, description="Duration of each pipeline stage in seconds"
    )
    stage_throughput: dict[str, float] = Field(
        default_factory=dict, description="Throughput metrics for each stage"
    )
    cache_hit_rate: float = Field(
        default=0.0, description="Cache hit rate as a percentage"
    )
    cache_size_mb: float = Field(default=0.0, description="Total cache size in MB")


class PipelineStageResult(BaseModel):
    stage_name: str = Field(description="Name of the pipeline stage")
    status: PipelineStageStatus = Field(description="Current status of the stage")
    start_time: datetime = Field(description="When the stage started execution")
    end_time: datetime | None = Field(
        default=None, description="When the stage finished execution"
    )
    duration_seconds: float | None = Field(
        default=None, description="Duration of stage execution in seconds"
    )
    input_count: int = Field(default=0, description="Number of input items processed")
    output_count: int = Field(default=0, description="Number of output items generated")
    error_message: str | None = Field(
        default=None, description="Error message if stage failed"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Stage-specific performance metrics"
    )


class PipelineContext(BaseModel):
    pipeline_id: str = Field(
        description="A unique identifier for the current pipeline run."
    )
    config: PipelineConfig = Field(
        description="The configuration settings for the current run."
    )
    status: PipelineStageStatus = Field(
        description="The current execution status of the pipeline."
    )
    start_time: datetime = Field(
        description="The timestamp when the pipeline run began."
    )
    end_time: datetime | None = Field(
        default=None, description="The timestamp when the pipeline run concluded."
    )
    duration_seconds: float | None = Field(
        default=None,
        description="The total duration of the pipeline execution in seconds.",
    )
    source_directory: Path = Field(
        description="The source directory containing input documents."
    )
    documents: "list[Document]" = Field(
        default_factory=list, description="Documents loaded from the source directory."
    )
    text_units: "list[TextUnit]" = Field(
        default_factory=list, description="Text units created by chunking documents."
    )
    translated_units: "list[TextUnit]" = Field(
        default_factory=list, description="Text units after translation."
    )
    unit_extractions: "list[TextUnitExtraction]" = Field(
        default_factory=list,
        description="Each text unit's extraction record (its own entities and "
        "relationships), from graph extraction and gleaning; graph resolution "
        "merges them.",
    )
    claims: "list[Claim]" = Field(
        default_factory=list, description="Claims and covariates extracted from text."
    )
    resolved_entities: "list[Entity]" = Field(
        default_factory=list, description="Entities after resolution and deduplication."
    )
    resolved_relationships: "list[Relationship]" = Field(
        default_factory=list, description="Relationships after entity resolution."
    )
    resolved_claims: "list[Claim]" = Field(
        default_factory=list, description="Claims after entity resolution."
    )
    knowledge_graph: nx.Graph | None = Field(
        default=None,
        description="The final graph of entities, relationships, and claims.",
    )
    communities: "list[Community]" = Field(
        default_factory=list,
        description="Communities detected within the knowledge graph.",
    )
    community_reports: "list[CommunityReport]" = Field(
        default_factory=list, description="Human-readable reports for each community."
    )
    stage_results: list[PipelineStageResult] = Field(
        default_factory=list,
        description="A collection of results from each executed stage.",
    )
    global_metrics: PipelineMetrics | None = Field(
        default=None,
        description="Global metrics and statistics for the entire pipeline run.",
    )
    incremental_delta: DocumentDelta | None = Field(
        default=None,
        description="Corpus delta (new/changed/unchanged/deleted) when incremental "
        "indexing is active; drives delta upsert + stale-artifact pruning.",
    )
    incremental_fingerprints: dict[str, str] = Field(
        default_factory=dict,
        description="doc_id -> content hash for the current corpus, recorded to the "
        "doc-status registry after a successful incremental commit.",
    )
    incremental_scope: str | None = Field(
        default=None,
        description="Registry scope (index namespace + corpus source) the delta "
        "was computed in; recorded on every committed document.",
    )
    incremental_extraction_fingerprint: str | None = Field(
        default=None,
        description="Fingerprint of the run's extraction settings the delta "
        "was computed with; recorded on every committed document so a later "
        "run with other settings re-extracts it.",
    )
    incremental_extraction_settings: ExtractionSettings | None = Field(
        default=None,
        description="The extraction settings the delta was computed with "
        "(cache_keys.extraction_settings): recorded for the registry scope at "
        "commit, and compared field by field when the run resumes.",
    )
    incremental_retired_scopes: list[str] | None = Field(
        default=None,
        description="The indexing.retire_source_scopes values the delta "
        "retired (their records are in its deleted set). A run resumed after "
        "document_loading reuses the delta, so it warns about configured "
        "scopes missing here. None when unknown: a context saved before this "
        "field existed.",
    )
    failed_source_files: list[str] = Field(
        default_factory=list,
        description="Corpus files that failed to parse or load this run. "
        "Incremental delta detection keeps them out of the deleted set.",
    )
    failed_text_unit_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Ids of the text units an extraction stage failed on, by "
        "stage name. An incremental commit records their documents as FAILED "
        "so the next run re-extracts them.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_stage_result(self, result: PipelineStageResult) -> None:
        self.stage_results = [
            r for r in self.stage_results if r.stage_name != result.stage_name
        ]
        self.stage_results.append(result)
