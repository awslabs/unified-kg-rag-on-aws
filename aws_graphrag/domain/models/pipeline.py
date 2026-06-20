# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from .community import Community
from .community_report import CommunityReport
from .config import PipelineConfig
from .covariate import Claim
from .document import Document, DocumentDelta
from .entity import Entity
from .relationship import Relationship
from .text_unit import TextUnit


class PipelineStageStatus(str, Enum):
    CACHED = "cached"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    RUNNING = "running"
    SKIPPED = "skipped"


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
    gleaning_improvement_rate: float = Field(
        default=0.0, description="Improvement rate from gleaning process"
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
    cache_path: Path | None = Field(
        default=None, description="Path to cached stage results"
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
    entities: "list[Entity]" = Field(
        default_factory=list, description="Entities extracted from text units."
    )
    relationships: "list[Relationship]" = Field(
        default_factory=list, description="Relationships extracted between entities."
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
    graph_statistics: Any = Field(
        default=None,
        description="Statistical metrics and analysis of the knowledge graph.",
    )
    centrality_metrics: list[Any] = Field(
        default_factory=list,
        description="Node centrality metrics from the graph analysis.",
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

    class Config:
        arbitrary_types_allowed = True

    def add_stage_result(self, result: PipelineStageResult) -> None:
        self.stage_results = [
            r for r in self.stage_results if r.stage_name != result.stage_name
        ]
        self.stage_results.append(result)

    def get_stage_result(self, stage_name: str) -> PipelineStageResult | None:
        for result in self.stage_results:
            if result.stage_name == stage_name:
                return result
        return None
