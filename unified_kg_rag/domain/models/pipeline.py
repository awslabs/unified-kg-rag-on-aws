# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from unified_kg_rag.shared.exceptions import ErrorClass

from .community import Community
from .community_report import CommunityReport
from .config import PipelineConfig
from .covariate import Claim
from .document import Document, DocumentDelta, ExtractionSettings
from .entity import Entity
from .extraction import TextUnitExtraction
from .methodology import (
    IndexArtifact,
    Methodology,
    methodologies_holding,
    recorded_artifacts,
)
from .relationship import Relationship
from .text_unit import TextUnit

# The run's metadata file in its stage-cache directory (the stage results and
# context); the S3 sync uploads it last.
PIPELINE_METADATA_FILE = "pipeline_metadata.json"


class PipelineStageStatus(str, Enum):
    CACHED = "cached"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    RUNNING = "running"
    SKIPPED = "skipped"


class CommunityCounters(BaseModel):
    """How far incremental commits moved a suffix's communities from the last
    full clustering (``indexing.community_compaction``).

    A full build clusters the whole graph and starts the counters; each
    incremental commit appends the communities of its delta and deletes
    those citing only removed text units, and adds one run.
    """

    clustered_at: datetime = Field(
        description="When the full build that last clustered the whole graph "
        "finished indexing (UTC)."
    )
    communities_at_clustering: int = Field(
        ge=0, description="Communities (all levels) that full build wrote."
    )
    communities_now: int = Field(
        ge=0,
        description="communities_at_clustering plus the communities each "
        "incremental commit since appended, minus those its removals "
        "deleted. An upper bound: a delta community written under a stored "
        "community's id counts again.",
    )
    runs_since: int = Field(
        ge=0, description="Incremental commits since that full build."
    )
    last_commit: str | None = Field(
        default=None,
        description="The incremental commit last counted (see "
        "CommunityChange.commit): a resumed run that writes the record again "
        "does not count its commit twice.",
    )

    @property
    def growth_ratio(self) -> float | None:
        """``communities_now / communities_at_clustering`` (None without a
        clustered community)."""
        if not self.communities_at_clustering:
            return None
        return round(self.communities_now / self.communities_at_clustering, 4)


class CommunityChange(BaseModel):
    """An incremental commit's change to a suffix's communities, which
    ``IndexBuildRecord.over`` adds to the stored counters."""

    commit: str = Field(
        description="Identity of the commit: the run's pipeline id and delta "
        "(a resumed run commits the same delta again)."
    )
    added: int = Field(ge=0, description="Communities the commit appended.")
    deleted: int = Field(
        ge=0, description="Stored communities the commit's removals deleted."
    )


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
            "incremental run counts only its new and changed documents; one "
            "that rewrote a missing record, those the registry records)."
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
    communities: CommunityCounters | None = Field(
        default=None,
        description=(
            "The suffix's community counters since its last full build; None "
            "until a full build recorded them (a record written by an earlier "
            "version starts them at the next full build)."
        ),
    )
    artifacts: list[IndexArtifact] | None = Field(
        default=None,
        description=(
            "The index artifacts the suffix holds and its runs maintain "
            "(docs/design.md#46-methodology-profiles): a full build's are "
            "those it built, an incremental record keeps those both it and "
            "the stored record name. None in a record from before "
            "methodology profiles."
        ),
    )
    methodologies: list[Methodology] | None = Field(
        default=None,
        description=(
            "The methodologies whose required artifacts the suffix holds "
            "(derived from artifacts when the record is written)."
        ),
    )
    source_scopes: list[str] | None = Field(
        default=None,
        description=(
            "The source scopes whose documents the suffix held when the "
            "record was written, by the doc-status registry (the run's "
            "and the others its diff read). Its methodologies hold for all "
            "of them. None without the registry and in a record written "
            "by an earlier version."
        ),
    )
    visualization_export_run: str | None = Field(
        default=None,
        description=(
            "The ingestion run that exports this build's visualization data "
            "after it (graph.visualization.export_data), into its cache "
            "directory unless outputs_directory is set: a later run that "
            "writes nothing keeps that export, uploaded or local, instead of "
            "writing it again. None when the run exports none."
        ),
    )
    community_change: CommunityChange | None = Field(
        default=None,
        exclude=True,
        description=(
            "Incremental record, not stored: the commit's change, which "
            "over() adds to the stored counters."
        ),
    )

    def over(self, previous: Mapping[str, Any] | None) -> "IndexBuildRecord":
        """This record as it replaces ``previous`` (the stored record's dump):
        an incremental record keeps the full build ``previous`` is or keeps,
        and its community counters with this commit's change added.

        A full record with a ``community_change`` was written by a delta
        (the registry knew none of the scope's documents, so the suffix may
        hold what other scopes wrote): it adds to the stored counters too,
        and starts its own only where none are stored.
        """
        if not previous or (
            self.mode != "incremental" and self.community_change is None
        ):
            return self
        try:
            stored = IndexBuildRecord.model_validate(previous)
        except ValidationError:
            return self
        if self.mode != "incremental":
            if stored.communities is None:
                return self
            return self.model_copy(
                update={"communities": self._counted(stored.communities)}
            )
        update: dict[str, Any] = {
            "communities": self._counted(stored.communities),
            **self._kept_artifacts(stored.artifacts),
        }
        base = stored if stored.mode == "full" else stored.full_build
        if base is not None:
            update["full_build"] = base.model_copy(
                update={
                    "extraction_settings": None,
                    "full_build": None,
                    "communities": None,
                }
            )
        return self.model_copy(update=update)

    def with_artifacts(self, artifacts: Iterable[IndexArtifact]) -> "IndexBuildRecord":
        """This record naming ``artifacts`` and the methodologies they hold."""
        ordered = [a for a in IndexArtifact if a in set(artifacts)]
        return self.model_copy(
            update={
                "artifacts": ordered,
                "methodologies": list(methodologies_holding(ordered)),
            }
        )

    def _kept_artifacts(self, stored: list[IndexArtifact] | None) -> dict[str, Any]:
        """An incremental record's artifacts over the ``stored`` ones: those
        both name (one this run did not maintain is stale for its delta).
        Nothing to change when either names none."""
        if self.artifacts is None or stored is None:
            return {}
        kept = self.with_artifacts(
            recorded_artifacts(self.artifacts, stored, incremental=True)
        )
        return {"artifacts": kept.artifacts, "methodologies": kept.methodologies}

    def _counted(self, counters: CommunityCounters | None) -> CommunityCounters | None:
        """``counters`` with this commit's change added, once per commit."""
        change = self.community_change
        if counters is None or change is None or change.commit == counters.last_commit:
            return counters
        return counters.model_copy(
            update={
                "communities_now": max(
                    0, counters.communities_now + change.added - change.deleted
                ),
                "runs_since": counters.runs_since + 1,
                "last_commit": change.commit,
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
    community_growth_ratio: float = Field(
        default=0.0,
        description="Runs that recorded community counters: the stored "
        "communities over those the last full build clustered (the largest "
        "of the run's suffixes; 0 without communities)",
    )
    runs_since_clustering: int = Field(
        default=0,
        description="Runs that recorded community counters: incremental "
        "commits since the last full build clustered the graph (the largest "
        "of the run's suffixes)",
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
    total_unrecognised_claim_statuses: int = Field(
        default=0,
        description="Extracted claims whose status the model wrote outside the "
        "words the claim reader knows (stored as unknown)",
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
    total_documents_excluded: int = Field(
        default=0,
        description="Incremental runs: changed documents not retried, each "
        "having failed indexing.max_document_failures consecutive runs for "
        "an error of its own (kept FAILED)",
    )
    total_files_failed: int = Field(
        default=0,
        description="Corpus files that failed to parse or load; an incremental "
        "run keeps what they indexed before",
    )
    total_files_skipped: int = Field(
        default=0,
        description="Corpus files skipped for a file type no loader reads (an "
        "optional extra not installed, a loader plugin not imported); an "
        "incremental run keeps what they indexed before",
    )
    total_documents_deleted: int = Field(
        default=0,
        description="Incremental runs: documents gone from the corpus (or of a "
        "retired source scope) whose exclusive content and records the "
        "indexing stage removed",
    )
    total_documents_failed: int = Field(
        default=0,
        description="Documents the run recorded FAILED (retried next run); "
        "run-ingestion's JSON failed_documents",
    )
    total_units_failed: int = Field(
        default=0,
        description="Distinct text units a stage of the run failed on "
        "(extraction, gleaning, claims, translation, resolution, reports), "
        "indexed without that stage's output; run-ingestion's JSON "
        "units_failed. Counted in a full build too, which records no "
        "document FAILED",
    )
    total_documents_retrying: int = Field(
        default=0,
        description="Documents failed for an error not their own (an outage, a "
        "setting) that the run before failed too; run-ingestion's JSON "
        "documents_retrying",
    )
    total_documents_without_claims: int = Field(
        default=0,
        description="Documents indexed without the claims of some text units "
        "(their claim extraction failed for an error of their own); "
        "run-ingestion's JSON documents_without_claims",
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
    total_tree_summaries_failed: int = Field(
        default=0,
        description="Summaries of the RAPTOR summary tree that failed (their "
        "children moved up a level: the tree is partial until a run builds "
        "them again)",
    )
    total_tree_uncovered_leaves: int = Field(
        default=0,
        description="indexing.summary_tree.refresh: on_reset: stored text units "
        "no summary of the tree covers until a reset rebuilds it",
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
    error_class: ErrorClass | None = Field(
        default=None,
        description="The class of the failure (ErrorClass), its dominant "
        "cause's when its items failed (model_unreachable for a model server "
        "that could not be reached)",
    )
    error_type: str | None = Field(
        default=None, description="The class name of the error it failed with"
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
    member_summaries: "list[Entity]" = Field(
        default_factory=list,
        description="Summaries community detection made of stored members' "
        "descriptions on an incremental run, each marked with the description "
        "it was made from (summarized_from): the indexing reuses one when it "
        "derives that description again instead of summarizing it a second "
        "time. Not kept in the run metadata: a resumed indexing summarizes "
        "again.",
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
    incremental_loading_key: str | None = Field(
        default=None,
        description="Cache key of the document_loading output the delta was "
        "computed with. The delta is saved with the run's metadata, not with "
        "that output: a resume restores both only when they match.",
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
    skipped_source_files: list[str] = Field(
        default_factory=list,
        description="Corpus files the parsing stage skipped for a file type no "
        "loader reads (an optional extra not installed, a loader plugin not "
        "imported). Incremental delta detection keeps them out of the deleted "
        "set, as failed_source_files; unlike those, they do not make an empty "
        "corpus with seed graphs a failed run.",
    )
    failed_text_unit_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Ids of the text units an extraction stage failed on, by "
        "stage name. An incremental commit records their documents as FAILED "
        "so the next run re-extracts them.",
    )
    item_failed_text_unit_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description="The failed text units whose error was their own (a "
        "request the model rejects for its content, a response that does not "
        "parse or was cut off), by stage name: only their documents' failures "
        "count toward indexing.max_document_failures, including when such a "
        "stage fails the run, except claim extraction's: such a document "
        "whose other stages succeeded is indexed without those claims "
        "(incremental.run_failures). Any other error (an outage, "
        "credentials, a quota, a setting) counts for none.",
    )

    recorded_failed_doc_ids: list[str] = Field(
        default_factory=list,
        description="Registry ids of the documents this run recorded FAILED "
        "(the indexing commit, or a stage that stopped the run on errors of "
        "their own), not those indexed without some of their claims (recorded "
        "FAILED once so the next run extracts them again); the run's "
        "failed_documents count.",
    )

    retrying_doc_ids: list[str] = Field(
        default_factory=list,
        description="Registry ids of the documents this run failed for an "
        "error not their own (an outage or a setting, never counted toward "
        "exclusion) that had failed so the run before too "
        "(DocStatusRecord.retry_count >= 2); the run's documents_retrying "
        "count.",
    )

    visualization_data_path: str | None = Field(
        default=None,
        description="Where the run wrote the namespace's visualization_data.json "
        "(graph.visualization.export_data), read from the stores once it "
        "indexed; None when it wrote none.",
    )
    claims_missing_doc_ids: list[str] = Field(
        default_factory=list,
        description="Registry ids of the documents this run indexed without "
        "the claims of some text units (their claim extraction failed for an "
        "error of their own, counted for none); the run's "
        "documents_without_claims count.",
    )

    def failed_unit_ids(self) -> set[str]:
        """The text units a stage of this run failed on (every stage's
        :attr:`failed_text_unit_ids`)."""
        return {
            unit_id for ids in self.failed_text_unit_ids.values() for unit_id in ids
        }

    def record_failed_documents(self, doc_ids: Iterable[str]) -> None:
        """Add ``doc_ids`` to :attr:`recorded_failed_doc_ids`."""
        self.recorded_failed_doc_ids = sorted({*self.recorded_failed_doc_ids, *doc_ids})

    def record_claims_missing_documents(self, doc_ids: Iterable[str]) -> None:
        """Add ``doc_ids`` to :attr:`claims_missing_doc_ids`."""
        self.claims_missing_doc_ids = sorted({*self.claims_missing_doc_ids, *doc_ids})

    def record_retrying_documents(self, doc_ids: Iterable[str]) -> None:
        """Add ``doc_ids`` to :attr:`retrying_doc_ids`."""
        self.retrying_doc_ids = sorted({*self.retrying_doc_ids, *doc_ids})

    def documents_of_units(self, unit_ids: Iterable[str]) -> "list[Document]":
        """The run's documents the text units ``unit_ids`` belong to."""
        wanted = set(unit_ids)
        if not wanted:
            return []
        document_ids = {
            document_id
            for unit in self.translated_units or self.text_units or []
            if unit.id in wanted
            for document_id in unit.document_ids or []
        }
        return [doc for doc in self.documents if doc.document_id in document_ids]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_stage_result(self, result: PipelineStageResult) -> None:
        self.stage_results = [
            r for r in self.stage_results if r.stage_name != result.stage_name
        ]
        self.stage_results.append(result)
