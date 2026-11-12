# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The metrics of an ingestion run: built from its context, emitted to the
pipeline's metrics sink after each stage and when the run ends, and the
summary logged at its end.

The functions that read the pipeline (its cache manager, enabled stages
and metrics sink) take the ``DataIngestionPipeline``, which keeps a
delegating method for each of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from unified_kg_rag.domain.models import (
    PipelineContext,
    PipelineMetrics,
    PipelineStageStatus,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.usage import merge_usage

if TYPE_CHECKING:
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

logger = get_logger(__name__)

# The PipelineStageResult.metrics key of the stage's LLM usage
# (shared.usage snapshot).
LLM_USAGE_METRIC = "llm_usage"


def run_token_usage(
    context: PipelineContext, executed_stages: frozenset[str]
) -> dict[str, Any]:
    """The LLM usage of the stages a run executed (a resumed run's restored
    stages spent nothing in it): the totals, ``by_role``, ``by_model`` and
    ``stages`` (each executed stage's usage)."""
    stages = {
        r.stage_name: r.metrics[LLM_USAGE_METRIC]
        for r in context.stage_results
        if r.stage_name in executed_stages
        and isinstance(r.metrics.get(LLM_USAGE_METRIC), dict)
    }
    return {**merge_usage(list(stages.values())), "stages": stages}


# Emitted metric -> the stages whose execution produces it. A metric is
# emitted once, right after the last of its stages this run has enabled
# executes (a resumed run restores the others from the cache and does not
# re-emit their counts, or Sum dashboards would over-count). The other
# metrics describe the whole run and are emitted when it ends.
METRIC_STAGES: dict[str, frozenset[str]] = {
    "total_documents_processed": frozenset({"document_parsing", "document_loading"}),
    "total_text_units_created": frozenset({"text_chunking"}),
    "total_translated_units": frozenset({"translation"}),
    "total_entities_extracted": frozenset({"graph_extraction", "gleaning"}),
    "total_relationships_extracted": frozenset({"graph_extraction", "gleaning"}),
    "total_claims_extracted": frozenset({"claim_extraction"}),
    "total_communities_detected": frozenset({"community_detection"}),
    "total_community_reports_generated": frozenset({"community_detection"}),
    "gleaning_improvement_rate": frozenset({"gleaning"}),
    "entity_resolution_merge_rate": frozenset({"graph_resolution"}),
    "relationship_resolution_merge_rate": frozenset({"graph_resolution"}),
    "claim_resolution_merge_rate": frozenset({"claim_resolution"}),
    "community_modularity_score": frozenset({"community_detection"}),
    "total_items_indexed": frozenset({"indexing"}),
    "total_items_index_failed": frozenset({"indexing"}),
    "relationships_indexed": frozenset({"indexing"}),
    "community_drift_overlap_share": frozenset({"indexing"}),
    "community_rebuild_recommended": frozenset({"indexing"}),
    "community_growth_ratio": frozenset({"indexing"}),
    "runs_since_clustering": frozenset({"indexing"}),
    "total_extraction_failures": frozenset({"graph_extraction"}),
    "total_documents_quarantined": frozenset({"document_loading"}),
    "total_documents_deferred": frozenset({"document_loading"}),
    "total_documents_excluded": frozenset({"document_loading"}),
    "total_files_failed": frozenset({"document_parsing", "document_loading"}),
    "total_files_skipped": frozenset({"document_parsing"}),
    "total_documents_deleted": frozenset({"indexing"}),
    "total_records_not_stored": frozenset({"graph_resolution", "claim_extraction"}),
    "underived_items": frozenset({"indexing"}),
    "total_tree_summaries_failed": frozenset({"summary_tree"}),
    "total_tree_uncovered_leaves": frozenset({"summary_tree"}),
    "total_gleaning_failures": frozenset({"gleaning"}),
    "total_claim_extraction_failures": frozenset({"claim_extraction"}),
    "total_unrecognised_claim_statuses": frozenset({"claim_extraction"}),
}


def create_pipeline_metrics(
    pipeline: DataIngestionPipeline, context: PipelineContext
) -> PipelineMetrics:
    cache_stats = pipeline.cache_manager.get_cache_stats(context.pipeline_id)
    stage_durations, stage_throughput = calculate_stage_performance(context)

    metric_data: dict[str, Any] = {
        "pipeline_id": context.pipeline_id,
        # Not known until the run ends (a stage's metrics are built before).
        "total_duration_seconds": context.duration_seconds or 0.0,
        "total_documents_processed": len(context.documents or []),
        "total_text_units_created": len(context.text_units or []),
        "total_translated_units": len(context.translated_units or []),
        "total_entities_extracted": len(
            {e.id for r in context.unit_extractions for e in r.entities}
        ),
        "total_relationships_extracted": len(
            {rel.id for r in context.unit_extractions for rel in r.relationships}
        ),
        "total_claims_extracted": len(context.claims or []),
        "total_communities_detected": len(context.communities or []),
        "total_community_reports_generated": len(context.community_reports or []),
        "gleaning_improvement_rate": get_stage_metric(
            context, "gleaning", "improvement_rate"
        ),
        "entity_resolution_merge_rate": get_stage_metric(
            context, "graph_resolution", "entity_merge_rate"
        ),
        "relationship_resolution_merge_rate": get_stage_metric(
            context, "graph_resolution", "relationship_merge_rate"
        ),
        "claim_resolution_merge_rate": get_stage_metric(
            context, "claim_resolution", "claim_merge_rate"
        ),
        "community_modularity_score": get_stage_metric(
            context, "community_detection", "modularity_score"
        ),
        "total_items_indexed": int(
            get_stage_metric(context, "indexing", "total_indexed")
        ),
        # Include a hard-failure sentinel: when the indexing stage RAISES
        # (e.g. clear/initialize/connection error), its result carries no
        # metrics, so "total_failed" reads 0 and the IndexingFailures alarm
        # (which watches this metric) can never fire on a total indexing
        # failure. Emit >=1 in that case so the alarm surfaces the crash.
        "total_items_index_failed": int(
            get_stage_metric(context, "indexing", "total_failed")
        )
        or (1 if _stage_failed(context, "indexing") else 0),
        "relationships_indexed": int(
            get_stage_metric(context, "indexing", "relationships_indexed")
        ),
        "community_drift_overlap_share": get_stage_metric(
            context, "indexing", "community_drift_overlap_share"
        ),
        "community_rebuild_recommended": int(
            get_stage_metric(context, "indexing", "community_rebuild_recommended")
        ),
        "community_growth_ratio": get_stage_metric(
            context, "indexing", "community_growth_ratio"
        ),
        "runs_since_clustering": int(
            get_stage_metric(context, "indexing", "runs_since_clustering")
        ),
        # Per-chunk LLM failures degrade a run without failing it; surface
        # them so the ExtractionFailures alarm can catch a partial graph.
        "total_extraction_failures": int(
            get_stage_metric(context, "graph_extraction", "failed_units")
        ),
        "total_gleaning_failures": int(
            get_stage_metric(context, "gleaning", "failed_units")
        ),
        "total_claim_extraction_failures": int(
            get_stage_metric(context, "claim_extraction", "failed_units")
        ),
        # A status the model wrote outside the words the claim reader knows
        # is stored as unknown: a rising count means the prompt's vocabulary
        # is not followed (another answer language, say).
        "total_unrecognised_claim_statuses": int(
            get_stage_metric(context, "claim_extraction", "unrecognised_claim_statuses")
        ),
        # Documents an incremental run stopped retrying because their
        # processing kept killing the run (the DocumentsQuarantined
        # alarm), and those it deferred while isolating one.
        "total_documents_quarantined": int(
            get_stage_metric(context, "document_loading", "documents_quarantined")
        ),
        "total_documents_deferred": int(
            get_stage_metric(context, "document_loading", "documents_deferred")
        ),
        # Documents an incremental run stopped retrying (each failed
        # indexing.max_document_failures runs for an error of its own).
        "total_documents_excluded": int(
            get_stage_metric(context, "document_loading", "documents_excluded")
        ),
        # Corpus files the run could not read: their documents are
        # neither processed nor deleted.
        "total_files_failed": sum(
            int(get_stage_metric(context, stage, "files_failed"))
            for stage in ("document_parsing", "document_loading")
        ),
        "total_files_skipped": int(
            get_stage_metric(context, "document_parsing", "files_skipped")
        ),
        "total_documents_deleted": int(
            get_stage_metric(context, "indexing", "documents_deleted")
        ),
        # The run's document outcomes, as run-ingestion's JSON result
        # reports them (failed_documents, documents_retrying,
        # documents_without_claims): recorded by the indexing commit or
        # a stage that stopped the run, so emitted when the run ends.
        "total_documents_failed": len(context.recorded_failed_doc_ids),
        "total_units_failed": len(context.failed_unit_ids()),
        "total_documents_retrying": len(context.retrying_doc_ids),
        "total_documents_without_claims": len(context.claims_missing_doc_ids),
        "total_records_not_stored": sum(
            int(get_stage_metric(context, stage, "records_not_stored"))
            for stage in ("graph_resolution", "claim_extraction")
        ),
        "underived_items": int(
            get_stage_metric(context, "indexing", "underived_items")
        ),
        # A partial summary tree degrades tree retrieval without failing the
        # run; so do the text units an on_reset tree leaves uncovered.
        "total_tree_summaries_failed": int(
            get_stage_metric(context, "summary_tree", "summaries_failed")
        ),
        "total_tree_uncovered_leaves": int(
            get_stage_metric(context, "summary_tree", "tree_uncovered_leaves")
        ),
        "cache_hit_rate": cache_stats.hit_rate,
        "cache_size_mb": cache_stats.total_size_mb,
        "stage_durations": stage_durations,
        "stage_throughput": stage_throughput,
    }
    return PipelineMetrics(**metric_data)


def emit_metrics(
    pipeline: DataIngestionPipeline,
    metrics: PipelineMetrics,
    pipeline_id: str,
    stage_name: str | None = None,
    extra: dict[str, float] | None = None,
) -> None:
    """Forward scalar pipeline metrics to the configured sink (best-effort).

    With ``stage_name`` (right after that stage ran): the metrics it
    completes (``METRIC_STAGES``, emitted after the last of their stages
    this run has enabled) and its duration, so a run that is stopped or
    times out still reports the stages it finished. Without: the metrics
    of the whole run, when it ends.
    """
    enabled = [stage.name for stage in getattr(pipeline, "stages", [])]

    def completes(stages: frozenset[str]) -> str | None:
        # The last enabled stage of ``stages``, in pipeline order.
        last = [name for name in enabled if name in stages]
        return last[-1] if last else None

    def emitted(name: str) -> bool:
        stages = METRIC_STAGES.get(name)
        if stage_name is None:
            return stages is None
        return stages is not None and completes(stages) == stage_name

    try:
        dumped = metrics.model_dump()
        scalars = {
            k: v
            for k, v in dumped.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and emitted(k)
        }
        # The per-stage duration map is flattened into scalar metrics (the
        # dict member is otherwise dropped by the scalar filter).
        duration = (metrics.stage_durations or {}).get(stage_name or "")
        if isinstance(duration, (int, float)):
            scalars[f"duration_{stage_name}_seconds"] = duration
        scalars.update(extra or {})
        pipeline.metrics_sink.emit(
            namespace="unified_kg_rag/ingestion",
            metrics=scalars,
            dimensions={"pipeline_id": pipeline_id},
        )
    except Exception as e:  # noqa: BLE001 - metrics never break a run
        logger.warning("Metrics emission failed: %s", e)


def emit_stage_metrics(
    pipeline: DataIngestionPipeline, context: PipelineContext, stage_name: str
) -> None:
    """Emit what ``stage_name`` completes, right after it ran."""
    try:
        metrics = pipeline._create_pipeline_metrics(context)
    except Exception as e:  # noqa: BLE001 - metrics never break a run
        logger.warning("Metrics emission failed: %s", e)
        return
    pipeline._emit_metrics(
        metrics,
        context.pipeline_id,
        stage_name,
        stage_usage_metrics(context, stage_name),
    )


def stage_usage_metrics(context: PipelineContext, stage_name: str) -> dict[str, float]:
    """The flat LLM usage metrics of a stage that called a model or was
    served from the response cache (none for a model-free stage, so no
    custom-metric series is created for it): ``llm_calls_<stage>``,
    ``llm_input_tokens_<stage>``, ``llm_output_tokens_<stage>``,
    ``llm_throttles_<stage>`` (requests the provider throttled), and,
    summed over the stages by a Sum statistic, ``llm_cache_hits``,
    ``llm_throttles`` and ``llm_requests`` (the requests sent: returned and
    failed calls and the requests resent inside them), whose ratio the CDK
    stack's throttling alarm reads."""
    usage = next(
        (
            r.metrics.get(LLM_USAGE_METRIC)
            for r in reversed(context.stage_results)
            if r.stage_name == stage_name
        ),
        None,
    )
    if not usage or not (
        usage.get("llm_calls") or usage.get("cache_hits") or usage.get("failed_calls")
    ):
        return {}
    throttles = usage.get("throttled_requests", 0)
    return {
        f"llm_calls_{stage_name}": usage.get("llm_calls", 0),
        f"llm_input_tokens_{stage_name}": usage.get("input_tokens", 0),
        f"llm_output_tokens_{stage_name}": usage.get("output_tokens", 0),
        f"llm_throttles_{stage_name}": throttles,
        "llm_cache_hits": usage.get("cache_hits", 0),
        "llm_throttles": throttles,
        "llm_requests": usage.get("llm_calls", 0)
        + usage.get("failed_calls", 0)
        + usage.get("request_retries", 0),
    }


def calculate_stage_performance(
    context: PipelineContext,
) -> tuple[dict[str, float], dict[str, float]]:
    durations = {}
    throughput = {}

    for result in context.stage_results:
        if result.duration_seconds is not None and result.duration_seconds > 0:
            durations[result.stage_name] = result.duration_seconds
            if result.output_count > 0:
                throughput[result.stage_name] = (
                    result.output_count / result.duration_seconds
                )

    return durations, throughput


def get_stage_metric(
    context: PipelineContext, stage_name_prefix: str, metric_name: str
) -> float:
    for result in context.stage_results:
        if result.stage_name.lower().startswith(stage_name_prefix):
            return float(result.metrics.get(metric_name, 0.0))
    return 0.0


def _stage_failed(context: PipelineContext, stage_name_prefix: str) -> bool:
    """True if the named stage ran and ended FAILED (crashed).

    A crashed stage carries no metrics, so the metric-based failure counts
    read 0; callers use this to still surface the failure (e.g. to the
    IndexingFailures alarm).
    """
    for result in context.stage_results:
        if result.stage_name.lower().startswith(stage_name_prefix):
            return result.status == PipelineStageStatus.FAILED
    return False


def log_pipeline_summary(context: PipelineContext) -> None:
    status_counts = dict.fromkeys(PipelineStageStatus, 0)
    for r in context.stage_results:
        status_counts[r.status] += 1

    completed = status_counts.get(PipelineStageStatus.COMPLETED, 0)
    failed = status_counts.get(PipelineStageStatus.FAILED, 0)
    running = status_counts.get(PipelineStageStatus.RUNNING, 0)

    total_documents = len(context.documents or [])
    total_entities = len(context.resolved_entities or [])
    total_relationships = len(context.resolved_relationships or [])
    total_claims = len(context.resolved_claims or [])
    total_communities = len(context.communities or [])

    summary = (
        f"Pipeline ID: {context.pipeline_id}\n"
        f"Status: {context.status.value}\n"
        f"Duration: {context.duration_seconds:.2f}s\n"
        f"Stages: {completed} completed, {failed} failed, {running} running\n"
        f"Data processed: {total_documents} documents -> "
        f"{total_entities} entities, {total_relationships} relationships, "
        f"{total_claims} claims -> {total_communities} communities"
    )
    if failed_units := context.failed_unit_ids():
        summary += f"\nText units failed: {len(failed_units)}"
    if context.claims_missing_doc_ids:
        summary += (
            f"\nIndexed without some of their claims: "
            f"{len(context.claims_missing_doc_ids)} documents"
        )
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info("%s", summary)
