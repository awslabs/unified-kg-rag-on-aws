# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import boto3
from pydantic import BaseModel

from aws_graphrag.adapters.aws import S3CacheManager
from aws_graphrag.application.ingestion.pipeline_stages import (
    ClaimExtractionStage,
    ClaimResolutionStage,
    CommunityDetectionStage,
    DocumentLoadingStage,
    DocumentParsingStage,
    GleaningStage,
    GraphAnalysisStage,
    GraphExtractionStage,
    GraphResolutionStage,
    IndexingStage,
    PipelineStage,
    TextChunkingStage,
    TranslationStage,
)
from aws_graphrag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Document,
    Entity,
    PipelineConfig,
    PipelineContext,
    PipelineMetrics,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    Relationship,
    TextUnit,
)
from aws_graphrag.shared import (
    PipelineExecutionError,
    PipelineResumeError,
    PipelineResumeManager,
    PipelineStageError,
    PipelineStateError,
    PipelineStateManager,
    get_cache_manager,
    get_logger,
)
from aws_graphrag.shared.metrics import MetricsSink, NullMetricsSink
from aws_graphrag.shared.utils import compute_hash

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class DataIngestionPipeline:
    STAGE_CLASSES = {
        PipelineStageType.DOCUMENT_PARSING: DocumentParsingStage,
        PipelineStageType.DOCUMENT_LOADING: DocumentLoadingStage,
        PipelineStageType.TEXT_CHUNKING: TextChunkingStage,
        PipelineStageType.TRANSLATION: TranslationStage,
        PipelineStageType.GRAPH_EXTRACTION: GraphExtractionStage,
        PipelineStageType.GLEANING: GleaningStage,
        PipelineStageType.GRAPH_RESOLUTION: GraphResolutionStage,
        PipelineStageType.CLAIM_EXTRACTION: ClaimExtractionStage,
        PipelineStageType.CLAIM_RESOLUTION: ClaimResolutionStage,
        PipelineStageType.GRAPH_ANALYSIS: GraphAnalysisStage,
        PipelineStageType.COMMUNITY_DETECTION: CommunityDetectionStage,
        PipelineStageType.INDEXING: IndexingStage,
    }

    INPUT_DIR_REQUIRED_STAGES = {
        PipelineStageType.DOCUMENT_PARSING,
        PipelineStageType.DOCUMENT_LOADING,
    }

    BOTO_REQUIRED_STAGES = {
        PipelineStageType.CLAIM_EXTRACTION,
        PipelineStageType.COMMUNITY_DETECTION,
        PipelineStageType.GLEANING,
        PipelineStageType.GRAPH_ANALYSIS,
        PipelineStageType.GRAPH_EXTRACTION,
        PipelineStageType.TEXT_CHUNKING,
        PipelineStageType.TRANSLATION,
    }

    STAGE_OUTPUT_MAPPING: dict[
        PipelineStageType, dict[str, tuple[type[BaseModel], str]]
    ] = {
        PipelineStageType.CLAIM_EXTRACTION: {"claims": (Claim, "claims")},
        PipelineStageType.CLAIM_RESOLUTION: {
            "resolved_claims": (Claim, "resolved_claims")
        },
        PipelineStageType.COMMUNITY_DETECTION: {
            "communities": (Community, "communities"),
            "community_reports": (CommunityReport, "community_reports"),
        },
        PipelineStageType.DOCUMENT_LOADING: {"documents": (Document, "documents")},
        PipelineStageType.DOCUMENT_PARSING: {"documents": (Document, "documents")},
        PipelineStageType.GLEANING: {
            "entities": (Entity, "entities"),
            "relationships": (Relationship, "relationships"),
        },
        PipelineStageType.GRAPH_ANALYSIS: {
            "resolved_entities": (Entity, "resolved_entities"),
            "resolved_relationships": (Relationship, "resolved_relationships"),
        },
        PipelineStageType.GRAPH_EXTRACTION: {
            "entities": (Entity, "entities"),
            "relationships": (Relationship, "relationships"),
        },
        PipelineStageType.GRAPH_RESOLUTION: {
            "resolved_entities": (Entity, "resolved_entities"),
            "resolved_relationships": (Relationship, "resolved_relationships"),
        },
        PipelineStageType.TEXT_CHUNKING: {"text_units": (TextUnit, "text_units")},
        PipelineStageType.TRANSLATION: {
            "translated_units": (TextUnit, "translated_units")
        },
    }

    def __init__(
        self,
        config: Config,
        pipeline_config: PipelineConfig,
        source_directory: Path | None = None,
        target_directory: Path | None = None,
        boto_session: boto3.Session | None = None,
        metrics_sink: MetricsSink | None = None,
    ) -> None:
        self.config = config
        self.pipeline_config = pipeline_config
        # Where pipeline metrics are forwarded (CloudWatch EMF, etc.). Defaults
        # to a no-op so the library assumes no monitoring backend.
        self.metrics_sink: MetricsSink = metrics_sink or NullMetricsSink()
        self.source_directory = source_directory or Path(
            self.config.processing.document_parsing.source_directory
        )
        self.target_directory = Path(
            target_directory
            or self.config.processing.document_parsing.target_directory
            or self.source_directory
        )
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )

        self._initialize_managers()
        self.stages, self.name_to_type_map = self._initialize_stages()

        logger.info(
            f"Successfully initialized pipeline with {len(self.stages)} stages: "
            f"'{', '.join([stage.name for stage in self.stages])}'"
        )

    def _initialize_managers(self) -> None:
        cache_config = self.config.cache
        chunking_config = cache_config.chunking

        cache_manager_class = get_cache_manager()
        self.cache_manager = cache_manager_class(
            config=self.config,
            cache_directory=self.pipeline_config.local_directory,
            ttl_seconds=cache_config.ttl_seconds,
            chunk_size=chunking_config.chunk_size,
            max_file_size_mb=chunking_config.max_file_size_mb,
            enable_chunking=chunking_config.enabled,
        )
        self.state_manager = PipelineStateManager(self.cache_manager)
        self.resume_manager = PipelineResumeManager(self.state_manager)

        self.s3_cache_manager: S3CacheManager | None = None
        if self.pipeline_config.s3_sync_enabled:
            self.s3_cache_manager = S3CacheManager(
                config=self.config,
                boto_session=self.boto_session,
                bucket_name=self.pipeline_config.s3_bucket_name,
                prefix=self.pipeline_config.s3_prefix,
            )

    def _initialize_stages(
        self,
    ) -> tuple[list[PipelineStage], dict[str, PipelineStageType]]:
        stages = []
        name_to_type_map = {}
        enabled_stages = []
        disabled_stages = []

        for stage_type, stage_class in self.STAGE_CLASSES.items():
            stage_enabled = self.pipeline_config.stages_enabled.get(stage_type, True)

            if stage_type == PipelineStageType.GLEANING:
                gleaning_config = self.config.processing.gleaning
                if gleaning_config and hasattr(gleaning_config, "enabled"):
                    stage_enabled = stage_enabled and gleaning_config.enabled

            elif stage_type in [
                PipelineStageType.CLAIM_EXTRACTION,
                PipelineStageType.CLAIM_RESOLUTION,
            ]:
                claim_config = getattr(self.config.processing, "claim_extraction", None)
                if claim_config and hasattr(claim_config, "enabled"):
                    stage_enabled = stage_enabled and claim_config.enabled

            if stage_enabled:
                kwargs: dict[str, Any] = {}
                if stage_type in self.INPUT_DIR_REQUIRED_STAGES:
                    kwargs["source_directory"] = self.source_directory
                if stage_type in self.BOTO_REQUIRED_STAGES:
                    kwargs["boto_session"] = self.boto_session
                if stage_type == PipelineStageType.DOCUMENT_PARSING:
                    kwargs["target_directory"] = self.target_directory

                kwargs["config"] = self.config

                stage_instance = stage_class(**kwargs)
                stages.append(stage_instance)
                name_to_type_map[stage_instance.name] = stage_type
                enabled_stages.append(stage_type.value)
            else:
                disabled_stages.append(stage_type.value)

        if disabled_stages:
            logger.info(f"Disabled stages: {disabled_stages}")

        return stages, name_to_type_map

    def run(
        self,
        source_directory: str | Path,
        pipeline_id: str | None = None,
        resume_from_stage: str | None = None,
    ) -> PipelineContext:
        try:
            source_path = Path(source_directory).resolve()
            logger.info(
                f"Starting pipeline run - source: '{source_path}', "
                f"pipeline_id: '{pipeline_id}', resume_from: '{resume_from_stage or 'auto-detect'}'"
            )

            self._validate_source_directory(source_path)

            resolved_pipeline_id = self._resolve_pipeline_id(source_path, pipeline_id)
            resolved_resume_stage = (
                resume_from_stage or self.pipeline_config.resume_from_stage
            )

            logger.info(f"Resolved pipeline ID: '{resolved_pipeline_id}'")

            if self.pipeline_config.s3_sync_enabled:
                self._sync_cache_with_s3(resolved_pipeline_id, "download")

            if resolved_pipeline_id and self.state_manager.pipeline_exists(
                resolved_pipeline_id
            ):
                logger.info(
                    f"Resuming existing pipeline '{resolved_pipeline_id}' "
                    f"from stage: '{resolved_resume_stage or 'auto-detect'}'"
                )
                context, start_stage_name = self._prepare_resume(
                    resolved_pipeline_id, resolved_resume_stage
                )
            else:
                logger.info(f"Starting new pipeline run: '{resolved_pipeline_id}'")
                context, start_stage_name = self._prepare_new_run(
                    source_path, resolved_pipeline_id, resolved_resume_stage
                )

            self._execute_pipeline(context, start_stage_name)

            logger.info(
                f"Pipeline execution completed successfully - ID: "
                f"'{context.pipeline_id}', Status: '{context.status}'"
            )
            return context

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise PipelineExecutionError(f"Failed to run pipeline: {e}") from e

    @staticmethod
    def _validate_source_directory(source_directory: Path) -> None:
        if not source_directory.exists() or not source_directory.is_dir():
            logger.error(f"Source directory validation failed: '{source_directory}'")
            raise FileNotFoundError(
                f"Source directory not found or not a directory: '{source_directory}'"
            )

    def _resolve_pipeline_id(
        self, source_directory: Path, pipeline_id: str | None
    ) -> str:
        return (
            pipeline_id
            or self.pipeline_config.pipeline_id
            or self._generate_pipeline_id(source_directory)
        )

    @staticmethod
    def _generate_pipeline_id(source_directory: Path | None = None) -> str:
        if source_directory is None:
            return f"pipeline-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdirs = [
            str(path.relative_to(source_directory))
            for path in source_directory.rglob("*")
            if path.is_dir()
        ]

        subdirs.sort()
        directory_identifier = "_".join(subdirs) if subdirs else source_directory.name
        directory_hash = compute_hash(directory_identifier, length=8)

        return f"pipeline-{directory_hash}-{timestamp}"

    def _sync_cache_with_s3(self, pipeline_id: str, direction: str) -> None:
        if not self.s3_cache_manager:
            logger.info("S3 cache manager not available, skipping sync")
            return

        cache_dir = self.cache_manager.get_pipeline_cache_dir(pipeline_id)

        try:
            if direction == "upload":
                self.s3_cache_manager.sync_pipeline_to_s3(pipeline_id, cache_dir)
            else:
                self.s3_cache_manager.sync_pipeline_from_s3(pipeline_id, cache_dir)
            logger.info(
                f"S3 sync {direction} completed successfully for "
                f"pipeline '{pipeline_id}'"
            )
        except Exception as e:
            logger.error(
                f"Failed to sync cache {direction} S3 for pipeline '{pipeline_id}': {e}"
            )
            raise

    def _prepare_resume(
        self, pipeline_id: str, resume_from_stage: str | None
    ) -> tuple[PipelineContext, str | None]:
        logger.info(
            f"Preparing pipeline resume for '{pipeline_id}' "
            f"from stage: '{resume_from_stage or 'auto-detect'}'"
        )

        try:
            start_stage_name, completed_stages = (
                self.resume_manager.determine_resume_strategy(
                    pipeline_id, resume_from_stage
                )
            )

            context = self.resume_manager.restore_pipeline_context(
                pipeline_id, completed_stages
            )

            self._prepare_context_for_resume(context, start_stage_name)

            logger.info(
                f"Pipeline resume prepared successfully - "
                f"will start from stage: '{start_stage_name}'"
            )
            return context, start_stage_name
        except (PipelineResumeError, PipelineStateError) as e:
            logger.error(f"Failed to prepare resume for pipeline '{pipeline_id}': {e}")
            raise PipelineExecutionError(f"Pipeline resume failed: {e}") from e

    def _prepare_context_for_resume(
        self, context: PipelineContext, start_stage_name: str
    ) -> None:
        stage_names = [stage.name for stage in self.stages]
        try:
            start_index = stage_names.index(start_stage_name)
            valid_results = [
                res
                for res in context.stage_results
                if stage_names.index(res.stage_name) < start_index
            ]
            context.stage_results = valid_results
            logger.info(
                f"Context prepared for resume from '{start_stage_name}' - "
                f"keeping {len(valid_results)} previous stage results"
            )
        except ValueError:
            logger.warning(
                f"Start stage '{start_stage_name}' not found in stage list, "
                f"starting from beginning"
            )
            context.stage_results = []

    def _prepare_new_run(
        self, source_directory: Path, pipeline_id: str, resume_from_stage: str | None
    ) -> tuple[PipelineContext, str | None]:
        context = PipelineContext(
            pipeline_id=pipeline_id,
            status=PipelineStageStatus.RUNNING,
            start_time=datetime.now(),
            config=self.pipeline_config,
            source_directory=source_directory,
        )
        start_stage_name = resume_from_stage or (
            self.stages[0].name if self.stages else None
        )

        return context, start_stage_name

    def _execute_pipeline(
        self, context: PipelineContext, start_stage_name: str | None
    ) -> None:
        total_start_time = time.time()
        self._execute_pipeline_stages(context, start_stage_name)
        self._finalize_pipeline_execution(context, total_start_time)

    def _execute_pipeline_stages(
        self, context: PipelineContext, start_stage_name: str | None
    ) -> None:
        stage_names = [stage.name for stage in self.stages]
        start_index = 0

        if start_stage_name:
            try:
                start_index = stage_names.index(start_stage_name)
            except ValueError:
                logger.warning(
                    f"Start stage '{start_stage_name}' not found, "
                    f"starting from beginning"
                )

        completed_stage_names = {
            r.stage_name
            for r in context.stage_results
            if r.status == PipelineStageStatus.COMPLETED
        }

        if completed_stage_names:
            logger.info(
                f"Found {len(completed_stage_names)} previously completed stages: "
                f"'{', '.join(list(completed_stage_names))}'"
            )

        stages_to_execute = self.stages[start_index:]
        logger.info(
            f"Executing {len(stages_to_execute)} stages: "
            f"'{', '.join([s.name for s in stages_to_execute])}'"
        )

        for stage in stages_to_execute:
            if stage.name in completed_stage_names:
                logger.info(f"Skipping previously completed stage: '{stage.name}'")
            else:
                logger.info(f"Executing stage: '{stage.name}'")
                result = self._execute_stage(stage, context)

                if self._should_stop_pipeline(result):
                    logger.warning(
                        f"Pipeline execution stopped due to stage failure: '{stage.name}'"
                    )
                    break

    def _execute_stage(
        self, stage: PipelineStage, context: PipelineContext
    ) -> PipelineStageResult:
        running_result = PipelineStageResult(
            stage_name=stage.name,
            status=PipelineStageStatus.RUNNING,
            start_time=datetime.now(),
        )
        self.state_manager.update_stage_result(context, running_result)

        try:
            result = stage.execute(context)
            self.state_manager.update_stage_result(context, result)

            if result.status == PipelineStageStatus.COMPLETED:
                self._save_stage_outputs_to_cache(context, stage.name)
            elif result.status == PipelineStageStatus.FAILED:
                logger.error(f"Stage '{stage.name}' failed: {result.error_message}")

            return result

        except PipelineStageError as e:
            logger.error(f"Stage '{stage.name}' validation failed: {e}")
            failed_result = PipelineStageResult(
                stage_name=stage.name,
                status=PipelineStageStatus.FAILED,
                start_time=running_result.start_time,
                end_time=datetime.now(),
                error_message=str(e),
            )
            self.state_manager.update_stage_result(context, failed_result)
            return failed_result

        except Exception as e:
            logger.error(
                f"Unexpected error in stage '{stage.name}': {e}", exc_info=True
            )
            failed_result = PipelineStageResult(
                stage_name=stage.name,
                status=PipelineStageStatus.FAILED,
                start_time=running_result.start_time,
                end_time=datetime.now(),
                error_message=f"Unexpected error: {str(e)}",
            )
            self.state_manager.update_stage_result(context, failed_result)
            return failed_result

    def _save_stage_outputs_to_cache(
        self, context: PipelineContext, stage_name: str
    ) -> None:
        if not self.pipeline_config.cache_enabled:
            logger.info(
                f"Cache disabled, skipping output save for stage: '{stage_name}'"
            )
            return

        stage_type = self.name_to_type_map.get(stage_name)
        if not stage_type or stage_type not in self.STAGE_OUTPUT_MAPPING:
            logger.info(
                f"No output mapping found for stage: '{stage_name}', "
                f"skipping cache save"
            )
            return

        try:
            saved_outputs = []
            for _, (_, context_attr) in self.STAGE_OUTPUT_MAPPING[stage_type].items():
                data_to_save = getattr(context, context_attr, None)
                if not data_to_save:
                    logger.debug(f"No data found for context attribute: {context_attr}")
                    continue

                logger.info(
                    f"Saving {len(data_to_save) if isinstance(data_to_save, list) else 1} {context_attr} to cache"
                )

                cache_entry = self.cache_manager.save_stage_result(
                    data=data_to_save,
                    cache_key=context_attr,
                    stage_name=stage_name,
                    pipeline_id=context.pipeline_id,
                    metadata={
                        "stage_type": (
                            stage_type.value
                            if hasattr(stage_type, "value")
                            else str(stage_type)
                        ),
                        "context_attribute": context_attr,
                    },
                )

                if cache_entry:
                    item_count = cache_entry.record_count or (
                        len(data_to_save) if isinstance(data_to_save, list) else 1
                    )
                    chunked_info = ""
                    if getattr(cache_entry.metadata, "is_chunked", False):
                        chunk_count = getattr(cache_entry.metadata, "chunk_count", 0)
                        chunked_info = f" (chunked into {chunk_count} files)"

                    saved_outputs.append(f"{item_count} {context_attr}{chunked_info}")
                    logger.info(f"Successfully cached {context_attr}")
                else:
                    logger.warning(f"Cache manager returned None for {context_attr}")

            if saved_outputs:
                logger.info(
                    f"Stage '{stage_name}' outputs cached: {', '.join(saved_outputs)}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to save outputs to cache for stage '{stage_name}': {e}"
            )

    def _should_stop_pipeline(self, result: PipelineStageResult) -> bool:
        if result.status == PipelineStageStatus.FAILED:
            if self.pipeline_config.continue_on_error:
                logger.warning(
                    f"Stage '{result.stage_name}' failed but continuing due to "
                    f"continue_on_error setting: {result.error_message}"
                )
                return False
            else:
                logger.error(
                    f"Stopping pipeline due to failure in stage '{result.stage_name}': "
                    f"{result.error_message}"
                )
                return True
        return False

    def _finalize_pipeline_execution(
        self, context: PipelineContext, start_time: float
    ) -> None:
        logger.info(f"Finalizing pipeline execution - ID: '{context.pipeline_id}'")

        context.end_time = datetime.now()
        total_duration = time.time() - start_time
        context.duration_seconds = total_duration
        context.global_metrics = self._create_pipeline_metrics(context)
        self._emit_metrics(context.global_metrics, context.pipeline_id)

        failed_stages = [
            r for r in context.stage_results if r.status == PipelineStageStatus.FAILED
        ]

        if failed_stages:
            context.status = PipelineStageStatus.FAILED
            logger.warning(
                f"Pipeline completed with {len(failed_stages)} failed stages: "
                f"{[r.stage_name for r in failed_stages]}"
            )
        else:
            context.status = PipelineStageStatus.COMPLETED
            logger.info("Pipeline completed successfully with all stages passed")

        self.state_manager.save_pipeline_metadata(context)

        if self.pipeline_config.s3_sync_enabled:
            self._sync_cache_with_s3(context.pipeline_id, "upload")

        self._log_pipeline_summary(context)

    def _create_pipeline_metrics(self, context: PipelineContext) -> PipelineMetrics:
        cache_stats = self.cache_manager.get_cache_stats(context.pipeline_id)
        stage_durations, stage_throughput = self._calculate_stage_performance(context)

        metric_data: dict[str, Any] = {
            "pipeline_id": context.pipeline_id,
            "total_duration_seconds": context.duration_seconds,
            "total_documents_processed": len(context.documents or []),
            "total_text_units_created": len(context.text_units or []),
            "total_translated_units": len(context.translated_units or []),
            "total_entities_extracted": len(context.entities or []),
            "total_relationships_extracted": len(context.relationships or []),
            "total_claims_extracted": len(context.claims or []),
            "total_communities_detected": len(context.communities or []),
            "total_community_reports_generated": len(context.community_reports or []),
            "gleaning_improvement_rate": self._get_stage_metric(
                context, "gleaning", "quality_improvement_rate"
            ),
            "entity_resolution_merge_rate": self._get_stage_metric(
                context, "graph_resolution", "entity_merge_rate"
            ),
            "relationship_resolution_merge_rate": self._get_stage_metric(
                context, "graph_resolution", "relationship_merge_rate"
            ),
            "claim_resolution_merge_rate": self._get_stage_metric(
                context, "claim_resolution", "claim_merge_rate"
            ),
            "community_modularity_score": self._get_stage_metric(
                context, "community_detection", "modularity_score"
            ),
            "cache_hit_rate": cache_stats.hit_rate,
            "cache_size_mb": cache_stats.total_size_mb,
            "stage_durations": stage_durations,
            "stage_throughput": stage_throughput,
        }
        return PipelineMetrics(**metric_data)

    def _emit_metrics(self, metrics: PipelineMetrics, pipeline_id: str) -> None:
        """Forward scalar pipeline metrics to the configured sink (best-effort)."""
        try:
            scalars = {
                k: v
                for k, v in metrics.model_dump().items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            self.metrics_sink.emit(
                namespace="aws_graphrag/ingestion",
                metrics=scalars,
                dimensions={"pipeline_id": pipeline_id},
            )
        except Exception as e:  # never let metrics emission break a run
            logger.warning("Metrics emission failed: %s", e)

    @staticmethod
    def _calculate_stage_performance(
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

    @staticmethod
    def _get_stage_metric(
        context: PipelineContext, stage_name_prefix: str, metric_name: str
    ) -> float:
        for result in context.stage_results:
            if result.stage_name.lower().startswith(stage_name_prefix):
                return float(result.metrics.get(metric_name, 0.0))
        return 0.0

    @staticmethod
    def _log_pipeline_summary(context: PipelineContext) -> None:
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
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(summary)

    def repair_pipeline_metadata(self, context: PipelineContext) -> bool:
        logger.info(
            f"Attempting to repair pipeline metadata for: {context.pipeline_id}"
        )

        try:
            result = self.state_manager.repair_pipeline_metadata(context)
            if result:
                logger.info(
                    f"Pipeline metadata repair successful for: {context.pipeline_id}"
                )
            else:
                logger.warning(
                    f"Pipeline metadata repair failed for: {context.pipeline_id}"
                )
            return result
        except Exception as e:
            logger.error(
                f"Error during pipeline metadata repair for {context.pipeline_id}: {e}"
            )
            return False

    def verify_pipeline_metadata(self, pipeline_id: str) -> bool:
        logger.info(f"Verifying pipeline metadata for: {pipeline_id}")

        try:
            result = self.state_manager.verify_pipeline_metadata(pipeline_id)
            if result:
                logger.info(
                    f"Pipeline metadata verification successful for: {pipeline_id}"
                )
            else:
                logger.warning(
                    f"Pipeline metadata verification failed for: {pipeline_id}"
                )
            return result
        except Exception as e:
            logger.error(
                f"Error during pipeline metadata verification for {pipeline_id}: {e}"
            )
            return False
