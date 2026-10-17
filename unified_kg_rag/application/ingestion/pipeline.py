# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import secrets
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel
from structlog.contextvars import bind_contextvars, reset_contextvars

from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.adapters.llm.response_cache import LLMResponseCache
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.ingestion.namespace_lease import NamespaceLease
from unified_kg_rag.application.ingestion.pipeline_stages import (
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
from unified_kg_rag.application.observability import configure_observability
from unified_kg_rag.domain.ingestion.delta_detector import validate_retired_scopes
from unified_kg_rag.domain.models import (
    Config,
    ExtractionSettings,
    PipelineConfig,
    PipelineContext,
    PipelineMetrics,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    validate_path_segment,
)
from unified_kg_rag.ports import DocStatusPort, GraphIndexer, VectorIndexer
from unified_kg_rag.shared import (
    CacheSyncError,
    ConfigurationError,
    NamespaceLeaseError,
    PipelineExecutionError,
    PipelineInterrupted,
    PipelineResumeError,
    PipelineResumeManager,
    PipelineStateError,
    PipelineStateManager,
    get_logger,
    observability,
)
from unified_kg_rag.shared.cache_manager import CacheManager, CacheStrategy
from unified_kg_rag.shared.metrics import MetricsSink, NullMetricsSink
from unified_kg_rag.shared.pipeline_manager import STAGE_OUTPUTS
from unified_kg_rag.shared.utils import (
    compute_hash,
    corpus_manifest_fingerprint,
    stage_cache_key,
)
from unified_kg_rag.shared.utils.cache_keys import (
    changed_settings,
    extraction_settings,
)
from unified_kg_rag.shared.utils.document_identity import config_registry_namespace

if TYPE_CHECKING:
    import boto3

    from unified_kg_rag.adapters.aws.s3_cache import S3CacheManager

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class DataIngestionPipeline:
    """The ingestion pipeline: documents in, an indexed knowledge graph out.

    Runs the stages of ``STAGE_CLASSES`` in order (parse, load, chunk,
    translate, extract, glean, resolve, claims, analyze, communities, index),
    skipping those ``pipeline_config.stages_enabled`` or a config section
    (``CONFIG_ENABLED``) turns off. Each stage's output is cached under the
    run's pipeline id in ``pipeline_config.local_directory`` (and, with
    ``s3_sync_enabled``, uploaded to S3 after every stage), so a re-run with
    the same pipeline id resumes after the last stage whose cached output is
    still valid. ``request_stop`` ends a run early without losing its finished
    stages. With the doc-status registry on, only new and changed documents
    are indexed (see ``IncrementalIndexer``).
    """

    # Run order; kept in PipelineStageType's declaration order (tested).
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

    # Stages switched off by a config section besides stages_enabled. Claim
    # extraction and resolution go together; community detection is optional
    # for a LightRAG-only (mix/hybrid/naive) ingestion, which needs no
    # communities (GraphRAG global/drift require it).
    CONFIG_ENABLED: dict[PipelineStageType, Callable[[Config], bool]] = {
        PipelineStageType.GLEANING: lambda c: c.processing.gleaning.enabled,
        PipelineStageType.CLAIM_EXTRACTION: (
            lambda c: c.processing.claim_extraction.enabled
        ),
        PipelineStageType.CLAIM_RESOLUTION: (
            lambda c: c.processing.claim_extraction.enabled
        ),
        PipelineStageType.COMMUNITY_DETECTION: (
            lambda c: c.graph.community_detection.enabled
        ),
    }

    # What each stage caches, derived from the single shared definition the
    # resume manager also reads, so save and restore cannot drift apart.
    STAGE_OUTPUT_MAPPING: dict[
        PipelineStageType, dict[str, tuple[type[BaseModel], str]]
    ] = {
        stage: {attr: (model, attr) for attr, model in outputs.items()}
        for stage, outputs in STAGE_OUTPUTS.items()
    }

    # Stages whose failure leaves the run's documents without some of their
    # artifacts (graph, claims, communities). When one crashes under
    # continue_on_error, every text unit of the run is recorded as failed for
    # it, so an incremental commit records the documents FAILED (re-extracted
    # next run) instead of PROCESSED with a partial graph.
    ARTIFACT_STAGES: frozenset[PipelineStageType] = frozenset(
        {
            PipelineStageType.TRANSLATION,
            PipelineStageType.GRAPH_EXTRACTION,
            PipelineStageType.GLEANING,
            PipelineStageType.GRAPH_RESOLUTION,
            PipelineStageType.CLAIM_EXTRACTION,
            PipelineStageType.CLAIM_RESOLUTION,
            PipelineStageType.GRAPH_ANALYSIS,
            PipelineStageType.COMMUNITY_DETECTION,
        }
    )

    # Stages that build the run's documents and text units. When one failed
    # (under continue_on_error), the run holds no per-unit record of what is
    # missing, so indexing does not run: an incremental commit would record
    # the documents PROCESSED without content, or remove every stored
    # document as deleted after a parsing crash.
    CORPUS_STAGES: tuple[PipelineStageType, ...] = (
        PipelineStageType.DOCUMENT_PARSING,
        PipelineStageType.DOCUMENT_LOADING,
        PipelineStageType.TEXT_CHUNKING,
    )

    # Emitted metric -> the stages whose execution produces it. A metric is
    # emitted once, right after the last of its stages this run has enabled
    # executes (a resumed run restores the others from the cache and does not
    # re-emit their counts, or Sum dashboards would over-count). The other
    # metrics describe the whole run and are emitted when it ends.
    METRIC_STAGES: dict[str, frozenset[str]] = {
        "total_documents_processed": frozenset(
            {"document_parsing", "document_loading"}
        ),
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
        "total_extraction_failures": frozenset({"graph_extraction"}),
        "total_documents_quarantined": frozenset({"document_loading"}),
        "total_gleaning_failures": frozenset({"gleaning"}),
        "total_claim_extraction_failures": frozenset({"claim_extraction"}),
    }

    # Names of the stages the current run executed (not skipped as completed).
    _executed_stages: frozenset[str] = frozenset()
    # The doc-status registry handed to the stages (None when off); the run
    # holds its namespace's write lease in it.
    _doc_status: DocStatusPort | None = None

    # Fingerprint of the run's corpus contents, computed in run(); part of every
    # stage cache key so a changed corpus is a cache miss, not a stale resume.
    corpus_fingerprint: str | None = None

    # Stop handling (see request_stop): the run's pipeline id once its S3
    # checkpoint is in place, whether a checkpoint is being written, and the
    # signal of a stop requested meanwhile.
    s3_cache_manager: S3CacheManager | None = None
    llm_response_cache: LLMResponseCache | None = None
    # Where the response cache keeps its segments, in the run's cache directory.
    LLM_CACHE_DIRECTORY = "llm-cache"
    _checkpoint_id: str | None = None
    _checkpointing: bool = False
    _stop_signal: int | None = None

    def __init__(
        self,
        config: Config,
        pipeline_config: PipelineConfig,
        source_directory: Path | None = None,
        target_directory: Path | None = None,
        boto_session: boto3.Session | None = None,
        metrics_sink: MetricsSink | None = None,
        *,
        providers: Providers | None = None,
        doc_status: DocStatusPort | None = None,
        vector_indexer: VectorIndexer | None = None,
        graph_indexer: GraphIndexer | None = None,
    ) -> None:
        """Build the stages; every backend may be injected.

        Args:
            providers: Model and store provider bundle (built from
                ``config.providers`` when omitted; see
                ``application.bootstrap.build_providers``).
            doc_status / vector_indexer / graph_indexer: Shorthand for a bundle
                built with ``doc_status=``, ``vector_indexer_factory=`` and
                ``graph_indexer_factory=`` (an injected registry turns
                incremental indexing on, as on ``Providers``). Set them on
                ``providers`` instead when passing one.

        Raises:
            ValueError: ``providers`` and one of the shorthand components are
                both given.
        """
        self.config = config
        # Before any stage builds a model, so the factories attach the tracing
        # callback (no-op unless observability.enabled).
        configure_observability(config.observability)
        overrides: dict[str, Any] = {}
        if doc_status is not None:
            overrides["doc_status"] = doc_status
        if vector_indexer is not None:
            overrides["vector_indexer_factory"] = lambda: vector_indexer
        if graph_indexer is not None:
            overrides["graph_indexer_factory"] = lambda: graph_indexer
        if providers is not None and overrides:
            raise ValueError(
                "Pass doc_status, vector_indexer and graph_indexer either to "
                "DataIngestionPipeline or on the Providers bundle, not both."
            )
        self.pipeline_config = pipeline_config
        # Where pipeline metrics are forwarded (CloudWatch EMF, etc.). Defaults
        # to a no-op so the library assumes no monitoring backend.
        self.metrics_sink: MetricsSink = metrics_sink or NullMetricsSink()
        self.source_directory = source_directory or Path(
            self.config.processing.document_parsing.source_directory
        )
        # Parsed-JSON export directory, if any. Unset means no export; it
        # must never default to the source corpus (see DocumentParsingStage).
        explicit_target = (
            target_directory or self.config.processing.document_parsing.target_directory
        )
        self.target_directory: Path | None = (
            Path(explicit_target) if explicit_target else None
        )
        # Composition root for ingestion: one provider bundle (session plus
        # LLM/embedding factories) built here or injected, and passed to every
        # stage, so an injected factory reaches every LLM/embedding adapter
        # and the default Bedrock clients are created once per pipeline.
        if providers is not None and boto_session is not None:
            raise ValueError(
                "Pass boto_session either to DataIngestionPipeline or on the "
                "Providers bundle (Providers(config, boto_session=...)), not both: "
                "the bundle's session is the one every stage uses."
            )
        self.providers = (
            providers
            if providers is not None
            else build_providers(config, boto_session, **overrides)
        )

        # The ingestion models the stages build below answer from (and add
        # to) the run's response cache, opened once the run's cache is in place.
        self.llm_response_cache: LLMResponseCache | None = None
        responses = config.cache.llm_responses
        if pipeline_config.cache_enabled and responses.enabled:
            self.llm_response_cache = LLMResponseCache(
                flush_every=responses.flush_every
            )
            self.providers.use_llm_response_cache(self.llm_response_cache)

        self._initialize_managers()
        self.stages, self.name_to_type_map = self._initialize_stages()

        logger.info(
            "Successfully initialized pipeline with %s stages: '%s'",
            len(self.stages),
            ", ".join([stage.name for stage in self.stages]),
        )

    def _initialize_managers(self) -> None:
        cache_config = self.config.cache
        chunking_config = cache_config.chunking

        # force_rebuild ignores existing cached stage outputs: FORCE_REFRESH makes
        # the cache treat every entry as a miss so stages actually re-execute.
        cache_strategy = (
            CacheStrategy.FORCE_REFRESH
            if self.pipeline_config.force_rebuild
            else CacheStrategy.CONTENT_HASH
        )
        self.cache_manager = CacheManager(
            config=self.config,
            cache_directory=self.pipeline_config.local_directory,
            strategy=cache_strategy,
            # An S3-synced run's checkpoints never expire here: the bucket
            # lifecycle expires them, and a 24 h TTL made a resume the next
            # day recompute every stage S3 still held.
            ttl_seconds=(
                None
                if self.pipeline_config.s3_sync_enabled
                else cache_config.ttl_seconds
            ),
            chunk_size=chunking_config.chunk_size,
            max_file_size_mb=chunking_config.max_file_size_mb,
            enable_chunking=chunking_config.enabled,
        )
        self.state_manager = PipelineStateManager(self.cache_manager)
        # The resume manager must reconstruct the same input-fingerprinted cache
        # keys the save path writes, so it needs the same config.
        self.resume_manager = PipelineResumeManager(self.state_manager, self.config)

        self.s3_cache_manager = None
        if self.pipeline_config.s3_sync_enabled:
            # Imported here: a run without S3 sync never loads the AWS SDK.
            from unified_kg_rag.adapters.aws.s3_cache import S3CacheManager

            self.s3_cache_manager = S3CacheManager(
                config=self.config,
                boto_session=self.providers.boto_session,
                bucket_name=self.pipeline_config.s3_bucket_name,
                prefix=self.pipeline_config.s3_prefix,
            )

    def _initialize_stages(
        self,
    ) -> tuple[list[PipelineStage], dict[str, PipelineStageType]]:
        # The doc-status registry is built once, here, and handed to the two
        # stages that use it (None when incremental indexing is off). Building
        # it does no I/O; a registry that cannot be reached fails the
        # document_loading stage (DocStatusRegistryError) rather than silently
        # turning the run into a full rebuild.
        doc_status = (
            self.providers.build_doc_status()
            if self.providers.doc_status_enabled
            else None
        )
        self._doc_status = doc_status
        cache_directory = Path(self.pipeline_config.local_directory)
        extra_kwargs: dict[PipelineStageType, dict[str, Any]] = {
            PipelineStageType.DOCUMENT_PARSING: {
                "source_directory": self.source_directory,
                "target_directory": self.target_directory,
                "cache_directory": cache_directory,
            },
            PipelineStageType.DOCUMENT_LOADING: {
                "source_directory": self.source_directory,
                "doc_status": doc_status,
            },
            # --force-rebuild re-extracts every unit (and refreshes the
            # extraction store) instead of reusing the stored records.
            PipelineStageType.GRAPH_EXTRACTION: {
                "reuse_stored": not self.pipeline_config.force_rebuild
            },
            PipelineStageType.CLAIM_EXTRACTION: {
                "reuse_stored": not self.pipeline_config.force_rebuild
            },
            PipelineStageType.COMMUNITY_DETECTION: {"cache_directory": cache_directory},
            PipelineStageType.INDEXING: {"doc_status": doc_status},
        }

        stages: list[PipelineStage] = []
        name_to_type_map: dict[str, PipelineStageType] = {}
        disabled_stages = []
        for stage_type, stage_class in self.STAGE_CLASSES.items():
            config_enabled = self.CONFIG_ENABLED.get(stage_type, lambda _: True)
            if not (
                self.pipeline_config.stages_enabled.get(stage_type, True)
                and config_enabled(self.config)
            ):
                disabled_stages.append(stage_type.value)
                continue
            stage = stage_class(
                config=self.config,
                providers=self.providers,
                **extra_kwargs.get(stage_type, {}),
            )
            stages.append(stage)
            name_to_type_map[stage.name] = stage_type

        if disabled_stages:
            logger.info("Disabled stages: %s", disabled_stages)

        return stages, name_to_type_map

    def close(self) -> None:
        """Release backend clients owned by the pipeline's stages.

        Only the indexing stage holds long-lived Neptune/OpenSearch connections;
        every other stage's ``close`` is a no-op. Best-effort: never raises, so
        the CLI can call it from a ``finally`` regardless of run outcome.
        """
        for stage in self.stages:
            try:
                stage.close()
            except Exception as e:  # noqa: BLE001 - teardown must never raise
                logger.debug("Error closing stage '%s': %s", stage.name, e)

    def run(
        self,
        source_directory: str | Path,
        pipeline_id: str | None = None,
        resume_from_stage: str | None = None,
    ) -> PipelineContext:
        """Run the pipeline over ``source_directory`` and return its context.

        Args:
            source_directory: The corpus root. When it differs from the
                directory the pipeline was built with, the parsing and loading
                stages (and with them the incremental registry's source scope)
                are rebuilt for it, so the run reads and diffs this corpus.
            pipeline_id: The run to resume (its cache and metadata); a new id
                is generated when neither this nor ``pipeline_config`` sets
                one.
            resume_from_stage: Stage name to restart from; ``None`` resumes
                after the last completed stage.

        Returns:
            The run's context. A failed stage is reported there (``status``
            ``FAILED`` and its stage result), not raised.

        Raises:
            PipelineExecutionError: The run could not be set up or crashed
                outside a stage (for example a missing source directory).
        """
        # The root span of an ingestion trace: stages, batches and model calls
        # nest under it. A failure leaving the block is recorded on it.
        with observability.span(
            "kg_rag.ingestion.run", resume_from_stage=resume_from_stage
        ) as span:
            context = self._run(source_directory, pipeline_id, resume_from_stage)
            span.set_attributes(
                {
                    "kg_rag.pipeline_id": context.pipeline_id,
                    "kg_rag.pipeline.status": context.status,
                    "kg_rag.stages_completed": sum(
                        r.status == PipelineStageStatus.COMPLETED
                        for r in context.stage_results
                    ),
                }
            )
            # _run reports a failed stage on the context instead of raising.
            if context.status == PipelineStageStatus.FAILED:
                span.set_error(f"pipeline '{context.pipeline_id}' failed")
            return context

    def _run(
        self,
        source_directory: str | Path,
        pipeline_id: str | None,
        resume_from_stage: str | None,
    ) -> PipelineContext:
        log_context: Mapping[str, Any] = {}
        self._stop_signal = None
        try:
            source_path = Path(source_directory).resolve()
            logger.info(
                "Starting pipeline run - source: '%s', pipeline_id: '%s', resume_from: '%s'",
                source_path,
                pipeline_id,
                resume_from_stage or "auto-detect",
            )

            self._validate_source_directory(source_path)
            self._use_source_directory(Path(source_directory))
            self._validate_retired_scopes(source_path)

            resolved_pipeline_id = self._resolve_pipeline_id(source_path, pipeline_id)
            # Every log line of the run, the S3 sync and the failure report
            # included, carries the id (also in worker threads, see
            # ContextThreadPoolExecutor).
            log_context = bind_contextvars(pipeline_id=resolved_pipeline_id)
            resolved_resume_stage = self.normalize_stage_name(
                resume_from_stage or self.pipeline_config.resume_from_stage
            )

            logger.info("Resolved pipeline ID: '%s'", resolved_pipeline_id)

            if self.pipeline_config.s3_sync_enabled:
                self._sync_cache_with_s3(resolved_pipeline_id, "download")
                self._checkpoint_id = resolved_pipeline_id

            # Part of every stage cache key, so it must be set before the
            # resume decision reads the cache.
            self._set_corpus_fingerprint(source_path)
            self._open_llm_response_cache(resolved_pipeline_id)

            # force_rebuild must ignore ALL existing cache: skip the resume path
            # entirely so the run re-executes from the beginning even when a
            # prior metadata file exists (FORCE_REFRESH alone only bypasses
            # per-stage cache reads, not the resume decision).
            can_resume = (
                resolved_pipeline_id
                and not self.pipeline_config.force_rebuild
                and self.state_manager.pipeline_exists(resolved_pipeline_id)
            )
            if can_resume:
                logger.info(
                    "Resuming existing pipeline '%s' from stage: '%s'",
                    resolved_pipeline_id,
                    resolved_resume_stage or "auto-detect",
                )
                context, start_stage_name = self._prepare_resume(
                    resolved_pipeline_id, resolved_resume_stage
                )
            else:
                if self.pipeline_config.force_rebuild:
                    logger.info(
                        "Force rebuild: ignoring any existing cache for '%s'",
                        resolved_pipeline_id,
                    )
                logger.info("Starting new pipeline run: '%s'", resolved_pipeline_id)
                context, start_stage_name = self._prepare_new_run(
                    source_path, resolved_pipeline_id, resolved_resume_stage
                )

            with self._namespace_lease():
                self._execute_pipeline(context, start_stage_name)

            if context.status == PipelineStageStatus.FAILED:
                logger.warning(
                    "Pipeline run finished with failed stages - ID: '%s'",
                    context.pipeline_id,
                )
            else:
                logger.info(
                    "Pipeline run completed - ID: '%s', Status: '%s'",
                    context.pipeline_id,
                    context.status.value,
                )
            return context

        except KeyboardInterrupt:
            self._checkpoint_on_stop()
            raise
        except NamespaceLeaseError:
            # Another run writes the namespace: nothing was read or written.
            raise
        except Exception as e:
            logger.exception("Pipeline execution failed: %s", e)
            raise PipelineExecutionError(f"Failed to run pipeline: {e}") from e
        finally:
            if self.llm_response_cache is not None:
                self.llm_response_cache.close()
            self._checkpoint_id = None
            reset_contextvars(**log_context)

    def _namespace_lease(self) -> AbstractContextManager[NamespaceLease | None]:
        """The write lease of the run's index namespace, held for the whole
        run when the doc-status registry is on (see ``namespace_lease``)."""
        if self._doc_status is None:
            return nullcontext()
        namespace = config_registry_namespace(
            self.config.indexing, self.config.processing.document_parsing.index_value
        )
        return NamespaceLease(
            self._doc_status, namespace, self.config.indexing.lease_seconds
        )

    def _open_llm_response_cache(self, pipeline_id: str) -> None:
        """Serve this run's ingestion model calls from the responses earlier
        runs of ``pipeline_id`` stored (none on a force rebuild)."""
        if self.llm_response_cache is None:
            return
        cache_dir = self.cache_manager.get_pipeline_cache_dir(pipeline_id)
        self.llm_response_cache.open(
            cache_dir / self.LLM_CACHE_DIRECTORY,
            lookups=not self.pipeline_config.force_rebuild,
            on_segment=(
                functools.partial(self._upload_file, pipeline_id, cache_dir)
                if self.s3_cache_manager is not None
                else None
            ),
        )

    def _upload_file(self, pipeline_id: str, cache_dir: Path, path: Path) -> None:
        """Upload one cache file now (best effort; the next checkpoint upload
        retries it). The response cache calls it from the model call that
        filled a segment, so a task killed mid-stage keeps those responses."""
        if self.s3_cache_manager is None:
            return
        try:
            self.s3_cache_manager.upload_file(pipeline_id, cache_dir, path)
        except CacheSyncError as e:
            logger.warning(
                "Upload of '%s' failed; the next checkpoint upload retries it: %s",
                path.name,
                e,
            )

    def request_stop(self, signum: int) -> None:
        """End the run: raise ``PipelineInterrupted``, uploading the
        checkpoint as the run unwinds.

        Meant for a signal handler in the main thread; ``run-ingestion``
        installs one for SIGTERM, which ECS sends on StopTask and when a Step
        Functions execution times out, before it kills the task. Every stage
        the run finished is in the checkpoint, so a re-run with the same
        pipeline id resumes at the stage that was running. A request made while
        a checkpoint is being written takes effect once it is written, so the
        stage that just finished is kept too.

        It does no I/O itself: a handler runs wherever the main thread was,
        possibly holding a lock (an executor's, say) the upload would wait on.
        """
        self._stop_signal = signum
        if not self._checkpointing:
            raise PipelineInterrupted(signum)

    @contextmanager
    def _checkpoint_write(self) -> Iterator[None]:
        """Defer a requested stop until the enclosed checkpoint write is done."""
        self._checkpointing = True
        try:
            yield
        finally:
            self._checkpointing = False
        if self._stop_signal is not None:
            raise PipelineInterrupted(self._stop_signal)

    def _checkpoint_on_stop(self) -> None:
        """Upload the checkpoint of a run that is stopping (``request_stop``
        or Ctrl-C), once the stage it interrupted has unwound."""
        if self._checkpoint_id is None:
            return
        logger.warning(
            "Run stopped (signal %s): uploading the checkpoint of '%s'",
            self._stop_signal or "SIGINT",
            self._checkpoint_id,
        )
        # A second stop request during the upload waits for it.
        self._checkpointing = True
        try:
            self._upload_checkpoint(self._checkpoint_id)
        finally:
            self._checkpointing = False

    def _upload_checkpoint(self, pipeline_id: str) -> None:
        """Upload the run's checkpoint to S3 (best effort).

        Files that fail are uploaded again by the next upload, and the final
        one after the last stage fails the run, so a transient S3 error does
        not stop the stages in between.
        """
        try:
            self._sync_cache_with_s3(pipeline_id, "upload")
        except CacheSyncError as e:
            logger.warning(
                "Checkpoint upload failed for '%s'; the next upload retries it: %s",
                pipeline_id,
                e,
            )

    @classmethod
    def normalize_stage_name(cls, stage_name: str | None) -> str | None:
        """Return the canonical (lowercase) name of ``stage_name``.

        Raises:
            ConfigurationError: ``stage_name`` names no pipeline stage.
        """
        if stage_name is None:
            return None
        normalized = stage_name.strip().lower()
        valid = [stage.value for stage in cls.STAGE_CLASSES]
        if normalized not in valid:
            raise ConfigurationError(
                f"Unknown pipeline stage '{stage_name}'. Valid stages: "
                f"{', '.join(valid)}"
            )
        return normalized

    @property
    def executed_stages(self) -> frozenset[str]:
        """Names of the stages the last run executed; the other completed
        stages of a resumed run were restored from the stage cache."""
        return self._executed_stages

    def _use_source_directory(self, source_directory: Path) -> None:
        """Point the stages at ``source_directory`` if they read another one.

        The parsing and loading stages capture the corpus root when they are
        built (parsing and discovery read it; the loading stage derives
        document ids and the registry's source scope from it), so running over
        another directory rebuilds those two rather than ingesting and diffing
        the old corpus. The other stages, which hold the backend clients, are
        kept.
        """
        if Path(self.source_directory).resolve() == source_directory.resolve():
            return
        logger.info(
            "Source directory changed from '%s' to '%s'; rebuilding the parsing "
            "and loading stages",
            self.source_directory,
            source_directory,
        )
        self.source_directory = source_directory
        for index, stage in enumerate(self.stages):
            if isinstance(stage, DocumentParsingStage):
                self.stages[index] = type(stage)(
                    config=self.config,
                    providers=self.providers,
                    source_directory=source_directory,
                    target_directory=stage.target_directory,
                    cache_directory=stage.cache_directory,
                )
            elif isinstance(stage, DocumentLoadingStage):
                self.stages[index] = type(stage)(
                    config=self.config,
                    providers=self.providers,
                    source_directory=source_directory,
                    doc_status=stage._doc_status,
                )

    def _set_corpus_fingerprint(self, source_directory: Path) -> None:
        self.corpus_fingerprint = self._compute_corpus_fingerprint(source_directory)
        self.resume_manager.corpus_fingerprint = self.corpus_fingerprint

    def validate_resume_integrity(
        self, source_directory: str | Path, pipeline_id: str
    ) -> tuple[bool, list[str]]:
        """Check that every completed stage of ``pipeline_id`` is still cached.

        Uses the cache keys the run itself will read, so the corpus
        fingerprint of ``source_directory`` is computed first. A failed check
        is not fatal for an auto resume: the run rewinds to the first stage
        whose output is missing and recomputes it.
        """
        self._set_corpus_fingerprint(Path(source_directory).resolve())
        return self.resume_manager.validate_pipeline_integrity(pipeline_id)

    def _compute_corpus_fingerprint(self, source_directory: Path) -> str:
        """Fingerprint the files this run can ingest (see cache_keys)."""
        extensions = {*ParserFactory.get_supported_extensions(), ".json"}
        owned = [Path(self.pipeline_config.local_directory)]
        if self.target_directory is not None:
            owned.append(self.target_directory)
        fingerprint = corpus_manifest_fingerprint(source_directory, extensions, owned)
        logger.info("Corpus manifest fingerprint: '%s'", fingerprint)
        return fingerprint

    @staticmethod
    def _validate_source_directory(source_directory: Path) -> None:
        if not source_directory.exists() or not source_directory.is_dir():
            logger.error("Source directory validation failed: '%s'", source_directory)
            raise FileNotFoundError(
                f"Source directory not found or not a directory: '{source_directory}'"
            )

    def _validate_retired_scopes(self, source_directory: Path) -> None:
        """Fail before any stage runs when ``indexing.retire_source_scopes``
        cannot be applied: without the doc-status registry, or when it names
        the run's own source scope (the loading stage checks again against
        the scope it actually uses)."""
        retired = self.config.indexing.retire_source_scopes
        if not retired:
            return
        if not self.providers.doc_status_enabled:
            raise ConfigurationError(
                "indexing.retire_source_scopes / --retire-source-scope needs "
                "incremental indexing (aws.dynamodb.enabled: true, or "
                "providers.doc_status.enabled: true): the retired records live "
                "in the doc-status registry"
            )
        parsing = self.config.processing.document_parsing
        validate_retired_scopes(
            retired,
            parsing.source_scope or source_directory.as_posix(),
            config_registry_namespace(self.config.indexing, parsing.index_value),
        )

    def _resolve_pipeline_id(
        self, source_directory: Path, pipeline_id: str | None
    ) -> str:
        # The id becomes a cache directory and an S3 prefix: no "/" or "..".
        return validate_path_segment(
            pipeline_id
            or self.pipeline_config.pipeline_id
            or self._generate_pipeline_id(source_directory),
            "pipeline_id",
        )

    @staticmethod
    def _generate_pipeline_id(source_directory: Path | None = None) -> str:
        # A random suffix keeps two runs started within the same second (one
        # per corpus subdirectory, say) from sharing a cache directory.
        timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
        if source_directory is None:
            return f"pipeline-{timestamp}"

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
        """Download (before the run) or upload (after each stage) the run's
        cache via S3.

        A failed or partial sync raises ``CacheSyncError``; ``run()`` turns it
        into ``PipelineExecutionError`` and the CLI exits non-zero, so a run
        that lost its checkpoint fails instead of the next one recomputing.
        """
        if not self.s3_cache_manager:
            logger.info("S3 cache manager not available, skipping sync")
            return

        cache_dir = self.cache_manager.get_pipeline_cache_dir(pipeline_id)

        try:
            if direction == "upload":
                if self.llm_response_cache is not None:
                    self.llm_response_cache.flush()
                self.s3_cache_manager.sync_pipeline_to_s3(pipeline_id, cache_dir)
            else:
                self.s3_cache_manager.sync_pipeline_from_s3(pipeline_id, cache_dir)
            logger.info(
                "S3 sync %s completed successfully for pipeline '%s'",
                direction,
                pipeline_id,
            )
        except Exception as e:
            logger.error(
                "Failed to sync cache %s S3 for pipeline '%s': %s",
                direction,
                pipeline_id,
                e,
            )
            raise

    def _prepare_resume(
        self, pipeline_id: str, resume_from_stage: str | None
    ) -> tuple[PipelineContext, str | None]:
        logger.info(
            "Preparing pipeline resume for '%s' from stage: '%s'",
            pipeline_id,
            resume_from_stage or "auto-detect",
        )

        try:
            start_stage_name, completed_stages = (
                self.resume_manager.determine_resume_strategy(
                    pipeline_id, resume_from_stage
                )
            )
            if resume_from_stage is None:
                start_stage_name, completed_stages = self._rewind_to_unrun_stage(
                    start_stage_name, completed_stages
                )
                start_stage_name, completed_stages = self._rewind_on_extraction_change(
                    pipeline_id, start_stage_name, completed_stages
                )

            self._check_upstream_complete(pipeline_id, start_stage_name)
            context = self.resume_manager.restore_pipeline_context(
                pipeline_id, completed_stages
            )

            self._prepare_context_for_resume(context, start_stage_name)

            logger.info(
                "Pipeline resume prepared successfully - will start from stage: '%s'",
                start_stage_name,
            )
            return context, start_stage_name
        except (PipelineResumeError, PipelineStateError) as e:
            logger.error(
                "Failed to prepare resume for pipeline '%s': %s", pipeline_id, e
            )
            raise PipelineExecutionError(f"Pipeline resume failed: {e}") from e

    def _rewind_to_unrun_stage(
        self, start_stage_name: str, completed_stages: list[str]
    ) -> tuple[str, list[str]]:
        """Move an auto resume back to an enabled stage that never completed.

        A stage switched on since the pipeline last ran (gleaning or claim
        extraction, say) has no recorded result, so the resume point the
        recorded stages give lies past it and the stage would never run.
        The resume starts at the first such stage instead, and only the
        completed stages upstream of it are restored.
        """
        canonical_order = [st.value for st in self.STAGE_CLASSES]
        if start_stage_name not in canonical_order:
            return start_stage_name, completed_stages
        start_position = canonical_order.index(start_stage_name)
        completed = set(completed_stages)
        for stage in self.stages:
            position = canonical_order.index(stage.name)
            if position >= start_position:
                break
            if stage.name not in completed:
                logger.warning(
                    "Stage '%s' is enabled but has not completed in this "
                    "pipeline (it was off when the pipeline last ran); resuming "
                    "from it instead of '%s'",
                    stage.name,
                    start_stage_name,
                )
                return stage.name, [
                    name
                    for name in completed_stages
                    if canonical_order.index(name) < position
                ]
        return start_stage_name, completed_stages

    def _rewind_on_extraction_change(
        self, pipeline_id: str, start_stage_name: str, completed_stages: list[str]
    ) -> tuple[str, list[str]]:
        """Move an auto resume back to document_loading when the extraction
        settings changed since its delta was computed.

        document_loading's cache key does not cover the extraction settings,
        so a resume past it keeps the previous run's delta and extraction
        fingerprint: the stages that rerun (a newly enabled stage, or one
        whose inputs changed) see only that delta's documents, and the
        commit records the old fingerprint, so the next run re-extracts
        every document. The same holds when the doc-status registry was
        turned off since: the restored delta would write records to the
        registry the user disabled. Rerunning document_loading diffs the
        corpus with this run's settings (or skips the diff without a
        registry); document_parsing stays cached.
        """
        loading = PipelineStageType.DOCUMENT_LOADING.value
        canonical_order = [st.value for st in self.STAGE_CLASSES]
        if (
            start_stage_name not in canonical_order
            or canonical_order.index(start_stage_name) <= canonical_order.index(loading)
            or loading not in {stage.name for stage in self.stages}
        ):
            return start_stage_name, completed_stages
        metadata = self.state_manager.load_pipeline_metadata(pipeline_id)
        recorded = metadata.get("incremental_extraction_settings")
        if recorded is None:
            return start_stage_name, completed_stages
        changed = changed_settings(
            ExtractionSettings.model_validate(recorded),
            extraction_settings(self.config),
        )
        if not self.providers.doc_status_enabled:
            reason = "the doc-status registry was disabled"
        elif changed:
            reason = f"the extraction settings changed ({', '.join(changed)})"
        else:
            return start_stage_name, completed_stages
        logger.warning(
            "%s since this pipeline computed its incremental delta; resuming "
            "from '%s' instead of '%s'",
            reason.capitalize(),
            loading,
            start_stage_name,
        )
        position = canonical_order.index(loading)
        return loading, [
            name
            for name in completed_stages
            if name in canonical_order and canonical_order.index(name) < position
        ]

    def _check_upstream_complete(self, pipeline_id: str, start_stage_name: str) -> None:
        """Fail when the resume point is a recorded stage upstream of every
        enabled stage: an earlier stage failed or lost its cached output, so
        this run's stages (a later phase, say) would run on missing inputs.
        A resume point downstream of the enabled stages means they are all
        done, and the run skips them."""
        if not self.stages:
            return
        canonical_order = [st.value for st in self.STAGE_CLASSES]
        if start_stage_name not in canonical_order or start_stage_name in {
            stage.name for stage in self.stages
        }:
            return
        first_enabled = canonical_order.index(self.stages[0].name)
        if canonical_order.index(start_stage_name) > first_enabled:
            return
        metadata = self.state_manager.load_pipeline_metadata(pipeline_id)
        recorded = {r.get("stage_name") for r in metadata.get("stage_results", [])}
        if start_stage_name in recorded:
            raise PipelineResumeError(
                f"Stage '{start_stage_name}', upstream of this run's enabled "
                f"stages, has not completed (or its cached output is gone). "
                f"Run it first, e.g. with --enabled-stages including "
                f"'{start_stage_name}'."
            )

    def _prepare_context_for_resume(
        self, context: PipelineContext, start_stage_name: str
    ) -> None:
        # Order by the canonical pipeline sequence (STAGE_CLASSES), NOT the
        # enabled-stage subset: phased execution narrows self.stages to one
        # phase's window (e.g. just ["indexing"]), so judging "is this an
        # upstream result?" against that subset would wrongly discard every
        # earlier phase's results — silently dropping resolved_relationships /
        # resolved_claims before indexing. The full order keeps them.
        canonical_order = [st.value for st in self.STAGE_CLASSES]
        try:
            start_index = canonical_order.index(start_stage_name)
        except ValueError:
            logger.warning(
                "Start stage '%s' not in canonical pipeline order; "
                "keeping all %s previous stage results",
                start_stage_name,
                len(context.stage_results),
            )
            return

        valid_results = [
            res
            for res in context.stage_results
            if res.stage_name in canonical_order
            and canonical_order.index(res.stage_name) < start_index
        ]
        context.stage_results = valid_results
        logger.info(
            "Context prepared for resume from '%s' - keeping %s previous stage results",
            start_stage_name,
            len(valid_results),
        )

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
        self._executed_stages = frozenset()
        self.state_manager.mark_rerun(None)
        self._execute_pipeline_stages(context, start_stage_name)
        self._finalize_pipeline_execution(context, total_start_time)

    def _execute_pipeline_stages(
        self, context: PipelineContext, start_stage_name: str | None
    ) -> None:
        start_index = self._start_index(start_stage_name)

        completed_stage_names = {
            r.stage_name
            for r in context.stage_results
            if r.status == PipelineStageStatus.COMPLETED
        }

        if completed_stage_names:
            logger.info(
                "Found %s previously completed stages: '%s'",
                len(completed_stage_names),
                ", ".join(list(completed_stage_names)),
            )

        stages_to_execute = self.stages[start_index:]
        logger.info(
            "Executing %s stages: '%s'",
            len(stages_to_execute),
            ", ".join([s.name for s in stages_to_execute]),
        )

        for stage in stages_to_execute:
            if stage.name in completed_stage_names:
                logger.info("Skipping previously completed stage: '%s'", stage.name)
            else:
                logger.info("Executing stage: '%s'", stage.name)
                result = self._execute_stage(stage, context)
                self._emit_stage_metrics(context, stage.name)
                with self._checkpoint_write():
                    if self.llm_response_cache is not None:
                        self.llm_response_cache.flush()
                    if self.s3_cache_manager is not None:
                        # Durable now, not only after the last stage: a task
                        # stopped later resumes from here.
                        self._upload_checkpoint(context.pipeline_id)

                if self._should_stop_pipeline(result):
                    logger.warning(
                        "Pipeline execution stopped due to stage failure: '%s'",
                        stage.name,
                    )
                    break

    def _start_index(self, start_stage_name: str | None) -> int:
        """Index in ``self.stages`` of the first stage to run.

        A start stage that is not enabled in this run (a phased run's window
        omits it, or its config section turns it off) maps to the first
        enabled stage at or after it in pipeline order; none means there is
        nothing left to run.
        """
        stage_names = [stage.name for stage in self.stages]
        if not start_stage_name:
            return 0
        if start_stage_name in stage_names:
            return stage_names.index(start_stage_name)
        canonical_order = [st.value for st in self.STAGE_CLASSES]
        if start_stage_name not in canonical_order:
            logger.warning(
                "Start stage '%s' not found, starting from beginning",
                start_stage_name,
            )
            return 0
        start_position = canonical_order.index(start_stage_name)
        for index, name in enumerate(stage_names):
            if canonical_order.index(name) >= start_position:
                logger.info(
                    "Start stage '%s' is not enabled in this run; starting "
                    "from '%s'",
                    start_stage_name,
                    name,
                )
                return index
        logger.info(
            "Resume point '%s' is past every enabled stage; nothing to run",
            start_stage_name,
        )
        return len(stage_names)

    def _execute_stage(
        self, stage: PipelineStage, context: PipelineContext
    ) -> PipelineStageResult:
        self._executed_stages = self._executed_stages | {stage.name}
        # The persisted results downstream of this stage are now stale.
        self.state_manager.mark_rerun(stage.name)
        # A failure recorded for this stage by an earlier attempt is replaced
        # by what this execution reports.
        context.failed_text_unit_ids.pop(stage.name, None)
        blocked_by = self._failed_corpus_stage(stage, context)
        if blocked_by is not None:
            result = PipelineStageResult(
                stage_name=stage.name,
                status=PipelineStageStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=(
                    f"Not indexing: stage '{blocked_by}' failed, so this run "
                    "does not hold every document's text units; committing "
                    "would record documents as processed without their content, "
                    "or delete them as missing from the corpus. Fix the failure "
                    "and resume with the same pipeline id."
                ),
            )
            logger.error("%s", result.error_message)
            self.state_manager.update_stage_result(context, result)
            return result
        result = self._run_stage(stage, context)
        if result.status == PipelineStageStatus.FAILED:
            self._mark_units_failed(stage, context)
        return result

    def _failed_corpus_stage(
        self, stage: PipelineStage, context: PipelineContext
    ) -> str | None:
        """The failed corpus stage that blocks ``stage`` (indexing), if any."""
        if self.name_to_type_map.get(stage.name) is not PipelineStageType.INDEXING:
            return None
        corpus_stages = {st.value for st in self.CORPUS_STAGES}
        for result in context.stage_results:
            if (
                result.stage_name in corpus_stages
                and result.status == PipelineStageStatus.FAILED
            ):
                return result.stage_name
        return None

    def _mark_units_failed(
        self, stage: PipelineStage, context: PipelineContext
    ) -> None:
        """Record every text unit as failed for a crashed artifact stage.

        Under continue_on_error the run goes on to indexing; without this the
        incremental commit would record the documents PROCESSED, and the next
        run would skip them as unchanged although their graph is partial.
        """
        if self.name_to_type_map.get(stage.name) not in self.ARTIFACT_STAGES:
            return
        units = context.translated_units or context.text_units or []
        context.failed_text_unit_ids[stage.name] = [unit.id for unit in units]
        self.state_manager.save_pipeline_metadata(context)
        if units:
            logger.warning(
                "Stage '%s' failed: recording all %s text units as failed so "
                "their documents are not committed as processed",
                stage.name,
                len(units),
            )

    def _run_stage(
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
            with self._checkpoint_write():
                self.state_manager.update_stage_result(context, result)
                if result.status == PipelineStageStatus.COMPLETED:
                    self._save_stage_outputs_to_cache(context, stage.name)
            if result.status == PipelineStageStatus.FAILED:
                logger.error("Stage '%s' failed: %s", stage.name, result.error_message)

            return result

        except Exception as e:
            logger.error(
                "Unexpected error in stage '%s': %s", stage.name, e, exc_info=True
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
                "Cache disabled, skipping output save for stage: '%s'", stage_name
            )
            return

        stage_type = self.name_to_type_map.get(stage_name)
        if not stage_type or stage_type not in self.STAGE_OUTPUT_MAPPING:
            logger.info(
                "No output mapping found for stage: '%s', skipping cache save",
                stage_name,
            )
            return

        try:
            saved_outputs = []
            for _, (_, context_attr) in self.STAGE_OUTPUT_MAPPING[stage_type].items():
                data_to_save = getattr(context, context_attr, None)
                # An empty list is still this stage's output under these
                # inputs (translation is off by default and completes with no
                # translated units). It is saved so that the resume can tell
                # "produced nothing" from "not in the cache" and does not
                # rewind to this stage on every run.
                if data_to_save is None:
                    logger.debug(
                        "No data found for context attribute: %s", context_attr
                    )
                    continue

                logger.info(
                    "Saving %s %s to cache",
                    len(data_to_save) if isinstance(data_to_save, list) else 1,
                    context_attr,
                )

                # The key carries a fingerprint of the inputs that produced this
                # output, so a changed prompt/model/config reads as a miss on the
                # next run instead of silently resuming from stale output.
                cache_entry = self.cache_manager.save_stage_result(
                    data=data_to_save,
                    cache_key=stage_cache_key(
                        self.config,
                        stage_type,
                        context_attr,
                        self.corpus_fingerprint,
                    ),
                    stage_name=stage_name,
                    pipeline_id=context.pipeline_id,
                    metadata={
                        "stage_type": stage_type.value,
                        "context_attribute": context_attr,
                    },
                )

                if cache_entry:
                    item_count = cache_entry.record_count or (
                        len(data_to_save) if isinstance(data_to_save, list) else 1
                    )
                    saved_outputs.append(f"{item_count} {context_attr}")
                    logger.info("Successfully cached %s", context_attr)
                else:
                    logger.warning("Cache manager returned None for %s", context_attr)

            if saved_outputs:
                logger.info(
                    "Stage '%s' outputs cached: %s",
                    stage_name,
                    ", ".join(saved_outputs),
                )
        except Exception as e:
            logger.warning(
                "Failed to save outputs to cache for stage '%s': %s", stage_name, e
            )

    def _should_stop_pipeline(self, result: PipelineStageResult) -> bool:
        if result.status == PipelineStageStatus.FAILED:
            if self.pipeline_config.continue_on_error:
                logger.warning(
                    "Stage '%s' failed but continuing due to continue_on_error setting: %s",
                    result.stage_name,
                    result.error_message,
                )
                return False
            else:
                logger.error(
                    "Stopping pipeline due to failure in stage '%s': %s",
                    result.stage_name,
                    result.error_message,
                )
                return True
        return False

    def _finalize_pipeline_execution(
        self, context: PipelineContext, start_time: float
    ) -> None:
        logger.info("Finalizing pipeline execution - ID: '%s'", context.pipeline_id)

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
                "Pipeline completed with %s failed stages: %s",
                len(failed_stages),
                [r.stage_name for r in failed_stages],
            )
        else:
            context.status = PipelineStageStatus.COMPLETED
            logger.info("Pipeline completed successfully with all stages passed")

        self.state_manager.save_pipeline_metadata(context)

        if self.pipeline_config.s3_sync_enabled:
            try:
                with self._checkpoint_write():
                    self._sync_cache_with_s3(context.pipeline_id, "upload")
            except CacheSyncError:
                # The stages (and indexing) already ran; the failure is only the
                # checkpoint upload. Log the summary before failing the run so
                # operators can see what was processed and indexed.
                logger.error(
                    "S3 cache upload failed after the pipeline stages finished "
                    "for '%s'; the summary below shows what this run processed",
                    context.pipeline_id,
                )
                self._log_pipeline_summary(context)
                raise

        self._log_pipeline_summary(context)

    def _create_pipeline_metrics(self, context: PipelineContext) -> PipelineMetrics:
        cache_stats = self.cache_manager.get_cache_stats(context.pipeline_id)
        stage_durations, stage_throughput = self._calculate_stage_performance(context)

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
            "gleaning_improvement_rate": self._get_stage_metric(
                context, "gleaning", "improvement_rate"
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
            "total_items_indexed": int(
                self._get_stage_metric(context, "indexing", "total_indexed")
            ),
            # Include a hard-failure sentinel: when the indexing stage RAISES
            # (e.g. clear/initialize/connection error), its result carries no
            # metrics, so "total_failed" reads 0 and the IndexingFailures alarm
            # (which watches this metric) can never fire on a total indexing
            # failure. Emit >=1 in that case so the alarm surfaces the crash.
            "total_items_index_failed": int(
                self._get_stage_metric(context, "indexing", "total_failed")
            )
            or (1 if self._stage_failed(context, "indexing") else 0),
            "relationships_indexed": int(
                self._get_stage_metric(context, "indexing", "relationships_indexed")
            ),
            "community_drift_overlap_share": self._get_stage_metric(
                context, "indexing", "community_drift_overlap_share"
            ),
            "community_rebuild_recommended": int(
                self._get_stage_metric(
                    context, "indexing", "community_rebuild_recommended"
                )
            ),
            # Per-chunk LLM failures degrade a run without failing it; surface
            # them so the ExtractionFailures alarm can catch a partial graph.
            "total_extraction_failures": int(
                self._get_stage_metric(context, "graph_extraction", "failed_units")
            ),
            "total_gleaning_failures": int(
                self._get_stage_metric(context, "gleaning", "failed_units")
            ),
            "total_claim_extraction_failures": int(
                self._get_stage_metric(context, "claim_extraction", "failed_units")
            ),
            # Documents an incremental run stopped retrying because their
            # indexing kept dying (the DocumentsQuarantined alarm).
            "total_documents_quarantined": int(
                self._get_stage_metric(
                    context, "document_loading", "documents_quarantined"
                )
            ),
            "cache_hit_rate": cache_stats.hit_rate,
            "cache_size_mb": cache_stats.total_size_mb,
            "stage_durations": stage_durations,
            "stage_throughput": stage_throughput,
        }
        return PipelineMetrics(**metric_data)

    def _emit_metrics(
        self,
        metrics: PipelineMetrics,
        pipeline_id: str,
        stage_name: str | None = None,
    ) -> None:
        """Forward scalar pipeline metrics to the configured sink (best-effort).

        With ``stage_name`` (right after that stage ran): the metrics it
        completes (``METRIC_STAGES``, emitted after the last of their stages
        this run has enabled) and its duration, so a run that is stopped or
        times out still reports the stages it finished. Without: the metrics
        of the whole run, when it ends.
        """
        enabled = [stage.name for stage in getattr(self, "stages", [])]

        def completes(stages: frozenset[str]) -> str | None:
            # The last enabled stage of ``stages``, in pipeline order.
            last = [name for name in enabled if name in stages]
            return last[-1] if last else None

        def emitted(name: str) -> bool:
            stages = self.METRIC_STAGES.get(name)
            if stage_name is None:
                return stages is None
            return stages is not None and completes(stages) == stage_name

        try:
            dumped = metrics.model_dump()
            scalars = {
                k: v
                for k, v in dumped.items()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and emitted(k)
            }
            # The per-stage duration map is flattened into scalar metrics (the
            # dict member is otherwise dropped by the scalar filter).
            duration = (metrics.stage_durations or {}).get(stage_name or "")
            if isinstance(duration, (int, float)):
                scalars[f"duration_{stage_name}_seconds"] = duration
            self.metrics_sink.emit(
                namespace="unified_kg_rag/ingestion",
                metrics=scalars,
                dimensions={"pipeline_id": pipeline_id},
            )
        except Exception as e:  # never let metrics emission break a run
            logger.warning("Metrics emission failed: %s", e)

    def _emit_stage_metrics(self, context: PipelineContext, stage_name: str) -> None:
        """Emit what ``stage_name`` completes, right after it ran."""
        try:
            metrics = self._create_pipeline_metrics(context)
        except Exception as e:  # never let metrics emission break a run
            logger.warning("Metrics emission failed: %s", e)
            return
        self._emit_metrics(metrics, context.pipeline_id, stage_name)

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

    def verify_pipeline_metadata(self, pipeline_id: str) -> bool:
        logger.info("Verifying pipeline metadata for: %s", pipeline_id)

        try:
            result = self.state_manager.verify_pipeline_metadata(pipeline_id)
            if result:
                logger.info(
                    "Pipeline metadata verification successful for: %s", pipeline_id
                )
            else:
                logger.warning(
                    "Pipeline metadata verification failed for: %s", pipeline_id
                )
            return result
        except Exception as e:
            logger.error(
                "Error during pipeline metadata verification for %s: %s", pipeline_id, e
            )
            return False
