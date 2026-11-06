# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import secrets
import signal
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel
from structlog.contextvars import bind_contextvars, reset_contextvars

from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.adapters.llm.response_cache import LLMResponseCache
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.bootstrap import (
    CheckpointSync,
    build_checkpoint_sync,
    build_providers,
    load_seed_graphs,
    resolve_extraction_schema,
)
from unified_kg_rag.application.ingestion import pipeline_metrics, resume
from unified_kg_rag.application.ingestion.cache_manager import (
    CacheManager,
)
from unified_kg_rag.application.ingestion.incremental import (
    close_attempts,
    open_attempts,
    other_source_scopes,
    record_item_failures,
    retired_scope_readings,
)
from unified_kg_rag.application.ingestion.namespace_lease import (
    NamespaceLease,
    longest_write_request_seconds,
)
from unified_kg_rag.application.ingestion.pipeline_manager import (
    STAGE_OUTPUTS,
    PipelineResumeManager,
    PipelineStateManager,
)
from unified_kg_rag.application.ingestion.pipeline_metrics import (
    LLM_USAGE_METRIC,
)
from unified_kg_rag.application.ingestion.pipeline_metrics import (
    run_token_usage as run_token_usage,
)
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
from unified_kg_rag.application.ingestion.summary_tree_stage import SummaryTreeStage
from unified_kg_rag.application.observability import configure_observability
from unified_kg_rag.domain.ingestion.cache_keys import (
    corpus_manifest_fingerprint,
    seed_corpus_fingerprint,
    stage_cache_key,
)
from unified_kg_rag.domain.ingestion.delta_detector import retired_scope_groups
from unified_kg_rag.domain.ingestion.document_identity import config_registry_namespace
from unified_kg_rag.domain.ingestion.methodologies import (
    missing_artifacts,
    selected_stages,
    stage_methodologies,
)
from unified_kg_rag.domain.ingestion.seed_graph import SeedConversion
from unified_kg_rag.domain.models import (
    CacheStrategy,
    Config,
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
    ConfirmationDeclinedError,
    NamespaceLeaseError,
    PipelineExecutionError,
    PipelineInterrupted,
    get_logger,
    observability,
)
from unified_kg_rag.shared.exceptions import StoreConnectionError
from unified_kg_rag.shared.metrics import MetricsSink, NullMetricsSink
from unified_kg_rag.shared.store_lookups import (
    install_lookup_cache,
    remove_lookup_cache,
)
from unified_kg_rag.shared.usage import usage_scope
from unified_kg_rag.shared.utils import compute_hash
from unified_kg_rag.shared.write_fence import (
    check_fences_held,
    install_write_fence,
    remove_write_fence,
)

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class _StopFence:
    """The write fence of a pipeline run that ends its worker threads' writes
    once a stop was requested (``request_stop``, or Ctrl-C). The thread that
    runs the pipeline gets the ``PipelineInterrupted`` (or
    ``KeyboardInterrupt``) itself, and its unwinding (closing the attempts,
    uploading the checkpoint) still writes."""

    def __init__(self, pipeline: DataIngestionPipeline) -> None:
        self._pipeline = pipeline
        self._run_thread = threading.current_thread()

    def check(self) -> None:
        signum = self._pipeline._stop_signal
        if signum is not None and threading.current_thread() is not self._run_thread:
            raise PipelineInterrupted(signum)

    # Nothing to hold back: the same check where nothing is written.
    check_held = check


class _Borrowed:
    """An indexer the pipeline uses but does not own: every attribute is the
    indexer's, and ``close`` does nothing (its owner closes it)."""

    def __init__(self, indexer: Any) -> None:
        self._indexer = indexer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._indexer, name)

    def close(self) -> None:
        return None


class DataIngestionPipeline:
    """The ingestion pipeline: documents in, an indexed knowledge graph out.

    Runs the stages of ``STAGE_CLASSES`` in order (parse, load, chunk,
    translate, extract, glean, resolve, claims, analyze, communities, index,
    summary tree),
    skipping those ``pipeline_config.stages_enabled``, a config section
    (``CONFIG_ENABLED``) or the selected ``indexing.methodologies`` turn off
    (``_stage_enabled``). Each stage's output is cached under the
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
        PipelineStageType.SUMMARY_TREE: SummaryTreeStage,
    }

    # Stage names in run order.
    _ORDER: tuple[str, ...] = tuple(stage.value for stage in STAGE_CLASSES)

    @classmethod
    def _before(cls, stage_names: Iterable[str], stage: str) -> list[str]:
        """The names in ``stage_names`` of the stages that run before
        ``stage``, in their order (the completed stages a rewind keeps)."""
        position = cls._ORDER.index(stage)
        return [
            name
            for name in stage_names
            if name in cls._ORDER and cls._ORDER.index(name) < position
        ]

    # Stages switched off by a config section besides stages_enabled and the
    # selected methodologies (_stage_enabled). Claim extraction and
    # resolution go together; community detection builds what graphrag
    # alone reads.
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

    # Names of the stages the current run executed (not skipped as completed).
    _executed_stages: frozenset[str] = frozenset()
    # The doc-status registry handed to the stages (None when off); the run
    # holds its namespace's write lease in it.
    _doc_status: DocStatusPort | None = None

    # Fingerprint of the run's corpus contents, computed in run(); part of every
    # stage cache key so a changed corpus is a cache miss, not a stale resume.
    corpus_fingerprint: str | None = None
    # The seed graphs the run imports, converted
    # (application.bootstrap.load_seed_graphs): part of its corpus, so their
    # content is folded into the corpus fingerprint. None imported yet.
    seed_conversions: tuple[SeedConversion, ...] = ()
    # Asks the questions a run needs answered (see __init__'s confirm and
    # confirm_deletions).
    _confirm: Callable[[str], bool] | None = None
    _confirm_deletions: Callable[[str], bool] | None = None
    # The id of the latest run, once resolved (the CLI reports it on a stop).
    pipeline_id: str | None = None

    # Stop handling (see request_stop): the run's pipeline id once its S3
    # checkpoint is in place, whether a checkpoint is being written, and the
    # signal of a stop requested meanwhile.
    s3_cache_manager: CheckpointSync | None = None
    llm_response_cache: LLMResponseCache | None = None
    # Where the response cache keeps its segments, in the run's cache directory.
    LLM_CACHE_DIRECTORY = "llm-cache"
    _checkpoint_id: str | None = None
    # The run rebuilds the namespace to add an artifact it lacks
    # (_rebuild_for_added_artifacts): indexing.reset is set for it alone.
    _reset_for_run: bool = False
    _checkpointing: bool = False
    _stop_signal: int | None = None

    def __init__(
        self,
        config: Config,
        pipeline_config: PipelineConfig,
        source_directory: Path | None = None,
        target_directory: Path | None = None,
        *,
        metrics_sink: MetricsSink | None = None,
        providers: Providers | None = None,
        doc_status: DocStatusPort | None = None,
        vector_indexer: VectorIndexer | None = None,
        graph_indexer: GraphIndexer | None = None,
        confirm: Callable[[str], bool] | None = None,
        confirm_deletions: Callable[[str], bool] | None = None,
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
                ``providers`` instead when passing one. The caller owns the
                indexers given here: the pipeline never closes them.
            confirm: Asks a yes/no question a run needs answered, today only
                a reset that drops another source scope's seed graph
                documents, or, from a corpus of seed graphs alone, its
                documents, and a run with the registry off that rebuilds an
                index holding items from seed graphs alone (the CLI asks on
                the terminal; ``--confirm-reset``,
                ``UKR_CONFIRM_RESET=1`` or ``--yes`` answers). ``None``
                cannot ask: such a run raises ``ConfirmationDeclinedError``
                before it writes, as a no does.
            confirm_deletions: Asks before an incremental run deletes more
                than ``indexing.max_deletion_share`` of its source scope's
                documents (the CLI asks on the terminal;
                ``--confirm-deletions``, ``UKR_CONFIRM_DELETIONS=1`` or
                ``--yes`` answers). ``None`` cannot ask: such a run raises
                ``ConfirmationDeclinedError`` before it writes.

        Raises:
            ValueError: ``providers`` and one of the shorthand components are
                both given.
            SchemaSourceError: The extraction schema source cannot be loaded.
            SeedGraphError: A seed graph cannot be imported.
        """
        # Loaded once: every stage reads the schema from the inline keys. A
        # source's AWS calls use the run's session.
        config = resolve_extraction_schema(config, providers=providers)
        # Against that schema, before any model is built or called: a seed
        # that cannot be imported fails here.
        self.seed_conversions = tuple(load_seed_graphs(config, providers))
        self._confirm = confirm
        self._confirm_deletions = confirm_deletions
        self.config = config
        # The write lease of the latest run (see _namespace_lease).
        self._lease: NamespaceLease | None = None
        # Before any stage builds a model, so the factories attach the tracing
        # callback (no-op unless observability.enabled).
        configure_observability(config.observability)
        overrides: dict[str, Any] = {}
        if doc_status is not None:
            overrides["doc_status"] = doc_status
        # The caller's instances, shared by every stage: each stage closes
        # what it builds, so they get a borrowed view whose close does
        # nothing (the caller closes them).
        if vector_indexer is not None:
            borrowed_vector = cast(VectorIndexer, _Borrowed(vector_indexer))
            overrides["vector_indexer_factory"] = lambda: borrowed_vector
        if graph_indexer is not None:
            borrowed_graph = cast(GraphIndexer, _Borrowed(graph_indexer))
            overrides["graph_indexer_factory"] = lambda: borrowed_graph
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
        self.providers = (
            providers if providers is not None else build_providers(config, **overrides)
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
        )
        self.state_manager = PipelineStateManager(self.cache_manager)
        # The resume manager must reconstruct the same input-fingerprinted cache
        # keys the save path writes, so it needs the same config.
        self.resume_manager = PipelineResumeManager(self.state_manager, self.config)

        self.s3_cache_manager = build_checkpoint_sync(
            self.config, self.pipeline_config, self.providers
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
                "seed_conversions": self.seed_conversions,
                "confirm": self._confirm,
                "confirm_deletions": self._confirm_deletions,
            },
            # --force-rebuild re-extracts every unit (and refreshes the
            # extraction store) instead of reusing the stored records, asks
            # the model for every pre-chunk's boundaries, description
            # summary and community report again and embeds every text
            # again.
            PipelineStageType.TEXT_CHUNKING: {
                "reuse_stored": not self.pipeline_config.force_rebuild
            },
            PipelineStageType.GRAPH_EXTRACTION: {
                "reuse_stored": not self.pipeline_config.force_rebuild
            },
            PipelineStageType.GRAPH_RESOLUTION: {
                "reuse_stored": not self.pipeline_config.force_rebuild
            },
            PipelineStageType.CLAIM_EXTRACTION: {
                "reuse_stored": not self.pipeline_config.force_rebuild
            },
            PipelineStageType.COMMUNITY_DETECTION: {
                "cache_directory": cache_directory,
                "doc_status": doc_status,
                "reuse_stored": not self.pipeline_config.force_rebuild,
            },
            PipelineStageType.INDEXING: {
                "doc_status": doc_status,
                "reuse_stored": not self.pipeline_config.force_rebuild,
            },
            PipelineStageType.SUMMARY_TREE: {
                "doc_status": doc_status,
                "reuse_stored": not self.pipeline_config.force_rebuild,
            },
        }

        stages: list[PipelineStage] = []
        name_to_type_map: dict[str, PipelineStageType] = {}
        disabled_stages = []
        for stage_type, stage_class in self.STAGE_CLASSES.items():
            if not self._stage_enabled(stage_type):
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

    def _stage_enabled(self, stage_type: PipelineStageType) -> bool:
        """Whether the run includes ``stage_type``: its config section
        (``CONFIG_ENABLED``) allows it, and an explicit ``stages_enabled``
        list (one that leaves a stage out) names it, or without one the
        selected methodologies need it (``methodologies.selected_stages``:
        every stage for the default ones). An explicit list wins either way
        (docs/design.md#46-methodology-profiles)."""
        if not self.CONFIG_ENABLED.get(stage_type, lambda _: True)(self.config):
            return False
        listed = self.pipeline_config.stages_enabled
        needed = selected_stages(self.config)
        if all(listed.get(stage, True) for stage in PipelineStageType):
            return stage_type in needed
        if not listed.get(stage_type, True):
            # Only a run that writes the index can leave a methodology out
            # of it (a phase before indexing writes none).
            methodologies = stage_methodologies(self.config, stage_type)
            if methodologies and listed.get(PipelineStageType.INDEXING, True):
                names = ", ".join(m.value for m in methodologies)
                logger.warning(
                    "Stage '%s' is not in the run's enabled stages, but %s "
                    "needs it: the index this run writes holds no %s unless "
                    "an earlier run of this pipeline id completed the stage",
                    stage_type.value,
                    names,
                    names,
                )
            return False
        if stage_type not in needed:
            logger.warning(
                "Stage '%s' is in the run's enabled stages, but no selected "
                "methodology (indexing.methodologies: %s) needs it: it runs, "
                "and the index records nothing it builds",
                stage_type.value,
                ", ".join(m.value for m in self.config.effective_methodologies),
            )
        return True

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
            ConfigurationError: The configuration cannot be run, for
                example the vector store refused to create an index with
                the run's settings (``StoreRequestRejectedError``) or an
                incremental run's live indices were built for other
                settings, both checked before any model call.
            StoreConnectionError: A store could not be reached outside a
                stage (the preflight before any model call, say).
            NamespaceLeaseError: Another run writes the namespace.
            ConfirmationDeclinedError: A question the run asked before it
                writes (``confirm``: a reset that drops another source
                scope's content; ``confirm_deletions``: a mass deletion) was
                answered no, or nothing could answer it.
        """
        # The root span of an ingestion trace: stages, batches and model calls
        # nest under it. A failure leaving the block is recorded on it.
        with observability.span(
            "kg_rag.ingestion.run", resume_from_stage=resume_from_stage
        ) as span:
            self.pipeline_id = None
            try:
                context = self._run(source_directory, pipeline_id, resume_from_stage)
            finally:
                # Also on a stop or a crash: the id the run resumes with
                # (_run sets it; read past mypy's narrowing to None above).
                if resolved := getattr(self, "pipeline_id", None):
                    span.set_attribute("kg_rag.pipeline_id", resolved)
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
        stop_fence = install_write_fence(_StopFence(self))
        # The run's alias and mapping lookups, read once (only its own
        # writes change them while it holds the namespace's lease).
        lookups = install_lookup_cache()
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
            self.pipeline_id = resolved_pipeline_id
            # Every log line of the run, the S3 sync and the failure report
            # included, carries the id (also in worker threads, see
            # ContextThreadPoolExecutor).
            log_context = bind_contextvars(pipeline_id=resolved_pipeline_id)
            resolved_resume_stage = self.normalize_stage_name(
                resume_from_stage or self.pipeline_config.resume_from_stage
            )

            logger.info("Resolved pipeline ID: '%s'", resolved_pipeline_id)

            # The lease covers the checkpoint too: it is downloaded only once
            # no other run writes the namespace, and uploaded before the
            # lease is released.
            try:
                with self._namespace_lease() as lease:
                    try:
                        context = self._run_leased(
                            source_path,
                            resolved_pipeline_id,
                            resolved_resume_stage,
                            lease,
                        )
                    finally:
                        # A stop (request_stop) from here on waits until the
                        # lease is given up: raised in its exit, it left the
                        # lease held for indexing.lease_seconds.
                        self._checkpointing = True
            finally:
                self._checkpointing = False
            # Stopped as the finished run gave the lease up: its last
            # checkpoint is uploaded.
            self._checkpoint_id = None
            self._raise_if_stopping()

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
        except ConfigurationError:
            # The store refused the run's index settings, or its live
            # indices were built for other settings, before any model call
            # (_check_index_creation): a usage error, not a crash.
            raise
        except StoreConnectionError:
            # A store could not be reached: raised as the facade exports it,
            # for a caller to tell an outage from a failed run.
            raise
        except ConfirmationDeclinedError:
            # A question answered no (a reset not confirmed): nothing was
            # written and nothing failed, so no traceback.
            raise
        except PipelineExecutionError:
            # Already says what failed (a refused resume, say): wrapped again
            # it read "Failed to run pipeline: Pipeline resume failed: ...".
            logger.exception("Pipeline execution failed")
            raise
        except Exception as e:
            self._raise_if_stopped_by(e)
            logger.exception("Pipeline execution failed: %s", e)
            raise PipelineExecutionError(f"Failed to run pipeline: {e}") from e
        finally:
            if self._reset_for_run:
                # Set for the run that needed it, not for a later one.
                self.config.indexing.reset = False
                self._reset_for_run = False
            if self.llm_response_cache is not None:
                self.llm_response_cache.close()
            self._checkpoint_id = None
            reset_contextvars(**log_context)
            remove_lookup_cache(lookups)
            remove_write_fence(stop_fence)

    def _run_leased(
        self,
        source_path: Path,
        pipeline_id: str,
        resume_stage: str | None,
        lease: NamespaceLease | None,
    ) -> PipelineContext:
        """Download the checkpoint, prepare the run and execute it (under the
        namespace's ``lease``), closing the run's processing attempts when it
        finishes, fails with an error or is stopped (KeyboardInterrupt,
        ``PipelineInterrupted``), and uploading the checkpoint of a stopped
        run."""
        context: PipelineContext | None = None
        try:
            if self.pipeline_config.s3_sync_enabled:
                self._sync_cache_with_s3(pipeline_id, "download")
                self._checkpoint_id = pipeline_id

            # Before any cache key is computed: a rebuild is keyed so.
            self._rebuild_for_added_artifacts(source_path)
            # Part of every stage cache key, so it must be set before the
            # resume decision reads the cache.
            self._set_corpus_fingerprint(source_path)
            self._open_llm_response_cache(pipeline_id)

            # force_rebuild must ignore ALL existing cache: skip the resume path
            # entirely so the run re-executes from the beginning even when a
            # prior metadata file exists (FORCE_REFRESH alone only bypasses
            # per-stage cache reads, not the resume decision).
            can_resume = (
                pipeline_id
                and not self.pipeline_config.force_rebuild
                and self.state_manager.pipeline_exists(pipeline_id)
            )
            if can_resume:
                logger.info(
                    "Resuming existing pipeline '%s' from stage: '%s'",
                    pipeline_id,
                    resume_stage or "auto-detect",
                )
                context, start_stage_name = self._prepare_resume(
                    pipeline_id, resume_stage
                )
                self._open_resumed_attempts(context, start_stage_name)
            else:
                if self.pipeline_config.force_rebuild:
                    logger.info(
                        "Force rebuild: ignoring any existing cache for '%s'",
                        pipeline_id,
                    )
                logger.info("Starting new pipeline run: '%s'", pipeline_id)
                context, start_stage_name = self._prepare_new_run(
                    source_path, pipeline_id, resume_stage
                )
            self._execute_pipeline(context, start_stage_name)
        except KeyboardInterrupt:
            # Ctrl-C did not go through request_stop: the stop fence must
            # still end the writes of the worker threads it left running.
            if self._stop_signal is None:
                self._stop_signal = signal.SIGINT
            # A second stop request waits for the attempts and the checkpoint.
            self._checkpointing = True
            try:
                self._close_attempts(context, lease)
                self._checkpoint_on_stop()
            finally:
                self._checkpointing = False
                self._checkpoint_id = None
            raise
        # Any other BaseException (SystemExit, or a test's simulated death)
        # leaves the attempts open, like a process that died.
        except Exception as e:
            if self._stop_signal is None:
                self._close_attempts(context, lease)
                raise
            # A stop that surfaced wrapped in the error (see
            # _raise_if_stopped_by) ends the run as a stop here, its
            # checkpoint uploaded while the lease is still held.
            self._checkpointing = True
            try:
                self._close_attempts(context, lease)
                self._checkpoint_on_stop()
            finally:
                self._checkpointing = False
                self._checkpoint_id = None
            raise PipelineInterrupted(self._stop_signal) from e
        self._close_attempts(context, lease)
        return context

    def _rebuild_for_added_artifacts(self, source_path: Path) -> None:
        """Rebuild the namespace (``indexing.reset`` for this run) when the
        run builds a corpus-wide artifact its build record does not name: a
        methodology added since (``graphrag``'s communities and reports after
        a ``lightrag``-only build). A delta would build it for the delta's
        documents only; the rebuild reuses the stored chunk boundaries,
        extraction records, summaries and vectors, so only the added
        artifacts' model calls are new. Only with the registry on (without
        it every run is a full build), and never on a record from before
        methodology profiles (``methodologies.missing_artifacts``).

        Raises:
            ConfigurationError: The namespace also holds other source
                scopes' documents, which a rebuild from this corpus would
                delete.
            DocStatusRegistryError: The namespace cannot be read.
        """
        if self._doc_status is None or self.config.indexing.reset:
            return
        indexer = self.providers.build_vector_indexer()
        try:
            record = indexer.read_build_record(
                self.config.processing.document_parsing.index_value
            )
        finally:
            indexer.close()
        missing = missing_artifacts(record, self.config)
        if not missing:
            return
        added = ", ".join(sorted(a.value for a in missing))
        if others := other_source_scopes(self._doc_status, self.config, source_path):
            raise ConfigurationError(
                f"indexing.methodologies reads {added}, which the index "
                "namespace does not hold: building it needs a rebuild of the "
                "namespace, and the namespace also holds documents of source "
                f"scope(s) {', '.join(others)}, which a rebuild from this "
                "corpus would delete. Keep the methodologies the namespace "
                "was built with, give each corpus its own namespace "
                "(processing.document_parsing.index_value), or rebuild it "
                "from this corpus with indexing.reset: true, deleting them."
            )
        logger.warning(
            "The index namespace holds no %s, which indexing.methodologies "
            "(%s) reads: this run rebuilds it from the whole corpus "
            "(indexing.reset), reusing the stored chunk boundaries, "
            "extractions, summaries and vectors, so only the model calls "
            "that build what it adds are new",
            added,
            ", ".join(m.value for m in self.config.effective_methodologies),
        )
        self.config.indexing.reset = True
        self._reset_for_run = True

    def _incremental_delta(self, context: PipelineContext | None) -> bool:
        """Whether ``context`` holds an incremental delta the registry tracks
        (not a reset's, whose rebuild opens no attempt)."""
        return (
            context is not None
            and context.incremental_delta is not None
            and self._doc_status is not None
            and not self.config.indexing.reset
        )

    def _open_resumed_attempts(
        self, context: PipelineContext, start_stage_name: str | None
    ) -> None:
        """Open the processing attempts of a run resumed past document_loading
        (which opens them on a fresh run, see ``incremental.open_attempts``)."""
        if not self._incremental_delta(context) or not self._resumes_past_loading(
            start_stage_name
        ):
            return
        assert context.incremental_delta is not None and self._doc_status is not None
        open_attempts(
            self._doc_status,
            context.incremental_delta,
            context.incremental_fingerprints,
            scope=context.incremental_scope,
            written_by=context.pipeline_id,
        )

    def _close_attempts(
        self, context: PipelineContext | None, lease: NamespaceLease | None
    ) -> None:
        """Close the run's processing attempts (best effort: an attempt left
        open counts as a death of the run, see ``domain.ingestion.attempts``).

        A run that lost its ``lease`` takes it back to close them unless
        another run holds it now: a lost lease is no death of the run."""
        if not self._incremental_delta(context):
            return
        assert context is not None and context.incremental_delta is not None
        assert self._doc_status is not None
        close = functools.partial(
            close_attempts,
            self._doc_status,
            context.incremental_delta,
            context.incremental_fingerprints,
            context.pipeline_id,
        )
        try:
            try:
                close()
            except NamespaceLeaseError:
                if lease is None or not lease.retake():
                    raise
                close()
        except Exception as e:  # noqa: BLE001 - counted as a death at worst
            logger.warning(
                "Closing the run's processing attempts failed (%s); the next "
                "run counts this one as a run that died",
                e,
            )

    def _resumes_past_loading(self, start_stage_name: str | None) -> bool:
        """Whether a resume at ``start_stage_name`` reuses the restored delta
        and still runs a stage up to indexing."""
        if start_stage_name not in self._ORDER:
            return False
        position = self._ORDER.index(start_stage_name)
        return self._ORDER.index(
            PipelineStageType.DOCUMENT_LOADING.value
        ) < position <= self._ORDER.index(
            PipelineStageType.INDEXING.value
        ) and self._start_index(
            start_stage_name
        ) < len(
            self.stages
        )

    def _namespace_lease(self) -> AbstractContextManager[NamespaceLease | None]:
        """The write lease of the run's index namespace, held for the whole
        run when the doc-status registry is on (see ``namespace_lease``)."""
        if self._doc_status is None:
            return nullcontext()
        namespace = config_registry_namespace(
            self.config.indexing, self.config.processing.document_parsing.index_value
        )
        seconds = self.config.indexing.lease_seconds
        longest = longest_write_request_seconds()
        if seconds < 3 * longest:
            logger.warning(
                "indexing.lease_seconds (%gs) is under three times the longest "
                "store write request (%gs): a request in flight when the lease "
                "runs out can still write after another run took it",
                seconds,
                longest,
            )
        self._lease = NamespaceLease(
            self._doc_status,
            namespace,
            seconds,
            interrupt=self._raise_if_stopping,
            in_process_write_seconds=self.config.search.store_timeout_seconds,
            in_process_stores=self._stores_in_process(),
        )
        return self._lease

    def _stores_in_process(self) -> bool:
        """Whether every store the run writes (the registry, the graph and
        the vector store) is embedded in this process, told by the adapters
        the run builds (their ``in_process``), not by the configured kinds:
        an injected remote store is not embedded whatever the config says."""
        if getattr(self._doc_status, "in_process", False) is not True:
            return False
        for build in (
            self.providers.build_graph_indexer,
            self.providers.build_vector_indexer,
        ):
            indexer = build()
            if getattr(indexer, "in_process", False) is not True:
                # An embedded one opens nothing until it writes; a remote
                # one may hold a client.
                try:
                    indexer.close()
                except Exception as e:  # noqa: BLE001 - only a probe
                    logger.debug("Error closing %s: %s", type(indexer).__name__, e)
                return False
        return True

    @property
    def next_run_writes_from(self) -> float:
        """After a run: when the next run on its namespace may write first
        (epoch seconds; 0 when at once). A stopped run with writes possibly
        in flight, its own or those of the run whose expired lease it took
        over, makes the next run wait (see ``namespace_lease``)."""
        return self._lease.next_writes_from if self._lease is not None else 0.0

    def _raise_if_stopping(self) -> None:
        """Raise ``PipelineInterrupted`` once a stop was requested (a takeover
        wait of the lease calls it, in any thread)."""
        if self._stop_signal is not None:
            raise PipelineInterrupted(self._stop_signal)

    def _raise_if_stopped_by(self, error: Exception) -> None:
        """End a run that failed with ``error`` once a stop was requested as
        a stop: the stop surfaced wrapped in it (raised inside a
        serialization pydantic wraps it in, say)."""
        if self._stop_signal is not None:
            self._checkpoint_on_stop()
            raise PipelineInterrupted(self._stop_signal) from error

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
        """Upload one cache file now; raises ``CacheSyncError`` on failure.
        The response cache calls it for each segment it writes, so a task
        killed mid-stage keeps those responses; it logs a failure and retries
        the segment at its next write (the checkpoint upload also sends it)."""
        if self.s3_cache_manager is not None:
            self.s3_cache_manager.upload_file(pipeline_id, cache_dir, path)

    def request_stop(self, signum: int) -> None:
        """End the run: raise ``PipelineInterrupted``, uploading the
        checkpoint as the run unwinds.

        Meant for a signal handler in the main thread; ``run-ingestion``
        installs one for SIGTERM, which ECS sends on StopTask and when a Step
        Functions execution times out, before it kills the task, and for
        Ctrl-C (SIGINT) unless the parent ignores it. Every stage
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
        """Defer a requested stop until the enclosed checkpoint write is done
        (the outermost one, when nested).

        Every metadata save goes through one: a stop raised inside its JSON
        serialization is wrapped by pydantic, as a warning (the run went on)
        or an error (the run failed instead of stopping).
        """
        outer = self._checkpointing
        self._checkpointing = True
        try:
            yield
        finally:
            self._checkpointing = outer
        if not outer and self._stop_signal is not None:
            raise PipelineInterrupted(self._stop_signal)

    def _checkpoint_on_stop(self) -> None:
        """Upload the checkpoint of a run that is stopping (``request_stop``
        or Ctrl-C), once the stage it interrupted has unwound."""
        if self._checkpoint_id is None:
            return
        try:
            # Held, not waited for: a stop during the takeover wait must not
            # lose the finished stages (a checkpoint is no store write).
            check_fences_held()
        except NamespaceLeaseError as e:
            # Another run may write this pipeline's checkpoint now.
            logger.warning(
                "Run stopped without uploading the checkpoint of '%s': %s",
                self._checkpoint_id,
                e,
            )
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
        if normalized not in cls._ORDER:
            raise ConfigurationError(
                f"Unknown pipeline stage '{stage_name}'. Valid stages: "
                f"{', '.join(cls._ORDER)}"
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
                    seed_conversions=stage.seed_conversions,
                    confirm=stage.confirm,
                    confirm_deletions=stage.confirm_deletions,
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
        """Fingerprint the files this run can ingest and the seed graphs it
        imports (see cache_keys)."""
        extensions = {*ParserFactory.get_supported_extensions(), ".json"}
        owned = [Path(self.pipeline_config.local_directory)]
        if self.target_directory is not None:
            owned.append(self.target_directory)
        fingerprint = seed_corpus_fingerprint(
            corpus_manifest_fingerprint(source_directory, extensions, owned),
            [conversion.digest for conversion in self.seed_conversions],
        )
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
                "incremental indexing (providers.doc_status.enabled: true): the "
                "retired records live "
                "in the doc-status registry"
            )
        parsing = self.config.processing.document_parsing
        retired_scope_groups(
            retired,
            parsing.source_scope or source_directory.as_posix(),
            config_registry_namespace(self.config.indexing, parsing.index_value),
            retired_scope_readings,
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

        if direction == "upload":
            # A run that lost its lease must not replace the checkpoint the
            # run holding it now may write. Held, not waited for (no store
            # write): the takeover wait would hold every stage's checkpoint.
            check_fences_held()
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
        """Restore a pipeline id's context and the stage its resume starts at
        (``resume.prepare_resume``)."""
        return resume.prepare_resume(self, pipeline_id, resume_from_stage)

    def _rewind_to_unrun_stage(
        self, start_stage_name: str, completed_stages: list[str]
    ) -> tuple[str, list[str]]:
        """``resume.rewind_to_unrun_stage``."""
        return resume.rewind_to_unrun_stage(self, start_stage_name, completed_stages)

    def _rewind_to_failed_stage(
        self,
        context: PipelineContext,
        start_stage_name: str | None,
        completed_stages: list[str],
        *,
        explicit: bool = False,
        rerun: bool = False,
    ) -> tuple[str, list[str]] | None:
        """``resume.rewind_to_failed_stage``."""
        return resume.rewind_to_failed_stage(
            self,
            context,
            start_stage_name,
            completed_stages,
            explicit=explicit,
            rerun=rerun,
        )

    def _rewind_on_missing_records(
        self,
        context: PipelineContext,
        start_stage_name: str | None,
        completed_stages: list[str],
    ) -> tuple[str, list[str]] | None:
        """``resume.rewind_on_missing_records``."""
        return resume.rewind_on_missing_records(
            self, context, start_stage_name, completed_stages
        )

    def _drop_unrun_failures(
        self, context: PipelineContext, start_stage_name: str | None
    ) -> None:
        """``resume.drop_unrun_failures``."""
        resume.drop_unrun_failures(self, context, start_stage_name)

    def _delta_is_stale(
        self, context: PipelineContext, start_stage_name: str | None
    ) -> bool:
        """``resume.delta_is_stale``."""
        return resume.delta_is_stale(self, context, start_stage_name)

    def _prepare_context_for_resume(
        self, context: PipelineContext, start_stage_name: str
    ) -> None:
        """``resume.prepare_context_for_resume``."""
        resume.prepare_context_for_resume(self, context, start_stage_name)

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

        index_creation_checked = False
        for stage in stages_to_execute:
            if stage.name in completed_stage_names:
                logger.info("Skipping previously completed stage: '%s'", stage.name)
            else:
                logger.info("Executing stage: '%s'", stage.name)
                # A run whose lease was lost meanwhile (in a long stage that
                # does not write) stops here, not at its next write. Not the
                # takeover wait: a stage that writes nothing need not wait.
                check_fences_held()
                if not index_creation_checked and (
                    stage.name not in self._MODEL_FREE_STAGES
                ):
                    self._check_index_creation(context)
                    index_creation_checked = True
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
                    self._record_item_failures(stage, context)
                    logger.warning(
                        "Pipeline execution stopped due to stage failure: '%s'",
                        stage.name,
                    )
                    break

    # The stages before the first one that may call a model (chunking may
    # embed, translation translates).
    _MODEL_FREE_STAGES = frozenset(
        {
            PipelineStageType.DOCUMENT_PARSING.value,
            PipelineStageType.DOCUMENT_LOADING.value,
        }
    )

    def _check_index_creation(self, context: PipelineContext) -> None:
        """Have the vector store refuse the run now, before any model call,
        rather than in the indexing stage after extraction: an incremental
        run's live indices built for other settings
        (``VectorIndexer.check_index_settings``), and the indices the run
        will create (``VectorIndexer.check_index_creation``: every one on a
        full build or reset, an incremental run's missing ones). An
        incremental run with no document to (re)extract (no change, or only
        deletions) creates no index: only its live indices are checked, so it
        is refused like the next run that extracts would be. Any other
        failure of the check is logged: the stages then meet the store's
        error themselves.

        Raises:
            ConfigurationError: The live indices do not match, or the store
                refused the settings or the caller (``StoreRequestRejectedError``,
                ``StoreAccessDeniedError``: HTTP 400, 401/403).
            StoreConnectionError: The store is unreachable (after the
                client's retries): the run would pay for the extraction and
                then fail to write it.
            NamespaceLeaseError: The run lost its write lease.
        """
        try:
            incremental = (
                context.incremental_delta is not None and not self.config.indexing.reset
            )
            suffix = self.config.processing.document_parsing.index_value
            indexer = self.providers.build_vector_indexer()
            try:
                if incremental:
                    indexer.check_index_settings(suffix)
                if not incremental or context.documents:
                    indexer.check_index_creation(suffix, rebuild=not incremental)
            finally:
                indexer.close()
        except (ConfigurationError, NamespaceLeaseError, StoreConnectionError):
            raise
        except Exception as e:  # noqa: BLE001 - the stages report the store
            logger.warning("Checking the vector store's index creation failed: %s", e)

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
        if start_stage_name not in self._ORDER:
            logger.warning(
                "Start stage '%s' not found, starting from beginning",
                start_stage_name,
            )
            return 0
        start_position = self._ORDER.index(start_stage_name)
        for index, name in enumerate(stage_names):
            if self._ORDER.index(name) >= start_position:
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
        context.item_failed_text_unit_ids.pop(stage.name, None)
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
            with self._checkpoint_write():
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
        with self._checkpoint_write():
            self.state_manager.save_pipeline_metadata(context)
        if units:
            logger.warning(
                "Stage '%s' failed: recording all %s text units as failed so "
                "their documents are not committed as processed",
                stage.name,
                len(units),
            )

    def _record_item_failures(
        self, stage: PipelineStage, context: PipelineContext
    ) -> None:
        """Record FAILED the documents ``stage``, which stops the run, failed
        on for errors of their own (``incremental.record_item_failures``).

        The indexing stage records a failed document only when the run gets
        there; a stage that failed on all of its units stops the run first,
        so without this a document that always fails (a request the model
        rejects, a response that never parses) never reaches
        ``indexing.max_document_failures`` and stops every later run, with
        the deletions of its delta. Errors of the model (throttling, 5xx,
        timeouts, connections) count for none. Best effort: the run fails
        either way.
        """
        unit_ids = context.item_failed_text_unit_ids.get(stage.name, [])
        retried = context.failed_text_unit_ids.get(stage.name, [])
        if not (unit_ids or retried) or not self._incremental_delta(context):
            return
        assert self._doc_status is not None
        try:
            record_item_failures(
                self._doc_status,
                context,
                unit_ids,
                self.config.indexing.max_document_failures,
                retried_unit_ids=retried,
            )
        except Exception as e:  # noqa: BLE001 - the run fails either way
            logger.warning(
                "Recording the documents stage '%s' failed on failed (%s); the "
                "next run retries them without counting this failure",
                stage.name,
                e,
            )

    def _run_stage(
        self, stage: PipelineStage, context: PipelineContext
    ) -> PipelineStageResult:
        running_result = PipelineStageResult(
            stage_name=stage.name,
            status=PipelineStageStatus.RUNNING,
            start_time=datetime.now(),
        )
        with self._checkpoint_write():
            self.state_manager.update_stage_result(context, running_result)

        stage_usage: dict[str, Any] | None = None
        try:
            # The stage's model calls and tokens (shared.usage), kept with its
            # result: emitted as llm_*_<stage> and run-ingestion's token_usage.
            with usage_scope() as meter:
                try:
                    result = stage.execute(context)
                finally:
                    stage_usage = meter.snapshot()
            result.metrics[LLM_USAGE_METRIC] = stage_usage
            if stage.stage_type is PipelineStageType.DOCUMENT_LOADING:
                # The delta is saved with the metadata, the documents it
                # filtered under this key: a resume restores them together
                # only (see PipelineResumeManager).
                context.incremental_loading_key = stage_cache_key(
                    self.config,
                    PipelineStageType.DOCUMENT_LOADING,
                    "documents",
                    self.corpus_fingerprint,
                )
            with self._checkpoint_write():
                self.state_manager.update_stage_result(context, result)
                if result.status == PipelineStageStatus.COMPLETED:
                    self._save_stage_outputs_to_cache(context, stage.name)
            # A failed stage logged its error (PipelineStage._execute_logged).
            return result

        except ConfirmationDeclinedError as e:
            # A question the stage asked before writing was answered no: the
            # run stops here, recorded like a failed stage for a resume.
            with self._checkpoint_write():
                self.state_manager.update_stage_result(
                    context,
                    PipelineStageResult(
                        stage_name=stage.name,
                        status=PipelineStageStatus.FAILED,
                        start_time=running_result.start_time,
                        end_time=datetime.now(),
                        error_message=str(e),
                    ),
                )
            raise
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
                metrics=(
                    {LLM_USAGE_METRIC: stage_usage} if stage_usage is not None else {}
                ),
            )
            with self._checkpoint_write():
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
        except Exception as e:  # noqa: BLE001 - the stage reruns on resume
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
                # The stage logged its error already, at ERROR.
                logger.info(
                    "Stopping pipeline due to failure in stage '%s'",
                    result.stage_name,
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
        self._warn_failed_units(context)

        with self._checkpoint_write():
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
                pipeline_metrics.log_pipeline_summary(context)
                raise

        pipeline_metrics.log_pipeline_summary(context)

    def _warn_failed_units(self, context: PipelineContext) -> None:
        """Warn, at the end of a run, of the text units its stages failed on:
        how many, by stage and cause, and what retries them.

        A run that completes with failed units is a partial graph. Its
        ``failed_documents`` counts the documents the doc-status registry
        recorded FAILED, so it reads 0 in a full build (no registry) however
        many units failed; the per-unit errors were each logged once, in the
        middle of the run.
        """
        failed = context.failed_unit_ids()
        if not failed:
            return
        stages = []
        for name, unit_ids in context.failed_text_unit_ids.items():
            if not unit_ids:
                continue
            own = len(
                set(unit_ids) & set(context.item_failed_text_unit_ids.get(name, []))
            )
            stages.append(
                f"{name} {len(unit_ids)} ({own} for an error of their own, such "
                f"as output cut off or unparseable; {len(unit_ids) - own} for "
                "another, such as throttling or an outage)"
            )
        if self._doc_status is not None:
            retry = (
                f"{len(context.recorded_failed_doc_ids)} documents recorded "
                "FAILED (failed_documents), processed again by the next run"
            )
        else:
            retry = (
                "no document is recorded FAILED without the doc-status registry "
                "(providers.doc_status.enabled), so failed_documents is 0: a "
                "re-run extracts those units again, reusing the stored "
                "extraction records of the others"
            )
        logger.warning(
            "%d text units failed and are indexed without that stage's output "
            "(units_failed): %s; %s",
            len(failed),
            "; ".join(stages),
            retry,
        )

    def _create_pipeline_metrics(self, context: PipelineContext) -> PipelineMetrics:
        """``pipeline_metrics.create_pipeline_metrics``."""
        return pipeline_metrics.create_pipeline_metrics(self, context)

    def _emit_metrics(
        self,
        metrics: PipelineMetrics,
        pipeline_id: str,
        stage_name: str | None = None,
        extra: dict[str, float] | None = None,
    ) -> None:
        """``pipeline_metrics.emit_metrics``."""
        pipeline_metrics.emit_metrics(self, metrics, pipeline_id, stage_name, extra)

    def _emit_stage_metrics(self, context: PipelineContext, stage_name: str) -> None:
        """``pipeline_metrics.emit_stage_metrics``."""
        pipeline_metrics.emit_stage_metrics(self, context, stage_name)

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
        except Exception as e:  # noqa: BLE001 - verification only reports
            logger.error(
                "Error during pipeline metadata verification for %s: %s",
                pipeline_id,
                e,
                exc_info=True,
            )
            return False
