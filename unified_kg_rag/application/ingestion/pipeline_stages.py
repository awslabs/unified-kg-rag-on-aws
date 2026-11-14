# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from pydantic import ValidationError
from structlog.contextvars import bind_contextvars, reset_contextvars

from unified_kg_rag import __version__
from unified_kg_rag.adapters.ingestion.chunker import ChunkerFactory
from unified_kg_rag.adapters.ingestion.claim_extractor import (
    ClaimExtractionStats,
    ClaimExtractor,
)
from unified_kg_rag.adapters.ingestion.community_detector import (
    CommunityDetector,
)
from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
    needs_summary,
)
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.adapters.ingestion.graph_extractor import (
    ExtractionStats,
    GraphExtractor,
)
from unified_kg_rag.adapters.ingestion.loader import DirectoryLoader
from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.adapters.ingestion.translator import TextUnitTranslator
from unified_kg_rag.adapters.llm.reachability import (
    is_unreachable_error,
    model_unreachable_error,
)
from unified_kg_rag.adapters.llm.retry import (
    FailureLedger,
    describe_cause,
    failure_ledger,
    item_error_predicate,
)
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.extraction_store import (
    ExtractionStore,
    with_record_keys,
)
from unified_kg_rag.application.ingestion.incremental import (
    artifact_owners,
    clear_registry_namespaces,
    context_lineages,
    corpus_is_seeds_only,
    dedup_preferences,
    filter_to_delta,
    index_delta,
    record_rebuilt_annotations,
    recorded_source_scopes,
    reset_namespaces,
    run_failures,
    run_incremental_indexer,
    run_registry_scope,
    unit_source_files,
    warn_about_unapplied_retired_scopes,
)
from unified_kg_rag.application.ingestion.source_parsing import (
    SourceFileParser,
    reusable_records,
)
from unified_kg_rag.application.storage.indexing_manager import (
    IndexingManager,
    stripped_descriptions,
)
from unified_kg_rag.domain.ingestion.cache_keys import (
    extraction_settings,
    is_excluded_source_path,
    leaves_source_root,
    real_source_path,
    unit_record_fingerprint,
)
from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver
from unified_kg_rag.domain.ingestion.community_drift import (
    CommunityDrift,
    commit_identity,
    reset_reasons,
    with_level0_community_ids,
)
from unified_kg_rag.domain.ingestion.contributions import (
    claims_from_payload,
    claims_payload,
    merge_contributions,
    record_from_payload,
    record_payload,
    unit_entities,
)
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
)
from unified_kg_rag.domain.ingestion.descriptions import merge_descriptions
from unified_kg_rag.domain.ingestion.document_identity import (
    RELATIVE_PATH_KEY,
    config_registry_namespace,
    is_unparsed,
    store_namespace_tag,
)
from unified_kg_rag.domain.ingestion.graph_analyzer import GraphAnalyzer
from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.ingestion.methodologies import (
    artifacts_to_build,
    built_artifacts,
    stored_artifacts,
    stored_artifacts_shown,
)
from unified_kg_rag.domain.ingestion.names import (
    compact_name_key,
    normalize_entity_type,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    DOCUMENT_SOURCE_KIND,
    SEED_RECORD_FINGERPRINT,
    SeedConversion,
    is_seed_document,
    is_seed_unit,
    parse_seed_unit,
    seed_only,
    seed_text_units,
)
from unified_kg_rag.domain.languages.corpus import (
    corpus_sample,
    sample_split,
    source_language_mismatch,
)
from unified_kg_rag.domain.models import (
    ChunkingStrategy,
    Claim,
    Community,
    CommunityChange,
    CommunityCounters,
    CommunityMetrics,
    CommunityReport,
    Config,
    Constants,
    DocStatus,
    Document,
    DocumentDelta,
    Entity,
    IndexArtifact,
    IndexBuildRecord,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.methodology import recorded_artifacts
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.ports.indexer import (
    BaseIndexer,
    GraphIndexer,
    IndexingStats,
    keeps_build_records,
)
from unified_kg_rag.shared import (
    ConfigurationError,
    ConfirmationDeclinedError,
    NamespaceLeaseError,
    PipelineStageError,
    StoreConnectionError,
    get_logger,
    observability,
)
from unified_kg_rag.shared.exceptions import classify, operator_error_text
from unified_kg_rag.shared.genai import ERROR_TYPE
from unified_kg_rag.shared.logging import log_error
from unified_kg_rag.shared.write_fence import check_write_fence

if TYPE_CHECKING:
    from unified_kg_rag.ports import DocStatusPort

logger = get_logger(__name__)


def _resolved_claims(context: PipelineContext) -> list[Claim]:
    """The claims the graph, the claim index and the lineage may hold.

    Only claim resolution ties a claim to an entity (``subject_id``); an
    extracted claim carries empty ids until then. So this never falls back to
    ``context.claims``: when no claim resolves, or claim resolution did not
    run (disabled through ``stages_enabled``), no claim is indexed rather than
    claims attached to no entity.
    """
    return [claim for claim in context.resolved_claims if claim.subject_id]


def is_incremental_build(context: PipelineContext, config: Config) -> bool:
    """Whether the run added its delta to what the namespace held (its
    build record's ``mode`` ``incremental``). A delta of new documents only
    into a namespace the registry held nothing of built the suffix from
    nothing: a full build, as is a reset or a run without the registry. One
    beside other source scopes' documents (``stored_scopes``) added to
    them."""
    delta = context.incremental_delta
    return (
        delta is not None
        and not config.indexing.reset
        and bool(
            delta.changed or delta.unchanged or delta.deleted or delta.stored_scopes
        )
    )


def _completed_stages(context: PipelineContext) -> set[PipelineStageType]:
    """The stages completed in ``context`` (this run's and, on a resume,
    its earlier phases')."""
    known = {stage.value: stage for stage in PipelineStageType}
    return {
        known[result.stage_name]
        for result in context.stage_results
        if result.status == PipelineStageStatus.COMPLETED and result.stage_name in known
    }


def _tree_pruned(results: Mapping[str, IndexingStats]) -> bool:
    """Whether the run's deletion of the summaries citing removed text
    succeeded (every ``vector_summary_nodes_*`` result written)."""
    return not any(
        stats.failed_items
        for key, stats in results.items()
        if key.startswith("vector_summary_nodes_")
    )


def _build_knowledge_graph(context: PipelineContext) -> Any:
    """Build the knowledge graph from the context's resolved graph outputs.

    graph_analysis builds it; community_detection rebuilds it the same way when
    a resumed run restored only the cached entities, relationships and claims
    (the graph itself is neither cached nor kept in the run metadata).
    """
    claims = _resolved_claims(context)
    return GraphBuilder(
        context.resolved_entities, context.resolved_relationships, claims
    ).build()


def _failed_files_note(failed_files: Sequence[str]) -> str:
    """`` It failed on N files: a, b ...`` (the first ten), or nothing."""
    if not failed_files:
        return ""
    return (
        f" It failed on {len(failed_files)} files: "
        + ", ".join(failed_files[:10])
        + (" ..." if len(failed_files) > 10 else "")
    )


class PipelineStage(ABC):
    MUST_HAVE_INPUT_STAGES = {
        PipelineStageType.COMMUNITY_DETECTION,
        PipelineStageType.GRAPH_ANALYSIS,
        PipelineStageType.GRAPH_EXTRACTION,
        PipelineStageType.GRAPH_RESOLUTION,
        PipelineStageType.DOCUMENT_LOADING,
        PipelineStageType.DOCUMENT_PARSING,
        PipelineStageType.TEXT_CHUNKING,
        PipelineStageType.TRANSLATION,
    }

    # The stages that build the graph from the extracted entities.
    GRAPH_STAGES = {
        PipelineStageType.CLAIM_EXTRACTION,
        PipelineStageType.CLAIM_RESOLUTION,
        PipelineStageType.COMMUNITY_DETECTION,
        PipelineStageType.GLEANING,
        PipelineStageType.GRAPH_ANALYSIS,
        PipelineStageType.GRAPH_EXTRACTION,
        PipelineStageType.GRAPH_RESOLUTION,
    }

    # Stages that may produce nothing from a non-empty input; every other
    # stage fails on zero outputs.
    OPTIONAL_OUTPUT_STAGES = {
        PipelineStageType.CLAIM_EXTRACTION,
        PipelineStageType.CLAIM_RESOLUTION,
        PipelineStageType.GLEANING,
    }

    # Whether the stage reads its input from the stores, not from earlier
    # stages' output (summary_tree): a run of such stages alone needs no
    # upstream stage cached and creates no artifact index of the run.
    READS_STORES: ClassVar[bool] = False

    def __init__(
        self,
        stage_type: PipelineStageType,
        config: Config,
        providers: Providers | None = None,
    ) -> None:
        self.config = config
        self.stage_type = stage_type
        # The pipeline passes its one provider bundle, so every stage's LLM /
        # embedding adapters share the session and any injected factory. The
        # session itself is only created when an AWS adapter asks for it.
        self.providers = Providers.resolve(config, providers)

    @property
    def name(self) -> str:
        return self.stage_type.value

    def close(self) -> None:  # noqa: B027 - intentional no-op default hook
        """Release any backend clients the stage owns.

        Default is a no-op: only ``IndexingStage`` holds long-lived backend
        connections. Not abstract on purpose — stages opt in by overriding.
        """

    def execute(self, context: PipelineContext) -> PipelineStageResult:
        with observability.span("kg_rag.ingestion.stage", stage=self.name) as span:
            result = self._execute_logged(context, span)
            span.set_attributes(
                {
                    "kg_rag.stage.status": result.status,
                    "kg_rag.input_count": result.input_count,
                    "kg_rag.output_count": result.output_count,
                    # Counters such as dropped units or failed writes show a
                    # partial degradation on a stage that still completed.
                    **{
                        f"kg_rag.stage.{key}": value
                        for key, value in result.metrics.items()
                        if isinstance(value, int | float)
                        and not isinstance(value, bool)
                    },
                }
            )
            if result.status == PipelineStageStatus.FAILED:
                span.set_error(f"stage '{self.name}' failed")
            return result

    def _execute_logged(
        self, context: PipelineContext, span: observability.SpanHandle
    ) -> PipelineStageResult:
        with failure_ledger() as ledger:
            return self._execute_in_ledger(context, span, ledger)

    def _execute_in_ledger(
        self,
        context: PipelineContext,
        span: observability.SpanHandle,
        ledger: FailureLedger,
    ) -> PipelineStageResult:
        tokens = bind_contextvars(stage=self.name)
        logger.info("Starting stage: '%s'", self.name)
        start_time = datetime.now()
        # Counted when the output check fails the stage, so the failed
        # result agrees with the error naming them.
        input_count = output_count = 0

        try:
            input_count, output_count, metrics = self._execute_core(context)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if not self._allows_empty_output(context):
                self._validate_critical_stage_output(
                    input_count,
                    output_count,
                    unit_source_files(
                        context, context.failed_text_unit_ids.get(self.name, ())
                    ),
                )

            logger.info(
                "Stage '%s' completed successfully in %.2f seconds. "
                "Inputs: %s, Outputs: %s",
                self.name,
                duration,
                input_count,
                output_count,
            )

            return self._create_result(
                PipelineStageStatus.COMPLETED,
                start_time,
                end_time,
                input_count,
                output_count,
                metrics,
            )

        except ConfirmationDeclinedError as e:
            # Not a failure: the run stops before it writes (the pipeline
            # records the stage and raises it on).
            log_error(
                logger, logging.WARNING, "Stage '%s' stopped: %s", self.name, error=e
            )
            raise
        except Exception as raised:  # noqa: BLE001 - logged, recorded, raised
            error, named_after = self._stage_error(raised, ledger)
            log_error(
                logger, logging.ERROR, "Stage '%s' failed: %s", self.name, error=error
            )
            span.record_exception(error)
            span.set_attribute(ERROR_TYPE, type(error).__name__)
            end_time = datetime.now()

            result = self._create_result(
                PipelineStageStatus.FAILED,
                start_time,
                end_time,
                input_count,
                output_count,
                # What a surface shows (an unexpected failure's text masked,
                # at DEBUG in the log): the result is persisted (the run's
                # metadata, its S3 checkpoint) and shown by every surface.
                error_message=operator_error_text(error),
            )
            result.error_class = classify(error)
            # The model's error a stage failure was named after, else its own.
            result.error_type = type(named_after or error).__name__
            return result
        finally:
            reset_contextvars(**tokens)

    @abstractmethod
    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        pass

    def _stage_error(
        self, error: Exception, ledger: FailureLedger
    ) -> tuple[Exception, BaseException | None]:
        """What the stage failed with: ``error``, or, when it failed because
        its model calls did (a critical stage that produced nothing), their
        dominant cause (``ledger``): a model server that could not be
        reached as the ``ModelUnreachableError`` naming it, a store as its
        ``StoreConnectionError``, a failure a known-good request met too as
        the model's or a setting's, and one that is no item's own (the
        throttling, server error or timeout that outlasted the retries) as
        the model's, its quota's or the service's, with how many calls it
        ended; not the documents'. With the error a named failure was named
        after (its ``error_type``), else ``None``."""
        if is_unreachable_error(error):
            return model_unreachable_error(error, self.config), None
        found = ledger.dominant_count()
        if not isinstance(error, PipelineStageError) or found is None:
            return error, None
        dominant, count = found
        if is_unreachable_error(dominant):
            return model_unreachable_error(dominant, self.config), None
        if isinstance(dominant, StoreConnectionError):
            return dominant, None
        # Named by its type, code and status, never its message, which may
        # quote the request's text; the error itself is the cause.
        cause = describe_cause(dominant)
        if ledger.model_failed:
            named = PipelineStageError(
                f"Stage '{self.name}' failed: every model call and a known-good "
                f"request failed alike ({cause}), a failure of the model or a "
                "setting, not of the documents."
                + _failed_files_note(error.failed_files),
                failed_files=error.failed_files,
                own_text=True,
            )
        elif not item_error_predicate(self.providers.llm_factory)(dominant):
            # Throttling, a 5xx or a timeout that outlasted the retries: the
            # known-good request is sent only for the items' own errors.
            named = PipelineStageError(
                f"Stage '{self.name}' failed: {count} of its "
                f"{ledger.failures} failed model calls ended in {cause} after "
                "their retries, a failure of the model, its quota or the "
                "service, not of the documents."
                + _failed_files_note(error.failed_files),
                failed_files=error.failed_files,
                own_text=True,
            )
        else:
            return error, None
        named.__cause__ = dominant
        return named, dominant

    def _allows_empty_output(self, context: PipelineContext) -> bool:
        """Stages may opt out of the zero-output critical check for valid cases.

        In incremental mode the doc-status registry can legitimately filter the
        corpus to zero documents to (re)extract — a deletion-only delta still
        propagates deletions (handled at the loading stage) and an all-unchanged
        delta is a no-op run. In that case every document-processing stage
        receives empty input, which is valid rather than a failure, so the
        must-have-input / critical-output checks are skipped pipeline-wide.

        Likewise the new and changed documents of an incremental delta may
        name no entity at all (an edit that removed the last one). The graph
        stages then produce, or receive, nothing; the documents are still
        committed (with an empty lineage) and the delta's deletions applied.
        Failing here would stop every later run at the same documents. The
        check stays when a text unit failed (an unreachable model also
        extracts nothing) and for a full build, where a corpus without a
        single entity points at a broken model or prompt. A failure of the
        claim stages does not count: claims are optional, and a unit they
        failed on still had its entities extracted.
        """
        if context.incremental_delta is None:
            return False
        if not context.documents:
            return True
        claim_stages = {
            PipelineStageType.CLAIM_EXTRACTION.value,
            PipelineStageType.CLAIM_RESOLUTION.value,
        }
        return (
            self.stage_type in self.GRAPH_STAGES
            and not self.config.indexing.reset
            and not any(
                unit_ids
                for name, unit_ids in context.failed_text_unit_ids.items()
                if name not in claim_stages
            )
        )

    def _empty_input_hint(self) -> str:
        return "Check that the previous stage completed successfully."

    def _validate_critical_stage_output(
        self, input_count: int, output_count: int, failed_files: Sequence[str] = ()
    ) -> None:
        """Raise when a stage that must have input got none, or a critical
        stage produced nothing; the message names ``failed_files``, the source
        files of the units the stage failed on. Not logged here: the stage
        logs the error it fails with (``_execute_logged``)."""
        is_critical = self.stage_type not in self.OPTIONAL_OUTPUT_STAGES
        must_have_input = self.stage_type in self.MUST_HAVE_INPUT_STAGES

        if input_count == 0:
            if must_have_input:
                error_msg = (
                    f"Stage '{self.name}' received 0 inputs but requires input "
                    f"to function. {self._empty_input_hint()}"
                )
                raise PipelineStageError(error_msg, own_text=True)
            logger.info("Stage '%s' had no input to process", self.name)
            return

        if is_critical and output_count == 0:
            error_msg = (
                f"Critical stage '{self.name}' processed "
                f"{input_count} inputs but produced 0 outputs. Check the stage "
                f"configuration and input data quality."
            )
            raise PipelineStageError(
                error_msg + _failed_files_note(failed_files),
                failed_files=list(failed_files),
                own_text=True,
            )

        if output_count == 0:
            logger.warning(
                "Stage '%s' processed %s inputs but produced 0 outputs",
                self.name,
                input_count,
            )

    def _create_result(
        self,
        status: PipelineStageStatus,
        start_time: datetime,
        end_time: datetime,
        input_count: int = 0,
        output_count: int = 0,
        metrics: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> PipelineStageResult:
        duration = (end_time - start_time).total_seconds()

        return PipelineStageResult(
            stage_name=self.name,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            input_count=input_count,
            output_count=output_count,
            metrics=metrics or {},
            error_message=error_message,
        )


class _RegistryStage(PipelineStage):
    """A stage that reads or writes the doc-status registry: the injected one
    (``doc_status``), else the one ``config.providers`` selects."""

    _doc_status: "DocStatusPort | None" = None

    @property
    def _registry_on(self) -> bool:
        """Whether incremental indexing is on (a registry injected or enabled)."""
        return self._doc_status is not None or self.config.doc_status_enabled

    def _build_doc_status_store(self) -> "DocStatusPort":
        if self._doc_status is not None:
            return self._doc_status
        return self.providers.build_doc_status()


class DocumentLoadingStage(_RegistryStage):
    """Produce the run's document set (dedup + incremental filter).

    When the parsing stage completed in this pipeline, its parsed documents are
    already on the context (or restored from its stage cache on resume), so they
    are reused directly: re-reading them from disk added nothing and, with the
    parsed JSON written next to the raw corpus, made the loader try to read raw
    files as JSON. Only without a parsing stage (a corpus of pre-parsed
    ``Document`` JSON files) does this stage read ``source_directory``, and then
    only ``.json`` files.
    """

    # The run's seed graphs, and who answers a reset's question and a mass
    # deletion's (__init__).
    seed_conversions: tuple[SeedConversion, ...] = ()
    confirm: Callable[[str], bool] | None = None
    confirm_deletions: Callable[[str], bool] | None = None
    rebuild_for_artifacts: Callable[[], bool] | None = None
    deletions: Callable[[], Sequence[str] | None] | None = None

    def __init__(
        self,
        config: Config,
        source_directory: Path,
        doc_status: "DocStatusPort | None" = None,
        providers: Providers | None = None,
        seed_conversions: Sequence[SeedConversion] = (),
        confirm: Callable[[str], bool] | None = None,
        confirm_deletions: Callable[[str], bool] | None = None,
        rebuild_for_artifacts: Callable[[], bool] | None = None,
        deletions: Callable[[], Sequence[str] | None] | None = None,
    ):
        super().__init__(PipelineStageType.DOCUMENT_LOADING, config, providers)
        # The run's seed graphs (bootstrap.load_seed_graphs): their shards
        # are virtual documents of the corpus.
        self.seed_conversions = tuple(seed_conversions)
        # Answers the reset's question when it drops another scope's seed
        # graph documents (incremental.filter_to_delta), and a registry-off
        # run's from seeds alone (_confirm_seeds_alone); None cannot.
        self.confirm = confirm
        # Answers an incremental run's question before it deletes more than
        # indexing.max_deletion_share of its scope's documents; None cannot.
        self.confirm_deletions = confirm_deletions
        # Whether the run's indexing.reset is the pipeline's rebuild for an
        # artifact the namespace lacks (not the user's): its deletions are
        # asked like an incremental run's.
        self.rebuild_for_artifacts = rebuild_for_artifacts
        # The ids a run of document changes deletes (DataIngestionPipeline.
        # insert_documents / delete_documents); None for a corpus run.
        self.deletions = deletions
        # Without parsing, the directory holds pre-parsed Document JSON.
        # Deduplication runs in this stage, after document identity is
        # assigned, so an incremental run can prefer indexed documents.
        self.loader = DirectoryLoader(source_directory)
        # Injected by the pipeline (built once at the orchestration layer) when
        # incremental indexing is enabled; built lazily otherwise.
        self._doc_status = doc_status

    def _warn_of_another_script(
        self, documents: list[Document], context: PipelineContext
    ) -> None:
        """WARN when a sample of the corpus is written in another script
        than processing.translation.source_language says (a Korean corpus
        under the default 'en' is extracted with English prompts and
        analyzed as English); the run goes on. An incremental run reads
        only its delta: the unchanged documents are sampled from their
        stored text, each part its share of the corpus (``sample_split``)."""
        source_language = self.config.processing.translation.source_language
        texts = [
            document.page_content
            for document in documents
            if not is_seed_document(document) and document.page_content
        ]
        delta = context.incremental_delta
        of_read, of_stored = sample_split(
            len(texts), len(delta.unchanged) if delta is not None else 0
        )
        stored = self._stored_texts(of_stored) if of_stored else []
        sample = corpus_sample([*texts[:of_read], *stored] if stored else texts)
        if reason := source_language_mismatch(sample, source_language):
            logger.warning("Corpus language: %s", reason)

    def _stored_texts(self, limit: int) -> list[str]:
        """The stored text of up to ``limit`` documents of the namespace, a
        text unit of each, from one page of its text units; none when the
        store cannot page them or a read fails (the sample is the run's)."""
        suffix = resolve_namespace(self.config, None)
        try:
            indexer = self.providers.build_vector_indexer()
            try:
                pages = indexer.scan_text_unit_pages(
                    suffix, fields=["text", "document_ids"], page_size=limit * 5
                )
                page = next(iter(pages), []) if pages is not None else []
            finally:
                indexer.close()
        except Exception as e:  # noqa: BLE001 - the warning is advisory
            logger.debug("Not sampling the stored text of '%s': %s", suffix, e)
            return []
        texts: dict[str, str] = {}
        for row in page:
            text, documents = row.get("text"), row.get("document_ids") or [row["id"]]
            if isinstance(text, str) and text and len(texts) < limit:
                texts.setdefault(str(documents[0]), text)
        return list(texts.values())

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        # A context restored from a reused pipeline id carries the previous
        # run's delta in its metadata. This stage recomputes the document set,
        # so that delta no longer describes it: drop it before (re)deriving one,
        # or a run with the registry now disabled (or a failed diff under
        # continue_on_error) would index against stale fingerprints.
        context.incremental_delta = None
        context.incremental_fingerprints = {}
        context.incremental_scope = None
        context.incremental_extraction_fingerprint = None
        context.incremental_extraction_settings = None
        context.incremental_retired_scopes = []

        failed_files: list[str] = []
        if self._parsing_completed(context):
            documents = list(context.documents)
            input_count = len(documents)
            logger.info(
                "Using %s documents from the '%s' stage",
                input_count,
                PipelineStageType.DOCUMENT_PARSING.value,
            )
        else:
            discovered_files = self.loader.discover_files()
            input_count = len(discovered_files)
            # A deletion from memory stages no document: no empty corpus to
            # warn of ("No supported files found").
            deletion_only = self.deletions is not None and not discovered_files
            documents = [] if deletion_only else self.loader.load()
            for document in documents:
                assign_document_identity(
                    document,
                    self.loader.source_directory,
                    self.config.indexing.additional_suffix,
                    store_namespace_tag(self.config.indexing),
                )
                if self.deletions is not None:
                    # Documents given from memory, staged in a temporary
                    # directory: their path is their id.
                    document.file_path = document.metadata[RELATIVE_PATH_KEY]
            failed_files = self.loader.failed_files
            context.failed_source_files = list(failed_files)
            context.skipped_source_files = []

        incremental = self._registry_on
        if self.config.processing.deduplicate:
            # Deduplication compares the texts.
            documents = self._read_unparsed(documents)
            keep = avoid = None
            if incremental and not self.config.indexing.reset:
                keep, avoid = dedup_preferences(
                    documents,
                    context,
                    self.config,
                    self._build_doc_status_store,
                    self.loader.source_directory,
                )
            documents = self.loader.deduplicate_documents(
                documents, keep=keep, avoid=avoid
            )
        # After deduplication: a seed shard is no copy of a file.
        seeds = self._seed_documents()
        documents.extend(seeds)

        # Incremental indexing: when a doc-status registry is injected or the
        # DynamoDB one is enabled, diff against it, stash the delta/fingerprints
        # for the IndexingStage, and process only new/changed documents.
        delta_skipped = 0
        if incremental:
            documents, delta_skipped = self._apply_incremental_filter(
                documents, context
            )
        elif corpus_is_seeds_only(documents):
            self._confirm_seeds_alone()
        # What reaches the stages after this one is read (an unparsed
        # document of a parse cached by a run with the registry on).
        documents = self._read_unparsed(documents)

        context.documents = documents
        output_count = len(context.documents)
        seeds_kept = sum(1 for document in documents if is_seed_document(document))
        self._warn_of_another_script(documents, context)

        if failed_files:
            logger.warning("Failed to load %s files", len(failed_files))

        metrics = {
            "file_count": input_count,
            "success_count": output_count,
            "delta_skipped": delta_skipped,
            "seed_documents": len(seeds),
            "seed_documents_skipped": len(seeds) - seeds_kept,
            "documents_quarantined": (
                len(context.incremental_delta.quarantined)
                if context.incremental_delta is not None
                else 0
            ),
            "documents_deferred": (
                len(context.incremental_delta.deferred)
                if context.incremental_delta is not None
                else 0
            ),
            "documents_excluded": (
                len(context.incremental_delta.excluded)
                if context.incremental_delta is not None
                else 0
            ),
            "failed_files": failed_files,
            "files_failed": len(failed_files),
        }

        return input_count + len(seeds), output_count, metrics

    def _seed_documents(self) -> list[Document]:
        """The seed graphs' shard documents, identified like the corpus's:
        the run's index value, namespace and store tag."""
        index_value = self.config.processing.document_parsing.index_value
        documents = []
        for conversion in self.seed_conversions:
            for document in conversion.documents():
                if index_value is not None:
                    document.metadata[Constants.INDEX.value] = index_value
                assign_document_identity(
                    document,
                    self.loader.source_directory,
                    self.config.indexing.additional_suffix,
                    store_namespace_tag(self.config.indexing),
                )
                documents.append(document)
        if documents:
            logger.info(
                "Adding %s seed graph documents (%s seed graphs)",
                len(documents),
                len(self.seed_conversions),
            )
        return documents

    def _confirm_seeds_alone(self) -> None:
        """Ask before a registry-off run of seed graphs alone rebuilds an
        index that holds items.

        Without the registry every run is a full build of its corpus, and
        nothing tells an empty corpus from a missing one (a volume not
        mounted, an S3 sync that copied nothing): the seeds alone would
        replace every document the index holds. An index that holds no
        item yet (a first seed-only build) loses nothing and is built
        without asking (``holds_any_item`` and the vector store's
        ``holds_any_text_unit``, a bounded read each);
        a store that cannot tell, or whose read fails, counts as holding
        items. ``confirm`` answers as for a reset (``--confirm-reset``,
        ``UKR_CONFIRM_RESET=1`` or ``--yes``); ``None`` cannot, and the run
        stops before it writes.

        Raises:
            ConfirmationDeclinedError: The build is not confirmed.
        """
        index_value = self.config.processing.document_parsing.index_value
        suffix = resolve_namespace(self.config, None)
        if self._stores_hold_items(suffix) is False:
            logger.info(
                "No source document reached this run and the index holds no "
                "item yet: building it from the seed graphs alone"
            )
            return
        namespace = config_registry_namespace(self.config.indexing, index_value)
        question = (
            f"No source document reached this run, only its seed graphs, and "
            f"incremental indexing is off (providers.doc_status.enabled), so "
            f"an empty corpus looks like a missing one: rebuild the index "
            f"{namespace!r} from the seed graphs alone, replacing what it "
            f"holds; continue?"
        )
        if self.confirm is None:
            raise ConfirmationDeclinedError(
                f"{question} No one can confirm it (run-ingestion "
                "--confirm-reset does)"
            )
        if not self.confirm(question):
            raise ConfirmationDeclinedError(f"Build cancelled: {question}")

    def _stores_hold_items(self, suffix: str) -> bool | None:
        """Whether either store holds an entity or relationship of
        ``suffix``, or the vector store a text unit (of a corpus that
        stated no item); ``None`` when one cannot tell or a read fails."""
        answers: list[bool | None] = []
        try:
            graph = self.providers.build_graph_indexer()
            try:
                held = graph.holds_any_item(suffix)
            finally:
                graph.close()
            if held:
                return True
            answers.append(held)
            vector = self.providers.build_vector_indexer()
            try:
                for probe in (vector.holds_any_item, vector.holds_any_text_unit):
                    held = probe(suffix)
                    if held:
                        return True
                    answers.append(held)
            finally:
                vector.close()
        except Exception as e:  # noqa: BLE001 - read as holding: the run asks
            log_error(
                logger,
                logging.WARNING,
                "Could not tell whether the stores hold items of '%s' (%s)",
                suffix,
                error=e,
            )
            return None
        return None if None in answers else False

    def _apply_incremental_filter(
        self, documents: list[Document], context: PipelineContext
    ) -> tuple[list[Document], int]:
        """Keep the new and changed documents (see ``incremental.filter_to_delta``)."""
        return filter_to_delta(
            documents,
            context,
            self.config,
            self._build_doc_status_store,
            self.loader.source_directory,
            self.confirm,
            self.confirm_deletions,
            guard_reset_deletions=(
                self.rebuild_for_artifacts is not None and self.rebuild_for_artifacts()
            ),
            deletions=self.deletions() if self.deletions is not None else None,
            read_unparsed=self._source_parser().read_unparsed,
        )

    def _source_parser(self) -> SourceFileParser:
        return SourceFileParser(self.config, self.loader.source_directory)

    def _read_unparsed(self, documents: list[Document]) -> list[Document]:
        """``documents`` with every unparsed one read (one that fails to
        parse is dropped)."""
        if not any(is_unparsed(document) for document in documents):
            return documents
        return self._source_parser().read_unparsed(documents)[0]

    @staticmethod
    def _parsing_completed(context: PipelineContext) -> bool:
        return any(
            r.stage_name == PipelineStageType.DOCUMENT_PARSING.value
            and r.status == PipelineStageStatus.COMPLETED
            for r in context.stage_results
        )


class DocumentParsingStage(_RegistryStage):
    """Parse the raw corpus into ``Document`` objects.

    When ``target_directory`` is set, each parsed document is also exported as
    ``<stem>.json`` there for inspection; nothing reads the export back (the
    loading stage reuses the parsed documents directly). The target may not be
    the source directory: ``.json`` is itself a parseable source format, so a
    re-run would ingest the previous run's output as new (duplicate) documents.
    Discovery likewise skips the target and cache directories, which hold JSON
    this pipeline wrote.
    """

    def __init__(
        self,
        config: Config,
        source_directory: Path,
        target_directory: Path | None = None,
        cache_directory: Path | None = None,
        providers: Providers | None = None,
        doc_status: "DocStatusPort | None" = None,
    ):
        super().__init__(PipelineStageType.DOCUMENT_PARSING, config, providers)
        # The registry whose records tell which files need no parse
        # (source_parsing); injected by the pipeline when the loading stage
        # runs too, since it reads the documents that are not parsed here.
        self._doc_status = doc_status
        self.source_directory = Path(source_directory)
        self.target_directory = Path(target_directory) if target_directory else None
        self.cache_directory = Path(cache_directory) if cache_directory else None
        if (
            self.target_directory is not None
            and self.target_directory.resolve() == self.source_directory.resolve()
        ):
            raise ConfigurationError(
                "document_parsing.target_directory must not be the source "
                f"directory ('{self.source_directory}'): parsed .json output there "
                "is re-ingested as new documents on the next run"
            )
        self.supported_extensions = ParserFactory.get_supported_extensions()

    def _allows_empty_output(self, context: PipelineContext) -> bool:
        """An empty corpus is a seed-only run when seed graphs are
        configured: the loading stage adds their documents. A corpus whose
        every file failed to parse still fails."""
        seed_only_run = bool(self.config.processing.seed_graphs) and not (
            context.failed_source_files
        )
        return seed_only_run or super()._allows_empty_output(context)

    def _empty_input_hint(self) -> str:
        return (
            f"No supported source files found in '{self.source_directory}' "
            f"(supported: {', '.join(sorted(self.supported_extensions))}); see "
            "the warnings above for skipped files."
        )

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        files_to_parse, skipped_files = self._discover_source_files()
        input_count = len(files_to_parse)
        # Incremental delta detection must not mistake a file of a type no
        # loader reads (an optional extra dropped from the image, a loader
        # plugin not imported) for a deleted one: kept like a parse failure.
        context.skipped_source_files = skipped_files

        if not files_to_parse:
            if self.config.processing.seed_graphs:
                logger.info(
                    "No supported files found in '%s': the corpus is its seed "
                    "graphs",
                    self.source_directory,
                )
            else:
                logger.warning(
                    "No supported files found in '%s'", self.source_directory
                )
            # Cleared: a context restored from an earlier run keeps its files.
            context.documents = []
            context.failed_source_files = []
            return (
                0,
                0,
                {
                    "parsed_files": [],
                    "failed_files": [],
                    "files_failed": 0,
                    "files_skipped": len(skipped_files),
                },
            )

        logger.info("Found %s files to parse", input_count)

        parser = SourceFileParser(self.config, self.source_directory)
        digests: dict[Path, str] | None = None
        reusable: dict[Path, Document] = {}
        if self._registry_on:
            digests, reusable = self._reusable_parses(parser, files_to_parse)
        parsed, failed_files = parser.parse(
            [path for path in files_to_parse if path not in reusable], digests
        )
        parsed_documents = []
        for file_path in files_to_parse:
            if (unparsed := reusable.get(file_path)) is not None:
                parsed_documents.append(unparsed)
            elif (document := parsed.get(file_path)) is not None:
                parsed_documents.append(document)
                if self.target_directory is not None:
                    self._save_parsed_document(
                        document, file_path, self.target_directory
                    )
        if reusable:
            logger.info(
                "%d of %d files are unchanged since they were indexed (source "
                "digest): not parsed",
                len(reusable),
                input_count,
            )

        context.documents = parsed_documents
        # Incremental delta detection must not mistake a file that failed to
        # parse for a deleted one (that would remove its indexed content).
        context.failed_source_files = failed_files
        output_count = len(parsed_documents)

        metrics = {
            "input_files": input_count,
            "parsed_files": [
                doc.file_name for doc in parsed_documents if not is_unparsed(doc)
            ],
            "files_unparsed": len(reusable),
            "failed_files": failed_files,
            "files_failed": len(failed_files),
            "files_skipped": len(skipped_files),
            # A fraction, as every rate in the run metadata.
            "success_rate": output_count / input_count if input_count > 0 else 0,
        }

        return input_count, output_count, metrics

    def _reusable_parses(
        self, parser: SourceFileParser, files: list[Path]
    ) -> tuple[dict[Path, str], dict[Path, Document]]:
        """Every file's source digest, and the unparsed documents of the
        files whose PROCESSED record holds theirs
        (``source_parsing.reusable_records``): they need no parse. None
        without an injected registry (the pipeline injects it
        when the loading stage runs, which reads what is not parsed here) or
        when the run reads every document's text anyway: a reset,
        deduplication (it compares the texts) or an export of the parsed
        documents (``target_directory``)."""
        digests = {}
        for path in files:
            try:
                digests[path] = parser.source_digest(path)
            except Exception:  # noqa: BLE001 - its parse reports it
                continue
        if (
            self._doc_status is None
            or self.config.indexing.reset
            or self.config.processing.deduplicate
            or self.target_directory is not None
        ):
            return digests, {}
        # Resolved, as the loading stage keys the run (DirectoryLoader).
        _, _, source_scope, _ = run_registry_scope(
            self.config, self.source_directory.resolve(), []
        )
        try:
            return digests, reusable_records(
                parser, digests, self._doc_status, source_scope
            )
        except Exception as e:  # noqa: BLE001 - every file is parsed instead
            log_error(
                logger,
                logging.WARNING,
                "Could not read the doc-status registry to skip unchanged files "
                "(%s): parsing every file",
                error=e,
            )
            return digests, {}

    def _discover_source_files(self) -> tuple[list[Path], list[str]]:
        """The files to parse, and the files skipped for a file type no
        loader reads (one WARNING names them by extension)."""
        if not self.source_directory.exists():
            raise FileNotFoundError(
                f"Source directory not found: {self.source_directory}"
            )

        # Pipeline-written trees nested under the source dir (the export
        # target, or the cache dir, e.g. source "." with cache "./cache") hold
        # JSON this pipeline produced, never source documents.
        owned_dirs = [
            d.resolve() for d in (self.target_directory, self.cache_directory) if d
        ]
        files = []
        skipped_files: list[str] = []
        skipped: Counter[str] = Counter()
        outside: list[str] = []
        root = self.source_directory
        resolved_root = root.resolve()
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if is_excluded_source_path(file_path, root, resolved_root):
                if leaves_source_root(file_path, root, resolved_root):
                    outside.append(str(file_path))
                continue
            real = real_source_path(file_path, root, resolved_root)
            if any(real.is_relative_to(d) for d in owned_dirs):
                continue
            suffix = file_path.suffix.lower()
            if suffix in self.supported_extensions:
                files.append(file_path)
            elif suffix:
                skipped[suffix] += 1
                skipped_files.append(str(file_path))

        if skipped:
            hints = {s: ParserFactory.install_hint(s) for s in sorted(skipped)}
            others = [suffix for suffix, hint in hints.items() if hint is None]
            remedies = [
                f"for {suffix}, {hint}" for suffix, hint in hints.items() if hint
            ]
            if others:
                remedies.append(
                    f"for {', '.join(others)}, convert them to a supported "
                    "format or register a loader"
                )
            logger.warning(
                "Skipping %s file(s) in '%s' of a type no loader reads (%s): "
                "they are not indexed, and an incremental run keeps what they "
                "indexed before (not treated as deleted); to ingest them, %s",
                sum(skipped.values()),
                self.source_directory,
                ", ".join(
                    f"{count} '{suffix}'" for suffix, count in sorted(skipped.items())
                ),
                "; ".join(remedies),
            )
        if outside:
            logger.warning(
                "Skipping %s file(s) in '%s' that resolve outside it (a symlink "
                "leading elsewhere): they are not indexed (%s); copy them into "
                "the source directory to ingest them",
                len(outside),
                self.source_directory,
                ", ".join(sorted(outside)[:5]) + (", ..." if len(outside) > 5 else ""),
            )
        return sorted(files), sorted(skipped_files)

    def _save_parsed_document(
        self, document: Document, original_path: Path, output_directory: Path
    ) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_filename = f"{original_path.stem}.json"
        output_path = output_directory / output_filename

        try:
            document.to_json_file(output_path)
            logger.debug("Saved parsed document: '%s'", output_path)
        except (OSError, ValueError) as e:
            log_error(
                logger,
                logging.ERROR,
                "Failed to save parsed document '%s': %s",
                output_path,
                error=e,
            )


class TextChunkingStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
        reuse_stored: bool = True,
    ):
        super().__init__(PipelineStageType.TEXT_CHUNKING, config, providers)

        chunker_type = self.config.processing.chunking.chunker_type
        # The intelligent chunker's boundary answers are kept in the
        # extraction store (by pre-chunk text and chunking settings): a run
        # reads them instead of asking the model again, unless it rebuilds
        # everything (--force-rebuild), and refreshes them either way.
        self.store = (
            ExtractionStore(
                self.config, self.providers.build_vector_indexer, reuse=reuse_stored
            )
            if chunker_type is ChunkingStrategy.INTELLIGENT
            else None
        )
        self.chunker = ChunkerFactory.create_chunker(
            config=self.config,
            chunker_type=chunker_type,
            providers=self.providers,
            boundary_records=self.store,
        )
        self.chunker_type = chunker_type

    def close(self) -> None:
        if self.store is not None:
            self.store.close()

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        # One call for the whole corpus: the intelligent chunker batches its
        # boundary-model calls across documents, and the logged stats cover
        # the stage rather than its last document. A seed graph's shard is
        # cut into its units as written (no model): document units first.
        documents = [d for d in context.documents if not is_seed_document(d)]
        text_units = self.chunker.chunk_documents(documents)
        seed_units = [
            unit
            for document in context.documents
            if is_seed_document(document)
            for unit in seed_text_units(document, count_tokens=estimate_token_count)
        ]
        text_units.extend(seed_units)
        context.text_units = text_units
        total_chunks = len(text_units)
        doc_count = len(context.documents)

        metrics = {
            "chunker_type": self.chunker_type.value,
            "documents_processed": doc_count,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": total_chunks / doc_count if doc_count else 0.0,
            "seed_units": len(seed_units),
        }

        return doc_count, total_chunks, metrics


class TranslationStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
        reuse_stored: bool = True,
    ):
        super().__init__(PipelineStageType.TRANSLATION, config, providers)
        self.translation_config = self.config.processing.translation
        self.target_language = self.translation_config.target_language.value
        # --force-rebuild translates every unit again (and refreshes the
        # stored translations).
        self.reuse_stored = reuse_stored
        # Build the Bedrock-backed translator lazily so a disabled / no-op stage
        # costs nothing (no LLM client, no calls).
        self._translator: TextUnitTranslator | None = None
        self._store: ExtractionStore | None = None

    @property
    def translator(self) -> TextUnitTranslator:
        if self._translator is None:
            # The units' translations are kept in the extraction store (by
            # text and translation settings): a reset or an edited
            # document translates only the text that changed.
            self._store = ExtractionStore(
                self.config,
                self.providers.build_vector_indexer,
                reuse=self.reuse_stored,
            )
            self._translator = TextUnitTranslator(
                self.config, providers=self.providers, records=self._store
            )
        return self._translator

    def close(self) -> None:
        if self._store is not None:
            self._store.close()

    def _should_skip(self) -> str | None:
        """Return a reason to skip translation, or None to run it."""
        if not self.translation_config.enabled:
            return "translation disabled in config"
        if self.translation_config.is_noop:
            return (
                f"source and target language are both "
                f"'{self.target_language}' with no additional targets"
            )
        return None

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        skip_reason = self._should_skip()
        if skip_reason is not None:
            logger.info("Skipping translation: %s", skip_reason)
            # Leave translated_units empty; downstream stages fall back to
            # context.text_units, so this is a true no-op.
            context.translated_units = []
            context.failed_text_unit_ids[self.name] = []
            metrics = {
                "target_language": self.target_language,
                "units_processed": len(context.text_units),
                "units_translated": 0,
                "skipped": True,
                "skip_reason": skip_reason,
            }
            return len(context.text_units), len(context.text_units), metrics

        # A seed unit's statements are taken as written: never translated.
        translated = {
            unit.id: unit
            for unit in self.translator.translate_text_units(
                [u for u in context.text_units if not is_seed_unit(u.id)]
            )
        }
        translated_units = [translated.get(u.id, u) for u in context.text_units]
        context.translated_units = translated_units
        # Units left untranslated are extracted in the source language; their
        # documents are recorded FAILED and retried, like an extraction failure.
        stats = self.translator.stats
        context.failed_text_unit_ids[self.name] = (
            list(stats.failed_text_unit_ids) if stats else []
        )
        context.item_failed_text_unit_ids[self.name] = (
            list(stats.item_failed_text_unit_ids) if stats else []
        )

        translation_stats = stats.model_dump() if stats else {}

        units_translated_count = len(
            [
                u
                for u in translated_units
                if u.translated_texts and self.target_language in u.translated_texts
            ]
        )

        metrics = {
            "target_language": self.target_language,
            "units_processed": len(context.text_units),
            "units_translated": units_translated_count,
            "failed_units": len(context.failed_text_unit_ids[self.name]),
            "translation_stats": translation_stats,
        }

        return len(context.text_units), len(translated_units), metrics


def _failed_record_units(context: PipelineContext) -> set[str]:
    """The text units whose extraction or gleaning failed: their records
    (if any) are not final, so they are neither stored nor their claims."""
    return {
        unit_id
        for stage in (PipelineStageType.GRAPH_EXTRACTION, PipelineStageType.GLEANING)
        for unit_id in context.failed_text_unit_ids.get(stage.value, [])
    }


def _with_record_keys(
    config: Config, context: PipelineContext, text_units: list[TextUnit]
) -> list[TextUnit]:
    """The text units to write, naming the run's stored records of them.

    Computed from what the run's stages recorded (their records and failed
    units), never set on the cached units, so a run resumed after any stage
    writes the keys a run without the interruption writes.
    """
    failures = context.failed_text_unit_ids
    graph_failed = _failed_record_units(context) | set(
        failures.get(PipelineStageType.GRAPH_RESOLUTION.value, [])
    )
    claim_stage = PipelineStageType.CLAIM_EXTRACTION.value
    claims = any(
        result.stage_name == claim_stage
        and result.status == PipelineStageStatus.COMPLETED
        for result in context.stage_results
    )
    return with_record_keys(
        config,
        text_units,
        context.unit_extractions,
        graph_failed=graph_failed,
        claims=claims,
        claims_failed={*graph_failed, *failures.get(claim_stage, [])},
    )


def _distinct_counts(records: list[TextUnitExtraction]) -> tuple[int, int]:
    """Distinct entity and relationship ids over the records."""
    return (
        len({e.id for r in records for e in r.entities}),
        len({rel.id for r in records for rel in r.relationships}),
    )


class GraphExtractionStage(PipelineStage):
    """Extract each text unit's record, reusing the stored ones.

    A unit whose text and extraction settings have a record in the
    extraction store takes it (already gleaned when gleaning is on) instead of
    an LLM call; the others are extracted. Graph resolution stores the new
    records once gleaning has refined them.
    """

    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
        reuse_stored: bool = True,
    ):
        super().__init__(PipelineStageType.GRAPH_EXTRACTION, config, providers)
        self.extractor = GraphExtractor(self.config, providers=self.providers)
        self.store = ExtractionStore(
            self.config, self.providers.build_vector_indexer, reuse=reuse_stored
        )

    def close(self) -> None:
        self.store.close()

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        text_units = context.translated_units or context.text_units
        seed_units = [unit for unit in text_units if is_seed_unit(unit.id)]
        document_units = [unit for unit in text_units if not is_seed_unit(unit.id)]
        gleaned = self.config.processing.gleaning.enabled
        by_unit = self.store.load(
            document_units,
            unit_record_fingerprint(self.config, gleaned=gleaned),
            record_from_payload,
        )
        # A seed unit's record is its parsed text (no model call), stored
        # under the seed fingerprint like any record (graph resolution).
        by_unit.update(
            self.store.load(seed_units, SEED_RECORD_FINGERPRINT, record_from_payload)
        )
        by_unit.update(
            (unit.id, parse_seed_unit(unit))
            for unit in seed_units
            if unit.id not in by_unit
        )
        to_extract = [unit for unit in document_units if unit.id not in by_unit]
        stats = ExtractionStats()
        if to_extract:
            records, stats = self.extractor.extract_units(to_extract)
            by_unit.update((record.text_unit_id, record) for record in records)
        logger.info(
            "Graph extraction: %s text units reused from the extraction store, "
            "%s extracted, %s seed graph units",
            len(document_units) - len(to_extract),
            len(to_extract),
            len(seed_units),
        )
        context.unit_extractions = [
            by_unit[unit.id] for unit in text_units if unit.id in by_unit
        ]
        context.failed_text_unit_ids[self.name] = stats.failed_text_unit_ids
        context.item_failed_text_unit_ids[self.name] = stats.item_failed_text_unit_ids

        entities_count, relationships_count = _distinct_counts(context.unit_extractions)
        seed_entities, seed_relationships = _distinct_counts(
            [by_unit[unit.id] for unit in seed_units]
        )
        metrics = {
            "text_units_processed": len(text_units),
            "text_units_reused": len(document_units) - len(to_extract),
            "entities_extracted": entities_count,
            "relationships_extracted": relationships_count,
            "seed_units": len(seed_units),
            "seed_entities": seed_entities,
            "seed_relationships": seed_relationships,
            "failed_units": stats.num_failed_extractions,
            "extraction_stats": stats.model_dump(),
        }

        return len(text_units), entities_count + relationships_count, metrics


class GleaningStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
    ):
        super().__init__(PipelineStageType.GLEANING, config, providers)
        self.gleaner = GraphGleaner(self.config, providers=self.providers)

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        text_units = context.translated_units or context.text_units
        initial_entities, initial_relationships = _distinct_counts(
            context.unit_extractions
        )

        # A seed unit's record is final: the gleaner gets the document
        # units only and hands the seed records back as they are.
        records, gleaning_stats = self.gleaner.glean(
            [unit for unit in text_units if not is_seed_unit(unit.id)],
            context.unit_extractions,
        )

        context.unit_extractions = records
        context.failed_text_unit_ids[self.name] = gleaning_stats.failed_text_unit_ids
        context.item_failed_text_unit_ids[self.name] = (
            gleaning_stats.item_failed_text_unit_ids
        )
        final_entities, final_relationships = _distinct_counts(records)

        # Graph growth from gleaning: items it added per extracted item.
        initial_items = initial_entities + initial_relationships
        improvement_rate = 0.0
        if initial_items > 0:
            improvement_rate = (
                gleaning_stats.total_entities_added
                + gleaning_stats.total_relationships_added
            ) / initial_items

        metrics = {
            "text_units_processed": len(text_units),
            "initial_entities": initial_entities,
            "final_entities": final_entities,
            "initial_relationships": initial_relationships,
            "final_relationships": final_relationships,
            "entity_improvement": final_entities - initial_entities,
            "relationship_improvement": final_relationships - initial_relationships,
            "improvement_rate": improvement_rate,
            "iterations_completed": gleaning_stats.total_rounds,
            "refinement_calls": gleaning_stats.total_refinement_calls,
            "failed_units": gleaning_stats.num_failed_units,
            "gleaning_stats": gleaning_stats.model_dump(),
        }

        input_count = len(text_units) + initial_items
        output_count = final_entities + final_relationships
        return input_count, output_count, metrics


def _graph_is_corpus(config: Config, context: PipelineContext) -> bool:
    """Whether the run's graph is the whole corpus's, as a full build's:
    without the registry, on a reset, or with no document left unchanged
    (a first run). A reused summary or report answer is then the one a
    full build of the corpus would ask for."""
    delta = context.incremental_delta
    return delta is None or config.indexing.reset or not delta.unchanged


class GraphResolutionStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
        reuse_stored: bool = True,
    ):
        super().__init__(PipelineStageType.GRAPH_RESOLUTION, config, providers)
        self.resolver = GraphResolver(config)
        # Resolution merges descriptions (concatenation); re-summarize the
        # over-long ones with an LLM here (parity with MS/LightRAG).
        self.description_summarizer = DescriptionSummarizer(
            config, providers=self.providers
        )
        self.store = ExtractionStore(self.config, self.providers.build_vector_indexer)
        # A full build reuses the stored summaries of unchanged descriptions
        # (_reuse_stored_summaries), unless it rebuilds everything.
        self.reuse_stored = reuse_stored

    def close(self) -> None:
        self.store.close()

    def _store_records(self, context: PipelineContext) -> set[str]:
        """Store the run's new records: the final ones, not a failed unit's.

        Returns the units whose record was not stored: they are failed (see
        ``extraction_store``), so their documents are extracted again.
        """
        failed = _failed_record_units(context)
        units = {u.id: u for u in context.translated_units or context.text_units}
        fresh = [
            record
            for record in context.unit_extractions
            if not record.stored
            and record.text_unit_id not in failed
            and record.text_unit_id in units
        ]
        unsaved: set[str] = set()
        for gleaned in (False, True):
            unsaved |= self.store.save(
                [
                    (units[record.text_unit_id], record_payload(record))
                    for record in fresh
                    if record.gleaned is gleaned
                    and not is_seed_unit(record.text_unit_id)
                ],
                unit_record_fingerprint(self.config, gleaned=gleaned),
            )
        unsaved |= self.store.save(
            [
                (units[record.text_unit_id], record_payload(record))
                for record in fresh
                if is_seed_unit(record.text_unit_id)
            ],
            SEED_RECORD_FINGERPRINT,
        )
        for record in fresh:
            record.stored = record.text_unit_id not in unsaved
        if unsaved:
            logger.warning(
                "%d text units' extraction records were not stored: recording "
                "them failed so their documents are extracted again",
                len(unsaved),
            )
        return unsaved

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        unsaved = self._store_records(context)
        context.failed_text_unit_ids[self.name] = sorted(unsaved)
        entities, relationships = merge_contributions(context.unit_extractions)
        resolution_result, stats = self.resolver.resolve_graph(entities, relationships)

        original_entities_count = len(entities)
        original_relationships_count = len(relationships)

        resolved_entities = resolution_result["entities"]
        resolved_relationships = resolution_result["relationships"]
        reused_entities, reused_relationships = self._reuse_stored_summaries(
            context, resolved_entities, resolved_relationships
        )
        self.description_summarizer.summarize_entities(
            [e for e in resolved_entities if e.id not in reused_entities]
        )
        self.description_summarizer.summarize_relationships(
            [r for r in resolved_relationships if r.id not in reused_relationships]
        )
        context.resolved_entities = resolved_entities
        context.resolved_relationships = resolved_relationships

        resolved_entities_count = len(context.resolved_entities)
        resolved_relationships_count = len(context.resolved_relationships)

        metrics = {
            "original_entities": original_entities_count,
            "resolved_entities": resolved_entities_count,
            "original_relationships": original_relationships_count,
            "resolved_relationships": resolved_relationships_count,
            "entity_merge_rate": (
                1 - (resolved_entities_count / original_entities_count)
                if original_entities_count
                else 0
            ),
            "relationship_merge_rate": (
                1 - (resolved_relationships_count / original_relationships_count)
                if original_relationships_count
                else 0
            ),
            "resolution_stats": stats.model_dump(),
            "records_not_stored": len(unsaved),
            "summaries_reused": len(reused_entities) + len(reused_relationships),
        }

        input_count = original_entities_count + original_relationships_count
        output_count = resolved_entities_count + resolved_relationships_count
        return input_count, output_count, metrics

    def _reuse_stored_summaries(
        self,
        context: PipelineContext,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> tuple[set[str], set[str]]:
        """On a full build, give the items over the summarization threshold
        the summaries their stored copies were made from the same
        description with (``DescriptionSummarizer.reuse_stored``), in place,
        so a reset of an unchanged corpus asks the model for none; returns
        the ids of the entities and relationships given one.

        An incremental run's items hold its delta's descriptions only (the
        indexing derives the stored ones and reuses their summaries), unless
        no document is left unchanged (``_graph_is_corpus``). One
        read of the graph per suffix; a stored description the graph capped
        (``indexing.neptune.property_max_length``) is not reused, and a
        read that fails reuses nothing (the stores are a cache here).
        """
        if not self.reuse_stored or not _graph_is_corpus(self.config, context):
            return set(), set()
        wanted: dict[str, tuple[list[Entity], list[Relationship]]] = defaultdict(
            lambda: ([], [])
        )
        for entity in entities:
            if needs_summary(self.config, entity.description):
                wanted[BaseIndexer.get_suffix(entity)][0].append(entity)
        for rel in relationships:
            if needs_summary(self.config, rel.description):
                wanted[BaseIndexer.get_suffix(rel)][1].append(rel)
        if not wanted:
            return set(), set()
        limit = self.config.indexing.neptune.property_max_length
        reused_entities: set[str] = set()
        reused_relationships: set[str] = set()
        try:
            graph = self.providers.build_graph_indexer()
            try:
                for suffix, (over_entities, over_rels) in sorted(wanted.items()):
                    stored_entities = graph.read_entities(
                        [e.id for e in over_entities], suffix=suffix
                    )
                    stored_rels = graph.read_relationships(
                        [r.id for r in over_rels], suffix=suffix
                    )
                    reused_entities |= self.description_summarizer.reuse_stored(
                        over_entities,
                        [
                            e
                            for e in stored_entities
                            if len(e.description or "") < limit
                        ],
                    )
                    reused_relationships |= self.description_summarizer.reuse_stored(
                        over_rels,
                        [r for r in stored_rels if len(r.description or "") < limit],
                    )
            finally:
                graph.close()
        except Exception as e:  # noqa: BLE001 - the stored summaries are a cache
            log_error(
                logger,
                logging.WARNING,
                "Reading the stored summaries failed; summarizing every "
                "description over the threshold: %s",
                error=e,
            )
        if reused_entities or reused_relationships:
            logger.info(
                "Reused the stored summaries of %d entities and %d relationships",
                len(reused_entities),
                len(reused_relationships),
            )
        return reused_entities, reused_relationships


class ClaimExtractionStage(PipelineStage):
    """Extract each text unit's claims, reusing the stored ones.

    A unit's prompt lists the entities its own record names, so its claims
    depend on the unit alone and are stored per unit like its record (under
    the record's fingerprint plus the claim settings).
    """

    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
        reuse_stored: bool = True,
    ):
        super().__init__(PipelineStageType.CLAIM_EXTRACTION, config, providers)
        self.extractor = ClaimExtractor(self.config, providers=self.providers)
        self.store = ExtractionStore(
            self.config, self.providers.build_vector_indexer, reuse=reuse_stored
        )

    def close(self) -> None:
        self.store.close()

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        text_units = context.translated_units or context.text_units
        records = {r.text_unit_id: r for r in context.unit_extractions}
        failed = _failed_record_units(context)
        # A seed unit states no claim (its statements are no prose).
        text_units = [unit for unit in text_units if not is_seed_unit(unit.id)]
        # Only a final record's claims are a function of the unit's text and
        # settings, so only those are reused and stored.
        storable = {
            unit.id: unit
            for unit in text_units
            if unit.id in records and unit.id not in failed
        }
        per_unit: dict[str, list[Claim]] = {}
        for gleaned in (False, True):
            units = [u for u in storable.values() if records[u.id].gleaned is gleaned]
            per_unit.update(
                self.store.load(
                    units,
                    unit_record_fingerprint(self.config, gleaned=gleaned, claims=True),
                    claims_from_payload,
                )
            )
        reused = len(per_unit)

        pending = [unit for unit in text_units if unit.id not in per_unit]
        # A claim's subject resolves only against the entities its own unit
        # states (ClaimResolver): a unit that states none has no claim, and
        # is not asked.
        entities_of = {
            unit.id: unit_entities(records[unit.id]) if unit.id in records else []
            for unit in pending
        }
        to_extract = [unit for unit in pending if entities_of[unit.id]]
        stats = ClaimExtractionStats()
        unsaved: set[str] = set()
        extracted: dict[str, list[Claim]] = {}
        if to_extract:
            extracted, stats = self.extractor.extract_units(
                to_extract,
                [entity for unit in to_extract for entity in entities_of[unit.id]],
            )
        answered = {
            **{unit.id: [] for unit in pending if not entities_of[unit.id]},
            **extracted,
        }
        per_unit.update(answered)
        if answered:
            for gleaned in (False, True):
                unsaved |= self.store.save(
                    [
                        (storable[unit_id], claims_payload(claims))
                        for unit_id, claims in answered.items()
                        if unit_id in storable and records[unit_id].gleaned is gleaned
                    ],
                    unit_record_fingerprint(self.config, gleaned=gleaned, claims=True),
                )
            if unsaved:
                # Not stored: failed like an extraction (see extraction_store).
                logger.warning(
                    "%d text units' claim records were not stored: recording "
                    "them failed so their documents are extracted again",
                    len(unsaved),
                )
                stats.failed_text_unit_ids = sorted(
                    {*stats.failed_text_unit_ids, *unsaved}
                )
        # Each unit's statements, in text-unit id order, not merged by the
        # names they state: each resolves against the entities its own unit
        # states (ClaimResolver), as derivation resolves the stored claims'
        # units (``derivation.derive``), and the resolver merges those that
        # resolve to one claim.
        claims = [
            claim.model_copy(deep=True)
            for unit_id in sorted(per_unit)
            for claim in per_unit[unit_id]
        ]
        context.claims = claims
        context.failed_text_unit_ids[self.name] = stats.failed_text_unit_ids
        context.item_failed_text_unit_ids[self.name] = stats.item_failed_text_unit_ids

        metrics = {
            "text_units_processed": len(text_units),
            "text_units_reused": reused,
            "text_units_without_entities": len(answered) - len(extracted),
            "claims_extracted": len(claims),
            "unrecognised_claim_statuses": stats.unrecognised_statuses,
            "failed_units": stats.num_failed_extractions,
            "records_not_stored": len(unsaved),
            "extraction_stats": stats.model_dump(),
        }

        return len(text_units), len(claims), metrics


class ClaimResolutionStage(PipelineStage):
    def __init__(self, config: Config, providers: Providers | None = None):
        super().__init__(PipelineStageType.CLAIM_RESOLUTION, config, providers)
        self.resolver = ClaimResolver(config)

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        if not context.claims or not context.resolved_entities:
            # Nothing can resolve: say so explicitly, so no later stage reads a
            # stale (resumed) result or the unresolved claims.
            context.resolved_claims = []
            logger.warning(
                "Skipping claim-entity resolution due to missing claims or "
                "resolved entities."
            )
            return 0, 0, {"claims_processed": 0, "entities_available": 0}

        entities = context.resolved_entities
        original_claims_count = len(context.claims)
        resolved_claims, stats = self.resolver.resolve(context.claims, entities)
        context.resolved_claims = resolved_claims
        if stats.unresolved_claims:
            logger.warning(
                "Dropped %d of %d claim statements: their subject is not an "
                "entity their own text units state",
                stats.unresolved_claims,
                stats.original_claims,
            )

        resolved_count = len(resolved_claims)
        removed_count = original_claims_count - resolved_count

        metrics = {
            "claims_processed": original_claims_count,
            "claims_resolved": resolved_count,
            "claims_removed": removed_count,
            "claim_merge_rate": (
                1 - (resolved_count / original_claims_count)
                if original_claims_count > 0
                else 0.0
            ),
            "resolution_stats": stats.model_dump(),
        }

        return original_claims_count, resolved_count, metrics


def _nothing_to_index(context: PipelineContext) -> bool:
    """An incremental run without a new or changed document (an unchanged
    corpus, or deletions only): the graph stages receive an empty graph,
    which is no fault, so they do not warn about it."""
    return context.incremental_delta is not None and not context.documents


class GraphAnalysisStage(PipelineStage):
    def __init__(self, config: Config, providers: Providers | None = None):
        super().__init__(PipelineStageType.GRAPH_ANALYSIS, config, providers)
        self.analyzer = GraphAnalyzer(config)

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        entities = context.resolved_entities
        relationships = context.resolved_relationships
        claims = _resolved_claims(context)

        context.knowledge_graph = _build_knowledge_graph(context)
        self.analyzer.graph = context.knowledge_graph

        # Centrality is not computed here: no later stage reads it (the
        # visualization computes its own), and betweenness alone takes
        # minutes on a graph of tens of thousands of entities.
        graph_stats = self.analyzer.get_graph_statistics(
            empty_expected=_nothing_to_index(context)
        )

        metrics = {
            "entities_analyzed": len(entities),
            "relationships_analyzed": len(relationships),
            "claims_analyzed": len(claims) if claims else 0,
            "graph_metrics": {
                "num_nodes": graph_stats.num_nodes,
                "num_edges": graph_stats.num_edges,
                "density": graph_stats.density,
                "num_connected_components": graph_stats.num_connected_components,
                "largest_component_size": graph_stats.largest_component_size,
            },
        }

        input_count = len(entities) + len(relationships)
        output_count = len(entities)
        return input_count, output_count, metrics


# Incident edges read per batch while a delta component's stored
# neighbourhood is walked (see _components_reaching).
_COMPONENT_EDGE_BATCH = 64


def _components_reaching(
    graph: GraphIndexer,
    components: list[set[Any]],
    suffixes: dict[str, str],
    removed: set[str],
    min_size: int,
    delta_neighbors: Mapping[str, Iterable[str]] | None = None,
    delta_entities: Sequence[Entity] = (),
) -> tuple[set[Any], int]:
    """The entities of ``components`` (of the delta graph) whose component of
    the whole graph holds ``min_size`` entities or more, and how many of the
    stored entities the walk reached have no community (a lower bound: the
    walk stops at ``min_size``).

    The whole graph is the stored one, without the relationships that cite
    only ``removed`` text units (the ones a full build no longer extracts),
    joined with the delta graph (``delta_neighbors``: the other delta
    components). A stored entity of another spelling of a delta entity's
    name (its ``compact_name_key``, of the same type) is the same entity
    there: the walk starts from it too. Each component is walked out from,
    a hop at a time, reading the incident edges of a hop's entities a batch
    at a time, and stops once it reaches ``min_size``: at most ``min_size``
    hops, and about ``min_size`` edges past it per hop rather than a hub's
    whole neighbourhood.
    """

    def kept(item: Relationship) -> bool:
        return bool(set(item.text_unit_ids or []) - removed)

    delta_neighbors = delta_neighbors or {}
    suffix_of = {
        str(n): suffixes.get(str(n), Constants.DEFAULT_SUFFIX.value)
        for c in components
        for n in c
    }
    canon = _stored_spellings(graph, components, suffix_of, delta_entities)
    aliases: dict[str, set[str]] = defaultdict(set)
    for stored_id, delta_id in canon.items():
        aliases[delta_id].add(stored_id)
        suffix_of.setdefault(stored_id, suffix_of[delta_id])

    def same(node: str) -> str:
        return canon.get(node, node)

    def nodes_of(ids: Iterable[str]) -> set[str]:
        return {n for i in ids for n in (i, *aliases.get(i, ()))}

    members = [{str(n) for n in c} for c in components]
    frontier = [nodes_of(m) for m in members]
    neighbors: dict[str, set[str]] = {}
    open_ = set(range(len(components)))
    reached: set[int] = set()

    def by_suffix(ids: Iterable[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for node in sorted(ids):
            groups.setdefault(suffix_of[node], []).append(node)
        return groups

    def grown(k: int) -> set[str]:
        return members[k] | {same(b) for n in frontier[k] for b in neighbors.get(n, ())}

    def add(edges: Iterable[Relationship], suffix: str) -> list[tuple[str, str]]:
        """Record the kept ``edges``; returns their (node, neighbour) pairs."""
        added = []
        for edge in edges:
            if not kept(edge):
                continue
            for a, b in (
                (edge.source_id, edge.target_id),
                (edge.target_id, edge.source_id),
            ):
                if a in neighbors:
                    neighbors[a].add(b)
                    suffix_of.setdefault(b, suffix)
                    added.append((a, b))
        return added

    # A hop reads the edges of its entities of at most ``step`` neighbours
    # at once. A hub's are read a batch at a time, and only while one of
    # the components walking through it (its owners, their sizes counted as
    # the batches come in) has not reached min_size: a component that
    # reached it stops the reads, and one that does not walk through the hub
    # never drives them. An owner that cannot reach min_size still reads the
    # hub's whole neighbourhood.
    step = max(min_size, _COMPONENT_EDGE_BATCH)
    while open_:
        hop = set().union(*(frontier[k] for k in open_)) - neighbors.keys()
        for node in hop:
            neighbors[node] = {str(b) for b in delta_neighbors.get(node, ())}
            for b in neighbors[node]:
                suffix_of.setdefault(b, suffix_of[node])
        for suffix, group in by_suffix(hop).items():
            degree = graph.count_neighbors(group, suffix=suffix, at_most=step + 1)
            hubs = sorted(n for n in group if degree.get(n, 0) > step)
            hub_set = set(hubs)
            small = [n for n in group if n not in hub_set]
            ids = (
                graph.find_incident_relationship_ids(small, suffix=suffix)
                if small
                else []
            )
            for start in range(0, len(ids), step):
                add(
                    graph.read_relationships(ids[start : start + step], suffix=suffix),
                    suffix,
                )
            for hub in hubs:
                owners = [k for k in sorted(open_) if hub in frontier[k]]
                seen = {k: grown(k) for k in owners}
                ids = graph.find_incident_relationship_ids([hub], suffix=suffix)
                for start in range(0, len(ids), step):
                    if all(len(seen[k]) >= min_size for k in owners):
                        break
                    edges = graph.read_relationships(
                        ids[start : start + step], suffix=suffix
                    )
                    for a, b in add(edges, suffix):
                        for k in owners:
                            if a in frontier[k]:
                                seen[k].add(same(b))
        for k in sorted(open_):
            new = grown(k) - members[k]
            members[k] |= new
            frontier[k] = nodes_of(new)
            if len(members[k]) >= min_size:
                reached.add(k)
            if len(members[k]) >= min_size or not new:
                open_.discard(k)
    found = {n for k in reached for n in components[k]}
    linked = {n for k in reached for n in members[k]} - {str(n) for n in found}
    unclustered = sum(
        1
        for suffix, group in by_suffix(linked).items()
        for entity in graph.read_entities(group, suffix=suffix)
        if not entity.community_ids
    )
    return found, unclustered


class _StoredMembers(NamedTuple):
    """What the stores hold of an incremental run's report members (see
    ``CommunityDetectionStage._evidenced_by_stored_members``)."""

    # The communities a stored member evidences; None when the read failed.
    evidenced: set[str] | None
    # The document-backed stored members by id, each with the description
    # its report reads before the run's lines.
    backed: dict[str, Entity]
    # The stored copies of the members the run states with neither a type
    # nor a description, by id.
    unattributed: dict[str, Entity]


def _in_suffix(entity: Entity, suffix: str) -> Entity:
    """``entity`` (a copy) tagged with the index ``suffix`` it was read
    from, as the indexing groups items."""
    entity.attributes = {**(entity.attributes or {}), Constants.INDEX.value: suffix}
    return entity


def _document_backed(entity: Entity, removed: set[str]) -> bool:
    """Whether the stored ``entity`` is stated by a document a full build
    keeps: it cites a document unit not ``removed``, or, stored without
    lineage, its ``source_kinds`` is absent or names documents."""
    units = entity.text_unit_ids or []
    if units:
        return any(not is_seed_unit(u) and u not in removed for u in units)
    return not entity.source_kinds or DOCUMENT_SOURCE_KIND in entity.source_kinds


def _stored_spellings(
    graph: GraphIndexer,
    components: list[set[Any]],
    suffix_of: Mapping[str, str],
    delta_entities: Sequence[Entity],
) -> dict[str, str]:
    """``{stored id: delta id}`` of the stored entities of another id whose
    name has a component entity's ``compact_name_key`` and whose type is
    its (a full build merges the two)."""
    in_components = {str(n) for c in components for n in c}
    keyed: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    for entity in delta_entities:
        if entity.id in in_components:
            keyed[(suffix_of[entity.id], compact_name_key(entity.name))].append(entity)
    found: dict[str, str] = {}
    by_suffix: dict[str, set[str]] = defaultdict(set)
    for suffix, key in keyed:
        by_suffix[suffix].add(key)
    for suffix, keys in sorted(by_suffix.items()):
        ids = graph.find_entity_ids_by_name_key(sorted(keys), suffix) or []
        ids = [i for i in ids if i not in in_components]
        for stored in graph.read_entities(ids, suffix=suffix) if ids else []:
            for entity in keyed.get((suffix, compact_name_key(stored.name)), ()):
                if normalize_entity_type(entity.type) == normalize_entity_type(
                    stored.type
                ):
                    found.setdefault(stored.id, entity.id)
    return found


def _neighbor_ids(graph: Any) -> dict[str, set[str]]:
    """``{node: its neighbours}`` of the run's delta graph (networkx)."""
    if graph is None:
        return {}
    return {str(n): {str(m) for m in graph.neighbors(n)} for n in graph.nodes}


class CommunityDetectionStage(_RegistryStage):
    # Stored entities without a community linked to the small delta
    # components the last run clustered (``_whole_graph``).
    unclustered_stored = 0

    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
        doc_status: "DocStatusPort | None" = None,
        reuse_stored: bool = True,
    ):
        super().__init__(PipelineStageType.COMMUNITY_DETECTION, config, providers)
        self.detector = CommunityDetector(config, providers=self.providers)
        self._doc_status = doc_status
        self._description_summarizer: DescriptionSummarizer | None = None
        # A full build keeps the report answers in the extraction store (by
        # their input) and reuses an unchanged input's, unless it rebuilds
        # everything (--force-rebuild: it asks again and refreshes them).
        self.store = ExtractionStore(
            self.config, self.providers.build_vector_indexer, reuse=reuse_stored
        )

    def close(self) -> None:
        self.store.close()

    def _check_report_failures(
        self, communities: list[Community], failed_community_ids: list[str]
    ) -> list[str]:
        """Text units of the base-level communities left without a report.

        Their documents are recorded FAILED, so the next incremental run
        re-processes them and generates the missing reports, the same way an
        extraction failure is retried. A failed parent-level report is only
        logged (and counted in the stage metrics): a parent spans many
        base-level communities, so recording its text units would re-extract
        most of the corpus on the next run for one report. More failures
        than ``indexing.max_failure_rate`` of the communities fail the stage
        (the reports are the community-report index's items). Seed units are
        left out: a failed report never imports a seed graph's shard again
        (its statements need no model call).
        """
        if not failed_community_ids:
            return []
        max_failure_rate = self.config.indexing.max_failure_rate
        failure_rate = len(failed_community_ids) / max(len(communities), 1)
        if failure_rate > max_failure_rate:
            raise PipelineStageError(
                f"Community report generation failed for "
                f"{len(failed_community_ids)} of {len(communities)} communities "
                f"({failure_rate:.0%} > {max_failure_rate:.0%} tolerated by "
                f"indexing.max_failure_rate). Check the report model errors above.",
                own_text=True,
            )
        failed_ids = set(failed_community_ids)
        failed = [c for c in communities if c.id in failed_ids]
        parents = [c.id for c in failed if str(c.level) != "0"]
        if parents:
            logger.warning(
                "%s parent-level community report(s) failed and are missing "
                "from the index until the communities are rebuilt: %s",
                len(parents),
                ", ".join(parents),
            )
        return sorted(
            {
                unit_id
                for community in failed
                if str(community.level) == "0"
                for unit_id in community.text_unit_ids or []
                if not is_seed_unit(unit_id)
            }
        )

    def _allows_empty_output(self, context: PipelineContext) -> bool:
        """Also when every entity is in a component too small to cluster.

        Such entities get no community (``CommunityDetector``), so a corpus,
        or a delta, of only isolated entities and pairs has none to report;
        entities without a single community mean nothing else.
        """
        return super()._allows_empty_output(context) or (
            bool(context.resolved_entities) and not context.communities
        )

    def _whole_graph(
        self, context: PipelineContext
    ) -> Callable[[list[set[Any]]], set[Any]] | None:
        """On an incremental run, the entities of the delta's small
        components that reach ``min_community_size`` entities in the whole
        graph (``CommunityDetector.__call__``).

        The whole graph is the one a full build of the corpus would cluster:
        the stored graph without what this run replaces or removes (the
        text units of the documents it processes, deletes or quarantines).
        A read failure keeps the components left out, as a full build of a
        smaller graph would.
        """
        delta = context.incremental_delta
        if delta is None or not delta.unchanged:
            return None
        suffixes = {e.id: BaseIndexer.get_suffix(e) for e in context.resolved_entities}
        min_size = self.config.graph.community_detection.min_community_size

        def large(components: list[set[Any]]) -> set[Any]:
            try:
                removed = self._replaced_unit_ids(context, delta)
                graph = self.providers.build_graph_indexer()
                try:
                    found, unclustered = _components_reaching(
                        graph,
                        components,
                        suffixes,
                        removed,
                        min_size,
                        delta_neighbors=_neighbor_ids(context.knowledge_graph),
                        delta_entities=context.resolved_entities,
                    )
                finally:
                    graph.close()
            except Exception as e:  # noqa: BLE001 - left unclustered
                log_error(
                    logger,
                    logging.WARNING,
                    "Could not read the stored graph around %s small delta "
                    "components; leaving them out of the communities: %s",
                    len(components),
                    error=e,
                )
                return set()
            if unclustered:
                # Reported, not a rebuild signal (community_drift's
                # rebuild_recommended does not count them).
                logger.info(
                    "%s stored entities without a community are linked to "
                    "small delta components this run clusters: a full build "
                    "clusters them with those, an incremental run clusters "
                    "only its delta (community_drift.unclustered_entities; "
                    "indexing.reset: true clusters them).",
                    unclustered,
                )
            self.unclustered_stored = unclustered
            return found

        return large

    def _replaced_unit_ids(
        self, context: PipelineContext, delta: DocumentDelta
    ) -> set[str]:
        """The stored text units a full build of the corpus no longer
        holds: this run's units (of the documents it processes) and the
        recorded units of the documents it processes, deletes or
        quarantines."""
        records = self._build_doc_status_store().get_many(
            [*delta.to_process, *delta.deleted, *delta.quarantined]
        )
        return {u.id for u in context.text_units or context.translated_units}.union(
            *(r.text_unit_ids for r in records.values())
        )

    def _evidenced_by_stored_members(
        self, context: PipelineContext, communities: list[Community]
    ) -> _StoredMembers:
        """The ids of ``communities`` (every member seed-only in this run's
        graph) that a stored member makes document-evidenced, on an
        incremental run, and the stored copies of those members by id: a
        full build's report reads their descriptions merged with the seed's
        lines (``generate_reports(descriptions=)``). A member citing a unit
        the run replaces (a seed's earlier version) has its description
        derived again from the records of its other units first
        (``stripped_descriptions``, as the indexing's strip does): the
        replaced unit's lines are not the corpus's. Such a derived
        description over the summarization threshold is summarized, as the
        strip's indexing summarizes it (``_summarize_members``).

        An incremental run's graph holds its delta only: a seed added or
        edited beside unchanged documents states their entities with seed
        units alone, while the stored entities still cite the documents'. A
        stored member is evidenced when it cites a document unit the run
        does not replace (``_replaced_unit_ids``), or, stored without
        lineage, when its ``source_kinds`` is absent (written before seed
        graphs, or document-only) or names documents. One batched read of
        the members per suffix (and of the records of the stripped ones'
        units); a failure leaves ``evidenced`` ``None``: the communities
        are stored without a report and count as failed ones.

        A member the run states with neither a type nor a description (the
        endpoint of a seed edge whose own node is in a shard the run keeps)
        is in ``unattributed`` with its stored copy, its description derived
        alike: the report reads the stored type and description, as a full
        build reads the node's.
        """
        delta = context.incremental_delta
        if not communities or delta is None or self.config.indexing.reset:
            return _StoredMembers(set(), {}, {})
        members = {str(e) for c in communities for e in c.entity_ids or []}
        groups: dict[str, list[str]] = defaultdict(list)
        for entity in context.resolved_entities:
            if entity.id in members:
                groups[BaseIndexer.get_suffix(entity)].append(entity.id)
        bare = {
            e.id for e in context.resolved_entities if not (e.type or e.description)
        }
        backed: dict[str, Entity] = {}
        unattributed: dict[str, Entity] = {}
        stripped: list[Entity] = []
        try:
            removed = self._replaced_unit_ids(context, delta)
            graph = self.providers.build_graph_indexer()
            try:
                read = {
                    suffix: graph.read_entities(sorted(ids), suffix=suffix)
                    for suffix, ids in sorted(groups.items())
                }
            finally:
                graph.close()
            stored = {
                suffix: [
                    e for e in entities if _document_backed(e, removed) or e.id in bare
                ]
                for suffix, entities in read.items()
            }
            # The lines of the units this run replaces (a seed's earlier
            # version) are not the corpus's: their descriptions are derived
            # again without them, as the indexing's strip does before it
            # merges this run's lines in.
            stale = any(
                removed.intersection(entity.text_unit_ids or [])
                for entities in stored.values()
                for entity in entities
            )
            vector = self.providers.build_vector_indexer() if stale else None
            try:
                for suffix, entities in stored.items():
                    descriptions = (
                        stripped_descriptions(
                            self.config, vector, suffix, entities, removed
                        )
                        if vector is not None
                        else {}
                    )
                    for entity in entities:
                        copy = _in_suffix(
                            entity.model_copy(
                                update={
                                    "description": descriptions.get(
                                        entity.id, entity.description
                                    )
                                }
                            ),
                            suffix,
                        )
                        if _document_backed(entity, removed):
                            backed[entity.id] = copy
                        if entity.id in bare:
                            unattributed[entity.id] = copy
                        if removed.intersection(entity.text_unit_ids or []):
                            stripped.append(copy)
            finally:
                if vector is not None:
                    vector.close()
        except Exception as e:  # noqa: BLE001 - counted in reports_failed
            log_error(
                logger,
                logging.WARNING,
                "Could not read the stored members of %s communities only seed "
                "graphs state in this run; storing them without a report, "
                "counted in reports_failed (indexing.reset: true reports "
                "them): %s",
                len(communities),
                error=e,
            )
            return _StoredMembers(None, {}, {})
        self._summarize_members(context, stripped)
        evidenced = {
            c.id
            for c in communities
            if backed.keys() & set(map(str, c.entity_ids or []))
        }
        return _StoredMembers(evidenced, backed, unattributed)

    @property
    def description_summarizer(self) -> DescriptionSummarizer:
        """The summarizer of the stored members' descriptions, built on
        first use (most runs summarize none here)."""
        if self._description_summarizer is None:
            self._description_summarizer = DescriptionSummarizer(
                self.config, providers=self.providers
            )
        return self._description_summarizer

    def _summarize_members(
        self, context: PipelineContext, members: list[Entity]
    ) -> None:
        """Summarize the descriptions of ``members`` (copies of the run's
        report members) over the summarization threshold, in place, as the
        indexing summarizes the descriptions it derives (``needs_summary``),
        and keep each summary in ``context.member_summaries``: marked with
        the description it was made from (``summarized_from``), it is the
        one the indexing reuses (``reuse_summaries``) when it derives that
        description again, so the run asks the model once. A failed summary
        leaves the description as derived (the indexing tries again)."""
        over = [m for m in members if needs_summary(self.config, m.description)]
        if not over:
            return
        derived = [m.description for m in over]
        self.description_summarizer.summarize_entities(over)
        context.member_summaries.extend(
            m for m, d in zip(over, derived, strict=True) if m.description != d
        )

    def _report_descriptions(
        self,
        context: PipelineContext,
        stored: _StoredMembers,
    ) -> dict[str, str]:
        """The description each report reads of a member the stores hold
        (see :meth:`_evidenced_by_stored_members`), by id: a stored
        member's description followed by the run's lines, as the full
        build's merge orders them (documents first) and the indexing merges
        them, summarized when over the threshold as the indexing summarizes
        the merge (``_summarize_members``); and an unattributed member's
        stored description."""
        merged: list[Entity] = []
        for entity in context.resolved_entities:
            copy = stored.backed.get(entity.id)
            if copy is None:
                continue
            merged.append(
                copy.model_copy(
                    update={
                        "description": merge_descriptions(
                            (copy.description, entity.description)
                        )
                    }
                )
            )
        self._summarize_members(context, merged)
        descriptions = {
            entity_id: entity.description
            for entity_id, entity in stored.unattributed.items()
            if entity.description and entity_id not in stored.backed
        }
        descriptions.update({m.id: m.description for m in merged if m.description})
        return descriptions

    @staticmethod
    def _get_detection_stats_dict(metrics_obj: CommunityMetrics) -> dict[str, Any]:
        return {
            "num_communities": metrics_obj.num_communities,
            "average_community_size": metrics_obj.average_community_size,
            "largest_community_size": metrics_obj.largest_community_size,
            "smallest_community_size": metrics_obj.smallest_community_size,
            "community_size_distribution": metrics_obj.community_size_distribution,
        }

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        entities = context.resolved_entities
        relationships = context.resolved_relationships

        if context.knowledge_graph is None:
            # A run resumed at this stage restores graph_analysis's cached
            # entities and relationships, not the graph built from them.
            logger.info(
                "Rebuilding the knowledge graph from %s resolved entities and %s "
                "relationships (resumed run)",
                len(entities),
                len(relationships),
            )
            context.knowledge_graph = _build_knowledge_graph(context)

        self.unclustered_stored = 0
        self.detector(
            context.knowledge_graph,
            self._whole_graph(context),
            empty_expected=_nothing_to_index(context),
        )
        community_objects = self.detector.generate_community_objects()
        # A community no document evidences (every member stated by a seed
        # graph only) is stored without a report: the corpus says nothing
        # about it, and global search answers from the corpus. A parent
        # holds its children's members, so one evidenced child evidences it.
        # On an incremental run the stored members count too.
        looks_seed_only = [c for c in community_objects if seed_only(c.text_unit_ids)]
        context.member_summaries = []
        stored = self._evidenced_by_stored_members(context, looks_seed_only)
        # Communities whose stored members could not be read: without a
        # report, and failed rather than seed-only (the corpus may evidence
        # them).
        unread = [c.id for c in looks_seed_only] if stored.evidenced is None else []
        stored_evidence = stored.evidenced or set()
        evidenced = [
            c
            for c in community_objects
            if not seed_only(c.text_unit_ids) or c.id in stored_evidence
        ]
        seed_only_count = len(community_objects) - len(evidenced) - len(unread)
        if seed_only_count:
            logger.info(
                "%s of %s communities have members stated only by seed graphs: "
                "stored without a report",
                seed_only_count,
                len(community_objects),
            )
        # Text units resolve each report's source documents (report lineage).
        # A stored member's description precedes the seed's lines, as in the
        # full build's merge (documents first).
        community_reports, failed_community_ids = self.detector.generate_reports(
            evidenced,
            text_units=context.text_units or context.translated_units,
            descriptions=self._report_descriptions(context, stored),
            types={
                entity_id: entity.type
                for entity_id, entity in stored.unattributed.items()
                if entity.type
            },
            # A full build's reports: an incremental run reports the delta's
            # communities, which a later full build clusters otherwise.
            records=self.store if _graph_is_corpus(self.config, context) else None,
            empty_expected=_nothing_to_index(context),
        )
        metrics_obj = self.detector.get_community_metrics()

        context.communities = community_objects
        context.community_reports = community_reports
        context.failed_text_unit_ids[self.name] = self._check_report_failures(
            evidenced, failed_community_ids
        )

        communities_count = len(context.communities)
        reports_count = len(context.community_reports)
        metrics = {
            "entities_processed": len(entities),
            "relationships_processed": len(relationships),
            "communities_detected": communities_count,
            "reports_generated": reports_count,
            "reports_failed": len(failed_community_ids) + len(unread),
            "communities_seed_only": seed_only_count,
            "unclustered_stored_entities": self.unclustered_stored,
            "modularity_score": metrics_obj.modularity if metrics_obj else 0.0,
            "detection_stats": (
                self._get_detection_stats_dict(metrics_obj) if metrics_obj else {}
            ),
        }

        input_count = len(entities) + len(relationships)
        output_count = communities_count + reports_count
        return input_count, output_count, metrics


class IndexingStage(_RegistryStage):
    def __init__(
        self,
        config: Config,
        doc_status: "DocStatusPort | None" = None,
        providers: Providers | None = None,
        reuse_stored: bool = True,
    ):
        super().__init__(PipelineStageType.INDEXING, config, providers)
        self.indexing_manager = IndexingManager(
            config=self.config, providers=self.providers
        )
        # A text embedded unchanged takes its stored vector (the indexer's
        # default), unless the run rebuilds everything (--force-rebuild).
        if not reuse_stored:
            self.indexing_manager.vector_indexer.set_embedding_reuse(False)
        # Injected by the pipeline for the incremental commit/registry write-back.
        self._doc_status = doc_status

    def close(self) -> None:
        """Close the indexers' Neptune/OpenSearch clients (best-effort)."""
        self.indexing_manager.close()

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        if self.config.indexing.reset:
            check_write_fence()
            # The stores are not cleared first: the full build below writes
            # new vector indices and swaps their aliases, and rewrites the
            # graph in place before dropping what it did not write
            # (GraphIndexer.begin_rebuild), so the suffix is never empty
            # meanwhile (queries read rewritten items next to the previous
            # ones not reached yet). The registry keeps its records until
            # the rebuild recorded the corpus (below).

        self.indexing_manager.initialize()
        self.indexing_manager.underived_items = 0
        self.indexing_manager.annotations_rewritten = 0
        # A pipeline can be run again (library use): nothing an earlier run's
        # derivation decided carries over (a failed commit's re-keys would
        # delete old ids whose replacements were never written).
        self.indexing_manager.reset_run_state()
        # The summaries community detection made of the stored members'
        # descriptions: the indexing derives those descriptions again.
        self.indexing_manager.offer_summaries(context.member_summaries)

        text_units = _with_record_keys(
            self.config, context, context.translated_units or context.text_units
        )
        entities = context.resolved_entities
        relationships = context.resolved_relationships
        communities = context.communities or []
        community_reports = context.community_reports or []
        # Each entity records its level-0 community, so a later delta run
        # can tell which of its entities an earlier clustering placed.
        entities = with_level0_community_ids(entities, communities)
        claims = _resolved_claims(context)
        if context.claims and not claims:
            logger.info(
                "Indexing no claims: none of the %s extracted claims resolved to "
                "an entity (or claim resolution did not run)",
                len(context.claims),
            )

        input_count = (
            len(text_units)
            + len(entities)
            + len(relationships)
            + len(communities)
            + len(community_reports)
            + len(claims)
        )

        if (
            context.incremental_delta is None
            and not self.config.indexing.reset
            and self._registry_on
        ):
            # Incremental indexing is on but no delta reached this stage (the
            # loading stage failed under continue_on_error, or the run was
            # resumed from metadata written with the registry off). The full
            # path would replace the live index content with this run's
            # documents only, so stop instead.
            raise PipelineStageError(
                "Incremental indexing is enabled but no document delta was "
                "computed for this run, so the indexing stage cannot tell "
                "which stored documents to keep. Re-run from the "
                "document_loading stage (--resume-from-stage document_loading) "
                "once the doc-status registry is reachable, or set "
                "indexing.reset: true to rebuild the stores.",
                own_text=True,
            )

        drift = None
        documents_deleted = 0
        # The build record, written before the registry records the run's
        # documents when the registry is on (see IncrementalIndexer.record):
        # a run killed after the registry write leaves it.
        counters: dict[str, CommunityCounters] | None = None
        # What became of the record (the run result's ``build_record``).
        record_state = "failed"

        def record_build(results: Mapping[str, IndexingStats]) -> None:
            nonlocal counters, record_state
            write = self._build_record_write(context, results)
            if write is None:
                return
            try:
                counters = self._record_build(
                    context,
                    text_units,
                    communities,
                    tree_pruned=_tree_pruned(results),
                    rewrite=write == "rewrite",
                )
                if counters is not None:
                    record_state = "written"
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - written again after the registry
                log_error(
                    logger,
                    logging.WARNING,
                    "Writing the build record failed (%s); retried once the "
                    "documents are recorded",
                    error=e,
                )
            if counters is None:
                logger.warning(
                    "The build record was not written on every index; retried "
                    "once the documents are recorded"
                )

        if context.incremental_delta is not None and not self.config.indexing.reset:
            warn_about_unapplied_retired_scopes(self.config, context)
            # Applied once the stage completes: a failed removal fails it.
            documents_deleted = len(context.incremental_delta.deleted)
            indexing_results, drift = self._index_incremental(
                context,
                text_units,
                entities,
                relationships,
                communities,
                community_reports,
                claims,
                record_build,
            )
        else:
            rebuild = None
            if context.incremental_delta is not None:
                # Reset with the registry enabled: record the rebuilt corpus
                # for the next delta run, PENDING first (see
                # IncrementalIndexer.write_ahead) so an interrupted rebuild
                # leaves nothing unreferenced.
                incremental = run_incremental_indexer(
                    self._build_doc_status_store(),
                    self.indexing_manager,
                    context,
                    text_units,
                )
                lineages = context_lineages(context, incremental.suffix, text_units)
                # And the documents it drops: once it swaps an index in, an
                # interrupted rebuild left their records PROCESSED over stores
                # that no longer hold them (diffed unchanged, never indexed
                # again). Not a document of the corpus (quarantined ones
                # included): write_ahead takes them as listed.
                kept = set(context.incremental_fingerprints or ())
                dropped = [
                    record
                    for namespace in sorted(
                        reset_namespaces(self.config, context, text_units)
                    )
                    for record in incremental.doc_status.list_namespace(namespace)
                    if record.doc_id not in kept
                    and record.status is DocStatus.PROCESSED
                ]
                incremental.write_ahead(
                    context.incremental_delta, lineages, dropped=dropped
                )
                # Until the rebuild wrote every item, the stores may hold
                # the annotations of two schemas (annotations_due).
                record_rebuilt_annotations(
                    incremental.doc_status,
                    self.indexing_manager,
                    reset_namespaces(self.config, context, text_units),
                    rebuilt=False,
                )
                rebuild = (incremental, lineages)
            # A build the registry does not record must not be updated
            # incrementally later (see VectorIndexer.check_index_settings).
            self.indexing_manager.vector_indexer.set_doc_status_tracking(
                rebuild is not None
            )
            indexing_results = self.indexing_manager.index_all_data(
                text_units=text_units,
                entities=entities,
                relationships=relationships,
                communities=communities,
                community_reports=community_reports,
                claims=claims,
            )
            if rebuild is not None:
                incremental, lineages = rebuild
                failures = run_failures(context, text_units)
                written = [
                    *text_units,
                    *entities,
                    *relationships,
                    *communities,
                    *community_reports,
                    *claims,
                ]
                recorded = incremental.record(
                    lineages,
                    context.incremental_fingerprints,
                    indexing_results,
                    failures.failed,
                    artifact_owners(lineages, written),
                    written=written,
                    counted_failed_doc_ids=failures.counted,
                    claims_missing_doc_ids=failures.claims_missing,
                    before_registry=record_build,
                )
                context.record_failed_documents(incremental.recorded_failed_doc_ids)
                context.record_retrying_documents(incremental.retrying_doc_ids)
                context.record_claims_missing_documents(
                    incremental.claims_missing_doc_ids
                )
                if recorded:
                    # The records of documents the rebuild replaced go; a
                    # failed rebuild keeps them (their artifacts may remain).
                    clear_registry_namespaces(
                        self._build_doc_status_store,
                        reset_namespaces(self.config, context, text_units),
                        keep=context.incremental_fingerprints,
                    )
                if recorded and not any(
                    stats.failed_items for stats in indexing_results.values()
                ):
                    # Every stored item was written by the rebuild (a failed
                    # graph write keeps the earlier graph's items).
                    record_rebuilt_annotations(
                        incremental.doc_status,
                        self.indexing_manager,
                        reset_namespaces(self.config, context, text_units),
                        rebuilt=True,
                    )

        total_indexed = sum(
            stats.successful_items for stats in indexing_results.values()
        )
        total_failed = sum(stats.failed_items for stats in indexing_results.values())
        # Relationships indexed (graph backend), surfaced as a top-level metric so
        # the silent-drop failure mode (relationships extracted but 0 indexed) is
        # observable/alarmable rather than hidden inside per-backend results.
        relationships_indexed = sum(
            stats.successful_items
            for key, stats in indexing_results.items()
            if "relationship" in key
        )

        # Validate that no backend has completely failed
        self._validate_backend_success(indexing_results)
        delta = context.incremental_delta
        if counters is None:
            write = self._build_record_write(context, indexing_results)
            if write is not None:
                counters = self._record_build(
                    context,
                    text_units,
                    communities,
                    tree_pruned=_tree_pruned(indexing_results),
                    rewrite=write == "rewrite",
                )
                if counters is not None:
                    record_state = "written"
                else:
                    logger.warning(
                        "The build record was not written on every index: until "
                        "a run writes it, queries read the namespace as holding "
                        "what this config builds (a strategy whose methodology "
                        "it does not hold answers from what it finds instead of "
                        "failing). The next run writes it again."
                    )
            else:
                self._record_removed_artifacts(text_units, delta)
                record_state = self._held_record_state()

        metrics = {
            "indexing_results": {k: v.to_dict() for k, v in indexing_results.items()},
            "total_indexed": total_indexed,
            "total_failed": total_failed,
            "relationships_indexed": relationships_indexed,
            # Stored items an incremental run could not derive (a text unit
            # they cite names no stored record): stale until a rebuild.
            "underived_items": self.indexing_manager.underived_items,
            # Stored items whose schema annotations a schema change rewrote.
            "annotations_rewritten": self.indexing_manager.annotations_rewritten,
            # Documents gone from the corpus (or of a retired scope) whose
            # exclusive content and records this incremental run removed.
            "documents_deleted": documents_deleted,
            # The suffix's build record after the run: "written" by it,
            # "kept" (it wrote nothing; the stored one stands), "failed" (no
            # index took it), "missing" (none stored, none written), or
            # "unsupported" (a vector store without index metadata).
            "build_record": (
                record_state
                if keeps_build_records(self.indexing_manager.vector_indexer)
                else "unsupported"
            ),
            "success_rate": (
                total_indexed / (total_indexed + total_failed)
                if (total_indexed + total_failed) > 0
                else 0
            ),
        }
        if drift is not None:
            metrics["community_drift"] = drift.model_dump()
            metrics["community_drift_overlap_share"] = drift.overlap_share
            metrics["community_rebuild_recommended"] = int(drift.rebuild_recommended)
        if counters:
            metrics.update(self._check_community_counters(counters, drift))

        return input_count, total_indexed, metrics

    def _check_community_counters(
        self, counters: dict[str, CommunityCounters], drift: CommunityDrift | None
    ) -> dict[str, Any]:
        """Recommend a rebuild (one WARNING per suffix) where the community
        counters crossed an ``indexing.community_compaction`` threshold; the
        counters' metrics (the largest over the suffixes)."""
        compaction = self.config.indexing.community_compaction
        for suffix, counted in sorted(counters.items()):
            reasons = reset_reasons(
                counted,
                compaction,
                drift_recommended=drift is not None and drift.rebuild_recommended,
            )
            if reasons:
                logger.warning(
                    "Communities of '%s' are due for re-clustering: %s. "
                    "Incremental runs append their delta's communities "
                    "instead of re-clustering the graph; run once with "
                    "indexing.reset: true, which reuses the unchanged "
                    "summaries and reports.",
                    suffix,
                    "; ".join(reasons),
                )
        return {
            "community_counters": {
                suffix: counted.model_dump(mode="json")
                for suffix, counted in sorted(counters.items())
            },
            "community_growth_ratio": max(
                (c.growth_ratio or 0.0 for c in counters.values()), default=0.0
            ),
            "runs_since_clustering": max(
                (c.runs_since for c in counters.values()), default=0
            ),
        }

    def _build_record_write(
        self, context: PipelineContext, results: Mapping[str, IndexingStats]
    ) -> str | None:
        """Whether the run writes the build record: ``"run"`` unless it is
        an incremental run that found nothing to index and wrote no item
        (the indices keep the record of the run that did); then
        ``"rewrite"`` when the registry holds documents of the configured
        suffix and its indices hold no record (a write the store rejected,
        or a build from before records), else ``None``."""
        delta = context.incremental_delta
        if (
            delta is None
            or self.config.indexing.reset
            or not delta.is_empty
            or any(s.total_items or s.failed_items for s in results.values())
        ):
            return "run"
        vector_indexer = self.indexing_manager.vector_indexer
        if (
            delta.unchanged
            and keeps_build_records(vector_indexer)
            and vector_indexer.read_build_record(resolve_namespace(self.config, None))
            is None
        ):
            return "rewrite"
        return None

    def _held_record_state(self) -> str:
        """The build record of a run that wrote none: ``"kept"`` when the
        configured suffix holds one, else ``"missing"``."""
        vector_indexer = self.indexing_manager.vector_indexer
        if not keeps_build_records(vector_indexer):
            return "missing"
        held = vector_indexer.read_build_record(resolve_namespace(self.config, None))
        return "kept" if held is not None else "missing"

    def _artifacts_held(
        self, suffix: str
    ) -> tuple[frozenset[IndexArtifact], dict[str, int | None]] | None:
        """The artifacts of this config the stores of ``suffix`` show they
        hold (``stored_artifacts_shown``) and the documents of each vector
        index kind, ``None`` when the vector store cannot describe its
        indices."""
        vector_indexer = self.indexing_manager.vector_indexer
        described = vector_indexer.describe_indices(suffix)
        if described is None:
            return None
        documents = {entry["kind"]: entry.get("documents") for entry in described}
        graph = self.indexing_manager.graph_indexer
        return (
            stored_artifacts_shown(
                self.config,
                documents,
                graph_holds=graph.holds_any_item(suffix),
                graph_edges=graph.holds_any_relationship(suffix),
                tree=bool(vector_indexer.read_summary_tree_record(suffix)),
            ),
            documents,
        )

    def _record_build(
        self,
        context: PipelineContext,
        text_units: list[TextUnit],
        communities: Sequence[Community] = (),
        tree_pruned: bool = True,
        rewrite: bool = False,
    ) -> dict[str, CommunityCounters] | None:
        """Record on each written suffix's vector indices what built them.

        The summary tree is built after this record, by the summary_tree
        stage. A suffix holding a live tree keeps naming it when ``raptor``
        is selected and the run's deletion of the summaries citing removed
        text succeeded (``tree_pruned``): the tree then cites only stored
        text, so queries read it while the stage rebuilds it (and after a
        crash before it did). The stage drops it when its build fails.

        ``run-eval`` copies the record into its run manifest, so a score can be
        traced to the ingestion run, its extraction settings and its corpus
        size. Written after the backends succeeded; best-effort. A full
        build starts each suffix's community counters with the
        ``communities`` it wrote; an incremental record adds the ones it
        appended and subtracts those its removals deleted, once per commit
        (``IndexBuildRecord.over``). Returns the counters written, by suffix
        (none where the stored record predates them), or ``None`` when a
        suffix's record was written on no index (the store rejected it).

        A run that ``rewrite``s a missing record indexed nothing: the record
        stands for the build the stores hold, a full one (later incremental
        records carry it as their ``full_build``). It names the artifacts
        the stores show they hold (``_artifacts_held``; none is written when
        a read of them fails), the documents the registry records, and
        community counters started from the reports stored, not what the
        config would build.
        """
        delta = context.incremental_delta
        incremental = is_incremental_build(context, self.config)
        suffixes = {BaseIndexer.get_suffix(unit) for unit in text_units} or {
            resolve_namespace(self.config, None)
        }
        documents = {
            doc_id for unit in text_units for doc_id in unit.document_ids or []
        }
        artifacts = built_artifacts(self.config, _completed_stages(context))
        added = Counter(BaseIndexer.get_suffix(c) for c in communities)
        rewritten = rewrite and delta is not None
        if rewritten and delta is not None:
            suffix = resolve_namespace(self.config, None)
            try:
                held = self._artifacts_held(suffix)
            except Exception as e:  # noqa: BLE001 - best-effort, as the write
                log_error(
                    logger,
                    logging.WARNING,
                    "Not rewriting the missing build record of '%s': its "
                    "indices could not be described (%s)",
                    suffix,
                    error=e,
                )
                return None
            incremental = False
            documents = set(delta.unchanged)
            if held is not None:
                artifacts, stored = held
                reports = stored.get("community_reports")
                if IndexArtifact.COMMUNITIES in artifacts and reports:
                    added = Counter({suffix: reports})
        settings = context.incremental_extraction_settings or extraction_settings(
            self.config
        )
        record = IndexBuildRecord(
            pipeline_id=context.pipeline_id,
            mode="incremental" if incremental else "full",
            built_at=datetime.now(UTC),
            documents=len(documents),
            extraction_fingerprint=(
                context.incremental_extraction_fingerprint or settings.fingerprint
            ),
            extraction_settings=settings,
            package_version=__version__,
            source_scopes=recorded_source_scopes(context, self.config.indexing.reset),
            # Only a full build exports (DataIngestionPipeline's export).
            visualization_export_run=(
                context.pipeline_id
                if self.config.graph.visualization.export_data and not incremental
                else None
            ),
        ).with_artifacts(artifacts)
        deleted = self.indexing_manager.deleted_communities
        # Written by a delta (an incremental run, or one whose scope the
        # registry knew no document of): counted once per commit.
        commit = (
            commit_identity(
                context.pipeline_id, delta, context.incremental_fingerprints
            )
            if delta is not None and not self.config.indexing.reset and not rewritten
            else None
        )
        counters: dict[str, CommunityCounters] | None = {}
        keeps_tree = tree_pruned and IndexArtifact.SUMMARY_TREE in (
            artifacts_to_build(self.config)
        )
        vector_indexer = self.indexing_manager.vector_indexer
        for suffix in sorted(suffixes):
            update: dict[str, Any] = {}
            if keeps_tree and vector_indexer.read_summary_tree_record(suffix):
                update = record.with_artifacts(
                    [*(record.artifacts or []), IndexArtifact.SUMMARY_TREE]
                ).model_dump(include={"artifacts", "methodologies"})
            if commit is not None:
                update["community_change"] = CommunityChange(
                    commit=commit,
                    added=added[suffix],
                    deleted=len(deleted.get(suffix, ())),
                )
            if not incremental and (not rewritten or added[suffix]):
                update["communities"] = CommunityCounters(
                    clustered_at=record.built_at,
                    communities_at_clustering=added[suffix],
                    communities_now=added[suffix],
                    runs_since=0,
                    last_commit=commit,
                )
            written = vector_indexer.record_build(
                record.model_copy(update=update), suffix
            )
            if written is None:
                if keeps_build_records(vector_indexer):
                    counters = None
            elif counters is not None and written.communities is not None:
                counters[suffix] = written.communities
        return counters

    def _record_removed_artifacts(
        self, text_units: list[TextUnit], delta: DocumentDelta | None = None
    ) -> None:
        """Drop from each suffix's stored build record the artifacts this
        run's methodologies no longer build (``artifacts_to_build``).

        An incremental run with nothing to index writes no record (the
        indices keep the one of the run that did), but a run under a
        narrower profile stops maintaining the removed methodologies'
        artifacts all the same: unrecorded, their strategies would still be
        served until a run with a delta. The suffixes are the configured
        one and those the registry records of the ``delta``'s unchanged
        documents name (a document with an index value of its own), as the
        diff read them (``DocumentDelta.unchanged_suffixes``) or, from a
        registry whose diff does not, read again. Best-effort, as
        ``record_build``.
        """
        building = artifacts_to_build(self.config)
        vector_indexer = self.indexing_manager.vector_indexer
        suffixes = {BaseIndexer.get_suffix(unit) for unit in text_units} or {
            resolve_namespace(self.config, None)
        }
        if delta is not None and delta.unchanged_suffixes is not None:
            suffixes |= set(delta.unchanged_suffixes)
        elif delta is not None and delta.unchanged:
            try:
                records = self._build_doc_status_store().get_many(delta.unchanged)
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - best-effort, as the record
                log_error(
                    logger,
                    logging.WARNING,
                    "Could not read the unchanged documents' index suffixes; "
                    "only '%s' records the methodologies no longer built: %s",
                    ", ".join(sorted(suffixes)),
                    error=e,
                )
            else:
                suffixes |= {r.suffix for r in records.values() if r.suffix}
        for suffix in sorted(suffixes):
            stored = vector_indexer.read_build_record(suffix)
            held = stored_artifacts(stored)
            if stored is None or held is None or held <= building:
                continue
            try:
                record = IndexBuildRecord.model_validate(stored)
            except ValidationError:
                continue
            kept = record.with_artifacts(
                recorded_artifacts(building, held, incremental=True)
            )
            logger.info(
                "Methodologies of '%s' no longer built: %s; recorded as gone "
                "(their stored items stay until a reset)",
                suffix,
                ", ".join(
                    m
                    for m in record.methodologies or []
                    if m not in (kept.methodologies or [])
                )
                or "none",
            )
            vector_indexer.record_build(kept, suffix)

    def _index_incremental(
        self,
        context: PipelineContext,
        text_units: list[TextUnit],
        entities: list[Entity],
        relationships: list[Relationship],
        communities: list[Community],
        community_reports: list[CommunityReport],
        claims: list[Claim],
        before_registry: Callable[[Mapping[str, IndexingStats]], None] | None = None,
    ) -> tuple[dict[str, Any], CommunityDrift | None]:
        """Index the run's delta and its community drift (``index_delta``)."""
        return index_delta(
            self._build_doc_status_store,
            self.indexing_manager,
            context,
            text_units,
            entities,
            relationships,
            communities,
            community_reports,
            claims,
            before_registry,
        )

    def _validate_backend_success(self, indexing_results: dict[str, Any]) -> None:
        # Fail if any index type failed entirely (0 successes) OR exceeded the
        # tolerated partial-failure rate. The partial-failure gate is what stops
        # a run where, say, most relationship edges are dropped by a backend
        # error from still being reported as a successful pipeline (the write
        # failures otherwise only surface in per-stage stats). A wholly failed
        # backend fails all its index types, so it is caught here too.
        max_failure_rate = self.config.indexing.max_failure_rate
        failed_index_types = []
        for key, stats in indexing_results.items():
            # An index type that raised before _perform_indexing seeds
            # total_items surfaces as total_items=0, failed_items=N. Treat that
            # (failed_items>0, zero successes) as a hard failure rather than
            # skipping it, otherwise a complete write failure passes as success.
            if stats.total_items <= 0:
                if stats.failed_items > 0 and stats.successful_items == 0:
                    failed_index_types.append(key)
                    logger.error(
                        "Index type '%s' failed: %s items failed before any "
                        "were counted (complete write failure).",
                        key,
                        stats.failed_items,
                    )
                continue
            failure_rate = stats.failed_items / stats.total_items
            fully_failed = stats.successful_items == 0
            if fully_failed or failure_rate > max_failure_rate:
                failed_index_types.append(key)
                logger.error(
                    "Index type '%s' failed: %s/%s items failed (%.1f%% > %.1f%% "
                    "tolerated).",
                    key,
                    stats.failed_items,
                    stats.total_items,
                    failure_rate * 100,
                    max_failure_rate * 100,
                )

        if failed_index_types:
            first_error = next(
                (
                    indexing_results[key].errors[0]
                    for key in failed_index_types
                    if indexing_results[key].errors
                ),
                None,
            )
            # Shown and logged whole: a write's errors are log-safe as
            # recorded (IndexingStats.add_error), so the first names the cause.
            message = (
                f"Indexing failed: {', '.join(failed_index_types)} "
                f"exceeded the tolerated failure rate ({max_failure_rate:.0%}) or "
                f"failed to index any items. This indicates a configuration or "
                f"connectivity issue; the WARNING and ERROR lines above name "
                "each failed write."
            ) + (f" First error: {first_error}" if first_error else "")
            raise PipelineStageError(message, own_text=True)
