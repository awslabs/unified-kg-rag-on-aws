# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from structlog.contextvars import bind_contextvars, reset_contextvars

from unified_kg_rag.adapters.ingestion.chunker import ChunkerFactory
from unified_kg_rag.adapters.ingestion.claim_extractor import (
    ClaimExtractionStats,
    ClaimExtractor,
)
from unified_kg_rag.adapters.ingestion.community_detector import (
    CommunityDetector,
    CommunityMetrics,
)
from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
)
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.adapters.ingestion.graph_extractor import (
    ExtractionStats,
    GraphExtractor,
)
from unified_kg_rag.adapters.ingestion.loader import DirectoryLoader
from unified_kg_rag.adapters.ingestion.parser import (
    UNSTRUCTURED_EXTENSIONS,
    UNSTRUCTURED_INSTALL_HINT,
    ParserFactory,
)
from unified_kg_rag.adapters.ingestion.translator import TextUnitTranslator
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.extraction_store import (
    ExtractionStore,
    with_record_keys,
)
from unified_kg_rag.application.ingestion.incremental import (
    artifact_owners,
    clear_registry_namespaces,
    context_lineages,
    dedup_preferences,
    filter_to_delta,
    index_delta,
    reset_namespaces,
    run_failures,
    run_incremental_indexer,
    unit_source_files,
    warn_about_unapplied_retired_scopes,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.base_resolver import (
    compact_name_key,
    normalize_entity_type,
)
from unified_kg_rag.domain.ingestion.cache_keys import (
    extraction_settings,
    is_excluded_source_path,
    unit_record_fingerprint,
)
from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver
from unified_kg_rag.domain.ingestion.community_drift import (
    CommunityDrift,
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
from unified_kg_rag.domain.ingestion.document_identity import store_namespace_tag
from unified_kg_rag.domain.ingestion.graph_analyzer import GraphAnalyzer
from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Constants,
    DocStatus,
    Document,
    Entity,
    IndexBuildRecord,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.ports.indexer import BaseIndexer, GraphIndexer
from unified_kg_rag.shared import (
    ConfigurationError,
    PipelineStageError,
    get_logger,
    observability,
)
from unified_kg_rag.shared.genai import ERROR_TYPE
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

        except Exception as e:
            logger.exception("Stage '%s' failed: %s", self.name, e)
            span.record_exception(e)
            span.set_attribute(ERROR_TYPE, type(e).__name__)
            end_time = datetime.now()

            return self._create_result(
                PipelineStageStatus.FAILED,
                start_time,
                end_time,
                input_count,
                output_count,
                error_message=str(e),
            )
        finally:
            reset_contextvars(**tokens)

    @abstractmethod
    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        pass

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
                raise PipelineStageError(error_msg)
            logger.info("Stage '%s' had no input to process", self.name)
            return

        if is_critical and output_count == 0:
            error_msg = (
                f"Critical stage '{self.name}' processed "
                f"{input_count} inputs but produced 0 outputs. Check the stage "
                f"configuration and input data quality."
            )
            if failed_files:
                error_msg += (
                    f" It failed on {len(failed_files)} files: "
                    + ", ".join(failed_files[:10])
                    + (" ..." if len(failed_files) > 10 else "")
                )
            raise PipelineStageError(error_msg)

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

    def __init__(
        self,
        config: Config,
        source_directory: Path,
        doc_status: "DocStatusPort | None" = None,
        providers: Providers | None = None,
    ):
        super().__init__(PipelineStageType.DOCUMENT_LOADING, config, providers)
        # Without parsing, the directory holds pre-parsed Document JSON.
        # Deduplication runs in this stage, after document identity is
        # assigned, so an incremental run can prefer indexed documents.
        self.loader = DirectoryLoader(source_directory)
        # Injected by the pipeline (built once at the orchestration layer) when
        # incremental indexing is enabled; built lazily otherwise.
        self._doc_status = doc_status

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
            documents = self.loader.load()
            for document in documents:
                assign_document_identity(
                    document,
                    self.loader.source_directory,
                    self.config.indexing.additional_suffix,
                    store_namespace_tag(self.config.indexing),
                )
            failed_files = self.loader.failed_files
            context.failed_source_files = list(failed_files)

        incremental = self._registry_on
        if self.config.processing.deduplicate:
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

        # Incremental indexing: when a doc-status registry is injected or the
        # DynamoDB one is enabled, diff against it, stash the delta/fingerprints
        # for the IndexingStage, and process only new/changed documents.
        delta_skipped = 0
        if incremental:
            documents, delta_skipped = self._apply_incremental_filter(
                documents, context
            )

        context.documents = documents
        output_count = len(context.documents)

        if failed_files:
            logger.warning("Failed to load %s files", len(failed_files))

        metrics = {
            "file_count": input_count,
            "success_count": output_count,
            "delta_skipped": delta_skipped,
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
        }

        return input_count, output_count, metrics

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
        )

    @staticmethod
    def _parsing_completed(context: PipelineContext) -> bool:
        return any(
            r.stage_name == PipelineStageType.DOCUMENT_PARSING.value
            and r.status == PipelineStageStatus.COMPLETED
            for r in context.stage_results
        )


class DocumentParsingStage(PipelineStage):
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
    ):
        super().__init__(PipelineStageType.DOCUMENT_PARSING, config, providers)
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

    def _empty_input_hint(self) -> str:
        return (
            f"No supported source files found in '{self.source_directory}' "
            f"(supported: {', '.join(sorted(self.supported_extensions))}); see "
            "the warnings above for skipped files."
        )

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        files_to_parse = self._discover_files()
        input_count = len(files_to_parse)

        if not files_to_parse:
            logger.warning("No supported files found in '%s'", self.source_directory)
            return 0, 0, {"parsed_files": [], "failed_files": []}

        logger.info("Found %s files to parse", input_count)

        parsed_documents = []
        failed_files = []

        for file_path in files_to_parse:
            try:
                parser = ParserFactory.create_parser(file_path, self.config)
                document = parser.parse_file(
                    file_path, self.config.processing.document_parsing.index_value
                )
                assign_document_identity(
                    document,
                    self.source_directory,
                    self.config.indexing.additional_suffix,
                    store_namespace_tag(self.config.indexing),
                )
                parsed_documents.append(document)

                if self.target_directory is not None:
                    self._save_parsed_document(
                        document, file_path, self.target_directory
                    )

            except Exception as e:
                logger.error("Failed to parse '%s': %s", file_path, e)
                failed_files.append(str(file_path))

        context.documents = parsed_documents
        # Incremental delta detection must not mistake a file that failed to
        # parse for a deleted one (that would remove its indexed content).
        context.failed_source_files = failed_files
        output_count = len(parsed_documents)

        metrics = {
            "input_files": input_count,
            "parsed_files": [doc.file_name for doc in parsed_documents],
            "failed_files": failed_files,
            "success_rate": (
                (output_count / input_count * 100) if input_count > 0 else 0
            ),
        }

        return input_count, output_count, metrics

    def _discover_files(self) -> list[Path]:
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
        skipped: Counter[str] = Counter()
        for file_path in self.source_directory.rglob("*"):
            if (
                not file_path.is_file()
                or is_excluded_source_path(file_path, self.source_directory)
                or any(file_path.resolve().is_relative_to(d) for d in owned_dirs)
            ):
                continue
            suffix = file_path.suffix.lower()
            if suffix in self.supported_extensions:
                files.append(file_path)
            elif suffix:
                skipped[suffix] += 1

        for suffix, count in sorted(skipped.items()):
            remedy = (
                UNSTRUCTURED_INSTALL_HINT
                if suffix in UNSTRUCTURED_EXTENSIONS
                else "convert them to a supported format"
            )
            logger.warning(
                "Skipping %s '%s' file(s) in '%s': unsupported file type; to "
                "ingest them, %s",
                count,
                suffix,
                self.source_directory,
                remedy,
            )
        return sorted(files)

    def _save_parsed_document(
        self, document: Document, original_path: Path, output_directory: Path
    ) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_filename = f"{original_path.stem}.json"
        output_path = output_directory / output_filename

        try:
            document.to_json_file(output_path)
            logger.debug("Saved parsed document: '%s'", output_path)
        except Exception as e:
            logger.error("Failed to save parsed document '%s': %s", output_path, e)


class TextChunkingStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
    ):
        super().__init__(PipelineStageType.TEXT_CHUNKING, config, providers)

        chunker_type = self.config.processing.chunking.chunker_type
        self.chunker = ChunkerFactory.create_chunker(
            config=self.config,
            chunker_type=chunker_type,
            providers=self.providers,
        )
        self.chunker_type = chunker_type

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        # One call for the whole corpus: the intelligent chunker batches its
        # boundary-model calls across documents, and the logged stats cover
        # the stage rather than its last document.
        text_units = self.chunker.chunk_documents(context.documents)
        context.text_units = text_units
        total_chunks = len(text_units)
        doc_count = len(context.documents)

        metrics = {
            "chunker_type": self.chunker_type.value,
            "documents_processed": doc_count,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": total_chunks / doc_count if doc_count else 0.0,
        }

        return doc_count, total_chunks, metrics


class TranslationStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
    ):
        super().__init__(PipelineStageType.TRANSLATION, config, providers)
        self.translation_config = self.config.processing.translation
        self.target_language = self.translation_config.target_language.value
        # Build the Bedrock-backed translator lazily so a disabled / no-op stage
        # costs nothing (no LLM client, no calls).
        self._translator: TextUnitTranslator | None = None

    @property
    def translator(self) -> TextUnitTranslator:
        if self._translator is None:
            self._translator = TextUnitTranslator(self.config, providers=self.providers)
        return self._translator

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

        translated_units = self.translator.translate_text_units(context.text_units)
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
        fingerprint = unit_record_fingerprint(
            self.config, gleaned=self.config.processing.gleaning.enabled
        )
        by_unit = self.store.load(text_units, fingerprint, record_from_payload)
        to_extract = [unit for unit in text_units if unit.id not in by_unit]
        stats = ExtractionStats()
        if to_extract:
            records, stats = self.extractor.extract_units(to_extract)
            by_unit.update((record.text_unit_id, record) for record in records)
        logger.info(
            "Graph extraction: %s text units reused from the extraction store, "
            "%s extracted",
            len(text_units) - len(to_extract),
            len(to_extract),
        )
        context.unit_extractions = [
            by_unit[unit.id] for unit in text_units if unit.id in by_unit
        ]
        context.failed_text_unit_ids[self.name] = stats.failed_text_unit_ids
        context.item_failed_text_unit_ids[self.name] = stats.item_failed_text_unit_ids

        entities_count, relationships_count = _distinct_counts(context.unit_extractions)
        metrics = {
            "text_units_processed": len(text_units),
            "text_units_reused": len(text_units) - len(to_extract),
            "entities_extracted": entities_count,
            "relationships_extracted": relationships_count,
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

        records, gleaning_stats = self.gleaner.glean(
            text_units, context.unit_extractions
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


class GraphResolutionStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
    ):
        super().__init__(PipelineStageType.GRAPH_RESOLUTION, config, providers)
        self.resolver = GraphResolver(config)
        # Resolution merges descriptions (concatenation); re-summarize the
        # over-long ones with an LLM here (parity with MS/LightRAG).
        self.description_summarizer = DescriptionSummarizer(
            config, providers=self.providers
        )
        self.store = ExtractionStore(self.config, self.providers.build_vector_indexer)

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
                ],
                unit_record_fingerprint(self.config, gleaned=gleaned),
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

        context.resolved_entities = self.description_summarizer.summarize_entities(
            resolution_result["entities"]
        )
        context.resolved_relationships = (
            self.description_summarizer.summarize_relationships(
                resolution_result["relationships"]
            )
        )

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
        }

        input_count = original_entities_count + original_relationships_count
        output_count = resolved_entities_count + resolved_relationships_count
        return input_count, output_count, metrics


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
            fingerprint = unit_record_fingerprint(
                self.config, gleaned=gleaned, claims=True
            )
            per_unit.update(self.store.load(units, fingerprint, claims_from_payload))
        reused = len(per_unit)

        to_extract = [unit for unit in text_units if unit.id not in per_unit]
        stats = ClaimExtractionStats()
        unsaved: set[str] = set()
        if to_extract:
            entities = [
                entity
                for unit in to_extract
                if unit.id in records
                for entity in unit_entities(records[unit.id])
            ]
            extracted, stats = self.extractor.extract_units(to_extract, entities)
            per_unit.update(extracted)
            for gleaned in (False, True):
                unsaved |= self.store.save(
                    [
                        (storable[unit_id], claims_payload(claims))
                        for unit_id, claims in extracted.items()
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
            "claims_extracted": len(claims),
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


class GraphAnalysisStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
    ):
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
        graph_stats = self.analyzer.get_graph_statistics()

        metrics = {
            "entities_analyzed": len(entities),
            "relationships_analyzed": len(relationships),
            "claims_analyzed": len(claims) if claims else 0,
            "graph_metrics": {
                "num_nodes": graph_stats.num_nodes,
                "num_edges": graph_stats.num_edges,
                "density": graph_stats.density,
                "average_clustering": graph_stats.average_clustering,
                "diameter": graph_stats.diameter,
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
        cache_directory: Path | None = None,
        providers: Providers | None = None,
        doc_status: "DocStatusPort | None" = None,
    ):
        super().__init__(PipelineStageType.COMMUNITY_DETECTION, config, providers)
        self.detector = CommunityDetector(config, providers=self.providers)
        self.cache_directory = Path(cache_directory) if cache_directory else None
        self._doc_status = doc_status

    def _visualization_outputs_dir(self, context: PipelineContext) -> Path | None:
        # Unset outputs_directory -> the pipeline's cache dir, which the S3 cache
        # sync uploads (a deployed task's local outputs/ is lost on exit).
        if (
            self.config.graph.visualization.outputs_directory
            or not self.cache_directory
        ):
            return None
        return self.cache_directory / context.pipeline_id / "visualization"

    def _export_visualization_data(self, context: PipelineContext) -> None:
        """Write the graph and communities for ``run-visualization`` (no render).

        Best effort, like rendering: a failed write never fails the stage.
        """
        import json

        from unified_kg_rag.domain.ingestion.visualization_data import (
            DEFAULT_OUTPUTS_DIRECTORY,
            VISUALIZATION_DATA_FILENAME,
            graph_data,
        )

        viz = self.config.graph.visualization
        outputs_dir = self._visualization_outputs_dir(context) or Path(
            viz.outputs_directory or DEFAULT_OUTPUTS_DIRECTORY
        )
        path = outputs_dir / VISUALIZATION_DATA_FILENAME
        try:
            data = graph_data(
                context.knowledge_graph,
                self.detector.export_community_data(),
                {
                    str(node): str(community)
                    for node, community in self.detector.node_to_community_l0.items()
                },
            )
            outputs_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Visualization data export failed: %s", e)
            return
        logger.info("Exported visualization data to '%s'", path)

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
        (the reports are the community-report index's items).
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
                f"indexing.max_failure_rate). Check the report model errors above."
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
                records = self._build_doc_status_store().get_many(
                    [*delta.to_process, *delta.deleted, *delta.quarantined]
                )
                removed = {
                    u.id for u in context.text_units or context.translated_units
                }.union(*(r.text_unit_ids for r in records.values()))
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
            except Exception as e:
                logger.warning(
                    "Could not read the stored graph around %s small delta "
                    "components; leaving them out of the communities: %s",
                    len(components),
                    e,
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
        self.detector(context.knowledge_graph, self._whole_graph(context))
        community_objects = self.detector.generate_community_objects()
        # Text units resolve each report's source documents (report lineage).
        community_reports, failed_community_ids = self.detector.generate_reports(
            community_objects,
            text_units=context.text_units or context.translated_units,
        )
        metrics_obj = self.detector.get_community_metrics()

        context.communities = community_objects
        context.community_reports = community_reports
        context.failed_text_unit_ids[self.name] = self._check_report_failures(
            community_objects, failed_community_ids
        )

        viz = self.config.graph.visualization
        delta = context.incremental_delta
        if (viz.enabled or viz.export_data) and delta is not None and delta.unchanged:
            # An incremental run's graph holds only the delta's documents:
            # rendering or exporting it would replace the full graph's
            # visualization with a fragment.
            logger.info(
                "Visualization skipped: this incremental run's graph holds only "
                "its %d new or changed documents, not the %d unchanged ones. "
                "To visualize the whole graph, run a full build "
                "(indexing.reset: true) and then run-visualization on its "
                "visualization_data.json.",
                len(delta.to_process),
                len(delta.unchanged),
            )
        elif self.config.graph.visualization.enabled and context.knowledge_graph:
            try:
                # Imported on use: rendering is opt-in, and the visualization
                # package loads umap and scikit-learn (seconds of import time).
                from unified_kg_rag.visualization import GraphVisualizationManager

                logger.info("Generating graph visualizations...")
                analyzer = GraphAnalyzer(self.config)
                analyzer.graph = context.knowledge_graph

                visualization_manager = GraphVisualizationManager(
                    config=self.config,
                    graph_analyzer=analyzer,
                    community_detector=self.detector,
                    outputs_dir=self._visualization_outputs_dir(context),
                    providers=self.providers,
                )

                visualization_manager.run()
                logger.info("Graph visualizations generated successfully")
            except Exception as e:
                logger.warning("Visualization generation failed: %s", e)
        elif self.config.graph.visualization.export_data and context.knowledge_graph:
            self._export_visualization_data(context)

        communities_count = len(context.communities)
        reports_count = len(context.community_reports)
        metrics = {
            "entities_processed": len(entities),
            "relationships_processed": len(relationships),
            "communities_detected": communities_count,
            "reports_generated": reports_count,
            "reports_failed": len(failed_community_ids),
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
    ):
        super().__init__(PipelineStageType.INDEXING, config, providers)
        self.indexing_manager = IndexingManager(
            config=self.config, providers=self.providers
        )
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
        # A pipeline can be run again (library use): nothing an earlier run's
        # derivation decided carries over (a failed commit's re-keys would
        # delete old ids whose replacements were never written).
        self.indexing_manager.reset_run_state()

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
                "indexing.reset: true to rebuild the stores."
            )

        drift = None
        if context.incremental_delta is not None and not self.config.indexing.reset:
            warn_about_unapplied_retired_scopes(self.config, context)
            indexing_results, drift = self._index_incremental(
                context,
                text_units,
                entities,
                relationships,
                communities,
                community_reports,
                claims,
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
                rebuild = (incremental, lineages)
            # A build the registry does not record must not be updated
            # incrementally later (see VectorIndexer.check_index_settings).
            self.indexing_manager.opensearch_indexer.set_doc_status_tracking(
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
                recorded = incremental.record(
                    lineages,
                    context.incremental_fingerprints,
                    indexing_results,
                    failures.failed,
                    artifact_owners(
                        lineages,
                        [
                            *text_units,
                            *entities,
                            *relationships,
                            *communities,
                            *community_reports,
                            *claims,
                        ],
                    ),
                    counted_failed_doc_ids=failures.counted,
                    claims_missing_doc_ids=failures.claims_missing,
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
        self._record_build(context, text_units)

        metrics = {
            "indexing_results": {k: v.to_dict() for k, v in indexing_results.items()},
            "total_indexed": total_indexed,
            "total_failed": total_failed,
            "relationships_indexed": relationships_indexed,
            # Stored items an incremental run could not derive (a text unit
            # they cite names no stored record): stale until a rebuild.
            "underived_items": self.indexing_manager.underived_items,
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

        return input_count, total_indexed, metrics

    def _record_build(
        self, context: PipelineContext, text_units: list[TextUnit]
    ) -> None:
        """Record on each written suffix's vector indices what built them.

        ``run-eval`` copies the record into its run manifest, so a score can be
        traced to the ingestion run, its extraction settings and its corpus
        size. Written after the backends succeeded; best-effort.
        """
        delta = context.incremental_delta
        # A delta of new documents only (the registry knew none of the
        # scope's documents) built the suffix from nothing: a full build.
        incremental = (
            delta is not None
            and not self.config.indexing.reset
            and bool(delta.changed or delta.unchanged or delta.deleted)
        )
        suffixes = {BaseIndexer.get_suffix(unit) for unit in text_units} or {
            self.config.processing.document_parsing.index_value
            or Constants.DEFAULT_SUFFIX.value
        }
        try:
            package_version: str | None = version("unified-kg-rag-on-aws")
        except PackageNotFoundError:
            package_version = None
        documents = {
            doc_id for unit in text_units for doc_id in unit.document_ids or []
        }
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
            package_version=package_version,
        )
        for suffix in sorted(suffixes):
            self.indexing_manager.opensearch_indexer.record_build(record, suffix)

    def _index_incremental(
        self,
        context: PipelineContext,
        text_units: list[TextUnit],
        entities: list[Entity],
        relationships: list[Relationship],
        communities: list[Community],
        community_reports: list[CommunityReport],
        claims: list[Claim],
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
            error_msg = (
                f"Indexing failed: {', '.join(failed_index_types)} "
                f"exceeded the tolerated failure rate ({max_failure_rate:.0%}) or "
                f"failed to index any items. This indicates a configuration or "
                f"connectivity issue. "
                + (
                    f"First error: {first_error}"
                    if first_error
                    else "Check the logs above for specific errors."
                )
            )
            raise PipelineStageError(error_msg)
