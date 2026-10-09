# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
from structlog.contextvars import bind_contextvars, reset_contextvars

from unified_kg_rag.adapters.ingestion.chunker import ChunkerFactory
from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractor
from unified_kg_rag.adapters.ingestion.community_detector import (
    CommunityDetector,
    CommunityMetrics,
)
from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
)
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.adapters.ingestion.loader import DirectoryLoader
from unified_kg_rag.adapters.ingestion.parser import (
    UNSTRUCTURED_EXTENSIONS,
    UNSTRUCTURED_INSTALL_HINT,
    ParserFactory,
)
from unified_kg_rag.adapters.ingestion.translator import TextUnitTranslator
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
    document_doc_id,
)
from unified_kg_rag.domain.ingestion.graph_analyzer import GraphAnalyzer
from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Constants,
    Document,
    DocumentDelta,
    DocumentLineage,
    Entity,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    Relationship,
    TextUnit,
)
from unified_kg_rag.shared import (
    DocStatusRegistryError,
    PipelineStageError,
    get_logger,
)

if TYPE_CHECKING:
    from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
    from unified_kg_rag.ports import DocStatusPort, GraphIndexer, VectorIndexer

logger = get_logger(__name__)


def _build_knowledge_graph(context: PipelineContext) -> Any:
    """Build the knowledge graph from the context's resolved graph outputs.

    graph_analysis builds it; community_detection rebuilds it the same way when
    a resumed run restored only the cached entities, relationships and claims
    (the graph itself is neither cached nor kept in the run metadata).
    """
    claims = context.resolved_claims or context.claims
    return GraphBuilder(
        context.resolved_entities, context.resolved_relationships, claims
    ).build()


class PipelineStage(ABC):
    CRITICAL_STAGES = {
        PipelineStageType.COMMUNITY_DETECTION,
        PipelineStageType.DOCUMENT_LOADING,
        PipelineStageType.DOCUMENT_PARSING,
        PipelineStageType.GRAPH_ANALYSIS,
        PipelineStageType.GRAPH_EXTRACTION,
        PipelineStageType.GRAPH_RESOLUTION,
        PipelineStageType.INDEXING,
        PipelineStageType.TEXT_CHUNKING,
        PipelineStageType.TRANSLATION,
    }

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

    OPTIONAL_OUTPUT_STAGES = {
        PipelineStageType.CLAIM_EXTRACTION,
        PipelineStageType.CLAIM_RESOLUTION,
        PipelineStageType.GLEANING,
    }

    def __init__(
        self,
        stage_type: PipelineStageType,
        config: Config,
        boto_session: boto3.Session | None = None,
        providers: Providers | None = None,
    ) -> None:
        self.config = config
        self.stage_type = stage_type
        # The pipeline passes its one provider bundle, so every stage's LLM /
        # embedding adapters share the session and any injected factory.
        self.providers = Providers.resolve(config, providers, boto_session)
        self.boto_session = self.providers.boto_session

    @property
    def name(self) -> str:
        return self.stage_type.value

    def close(self) -> None:  # noqa: B027 - intentional no-op default hook
        """Release any backend clients the stage owns.

        Default is a no-op: only ``IndexingStage`` holds long-lived backend
        connections. Not abstract on purpose — stages opt in by overriding.
        """

    def execute(self, context: PipelineContext) -> PipelineStageResult:
        tokens = bind_contextvars(stage=self.name)
        logger.info("Starting stage: '%s'", self.name)
        start_time = datetime.now()

        try:
            input_count, output_count, metrics = self._execute_core(context)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if not self._allows_empty_output(context):
                self._validate_critical_stage_output(input_count, output_count)

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
            end_time = datetime.now()

            return self._create_result(
                PipelineStageStatus.FAILED, start_time, end_time, error_message=str(e)
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
        """
        return context.incremental_delta is not None and not context.documents

    def _empty_input_hint(self) -> str:
        return "Check that the previous stage completed successfully."

    def _validate_critical_stage_output(
        self, input_count: int, output_count: int
    ) -> None:
        is_critical = self._should_validate_output()
        must_have_input = self.stage_type in self.MUST_HAVE_INPUT_STAGES

        if input_count == 0:
            if must_have_input:
                error_msg = (
                    f"Stage '{self.name}' received 0 inputs but requires input "
                    f"to function. {self._empty_input_hint()}"
                )
                logger.error(error_msg)
                raise PipelineStageError(error_msg)
            logger.info("Stage '%s' had no input to process", self.name)
            return

        if is_critical and output_count == 0:
            error_msg = (
                f"Critical stage '{self.name}' processed "
                f"{input_count} inputs but produced 0 outputs. Check the stage "
                f"configuration and input data quality."
            )
            logger.error(error_msg)
            raise PipelineStageError(error_msg)

        if output_count == 0:
            logger.warning(
                "Stage '%s' processed %s inputs but produced 0 outputs",
                self.name,
                input_count,
            )

    def _should_validate_output(self) -> bool:
        if self.stage_type in self.CRITICAL_STAGES:
            return True
        if self.stage_type in self.OPTIONAL_OUTPUT_STAGES:
            return False
        return True

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
            cache_path=None,
        )

    @staticmethod
    def _stats_to_dict(stats_obj: Any) -> dict[str, Any]:
        if not stats_obj:
            return {}
        if hasattr(stats_obj, "to_dict") and callable(stats_obj.to_dict):
            result = stats_obj.to_dict()
            if isinstance(result, dict):
                return result
        if hasattr(stats_obj, "__dict__"):
            return dict(stats_obj.__dict__)
        logger.warning(
            "Could not convert stats object of type %s to dict", type(stats_obj)
        )
        return {"stats": str(stats_obj)}


class DocumentLoadingStage(PipelineStage):
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
        boto_session: boto3.Session | None = None,
        parse_files: bool = False,
        doc_status: "DocStatusPort | None" = None,
    ):
        super().__init__(PipelineStageType.DOCUMENT_LOADING, config, boto_session)
        self.loader = DirectoryLoader(
            source_directory,
            config=self.config,
            # Without parsing, the directory holds pre-parsed Document JSON; any
            # other file would only fail Document.from_json_file.
            supported_extensions=None if parse_files else {".json"},
            deduplicate=self.config.processing.deduplicate,
            parse_files=parse_files,
        )
        # Injected by the pipeline (built once at the orchestration layer) when
        # incremental indexing is enabled; built lazily otherwise.
        self._doc_status = doc_status

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("DOCUMENT LOADING STAGE - STARTED")
        logger.info("=" * 60)

        # A context restored from a reused pipeline id carries the previous
        # run's delta in its metadata. This stage recomputes the document set,
        # so that delta no longer describes it: drop it before (re)deriving one,
        # or a run with the registry now disabled (or a failed diff under
        # continue_on_error) would index against stale fingerprints.
        context.incremental_delta = None
        context.incremental_fingerprints = {}
        context.incremental_scope = None

        failed_files: list[str] = []
        if self._parsing_completed(context):
            documents = list(context.documents)
            input_count = len(documents)
            logger.info(
                "Using %s documents from the '%s' stage",
                input_count,
                PipelineStageType.DOCUMENT_PARSING.value,
            )
            if self.loader.deduplicate:
                documents = self.loader.deduplicate_documents(documents)
        else:
            discovered_files = self.loader.discover_files()
            input_count = len(discovered_files)
            documents = self.loader.load()
            for document in documents:
                assign_document_identity(
                    document,
                    self.loader.source_directory,
                    self.config.indexing.additional_suffix,
                )
            failed_files = self.loader.failed_files
            context.failed_source_files = list(failed_files)

        # Incremental indexing: when a doc-status registry is injected or the
        # DynamoDB one is enabled, diff against it, stash the delta/fingerprints
        # for the IndexingStage, and process only new/changed documents.
        delta_skipped = 0
        if self._doc_status is not None or self.config.aws.dynamodb.enabled:
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
            "failed_files": failed_files,
        }

        logger.info("=" * 60)
        logger.info(
            "DOCUMENT LOADING STAGE - COMPLETED (%s documents loaded)", output_count
        )
        logger.info("=" * 60)

        return input_count, output_count, metrics

    @staticmethod
    def _parsing_completed(context: PipelineContext) -> bool:
        return any(
            r.stage_name == PipelineStageType.DOCUMENT_PARSING.value
            and r.status == PipelineStageStatus.COMPLETED
            for r in context.stage_results
        )

    def _apply_incremental_filter(
        self, documents: list[Document], context: PipelineContext
    ) -> tuple[list[Document], int]:
        """Keep only new/changed documents per the DynamoDB doc-status registry.

        Also stashes the computed delta + fingerprints on the context so the
        IndexingStage can prune stale artifacts, propagate deletions, and record
        per-document lineage (the full incremental commit path).

        Raises :class:`DocStatusRegistryError` when the registry cannot be
        read. There is no safe fallback: without a delta the indexing stage
        takes the full-rebuild path, which replaces the suffix's live index
        content with this run's documents and never records them, so other
        scopes' documents would be lost and every later run would rebuild.
        Imported lazily to avoid a hard dependency when the feature is off.
        """
        from unified_kg_rag.domain.ingestion.delta_detector import (
            assign_registry_source,
            detect_delta,
            filter_documents_to_process,
            fingerprint_documents,
            legacy_doc_id,
        )

        scope, source_scope, failed_doc_ids = self._registry_scope(context)
        for document in documents:
            assign_registry_source(document, source_scope)
        if self.config.indexing.reset:
            # A reset clears the stores and the registry before indexing, so
            # every document is new: diffing against the registry here would
            # rebuild from the delta alone and lose the unchanged documents.
            fingerprints = fingerprint_documents(documents)
            context.incremental_delta = DocumentDelta(new=list(fingerprints))
            context.incremental_fingerprints = fingerprints
            context.incremental_scope = scope
            logger.info(
                "Incremental filter skipped (indexing.reset): processing all "
                "%d documents",
                len(documents),
            )
            return documents, 0

        try:
            store = self._build_doc_status_store()
            delta, fingerprints = detect_delta(
                documents,
                store,
                scope=scope,
                failed_doc_ids=list(failed_doc_ids),
                max_failures=self.config.indexing.max_document_failures,
                legacy_doc_ids={
                    **failed_doc_ids,
                    **{
                        document_doc_id(document): legacy_doc_id(document)
                        for document in documents
                    },
                },
            )
        except DocStatusRegistryError:
            raise
        except Exception as e:
            raise DocStatusRegistryError(
                f"Doc-status registry: could not diff the corpus against the "
                f"registry ({type(e).__name__}: {e}). Incremental indexing needs "
                "the registry; fix it and re-run, or set aws.dynamodb.enabled: "
                "false to run without it."
            ) from e
        context.incremental_delta = delta
        context.incremental_fingerprints = fingerprints
        context.incremental_scope = scope
        if delta.is_empty:
            logger.info("Incremental: no new/changed documents detected")
        to_process = filter_documents_to_process(documents, delta)
        skipped = len(documents) - len(to_process)
        logger.info(
            "Incremental filter: %d to process, %d unchanged (skipped), %d deleted",
            len(to_process),
            skipped,
            len(delta.deleted),
        )
        return to_process, skipped

    def _registry_scope(
        self, context: PipelineContext
    ) -> tuple[str, str, dict[str, str]]:
        """The run's registry scope, its source scope and its failed files.

        The scope is the index namespace the run writes (``index_value`` +
        ``indexing.additional_suffix``) and the corpus source
        (``document_parsing.source_scope``, else the resolved source
        directory), so a run never deletes another tenant's or another
        corpus's documents. The source scope is also part of every document's
        registry key, so two corpora written to one namespace never share a
        record. Failed files are keyed the same way their documents would have
        been, mapped to their legacy key (see ``legacy_doc_id``).
        """
        from unified_kg_rag.domain.ingestion.delta_detector import registry_scope
        from unified_kg_rag.shared.utils.document_identity import (
            compute_doc_id,
            normalize_source_path,
            registry_namespace,
            relative_source_path,
        )

        parsing = self.config.processing.document_parsing
        namespace = registry_namespace(
            parsing.index_value, self.config.indexing.additional_suffix
        )
        root = self.loader.source_directory
        source_scope = parsing.source_scope or root.as_posix()
        failed_doc_ids: dict[str, str] = {}
        for path in context.failed_source_files:
            relative = relative_source_path(path, root) or normalize_source_path(path)
            failed_doc_ids[compute_doc_id(relative, namespace, source_scope)] = (
                compute_doc_id(relative, namespace)
            )
        return registry_scope(namespace, source_scope), source_scope, failed_doc_ids

    def _build_doc_status_store(self) -> "DocStatusPort":
        if self._doc_status is not None:
            return self._doc_status
        # Fallback: construct the default DynamoDB adapter (e.g. when the stage
        # is used standalone in tests without injection).
        from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore

        return DynamoDBDocStatusStore(self.config, boto_session=self.boto_session)


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
        boto_session: boto3.Session | None = None,
        cache_directory: Path | None = None,
    ):
        super().__init__(PipelineStageType.DOCUMENT_PARSING, config, boto_session)
        self.source_directory = Path(source_directory)
        self.target_directory = Path(target_directory) if target_directory else None
        self.cache_directory = Path(cache_directory) if cache_directory else None
        if (
            self.target_directory is not None
            and self.target_directory.resolve() == self.source_directory.resolve()
        ):
            raise ValueError(
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
        logger.info("=" * 60)
        logger.info("DOCUMENT PARSING STAGE - STARTED")
        logger.info("=" * 60)

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

        logger.info("=" * 60)
        logger.info(
            "DOCUMENT PARSING STAGE - COMPLETED (%s/%s files parsed)",
            output_count,
            input_count,
        )
        logger.info("=" * 60)

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
                or self._should_exclude_file(file_path)
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

    @staticmethod
    def _should_exclude_file(file_path: Path) -> bool:
        exclude_patterns = {".*", "*.pyc", "__pycache__"}

        for pattern in exclude_patterns:
            if file_path.match(pattern):
                return True

        return False

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
        boto_session: boto3.Session | None = None,
        providers: Providers | None = None,
    ):
        super().__init__(
            PipelineStageType.TEXT_CHUNKING, config, boto_session, providers
        )

        chunker_type = self.config.processing.chunking.chunker_type
        self.chunker = ChunkerFactory.create_chunker(
            config=self.config,
            boto_session=self.boto_session,
            chunker_type=chunker_type,
            providers=self.providers,
        )
        self.chunker_type = chunker_type

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("TEXT CHUNKING STAGE - STARTED")
        logger.info("=" * 60)

        text_units = []
        for doc in context.documents:
            chunks = self.chunker.chunk_documents([doc])
            text_units.extend(chunks)

        context.text_units = text_units
        total_chunks = len(text_units)
        doc_count = len(context.documents)

        metrics = {
            "chunker_type": self.chunker_type.value,
            "documents_processed": doc_count,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": total_chunks / doc_count if doc_count else 0.0,
        }

        logger.info("=" * 60)
        logger.info(
            "TEXT CHUNKING STAGE - COMPLETED (%s text units created)", total_chunks
        )
        logger.info("=" * 60)

        return doc_count, total_chunks, metrics


class TranslationStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        providers: Providers | None = None,
    ):
        super().__init__(PipelineStageType.TRANSLATION, config, boto_session, providers)
        self.translation_config = self.config.processing.translation
        self.target_language = self.translation_config.target_language.value
        # Build the Bedrock-backed translator lazily so a disabled / no-op stage
        # costs nothing (no LLM client, no calls).
        self._translator: TextUnitTranslator | None = None

    @property
    def translator(self) -> TextUnitTranslator:
        if self._translator is None:
            self._translator = TextUnitTranslator(
                self.config, boto_session=self.boto_session, providers=self.providers
            )
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
        logger.info("=" * 60)
        logger.info("TRANSLATION STAGE - STARTED")
        logger.info("=" * 60)

        skip_reason = self._should_skip()
        if skip_reason is not None:
            logger.info("TRANSLATION STAGE - SKIPPED (%s)", skip_reason)
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
            logger.info("=" * 60)
            return len(context.text_units), len(context.text_units), metrics

        translated_units = self.translator.translate_text_units(context.text_units)
        context.translated_units = translated_units
        # Units left untranslated are extracted in the source language; their
        # documents are recorded FAILED and retried, like an extraction failure.
        stats = self.translator.stats
        context.failed_text_unit_ids[self.name] = (
            list(stats.failed_text_unit_ids) if stats else []
        )

        translation_stats = self._stats_to_dict(stats)

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

        logger.info("=" * 60)
        logger.info(
            "TRANSLATION STAGE - COMPLETED (Translated %s text units)",
            units_translated_count,
        )
        logger.info("=" * 60)

        return len(context.text_units), len(translated_units), metrics


class GraphExtractionStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        providers: Providers | None = None,
    ):
        super().__init__(
            PipelineStageType.GRAPH_EXTRACTION, config, boto_session, providers
        )
        self.extractor = GraphExtractor(self.config, providers=self.providers)

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("GRAPH EXTRACTION STAGE - STARTED")
        logger.info("=" * 60)

        text_units = context.translated_units or context.text_units

        entities, relationships, stats = self.extractor.extract_from_text_units(
            text_units
        )
        context.entities = entities
        context.relationships = relationships
        context.failed_text_unit_ids[self.name] = (
            stats.failed_text_unit_ids if stats else []
        )

        entities_count = len(context.entities)
        relationships_count = len(context.relationships)

        metrics = {
            "text_units_processed": len(text_units),
            "entities_extracted": entities_count,
            "relationships_extracted": relationships_count,
            "failed_units": stats.num_failed_extractions if stats else 0,
            "extraction_stats": self._stats_to_dict(stats),
        }

        logger.info("=" * 60)
        logger.info(
            "GRAPH EXTRACTION STAGE - COMPLETED (%s entities, %s relationships)",
            entities_count,
            relationships_count,
        )
        logger.info("=" * 60)

        return len(text_units), entities_count + relationships_count, metrics


class GleaningStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        providers: Providers | None = None,
    ):
        super().__init__(PipelineStageType.GLEANING, config, boto_session, providers)
        self.gleaner = GraphGleaner(self.config, providers=self.providers)

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("GLEANING STAGE - STARTED")
        logger.info("=" * 60)

        text_units = context.translated_units or context.text_units
        initial_entities = context.entities
        initial_relationships = context.relationships

        entities, relationships, gleaning_stats = self.gleaner.glean_graph(
            text_units, initial_entities, initial_relationships
        )

        context.entities = entities
        context.relationships = relationships
        context.failed_text_unit_ids[self.name] = (
            gleaning_stats.failed_text_unit_ids if gleaning_stats else []
        )
        final_entities_count = len(context.entities)
        final_relationships_count = len(context.relationships)

        # Graph growth from gleaning: items it added per extracted item.
        initial_items = len(initial_entities) + len(initial_relationships)
        improvement_rate = 0.0
        if gleaning_stats and initial_items > 0:
            improvement_rate = (
                gleaning_stats.total_entities_added
                + gleaning_stats.total_relationships_added
            ) / initial_items

        metrics = {
            "text_units_processed": len(text_units),
            "initial_entities": len(initial_entities),
            "final_entities": final_entities_count,
            "initial_relationships": len(initial_relationships),
            "final_relationships": final_relationships_count,
            "entity_improvement": final_entities_count - len(initial_entities),
            "relationship_improvement": final_relationships_count
            - len(initial_relationships),
            "improvement_rate": improvement_rate,
            "iterations_completed": (
                gleaning_stats.total_rounds if gleaning_stats else 0
            ),
            "refinement_calls": (
                gleaning_stats.total_refinement_calls if gleaning_stats else 0
            ),
            "failed_units": gleaning_stats.num_failed_units if gleaning_stats else 0,
            "gleaning_stats": self._stats_to_dict(gleaning_stats),
        }

        logger.info("=" * 60)
        logger.info(
            "GLEANING STAGE - COMPLETED (%s -> %s entities, %s -> %s relationships improved)",
            len(initial_entities),
            final_entities_count,
            len(initial_relationships),
            final_relationships_count,
        )
        logger.info("=" * 60)

        input_count = (
            len(text_units) + len(initial_entities) + len(initial_relationships)
        )
        output_count = final_entities_count + final_relationships_count
        return input_count, output_count, metrics


class GraphResolutionStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        providers: Providers | None = None,
    ):
        super().__init__(
            PipelineStageType.GRAPH_RESOLUTION, config, boto_session, providers
        )
        self.resolver = GraphResolver(config)
        # Resolution merges descriptions (concatenation); re-summarize the
        # over-long ones with an LLM here (parity with MS/LightRAG). Needs Bedrock,
        # hence GRAPH_RESOLUTION is in BOTO_REQUIRED_STAGES.
        self.description_summarizer = DescriptionSummarizer(
            config, providers=self.providers
        )

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("GRAPH RESOLUTION STAGE - STARTED")
        logger.info("=" * 60)

        resolution_result, stats = self.resolver.resolve_graph(
            context.entities, context.relationships
        )

        original_entities_count = len(context.entities)
        original_relationships_count = len(context.relationships)

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
            "resolution_stats": self._stats_to_dict(stats),
        }

        logger.info("=" * 60)
        logger.info(
            "GRAPH RESOLUTION STAGE - COMPLETED (%s -> %s entities, %s -> %s relationships resolved)",
            original_entities_count,
            resolved_entities_count,
            original_relationships_count,
            resolved_relationships_count,
        )
        logger.info("=" * 60)

        input_count = original_entities_count + original_relationships_count
        output_count = resolved_entities_count + resolved_relationships_count
        return input_count, output_count, metrics


class ClaimExtractionStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        providers: Providers | None = None,
    ):
        super().__init__(
            PipelineStageType.CLAIM_EXTRACTION, config, boto_session, providers
        )
        self.extractor = ClaimExtractor(self.config, providers=self.providers)

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("CLAIM EXTRACTION STAGE - STARTED")
        logger.info("=" * 60)

        text_units = context.translated_units or context.text_units
        claims, extraction_stats = self.extractor.extract_from_text_units(
            text_units, context.resolved_entities
        )
        context.claims = claims
        context.failed_text_unit_ids[self.name] = (
            extraction_stats.failed_text_unit_ids if extraction_stats else []
        )

        metrics = {
            "text_units_processed": len(text_units),
            "claims_extracted": len(claims),
            "failed_units": (
                extraction_stats.num_failed_extractions if extraction_stats else 0
            ),
            "extraction_stats": self._stats_to_dict(extraction_stats),
        }

        logger.info("=" * 60)
        logger.info(
            "CLAIM EXTRACTION STAGE - COMPLETED (%s claims extracted)", len(claims)
        )
        logger.info("=" * 60)

        return len(text_units), len(claims), metrics


class ClaimResolutionStage(PipelineStage):
    def __init__(self, config: Config):
        super().__init__(PipelineStageType.CLAIM_RESOLUTION, config)
        self.resolver = ClaimResolver(config)

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("CLAIM RESOLUTION STAGE - STARTED")
        logger.info("=" * 60)

        if not context.claims or not context.resolved_entities:
            logger.warning(
                "Skipping claim-entity resolution due to missing claims or "
                "resolved entities."
            )
            return 0, 0, {"claims_processed": 0, "entities_available": 0}

        entities = context.resolved_entities
        original_claims_count = len(context.claims)
        resolved_claims, stats = self.resolver.resolve(context.claims, entities)
        context.resolved_claims = resolved_claims

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
            "resolution_stats": self._stats_to_dict(stats),
        }

        logger.info("=" * 60)
        logger.info(
            "CLAIM RESOLUTION STAGE - COMPLETED (%s -> %s claims resolved, %s claims removed)",
            original_claims_count,
            resolved_count,
            removed_count,
        )
        logger.info("=" * 60)

        return original_claims_count, resolved_count, metrics


class GraphAnalysisStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        providers: Providers | None = None,
    ):
        super().__init__(
            PipelineStageType.GRAPH_ANALYSIS, config, boto_session, providers
        )
        self.analyzer = GraphAnalyzer(config)

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("GRAPH ANALYSIS STAGE - STARTED")
        logger.info("=" * 60)

        entities = context.resolved_entities
        relationships = context.resolved_relationships
        claims = context.resolved_claims or context.claims

        context.knowledge_graph = _build_knowledge_graph(context)
        self.analyzer.graph = context.knowledge_graph

        centrality_data = self.analyzer.calculate_centrality()
        graph_stats = self.analyzer.get_graph_statistics()

        context.graph_statistics = graph_stats
        context.centrality_metrics = list(centrality_data.values())

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
            "centrality_stats": {
                "nodes_with_centrality": len(centrality_data),
                "centrality_types_calculated": len(self.analyzer.centrality_cache),
            },
        }

        logger.info("=" * 60)
        logger.info(
            "GRAPH ANALYSIS STAGE - COMPLETED (%s nodes, %s edges)",
            graph_stats.num_nodes,
            graph_stats.num_edges,
        )
        logger.info("=" * 60)

        input_count = len(entities) + len(relationships)
        output_count = len(entities)
        return input_count, output_count, metrics


class CommunityDetectionStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        cache_directory: Path | None = None,
        providers: Providers | None = None,
    ):
        super().__init__(
            PipelineStageType.COMMUNITY_DETECTION, config, boto_session, providers
        )
        self.detector = CommunityDetector(config, providers=self.providers)
        self.cache_directory = Path(cache_directory) if cache_directory else None

    def _visualization_outputs_dir(self, context: PipelineContext) -> Path | None:
        # Unset outputs_directory -> the pipeline's cache dir, which the S3 cache
        # sync uploads (a deployed task's local outputs/ is lost on exit).
        if (
            self.config.graph.visualization.outputs_directory
            or not self.cache_directory
        ):
            return None
        return self.cache_directory / context.pipeline_id / "visualization"

    def _check_report_failures(
        self, communities: list[Community], failed_community_ids: list[str]
    ) -> list[str]:
        """Text units of the communities left without a report.

        Their documents are recorded FAILED, so the next incremental run
        re-processes them and generates the missing reports, the same way an
        extraction failure is retried. More failures than
        ``indexing.max_failure_rate`` of the communities fail the stage (the
        reports are the community-report index's items).
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
        failed = set(failed_community_ids)
        return sorted(
            {
                unit_id
                for community in communities
                if community.id in failed
                for unit_id in community.text_unit_ids or []
            }
        )

    @staticmethod
    def _get_detection_stats_dict(metrics_obj: CommunityMetrics) -> dict[str, Any]:
        if not metrics_obj:
            return {
                "num_communities": 0,
                "average_community_size": 0.0,
                "largest_community_size": 0,
                "smallest_community_size": 0,
                "community_size_distribution": {},
            }
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
        logger.info("=" * 60)
        logger.info("COMMUNITY DETECTION STAGE - STARTED")
        logger.info("=" * 60)

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

        self.detector(context.knowledge_graph)
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

        if self.config.graph.visualization.enabled and context.knowledge_graph:
            try:
                # Imported lazily to avoid an import cycle (visualization.base
                # imports ingestion, which imports this module).
                from unified_kg_rag.visualization import GraphVisualizationManager

                logger.info("Generating graph visualizations...")
                analyzer = GraphAnalyzer(self.config)
                analyzer.graph = context.knowledge_graph

                visualization_manager = GraphVisualizationManager(
                    config=self.config,
                    graph_analyzer=analyzer,
                    community_detector=self.detector,
                    outputs_dir=self._visualization_outputs_dir(context),
                    boto_session=self.boto_session,
                    providers=self.providers,
                )

                visualization_manager.run()
                logger.info("Graph visualizations generated successfully")
            except Exception as e:
                logger.warning("Visualization generation failed: %s", e)

        communities_count = len(context.communities)
        reports_count = len(context.community_reports)
        metrics = {
            "entities_processed": len(entities),
            "relationships_processed": len(relationships),
            "communities_detected": communities_count,
            "reports_generated": reports_count,
            "reports_failed": len(failed_community_ids),
            "modularity_score": metrics_obj.modularity if metrics_obj else 0.0,
            "detection_stats": (
                self._get_detection_stats_dict(metrics_obj) if metrics_obj else {}
            ),
        }

        logger.info("=" * 60)
        logger.info(
            "COMMUNITY DETECTION STAGE - COMPLETED (%s communities, %s reports)",
            communities_count,
            reports_count,
        )
        logger.info("=" * 60)

        input_count = len(entities) + len(relationships)
        output_count = communities_count + reports_count
        return input_count, output_count, metrics


class IndexingStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        doc_status: "DocStatusPort | None" = None,
        providers: Providers | None = None,
        vector_indexer: "VectorIndexer | None" = None,
        graph_indexer: "GraphIndexer | None" = None,
    ):
        super().__init__(PipelineStageType.INDEXING, config, boto_session, providers)
        self.indexing_manager = IndexingManager(
            config=self.config,
            providers=self.providers,
            vector_indexer=vector_indexer,
            graph_indexer=graph_indexer,
        )
        # Injected by the pipeline for the incremental commit/registry write-back.
        self._doc_status = doc_status

    def close(self) -> None:
        """Close the indexers' Neptune/OpenSearch clients (best-effort)."""
        self.indexing_manager.close()

    def _build_doc_status_store(self) -> "DocStatusPort":
        if self._doc_status is not None:
            return self._doc_status
        from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore

        return DynamoDBDocStatusStore(self.config, boto_session=self.boto_session)

    def _clear_doc_status_registry(self) -> None:
        """Delete all records from the doc-status registry (reset path).

        Uses the port's list_all + delete so no new port method is needed.
        Best-effort: a registry-clear failure should not abort the reset run
        (the graph/index were already cleared).
        """
        try:
            store = self._build_doc_status_store()
            records = store.list_all()
            for record in records:
                store.delete(record.doc_id)
            logger.info("Cleared %s doc-status registry records on reset", len(records))
        except Exception as e:  # noqa: BLE001 - reset cleanup is best-effort
            logger.warning("Failed to clear doc-status registry on reset: %s", e)

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("INDEXING STAGE - STARTED")
        logger.info("=" * 60)

        if self.config.indexing.reset:
            logger.info("Clearing all existing data")
            text_units = context.translated_units or context.text_units
            if not self.indexing_manager.clear_all_data(text_units=text_units):
                raise RuntimeError("Failed to clear existing data before indexing")
            # Also clear the DynamoDB doc-status registry. clear_all_data only
            # wipes OpenSearch + Neptune; leaving stale content-hash lineage would
            # make the NEXT incremental run classify re-ingested docs as
            # "unchanged" (skipping them despite the graph having been wiped) and
            # leave rows for docs no longer in the corpus. Only relevant when the
            # registry is in use (incremental mode).
            if self._doc_status is not None or self.config.aws.dynamodb.enabled:
                self._clear_doc_status_registry()

        if not self.indexing_manager.initialize():
            raise RuntimeError("Failed to initialize indexing pipeline")

        text_units = context.translated_units or context.text_units
        entities = context.resolved_entities
        relationships = context.resolved_relationships
        communities = context.communities or []
        community_reports = context.community_reports or []
        claims = context.resolved_claims or context.claims or []

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
            and (self._doc_status is not None or self.config.aws.dynamodb.enabled)
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

        if context.incremental_delta is not None and not self.config.indexing.reset:
            indexing_results = self._index_incremental(
                context,
                text_units,
                entities,
                relationships,
                communities,
                community_reports,
                claims,
            )
        else:
            indexing_results = self.indexing_manager.index_all_data(
                text_units=text_units,
                entities=entities,
                relationships=relationships,
                communities=communities,
                community_reports=community_reports,
                claims=claims,
            )
            if context.incremental_delta is not None:
                # Reset with the registry enabled: the registry was cleared with
                # the stores, so record the rebuilt corpus for the next delta run.
                incremental = self._incremental_indexer(context, text_units)
                incremental.record(
                    self._document_lineages(
                        context,
                        incremental.suffix,
                        text_units,
                        entities,
                        relationships,
                        communities,
                        community_reports,
                        claims,
                    ),
                    context.incremental_fingerprints,
                    indexing_results,
                    self._documents_with_failed_units(context, text_units),
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

        metrics = {
            "indexing_results": {k: v.to_dict() for k, v in indexing_results.items()},
            "total_indexed": total_indexed,
            "total_failed": total_failed,
            "relationships_indexed": relationships_indexed,
            "success_rate": (
                total_indexed / (total_indexed + total_failed)
                if (total_indexed + total_failed) > 0
                else 0
            ),
        }

        logger.info("=" * 60)
        logger.info(
            "INDEXING STAGE - COMPLETED (%s items indexed, %s failed)",
            total_indexed,
            total_failed,
        )
        logger.info("=" * 60)

        return input_count, total_indexed, metrics

    def _index_incremental(
        self,
        context: PipelineContext,
        text_units: list[TextUnit],
        entities: list[Entity],
        relationships: list[Relationship],
        communities: list[Community],
        community_reports: list[CommunityReport],
        claims: list[Claim],
    ) -> dict[str, Any]:
        """Idempotent delta indexing + stale-artifact pruning + registry write-back.

        Routed to when a doc-status delta is present (incremental mode). Stale
        artifacts of changed/deleted documents are removed first, then the freshly
        extracted delta is upserted and the registry updated with per-document
        lineage so subsequent runs diff correctly.
        """
        delta = context.incremental_delta
        if delta is None:  # defensive; caller already guards
            return self.indexing_manager.index_all_data(
                text_units=text_units,
                entities=entities,
                relationships=relationships,
                communities=communities,
                community_reports=community_reports,
                claims=claims,
            )
        incremental = self._incremental_indexer(context, text_units)

        # Drop the stale artifacts of changed and deleted docs in ONE removal
        # plan (before re-upsert): planning them separately keeps whatever a
        # changed and a deleted doc share, orphaning it. A failed removal must
        # stop the run before commit when docs changed: committing would
        # overwrite their lineage and orphan the stale artifacts. A failure
        # that only concerns deleted docs keeps their registry rows for a
        # retry and need not hold back the delta, so it fails the stage only
        # after the commit.
        removed = incremental.remove_changed_and_deleted(delta)
        if not removed and delta.changed:
            raise PipelineStageError(
                "Removing stale artifacts of changed or deleted documents "
                "failed; not committing the delta so the registry keeps their "
                "old lineage and the next run retries the removal"
            )

        lineages = self._document_lineages(
            context,
            incremental.suffix,
            text_units,
            entities,
            relationships,
            communities,
            community_reports,
            claims,
        )
        results = incremental.commit(
            lineages=lineages,
            fingerprints=context.incremental_fingerprints,
            failed_doc_ids=self._documents_with_failed_units(context, text_units),
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            communities=communities,
            community_reports=community_reports,
            claims=claims,
        )
        if not removed:
            raise PipelineStageError(
                "Removing the artifacts of deleted documents failed; their "
                "registry records are kept so the next run retries the removal"
            )
        return results

    @staticmethod
    def _documents_with_failed_units(
        context: PipelineContext, text_units: list[TextUnit]
    ) -> set[str]:
        """Registry ids of the documents an extraction stage failed on."""
        failed_units = {
            unit_id for ids in context.failed_text_unit_ids.values() for unit_id in ids
        }
        run_document_ids = {
            document_id
            for unit in text_units
            if unit.id in failed_units
            for document_id in unit.document_ids or []
        }
        return {
            document_doc_id(document)
            for document in context.documents
            if document.document_id in run_document_ids
        }

    def _incremental_indexer(
        self, context: PipelineContext, text_units: list[TextUnit]
    ) -> "IncrementalIndexer":
        from unified_kg_rag.application.ingestion.incremental import (
            IncrementalIndexer,
        )
        from unified_kg_rag.ports.indexer import BaseIndexer

        # Artifacts carry their own index suffix (multi-tenant/version aware);
        # derive the run's suffix from the text units the same way the indexers do.
        suffix = (
            BaseIndexer.get_suffix(text_units[0])
            if text_units
            else Constants.DEFAULT_SUFFIX.value
        )
        return IncrementalIndexer(
            self._build_doc_status_store(),
            self.indexing_manager,
            suffix=suffix,
            scope=context.incremental_scope,
        )

    @staticmethod
    def _document_lineages(
        context: PipelineContext,
        suffix: str,
        text_units: list[TextUnit],
        entities: list[Entity],
        relationships: list[Relationship],
        communities: list[Community],
        community_reports: list[CommunityReport],
        claims: list[Claim],
    ) -> list[DocumentLineage]:
        from unified_kg_rag.application.ingestion.incremental import (
            build_document_lineage,
        )

        return build_document_lineage(
            documents=context.documents,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            communities=communities,
            claims=claims,
            community_reports=community_reports,
            suffix=suffix,
        )

    def _validate_backend_success(self, indexing_results: dict[str, Any]) -> None:
        # 1) Per-index-type validation: fail if any individual index type failed
        #    entirely (0 successes) OR exceeded the tolerated partial-failure rate.
        #    The partial-failure gate is what stops a run where, say, most
        #    relationship edges are dropped by a backend error from still being
        #    reported as a successful pipeline (the write failures otherwise only
        #    surface in per-stage stats and never affect stage/pipeline success).
        max_failure_rate = self.config.indexing.max_failure_rate
        failed_index_types = []
        for key, stats in indexing_results.items():
            if not stats:
                continue
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
            error_msg = (
                f"Indexing failed: {', '.join(failed_index_types)} "
                f"exceeded the tolerated failure rate ({max_failure_rate:.0%}) or "
                f"failed to index any items. This indicates a configuration or "
                f"connectivity issue. Check the logs above for specific errors."
            )
            raise PipelineStageError(error_msg)

        # 2) Backend-level validation (defensive): fail if entire backend has zero successes
        opensearch_keys = [
            k for k in indexing_results.keys() if k.startswith("opensearch_")
        ]
        neptune_keys = [k for k in indexing_results.keys() if k.startswith("neptune_")]

        backend_groups = {
            "OpenSearch": opensearch_keys,
            "Neptune": neptune_keys,
        }

        failed_backends = []

        for backend_name, keys in backend_groups.items():
            if not keys:
                continue

            backend_total = 0
            backend_successful = 0
            backend_failed = 0

            for key in keys:
                stats = indexing_results.get(key)
                if stats:
                    backend_total += stats.total_items
                    backend_successful += stats.successful_items
                    backend_failed += stats.failed_items

            # backend_total counts only items that reached _perform_indexing; an
            # index type that raised earlier contributes to backend_failed only.
            # Gate on either so a wholly-exception backend (total=0, failed>0) is
            # still flagged.
            backend_attempted = backend_total + backend_failed
            if backend_attempted > 0 and backend_successful == 0:
                failed_backends.append(backend_name)
                logger.error(
                    "%s backend completely failed: 0/%s items indexed successfully",
                    backend_name,
                    backend_attempted,
                )

        if failed_backends:
            error_msg = (
                f"Indexing failed: {', '.join(failed_backends)} backend(s) "
                f"completely failed to index any items. "
                f"This indicates a critical configuration or connectivity issue. "
                f"Check the logs above for specific error details."
            )
            raise PipelineStageError(error_msg)
