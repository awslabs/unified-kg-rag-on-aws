# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3

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
from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.adapters.ingestion.translator import TextUnitTranslator
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver
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
    Entity,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    Relationship,
    TextUnit,
)
from unified_kg_rag.shared import PipelineStageError, get_logger

if TYPE_CHECKING:
    from unified_kg_rag.ports import DocStatusPort

logger = get_logger(__name__)


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
    ) -> None:
        self.config = config
        self.stage_type = stage_type
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )

    @property
    def name(self) -> str:
        return self.stage_type.value

    def close(self) -> None:  # noqa: B027 - intentional no-op default hook
        """Release any backend clients the stage owns.

        Default is a no-op: only ``IndexingStage`` holds long-lived backend
        connections. Not abstract on purpose — stages opt in by overriding.
        """

    def execute(self, context: PipelineContext) -> PipelineStageResult:
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

    def _validate_critical_stage_output(
        self, input_count: int, output_count: int
    ) -> None:
        is_critical = self._should_validate_output()
        must_have_input = self.stage_type in self.MUST_HAVE_INPUT_STAGES

        if input_count == 0:
            if must_have_input:
                error_msg = (
                    f"Stage '{self.name}' received 0 inputs "
                    f"but requires input to function. Check that the previous stage "
                    f"completed successfully."
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

        discovered_files = self.loader.discover_files()
        input_count = len(discovered_files)

        result = self.loader.load()
        documents = [Document(**doc.model_dump()) for doc in result]

        # Incremental indexing: when the DynamoDB doc-status registry is enabled,
        # diff against it, stash the delta/fingerprints for the IndexingStage, and
        # process only new/changed documents.
        delta_skipped = 0
        if self.config.aws.dynamodb.enabled:
            documents, delta_skipped = self._apply_incremental_filter(
                documents, context
            )

        context.documents = documents
        output_count = len(context.documents)

        if self.loader.failed_files:
            logger.warning("Failed to load %s files", len(self.loader.failed_files))

        metrics = {
            "file_count": input_count,
            "success_count": output_count,
            "delta_skipped": delta_skipped,
            "failed_files": self.loader.failed_files,
        }

        logger.info("=" * 60)
        logger.info(
            "DOCUMENT LOADING STAGE - COMPLETED (%s documents loaded)", output_count
        )
        logger.info("=" * 60)

        return input_count, output_count, metrics

    def _apply_incremental_filter(
        self, documents: list[Document], context: PipelineContext
    ) -> tuple[list[Document], int]:
        """Keep only new/changed documents per the DynamoDB doc-status registry.

        Also stashes the computed delta + fingerprints on the context so the
        IndexingStage can prune stale artifacts, propagate deletions, and record
        per-document lineage (the full incremental commit path).

        Degrades to processing everything (and logs) if the registry is
        unreachable, so an enabled-but-misconfigured registry never blocks a run.
        Imported lazily to avoid a hard dependency when the feature is off.
        """
        try:
            from unified_kg_rag.domain.ingestion.delta_detector import (
                detect_delta,
                filter_documents_to_process,
            )

            store = self._build_doc_status_store()
            delta, fingerprints = detect_delta(documents, store)
            context.incremental_delta = delta
            context.incremental_fingerprints = fingerprints
            if delta.is_empty:
                logger.info("Incremental: no new/changed documents detected")
            to_process = filter_documents_to_process(documents, delta)
            skipped = len(documents) - len(to_process)
            logger.info(
                "Incremental filter: %d to process, %d unchanged (skipped), "
                "%d deleted",
                len(to_process),
                skipped,
                len(delta.deleted),
            )
            return to_process, skipped
        except Exception as e:
            logger.warning(
                "Incremental filter unavailable (%s); processing all documents", e
            )
            return documents, 0

    def _build_doc_status_store(self) -> "DocStatusPort":
        if self._doc_status is not None:
            return self._doc_status
        # Fallback: construct the default DynamoDB adapter (e.g. when the stage
        # is used standalone in tests without injection).
        from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore

        return DynamoDBDocStatusStore(self.config, boto_session=self.boto_session)


class DocumentParsingStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        source_directory: Path,
        target_directory: Path | None = None,
        boto_session: boto3.Session | None = None,
    ):
        super().__init__(PipelineStageType.DOCUMENT_PARSING, config, boto_session)
        self.source_directory = Path(source_directory)
        self.target_directory = Path(target_directory) if target_directory else None
        self.supported_extensions = ParserFactory.get_supported_extensions()

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
                parsed_documents.append(document)

                if self.target_directory:
                    self._save_parsed_document(document, file_path)

            except Exception as e:
                logger.error("Failed to parse '%s': %s", file_path, e)
                failed_files.append(str(file_path))

        context.documents = parsed_documents
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

        files = []
        for file_path in self.source_directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
                and not self._should_exclude_file(file_path)
            ):
                files.append(file_path)

        return sorted(files)

    @staticmethod
    def _should_exclude_file(file_path: Path) -> bool:
        exclude_patterns = {".*", "*.pyc", "__pycache__"}

        for pattern in exclude_patterns:
            if file_path.match(pattern):
                return True

        return False

    def _save_parsed_document(self, document: Document, original_path: Path) -> None:
        if not self.target_directory:
            return

        self.target_directory.mkdir(parents=True, exist_ok=True)
        output_filename = f"{original_path.stem}.json"
        output_path = self.target_directory / output_filename

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
    ):
        super().__init__(PipelineStageType.TEXT_CHUNKING, config, boto_session)

        chunker_type = self.config.processing.chunking.chunker_type
        self.chunker = ChunkerFactory.create_chunker(
            config=self.config,
            boto_session=self.boto_session,
            chunker_type=chunker_type,
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
    ):
        super().__init__(PipelineStageType.TRANSLATION, config, boto_session)
        self.translation_config = self.config.processing.translation
        self.target_language = self.translation_config.target_language.value
        # Build the Bedrock-backed translator lazily so a disabled / no-op stage
        # costs nothing (no LLM client, no calls).
        self._translator: TextUnitTranslator | None = None

    @property
    def translator(self) -> TextUnitTranslator:
        if self._translator is None:
            self._translator = TextUnitTranslator(
                self.config, boto_session=self.boto_session
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

        translation_stats: dict[str, Any] = {}
        try:
            if hasattr(self.translator, "stats"):
                stats = self.translator.stats
                if stats is not None:
                    translation_stats = self._stats_to_dict(stats)
        except Exception as e:
            logger.warning("Failed to get translation stats: %s", e)

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
    ):
        super().__init__(PipelineStageType.GRAPH_EXTRACTION, config, boto_session)
        self.extractor = GraphExtractor(self.config, boto_session=self.boto_session)

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

        entities_count = len(context.entities)
        relationships_count = len(context.relationships)

        metrics = {
            "text_units_processed": len(text_units),
            "entities_extracted": entities_count,
            "relationships_extracted": relationships_count,
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
    ):
        super().__init__(PipelineStageType.GLEANING, config, boto_session)
        self.gleaner = GraphGleaner(self.config, boto_session=self.boto_session)

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
        final_entities_count = len(context.entities)
        final_relationships_count = len(context.relationships)

        improvement_rate = 0.0
        if gleaning_stats and gleaning_stats.initial_quality_score > 0:
            improvement_rate = (
                gleaning_stats.final_quality_score
                - gleaning_stats.initial_quality_score
            ) / gleaning_stats.initial_quality_score

        metrics = {
            "text_units_processed": len(text_units),
            "initial_entities": len(initial_entities),
            "final_entities": final_entities_count,
            "initial_relationships": len(initial_relationships),
            "final_relationships": final_relationships_count,
            "entity_improvement": final_entities_count - len(initial_entities),
            "relationship_improvement": final_relationships_count
            - len(initial_relationships),
            "quality_improvement_rate": improvement_rate,
            "iterations_completed": (
                gleaning_stats.total_rounds if gleaning_stats else 0
            ),
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
    def __init__(self, config: Config, boto_session: boto3.Session | None = None):
        super().__init__(PipelineStageType.GRAPH_RESOLUTION, config, boto_session)
        self.resolver = GraphResolver(config)
        # Resolution merges descriptions (concatenation); re-summarize the
        # over-long ones with an LLM here (parity with MS/LightRAG). Needs Bedrock,
        # hence GRAPH_RESOLUTION is in BOTO_REQUIRED_STAGES.
        self.description_summarizer = DescriptionSummarizer(config, boto_session)

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
    ):
        super().__init__(PipelineStageType.CLAIM_EXTRACTION, config, boto_session)
        self.extractor = ClaimExtractor(self.config, boto_session=self.boto_session)

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

        metrics = {
            "text_units_processed": len(text_units),
            "claims_extracted": len(claims),
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
    def __init__(self, config: Config, boto_session: boto3.Session | None = None):
        super().__init__(PipelineStageType.GRAPH_ANALYSIS, config, boto_session)
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

        graph_builder = GraphBuilder(entities, relationships, claims)
        context.knowledge_graph = graph_builder.build()
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
    ):
        super().__init__(PipelineStageType.COMMUNITY_DETECTION, config, boto_session)
        self.detector = CommunityDetector(config, boto_session=self.boto_session)

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
            raise ValueError("Knowledge graph is required for community detection")

        self.detector(context.knowledge_graph)
        community_objects = self.detector.generate_community_objects()
        community_reports = self.detector.generate_reports(community_objects)
        metrics_obj = self.detector.get_community_metrics()

        context.communities = community_objects
        context.community_reports = community_reports

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
                    boto_session=self.boto_session,
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
    ):
        super().__init__(PipelineStageType.INDEXING, config, boto_session)
        self.indexing_manager = IndexingManager(config=self.config)
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
        from unified_kg_rag.application.ingestion.incremental import (
            IncrementalIndexer,
            build_document_lineage,
        )
        from unified_kg_rag.ports.indexer import BaseIndexer

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
        store = self._build_doc_status_store()
        # Artifacts carry their own index suffix (multi-tenant/version aware);
        # derive the run's suffix from the text units the same way the indexers do.
        suffix = (
            BaseIndexer.get_suffix(text_units[0])
            if text_units
            else Constants.DEFAULT_SUFFIX.value
        )
        incremental = IncrementalIndexer(store, self.indexing_manager, suffix=suffix)

        # Drop stale artifacts of changed docs (before re-upsert) and of deleted docs.
        incremental.prune_changed(delta)
        incremental.remove_deleted(delta)

        lineages = build_document_lineage(
            documents=context.documents,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            communities=communities,
            claims=claims,
            community_reports=community_reports,
            suffix=suffix,
        )
        return incremental.commit(
            lineages=lineages,
            fingerprints=context.incremental_fingerprints,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            communities=communities,
            community_reports=community_reports,
            claims=claims,
        )

    def _validate_backend_success(self, indexing_results: dict[str, Any]) -> None:
        # 1) Per-index-type validation: fail if any individual index type completely failed
        failed_index_types = []
        for key, stats in indexing_results.items():
            if stats and stats.total_items > 0 and stats.successful_items == 0:
                failed_index_types.append(key)
                logger.error(
                    "Index type '%s' completely failed: 0/%s items indexed successfully",
                    key,
                    stats.total_items,
                )

        if failed_index_types:
            error_msg = (
                f"Indexing failed: {', '.join(failed_index_types)} "
                f"completely failed to index any items. "
                f"This indicates a critical configuration or connectivity issue. "
                f"Check the logs above for specific error details."
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

            for key in keys:
                stats = indexing_results.get(key)
                if stats:
                    backend_total += stats.total_items
                    backend_successful += stats.successful_items

            if backend_total > 0 and backend_successful == 0:
                failed_backends.append(backend_name)
                logger.error(
                    "%s backend completely failed: 0/%s items indexed successfully",
                    backend_name,
                    backend_total,
                )

        if failed_backends:
            error_msg = (
                f"Indexing failed: {', '.join(failed_backends)} backend(s) "
                f"completely failed to index any items. "
                f"This indicates a critical configuration or connectivity issue. "
                f"Check the logs above for specific error details."
            )
            raise PipelineStageError(error_msg)
