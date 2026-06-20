# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3

from aws_graphrag.core import PipelineStageError, get_logger
from aws_graphrag.models import (
    Config,
    Document,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
)
from aws_graphrag.storage import IndexingManager

from .chunker import ChunkerFactory
from .claim_extractor import ClaimExtractor
from .claim_resolver import ClaimResolver
from .community_detector import CommunityDetector, CommunityMetrics
from .gleaner import GraphGleaner
from .graph_analyzer import GraphAnalyzer
from .graph_builder import GraphBuilder
from .graph_extractor import GraphExtractor
from .graph_resolver import GraphResolver
from .loader import DirectoryLoader
from .parser import ParserFactory
from .translator import TextUnitTranslator

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

    def execute(self, context: PipelineContext) -> PipelineStageResult:
        logger.info(f"Starting stage: '{self.name}'")
        start_time = datetime.now()

        try:
            input_count, output_count, metrics = self._execute_core(context)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self._validate_critical_stage_output(input_count, output_count)

            logger.info(
                f"Stage '{self.name}' completed successfully in "
                f"{duration:.2f} seconds. "
                f"Inputs: {input_count}, Outputs: {output_count}"
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
            logger.error(f"Stage '{self.name}' failed: {e}", exc_info=True)
            end_time = datetime.now()

            return self._create_result(
                PipelineStageStatus.FAILED, start_time, end_time, error_message=str(e)
            )

    @abstractmethod
    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        pass

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
            logger.info(f"Stage '{self.name}' had no input to process")
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
                f"Stage '{self.name}' processed {input_count} inputs but produced "
                f"0 outputs"
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
            f"Could not convert stats object of type {type(stats_obj)} to dict"
        )
        return {"stats": str(stats_obj)}


class DocumentLoadingStage(PipelineStage):
    def __init__(
        self,
        config: Config,
        source_directory: Path,
        boto_session: boto3.Session | None = None,
        parse_files: bool = False,
    ):
        super().__init__(PipelineStageType.DOCUMENT_LOADING, config, boto_session)
        self.loader = DirectoryLoader(
            source_directory,
            config=self.config,
            deduplicate=self.config.processing.deduplicate,
            parse_files=parse_files,
        )

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
        # diff against it and process only new/changed documents.
        delta_skipped = 0
        if self.config.aws.dynamodb.enabled:
            documents, delta_skipped = self._apply_incremental_filter(documents)

        context.documents = documents
        output_count = len(context.documents)

        if self.loader.failed_files:
            logger.warning(f"Failed to load {len(self.loader.failed_files)} files")

        metrics = {
            "file_count": input_count,
            "success_count": output_count,
            "delta_skipped": delta_skipped,
            "failed_files": self.loader.failed_files,
        }

        logger.info("=" * 60)
        logger.info(
            f"DOCUMENT LOADING STAGE - COMPLETED ({output_count} documents loaded)"
        )
        logger.info("=" * 60)

        return input_count, output_count, metrics

    def _apply_incremental_filter(
        self, documents: list[Document]
    ) -> tuple[list[Document], int]:
        """Keep only new/changed documents per the DynamoDB doc-status registry.

        Degrades to processing everything (and logs) if the registry is
        unreachable, so an enabled-but-misconfigured registry never blocks a run.
        Imported lazily to avoid a hard dependency when the feature is off.
        """
        try:
            from aws_graphrag.adapters.aws import DynamoDBDocStatusStore
            from aws_graphrag.ingestion.delta_detector import (
                detect_delta,
                filter_documents_to_process,
            )

            store = DynamoDBDocStatusStore(self.config, boto_session=self.boto_session)
            delta, _ = detect_delta(documents, store)
            if delta.is_empty:
                logger.info("Incremental: no new/changed documents detected")
            to_process = filter_documents_to_process(documents, delta)
            skipped = len(documents) - len(to_process)
            logger.info(
                "Incremental filter: %d to process, %d unchanged (skipped)",
                len(to_process),
                skipped,
            )
            return to_process, skipped
        except Exception as e:
            logger.warning(
                "Incremental filter unavailable (%s); processing all documents", e
            )
            return documents, 0


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
            logger.warning(f"No supported files found in '{self.source_directory}'")
            return 0, 0, {"parsed_files": [], "failed_files": []}

        logger.info(f"Found {input_count} files to parse")

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
                logger.error(f"Failed to parse '{file_path}': {e}")
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
            f"DOCUMENT PARSING STAGE - COMPLETED ({output_count}/{input_count} files parsed)"
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
            logger.debug(f"Saved parsed document: '{output_path}'")
        except Exception as e:
            logger.error(f"Failed to save parsed document '{output_path}': {e}")


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
            f"TEXT CHUNKING STAGE - COMPLETED ({total_chunks} text units created)"
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
        target_language = self.config.processing.translation.target_language.value
        self.translator = TextUnitTranslator(config, boto_session=self.boto_session)
        self.target_language = target_language

    def _execute_core(
        self, context: PipelineContext
    ) -> tuple[int, int, dict[str, Any] | None]:
        logger.info("=" * 60)
        logger.info("TRANSLATION STAGE - STARTED")
        logger.info("=" * 60)

        translated_units = self.translator.translate_text_units(context.text_units)
        context.translated_units = translated_units

        translation_stats: dict[str, Any] = {}
        try:
            if hasattr(self.translator, "stats"):
                stats = self.translator.stats
                if stats is not None:
                    translation_stats = self._stats_to_dict(stats)
        except Exception as e:
            logger.warning(f"Failed to get translation stats: {e}")

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
            f"TRANSLATION STAGE - COMPLETED "
            f"(Translated {units_translated_count} text units)"
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
            f"GRAPH EXTRACTION STAGE - COMPLETED "
            f"({entities_count} entities, {relationships_count} relationships)"
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
            "GLEANING STAGE - COMPLETED "
            f"({len(initial_entities)} -> {final_entities_count} entities, "
            f"{len(initial_relationships)} -> {final_relationships_count} "
            "relationships improved)"
        )
        logger.info("=" * 60)

        input_count = (
            len(text_units) + len(initial_entities) + len(initial_relationships)
        )
        output_count = final_entities_count + final_relationships_count
        return input_count, output_count, metrics


class GraphResolutionStage(PipelineStage):
    def __init__(self, config: Config):
        super().__init__(PipelineStageType.GRAPH_RESOLUTION, config)
        self.resolver = GraphResolver(config)

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

        context.resolved_entities = resolution_result["entities"]
        context.resolved_relationships = resolution_result["relationships"]

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
            "GRAPH RESOLUTION STAGE - COMPLETED "
            f"({original_entities_count} -> {resolved_entities_count} entities, "
            f"{original_relationships_count} -> {resolved_relationships_count} "
            "relationships resolved)"
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
            f"CLAIM EXTRACTION STAGE - COMPLETED ({len(claims)} claims extracted)"
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
            "CLAIM RESOLUTION STAGE - COMPLETED "
            f"({original_claims_count} -> {resolved_count} claims resolved, "
            f"{removed_count} claims removed)"
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
            "GRAPH ANALYSIS STAGE - COMPLETED "
            f"({graph_stats.num_nodes} nodes, {graph_stats.num_edges} edges)"
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
                from aws_graphrag.visualization import GraphVisualizationManager

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
                logger.warning(f"Visualization generation failed: {e}")

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
            "COMMUNITY DETECTION STAGE - COMPLETED "
            f"({communities_count} communities, {reports_count} reports)"
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
    ):
        super().__init__(PipelineStageType.INDEXING, config, boto_session)
        self.indexing_manager = IndexingManager(config=self.config)

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

        # Validate that no backend has completely failed
        self._validate_backend_success(indexing_results)

        metrics = {
            "indexing_results": {k: v.to_dict() for k, v in indexing_results.items()},
            "total_indexed": total_indexed,
            "total_failed": total_failed,
            "success_rate": (
                total_indexed / (total_indexed + total_failed)
                if (total_indexed + total_failed) > 0
                else 0
            ),
        }

        logger.info("=" * 60)
        logger.info(
            "INDEXING STAGE - COMPLETED "
            f"({total_indexed} items indexed, {total_failed} failed)"
        )
        logger.info("=" * 60)

        return input_count, total_indexed, metrics

    def _validate_backend_success(self, indexing_results: dict[str, Any]) -> None:
        # 1) Per-index-type validation: fail if any individual index type completely failed
        failed_index_types = []
        for key, stats in indexing_results.items():
            if stats and stats.total_items > 0 and stats.successful_items == 0:
                failed_index_types.append(key)
                logger.error(
                    f"Index type '{key}' completely failed: "
                    f"0/{stats.total_items} items indexed successfully"
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
                    f"{backend_name} backend completely failed: "
                    f"0/{backend_total} items indexed successfully"
                )

        if failed_backends:
            error_msg = (
                f"Indexing failed: {', '.join(failed_backends)} backend(s) "
                f"completely failed to index any items. "
                f"This indicates a critical configuration or connectivity issue. "
                f"Check the logs above for specific error details."
            )
            raise PipelineStageError(error_msg)
