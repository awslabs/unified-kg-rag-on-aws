# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unified_kg_rag.adapters.ingestion.chunker import (
    ChunkerFactory,
    ChunkingStats,
    ChunkingStrategy,
    ChunkProcessor,
    ChunkQualityValidator,
    IntelligentTextChunker,
    SimpleTextChunker,
)
from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractor
from unified_kg_rag.adapters.ingestion.community_detector import (
    CommunityDetector,
    HierarchicalCommunity,
)
from unified_kg_rag.adapters.ingestion.gleaner import (
    GleaningRound,
    GleaningStats,
    GraphGleaner,
)
from unified_kg_rag.adapters.ingestion.graph_extractor import (
    ExtractionStats,
    GraphExtractor,
)
from unified_kg_rag.adapters.ingestion.loader import DirectoryLoader
from unified_kg_rag.adapters.ingestion.parser import (
    BaseParser,
    ParserFactory,
    ParsingStats,
)
from unified_kg_rag.adapters.ingestion.translator import (
    TextUnitTranslator,
    TranslationStats,
)
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
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
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.base_resolver import BaseResolver, FuzzyMatcher
from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_content_hash,
    compute_doc_id,
    detect_delta,
    filter_documents_to_process,
    fingerprint_documents,
)
from unified_kg_rag.domain.ingestion.graph_analyzer import (
    CentralityMetrics,
    GraphAnalyzer,
    GraphStatistics,
)
from unified_kg_rag.domain.ingestion.graph_resolver import (
    EntityResolver,
    GraphResolver,
    RelationshipResolver,
)
from unified_kg_rag.domain.models import CommunityMetrics

__all__ = [
    "BaseParser",
    "BaseProcessor",
    "BaseResolver",
    "CentralityMetrics",
    "ChunkProcessor",
    "ChunkQualityValidator",
    "ChunkerFactory",
    "ChunkingStats",
    "ChunkingStrategy",
    "ClaimExtractor",
    "ClaimExtractionStage",
    "ClaimResolutionStage",
    "CommunityDetectionStage",
    "CommunityDetector",
    "CommunityMetrics",
    "DataIngestionPipeline",
    "DirectoryLoader",
    "DocumentLoadingStage",
    "IncrementalIndexer",
    "compute_content_hash",
    "compute_doc_id",
    "detect_delta",
    "filter_documents_to_process",
    "fingerprint_documents",
    "DocumentParsingStage",
    "EntityResolver",
    "ExtractionStats",
    "FuzzyMatcher",
    "GleaningRound",
    "GleaningStage",
    "GleaningStats",
    "GraphAnalysisStage",
    "GraphAnalyzer",
    "GraphExtractionStage",
    "GraphExtractor",
    "GraphGleaner",
    "GraphResolutionStage",
    "GraphResolver",
    "GraphStatistics",
    "HierarchicalCommunity",
    "IndexingStage",
    "IntelligentTextChunker",
    "ParserFactory",
    "ParsingStats",
    "PipelineStage",
    "RelationshipResolver",
    "SimpleTextChunker",
    "TextChunkingStage",
    "TextUnitTranslator",
    "TranslationStage",
    "TranslationStats",
]
