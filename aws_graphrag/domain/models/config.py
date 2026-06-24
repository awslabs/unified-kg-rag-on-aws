# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr, model_validator

from .evaluation import EvaluationMetricType, EvaluatorType
from .retrieval import FusionMethod


class PipelineStageType(Enum):
    CLAIM_EXTRACTION = "claim_extraction"
    CLAIM_RESOLUTION = "claim_resolution"
    COMMUNITY_DETECTION = "community_detection"
    DOCUMENT_LOADING = "document_loading"
    DOCUMENT_PARSING = "document_parsing"
    GLEANING = "gleaning"
    GRAPH_ANALYSIS = "graph_analysis"
    GRAPH_EXTRACTION = "graph_extraction"
    GRAPH_RESOLUTION = "graph_resolution"
    INDEXING = "indexing"
    TEXT_CHUNKING = "text_chunking"
    TRANSLATION = "translation"


class ChunkingStrategy(str, Enum):
    SIMPLE = "simple"
    INTELLIGENT = "intelligent"


class Constants(str, Enum):
    ATTRIBUTE_PREFIX = "attr"
    DEFAULT_SUFFIX = "default"
    FILTERS = "filters"
    INDEX = "index"


class LanguageCode(str, Enum):
    DE = "de"
    EN = "en"
    ES = "es"
    FR = "fr"
    IT = "it"
    JA = "ja"
    KO = "ko"
    PT = "pt"
    RU = "ru"
    ZH = "zh"


class ResolutionMethod(str, Enum):
    MINHASH = "minhash"
    SEQUENCE_MATCHER = "sequence_matcher"


class RetrieverType(str, Enum):
    NEPTUNE = "neptune"
    OPENSEARCH = "opensearch"


class S3EncryptionType(str, Enum):
    NONE = "NONE"
    AES256 = "AES256"
    KMS = "aws:kms"


class EmbeddingModelId(str, Enum):
    EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"
    EMBED_V4 = "cohere.embed-v4:0"
    EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    TITAN_EMBED_V1 = "amazon.titan-embed-text-v1"
    TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
    # NOTE: add new models here


class LanguageModelId(str, Enum):
    CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_V3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_V3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_V4_5_HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
    CLAUDE_V3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_V3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_V3_7_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_V4_SONNET = "anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_V4_5_SONNET = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_V4_OPUS = "anthropic.claude-opus-4-20250514-v1:0"
    CLAUDE_V4_1_OPUS = "anthropic.claude-opus-4-1-20250805-v1:0"
    CLAUDE_V4_5_OPUS = "anthropic.claude-opus-4-5-20251101-v1:0"
    # NOTE: add new models here


class RerankModelId(str, Enum):
    AMAZON_RERANK_V1 = "amazon.rerank-v1:0"
    COHERE_RERANK_V3_5 = "cohere.rerank-v3-5:0"
    # NOTE: add new models here


class GuardrailConfig(BaseModel):
    """Amazon Bedrock Guardrails applied to every model invocation.

    Disabled by default; set ``identifier`` (and optionally ``version``) to
    enforce content/PII/grounding policies on prompts and completions — the
    WAF security-pillar control for a user-facing LLM RAG application.
    """

    identifier: str | None = Field(
        default=None,
        description="Bedrock guardrail identifier (ID or ARN). Enables guardrails when set.",
    )
    version: str = Field(
        default="DRAFT",
        min_length=1,
        description="Guardrail version to apply (e.g. 'DRAFT' or a published version number)",
    )
    trace: bool = Field(
        default=False,
        description="Emit guardrail trace details for observability/auditing",
    )

    @property
    def enabled(self) -> bool:
        return bool(self.identifier)


class BedrockConfig(BaseModel):
    region_name: str = Field(
        default="us-west-2", min_length=1, description="AWS Bedrock service region"
    )
    assumed_role_arn: str | None = Field(
        default=None, description="AWS assumed role ARN for Bedrock service"
    )
    enable_global_profile: bool = Field(
        default=True, description="Enable global profile for Bedrock service"
    )
    guardrail: GuardrailConfig = Field(
        default_factory=GuardrailConfig,
        description="Amazon Bedrock Guardrails configuration (disabled unless identifier set)",
    )


class NeptuneConfig(BaseModel):
    endpoint: str | None = Field(
        default=None, description="Neptune database endpoint URL"
    )
    port: int = Field(
        default=8182, ge=1, le=65535, description="Neptune database connection port"
    )
    use_iam: bool = Field(
        default=True, description="Enable IAM authentication for Neptune"
    )
    pool_size: int = Field(
        default=4,
        ge=1,
        description=(
            "Gremlin DriverRemoteConnection pool size (max concurrent in-flight "
            "requests over the websocket). Set >= indexing.neptune."
            "index_concurrency so concurrent write batches are not serialized on "
            "a single connection."
        ),
    )


class OpenSearchConfig(BaseModel):
    endpoint: str | None = Field(
        default=None, description="OpenSearch cluster endpoint URL"
    )
    port: int = Field(
        default=443, ge=1, le=65535, description="OpenSearch cluster connection port"
    )
    username: str | None = Field(
        default=None, description="OpenSearch authentication username"
    )
    password: SecretStr | None = Field(
        default=None,
        description="OpenSearch authentication password (masked in logs/repr)",
    )
    use_ssl: bool = Field(default=True, description="Enable SSL/TLS connection")
    verify_certs: bool = Field(default=True, description="Verify SSL/TLS certificates")
    use_iam: bool = Field(
        default=False, description="Enable IAM authentication for OpenSearch"
    )


class S3EncryptionConfig(BaseModel):
    encryption_type: S3EncryptionType = Field(
        default=S3EncryptionType.AES256, description="S3 server-side encryption method"
    )
    kms_key_id: str | None = Field(
        default=None,
        description="AWS KMS key ID for encryption (required when encryption type is KMS)",
    )

    @model_validator(mode="after")
    def validate_kms_key_id(self) -> "S3EncryptionConfig":
        if self.encryption_type == S3EncryptionType.KMS and not self.kms_key_id:
            raise ValueError("kms_key_id is required when encryption type is KMS")
        return self


class S3Config(BaseModel):
    bucket_name: str | None = Field(
        default=None, description="S3 bucket name for data storage"
    )
    encryption: S3EncryptionConfig = Field(
        default_factory=S3EncryptionConfig,
        description="S3 server-side encryption configuration",
    )


class DynamoDBConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Enable the DynamoDB document-status registry for incremental indexing",
    )
    table_name: str = Field(
        default="aws-graphrag-doc-status",
        min_length=1,
        description="DynamoDB table holding per-document status and lineage",
    )
    create_table_if_missing: bool = Field(
        default=True,
        description="Create the doc-status table on first use if it does not exist",
    )
    billing_mode: str = Field(
        default="PAY_PER_REQUEST",
        description="Billing mode used when auto-creating the table",
    )


class AWSConfig(BaseModel):
    region_name: str = Field(
        default="ap-northeast-2", min_length=1, description="AWS region name"
    )
    profile_name: str | None = Field(
        default=None, description="AWS profile name for authentication"
    )
    bedrock: BedrockConfig = Field(
        default_factory=BedrockConfig, description="AWS Bedrock service configuration"
    )
    neptune: NeptuneConfig = Field(
        default_factory=NeptuneConfig, description="AWS Neptune database configuration"
    )
    opensearch: OpenSearchConfig = Field(
        default_factory=OpenSearchConfig,
        description="AWS OpenSearch service configuration",
    )
    s3: S3Config = Field(
        default_factory=S3Config, description="AWS S3 storage configuration"
    )
    dynamodb: DynamoDBConfig = Field(
        default_factory=DynamoDBConfig,
        description="AWS DynamoDB document-status registry configuration",
    )


class FixingConfig(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable automatic fixing of malformed model responses"
    )
    fixing_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model for output correction",
    )


class DocumentParsingConfig(BaseModel):
    source_directory: str | Path = Field(
        default="source", description="Directory to load documents from"
    )
    target_directory: str | Path | None = Field(
        default=None, description="Directory to save parsed documents to"
    )
    index_value: str | None = Field(
        default=None, description="Value to index the parsed documents with"
    )


class ChunkingConfig(BaseModel):
    chunker_type: ChunkingStrategy = Field(
        default=ChunkingStrategy.INTELLIGENT,
        description="Text chunking strategy to use",
    )
    chunking_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU,
        description="Language model for intelligent chunking",
    )
    content_type: str = Field(
        default="markdown",
        pattern="^(text|html|markdown)$",
        description="Content format for processing",
    )
    min_chunk_size: int = Field(
        default=5000, ge=1, description="Minimum chunk size in characters"
    )
    max_chunk_size: int = Field(
        default=50000,
        ge=1,
        description="Maximum chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=500, ge=0, description="Chunk overlap in characters"
    )
    pre_chunk_size: int = Field(
        default=50000, ge=1, description="Pre-chunk size in characters"
    )
    pre_chunk_overlap: int = Field(
        default=500, ge=0, description="Pre-chunk overlap in characters"
    )
    fallback_chunk_size: int = Field(
        default=50000,
        ge=1,
        description="Fallback chunk size when intelligent chunking fails",
    )
    max_marker_miss_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Maximum allowed boundary marker miss rate",
    )

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "ChunkingConfig":
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")
        if self.chunk_overlap >= self.min_chunk_size:
            raise ValueError("chunk_overlap must be less than min_chunk_size")
        if self.pre_chunk_overlap >= self.pre_chunk_size:
            raise ValueError("pre_chunk_overlap must be less than pre_chunk_size")
        return self


class TranslationConfig(BaseModel):
    translation_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU,
        description="Language model for text translation",
    )
    target_language: LanguageCode = Field(
        default=LanguageCode.EN, description="Target language code for translation"
    )
    additional_target_languages: list[LanguageCode] | None = Field(
        default=None, description="Additional target languages for translation"
    )


class DescriptionSummarizationConfig(BaseModel):
    """LLM re-summarization of merged entity/relationship descriptions.

    Without this, merging an entity that appears in many chunks simply
    concatenates every chunk's description, so a popular entity's description
    grows unbounded — bloating prompts/embeddings and degrading quality. This
    mirrors MS GraphRAG ``summarize_descriptions`` and LightRAG
    ``_handle_entity_relation_summary``: after merge, any description over the
    token budget is re-summarized by a cheap LLM into one coherent, deduplicated
    text. Cheap items (below the threshold) skip the LLM entirely.
    """

    enabled: bool = Field(
        default=True,
        description="Re-summarize over-long merged descriptions with an LLM "
        "(parity with MS GraphRAG/LightRAG). When disabled, descriptions are only "
        "concatenated and may grow unbounded for frequently-mentioned entities.",
    )
    summary_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU,
        description="Language model for description summarization. Summarization "
        "is mechanical, so a cheap/fast model is the default.",
    )
    force_summary_threshold_tokens: int = Field(
        default=600,
        ge=1,
        description="Re-summarize a merged description only when its estimated "
        "token count exceeds this threshold. Descriptions at or below it are left "
        "as-is, so cheap entities never incur an LLM call.",
    )
    max_summary_tokens: int = Field(
        default=256,
        ge=1,
        description="Target length (in tokens) of the produced summary; injected "
        "into the summarization prompt as the budget.",
    )


class GraphExtractionConfig(BaseModel):
    extraction_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model for entity and relationship extraction",
    )
    max_entities_per_chunk: int = Field(
        default=50, ge=1, description="Maximum entities per text chunk"
    )
    max_relationships_per_chunk: int = Field(
        default=50, ge=1, description="Maximum relationships per text chunk"
    )
    entity_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for filtering entities. "
        "Entities below this threshold are excluded. Set to 0.0 to disable filtering.",
    )
    enable_confidence_extraction: bool = Field(
        default=True,
        description="Enable confidence score extraction from LLM. "
        "When disabled, all entities default to confidence 1.0.",
    )
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "PERSON: Names, individuals, roles, titles",
            "ORGANIZATION: Companies, institutions, departments, groups",
            "LOCATION: Places, addresses, geographic areas, facilities",
            "CONCEPT: Ideas, theories, methodologies, frameworks, principles",
            "OBJECT: Documents, tools, products, systems, technologies",
            "EVENT: Meetings, projects, activities, processes, incidents",
            "TEMPORAL: Dates, time periods, schedules, deadlines",
        ],
        description="Domain entity categories the extractor may use, injected "
        "into the extraction prompt's {entity_types} slot. Override this to adapt "
        "to a domain (e.g. ['GENE: ...', 'DISEASE: ...']) WITHOUT rewriting the "
        "whole prompt. Each item is 'LABEL: short description' (description "
        "optional). Empty list lets the model choose any relevant types.",
    )
    description_summarization: DescriptionSummarizationConfig = Field(
        default_factory=DescriptionSummarizationConfig,
        description="LLM re-summarization of over-long merged descriptions",
    )


class GleaningConfig(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable gleaning for improved extraction"
    )
    graph_refinement_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model for graph refinement",
    )
    max_rounds: int = Field(default=3, ge=1, description="Maximum gleaning rounds")
    max_entities_per_prompt: int = Field(
        default=100, ge=1, description="Maximum entities per gleaning prompt"
    )
    max_relationships_per_prompt: int = Field(
        default=100,
        ge=1,
        description="Maximum relationships per gleaning prompt",
    )
    convergence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Convergence threshold for early stopping",
    )
    quality_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Quality threshold for completion"
    )
    min_improvement_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum improvement required between rounds",
    )
    quality_completeness_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight of the LLM completeness score in the blended graph-quality score (accuracy gets the remainder)",
    )
    initial_quality_entity_scale: int = Field(
        default=50,
        ge=1,
        description="Entity count at which initial completeness saturates (scales the count-based seed quality estimate)",
    )
    initial_quality_relationship_scale: int = Field(
        default=100,
        ge=1,
        description="Relationship count at which initial completeness saturates",
    )
    convergence_change_scale: int = Field(
        default=20,
        ge=1,
        description="New entities+relationships per round treated as a full unit of change when scoring convergence",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "GleaningConfig":
        if self.convergence_threshold >= self.quality_threshold:
            raise ValueError(
                "convergence_threshold must be less than quality_threshold"
            )
        return self


class ClaimExtractionConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Enable claim (covariate) extraction. OFF by default: it "
        "incurs an LLM call per text unit. When ON, the local search strategy "
        "retrieves matching claims from the claims index and injects them into "
        "its context (mirroring MS GraphRAG covariates), and simple search "
        "includes the claims index in its sweep; claims subject/object are still "
        "not linked into the entity graph. The pipeline honors this flag "
        "(DataIngestionPipeline._initialize_stages).",
    )
    extraction_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model for claim extraction",
    )
    max_entities_per_prompt: int = Field(
        default=100,
        ge=0,
        description="Maximum entities per claim extraction prompt",
    )


class ProcessingConfig(BaseModel):
    max_concurrency: int = Field(
        default=20,
        ge=1,
        description="Maximum number of concurrent LLM operations within a batch. "
        "These stages are Bedrock-I/O-bound (CPU/memory near-idle), so this can be "
        "well above the CPU count.",
    )
    chunk_concurrency: int = Field(
        default=4,
        ge=1,
        description="How many mini-batch chunks to run concurrently. Overlaps "
        "chunks' Bedrock network waits instead of processing them serially; 1 = "
        "legacy strictly-serial behaviour.",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of items to process in each batch for optimal memory usage and performance",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retry attempts for failed operations",
    )
    ignore_errors: bool = Field(
        default=False,
        description="Ignore errors and continue processing",
    )
    deduplicate: bool = Field(
        default=False, description="Enable document deduplication"
    )
    resolution_method: ResolutionMethod = Field(
        default=ResolutionMethod.MINHASH,
        description="Entity/relationship resolution method",
    )
    similarity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for entity/relationship resolution",
    )
    document_parsing: DocumentParsingConfig = Field(
        default_factory=DocumentParsingConfig,
        description="Document parsing configuration",
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig, description="Text chunking configuration"
    )
    translation: TranslationConfig = Field(
        default_factory=TranslationConfig, description="Text translation configuration"
    )
    graph_extraction: GraphExtractionConfig = Field(
        default_factory=GraphExtractionConfig,
        description="Graph extraction configuration",
    )
    gleaning: GleaningConfig = Field(
        default_factory=GleaningConfig,
        description="Iterative extraction refinement configuration",
    )
    claim_extraction: ClaimExtractionConfig = Field(
        default_factory=ClaimExtractionConfig,
        description="Claim extraction configuration",
    )


class CentralityConfig(BaseModel):
    calculate_degree: bool = Field(
        default=True,
        description="Calculate degree centrality to measure node connectivity",
    )
    calculate_betweenness: bool = Field(
        default=True,
        description="Calculate betweenness centrality to identify bridge nodes",
    )
    calculate_pagerank: bool = Field(
        default=True,
        description="Calculate PageRank centrality to rank node importance",
    )
    calculate_closeness: bool = Field(
        default=False,
        description="Calculate closeness centrality to measure node proximity",
    )
    calculate_eigenvector: bool = Field(
        default=False,
        description="Calculate eigenvector centrality to measure influence based on connections",
    )
    pagerank_alpha: float = Field(
        default=0.85, ge=0.0, le=1.0, description="PageRank damping factor"
    )
    pagerank_max_iter: int = Field(
        default=100,
        ge=1,
        description="Maximum iterations for PageRank convergence",
    )
    betweenness_k: int | None = Field(
        default=None,
        ge=1,
        description="Sample size for betweenness calculation (None for all nodes)",
    )
    eigenvector_max_iter: int = Field(
        default=1000,
        ge=1,
        description="Maximum iterations for eigenvector centrality convergence",
    )
    eigenvector_tol: float = Field(
        default=1.0e-3,
        gt=0.0,
        description="Convergence tolerance for eigenvector centrality",
    )


class StatisticsConfig(BaseModel):
    calculate_density: bool = Field(
        default=True,
        description="Calculate graph density to measure network connectivity",
    )
    calculate_clustering: bool = Field(
        default=True,
        description="Calculate clustering coefficient to measure local connectivity",
    )
    calculate_diameter: bool = Field(
        default=False,
        description="Calculate graph diameter (computationally expensive)",
    )
    calculate_components: bool = Field(
        default=True,
        description="Analyze connected components to identify isolated subgraphs",
    )


class GraphAnalysisConfig(BaseModel):
    centrality: CentralityConfig = Field(
        default_factory=CentralityConfig,
        description="Node centrality metrics configuration",
    )
    statistics: StatisticsConfig = Field(
        default_factory=StatisticsConfig,
        description="Graph-level statistics configuration",
    )


class ReportGenerationConfig(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable automatic community report generation"
    )
    report_generation_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model for community report generation",
    )
    max_entities_per_report: int = Field(
        default=50, ge=1, description="Maximum entities per community report"
    )
    content_length: str = Field(
        default="medium",
        pattern="^(short|medium|long)$",
        description="Report length: short (6-8 paragraphs), medium (10-12), long (15-18)",
    )
    include_statistics: bool = Field(
        default=True, description="Include statistical metrics in community reports"
    )
    include_key_entities: bool = Field(
        default=True, description="Highlight key entities in community reports"
    )


class CommunityDetectionConfig(BaseModel):
    resolution: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Community detection resolution parameter (higher = fewer communities)",
    )
    random_state: int = Field(
        default=42, description="Random seed for reproducible results"
    )
    max_levels: int = Field(default=5, ge=1, description="Maximum hierarchy levels")
    trials: int = Field(
        default=3,
        ge=1,
        description="Number of independent Leiden runs to find best partition",
    )
    extra_forced_iterations: int = Field(
        default=2,
        ge=0,
        description="Additional optimization iterations after convergence",
    )
    min_community_size: int = Field(
        default=3,
        ge=1,
        description="Minimum nodes per community (smaller ones merged to neighbors)",
    )
    auto_resolution: bool = Field(
        default=True,
        description="Automatically find optimal resolution via modularity maximization",
    )
    auto_resolution_candidates: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
        description="Resolution values swept when auto_resolution is enabled; the "
        "one maximizing modularity is chosen.",
    )
    report_generation: ReportGenerationConfig = Field(
        default_factory=ReportGenerationConfig,
        description="Community report generation configuration",
    )


class VisualizationConfig(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable or disable the entire visualization pipeline."
    )
    outputs_directory: str | Path = Field(
        default="outputs/visualization",
        description="Directory to save visualization files.",
    )
    embedding_method: str = Field(
        default="node2vec",
        pattern="^(node2vec|none)$",
        description="Method for node embedding ('node2vec' or 'none').",
    )
    layout_method: str = Field(
        default="umap",
        pattern="^(umap|tsne|pca)$",
        description="Method for dimensionality reduction ('umap', 'tsne', 'pca').",
    )
    embeddings: dict[str, Any] = Field(
        default={
            "node2vec": {"dimensions": 128, "num_walks": 10, "walk_length": 80},
        },
        description="Configuration parameters for embedding methods.",
    )
    layout: dict[str, Any] = Field(
        default={
            "umap": {"n_neighbors": 15, "min_dist": 0.1},
            "tsne": {},
            "pca": {},
        },
        description="Configuration parameters for layout algorithms.",
    )
    interactive: dict[str, Any] = Field(
        default={"physics_enabled": False},
        description="Configuration for interactive visualization rendering.",
    )
    static: dict[str, Any] = Field(
        default={"figure_width": 900, "figure_height": 600},
        description="Configuration for static visualization rendering.",
    )


class GraphConfig(BaseModel):
    analysis: GraphAnalysisConfig = Field(
        default_factory=GraphAnalysisConfig, description="Graph analysis configuration"
    )
    community_detection: CommunityDetectionConfig = Field(
        default_factory=CommunityDetectionConfig,
        description="Community detection configuration",
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization configuration",
    )


class NeptuneIndexingConfig(BaseModel):
    entity_label_prefix: str = Field(
        default="Entity",
        min_length=1,
        max_length=100,
        description="Prefix for entity node labels in Neptune",
    )
    community_label_prefix: str = Field(
        default="Community",
        min_length=1,
        max_length=100,
        description="Prefix for community node labels in Neptune",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        description="Number of items to process per batch in Neptune operations",
    )
    index_concurrency: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of Neptune write batches to submit concurrently. 1 (default) "
            "preserves sequential indexing; >1 fans batches over a thread pool, "
            "multiplexed across the Gremlin connection pool (size aws.neptune."
            "pool_size to match). Each batch accumulates its own stats, merged "
            "on completion."
        ),
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retry attempts for failed Neptune operations",
    )
    retry_delay_seconds: int = Field(
        default=2, ge=0, description="Delay in seconds between retry attempts"
    )
    property_max_length: int = Field(
        default=1000,
        ge=1,
        description="Maximum character length for Neptune property values",
    )
    max_hops: int = Field(
        default=3,
        ge=1,
        description="Maximum number of hops for graph traversal queries",
    )
    max_results_per_hop: int = Field(
        default=50,
        ge=1,
        description="Maximum number of results to return per traversal hop",
    )
    min_entity_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum importance score for entities to be included in query results",
    )

    @model_validator(mode="after")
    def validate_retry_configuration(self) -> "NeptuneIndexingConfig":
        if self.max_retries > 0 and self.retry_delay_seconds == 0:
            raise ValueError(
                "retry_delay_seconds should be greater than 0 when max_retries is enabled"
            )
        return self


class OpenSearchIndexingConfig(BaseModel):
    text_units_index_prefix: str = Field(
        default="graphrag-text-units",
        min_length=1,
        max_length=100,
        description="Index name prefix for text unit documents",
    )
    entities_index_prefix: str = Field(
        default="graphrag-entities",
        min_length=1,
        max_length=100,
        description="Index name prefix for entity documents",
    )
    community_reports_index_prefix: str = Field(
        default="graphrag-community-reports",
        min_length=1,
        max_length=100,
        description="Index name prefix for community report documents",
    )
    relationships_index_prefix: str = Field(
        default="graphrag-relationships",
        min_length=1,
        max_length=100,
        description="Index name prefix for relationship documents (LightRAG global retrieval)",
    )
    claims_index_prefix: str = Field(
        default="graphrag-claims",
        min_length=1,
        max_length=100,
        description="Index name prefix for claim (covariate) documents",
    )
    hybrid_search_pipeline_name: str = Field(
        default="graphrag-hybrid-search-pipeline",
        min_length=1,
        max_length=100,
        description="OpenSearch pipeline name for combining lexical and vector search results",
    )
    default_analyzer: str = Field(
        default="standard",
        min_length=1,
        description="OpenSearch text analyzer used when the language has no specific mapping",
    )
    language_analyzers: dict[str, str] = Field(
        default_factory=lambda: {"en": "english", "ko": "nori"},
        description="Maps a language code to its OpenSearch text analyzer; extend without code changes",
    )
    embedding_model_id: EmbeddingModelId = Field(
        default=EmbeddingModelId.TITAN_EMBED_V2,
        description="Embedding model identifier for vector generation",
    )
    embedding_dimension: int | None = Field(
        default=None,
        ge=1,
        description="Vector embedding dimension (automatically detected from model if None)",
    )
    refresh_after_batch: bool = Field(
        default=True,
        description="Whether to refresh OpenSearch indices after each batch operation for immediate visibility",
    )
    persist_embedding_cache: bool = Field(
        default=False,
        description="Persist the content-hash embedding cache to S3 so unchanged "
        "text is not re-embedded across separate runs/phases (each Fargate phase "
        "is a fresh process). Requires aws.s3.bucket_name. Off by default.",
    )
    embedding_cache_s3_key: str = Field(
        default="embedding-cache/cache.json",
        description="S3 key (under the cache bucket) for the persisted embedding "
        "cache when persist_embedding_cache is enabled.",
    )
    max_query_size: int = Field(
        default=100,
        ge=1,
        description="Maximum number of hits returned per OpenSearch query",
    )
    terms_batch_size: int = Field(
        default=150,
        ge=1,
        description="Batch size when partitioning large terms filters to stay under the clause limit",
    )
    max_total_clauses: int = Field(
        default=600,
        ge=1,
        description="Upper bound on boolean clauses per query (must stay under the cluster's max_clause_count)",
    )
    reserved_clauses: int = Field(
        default=300,
        ge=0,
        description="Clause budget reserved for non-filter query parts when batching terms filters",
    )
    index_settings: dict[str, Any] = Field(
        default_factory=lambda: {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
        },
        description="OpenSearch index configuration settings for performance tuning",
    )
    vector_search: dict[str, Any] = Field(
        default_factory=lambda: {
            "engine": "nmslib",
            "space_type": "cosinesimil",
            "ef_construction": 128,
            "m": 24,
            "ef_search": 100,
        },
        description="HNSW algorithm parameters for approximate nearest neighbor search",
    )


class IndexingConfig(BaseModel):
    reset: bool = Field(
        default=False,
        description="Whether to clear all existing indexed data before starting new indexing",
    )
    additional_suffix: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Optional additional suffix to append to index names for isolation",
    )
    cross_run_merge: bool = Field(
        default=False,
        description="On incremental (delta) runs, read existing graph entities/"
        "relationships and union them with the delta (description/text_unit_ids/"
        "frequency/weight) before upsert, instead of overwriting. Requires a graph "
        "adapter that supports read-back; off by default.",
    )
    neptune: NeptuneIndexingConfig = Field(
        default_factory=NeptuneIndexingConfig,
        description="Configuration settings for Neptune graph database indexing",
    )
    opensearch: OpenSearchIndexingConfig = Field(
        default_factory=OpenSearchIndexingConfig,
        description="Configuration settings for OpenSearch vector and text indexing",
    )


class HybridConfig(BaseModel):
    lexical_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for lexical search results"
    )
    vector_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for vector search results"
    )

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "HybridConfig":
        total = self.lexical_weight + self.vector_weight
        if math.isclose(total, 0.0):
            raise ValueError(
                "The sum of vector_weight and lexical_weight cannot be zero."
            )

        if not math.isclose(total, 1.0):
            self.lexical_weight /= total
            self.vector_weight /= total
        return self


class FusionConfig(BaseModel):
    method: FusionMethod = Field(
        default=FusionMethod.RRF,
        description="Method used for fusing search results from multiple sources",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF parameter k for reciprocal rank fusion algorithm",
    )
    diversity_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "MMR lambda for diversity filtering: score = lambda*relevance - "
            "(1-lambda)*max_similarity. 1.0 = pure relevance (no diversity), "
            "0.0 = maximum diversity. Lower values penalize redundant results "
            "more strongly. Filtering is skipped at 1.0 (no diversity benefit)."
        ),
    )
    fusion_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "graph_entities": 1.0,
            "text_units": 1.0,
            "lightrag_entities": 1.0,
            "lightrag_relationships": 1.0,
            "lightrag_chunks": 1.0,
            "opensearch_all": 1.0,
            "opensearch_community_reports": 1.0,
            "opensearch_candidate_community_reports": 1.0,
            "opensearch_expanded_community_reports": 1.0,
            "results": 1.0,
        },
        description=(
            "Per-source-bucket weights applied during weighted fusion "
            "(FusionMethod.WEIGHTED). Keys are the retrieval source buckets "
            "emitted by the search strategies (graph_entities, text_units, "
            "lightrag_entities/relationships/chunks, opensearch_all, the "
            "global-search community-report buckets, and drift's 'results'). A "
            "bucket without a key defaults to 1.0. Unused by the default RRF fusion."
        ),
    )


class RerankingConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Enable reranking of search results for improved relevance",
    )
    rerank_model_id: RerankModelId = Field(
        default=RerankModelId.COHERE_RERANK_V3_5,
        description="Bedrock reranking model identifier",
    )
    top_k: int = Field(
        default=100,
        ge=1,
        description="Maximum number of top results to rerank",
    )


class GlobalSearchConfig(BaseModel):
    community_relevance_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU,
        description="Language model used for scoring community relevance to search queries",
    )
    map_reduce_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU,
        description="Language model used for map-reduce summarization operations",
    )
    max_communities: int = Field(
        default=10,
        ge=1,
        description="Maximum number of communities to consider during global search",
    )
    relevance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score required for community selection",
    )
    use_dynamic_selection: bool = Field(
        default=True,
        description="Enable dynamic community selection based on query characteristics",
    )
    enable_map_reduce: bool = Field(
        default=True,
        description="Enable map-reduce processing for large-scale summarization",
    )
    max_text_units: int = Field(
        default=100,
        ge=1,
        description="Upper bound on text units pulled into the global-search "
        "context (caps context size regardless of top_k).",
    )
    map_reduce_min_results: int = Field(
        default=3,
        ge=1,
        description="Minimum community results before map-reduce synthesis is "
        "applied; below this the results are returned directly.",
    )
    graph_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout (seconds) for the Neptune community-graph retrieval "
        "in global search; raise for very large graphs or slow clusters.",
    )


class LocalSearchConfig(BaseModel):
    entity_frequency_threshold: int = Field(
        default=20,
        ge=1,
        description="Drop graph-expanded entities appearing in more than this "
        "many text units (too generic to be discriminative for local search).",
    )


class DriftSearchConfig(BaseModel):
    enable_query_refinement: bool = Field(
        default=True,
        description="Enable iterative query refinement during drift search",
    )
    enable_keyword_extraction: bool = Field(
        default=True,
        description="Enable automatic keyword extraction from search results",
    )
    query_refinement_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU,
        description="Language model used for refining search queries based on intermediate results",
    )
    keyword_expansion_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU,
        description="Language model used for expanding keywords from discovered entities",
    )
    convergence_assessment_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU,
        description="Language model used for assessing search convergence",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum number of drift search iterations allowed",
    )
    initial_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of initial community reports to retrieve as search seeds",
    )
    summary_length: int = Field(
        default=5,
        ge=1,
        description="Maximum number of result summaries to include in query evolution",
    )
    n_entities: int = Field(
        default=5,
        ge=1,
        description="Number of top entities to extract for keyword expansion",
    )
    convergence_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Convergence score threshold for early termination",
    )
    improvement_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum improvement ratio required to continue iterations",
    )


class TokenManagerConfig(BaseModel):
    max_context_tokens: int = Field(
        default=200000,
        ge=1024,
        description="Maximum number of tokens allowed in the context window for optimal performance",
    )
    token_count_cache_size: int = Field(
        default=1024,
        ge=1,
        description="Maximum number of entries in the LRU cache for Bedrock token counting",
    )


class LightRAGSearchConfig(BaseModel):
    raw_query_fallback_max_len: int = Field(
        default=50,
        ge=0,
        description=(
            "When dual-level keyword extraction yields no keywords, a query whose "
            "length is below this (0 disables the gate) falls back to using the raw "
            "query as a low-level keyword, mirroring LightRAG. Longer queries skip "
            "the fallback to avoid an over-broad graph scan."
        ),
    )


class SearchConfig(BaseModel):
    translation_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU,
        description="Language model identifier used for translating queries into the target language",
    )
    entity_extraction_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model identifier used for extracting named entities from user queries",
    )
    strategy_selection_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model identifier used for automatically selecting the optimal search strategy",
    )
    context_building_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model identifier used for building and structuring contextual information",
    )
    answer_generation_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model identifier used for generating final answers from retrieved context",
    )
    hybrid: HybridConfig = Field(
        default_factory=HybridConfig, description="Hybrid search configuration"
    )
    fusion: FusionConfig = Field(
        default_factory=FusionConfig, description="Search result fusion configuration"
    )
    reranking: RerankingConfig = Field(
        default_factory=RerankingConfig, description="Reranking configuration"
    )
    global_search: GlobalSearchConfig = Field(
        default_factory=GlobalSearchConfig, description="Global search configuration"
    )
    local_search: LocalSearchConfig = Field(
        default_factory=LocalSearchConfig, description="Local search configuration"
    )
    drift_search: DriftSearchConfig = Field(
        default_factory=DriftSearchConfig, description="Drift search configuration"
    )
    lightrag_search: LightRAGSearchConfig = Field(
        default_factory=LightRAGSearchConfig,
        description="LightRAG dual-level keyword search configuration",
    )
    token_manager: TokenManagerConfig = Field(
        default_factory=TokenManagerConfig, description="Token management configuration"
    )


class MemoryConfig(BaseModel):
    max_conversations: int = Field(
        default=100,
        ge=1,
        description="Maximum number of conversations to keep in memory",
    )
    max_messages_per_conversation: int = Field(
        default=20,
        ge=1,
        description="Maximum number of messages to store per conversation",
    )
    max_conversation_age_hours: int = Field(
        default=168,
        ge=1,
        description="Maximum age of a conversation in hours before being eligible for cleanup",
    )


class CacheChunkingConfig(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable cache data chunking for large datasets"
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of items per cache chunk",
    )
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        description="Maximum file size in MB before triggering chunking",
    )


class CacheConfig(BaseModel):
    ttl_seconds: int | None = Field(
        default=86400,
        ge=1,
        description="Cache entry time-to-live in seconds (None for no expiration)",
    )
    chunking: CacheChunkingConfig = Field(
        default_factory=CacheChunkingConfig,
        description="Configuration for cache data chunking behavior",
    )


class LoggingConfig(BaseModel):
    level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level",
    )
    log_format: str = Field(
        default="structured",
        pattern="^(structured|plain)$",
        description="Log format style",
    )
    log_to_file: bool = Field(default=True, description="Enable file logging")
    log_file_path: str = Field(
        default="logs/log.txt",
        min_length=1,
        max_length=255,
        description="Log file path",
    )


class CustomPromptConfig(BaseModel):
    graph_extraction_system: str | None = Field(
        default=None,
        description="Custom system prompt for entity and relationship extraction from text",
    )
    graph_extraction_human: str | None = Field(
        default=None,
        description="Custom human prompt for entity and relationship extraction from text",
    )
    claim_extraction_system: str | None = Field(
        default=None,
        description="Custom system prompt for extracting claims and assertions from documents",
    )
    claim_extraction_human: str | None = Field(
        default=None,
        description="Custom human prompt for extracting claims and assertions from documents",
    )
    description_summarization_system: str | None = Field(
        default=None,
        description="Custom system prompt for re-summarizing over-long merged entity/relationship descriptions",
    )
    description_summarization_human: str | None = Field(
        default=None,
        description="Custom human prompt for re-summarizing over-long merged entity/relationship descriptions",
    )
    graph_refinement_system: str | None = Field(
        default=None,
        description="Custom system prompt for improving and refining extracted graph entities and relationships",
    )
    graph_refinement_human: str | None = Field(
        default=None,
        description="Custom human prompt for improving and refining extracted graph entities and relationships",
    )
    community_report_system: str | None = Field(
        default=None,
        description="Custom system prompt for generating community analysis reports",
    )
    community_report_human: str | None = Field(
        default=None,
        description="Custom human prompt for generating community analysis reports",
    )
    answer_generation_system: str | None = Field(
        default=None,
        description="Custom system prompt for generating answers from knowledge graph",
    )
    answer_generation_human: str | None = Field(
        default=None,
        description="Custom human prompt for generating answers from knowledge graph",
    )
    context_building_system: str | None = Field(
        default=None,
        description="Custom system prompt for building context from knowledge graph",
    )
    context_building_human: str | None = Field(
        default=None,
        description="Custom human prompt for building context from knowledge graph",
    )
    entity_extraction_system: str | None = Field(
        default=None,
        description="Custom system prompt for extracting named entities from user queries",
    )
    entity_extraction_human: str | None = Field(
        default=None,
        description="Custom human prompt for extracting named entities from user queries",
    )
    keywords_extraction_system: str | None = Field(
        default=None,
        description="Custom system prompt for dual-level (high/low) keyword extraction (LightRAG)",
    )
    keywords_extraction_human: str | None = Field(
        default=None,
        description="Custom human prompt for dual-level (high/low) keyword extraction (LightRAG)",
    )
    corpus_profile_system: str | None = Field(
        default=None,
        description="Custom system prompt for corpus profiling during prompt tuning",
    )
    corpus_profile_human: str | None = Field(
        default=None,
        description="Custom human prompt for corpus profiling during prompt tuning",
    )
    extraction_examples_system: str | None = Field(
        default=None,
        description="Custom system prompt for generating few-shot extraction examples during prompt tuning",
    )
    extraction_examples_human: str | None = Field(
        default=None,
        description="Custom human prompt for generating few-shot extraction examples during prompt tuning",
    )
    keyword_expansion_system: str | None = Field(
        default=None,
        description="Custom system prompt for expanding search queries with relevant keywords",
    )
    keyword_expansion_human: str | None = Field(
        default=None,
        description="Custom human prompt for expanding search queries with relevant keywords",
    )
    query_refinement_system: str | None = Field(
        default=None,
        description="Custom system prompt for refining queries based on intermediate results",
    )
    query_refinement_human: str | None = Field(
        default=None,
        description="Custom human prompt for refining queries based on intermediate results",
    )
    strategy_selection_system: str | None = Field(
        default=None,
        description="Custom system prompt for selecting search strategy based on user query",
    )
    strategy_selection_human: str | None = Field(
        default=None,
        description="Custom human prompt for selecting search strategy based on user query",
    )


class EvaluationConfig(BaseModel):
    outputs_directory: str | Path = Field(
        default="outputs/evaluation",
        description="Directory to save evaluation results",
    )
    embedding_model_id: EmbeddingModelId = Field(
        default=EmbeddingModelId.TITAN_EMBED_V2,
        description="Embedding model identifier for evaluation",
    )
    evaluation_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET,
        description="Language model identifier used for evaluation",
    )
    enabled_evaluators: list[EvaluatorType] = Field(
        default=[EvaluatorType.LANGCHAIN, EvaluatorType.RAGAS],
        description="List of evaluator types to enable for this evaluation run.",
    )
    langchain_metrics: list[EvaluationMetricType] = Field(
        default=[
            EvaluationMetricType.CORRECTNESS,
            EvaluationMetricType.PARTIAL_CORRECTNESS,
        ],
        description="Specific evaluation metrics to calculate when using LangChain evaluator.",
    )
    ragas_metrics: list[EvaluationMetricType] = Field(
        default=[
            EvaluationMetricType.ANSWER_CORRECTNESS,
            EvaluationMetricType.ANSWER_RELEVANCY,
            EvaluationMetricType.CONTEXT_PRECISION,
            EvaluationMetricType.CONTEXT_RECALL,
            EvaluationMetricType.FAITHFULNESS,
        ],
        description="Specific evaluation metrics to calculate when using RAGAS evaluator.",
    )
    max_context_tokens: int = Field(
        default=8192,
        ge=1,
        description="Maximum number of tokens allowed in context for evaluation processing",
    )
    save_detailed_results: bool = Field(
        default=True,
        description="Whether to save detailed evaluation results and reports.",
    )
    save_summary_only: bool = Field(
        default=False,
        description="Whether to save only the evaluation summary (excludes detailed results).",
    )


class Config(BaseModel):
    aws: AWSConfig = Field(
        default_factory=AWSConfig, description="AWS services configuration"
    )
    fixing: FixingConfig = Field(
        default_factory=FixingConfig, description="Output fixing configuration"
    )
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig, description="Text processing configuration"
    )
    graph: GraphConfig = Field(
        default_factory=GraphConfig, description="Graph analysis configuration"
    )
    indexing: IndexingConfig = Field(
        default_factory=IndexingConfig, description="Data indexing configuration"
    )
    search: SearchConfig = Field(
        default_factory=SearchConfig, description="Search configuration"
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Memory system configuration"
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig, description="Caching system configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging system configuration"
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation configuration"
    )
    custom_prompts: CustomPromptConfig = Field(
        default_factory=CustomPromptConfig, description="Custom prompt configuration"
    )


class PipelineConfig(BaseModel):
    stages_enabled: dict[PipelineStageType, bool] = Field(
        default_factory=lambda: dict.fromkeys(PipelineStageType, True),
        description="Specifies which pipeline stages are enabled.",
    )
    cache_enabled: bool = Field(
        default=True, description="Enables caching for pipeline stage outputs."
    )
    local_directory: str | Path = Field(
        default="cache",
        description="The local filesystem path for cache storage.",
    )
    s3_sync_enabled: bool = Field(
        default=False,
        description="Enables synchronization of the cache with an S3 bucket.",
    )
    s3_bucket_name: str | None = Field(
        default=None, description="The S3 bucket name for cache synchronization."
    )
    s3_prefix: str = Field(
        default="pipeline-runs",
        min_length=1,
        description="The S3 key prefix for storing cache objects.",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="The number of items to process in a single batch.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="The maximum number of retries for a failed operation.",
    )
    continue_on_error: bool = Field(
        default=False,
        description="If true, the pipeline continues execution even if a stage fails.",
    )
    force_rebuild: bool = Field(
        default=False,
        description="If true, ignores any existing cache and rebuilds all outputs.",
    )
    pipeline_id: str | None = Field(
        default=None,
        min_length=1,
        description="The ID of a previous pipeline run to resume.",
    )
    resume_from_stage: str | None = Field(
        default=None,
        min_length=1,
        description="The name of the stage from which to resume execution.",
    )
