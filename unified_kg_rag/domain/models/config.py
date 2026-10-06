# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    SecretStr,
    StringConstraints,
    model_validator,
)

from .evaluation import EvaluationMetricType, EvaluatorType
from .retrieval import FusionMethod, SearchStrategy


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
    """Server-side encryption header sent on cache uploads.

    ``BUCKET_DEFAULT`` sends no SSE header, so S3 applies the bucket's default
    encryption (SSE-S3 at minimum, or the bucket's SSE-KMS CMK). ``AES256`` /
    ``aws:kms`` force a per-object header that OVERRIDES the bucket default.
    The legacy value ``"NONE"`` is still accepted and means ``BUCKET_DEFAULT``:
    S3 encrypts every new object, so it never meant "unencrypted".
    """

    BUCKET_DEFAULT = "BUCKET_DEFAULT"
    AES256 = "AES256"
    KMS = "aws:kms"

    @classmethod
    def _missing_(cls, value: object) -> "S3EncryptionType | None":
        return cls.BUCKET_DEFAULT if value == "NONE" else None


class EmbeddingModelId(str, Enum):
    EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"
    EMBED_V4 = "cohere.embed-v4:0"
    EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    TITAN_EMBED_V1 = "amazon.titan-embed-text-v1"
    TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
    # NOTE: add new models here


class LanguageModelId(str, Enum):
    """Named constants for the language models with a curated capability record.

    Model-id config fields accept ANY Bedrock model id string, so a new model
    can be tried without a release; these members are only convenience names
    (and the keys of the adapter's capability table). ``str()`` and
    f-strings render the bare id, so a member can be used wherever an id
    string is expected.
    """

    # Claude 4.7+ / 5.x ids carry no date suffix and no ':0' revision, unlike
    # earlier generations. Cross-region resolution still applies the same
    # 'us.'/'apac.'/'global.' inference-profile prefixes.
    CLAUDE_V5_5_SONNET = "anthropic.claude-sonnet-5-5"
    CLAUDE_V5_5_OPUS = "anthropic.claude-opus-5-5"
    CLAUDE_V5_SONNET = "anthropic.claude-sonnet-5"
    CLAUDE_V5_OPUS = "anthropic.claude-opus-5"
    CLAUDE_V4_8_OPUS = "anthropic.claude-opus-4-8"
    CLAUDE_V4_7_OPUS = "anthropic.claude-opus-4-7"
    CLAUDE_V4_6_OPUS = "anthropic.claude-opus-4-6-v1"
    CLAUDE_V4_6_SONNET = "anthropic.claude-sonnet-4-6"
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
    # OpenAI proprietary GPT models (open-weight gpt-oss is intentionally not
    # offered). Ids use dotted versions and are served through the Converse API
    # on cross-region inference profiles only.
    GPT_V6_1_SOL = "openai.gpt-6.1-sol"
    GPT_V6_ASTRA = "openai.gpt-6-astra"
    GPT_V6_SOL = "openai.gpt-6-sol"
    GPT_V6_LUNA = "openai.gpt-6-luna"
    GPT_V5_6_SOL = "openai.gpt-5.6-sol"
    GPT_V5_6_TERRA = "openai.gpt-5.6-terra"
    GPT_V5_6_LUNA = "openai.gpt-5.6-luna"
    GPT_V5_5 = "openai.gpt-5.5"
    GPT_V5_4 = "openai.gpt-5.4"
    # NOTE: add new models here

    def __str__(self) -> str:
        return str(self.value)


ModelTier = Literal["default", "fast"]
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]

# The two shipped model tiers. Every per-role model field declares one of them
# and inherits aws.bedrock.default_model_id / aws.bedrock.fast_model_id unless
# set explicitly, so changing a tier default is a one-line edit here.
DEFAULT_MODEL_ID: str = LanguageModelId.CLAUDE_V5_5_SONNET.value
FAST_MODEL_ID: str = LanguageModelId.CLAUDE_V4_5_HAIKU.value
_TIER_DEFAULT_MODEL_IDS: dict[str, str] = {
    "default": DEFAULT_MODEL_ID,
    "fast": FAST_MODEL_ID,
}
_MODEL_TIER_KEY = "model_tier"


def _coerce_model_id(value: Any) -> Any:
    # A LanguageModelId member (or any str enum) validates as its bare id.
    if isinstance(value, Enum):
        value = value.value
    return value.strip() if isinstance(value, str) else value


# Any Bedrock model id (or cross-region inference-profile id) as a plain str.
BedrockModelId = Annotated[
    str, BeforeValidator(_coerce_model_id), StringConstraints(min_length=1)
]


def role_model_field(tier: ModelTier, description: str) -> Any:
    """A per-role model-id field that inherits its tier's model when unset."""
    return Field(
        default=_TIER_DEFAULT_MODEL_IDS[tier],
        description=f"{description.rstrip('.')}. Defaults to aws.bedrock.{tier}_model_id.",
        json_schema_extra={_MODEL_TIER_KEY: tier},
    )


def _apply_model_tiers(model: BaseModel, tiers: dict[str, str]) -> BaseModel:
    """Copy of ``model`` whose unset role-model fields follow ``tiers``.

    Walks nested sections recursively. A field the user set explicitly is in
    ``model_fields_set`` and is never touched.
    """
    updates: dict[str, Any] = {}
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        extra = field.json_schema_extra
        tier = extra.get(_MODEL_TIER_KEY) if isinstance(extra, dict) else None
        if isinstance(tier, str):
            if name not in model.model_fields_set and value != tiers[tier]:
                updates[name] = tiers[tier]
        elif isinstance(value, BaseModel):
            resolved = _apply_model_tiers(value, tiers)
            if resolved is not value:
                updates[name] = resolved
    if not updates:
        return model
    copied = model.model_copy()
    # Write through __dict__ so inherited values stay out of model_fields_set:
    # the section keeps following its tier if it is reused in another Config.
    copied.__dict__.update(updates)
    return copied


class ModelPurpose(str, Enum):
    """Why a language model is being created, so per-path policies can differ.

    Callers pass it to the LLM factory (``get_model(model_purpose=...)``). The
    Bedrock factory uses it to scope guardrails (``GuardrailConfig.apply_to``):
    a user-facing guardrail that anonymizes PII or blocks instruction-like text
    is appropriate for queries, but corrupts graph extraction over a corpus.
    Unspecified means ``QUERY`` so an unmarked call site keeps the guarded,
    conservative behaviour.
    """

    QUERY = "query"
    INGESTION = "ingestion"
    EVALUATION = "evaluation"


class RerankModelId(str, Enum):
    AMAZON_RERANK_V1 = "amazon.rerank-v1:0"
    COHERE_RERANK_V3_5 = "cohere.rerank-v3-5:0"
    # NOTE: add new models here


class GuardrailConfig(BaseModel):
    """Amazon Bedrock Guardrails applied to language-model invocations.

    Disabled by default; set ``identifier`` (and optionally ``version``) to
    enforce content/PII/grounding policies on prompts and completions — the
    WAF security-pillar control for a user-facing LLM RAG application.

    ``apply_to`` scopes the guardrail. The default ``"query"`` guards only the
    user-facing query path (answer generation, query refinement and
    query-time entity/keyword extraction, global/DRIFT map-reduce). Ingestion
    (chunking, translation, graph extraction, gleaning, claims, description
    summarization, community reports) and evaluation judges run unguarded,
    because a PII-anonymizing guardrail rewrites names to placeholders (every
    person merges into one entity) and a prompt-attack filter blocks
    instruction-like corpus text, which silently yields empty extractions.
    ``"all"`` restores the previous guard-everything behaviour.
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
    apply_to: Literal["query", "all"] = Field(
        default="query",
        description=(
            "Which model invocations the guardrail is attached to: 'query' "
            "(user-facing query path only; ingestion and evaluation models run "
            "unguarded) or 'all' (every model, including ingestion)"
        ),
    )

    @property
    def enabled(self) -> bool:
        return bool(self.identifier)

    def applies_to(self, purpose: ModelPurpose) -> bool:
        """Whether a model created for ``purpose`` gets this guardrail."""
        if not self.enabled:
            return False
        return self.apply_to == "all" or purpose is ModelPurpose.QUERY


class TransientRetryConfig(BaseModel):
    """Bounded retry on transient Bedrock errors that botocore does not retry.

    botocore's retry modes do not retry Bedrock's HTTP 424
    ``ModelErrorException`` (or ``ModelNotReadyException``), so without this a
    single transient model fault fails a user query or drops an item from the
    vector index. Applied to every embedding request and to query-time LLM
    chains; ingestion LLM chains are retried by ``BatchProcessor`` instead.
    Only transient errors are retried; validation and access errors still fail
    fast.
    """

    max_attempts: int = Field(
        default=5,
        ge=1,
        description="Total attempts per call, including the first (1 disables "
        "the retry)",
    )
    base_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Backoff ceiling for the first retry; doubles per attempt "
        "with equal jitter",
    )
    max_delay_seconds: float = Field(
        default=16.0,
        ge=0.0,
        description="Upper bound on a single backoff delay in seconds",
    )
    max_total_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Wall-clock retry budget per call; no retry starts once the "
        "next backoff would cross it",
    )


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
    default_model_id: BedrockModelId = Field(
        default=DEFAULT_MODEL_ID,
        description=(
            "Model for every role on the 'default' tier (extraction, gleaning, "
            "claims, community reports, output fixing, query entity/keyword "
            "extraction, context building, answer generation, evaluation). Any "
            "Bedrock model id; a role's own *_model_id overrides it."
        ),
    )
    fast_model_id: BedrockModelId = Field(
        default=FAST_MODEL_ID,
        description=(
            "Model for every role on the 'fast' tier (chunking, translation, "
            "description summarization, global/DRIFT search steps, query "
            "translation, strategy routing). Any Bedrock model id; a role's own "
            "*_model_id overrides it."
        ),
    )
    default_max_output_tokens: int | None = Field(
        default=16384,
        ge=1,
        description=(
            "max_tokens sent with each LLM request, clamped to the model's "
            "maximum. Bedrock reserves input + max_tokens against the "
            "tokens-per-minute quota when a request starts, so sending the "
            "model maximum (128K on Claude 5.x) throttles concurrent calls "
            "long before real usage does. Thinking tokens count toward it. "
            "Prompts with long outputs (graph/claim extraction, gleaning, "
            "community reports, document translation) declare a higher floor "
            "that wins over this value. null sends the model maximum."
        ),
    )
    model_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-model capability overrides keyed by model id, e.g. "
            "{'amazon.nova-pro-v1:0': {'context_window_size': 300000}}. Keys "
            "are capability-record fields (context_window_size, "
            "max_output_tokens, supports_thinking, supports_sampling_params, "
            "supports_prompt_caching, ...). Use it for a model without a "
            "curated record, or to correct one, without a release."
        ),
    )
    enable_1m_context: bool = Field(
        default=False,
        description=(
            "Opt into the 1M-token context window on models that support it as "
            "a beta (Claude Sonnet 4 / 4.5, Opus 4 / 4.1 / 4.5). Off by default: "
            "long-context requests are billed at a premium on those models. "
            "Claude 5 has a native 1M window and ignores this flag. Enabling it "
            "also widens the derived retrieval context budget."
        ),
    )
    default_effort: EffortLevel = Field(
        default="high",
        description=(
            "Reasoning effort for calls on the default_model_id. Sent as "
            "output_config.effort to Anthropic adaptive-thinking models (Claude "
            "4.6+), where it replaces the fixed thinking token budget, and as "
            "reasoning.effort to OpenAI GPT models. 'xhigh'/'max' are only "
            "accepted by some models; lower levels trade depth for cost/latency."
        ),
    )
    fast_effort: EffortLevel = Field(
        default="low",
        description=(
            "Reasoning effort for calls on the fast_model_id (when it differs "
            "from default_model_id). The shipped fast model (Claude Haiku 4.5) "
            "does not reason on these calls, so this only takes effect when "
            "fast_model_id is an adaptive-thinking or GPT model."
        ),
    )
    effort: EffortLevel | None = Field(
        default=None,
        description=(
            "Deprecated alias for default_effort, kept for existing configs. "
            "Used only when default_effort is not set."
        ),
    )
    guardrail: GuardrailConfig = Field(
        default_factory=GuardrailConfig,
        description="Amazon Bedrock Guardrails configuration (disabled unless identifier set)",
    )
    transient_retry: TransientRetryConfig = Field(
        default_factory=TransientRetryConfig,
        description="Retry for embedding and query-time LLM calls on transient "
        "Bedrock errors",
    )

    @model_validator(mode="after")
    def _apply_legacy_effort(self) -> "BedrockConfig":
        # Mirror the legacy key into default_effort so a dumped config reloads
        # with the same effort. Written through __dict__ to stay out of
        # model_fields_set, as tier_effort reads that to pick the winner.
        if self.effort is not None and "default_effort" not in self.model_fields_set:
            self.__dict__["default_effort"] = self.effort
        return self

    def model_tier(self, model_id: str) -> ModelTier:
        """Tier a call on ``model_id`` belongs to, for picking its effort.

        Only a model that is the fast tier's model (and not also the default
        tier's) counts as fast; anything else, including a role pinned to its
        own model, gets the default tier's effort.
        """
        model_id = str(model_id).strip()
        if model_id == self.fast_model_id and model_id != self.default_model_id:
            return "fast"
        return "default"

    def tier_effort(self, tier: ModelTier) -> EffortLevel:
        """Configured effort for ``tier``, honouring the legacy ``effort`` key.

        Resolved on read as well as at validation, so assigning ``effort`` or
        ``default_effort`` after construction behaves the same way.
        """
        if tier == "fast":
            return self.fast_effort
        if self.effort is not None and "default_effort" not in self.model_fields_set:
            return self.effort
        return self.default_effort


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
    sigv4_service_name: str = Field(
        default="es",
        description=(
            "SigV4 signing service for IAM auth: 'es' for a managed OpenSearch "
            "domain, 'aoss' for an OpenSearch Serverless collection. A wrong "
            "value fails auth (often surfacing as zero search hits)."
        ),
    )


class S3EncryptionConfig(BaseModel):
    encryption_type: S3EncryptionType = Field(
        default=S3EncryptionType.BUCKET_DEFAULT,
        description=(
            "S3 server-side encryption for cache uploads: BUCKET_DEFAULT (send no "
            "header; the bucket's default encryption, e.g. its KMS CMK, applies), "
            "AES256 (force SSE-S3) or aws:kms (force SSE-KMS with kms_key_id). "
            "NONE is a legacy alias of BUCKET_DEFAULT."
        ),
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
        default="unified-kg-rag-on-aws-doc-status",
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
    fixing_model_id: BedrockModelId = role_model_field(
        "default",
        description="Language model for output correction",
    )


class DocumentParsingConfig(BaseModel):
    source_directory: str | Path = Field(
        default="source", description="Directory to load documents from"
    )
    target_directory: str | Path | None = Field(
        default=None,
        description=(
            "Directory to export each parsed document to as <stem>.json for "
            "inspection. None = no export. Must not be the source directory."
        ),
    )
    index_value: str | None = Field(
        default=None, description="Value to index the parsed documents with"
    )
    source_scope: str | None = Field(
        default=None,
        description=(
            "Identity of the corpus source for incremental indexing: a run only "
            "treats registry documents of its own index suffix AND source scope "
            "as deleted. None = the resolved source directory. Set it when the "
            "local directory is a staging copy, e.g. the S3 URI a container "
            "syncs from (env GRAPHRAG_SOURCE_SCOPE)."
        ),
    )


class ChunkingConfig(BaseModel):
    chunker_type: ChunkingStrategy = Field(
        default=ChunkingStrategy.INTELLIGENT,
        description="Text chunking strategy to use",
    )
    chunking_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model for intelligent chunking",
    )
    content_type: str = Field(
        default="markdown",
        pattern="^(text|html|markdown)$",
        description="Content format for processing",
    )
    min_chunk_size: int = Field(
        default=1000,
        ge=1,
        description=(
            "Minimum chunk size in characters; shorter pieces are merged into a "
            "neighbour. Default 1,000 (~250 English tokens) keeps headings and "
            "short trailing paragraphs from becoming chunks of their own."
        ),
    )
    max_chunk_size: int = Field(
        default=8000,
        ge=1,
        description=(
            "Maximum chunk size in characters. Every chunk is embedded whole and "
            "reranked as one document, so it must fit the embedding model's "
            "input (Titan Text Embeddings V2: 8,192 tokens) and the reranker's "
            "per-document limit (Cohere Rerank 3.5: 4,096 tokens), or its tail "
            "is never embedded or reranked. Default 8,000 fits Titan V2 even at "
            "~1 character per token (CJK scripts) and is ~2K tokens of English, "
            "inside the reranker limit."
        ),
    )
    chunk_overlap: int = Field(
        default=500, ge=0, description="Chunk overlap in characters"
    )
    pre_chunk_size: int = Field(
        default=50000,
        ge=1,
        description=(
            "Pre-chunk size in characters: the window the intelligent chunker's "
            "LLM picks boundaries in. Not embedded itself (its chunks are capped "
            "by max_chunk_size), so it is sized for the chunking model's prompt."
        ),
    )
    pre_chunk_overlap: int = Field(
        default=500, ge=0, description="Pre-chunk overlap in characters"
    )
    fallback_chunk_size: int = Field(
        default=4800,
        ge=1,
        description=(
            "Target chunk size for the size-based splitter (the simple chunker, "
            "and intelligent chunking when the LLM fails or a chunk exceeds "
            "max_chunk_size). Default 4,800 characters ~= 1,200 English tokens, "
            "the default chunk size of MS GraphRAG and LightRAG."
        ),
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
    enabled: bool = Field(
        default=True,
        description="Run the translation pipeline stage. Defaults to True to "
        "preserve existing behavior. Even when enabled, translation is skipped "
        "as a no-op when source_language equals target_language and there are no "
        "additional_target_languages (e.g. an English-only corpus with an English "
        "target pays no LLM cost).",
    )
    translation_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model for text translation",
    )
    source_language: LanguageCode = Field(
        default=LanguageCode.EN,
        description="Predominant source language of the corpus. Used only for the "
        "same-language no-op skip; translation is content-driven so this need not "
        "be exact for mixed corpora.",
    )
    target_language: LanguageCode = Field(
        default=LanguageCode.EN, description="Target language code for translation"
    )
    additional_target_languages: list[LanguageCode] | None = Field(
        default=None, description="Additional target languages for translation"
    )

    @property
    def is_noop(self) -> bool:
        """True when translation would do nothing useful (same-language skip)."""
        return (
            self.source_language == self.target_language
            and not self.additional_target_languages
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
    summary_model_id: BedrockModelId = role_model_field(
        "fast",
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


class EntityGroundingConfig(BaseModel):
    """Provenance grounding for extracted entities.

    The extraction prompt asks the model to emit a verbatim ``source_text`` span
    for every entity. When enabled, an entity whose evidence span does not occur
    in its source chunk (verbatim or by token overlap) is treated as a
    hallucination — the model invented it from domain priors rather than the
    document — and is either dropped or confidence-penalized. OFF by default:
    it relies on the model emitting spans and is conservative, but turning it on
    is the defense against corpus-absent entities polluting the index.
    """

    enabled: bool = Field(
        default=False,
        description="Reject/penalize entities whose source_text evidence span is "
        "not found in the source chunk (hallucination guard). Requires the "
        "extraction model to emit a source_text span per entity.",
    )
    action: str = Field(
        default="drop",
        pattern="^(drop|penalize)$",
        description="What to do with an ungrounded entity: 'drop' removes it; "
        "'penalize' multiplies its confidence by penalty_factor so the "
        "confidence threshold can filter it without hard deletion.",
    )
    penalty_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence multiplier applied to ungrounded entities when "
        "action='penalize'.",
    )
    min_span_tokens: int = Field(
        default=4,
        ge=1,
        description="Evidence spans shorter than this (after normalization) are "
        "treated as grounded — too short to judge, so the gate never deletes "
        "short legitimate names on weak signal.",
    )
    min_overlap_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="When the span is not a verbatim substring, the fraction of "
        "its tokens that must appear in the chunk to still count as grounded "
        "(handles light paraphrase / whitespace edits).",
    )


class GraphExtractionConfig(BaseModel):
    extraction_model_id: BedrockModelId = role_model_field(
        "default",
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
        "Entities below this threshold are excluded. Set to 0.0 to disable filtering. "
        "Compared against the normalized confidence: the prompt's 1-10 score / 10 "
        "(an integer 1 -> 0.1; decimal values <= 1 are taken as fractions).",
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
    entity_grounding: EntityGroundingConfig = Field(
        default_factory=EntityGroundingConfig,
        description="Provenance grounding guard against hallucinated entities",
    )


class GleaningConfig(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable gleaning for improved extraction"
    )
    graph_refinement_model_id: BedrockModelId = role_model_field(
        "default",
        description="Language model for graph refinement",
    )
    max_rounds: int = Field(
        default=3,
        ge=1,
        description=(
            "Maximum gleaning rounds. Each round after the first re-sends only "
            "the text units that gained entities or relationships in the "
            "previous round, so later rounds cost less than the first. Set 1 "
            "for MS GraphRAG's single-gleaning default when ingestion cost "
            "matters more than graph recall."
        ),
    )
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
        description="New entities+relationships per gleaned text unit treated as a full unit of change when scoring convergence",
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
    extraction_model_id: BedrockModelId = role_model_field(
        "default",
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
    betweenness_auto_sample_threshold: int = Field(
        default=2000,
        ge=1,
        description="When betweenness_k is None (exact) AND the graph has more "
        "nodes than this, automatically switch to sampled betweenness (using "
        "this value as the pivot-sample size) instead of exact all-pairs "
        "shortest paths. Exact betweenness is O(V*E) and stalls on large real "
        "graphs; sampling keeps the analysis phase bounded. Raise this to force "
        "exact computation on larger graphs.",
    )
    betweenness_seed: int = Field(
        default=42,
        description="Random seed for sampled betweenness (only used when "
        "betweenness_k is set). Fixed for reproducibility — sampled betweenness "
        "picks random pivots, so without a seed centrality is non-deterministic.",
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
    report_generation_model_id: BedrockModelId = role_model_field(
        "default",
        description="Language model for community report generation",
    )
    max_entities_per_report: int = Field(
        default=50, ge=1, description="Maximum entities per community report"
    )
    max_report_context_tokens: int = Field(
        default=4000,
        ge=1,
        description="Token budget for the entity/relationship context packed "
        "into a community report prompt. Entities are degree-sorted (most "
        "connected first) and packed up to this budget, so truncation drops "
        "the least-important entities first. max_entities_per_report remains an "
        "additional upper bound on entity count.",
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
    enable_sub_community_rollup: bool = Field(
        default=True,
        description="When a parent community's raw entity/relationship context "
        "would exceed max_report_context_tokens, substitute summaries of its "
        "already-generated child sub-community reports for the lowest-priority "
        "raw context (MS GraphRAG parity), instead of simply truncating. "
        "Requires bottom-up per-level report generation (level 0 = finest "
        "first). Set false to keep the flat, truncate-on-overflow behaviour.",
    )


class CommunityDetectionConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description=(
            "Run Leiden community detection + LLM community-report generation "
            "during indexing. Required for GraphRAG global/drift search. Disable "
            "for a lighter, LightRAG-only ingestion (mix/hybrid/naive need only "
            "entities, relationships, and the relationship vector index — no "
            "communities), which skips the Leiden pass and all report LLM calls."
        ),
    )
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
    auto_resolution_max_nodes: int = Field(
        default=10000,
        ge=1,
        description="Skip the auto_resolution sweep (which runs one full Leiden "
        "partition + modularity computation per candidate, ~10x the base cost, "
        "at every hierarchy level) when the graph exceeds this many nodes, and "
        "use the fixed `resolution` instead. Prevents the sweep from dominating "
        "the analysis phase on large graphs. Raise to force the sweep on bigger "
        "graphs.",
    )
    report_generation: ReportGenerationConfig = Field(
        default_factory=ReportGenerationConfig,
        description="Community report generation configuration",
    )


class VisualizationConfig(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable or disable the entire visualization pipeline."
    )
    outputs_directory: str | Path | None = Field(
        default=None,
        description="Directory to save visualization files. When unset, "
        "ingestion writes to '<cache.local_directory>/<pipeline_id>/visualization' "
        "so visualization_data.json is synced to S3 with the cache; other callers "
        "fall back to 'outputs/visualization'.",
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
        default_factory=dict,
        description="Embedding parameters for node embedding. Only "
        "'bedrock_model_id' is honored (node embeddings come from Bedrock; the "
        "embedding dimensionality is determined by the chosen model). When "
        "unset, the OpenSearch embedding model is reused.",
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
            "on completion. NOTE (real-AWS finding): >1 can trigger Neptune "
            "ConcurrentModificationException when batches touch overlapping "
            "vertices/properties; the retry (exponential backoff) recovers but "
            "the conflict churn can make a large run SLOWER than sequential. "
            "Prefer 1 unless you have measured a net win on your data/cluster."
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
        default=4000,
        ge=1,
        description=(
            "Maximum character length for Neptune property values (entity and "
            "relationship descriptions are read back from Neptune into the "
            "local/DRIFT context). Keep it well above the longest description "
            "that is not re-summarized: processing.description_summarization "
            "only triggers above force_summary_threshold_tokens (600 tokens, "
            "~2,400 English characters), so the former 1,000-character default "
            "cut such descriptions mid-sentence."
        ),
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
    entity_importance_source: Literal["rank", "degree", "none"] = Field(
        default="rank",
        description=(
            "Graph-expansion relevance is the mean of an entity's importance and "
            "its proximity to the seeds. 'rank' uses the indexed entity `rank`, "
            "'degree' counts the entity's edges at query time (MS GraphRAG ranks "
            "entities by degree), and each is normalized by the largest value in "
            "the same result. 'none' gives every entity a neutral 0.5, the "
            "previous behaviour (it read an `importance` property that entity "
            "vertices never store), so expansion ranks by proximity alone."
        ),
    )
    traversal_fetch_multiplier: int = Field(
        default=3,
        ge=1,
        description=(
            "Graph expansion fetches top_k * retrieval_multiplier * this many "
            "entities, ranks them by relevance, then keeps the top "
            "top_k * retrieval_multiplier. 1 = the previous behaviour: the "
            "traversal's limit cut entities in Neptune's arbitrary emit order, "
            "before any ranking."
        ),
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
    build_relationship_vector_index: bool = Field(
        default=True,
        description="Build the OpenSearch relationship VECTOR index (embeds "
        "Relationship.description). Required only for LightRAG high-level keyword "
        "retrieval (mix/hybrid). Set false for a GraphRAG-only deployment "
        "(auto/drift/global/local/simple use community reports + entities + the "
        "Neptune graph, never this index) to skip the relationship embeddings. "
        "The Neptune relationship EDGES are always indexed regardless — both "
        "methodologies traverse them for graph expansion; this toggles only the "
        "vector index. Symmetric with graph.community_detection.enabled.",
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
            # lucene HNSW natively supports the cosinesimil space type on all
            # supported OpenSearch versions. faiss HNSW does NOT: it accepts only
            # l2/innerproduct until OpenSearch 2.19 (cosine support) / 2.18
            # (auto-normalization), so faiss + cosinesimil is rejected at index
            # creation with a mapper_parsing_exception on 2.13. lucene HNSW caps
            # at 1024 dimensions, which covers the default Titan Embed V2 (1024);
            # for >1024-dim models set engine: faiss with space_type:
            # innerproduct (normalized embeddings) instead. Kept in sync with the
            # index mapping default in OpenSearchIndexer.
            "engine": "lucene",
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
        default=True,
        description="On incremental (delta) runs, read existing graph entities/"
        "relationships and union them with the delta (description/text_unit_ids/"
        "frequency/weight) before upsert, instead of overwriting. Requires a graph "
        "adapter that supports read-back (an adapter without it degrades to "
        "overwrite). On by default: overwriting replaces an entity shared with "
        "unchanged documents by its delta-only description and text_unit_ids.",
    )
    cross_run_fuzzy_merge: bool = Field(
        default=False,
        description="Extend cross_run_merge with fuzzy entity resolution: match "
        "delta entities against existing entity names by similarity (not just "
        "exact normalized name), so near-duplicate surface forms ('Acme Corp' vs "
        "'Acme Corporation') converge across incremental runs the way a full "
        "rebuild groups them. Costs an entity-name projection per delta run; "
        "requires cross_run_merge and a graph adapter that supports read-back. "
        "Off by default.",
    )
    neptune: NeptuneIndexingConfig = Field(
        default_factory=NeptuneIndexingConfig,
        description="Configuration settings for Neptune graph database indexing",
    )
    opensearch: OpenSearchIndexingConfig = Field(
        default_factory=OpenSearchIndexingConfig,
        description="Configuration settings for OpenSearch vector and text indexing",
    )
    max_failure_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum tolerated per-index-type write failure rate before "
        "the indexing stage is marked FAILED. A fully-failed index type (0 "
        "successes) always fails regardless of this value; this additionally "
        "catches PARTIAL failures (e.g. most relationship edges dropped) that "
        "would otherwise be reported as a successful run. Set to 1.0 to disable "
        "the partial-failure gate (only total failures fail the stage).",
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
            "(1-lambda)*max_similarity, with relevance min-max normalized to "
            "[0, 1] over the fused candidates. 1.0 = pure relevance (no diversity), "
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
            "Per-source-bucket weights, applied by both fusion methods: RRF "
            "scales each bucket's 1/(rrf_k + rank) term and weighted fusion "
            "scales its scores. Keys are the retrieval source buckets "
            "emitted by the search strategies (graph_entities, text_units, "
            "lightrag_entities/relationships/chunks, opensearch_all, the "
            "global-search community-report buckets, and drift's 'results'). A "
            "bucket without a key defaults to 1.0."
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
    community_relevance_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model used for scoring community relevance to search queries",
    )
    map_reduce_model_id: BedrockModelId = role_model_field(
        "fast",
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
        default=False,
        description=(
            "Score every retrieved community report for query relevance with an "
            "LLM (one call per report) before map-reduce. Off by default: the map "
            "step already rates each report's key points for the query and drops "
            "the irrelevant ones, so this pre-filter repeats that judgement at the "
            "cost of max_communities extra LLM calls per query (MS GraphRAG's "
            "dynamic community selection is also off by default). Useful mainly "
            "with enable_map_reduce=false; when off, communities are ranked by "
            "their indexed rank and rating."
        ),
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
    map_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model used for the map step of MS GraphRAG "
        "map-reduce (rates community-report key points 0-100). Rating is cheap, "
        "so a fast/cheap model (e.g. Haiku) is the sensible default.",
    )
    map_batch_size: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of community reports packed into one map-step LLM call; map "
            "calls are fanned over batches concurrently via BatchProcessor. "
            "Default 5 keeps a map prompt near MS GraphRAG's 12K-token map "
            "context (data_max_tokens) for reports of 1-2K tokens, and halves the "
            "map calls per query compared with 2. Lower it if reports are long."
        ),
    )
    map_relevance_threshold: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Drop map-step key points whose 0-100 relevance score is at "
        "or below this threshold before ranking (0 keeps any positive score).",
    )
    max_map_reduce_tokens: int = Field(
        default=8000,
        ge=128,
        description="Token budget for the ranked key points packed into the "
        "reduce step; the highest-scored points are taken until this budget is "
        "reached, so the reduce LLM synthesizes from focused, ranked evidence.",
    )
    reduce_with_llm: bool = Field(
        default=False,
        description=(
            "Run the REDUCE step as its own LLM call that writes a summary of the "
            "packed key points, which the answer model then rewrites. Off by "
            "default: the packed points go straight to the answer model as one "
            "ranked context section, saving one LLM call per query and avoiding "
            "a second synthesis that can drop facts or assert that the summaries "
            "lack them. When off, the degraded path (map calls that failed) also "
            "passes the unrated reports through instead of summarizing them. "
            "true = the previous two-step behaviour."
        ),
    )
    reserve_report_slots: bool = Field(
        default=True,
        description=(
            "When fusing the selected community reports with their text units, "
            "reserve max_communities * retrieval_multiplier slots for the reports "
            "and cap the text units at text_unit_slots, and keep the synthesized "
            "map-reduce item in addition to (not inside) that width. false = the "
            "previous flat top_k cut, which reranked reports and chunks together "
            "and often kept mostly chunks, then cut to top_k after prepending the "
            "synthesized item (dropping one more result)."
        ),
    )
    text_unit_slots: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Text-unit slots next to the reserved report slots "
            "(reserve_report_slots). null = the query's top_k."
        ),
    )


class LocalSearchQuotaConfig(BaseModel):
    """Reserved fusion slots per section type, as multiples of the query's top_k.

    Without reserved slots a single type (usually entities, which arrive with many
    near-tie fusion scores) takes the whole flat top_k cut and crowds out the
    document chunks that carry the answer. MS GraphRAG's local search assembles
    its context the same way — proportional per-section budgets rather than one
    ranked list — with ``top_k_entities``/``top_k_relationships`` defaulting to 10.

    Each entry is ``max(multiplier * top_k, floor)``, so the shape holds as the
    caller's top_k scales while a small top_k still admits enough of each type.
    """

    text_multiplier: float = Field(
        default=2.0, gt=0.0, description="Chunk slots as a multiple of top_k"
    )
    text_floor: int = Field(
        default=20, ge=1, description="Minimum chunk slots regardless of top_k"
    )
    entity_multiplier: float = Field(
        default=1.0, gt=0.0, description="Entity slots as a multiple of top_k"
    )
    entity_floor: int = Field(
        default=10, ge=1, description="Minimum entity slots regardless of top_k"
    )
    relationship_multiplier: float = Field(
        default=1.0, gt=0.0, description="Relationship slots as a multiple of top_k"
    )
    relationship_floor: int = Field(
        default=10, ge=1, description="Minimum relationship slots regardless of top_k"
    )
    community_multiplier: float = Field(
        default=0.5, gt=0.0, description="Community report slots as a multiple of top_k"
    )
    community_floor: int = Field(
        default=2, ge=1, description="Minimum community report slots"
    )
    claim_multiplier: float = Field(
        default=0.5, gt=0.0, description="Claim slots as a multiple of top_k"
    )
    claim_floor: int = Field(default=2, ge=1, description="Minimum claim slots")


class LocalSearchConfig(BaseModel):
    entity_frequency_threshold: int = Field(
        default=20,
        ge=1,
        description="Drop graph-expanded entities appearing in more than this "
        "many text units (too generic to be discriminative for local search).",
    )
    type_quota: LocalSearchQuotaConfig = Field(
        default_factory=LocalSearchQuotaConfig,
        description="Reserved fusion slots per section type",
    )
    include_bridge_relationships: bool = Field(
        default=True,
        description=(
            "Also fetch the relationships incident to the graph-expanded "
            "entities, edges between two of those entities first (MS GraphRAG "
            "local's in-network relationships). These bridge edges carry the hops "
            "of a multi-hop chain, which the relationship vector query alone "
            "often misses. Needs the relationship index "
            "(indexing.opensearch.build_relationship_vector_index). false = the "
            "previous behaviour: only the vector query on the relationship index."
        ),
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
    query_refinement_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model used for refining search queries based on intermediate results",
    )
    keyword_expansion_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model used for expanding keywords from discovered entities",
    )
    enable_llm_convergence: bool = Field(
        default=False,
        description=(
            "Ask an LLM after each iteration whether DRIFT has converged and stop "
            "early when its score reaches convergence_threshold. Off by default: "
            "it adds one LLM call per iteration, while the deterministic checks "
            "(low unique-result gain, improvement_threshold) already stop the "
            "loop, and max_iterations bounds it."
        ),
    )
    convergence_assessment_model_id: BedrockModelId = role_model_field(
        "fast",
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
    summary_char_limit: int = Field(
        default=200,
        ge=1,
        description="Per-result character budget when summarizing retrieved "
        "content into the evolved DRIFT query. Caps how much of each result is "
        "fed back into the next iteration's query (was a hardcoded cutoff).",
    )
    n_entities: int = Field(
        default=5,
        ge=1,
        description="Number of top entities to extract for keyword expansion",
    )
    convergence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "LLM convergence score (0-1) at or above which DRIFT stops early "
            "(enable_llm_convergence only). Default 0.8 is the lower bound of the "
            "ConvergenceAssessmentPrompt's 'convergence achieved' band; lower "
            "values stop while the prompt still advises exploring."
        ),
    )
    improvement_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum improvement ratio required to continue iterations",
    )
    enable_primer: bool = Field(
        default=False,
        description="Restructure DRIFT into MS GraphRAG's primer -> follow-up "
        "flow: a HyDE primer drafts a hypothetical answer from the seed "
        "community reports and decomposes the query into specific follow-up "
        "sub-queries, each run as its own search iteration (instead of carrying "
        "one mutating query forward). Off by default — the primer adds one LLM "
        "call up front and turns the iteration budget into follow-up breadth.",
    )
    primer_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model for the DRIFT primer (HyDE answer + "
        "follow-up decomposition) when enable_primer is set.",
    )
    primer_follow_ups: int = Field(
        default=3,
        ge=1,
        description="Number of follow-up sub-queries the DRIFT primer emits; "
        "each becomes one follow-up search iteration (capped by max_iterations).",
    )


class ContextTypeBudgetConfig(BaseModel):
    """Per-section-type share of the answer context window.

    Both upstream methodologies split the context window into per-type
    sub-budgets and pack each type independently, so document chunks (which hold
    the multi-hop answer) are guaranteed representation instead of competing in a
    flat greedy fill of near-tie fusion scores: MS GraphRAG's ``mixed_context``
    packs community / entity / text-unit sections against independent budgets
    (``community_prop`` 0.15, ``text_unit_prop`` 0.5 by default), and LightRAG's
    ``_apply_token_truncation`` trims entities / relations / chunks against
    separate token limits (6000 / 8000 out of a 30000 window).

    Shares are relative weights, not fractions: they are renormalized over the
    section types actually present in a query's candidate set, so an absent type
    does not shrink the usable window. Raise ``text`` to favour verbatim source
    passages, raise ``community`` to favour high-level synthesis.
    """

    text: float = Field(
        default=0.50, ge=0.0, description="Relative share for document chunk sections"
    )
    entity: float = Field(
        default=0.20, ge=0.0, description="Relative share for graph entity sections"
    )
    relationship: float = Field(
        default=0.13,
        ge=0.0,
        description="Relative share for graph relationship sections",
    )
    community: float = Field(
        default=0.10,
        ge=0.0,
        description="Relative share for community report sections",
    )
    claim: float = Field(
        default=0.07,
        ge=0.0,
        description="Relative share for claim (covariate) sections",
    )
    general: float = Field(
        default=0.10,
        ge=0.0,
        description="Relative share for untyped sections — global search's "
        "map-reduce synthesis lands here.",
    )

    @model_validator(mode="after")
    def validate_at_least_one_positive_share(self) -> "ContextTypeBudgetConfig":
        if not any(
            share > 0.0
            for share in (
                self.text,
                self.entity,
                self.relationship,
                self.community,
                self.claim,
                self.general,
            )
        ):
            raise ValueError(
                "At least one context type budget share must be greater than zero; "
                "all-zero shares would leave every answer context empty."
            )
        return self


class TokenManagerConfig(BaseModel):
    max_context_tokens: int | None = Field(
        default=30_000,
        ge=1024,
        description=(
            "Prompt-side retrieval context budget in tokens. Default 30,000 "
            "matches upstream LightRAG's total context budget "
            "(DEFAULT_MAX_TOTAL_TOKENS); MS GraphRAG uses 12,000. Deriving the "
            "budget from the answer model's window gives ~785K tokens on a 1M "
            "model, so it never binds: the per-type budgets and priority "
            "ordering never trim anything and every query pays for the whole "
            "retrieved set. The value is always clamped to what the answer "
            "model can accept alongside its output reservation. Set null to "
            "derive the budget from the window instead."
        ),
    )
    context_window_headroom_ratio: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of the answer model's context window held back when "
            "deriving max_context_tokens, covering the system prompt, the "
            "question, conversation history, and tokenizer estimation error."
        ),
    )
    token_count_cache_size: int = Field(
        default=1024,
        ge=1,
        description="Maximum number of entries in the LRU cache for Bedrock token counting",
    )
    type_budgets: ContextTypeBudgetConfig = Field(
        default_factory=ContextTypeBudgetConfig,
        description="Per-section-type shares of the context window",
    )
    min_truncated_section_tokens: int = Field(
        default=64,
        ge=1,
        description="A section truncated below this many tokens is dropped rather "
        "than emitted as an unusable fragment of evidence.",
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
    kg_stream_top_k: int = Field(
        default=40,
        ge=1,
        description=(
            "Width of the entity and relationship vector queries (LightRAG's "
            "`QueryParam.top_k`, default 40). Separate from the chunk width: "
            "collapsing both onto the caller's single top_k starves the graph "
            "streams. Acts as a floor — a caller asking for a larger top_k gets it."
        ),
    )
    chunk_stream_top_k: int = Field(
        default=20,
        ge=1,
        description=(
            "Width of the chunk stream, and the cap on the merged chunk pool "
            "(LightRAG's `QueryParam.chunk_top_k`, default 20). Acts as a floor "
            "against the caller's top_k."
        ),
    )
    related_chunk_number: int = Field(
        default=5,
        ge=1,
        description=(
            "Chunks allocated per matched entity/relationship when following "
            "`text_unit_ids` lineage (LightRAG's `related_chunk_number`). Slots are "
            "distributed on a decreasing gradient so the last match still gets one."
        ),
    )
    enable_graph_expansion: bool = Field(
        default=False,
        description=(
            "hybrid/mix: also expand the entity hits and relationship endpoints "
            "through the Neptune graph (indexing.neptune.max_hops) and add the "
            "neighbourhood as entity candidates. Off by default: upstream LightRAG "
            "has no multi-hop traversal (its graph context is the matched items "
            "plus their one-hop incident edges and endpoint entities, which this "
            "strategy already retrieves), and the extra Neptune round trip adds "
            "latency and entities less related to the query."
        ),
    )


class SearchConfig(BaseModel):
    translation_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model identifier used for translating queries into the target language",
    )
    entity_extraction_model_id: BedrockModelId = role_model_field(
        "default",
        description="Language model identifier used for extracting named entities from user queries",
    )
    strategy_selection_model_id: BedrockModelId = role_model_field(
        "fast",
        description=(
            "Language model identifier used for automatically selecting the "
            "optimal search strategy (AUTO). A fast model by default: routing "
            "is a one-word classification that runs before retrieval, so its "
            "latency is added to every AUTO query."
        ),
    )
    auto_routable_strategies: list[SearchStrategy] = Field(
        default_factory=lambda: [
            SearchStrategy.LOCAL,
            SearchStrategy.MIX,
            SearchStrategy.GLOBAL,
            SearchStrategy.DRIFT,
        ],
        min_length=1,
        description=(
            "Strategies the AUTO router may pick. Default: local, mix, global, "
            "drift. simple is left out because the graph strategies already "
            "cover its direct lookups with graph context; mix (LightRAG) is "
            "included for multi-hop fact chains, which it serves from the same "
            "index. Every strategy stays selectable explicitly."
        ),
    )
    context_building_model_id: BedrockModelId = role_model_field(
        "default",
        description="Language model identifier used for building and structuring contextual information",
    )
    answer_generation_model_id: BedrockModelId = role_model_field(
        "default",
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
    library_levels: dict[str, str] = Field(
        default_factory=lambda: {
            "langchain_aws": "WARNING",
            "botocore": "WARNING",
            "urllib3": "WARNING",
        },
        description="Per-logger levels for chatty third-party libraries "
        "(logger name -> level)",
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
    drift_primer_system: str | None = Field(
        default=None,
        description="Custom system prompt for the DRIFT primer (HyDE answer + follow-up decomposition)",
    )
    drift_primer_human: str | None = Field(
        default=None,
        description="Custom human prompt for the DRIFT primer (HyDE answer + follow-up decomposition)",
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
    global_map_system: str | None = Field(
        default=None,
        description="Custom system prompt for the global-search map step "
        "(rate community-report key points 0-100 for relevance)",
    )
    global_map_human: str | None = Field(
        default=None,
        description="Custom human prompt for the global-search map step "
        "(rate community-report key points 0-100 for relevance)",
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
    evaluation_model_id: BedrockModelId = role_model_field(
        "default",
        description="Language model identifier used for evaluation",
    )
    enabled_evaluators: list[EvaluatorType] = Field(
        default=[
            EvaluatorType.LANGCHAIN,
            EvaluatorType.RAGAS,
            EvaluatorType.ANSWER_MATCH,
            EvaluatorType.RETRIEVAL,
            EvaluatorType.GRAPH_AWARE,
        ],
        description=(
            "List of evaluator types to enable for this evaluation run. The "
            "deterministic ones (answer_match, retrieval, graph_aware) are free "
            "and skip a query that lacks their dataset fields, so they are on "
            "by default alongside the LLM judges (langchain, ragas)."
        ),
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
    judge_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = Field(
        default="low",
        description=(
            "Reasoning effort for the LLM judge (RAGAS and LangChain evaluators) "
            "on adaptive-thinking models (Claude 4.7+). Judge prompts are short "
            "extraction/classification calls, so a low effort keeps each metric "
            "well inside the RAGAS timeout. null inherits the effort of the "
            "judge model's tier (aws.bedrock.default_effort by default). "
            "Ignored by models without adaptive thinking."
        ),
    )
    ragas_timeout: int = Field(
        default=300,
        ge=1,
        description=(
            "RAGAS RunConfig.timeout in seconds: the budget for scoring ONE "
            "metric on ONE sample, covering all of its LLM calls and retries. "
            "A timed-out job yields NaN for that metric. RAGAS' own default "
            "(180s) is too tight for thinking judges."
        ),
    )
    ragas_max_contexts: int | None = Field(
        default=20,
        ge=1,
        description=(
            "Maximum number of top-ranked retrieved contexts per sample scored "
            "by RAGAS context_precision, applied before max_context_tokens. "
            "Context precision makes one judge call per context, so its cost "
            "scales with the context count; strategies that report 100+ "
            "sources time out otherwise. Only that metric is capped (it becomes "
            "context_precision@N); faithfulness and context_recall see every "
            "context within max_context_tokens, and the answer model's context "
            "is unaffected. null disables the cap."
        ),
    )
    ragas_max_workers: int = Field(
        default=8,
        ge=1,
        description=(
            "RAGAS RunConfig.max_workers: concurrent metric jobs. Lower it if "
            "Bedrock throttles the judge model (throttled calls burn the "
            "per-job timeout)."
        ),
    )
    ragas_max_retries: int = Field(
        default=3,
        ge=1,
        description=(
            "RAGAS RunConfig.max_retries: total attempts per judge call (RAGAS "
            "applies it as stop_after_attempt, with exponential backoff). "
            "Attempts count against ragas_timeout."
        ),
    )
    retrieval_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Cutoff k for the retrieval evaluator's hit@k / recall@k: the number "
            "of top-ranked reported sources compared with reference_sources."
        ),
    )
    save_detailed_results: bool = Field(
        default=True,
        description="Whether to save detailed evaluation results and reports.",
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

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_llm_retry(cls, data: Any) -> Any:
        """Map the former ``search.llm_retry`` key to ``aws.bedrock.transient_retry``.

        An explicit ``aws.bedrock.transient_retry`` wins over the legacy key.
        """
        if not isinstance(data, dict):
            return data
        search = data.get("search")
        if not isinstance(search, dict) or "llm_retry" not in search:
            return data
        search = {k: v for k, v in search.items() if k != "llm_retry"}
        aws = dict(data.get("aws") or {})
        bedrock = dict(aws.get("bedrock") or {})
        bedrock.setdefault("transient_retry", data["search"]["llm_retry"])
        aws["bedrock"] = bedrock
        return {**data, "search": search, "aws": aws}

    @model_validator(mode="after")
    def _inherit_tier_models(self) -> "Config":
        """Point every role-model field the user did not set at its tier model."""
        tiers = {
            "default": self.aws.bedrock.default_model_id,
            "fast": self.aws.bedrock.fast_model_id,
        }
        for name in type(self).model_fields:
            section = getattr(self, name)
            if isinstance(section, BaseModel):
                resolved = _apply_model_tiers(section, tiers)
                if resolved is not section:
                    self.__dict__[name] = resolved
        return self


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
