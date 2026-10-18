# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import functools
import logging
import math
import re
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from string import Formatter
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from .evaluation import EvaluationMetricType, EvaluatorType
from .provider_config import (
    BEDROCK,
    CROSS_ENCODER,
    GREMLIN,
    Backend,
    EmbeddingProviderConfig,
    LLMProviderConfig,
    ProvidersConfig,
    apply_backend_preset,
)
from .retrieval import FusionMethod, SearchStrategy, validate_path_segment

# Stdlib logger: shared.logging imports this module, so get_logger would cycle.
_logger = logging.getLogger(__name__)


class PipelineStageType(Enum):
    """Ingestion stages, declared in pipeline order.

    The declaration order is the canonical stage order: the pipeline runs the
    stages, the cache keys fold in upstream inputs, and the run metadata
    sorts stage results in this order.
    """

    DOCUMENT_PARSING = "document_parsing"
    DOCUMENT_LOADING = "document_loading"
    TEXT_CHUNKING = "text_chunking"
    TRANSLATION = "translation"
    GRAPH_EXTRACTION = "graph_extraction"
    GLEANING = "gleaning"
    GRAPH_RESOLUTION = "graph_resolution"
    CLAIM_EXTRACTION = "claim_extraction"
    CLAIM_RESOLUTION = "claim_resolution"
    GRAPH_ANALYSIS = "graph_analysis"
    COMMUNITY_DETECTION = "community_detection"
    INDEXING = "indexing"


class ChunkingStrategy(str, Enum):
    SIMPLE = "simple"
    INTELLIGENT = "intelligent"


class Constants(str, Enum):
    ATTRIBUTE_PREFIX = "attr"
    # Type of a relationship extracted without one (its Neptune edge label).
    DEFAULT_RELATIONSHIP_TYPE = "RELATED_TO"
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


_LANGUAGE_NAMES: dict[LanguageCode, str] = {
    LanguageCode.DE: "German",
    LanguageCode.EN: "English",
    LanguageCode.ES: "Spanish",
    LanguageCode.FR: "French",
    LanguageCode.IT: "Italian",
    LanguageCode.JA: "Japanese",
    LanguageCode.KO: "Korean",
    LanguageCode.PT: "Portuguese",
    LanguageCode.RU: "Russian",
    LanguageCode.ZH: "Chinese",
}


def language_name(language: str) -> str:
    """The language as prompts name it: ``"ko"`` -> ``"Korean"``.

    A ``LanguageCode`` formats as ``LanguageCode.KO`` in a prompt template and
    a bare code is easy to misread, so prompts get the English name. Text that
    is not a supported code (e.g. a caller's ``"Brazilian Portuguese"``) is
    returned as given.
    """
    if isinstance(language, LanguageCode):
        return _LANGUAGE_NAMES[language]
    try:
        return _LANGUAGE_NAMES[LanguageCode(language.strip().lower())]
    except ValueError:
        return language


class ResolutionMethod(str, Enum):
    MINHASH = "minhash"
    SEQUENCE_MATCHER = "sequence_matcher"


class S3EncryptionType(str, Enum):
    """Server-side encryption header sent on cache uploads.

    ``BUCKET_DEFAULT`` sends no SSE header, so S3 applies the bucket's default
    encryption (SSE-S3 at minimum, or the bucket's SSE-KMS CMK). ``AES256`` /
    ``aws:kms`` force a per-object header that OVERRIDES the bucket default.
    """

    BUCKET_DEFAULT = "BUCKET_DEFAULT"
    AES256 = "AES256"
    KMS = "aws:kms"


class EmbeddingModelId(str, Enum):
    """The embedding models the package supports; a closed list.

    Unlike language-model ids, embedding-model config fields accept only these
    members: the model's vector dimension is fixed into the OpenSearch
    ``knn_vector`` mappings and validated against its capability record, so an
    unknown id has no safe default. Adding a model takes a member here plus a
    record in ``adapters/aws/bedrock_models.py``.
    """

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
    CLAUDE_V5_5_HAIKU = "anthropic.claude-haiku-5-5"
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
EFFORT_LEVELS: frozenset[str] = frozenset(get_args(EffortLevel))

# The Bedrock models of the two tiers when models.default / models.fast are
# unset, so changing a tier default is a one-line edit here.
DEFAULT_MODEL_ID: str = LanguageModelId.CLAUDE_V5_5_SONNET.value
FAST_MODEL_ID: str = LanguageModelId.CLAUDE_V5_5_HAIKU.value


def _coerce_model_id(value: Any) -> Any:
    # A LanguageModelId member (or any str enum) validates as its bare id.
    if isinstance(value, Enum):
        value = value.value
    return value.strip() if isinstance(value, str) else value


class ModelPurpose(str, Enum):
    """Why a language model is being created, so per-path policies can differ.

    Callers pass it to the LLM factory (``get_model(model_purpose=...)``):
    the response cache and the ingestion effort apply to ``INGESTION``
    models, and query chains get a transient-error retry. Unspecified means
    ``QUERY``.
    """

    QUERY = "query"
    INGESTION = "ingestion"
    EVALUATION = "evaluation"


class RerankModelId(str, Enum):
    """The rerank models the package supports; a closed list.

    Like ``EmbeddingModelId``, and unlike language-model ids, only these
    members are accepted. Adding a model takes a member here plus a record
    (document and token limits) in ``adapters/aws/bedrock_models.py``.
    """

    AMAZON_RERANK_V1 = "amazon.rerank-v1:0"
    COHERE_RERANK_V3_5 = "cohere.rerank-v3-5:0"
    # NOTE: add new models here


def model_id_value(model_id: Any) -> str:
    """The bare id string of a model-id config value (enum member or str)."""
    return str(model_id.value) if isinstance(model_id, Enum) else str(model_id)


def _known_member(enum_cls: type[Enum]) -> Callable[[Any], Any]:
    """Coerce a string naming a member of ``enum_cls`` to that member."""

    def coerce(value: Any) -> Any:
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, str):
            value = value.strip()
            try:
                return enum_cls(value)
            except ValueError:
                return value
        return value

    return coerce


# Embedding/rerank model ids: a catalog member when the id is one, else the
# bare string. A non-Bedrock provider serves any model name; for Bedrock the
# Config validator still requires a catalog member (the closed lists above).
EmbeddingModelRef = Annotated[
    EmbeddingModelId | Annotated[str, StringConstraints(min_length=1)],
    Field(union_mode="left_to_right"),
    BeforeValidator(_known_member(EmbeddingModelId)),
]
RerankModelRef = Annotated[
    RerankModelId | Annotated[str, StringConstraints(min_length=1)],
    Field(union_mode="left_to_right"),
    BeforeValidator(_known_member(RerankModelId)),
]


class ModelRole(str, Enum):
    """Every place the framework calls a language model, by its role.

    Each role runs on a tier (``ROLE_TIERS``) unless ``models.roles`` names
    another model or tier for it.
    """

    FIXING = "fixing"
    CHUNKING = "chunking"
    TRANSLATION = "translation"
    GRAPH_EXTRACTION = "graph_extraction"
    DESCRIPTION_SUMMARY = "description_summary"
    GLEANING = "gleaning"
    CLAIM_EXTRACTION = "claim_extraction"
    COMMUNITY_REPORT = "community_report"
    QUERY_TRANSLATION = "query_translation"
    ENTITY_EXTRACTION = "entity_extraction"
    STRATEGY_SELECTION = "strategy_selection"
    QUERY_REWRITE = "query_rewrite"
    ANSWER_GENERATION = "answer_generation"
    COMMUNITY_RELEVANCE = "community_relevance"
    GLOBAL_MAP = "global_map"
    GLOBAL_REDUCE = "global_reduce"
    DRIFT_PRIMER = "drift_primer"
    DRIFT_ACTION = "drift_action"
    EVALUATION = "evaluation"


_ROLE_DESCRIPTIONS: dict[ModelRole, tuple[ModelTier, str]] = {
    ModelRole.FIXING: ("default", "repairs malformed structured model output"),
    ModelRole.CHUNKING: ("fast", "places chunk boundaries (intelligent chunker)"),
    ModelRole.TRANSLATION: ("fast", "translates the corpus at ingestion"),
    ModelRole.GRAPH_EXTRACTION: ("default", "extracts entities and relationships"),
    ModelRole.DESCRIPTION_SUMMARY: (
        "fast",
        "summarizes the merged descriptions of an entity or relationship",
    ),
    ModelRole.GLEANING: ("default", "refines each text unit's extraction"),
    ModelRole.CLAIM_EXTRACTION: ("default", "extracts claims"),
    ModelRole.COMMUNITY_REPORT: ("default", "writes community reports"),
    ModelRole.QUERY_TRANSLATION: ("fast", "translates the query"),
    ModelRole.ENTITY_EXTRACTION: (
        "default",
        "extracts the query's entities and keywords",
    ),
    ModelRole.STRATEGY_SELECTION: ("fast", "routes an auto query to a strategy"),
    ModelRole.QUERY_REWRITE: ("fast", "rewrites a follow-up into a standalone query"),
    ModelRole.ANSWER_GENERATION: ("default", "writes the answer"),
    ModelRole.COMMUNITY_RELEVANCE: (
        "fast",
        "rates community relevance for global search's dynamic selection",
    ),
    ModelRole.GLOBAL_MAP: ("fast", "answers from each report batch (global map)"),
    ModelRole.GLOBAL_REDUCE: ("fast", "combines the map answers (global reduce)"),
    ModelRole.DRIFT_PRIMER: ("fast", "writes DRIFT's primer and follow-ups"),
    ModelRole.DRIFT_ACTION: ("fast", "answers each DRIFT follow-up"),
    ModelRole.EVALUATION: ("default", "judges answers in run-eval"),
}
# The tier each role runs on unless models.roles says otherwise.
ROLE_TIERS: dict[ModelRole, ModelTier] = {
    role: tier for role, (tier, _) in _ROLE_DESCRIPTIONS.items()
}


def _role_field(role: ModelRole) -> Any:
    tier, what = _ROLE_DESCRIPTIONS[role]
    return Field(
        default=None,
        min_length=1,
        description=(
            f"Model that {what}: a model id, or a tier name ('default' / "
            f"'fast'). null runs it on the '{tier}' tier."
        ),
    )


class RoleModelsConfig(BaseModel):
    """Per-role model overrides (``models.roles``)."""

    fixing: str | None = _role_field(ModelRole.FIXING)
    chunking: str | None = _role_field(ModelRole.CHUNKING)
    translation: str | None = _role_field(ModelRole.TRANSLATION)
    graph_extraction: str | None = _role_field(ModelRole.GRAPH_EXTRACTION)
    description_summary: str | None = _role_field(ModelRole.DESCRIPTION_SUMMARY)
    gleaning: str | None = _role_field(ModelRole.GLEANING)
    claim_extraction: str | None = _role_field(ModelRole.CLAIM_EXTRACTION)
    community_report: str | None = _role_field(ModelRole.COMMUNITY_REPORT)
    query_translation: str | None = _role_field(ModelRole.QUERY_TRANSLATION)
    entity_extraction: str | None = _role_field(ModelRole.ENTITY_EXTRACTION)
    strategy_selection: str | None = _role_field(ModelRole.STRATEGY_SELECTION)
    query_rewrite: str | None = _role_field(ModelRole.QUERY_REWRITE)
    answer_generation: str | None = _role_field(ModelRole.ANSWER_GENERATION)
    community_relevance: str | None = _role_field(ModelRole.COMMUNITY_RELEVANCE)
    global_map: str | None = _role_field(ModelRole.GLOBAL_MAP)
    global_reduce: str | None = _role_field(ModelRole.GLOBAL_REDUCE)
    drift_primer: str | None = _role_field(ModelRole.DRIFT_PRIMER)
    drift_action: str | None = _role_field(ModelRole.DRIFT_ACTION)
    evaluation: str | None = _role_field(ModelRole.EVALUATION)

    @field_validator("*", mode="before")
    @classmethod
    def _bare_id(cls, value: Any) -> Any:
        return _coerce_model_id(value)


class EmbeddingModelConfig(BaseModel):
    """The one embedding model (``models.embedding``): index, queries,
    evaluation and the visualization layout all embed with it."""

    model_id: EmbeddingModelRef | None = Field(
        default=None,
        description=(
            "Embedding model. With the bedrock embedding provider one of the "
            "EmbeddingModelId values (the vector dimension must be known for "
            "the index mapping), null = amazon.titan-embed-text-v2:0; with "
            "another provider any model name, required."
        ),
    )
    dimension: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Vector dimension, fixed into the index mappings. Bedrock: one the "
            "model supports (Titan Embed V2: 256, 512 or 1024), null = its "
            "largest. Another provider: the model's dimension, required."
        ),
    )


class ModelsConfig(BaseModel):
    """Which model serves each role (``models``).

    The provider of each family is ``providers.llm.kind`` /
    ``providers.embedding.kind`` / ``providers.rerank.kind``; the ids here are
    checked against it. Read the resolved models through ``Config.model_id``,
    ``Config.embedding_model_id`` and ``Config.rerank_model_id``.
    """

    default: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Model of the 'default' tier (extraction, gleaning, claims, "
            "community reports, output fixing, query entity/keyword "
            "extraction, answer generation, evaluation). null = "
            f"{DEFAULT_MODEL_ID} on Bedrock; required with another provider."
        ),
    )
    fast: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Model of the 'fast' tier (chunking, translation, description "
            "summarization, global/DRIFT search steps, query translation, "
            f"rewriting and routing). null = {FAST_MODEL_ID} on Bedrock, the "
            "default tier's model with another provider."
        ),
    )
    roles: RoleModelsConfig = Field(
        default_factory=RoleModelsConfig,
        description="Per-role overrides: a model id or a tier name per role.",
    )
    embedding: EmbeddingModelConfig = Field(
        default_factory=EmbeddingModelConfig,
        description="The embedding model and its dimension.",
    )
    rerank: RerankModelRef | None = Field(
        default=None,
        description=(
            "Rerank model. Bedrock: one of the RerankModelId values, null = "
            "cohere.rerank-v3-5:0. cross_encoder: a sentence-transformers "
            "model name, required. Ignored with providers.rerank.kind none."
        ),
    )

    @field_validator("default", "fast", mode="before")
    @classmethod
    def _bare_id(cls, value: Any) -> Any:
        return _coerce_model_id(value)


# Moved settings -> the key that replaces each. A config that sets one fails
# validation naming its replacement.
MOVED_KEYS: dict[str, str] = {
    "graph.analysis.centrality": "graph.visualization.centrality",
    "aws.bedrock.default_model_id": "models.default",
    "aws.bedrock.fast_model_id": "models.fast",
    "providers.llm.default_model_id": "models.default",
    "providers.llm.fast_model_id": "models.fast",
    "fixing.fixing_model_id": "models.roles.fixing",
    "processing.chunking.chunking_model_id": "models.roles.chunking",
    "processing.translation.translation_model_id": "models.roles.translation",
    "processing.graph_extraction.extraction_model_id": "models.roles.graph_extraction",
    "processing.graph_extraction.description_summarization.summary_model_id": (
        "models.roles.description_summary"
    ),
    "processing.gleaning.graph_refinement_model_id": "models.roles.gleaning",
    "processing.claim_extraction.extraction_model_id": "models.roles.claim_extraction",
    "graph.community_detection.report_generation.report_generation_model_id": (
        "models.roles.community_report"
    ),
    "search.translation_model_id": "models.roles.query_translation",
    "search.entity_extraction_model_id": "models.roles.entity_extraction",
    "search.strategy_selection_model_id": "models.roles.strategy_selection",
    "search.query_rewrite_model_id": "models.roles.query_rewrite",
    "search.answer_generation_model_id": "models.roles.answer_generation",
    "search.global_search.community_relevance_model_id": (
        "models.roles.community_relevance"
    ),
    "search.global_search.map_model_id": "models.roles.global_map",
    "search.global_search.map_reduce_model_id": "models.roles.global_reduce",
    "search.drift_search.primer_model_id": "models.roles.drift_primer",
    "search.drift_search.action_model_id": "models.roles.drift_action",
    "evaluation.evaluation_model_id": "models.roles.evaluation",
    "providers.embedding.model_id": "models.embedding.model_id",
    "providers.embedding.dimension": "models.embedding.dimension",
    "indexing.opensearch.embedding_model_id": "models.embedding.model_id",
    "indexing.opensearch.embedding_dimension": "models.embedding.dimension",
    "evaluation.embedding_model_id": "models.embedding.model_id",
    "graph.visualization.embeddings": "models.embedding.model_id",
    "providers.rerank.model_id": "models.rerank",
    "aws.dynamodb.enabled": "providers.doc_status.enabled",
    "search.reranking.rerank_model_id": "models.rerank",
}


# Removed settings -> why they are gone. A config that sets one loads with
# a WARNING and the setting ignored.
REMOVED_KEYS: dict[str, str] = {
    "aws.bedrock.guardrail.apply_to": (
        "the guardrail screens the user query and the answer only; ingestion "
        "and evaluation models are never guarded"
    ),
    "indexing.cross_run_merge": (
        "an incremental run always merges its delta into the stored graph; "
        "turning that off overwrote entities other documents still cite"
    ),
}


def _sets(data: Any, path: str) -> bool:
    node = data
    for part in path.split("."):
        node = node.get(part, _ABSENT) if isinstance(node, dict) else _ABSENT
    return node is not _ABSENT


def moved_keys(data: Any) -> list[str]:
    """``"old -> new"`` for each moved setting ``data`` sets."""
    return [f"{old} -> {new}" for old, new in MOVED_KEYS.items() if _sets(data, old)]


def removed_keys(data: Any) -> list[str]:
    """The removed settings ``data`` sets (see ``REMOVED_KEYS``)."""
    return [path for path in REMOVED_KEYS if _sets(data, path)]


_ABSENT: Any = object()


class GuardrailConfig(BaseModel):
    """Amazon Bedrock Guardrails on the user-facing query path.

    Disabled by default; set ``identifier`` (and optionally ``version``) to
    enforce content/PII/denied-topic/prompt-attack policies — the WAF
    security-pillar control for a user-facing LLM RAG application.

    The query chain screens texts with the ``ApplyGuardrail`` API rather than
    attaching the guardrail to its model calls: ``check_input`` screens the
    raw user query once, before memory, rewrite, routing and retrieval, and
    ``check_output`` the final answer. A blocked query is answered with the
    guardrail's blocked message and never retrieved; a masked one (PII
    anonymized) continues with the masked text. The internal query calls
    (rewrite, routing, keyword extraction, map steps, answer generation) run
    unguarded: their prompts carry retrieved corpus text and developer
    instructions, which an attached guardrail would judge as user input.
    Ingestion and evaluation models run unguarded: a PII-anonymizing
    guardrail rewrites names to placeholders (every person merges into one
    entity) and a prompt-attack filter blocks instruction-like corpus text,
    which silently yields empty extractions.
    """

    identifier: str | None = Field(
        default=None,
        description=(
            "Bedrock guardrail ID or ARN; setting it enables the guardrail on "
            "the query path. The guardrail must exist in "
            "aws.bedrock.region_name."
        ),
    )
    version: str = Field(
        default="DRAFT",
        min_length=1,
        description="Guardrail version to apply (e.g. 'DRAFT' or a published version number)",
    )
    trace: bool = Field(
        default=False,
        description="Emit guardrail trace details for observability/auditing "
        "(full assessments, including non-detected entries)",
    )
    check_input: bool = Field(
        default=True,
        description="Screen each raw user query (ApplyGuardrail, source INPUT) "
        "before retrieval: prompt attacks, denied topics, PII",
    )
    check_output: bool = Field(
        default=True,
        description="Screen each final answer (ApplyGuardrail, source OUTPUT). "
        "A streamed answer is then delivered in one chunk, after the check",
    )

    @property
    def enabled(self) -> bool:
        return bool(self.identifier)


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
        description=(
            "Attempts per call, including the first, for transient Bedrock "
            "errors that botocore does not retry (such as HTTP 424), on "
            "embeddings and query-time calls. 1 disables the retry."
        ),
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
    region_name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Region for Bedrock model, embedding and rerank calls, and where "
            "the guardrail must exist. null = aws.region_name. Set it only "
            "when your models are enabled in another region; in a private VPC "
            "whose only Bedrock route is a VPC endpoint, leave it null or set "
            "the VPC's region."
        ),
    )
    assumed_role_arn: str | None = Field(
        default=None, description="AWS assumed role ARN for Bedrock service"
    )
    enable_global_profile: bool = Field(
        default=True,
        description=(
            "Resolve cross-region (global) inference profiles. "
            "Keep it on: Claude 4.7+ and OpenAI GPT models are "
            "invocable only through a profile, which also needs "
            "bedrock:ListInferenceProfiles."
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
            "that wins over this value. Raise it if answers are cut off "
            "(stopReason max_tokens); null sends the model maximum."
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
            "curated record, or to correct one, without a release. Language "
            "models only: embedding and rerank ids are a closed list "
            "(EmbeddingModelId, RerankModelId)."
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
            "Reasoning effort for calls on the default tier's model "
            "(models.default). Sent as output_config.effort to Anthropic "
            "adaptive-thinking models (Claude 4.6+), where it replaces the "
            "fixed thinking token budget, and as reasoning.effort to OpenAI "
            "GPT models. 'xhigh'/'max' are only accepted by some models, and "
            "an unsupported level fails fast; lower it to cut cost and "
            "latency."
        ),
    )
    fast_effort: EffortLevel = Field(
        default="low",
        description=(
            "Reasoning effort for calls on the fast tier's model (models.fast; when it differs "
            "from the default tier's). The shipped fast model, Claude Haiku 5.5, "
            "thinks adaptively, so this sets how much it reasons on fast-tier "
            "calls; it has no effect on a fast model that does not reason."
        ),
    )
    ingestion_effort: EffortLevel | None = Field(
        default=None,
        description=(
            "Reasoning effort for ingestion calls on the default tier (graph "
            "extraction, gleaning, claim extraction, community reports, prompt "
            "tuning, and the output fixer when models.roles.fixing is a "
            "default-tier model). These make up most of an "
            "ingestion's default-tier tokens, so lowering this cuts ingestion "
            "cost without changing query-time effort. null inherits "
            "default_effort. Fast-tier ingestion calls keep fast_effort."
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

    def tier_effort(
        self, tier: ModelTier, purpose: "ModelPurpose | None" = None
    ) -> EffortLevel:
        """Configured effort for ``tier``, for a call made for ``purpose``.

        A default-tier ingestion call uses ``ingestion_effort`` when it is set.
        """
        if tier == "fast":
            return self.fast_effort
        if purpose is ModelPurpose.INGESTION and self.ingestion_effort:
            return self.ingestion_effort
        return self.default_effort


class NeptuneConfig(BaseModel):
    endpoint: str | None = Field(
        default=None,
        description=(
            "Neptune cluster endpoint. Required with the neptune"
            " graph store (the CDK stack injects it as "
            "UKR__AWS__NEPTUNE__ENDPOINT)."
        ),
    )
    port: int = Field(
        default=8182, ge=1, le=65535, description="Neptune database connection port"
    )
    use_iam: bool = Field(
        default=True,
        description=("SigV4-sign Neptune requests (IAM database " "authentication)."),
    )
    use_ssl: bool = Field(
        default=True,
        description=(
            "Connect over TLS (wss://). Neptune requires it; set false only for "
            "a plain TinkerPop Gremlin Server (ws://), together with "
            "use_iam: false."
        ),
    )
    pool_size: int = Field(
        default=4,
        ge=1,
        description=(
            "Gremlin DriverRemoteConnection pool size (max concurrent in-flight "
            "requests over the websocket). The client raises it to indexing."
            "neptune.index_concurrency when that is larger, so concurrent write "
            "batches are not serialized on fewer connections. Query traversals "
            "beyond it wait on the event loop. The default matches the 4 query "
            "threads (2 x vCPUs) of the default db.r6g.large instance; more "
            "in-flight requests only queue on the server, so raise it with the "
            "instance size."
        ),
    )


class OpenSearchConfig(BaseModel):
    endpoint: str | None = Field(
        default=None,
        description=(
            "OpenSearch domain endpoint. Required with the "
            "opensearch vector store (the CDK stack injects it "
            "as UKR__AWS__OPENSEARCH__ENDPOINT)."
        ),
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
        default=False,
        description=(
            "SigV4-sign OpenSearch requests. false uses basic "
            "auth with username / password, usually from "
            "UKR__AWS__OPENSEARCH__USERNAME and "
            "UKR__AWS__OPENSEARCH__PASSWORD (for example in "
            ".env)."
        ),
    )
    allow_anonymous: bool = Field(
        default=False,
        description=(
            "Connect without authentication, for a local OpenSearch with the "
            "security plugin disabled. Mutually exclusive with use_iam and "
            "username/password."
        ),
    )
    sigv4_service_name: str = Field(
        default="es",
        description=(
            "SigV4 signing service for IAM auth: 'es' for a managed OpenSearch "
            "domain, 'aoss' for an OpenSearch Serverless collection. A wrong "
            "value fails auth (often surfacing as zero search hits)."
        ),
    )

    @model_validator(mode="after")
    def validate_anonymous_auth(self) -> "OpenSearchConfig":
        if self.allow_anonymous and (self.use_iam or self.username):
            raise ValueError(
                "allow_anonymous cannot be combined with use_iam or username/password"
            )
        return self


class S3EncryptionConfig(BaseModel):
    encryption_type: S3EncryptionType = Field(
        default=S3EncryptionType.BUCKET_DEFAULT,
        description=(
            "S3 server-side encryption for cache uploads: BUCKET_DEFAULT (send no "
            "header; the bucket's default encryption, e.g. its KMS CMK, applies), "
            "AES256 (force SSE-S3) or aws:kms (force SSE-KMS with kms_key_id)."
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
        default=None,
        description=(
            "Bucket for the stage-cache sync (run-ingestion "
            "--s3-sync) and the persisted embedding cache."
        ),
    )
    encryption: S3EncryptionConfig = Field(
        default_factory=S3EncryptionConfig,
        description="S3 server-side encryption configuration",
    )


class DynamoDBConfig(BaseModel):
    table_name: str = Field(
        default="unified-kg-rag-on-aws-doc-status-v2",
        min_length=1,
        description=(
            "DynamoDB table holding each document's status and text units, "
            "partitioned by index namespace."
        ),
    )
    create_table_if_missing: bool = Field(
        default=True,
        description=(
            "Create the doc-status table on first use. Set false when the "
            "table is managed by IaC (the CDK stack does)."
        ),
    )
    billing_mode: str = Field(
        default="PAY_PER_REQUEST",
        description="Billing mode used when auto-creating the table",
    )


class AWSConfig(BaseModel):
    # us-west-2 offers every default model, including both rerank models,
    # which several regions (e.g. ap-northeast-2) do not.
    region_name: str = Field(
        default="us-west-2",
        min_length=1,
        description=(
            "Region of Neptune, OpenSearch,"
            " S3 and DynamoDB, and of "
            "Bedrock unless "
            "aws.bedrock.region_name is "
            "set. AWS_REGION fills it only "
            "when the file leaves it unset."
            " The default offers every "
            "default model, including both "
            "rerank models, which some "
            "regions (e.g. ap-northeast-2) "
            "do not."
        ),
    )
    profile_name: str | None = Field(
        default=None,
        description=(
            "Named AWS profile. null uses the default credential"
            " chain (environment variables, instance or task "
            "role, ...); AWS_PROFILE fills it only when the file"
            " leaves it unset."
        ),
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

    @property
    def bedrock_region(self) -> str:
        """Region Bedrock calls go to: aws.bedrock.region_name, else aws.region_name."""
        return self.bedrock.region_name or self.region_name


class FixingConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description=(
            "When a structured stage gets malformed JSON from "
            "the model, ask a model (models.roles.fixing) to "
            "repair it instead of failing the item. Leave it on."
        ),
    )


class DocumentParsingConfig(BaseModel):
    source_directory: str | Path = Field(
        default="source",
        description=(
            "Directory to load documents from, for library "
            "callers. run-ingestion requires "
            "--source-directory (or UKR_SOURCE_DIRECTORY) "
            "instead."
        ),
    )
    target_directory: str | Path | None = Field(
        default=None,
        description=(
            "Directory to export each parsed document to as <stem>.json for "
            "inspection (run-ingestion --output-directory). null = no export. "
            "Must not be the source directory."
        ),
    )
    index_value: str | None = Field(
        default=None,
        description=(
            "Index suffix the run writes: every OpenSearch index and Neptune "
            "label is named <prefix>-<index_value>. null = 'default'. Query "
            "the corpus with the same value as run-rag / run-eval --suffix. "
            "run-ingestion has no flag for it: use one config file per tenant "
            "or corpus version."
        ),
    )
    source_scope: str | None = Field(
        default=None,
        description=(
            "Identity of the corpus source for incremental indexing: a run only "
            "treats registry documents of its own index suffix AND source scope "
            "as deleted. None = the resolved source directory. Set it when the "
            "local directory is a staging copy, e.g. the S3 URI a container "
            "syncs from (env UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE)."
        ),
    )


class ChunkingConfig(BaseModel):
    chunker_type: ChunkingStrategy = Field(
        default=ChunkingStrategy.INTELLIGENT,
        description=(
            "intelligent lets the chunking model pick semantic boundaries (one"
            " fast LLM call per pre-chunk); simple splits by size with no LLM "
            "call."
        ),
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
            "Maximum chunk size in characters. Every chunk is embedded whole "
            "and reranked as one document, so it must fit the embedding "
            "model's input (Titan Text Embeddings V2: 8,192 tokens) and the "
            "reranker's per-document limit (Cohere Rerank 3.5: 4,096 tokens), "
            "or its tail is never embedded or reranked. Sizes count "
            "characters: English averages ~4 characters per token, a "
            "Han/Hangul/Kana character ~1, so for CJK corpora with reranking "
            "use about half the default sizes."
        ),
    )
    chunk_overlap: int = Field(
        default=500,
        ge=0,
        description=("Overlap between consecutive chunks, in " "characters."),
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
            "Target chunk size for the size-based splitter (the simple "
            "chunker, and intelligent chunking when the LLM fails or a chunk "
            "exceeds max_chunk_size). It splits at paragraphs, then lines, "
            "then after CJK sentence terminators, then spaces. 4,800 "
            "characters is ~1,200 English tokens, the default chunk size of MS"
            " GraphRAG and LightRAG."
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
        description=(
            "Run the translation stage. It is skipped at no cost when "
            "source_language equals target_language and there are no "
            "additional_target_languages."
        ),
    )
    source_language: LanguageCode = Field(
        default=LanguageCode.EN,
        description="Predominant source language of the corpus. Selects the "
        "OpenSearch analyzer of the original chunk text and drives the "
        "same-language no-op skip; translation itself is content-driven.",
    )
    target_language: LanguageCode = Field(
        default=LanguageCode.EN,
        description=(
            "Language the corpus is translated into. "
            "Extraction, embeddings and the answer "
            "context use it, and entity and "
            "relationship descriptions are written in"
            " it."
        ),
    )
    additional_target_languages: list[LanguageCode] | None = Field(
        default=None,
        description=(
            "Extra languages to translate into. Each is indexed in its own "
            "translated_text_<language> field with that language's analyzer "
            "and searched lexically (useful with a per-query "
            "RAGInput.target_language); extraction, embeddings and the answer "
            "context use target_language only. Each language is one more LLM "
            "call per chunk."
        ),
    )

    @property
    def translated_languages(self) -> list[str]:
        """Language codes with a translated text field: target first, no repeats."""
        languages = [self.target_language, *(self.additional_target_languages or [])]
        return list(dict.fromkeys(language.value for language in languages))

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
        description=(
            "Re-summarize merged descriptions longer than "
            "force_summary_threshold_tokens with an LLM (as MS GraphRAG and "
            "LightRAG do). Off, descriptions are only concatenated and grow "
            "without bound for frequently mentioned entities."
        ),
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
        description=(
            "Hallucination guard: drop (or, with action 'penalize', "
            "down-weight) entities and relationships whose verbatim "
            "source_text span is not found in their chunk. Punctuation is "
            "ignored and Han/Hangul/Kana spans are compared by character "
            "bigrams, so a changed Korean particle still counts as present. It"
            " applies to gleaning additions too. Requires an extraction model "
            "that emits a source_text span per item."
        ),
    )
    action: str = Field(
        default="drop",
        pattern="^(drop|penalize)$",
        description="What to do with an ungrounded entity: 'drop' removes it; "
        "'penalize' multiplies its confidence by penalty_factor so the "
        "confidence threshold can filter it without hard deletion. An "
        "ungrounded relationship is likewise dropped, or kept at its weight "
        "times penalty_factor; 'penalize' still drops one naming an entity its "
        "chunk did not extract, which it would otherwise make cite the chunk. "
        "Gleaning additions follow the same action.",
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
        "short legitimate names on weak signal. A Han/Hangul/Kana character "
        "counts as one token.",
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
    max_entities_per_chunk: int = Field(
        default=50,
        ge=1,
        description=(
            "Most entities the extraction prompt asks for "
            "per text chunk; it also sizes the extraction "
            "call's output-token floor."
        ),
    )
    max_relationships_per_chunk: int = Field(
        default=50, ge=1, description="Maximum relationships per text chunk"
    )
    entity_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Drop an entity from each chunk that scored it below this "
            "confidence, with that chunk's relationships naming it (per chunk,"
            " so the result does not depend on which documents are indexed "
            "together). Compared against the normalized confidence: the "
            "prompt's 1-10 score / 10. 0.0 keeps everything."
        ),
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
        description=(
            "Entity categories injected into the extraction prompt, each "
            "'LABEL: short description' (the description is optional), e.g. "
            "['GENE: ...', 'DISEASE: ...']. The cheapest and most effective "
            "domain-adaptation knob; an empty list lets the model choose any "
            "relevant types."
        ),
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
        default=True,
        description=(
            "Extra extraction passes per text unit that ask the "
            "model for the entities and relationships it missed."
        ),
    )
    max_rounds: int = Field(
        default=3,
        ge=1,
        description=(
            "Maximum gleaning rounds per text unit. Each round after the first"
            " re-sends only the text units whose previous answer added a new "
            "entity or relationship, so a unit stops once an answer adds "
            "nothing new. 1 is MS GraphRAG's default, for when ingestion cost "
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


class ClaimExtractionConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description=(
            "Extract claims (covariates): one more LLM call per text unit. "
            "When on, local search injects the matching claims into its "
            "context and simple search includes the claims index in its sweep."
            " Claim subjects and objects are not linked into the entity graph."
        ),
    )
    max_entities_per_prompt: int = Field(
        default=100,
        ge=0,
        description="Maximum entities per claim extraction prompt",
    )


class ProcessingConfig(BaseModel):
    max_concurrency: int = Field(
        default=40,
        ge=1,
        description=(
            "LLM calls an ingestion stage keeps in flight: the next item "
            "starts as soon as one finishes, so this is both the peak and the "
            "steady concurrency. These stages are I/O-bound, so this can be "
            "well above the CPU count. Bedrock quotas differ per account, "
            "model and Region: lower it if the logs show ThrottlingException "
            "retries, raise it if quota allows. The backend: local preset sets"
            " 20, since a local model server queues the calls it cannot run at"
            " once."
        ),
    )
    max_attempts: int = Field(
        default=5,
        ge=1,
        description=(
            "Attempts per ingestion LLM item, including the first call. "
            "Transient Bedrock errors, call timeouts and unparseable output "
            "are retried; other errors fail fast. 1 disables the retry."
        ),
    )
    io_workers: int = Field(
        default=64,
        ge=1,
        description=(
            "Threads for blocking I/O on the query path (the Bedrock calls "
            "LangChain runs in the event loop's default executor, Neptune "
            "traversals, reranking) in the CLIs and GraphRAGChain's sync "
            "methods. Python's default is min(32, CPUs + 4), six on a 2-vCPU "
            "task. An async host calls "
            "unified_kg_rag.configure_event_loop(asyncio.get_running_loop(), "
            "config.processing.io_workers) at startup."
        ),
    )
    ignore_errors: bool = Field(
        default=False,
        description=(
            "When a whole step fails (a batch call, a translation language, an"
            " evaluator, global map/reduce, the DRIFT primer, the auto "
            "router), log it and continue with an empty or degraded result "
            "instead of raising. Items whose own LLM call fails are always "
            "skipped and reported, whatever this says."
        ),
    )
    deduplicate: bool = Field(
        default=False,
        description=(
            "Drop near-duplicate documents before extraction. "
            "Two documents are near-duplicates when the Jaccard"
            " similarity of their character 3-gram sets, "
            "estimated with a 128-permutation MinHash, is at "
            "least 0.9 (fixed parameters); of each pair the "
            "first is kept, and an incremental run keeps the "
            "already-indexed copy."
        ),
    )
    resolution_method: ResolutionMethod = Field(
        default=ResolutionMethod.MINHASH,
        description=(
            "Name-similarity measure of fuzzy entity resolution "
            "(fuzzy_entity_resolution): minhash or sequence_matcher."
        ),
    )
    fuzzy_entity_resolution: bool = Field(
        default=False,
        description="Also merge entities whose names are only similar "
        "(resolution_method at similarity_threshold), in full builds and "
        "incremental merges alike. Off by default: names are merged when they "
        "are equal after normalization, legal-form removal ('Acme Inc' / "
        "'Acme') and dense-script spacing, because name similarity also joins "
        "different organizations that share most of their name ('... "
        "University' / '... University Hospital').",
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity threshold for fuzzy entity resolution "
            "(fuzzy_entity_resolution) and for matching claim subjects and "
            "objects to entity names. Raise it if distinct entities merge."
        ),
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
        description=(
            "Compute betweenness centrality (bridge nodes) for the "
            "visualization; ingestion itself computes no centrality. On graphs"
            " larger than betweenness_auto_sample_threshold nodes it is "
            "sampled instead of computed exactly."
        ),
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
    statistics: StatisticsConfig = Field(
        default_factory=StatisticsConfig,
        description="Graph-level statistics configuration",
    )


class ReportGenerationConfig(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable automatic community report generation"
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
        description="Report length, as the number of findings the report asks "
        "for: short (5-7), medium (7-10), long (10-15)",
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
        description=(
            "Leiden modularity resolution: higher gives more, smaller "
            "communities (and more report calls); lower gives fewer, larger "
            "ones."
        ),
    )
    random_state: int = Field(
        default=42,
        ge=1,
        description="Leiden random seed (a positive integer; the native Leiden "
        "rejects 0). The same graph and seed give the same communities.",
    )
    max_levels: int = Field(
        default=5, ge=1, description=("Maximum depth of the " "community hierarchy.")
    )
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
    max_cluster_size: int = Field(
        default=20,
        ge=2,
        description="Maximum members per community: entities at level 0, "
        "child communities above. A larger Leiden community is re-partitioned "
        "on its own subgraph, recursively, until every piece fits or Leiden "
        "finds no finer split; the next level regroups the pieces. Without the "
        "bound, modularity's resolution limit grows communities with the "
        "corpus beyond what one report's max_report_context_tokens can "
        "describe (or a parent's roll-up can summarize). MS GraphRAG's default "
        "is 10 with a larger report budget; 20 typically keeps a leaf's "
        "entities and relationships, or a parent's child summaries, within the "
        "default 4000-token report context.",
    )
    min_community_size: int = Field(
        default=3,
        ge=1,
        description=(
            "Minimum entities per level-0 community: a smaller one merges into"
            " the neighbour it shares the most edge weight with, unless that "
            "would exceed max_cluster_size."
        ),
    )
    auto_resolution: bool = Field(
        default=False,
        description=(
            "Sweep auto_resolution_candidates at every hierarchy level and "
            "keep the most modular resolution; otherwise resolution is used. "
            "Off by default: maximizing modularity on the coarse upper-level "
            "graphs picks low resolutions that merge unrelated communities, "
            "which cost global search accuracy in testing."
        ),
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

    @model_validator(mode="after")
    def _check_cluster_size_bounds(self) -> "CommunityDetectionConfig":
        if self.min_community_size > self.max_cluster_size:
            raise ValueError(
                f"min_community_size ({self.min_community_size}) must not exceed "
                f"max_cluster_size ({self.max_cluster_size})"
            )
        return self


class ReducerSettings(BaseModel):
    """A dimensionality reduction of the node layout (PCA; the base of the others)."""

    n_components: int = Field(default=2, ge=1, description="Output dimensions.")
    metric: str = Field(default="euclidean", description="Distance metric.")
    random_state: int | None = Field(
        default=42,
        description="Random seed, so a re-render gives the same layout; null "
        "for a random one (UMAP then runs in parallel).",
    )


class UMAPSettings(ReducerSettings):
    n_neighbors: int = Field(
        default=15, ge=2, description="Neighbours of the local approximation."
    )
    min_dist: float = Field(
        default=0.1, ge=0.0, description="Minimum distance between laid-out points."
    )


class TSNESettings(ReducerSettings):
    perplexity: int = Field(
        default=30, ge=1, description="Balance of local and global structure."
    )
    max_iter: int = Field(default=1000, ge=250, description="Optimization iterations.")


class LayoutSettings(BaseModel):
    """Parameters of each ``layout_method``."""

    umap: UMAPSettings = Field(default_factory=UMAPSettings)
    tsne: TSNESettings = Field(default_factory=TSNESettings)
    pca: ReducerSettings = Field(default_factory=ReducerSettings)


class InteractiveGraphSettings(BaseModel):
    """The interactive pyvis graph (``interactive_graph.html``)."""

    height: str = Field(default="900px", description="Canvas height (CSS).")
    width: str = Field(default="100%", description="Canvas width (CSS).")
    physics_enabled: bool = Field(
        default=False, description="Run the pyvis physics simulation."
    )
    cdn_resources: str = Field(
        default="in_line", description="pyvis cdn_resources: in_line, local or remote."
    )
    background_color: str = Field(default="#f8fafc", description="Canvas colour.")
    font_family: str = Field(default="Arial", description="Label font.")
    tooltip_delay: int = Field(default=200, ge=0, description="Tooltip delay (ms).")
    max_nodes: int | None = Field(
        default=2000,
        ge=0,
        description="Keep the top-N nodes by degree; 0 or null disables the cap "
        "(a browser cannot lay out tens of thousands of nodes).",
    )


class StaticChartSettings(BaseModel):
    """The static Bokeh charts."""

    figure_width: int = Field(default=900, ge=100, description="Chart width (px).")
    figure_height: int = Field(default=600, ge=100, description="Chart height (px).")
    font: str = Field(default="Arial, sans-serif", description="Chart font.")
    background_color: str = Field(default="#fdfdfd", description="Chart background.")
    log_scale_x: bool = Field(
        default=False, description="Logarithmic x axis on distributions."
    )


class VisualizationConfig(BaseModel):
    centrality: CentralityConfig = Field(
        default_factory=CentralityConfig,
        description="Node centrality metrics the visualization sizes nodes and "
        "plots by (computed at render time; no ingestion stage reads them)",
    )
    enabled: bool = Field(
        default=False,
        description=(
            "Render visualizations during ingestion. Off by default: it adds "
            "an embedding call per node (embedding_method) and a "
            "dimensionality reduction to every ingest; render on demand with "
            "run-visualization from the data export_data writes instead."
        ),
    )
    export_data: bool = Field(
        default=True,
        description="Write visualization_data.json (graph nodes, edges and the "
        "community hierarchy) at ingestion when enabled is false, so "
        "run-visualization can render on demand. One pass over the graph: no "
        "embedding or model call, no layout or centrality (run-visualization "
        "computes them). enabled: true always writes the file, with its layout.",
    )
    outputs_directory: str | Path | None = Field(
        default=None,
        description="Directory to save visualization files. When unset, "
        "ingestion writes to '<cache directory>/<pipeline_id>/visualization' "
        "(--cache-directory), so visualization_data.json is synced to S3 with the cache; other callers "
        "fall back to 'outputs/visualization'.",
    )
    embedding_method: str = Field(
        default="description",
        pattern="^(description|none)$",
        description="Node embedding for the layout: 'description' embeds each "
        "node's 'name: description' with the embedding model; 'none' uses a "
        "spring layout from the graph topology.",
    )
    layout_method: str = Field(
        default="umap",
        pattern="^(umap|tsne|pca)$",
        description=("Dimensionality reduction of the node layout: umap, tsne or pca."),
    )
    layout: LayoutSettings = Field(
        default_factory=LayoutSettings,
        description="Parameters per layout_method.",
    )
    interactive: InteractiveGraphSettings = Field(
        default_factory=InteractiveGraphSettings,
        description="Settings of the interactive pyvis graph.",
    )
    static: StaticChartSettings = Field(
        default_factory=StaticChartSettings,
        description="Settings of the static Bokeh charts.",
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
        description=("Items per Neptune write batch."),
    )
    index_concurrency: int = Field(
        default=1,
        ge=1,
        description=(
            "Neptune write batches (and relationship edges, one per request) "
            "submitted concurrently; the Gremlin connection pool grows to at "
            "least this. Above 1, batches touching the same vertices can raise"
            " ConcurrentModificationException; the retry recovers, but the "
            "conflicts can make a large run slower than sequential writes, so "
            "measure before raising it."
        ),
    )
    max_attempts: int = Field(
        default=4,
        ge=1,
        description=(
            "Attempts per Neptune indexer request (writes, read-backs, drops "
            "and label clears), including the first, with full-jitter "
            "exponential backoff (the wait after attempt n is random between 0"
            " and retry_delay_seconds * 2^(n-1) seconds). Errors a retry "
            "cannot fix (malformed query, access denied, bad parameter, "
            "read-only violation) fail on the first attempt. 1 disables the "
            "retry."
        ),
    )
    retry_delay_seconds: int = Field(
        default=2,
        ge=0,
        description="Base of the retry backoff in seconds (full jitter): the "
        "wait after attempt n is random between 0 and "
        "retry_delay_seconds * 2**(n-1). 0 retries at once.",
    )
    property_max_length: int = Field(
        default=4000,
        ge=1,
        description=(
            "Character cap per Neptune property value. Keep it above the "
            "longest description that is not re-summarized (summarization "
            "triggers above force_summary_threshold_tokens, 600 tokens or "
            "~2,400 English characters). OpenSearch keeps descriptions in "
            "full. Takes effect on re-ingestion."
        ),
    )
    max_hops: int = Field(
        default=3,
        ge=1,
        description=(
            "Depth of the graph expansion around the seed entities at query " "time."
        ),
    )
    max_results_per_hop: int = Field(
        default=50,
        ge=1,
        description="Maximum number of neighbours one node contributes per "
        "traversal hop, so a hub entity cannot fan the expansion out unbounded",
    )
    entity_importance_source: Literal["rank", "degree", "none"] = Field(
        default="rank",
        description=(
            "Graph-expansion relevance orders entities by distance from the "
            "seeds (seeds first) and, at the same distance, by this "
            "importance: 'rank' (the indexed entity rank, its degree), "
            "'degree' (the entity's edges counted at query time, as MS "
            "GraphRAG ranks entities) or 'none' (a neutral 0.5 for every "
            "entity, so expansion ranks by distance alone)."
        ),
    )
    traversal_fetch_multiplier: int = Field(
        default=3,
        ge=1,
        description=(
            "Graph expansion fetches top_k * retrieval_multiplier * this many "
            "entities besides the seeds, split evenly across the seeds, ranks "
            "them by relevance, then keeps every seed plus the top "
            "top_k * retrieval_multiplier of the others. 1 = no over-fetch: "
            "every traversed entity is kept, with no cut after ranking."
        ),
    )


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
        description=(
            "Build the relationship vector index (embeds "
            "Relationship.description). LightRAG mix/hybrid query it for "
            "high-level keywords and incident edges, GraphRAG local for its "
            "relationship section and bridge relationships, and simple and "
            "drift search it with the other indexes; only global does without "
            "it. false skips the relationship embeddings at the cost of those."
            " The Neptune relationship edges are indexed either way."
        ),
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
    extraction_records_index_prefix: str = Field(
        default="graphrag-extraction-records",
        min_length=1,
        max_length=100,
        description="Index name prefix for the extraction store: each text "
        "unit's extracted entities, relationships and claims, keyed by its "
        "text and the extraction settings, so an unchanged chunk is not "
        "extracted again (no embeddings). A full build or reset keeps only "
        "the records its text units name",
    )
    default_analyzer: str = Field(
        default="standard",
        min_length=1,
        description="OpenSearch text analyzer used when the language has no specific mapping",
    )
    language_analyzers: dict[str, str] = Field(
        default_factory=lambda: {"en": "english", "ko": "nori"},
        description=(
            "Text analyzer per language code (en: english, ko: nori); "
            "languages without an entry use default_analyzer. nori needs the "
            "analysis-nori plugin, which an Amazon OpenSearch Service domain "
            "has only once its optional package is associated (the CDK stack "
            "maps ko to cjk). Setting the key replaces the whole map."
        ),
    )
    refresh_after_batch: bool = Field(
        default=True,
        description="Whether to refresh OpenSearch indices after each batch operation for immediate visibility",
    )
    persist_embedding_cache: bool = Field(
        default=False,
        description=(
            "Persist the content-hash embedding cache to S3, so unchanged text"
            " is not embedded again by a later run (each Fargate task is a "
            "fresh process). Requires aws.s3.bucket_name."
        ),
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
    index_settings: dict[str, Any] = Field(
        default_factory=lambda: {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
        },
        description=(
            "OpenSearch settings of the indices this package creates "
            "(number_of_shards, number_of_replicas, refresh_interval, ...); "
            "they apply to indices created afterwards. Raise refresh_interval "
            '(or set "-1") for faster bulk loads, then reset it for live '
            "querying. Setting the key replaces the whole map."
        ),
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
        description=(
            "HNSW algorithm parameters for approximate nearest neighbor search. "
            "ef_search is the candidate list size of a query: on faiss and "
            "nmslib the index setting index.knn.algo_param.ef_search (fixed "
            "when the index is created), on lucene, which ignores that "
            "setting, the k of every kNN query (at least the hits requested), "
            "read at query time."
        ),
    )


class IndexingConfig(BaseModel):
    reset: bool = Field(
        default=False,
        description=(
            "Rebuild the run's index namespace from the whole corpus instead "
            "of applying a delta, without taking it offline: the vector "
            "indices swap aliases, the graph is rewritten in place and what "
            "the rebuild did not write is dropped at the end; then the "
            "namespace's registry records are written again. Set it for one "
            "run."
        ),
    )
    additional_suffix: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description=(
            "Appended after the run's suffix in every OpenSearch index name "
            "and Neptune label (<prefix>-<suffix>-<additional_suffix>; the "
            "suffix is processing.document_parsing.index_value at ingestion "
            "and --suffix at query time, both 'default' when unset). Set it "
            "identically for ingestion and queries, for versioned or "
            "multi-tenant separation."
        ),
    )
    community_drift_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="On incremental runs with communities on, recommend a "
        "rebuild (WARNING and the community_rebuild_recommended metric) when "
        "at least this share of the entities the run clustered also belongs "
        "to an earlier level-0 community. A delta run clusters only its own "
        "documents, so such an entity sits in two communities and the earlier "
        "community's report misses what the run added; indexing.reset: true "
        "re-clusters the whole graph.",
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
        description=(
            "Per-index-type write failure rate above which the indexing stage "
            "fails, and share of communities without a report above which "
            "community detection fails; below it the affected documents are "
            "recorded FAILED and retried. A fully failed index type always "
            "fails the stage. 1.0 disables the partial-failure gate."
        ),
    )
    max_document_failures: int = Field(
        default=3,
        ge=1,
        description="Incremental runs: a document recorded FAILED this many "
        "consecutive runs with unchanged content is no longer retried. It stays "
        "FAILED and is skipped as unchanged (with a WARNING) until its content "
        "changes or this value is raised.",
    )
    max_document_attempts: int = Field(
        default=3,
        ge=1,
        description="Incremental runs: a document whose indexing (of the same "
        "content) was started this many consecutive runs and never finished "
        "(the run died each time: out of memory, a task timeout) is "
        "quarantined: the next run removes what the attempts wrote, records "
        "it QUARANTINED and skips it until its content or the extraction "
        "settings change or this value is raised. Every document of a delta "
        "that keeps dying counts an attempt.",
    )
    lease_seconds: float = Field(
        default=300.0,
        ge=10.0,
        description="Incremental runs: the term of an index namespace's write "
        "lease, which a run holds from its first stage to its end so no other "
        "run writes the namespace meanwhile. The run renews it every third of "
        "the term; a run that dies stops renewing, so its lease expires (and "
        "another run can start) at most this long after.",
    )
    retire_source_scopes: list[str] = Field(
        default_factory=list,
        description="Incremental runs only: source scopes (a former "
        "document_parsing.source_scope value, or the absolute source directory "
        "a corpus was indexed from before it moved) whose registry records in "
        "this run's index namespace are treated as deleted. The run removes "
        "their exclusive artifacts (shared ones are kept) and deletes their "
        "records once the removal succeeded. Never the run's own scope. "
        "run-ingestion --retire-source-scope adds to this list.",
    )

    @field_validator("retire_source_scopes")
    @classmethod
    def _strip_retired_scopes(cls, scopes: list[str]) -> list[str]:
        stripped = [scope.strip() for scope in scopes]
        if not all(stripped):
            raise ValueError("retire_source_scopes entries must not be empty")
        return list(dict.fromkeys(stripped))


class HybridConfig(BaseModel):
    lexical_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the lexical side of a "
            "hybrid search. A hybrid search runs "
            "a lexical and a kNN search on each "
            "index, each fetching at least 50 "
            "hits, min-max normalizes each over "
            "its hits, takes their weighted mean "
            "(lexical_weight, vector_weight) and "
            "keeps the requested number, so the "
            "ranking does not depend on how many "
            "hits a query asks for."
        ),
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
        description=(
            "How results of different sources are fused: rrf (reciprocal rank "
            "fusion) or weighted (score-weighted)."
        ),
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description=(
            "The k constant of reciprocal rank fusion: each result adds 1 / "
            "(rrf_k + rank)."
        ),
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
        default_factory=dict,
        description=(
            "Per-source-bucket weights, applied by both fusion methods: RRF "
            "scales each bucket's 1/(rrf_k + rank) term and weighted fusion "
            "scales its scores. Keys are the source buckets the search "
            "strategies emit: local (graph_entities, text_units, relationships, "
            "bridge_relationships, community_reports, claims), LightRAG "
            "(lightrag_entities, lightrag_relationships, "
            "lightrag_endpoint_entities, lightrag_incident_relationships, "
            "lightrag_chunks, lightrag_linked_chunks), simple (opensearch_all), "
            "global (opensearch_community_reports, text_units) and drift "
            "(results). A bucket without a key weighs 1.0 (the default: none)."
        ),
    )


class RerankingConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description=(
            "Rerank the fused results with models.rerank. Bedrock rerank "
            "models are not offered in every region, and the Rerank API needs "
            "its own IAM statement (bedrock:Rerank on Resource '*')."
        ),
    )
    top_k: int = Field(
        default=100,
        ge=1,
        description=("Fused candidates sent to the reranker."),
    )
    required: bool = Field(
        default=False,
        description=(
            "Fail the query with RerankModelError when reranking is enabled "
            "but the rerank model cannot be built, the rerank call fails or it "
            "returns nothing usable. false keeps the fused order and reports "
            "the outcome in SearchResult.metadata['rerank']. No effect when "
            "reranking is disabled or providers.rerank.kind is none."
        ),
    )


class GlobalSearchConfig(BaseModel):
    """MS GraphRAG global search: map over a cut of the community hierarchy.

    The reports of one cut (``community_depth``), or of a root-down dynamic
    selection (``use_dynamic_selection``), are all mapped in token-sized
    batches; similarity only caps a cut larger than ``max_reports_per_level``.
    """

    community_depth: int = Field(
        default=2,
        ge=0,
        description=(
            "The cut of the community hierarchy whose reports global search "
            "maps over, counted from the root as MS GraphRAG's community_level "
            "(default 2 there too): the communities at this depth plus the "
            "leaves above it, so every entity is covered once. 0 = the root "
            "communities (fewest, broadest reports); a large value = the "
            "leaves. With use_dynamic_selection it bounds how deep the "
            "selection descends."
        ),
    )
    max_reports_per_level: int = Field(
        default=100,
        ge=1,
        description=(
            "Upper bound on the reports one cut (or one step of the dynamic "
            "selection) contributes. A cut with more reports is cut to the "
            "ones most similar to the query (a hybrid search restricted to the "
            "cut) and the search metadata says so (global_level_capped). "
            "Bounded by indexing.opensearch.max_query_size, the most hits one "
            "store request returns. Each report costs map-step input tokens."
        ),
    )
    use_dynamic_selection: bool = Field(
        default=False,
        description=(
            "MS GraphRAG dynamic community selection instead of a fixed cut: "
            "rate the root reports for the query with "
            "models.roles.community_relevance, descend into the children of "
            "the relevant ones (dropping a parent once a child is relevant), "
            "and rate the whole next depth when a depth yields none, down to "
            "community_depth. One fast-model call per rated report, "
            "processing.max_concurrency at a time; the relevant reports are "
            "then mapped. Off by default, as in MS GraphRAG."
        ),
    )
    relevance_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Dynamic selection keeps a report whose relevance (the 1-10 rating "
            "divided by 10) is at least this. 0.2 = anything rated above "
            "irrelevant, MS GraphRAG's threshold of 1 on its 0-5 scale: a "
            "pruned report prunes its whole subtree."
        ),
    )
    enable_map_reduce: bool = Field(
        default=True,
        description="Map the selected reports into scored key points. false = "
        "the reports go to the answer model as they are (budget-trimmed).",
    )
    map_reduce_min_results: int = Field(
        default=3,
        ge=1,
        description="Minimum community results before map-reduce synthesis is "
        "applied; below this the results are returned directly.",
    )
    map_context_tokens: int = Field(
        default=12000,
        ge=128,
        description=(
            "Report tokens packed into one map-step LLM call (MS GraphRAG's "
            "data_max_tokens, 12000). Reports are packed in order until the "
            "next one would exceed it; a longer report gets a call of its own. "
            "Map calls run processing.max_concurrency at a time."
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
            "Run the reduce step as its own LLM call that summarizes the "
            "packed key points, which the answer model then rewrites. Off by "
            "default: the packed points go straight to the answer model as one"
            " ranked context section, saving one LLM call per query and a "
            "second synthesis that can drop facts."
        ),
    )
    text_unit_slots: int = Field(
        default=0,
        ge=0,
        description=(
            "Text units of the communities behind the key points added to the "
            "context (retrieved for the query and reranked). 0 (default) = "
            "none: MS GraphRAG's global context is community reports only, and "
            "chunks next to them took most of the answer budget the reports "
            "and key points need. Local and mix search cover chunk evidence."
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
        "many text units (too generic to be discriminative for local search) "
        "from the context and the text-unit fan-out. The query's own matched "
        "(seed) entities are always kept.",
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
            "local's in-network relationships). These bridge edges carry the "
            "hops of a multi-hop chain, which the relationship vector query "
            "alone often misses. Needs "
            "indexing.opensearch.build_relationship_vector_index. false = only"
            " the vector query on the relationship index."
        ),
    )


class DriftSearchConfig(BaseModel):
    """MS GraphRAG DRIFT search: a community-report primer, then local search.

    The primer expands the query into a hypothetical answer (HyDE), reads the
    ``k_followups`` community reports most similar to it in ``primer_folds``
    calls and drafts intermediate answers plus follow-up queries. Each of
    ``n_depth`` rounds then runs the ``k_followups`` best-placed open follow-up
    queries as local searches, each answered by ``models.roles.drift_action`` with its
    own follow-up queries. The answer model reduces the ranked intermediate
    answers with the local evidence they came from.
    """

    k_followups: int = Field(
        default=20,
        ge=1,
        description=(
            "Community reports the primer reads, and follow-up queries run per"
            " round (MS GraphRAG drift_k_followups, 20). Each follow-up query "
            "is one local search plus one models.roles.drift_action call."
        ),
    )
    primer_folds: int = Field(
        default=5,
        ge=1,
        description="Primer calls the primer reports are split over (MS "
        "GraphRAG primer_folds, 5); each drafts its own intermediate answer and "
        "follow-up queries.",
    )
    n_depth: int = Field(
        default=3,
        ge=1,
        description="Rounds of follow-up queries (MS GraphRAG n_depth, 3). Up to "
        "k_followups * n_depth local searches per query.",
    )
    local_context_tokens: int = Field(
        default=12000,
        ge=128,
        description="Token budget of the local-search context one follow-up "
        "query is answered from (MS GraphRAG local_search_max_data_tokens).",
    )
    max_answer_tokens: int = Field(
        default=8000,
        ge=128,
        description="Token budget for the ranked intermediate answers handed to "
        "the answer model; the best-scored answers are taken first.",
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
    does not shrink the usable window. Budget a type leaves unused because all
    of its candidates fit goes to the types that still have candidates, by
    share (LightRAG mix/hybrid give it to chunks first, as upstream does).
    Raise ``text`` to favour verbatim source passages, raise ``community`` to
    favour high-level synthesis.
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
            "Retrieval context budget of the answer prompt, in tokens. 30,000 "
            "matches upstream LightRAG's total context budget (MS GraphRAG "
            "uses 12,000); a budget derived from a 1M-token window (~785K) "
            "would never bind, so the per-type budgets would never trim "
            "anything. Always clamped to what models.roles.answer_generation "
            "accepts beside its output reservation "
            "(aws.bedrock.default_max_output_tokens), with a warning. null "
            "derives it from that window minus the output reservation and "
            "context_window_headroom_ratio."
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


def _reject_auto_strategy(strategies: list[SearchStrategy]) -> list[SearchStrategy]:
    if SearchStrategy.AUTO in strategies:
        raise ValueError("auto_routable_strategies cannot include 'auto'")
    return strategies


class SearchConfig(BaseModel):
    auto_routable_strategies: Annotated[
        list[SearchStrategy], AfterValidator(_reject_auto_strategy)
    ] = Field(
        default_factory=lambda: [
            SearchStrategy.LOCAL,
            SearchStrategy.MIX,
            SearchStrategy.GLOBAL,
            SearchStrategy.DRIFT,
        ],
        min_length=1,
        description=(
            "Strategies the auto router may pick; its prompt describes exactly"
            " these, in this order. simple is left out by default because the "
            "graph strategies cover its direct lookups with graph context. "
            "global and drift are skipped when "
            "graph.community_detection.enabled is false (no community "
            "reports), and local is used when nothing is left. auto itself is "
            "rejected; every strategy stays selectable explicitly."
        ),
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
    store_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Seconds a query waits for one OpenSearch or Neptune request (and "
            "for opening a Neptune connection); a Neptune traversal, including"
            " its wait for a pooled connection, gets twice that. A store that "
            "does not answer fails the query instead of answering from no "
            "context. Ingestion keeps its own, longer limits."
        ),
    )
    store_max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Attempts of one OpenSearch query request that timed out, could "
            "not connect or got a 429/502/503/504, including the first, with a"
            " jittered backoff between them (0.25-0.5 s, then 0.5-1 s, "
            "doubling up to 2-4 s), so a short throttling burst is waited out."
            " A hung cluster fails a query after about store_max_attempts x "
            "store_timeout_seconds. Neptune query requests are not retried."
        ),
    )


class MemoryConfig(BaseModel):
    max_conversations: int = Field(
        default=100,
        ge=1,
        description=(
            "Conversations held in conversation memory; a new one beyond it "
            "drops the one idle longest."
        ),
    )
    max_messages_per_conversation: int = Field(
        default=20,
        ge=1,
        description=("Messages kept per conversation; older ones are dropped."),
    )
    max_conversation_age_hours: int = Field(
        default=168,
        ge=1,
        description="Hours a conversation may stay idle (since its last message); "
        "an older one is dropped on the next memory lookup and starts over",
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


class LLMResponseCacheConfig(BaseModel):
    """The ingestion LLM response cache (``adapters.llm.response_cache``)."""

    enabled: bool = Field(
        default=True,
        description="Keep the responses of the ingestion model calls with the "
        "run's stage cache, so a re-run with the same pipeline id (after a "
        "crash, a stop or a timeout) does not call the model again for a "
        "prompt that already answered. Query and evaluation calls are never "
        "cached",
    )
    flush_every: int = Field(
        default=100,
        ge=1,
        description="Responses per cache segment file; a segment is written "
        "(and uploaded with --s3-sync) when it is full and before each stage "
        "checkpoint, so a killed task loses at most this many responses",
    )


class CacheConfig(BaseModel):
    ttl_seconds: int | None = Field(
        default=86400,
        ge=1,
        description=(
            "Time-to-live of the local stage-cache entries, in seconds; null ="
            " never expire. Not applied with --s3-sync, where the bucket "
            "lifecycle expires the checkpoints."
        ),
    )
    chunking: CacheChunkingConfig = Field(
        default_factory=CacheChunkingConfig,
        description="Configuration for cache data chunking behavior",
    )
    llm_responses: LLMResponseCacheConfig = Field(
        default_factory=LLMResponseCacheConfig,
        description="Ingestion LLM response cache",
    )


class LoggingConfig(BaseModel):
    level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description=(
            "Log level: DEBUG, INFO, WARNING, ERROR or CRITICAL. DEBUG logs "
            "contain user and corpus text."
        ),
    )
    log_format: str = Field(
        default="structured",
        pattern="^(structured|plain)$",
        description=(
            "structured (JSON records, for CloudWatch and other log stores) or"
            " plain."
        ),
    )
    log_to_file: bool = Field(
        default=True,
        description=(
            "Also write the CLIs' logs to "
            "log_file_path (dated per day;"
            " a relative path resolves "
            "against the working "
            "directory). The package "
            "configures no handler when "
            "imported as a library."
        ),
    )
    log_file_path: str = Field(
        default="logs/log.txt",
        min_length=1,
        max_length=255,
        description="Log file path (CLIs only); relative to the working directory",
    )
    library_levels: dict[str, str] = Field(
        default_factory=lambda: {
            "langchain_aws": "WARNING",
            "botocore": "WARNING",
            "urllib3": "WARNING",
            # The OTLP exporter warns on every retry to an unreachable collector.
            "opentelemetry": "ERROR",
        },
        description=(
            "Per-logger levels for chatty third-party libraries (logger name "
            "-> level). Setting the key replaces the whole map."
        ),
    )


class ObservabilityBackend(str, Enum):
    """Where trace spans go when ``observability.enabled`` is true.

    ``none`` installs no exporter: spans go to the process's global
    OpenTelemetry tracer provider, which a host application or
    ``opentelemetry-instrument`` configures. ``otlp`` installs an OTLP
    exporter configured by the standard ``OTEL_EXPORTER_OTLP_*`` environment
    variables. ``langfuse`` and ``phoenix`` install the same OTLP exporter,
    pointed at that vendor's OTLP endpoint (from ``LANGFUSE_*`` /
    ``PHOENIX_*`` variables when the ``OTEL_*`` ones are unset), and add the
    vendor's attribute names (Langfuse observation types, OpenInference span
    kinds and token counts). No vendor SDK is needed.
    """

    NONE = "none"
    OTLP = "otlp"
    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"


class ObservabilityConfig(BaseModel):
    """OpenTelemetry tracing of ingestion and query runs (off by default).

    Needs the ``otel`` extra (``uv sync --extra otel``) for every backend. The
    exporter endpoint and headers come from environment variables
    (``OTEL_EXPORTER_OTLP_*``, or ``LANGFUSE_*`` / ``PHOENIX_*`` for those
    backends), never from this file, so no credential lands in a config file.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Emit OpenTelemetry spans for ingestion, queries and evaluation "
            "(needs the otel extra). OTEL_SDK_DISABLED=true turns it off "
            "regardless."
        ),
    )
    backend: ObservabilityBackend = Field(
        default=ObservabilityBackend.NONE,
        description=(
            "none: spans go to the process's global tracer provider, which a "
            "host application or opentelemetry-instrument sets up. otlp: the "
            "package installs an OTLP exporter configured by the "
            "OTEL_EXPORTER_OTLP_* variables. langfuse / phoenix: the same "
            "exporter pointed at Langfuse or Arize Phoenix, with that vendor's"
            " attribute names added."
        ),
    )
    capture_content: bool = Field(
        default=False,
        description=(
            "Also record prompt and completion text on model spans, and the "
            "query and answer on the kg_rag.query span (after "
            "redact_patterns). Off by default: spans carry counts, ids and "
            "token usage only. Leave it off for confidential corpora."
        ),
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of root traces sampled (backends otlp, langfuse, "
        "phoenix; with 'none' the host's provider samples); child spans follow "
        "their parent's decision",
    )
    service_name: str = Field(
        default="unified-kg-rag",
        min_length=1,
        description=(
            "OpenTelemetry service.name, and the Phoenix project; "
            "OTEL_SERVICE_NAME (and PHOENIX_PROJECT_NAME) win when set."
        ),
    )
    redact_patterns: list[str] = Field(
        default_factory=list,
        description="Regular expressions replaced with '[REDACTED]' in captured "
        "content and, for the backends this package installs (otlp, langfuse, "
        "phoenix), in every exported string attribute, event attribute and "
        "status message",
    )
    export_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
        description="Upper bound on one span-export request (backends otlp, "
        "langfuse, phoenix). OTEL_EXPORTER_OTLP_TRACES_TIMEOUT / "
        "OTEL_EXPORTER_OTLP_TIMEOUT win when set",
    )

    @field_validator("redact_patterns")
    @classmethod
    def _patterns_compile(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(
                    f"Invalid observability.redact_patterns entry {pattern!r}: {e}"
                ) from e
        return patterns


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
        description="Custom system prompt for the DRIFT primer (intermediate answer + follow-up queries)",
    )
    drift_primer_human: str | None = Field(
        default=None,
        description="Custom human prompt for the DRIFT primer (intermediate answer + follow-up queries)",
    )
    pairwise_comparison_system: str | None = Field(
        default=None,
        description="Custom system prompt for the run-eval --pairwise-judge LLM judge",
    )
    pairwise_comparison_human: str | None = Field(
        default=None,
        description="Custom human prompt for the run-eval --pairwise-judge LLM judge",
    )
    drift_hyde_system: str | None = Field(
        default=None,
        description="Custom system prompt for the DRIFT query expansion (HyDE)",
    )
    drift_hyde_human: str | None = Field(
        default=None,
        description="Custom human prompt for the DRIFT query expansion (HyDE)",
    )
    drift_action_system: str | None = Field(
        default=None,
        description="Custom system prompt for answering a DRIFT follow-up query "
        "from its local-search context",
    )
    drift_action_human: str | None = Field(
        default=None,
        description="Custom human prompt for answering a DRIFT follow-up query "
        "from its local-search context",
    )
    query_rewrite_system: str | None = Field(
        default=None,
        description="Custom system prompt for rewriting a follow-up message as a "
        "standalone question from the conversation history",
    )
    query_rewrite_human: str | None = Field(
        default=None,
        description="Custom human prompt for rewriting a follow-up message as a "
        "standalone question from the conversation history",
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

    @model_validator(mode="after")
    def _check_override_variables(self) -> "CustomPromptConfig":
        """Reject overrides that would fail or run on missing data at call time.

        A placeholder the prompt is not called with raises ``KeyError`` on every
        call (often a literal JSON brace), and an override without a data
        variable such as ``{input_text}`` runs with no document text. Other
        declared variables are tuning knobs an override may leave out.
        """
        # Lazy: the prompt modules reference this config type.
        from unified_kg_rag.domain import prompts

        errors: list[str] = []
        brace_hint = False
        for name in prompts.__all__:
            prompt = getattr(prompts, name)
            key = getattr(prompt, "prompt_key", None)
            if not key:
                continue
            system = getattr(self, f"{key}_system")
            human = getattr(self, f"{key}_human")
            if system is None and human is None:
                continue
            used: set[str] = set()
            for part, override, default in (
                ("system", system, prompt.system_prompt_template),
                ("human", human, prompt.human_prompt_template),
            ):
                try:
                    used |= _template_variables(override or default)
                except ValueError as e:
                    errors.append(f"custom_prompts.{key}_{part}: {e}")
                    brace_hint = True
            unknown = used - set(prompt.input_variables)
            if unknown:
                brace_hint = True
                errors.append(
                    f"custom_prompts.{key}_*: unknown variable(s) "
                    f"{', '.join('{' + v + '}' for v in sorted(unknown))}; "
                    f"available: "
                    f"{', '.join('{' + v + '}' for v in prompt.input_variables)}"
                )
            missing = _REQUIRED_PROMPT_VARIABLES.get(key, frozenset()) - used
            if missing:
                errors.append(
                    f"custom_prompts.{key}_*: the system or human template must "
                    f"contain {', '.join('{' + v + '}' for v in sorted(missing))}"
                )
            _warn_missing_output_tags(
                key,
                f"{system or prompt.system_prompt_template}\n"
                f"{human or prompt.human_prompt_template}",
                getattr(prompt, "required_output_tags", ()),
            )
        if errors:
            hint = " Write a literal brace as '{{' or '}}'." if brace_hint else ""
            raise ValueError("; ".join(errors) + "." + hint)
        return self


# The variables that carry each prompt's data: an override must keep them.
_REQUIRED_PROMPT_VARIABLES: dict[str, frozenset[str]] = {
    "answer_generation": frozenset({"query", "context"}),
    "claim_extraction": frozenset({"input_text"}),
    "community_report": frozenset({"entities", "relationships"}),
    "corpus_profile": frozenset({"corpus_sample"}),
    "description_summarization": frozenset({"descriptions"}),
    "drift_action": frozenset({"query", "context"}),
    "drift_hyde": frozenset({"query"}),
    "drift_primer": frozenset({"query", "community_reports"}),
    "entity_extraction": frozenset({"query"}),
    "global_map": frozenset({"query", "reports"}),
    "graph_extraction": frozenset({"input_text"}),
    "graph_refinement": frozenset({"text", "entities", "relationships"}),
    "keywords_extraction": frozenset({"query"}),
    "pairwise_comparison": frozenset({"query", "answer1", "answer2"}),
    "query_rewrite": frozenset({"query", "conversation_history"}),
    "strategy_selection": frozenset({"query", "strategies"}),
}


@functools.cache
def _warn_missing_output_tags(
    key: str, templates: str, required_tags: tuple[str, ...]
) -> None:
    """Warn when an override no longer teaches the XML tags its parser reads.

    A ``*_system`` override replaces the whole built-in system prompt, which is
    where the output schema lives for the XML-parsed prompts; without it the
    model answers in a free format the parser drops. Some users teach the
    format in prose or a human override, so this warns rather than fails.
    Once per override text: a config is validated several times per load
    (each layer, each rebuild under provider flags).
    """
    missing = [tag for tag in required_tags if tag not in templates]
    if missing:
        _logger.warning(
            "custom_prompts.%s_*: the system and human templates do not "
            "mention the output tag(s) %s that the response parser reads; "
            "keep the built-in output format (run-prompt-tuning output does) "
            "or results may be empty or ungrounded",
            key,
            ", ".join(missing),
        )


def _template_variables(template: str) -> set[str]:
    """Placeholder names in an f-string style template (``{{``/``}}`` are literal).

    Raises:
        ValueError: On an unmatched ``{`` or ``}``.
    """
    return {
        field.split(".")[0].split("[")[0]
        for _, field, _, _ in Formatter().parse(template)
        if field is not None
    }


class EvaluationConfig(BaseModel):
    outputs_directory: str | Path = Field(
        default="outputs/evaluation",
        description=("Directory the evaluation results are written to."),
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
    max_context_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Token cap on the retrieved context passed to the RAGAS judges. "
            "null (default) uses the answer model's budget "
            "(search.token_manager.max_context_tokens), so faithfulness and "
            "context_recall judge the answer against the context it was given; "
            "with that budget derived from the model window too, no extra cap "
            "applies (the contexts are already the answer model's selection). A "
            "value below the answer budget logs a warning: the judge then sees "
            "less context than the answer model did."
        ),
    )
    judge_effort: EffortLevel | None = Field(
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
    ragas_max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Total attempts per judge call, including the first (passed as "
            "RAGAS RunConfig.max_retries, which RAGAS applies as "
            "stop_after_attempt, with exponential backoff). Attempts count "
            "against ragas_timeout."
        ),
    )
    answer_normalization: Literal["default", "official_squad"] = Field(
        default="default",
        description=(
            "Normalization of the answer_match exact_match and token_f1 scores. "
            "'default' keeps meaning-bearing punctuation (minus signs, decimal "
            "points, 'C++') and folds Unicode; 'official_squad' is the official "
            "SQuAD v1.1 script (lowercase, ASCII punctuation and English "
            "articles removed), so scores compare with published MuSiQue / "
            "HotpotQA numbers. answer_contains always uses 'default'."
        ),
    )
    retrieval_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Cutoff k for the retrieval evaluator's hit@k / recall@k: the number "
            "of top-ranked distinct files (of the reported sources, in rank "
            "order) compared with reference_sources."
        ),
    )
    save_detailed_results: bool = Field(
        default=True,
        description="Whether to save detailed evaluation results and reports.",
    )


def _plain(value: Any) -> Any:
    """``value`` as plain data: a section instance as its full dump."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return copy.deepcopy(value)


def merge_over(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """``base`` with every leaf of ``top`` written over it (nested dicts merge)."""
    merged = dict(base)
    for key, value in top.items():
        below = merged.get(key)
        merged[key] = (
            merge_over(below, value)
            if isinstance(value, dict) and isinstance(below, dict)
            else value
        )
    return merged


def _changes(before: BaseModel, after: BaseModel) -> dict[str, Any]:
    """The fields of ``after`` that differ from ``before``, as nested data."""
    changes: dict[str, Any] = {}
    for name in type(after).model_fields:
        old, new = getattr(before, name), getattr(after, name)
        if isinstance(old, BaseModel) and type(old) is type(new):
            if nested := _changes(old, new):
                changes[name] = nested
        elif old != new:
            changes[name] = _plain(new)
    return changes


class Config(BaseModel):
    backend: Backend = Field(
        default=Backend.AWS,
        description=(
            "Provider preset. 'aws' (default): Bedrock, Neptune, OpenSearch and "
            "DynamoDB. 'local': fills in local providers (Ollama, a cross-encoder "
            "reranker, embedded SQLite graph/vector stores and registry) for every "
            "providers/aws.neptune/aws.opensearch key the config leaves unset."
        ),
    )
    providers: ProvidersConfig = Field(
        default_factory=ProvidersConfig,
        description="Per-role provider selection (models, stores, registry)",
    )
    models: ModelsConfig = Field(
        default_factory=ModelsConfig,
        description="Which model serves each role, on the providers above",
    )
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
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="OpenTelemetry tracing configuration",
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation configuration"
    )
    custom_prompts: CustomPromptConfig = Field(
        default_factory=CustomPromptConfig, description="Custom prompt configuration"
    )

    # The input this config was validated from, before its backend preset
    # (see as_input).
    _input: dict[str, Any] | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _apply_backend_preset(
        cls, data: Any, handler: ModelWrapValidatorHandler["Config"]
    ) -> "Config":
        """Refuse the moved keys, fill in the ``backend`` preset under the
        keys the config sets, and keep the input it was given."""
        if not isinstance(data, dict):
            return handler(data)
        if moved := moved_keys(data):
            raise ValueError("settings moved; replace " + "; ".join(moved))
        config = handler(apply_backend_preset(data))
        config._input = _plain(data)
        return config

    def __eq__(self, other: object) -> bool:
        """Equal settings make equal configs, whatever input they came from."""
        if not isinstance(other, Config):
            return NotImplemented
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def as_input(self) -> dict[str, Any]:
        """Input that validates to this config again, without its preset.

        The input it was validated from (the file and the environment, as
        given; not the ``backend`` preset's values, which validation fills
        in again), with every change made to the object since written over
        it. Rebuilding from it under another ``backend`` or provider keeps
        what the caller set, even a value equal to a preset's, and drops
        only what the old preset filled in.
        """
        if self._input is None:
            return self.model_dump(exclude_unset=True)
        built = type(self).model_validate(copy.deepcopy(self._input))
        return merge_over(copy.deepcopy(self._input), _changes(built, self))

    @model_validator(mode="after")
    def _check_models(self) -> "Config":
        """Check the models against the provider of their family.

        A Bedrock provider takes only catalog members for embedding and
        rerank (the vector dimension and rerank limits come from the
        catalog); another provider takes any name but needs one.
        """
        providers, models = self.providers, self.models
        errors: list[str] = []
        if providers.llm.kind != BEDROCK and models.default is None:
            errors.append(
                "models.default is required when providers.llm.kind is "
                f"'{providers.llm.kind}'"
            )
        embedding = models.embedding
        if providers.embedding.kind == BEDROCK:
            if embedding.model_id is not None and not isinstance(
                embedding.model_id, EmbeddingModelId
            ):
                errors.append(
                    f"models.embedding.model_id: '{embedding.model_id}' is not a "
                    "Bedrock embedding model; expected one of "
                    f"{[m.value for m in EmbeddingModelId]} (or select another "
                    "providers.embedding.kind)"
                )
        elif embedding.model_id is None or embedding.dimension is None:
            errors.append(
                "models.embedding.model_id and models.embedding.dimension are "
                "required when providers.embedding.kind is "
                f"'{providers.embedding.kind}'"
            )
        if providers.rerank.kind == BEDROCK:
            if models.rerank is not None and not isinstance(
                models.rerank, RerankModelId
            ):
                errors.append(
                    f"models.rerank: '{models.rerank}' is not a Bedrock rerank "
                    f"model; expected one of {[m.value for m in RerankModelId]} "
                    "(or select another providers.rerank.kind)"
                )
        elif providers.rerank.kind == CROSS_ENCODER and models.rerank is None:
            errors.append(
                "models.rerank is required when providers.rerank.kind is "
                f"'{CROSS_ENCODER}'"
            )
        if providers.graph.kind == GREMLIN and self.aws.neptune.use_iam:
            errors.append(
                "providers.graph.kind 'gremlin' needs aws.neptune.use_iam: false "
                "(SigV4 signing is Neptune-only)"
            )
        if errors:
            raise ValueError("; ".join(errors))
        return self

    # --- resolved models -----------------------------------------------------

    def tier_model_id(self, tier: ModelTier) -> str:
        """The model of ``tier``: ``models.<tier>``, else the provider's default."""
        models = self.models
        default = models.default or DEFAULT_MODEL_ID
        if tier == "default":
            return default
        if models.fast is not None:
            return models.fast
        return FAST_MODEL_ID if self.providers.llm.kind == BEDROCK else default

    def model_id(self, role: ModelRole) -> str:
        """The model that serves ``role``: its ``models.roles`` entry (a model
        id or a tier name), else its tier's model."""
        chosen: str | None = getattr(self.models.roles, role.value)
        if chosen is None:
            return self.tier_model_id(ROLE_TIERS[role])
        if chosen == "default":
            return self.tier_model_id("default")
        if chosen == "fast":
            return self.tier_model_id("fast")
        return chosen

    def model_tier(self, model_id: str) -> ModelTier:
        """Tier a call on ``model_id`` belongs to, for picking its effort.

        Only a model that is the fast tier's model (and not also the default
        tier's) counts as fast; anything else, including a role pinned to its
        own model, gets the default tier's effort.
        """
        model_id = str(model_id).strip()
        fast = self.tier_model_id("fast")
        if model_id == fast and fast != self.tier_model_id("default"):
            return "fast"
        return "default"

    @property
    def embedding_model_id(self) -> EmbeddingModelId | str:
        """The embedding model (``models.embedding.model_id``; Bedrock default
        Titan Embed V2)."""
        model_id = self.models.embedding.model_id
        return EmbeddingModelId.TITAN_EMBED_V2 if model_id is None else model_id

    @property
    def embedding_dimension(self) -> int | None:
        """``models.embedding.dimension`` (``None``: the Bedrock model's largest)."""
        return self.models.embedding.dimension

    @property
    def rerank_model_id(self) -> RerankModelId | str | None:
        """The rerank model (``models.rerank``; Bedrock default Cohere Rerank
        3.5); ``None`` with a reranker that takes no model."""
        if self.models.rerank is not None:
            return self.models.rerank
        if self.providers.rerank.kind == BEDROCK:
            return RerankModelId.COHERE_RERANK_V3_5
        return None

    def inactive_provider_settings(self) -> list[str]:
        """Settings the config sets that its providers do not read.

        The endpoint and sampling settings of ``providers.llm`` while the
        language models are Bedrock's, the Bedrock model knobs while they are
        not, the endpoint settings of ``providers.embedding`` while
        embeddings are Bedrock's, and the cross-encoder settings of
        ``providers.rerank`` under another reranker. Logged as a WARNING at
        load: such a setting is ignored.
        """
        providers = self.providers
        sections: list[tuple[str, BaseModel, frozenset[str], bool]] = [
            (
                "providers.llm",
                providers.llm,
                frozenset(LLMProviderConfig.model_fields) - {"kind"},
                providers.llm.kind == BEDROCK,
            ),
            (
                "aws.bedrock",
                self.aws.bedrock,
                frozenset(
                    {
                        "default_effort",
                        "fast_effort",
                        "ingestion_effort",
                        "enable_1m_context",
                        "model_overrides",
                    }
                ),
                providers.llm.kind != BEDROCK,
            ),
            (
                "providers.embedding",
                providers.embedding,
                frozenset(EmbeddingProviderConfig.model_fields) - {"kind"},
                providers.embedding.kind == BEDROCK,
            ),
            (
                "providers.rerank",
                providers.rerank,
                frozenset({"device", "max_length", "batch_size"}),
                providers.rerank.kind != CROSS_ENCODER,
            ),
        ]
        return [
            f"{prefix}.{name}"
            for prefix, section, names, inactive in sections
            if inactive
            for name in sorted(names & section.model_fields_set)
        ]

    @property
    def doc_status_enabled(self) -> bool:
        """Whether incremental indexing (the doc-status registry) is on:
        ``providers.doc_status.enabled``."""
        return self.providers.doc_status.enabled


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
    continue_on_error: bool = Field(
        default=False,
        description="If true, the pipeline continues execution even if a stage fails.",
    )
    force_rebuild: bool = Field(
        default=False,
        description="If true, ignores any existing cache and rebuilds all outputs.",
    )
    # The id names a local cache directory and an S3 prefix.
    pipeline_id: (
        Annotated[
            str, AfterValidator(lambda v: validate_path_segment(v, "pipeline_id"))
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "The ID of a previous pipeline run to resume (lowercase letters, "
            "digits, hyphens and underscores)."
        ),
    )
    resume_from_stage: str | None = Field(
        default=None,
        min_length=1,
        description="The name of the stage from which to resume execution.",
    )
