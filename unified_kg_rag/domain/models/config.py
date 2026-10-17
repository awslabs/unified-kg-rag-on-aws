# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from .evaluation import EvaluationMetricType, EvaluatorType
from .provider_config import (
    BEDROCK,
    GREMLIN,
    Backend,
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

# The two shipped model tiers. Every per-role model field declares one of them
# and inherits aws.bedrock.default_model_id / aws.bedrock.fast_model_id unless
# set explicitly, so changing a tier default is a one-line edit here.
DEFAULT_MODEL_ID: str = LanguageModelId.CLAUDE_V5_5_SONNET.value
FAST_MODEL_ID: str = LanguageModelId.CLAUDE_V5_5_HAIKU.value
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


def _resolve_role_models(
    model: BaseModel,
    roles: dict[str, tuple[str | None, type[Enum] | None]],
    errors: list[str],
    path: str = "",
) -> BaseModel:
    """Copy of ``model`` with its embedding/rerank model ids resolved.

    ``roles`` maps a role to ``(default, catalog)``: an id the config did not
    set takes ``default`` (when not None), and an id outside ``catalog`` (when
    not None) is reported in ``errors``. Walks nested sections recursively.
    """
    updates: dict[str, Any] = {}
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        extra = field.json_schema_extra
        role = extra.get(_MODEL_ROLE_KEY) if isinstance(extra, dict) else None
        if isinstance(role, str) and role in roles:
            default, catalog = roles[role]
            if default is not None and name not in model.model_fields_set:
                if value != default:
                    updates[name] = value = default
            if catalog is not None and not isinstance(value, catalog):
                errors.append(
                    f"{path}{name}: '{value}' is not a Bedrock {role} model; "
                    f"expected one of {[m.value for m in catalog]} (or select "
                    f"another providers.{role}.kind)"
                )
        elif isinstance(value, BaseModel):
            resolved = _resolve_role_models(value, roles, errors, f"{path}{name}.")
            if resolved is not value:
                updates[name] = resolved
    if not updates:
        return model
    copied = model.model_copy()
    # As in _apply_model_tiers: inherited values stay out of model_fields_set.
    copied.__dict__.update(updates)
    return copied


class ModelPurpose(str, Enum):
    """Why a language model is being created, so per-path policies can differ.

    Callers pass it to the LLM factory (``get_model(model_purpose=...)``). The
    Bedrock factory uses it to attach a guardrail to ingestion and evaluation
    models under ``GuardrailConfig.apply_to: all`` (query models never get
    one: the query chain screens the user's query and answer instead), and
    query chains get a transient-error retry. Unspecified means ``QUERY``.
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
# json_schema_extra key marking a field as an embedding/rerank model id, so the
# Config validator can find every one of them (see _resolve_role_models).
_MODEL_ROLE_KEY = "model_role"


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

    ``apply_to`` scopes the guardrail beyond queries. The default ``"query"``
    leaves ingestion (chunking, translation, graph extraction, gleaning,
    claims, description summarization, community reports) and evaluation
    judges unguarded, because a PII-anonymizing guardrail rewrites names to
    placeholders (every person merges into one entity) and a prompt-attack
    filter blocks instruction-like corpus text, which silently yields empty
    extractions. ``"all"`` also attaches the guardrail to every ingestion and
    evaluation model call.
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
    apply_to: Literal["query", "all"] = Field(
        default="query",
        description=(
            "'query': only the user query and answer are screened; ingestion "
            "and evaluation models run unguarded. 'all': the guardrail is also "
            "attached to every ingestion and evaluation model call"
        ),
    )

    @property
    def enabled(self) -> bool:
        return bool(self.identifier)

    def applies_to(self, purpose: ModelPurpose) -> bool:
        """Whether a model created for ``purpose`` gets the guardrail attached.

        Never for ``QUERY`` models: the query path is screened at its edges
        (``check_input`` / ``check_output``) instead.
        """
        if not self.enabled or purpose is ModelPurpose.QUERY:
            return False
        return self.apply_to == "all"


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
    region_name: str | None = Field(
        default=None,
        min_length=1,
        description="Region for Bedrock model, embedding and rerank calls. "
        "None = aws.region_name, so a private-VPC deployment reaches Bedrock "
        "through its own region's VPC endpoint without extra config.",
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
            "from default_model_id). The shipped fast model, Claude Haiku 5.5, "
            "thinks adaptively, so this sets how much it reasons on fast-tier "
            "calls; it has no effect on a fast model that does not reason."
        ),
    )
    ingestion_effort: EffortLevel | None = Field(
        default=None,
        description=(
            "Reasoning effort for ingestion calls on the default tier (graph "
            "extraction, gleaning, claim extraction, community reports, prompt "
            "tuning, and the output fixer when fixing_model_id is a "
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
        default=None, description="Neptune database endpoint URL"
    )
    port: int = Field(
        default=8182, ge=1, le=65535, description="Neptune database connection port"
    )
    use_iam: bool = Field(
        default=True, description="Enable IAM authentication for Neptune"
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
    # us-west-2 offers every default model, including both rerank models,
    # which several regions (e.g. ap-northeast-2) do not.
    region_name: str = Field(
        default="us-west-2", min_length=1, description="AWS region name"
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

    @property
    def bedrock_region(self) -> str:
        """Region Bedrock calls go to: aws.bedrock_region, else aws.region_name."""
        return self.bedrock.region_name or self.region_name


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
        default=None,
        description=(
            "Index suffix the run writes to (OpenSearch index and Neptune label "
            "names); query with the same value as run-rag/run-eval --suffix. "
            "None = 'default'."
        ),
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
        description="Predominant source language of the corpus. Selects the "
        "OpenSearch analyzer of the original chunk text and drives the "
        "same-language no-op skip; translation itself is content-driven.",
    )
    target_language: LanguageCode = Field(
        default=LanguageCode.EN, description="Target language code for translation"
    )
    additional_target_languages: list[LanguageCode] | None = Field(
        default=None,
        description="Additional target languages for translation. Each one is "
        "indexed in its own translated_text_<language> field (with that "
        "language's analyzer) and searched lexically; the target language "
        "alone feeds extraction, embeddings and the answer context.",
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
        "Applied per text unit: an entity a chunk scored below this threshold is "
        "dropped from that chunk, with the chunk's relationships naming it. Set to "
        "0.0 to disable filtering. "
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
            "Maximum gleaning rounds per text unit. Each round after the first "
            "re-sends only the text units whose previous answer added a new "
            "entity or relationship; a unit stops once an answer adds nothing "
            "new. Set 1 for MS GraphRAG's single-gleaning default when "
            "ingestion cost matters more than graph recall."
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
        default=40,
        ge=1,
        description="LLM calls an ingestion stage keeps in flight: the next "
        "item starts as soon as one finishes, so this is both the peak and the "
        "steady concurrency. These stages are Bedrock-I/O-bound (CPU/memory "
        "near-idle), so this can be well above the CPU count. Bedrock quotas "
        "differ per account, model and Region: lower it if the logs show "
        "ThrottlingException retries.",
    )
    max_attempts: int = Field(
        default=5,
        ge=1,
        description="Total attempts per ingestion LLM item, including the "
        "first call (1 disables the retry)",
    )
    io_workers: int = Field(
        default=64,
        ge=1,
        description="Threads for blocking I/O on the query path (Bedrock calls "
        "LangChain runs in the event loop's default executor, Neptune "
        "traversals, reranking) in the CLIs and GraphRAGChain's sync methods. "
        "Python's default is min(32, CPUs + 4), which caps concurrent Bedrock "
        "calls on small tasks; async hosts call "
        "unified_kg_rag.shared.utils.configure_event_loop themselves.",
    )
    ignore_errors: bool = Field(
        default=False,
        description="Degrade instead of raising when a whole step fails "
        "(a batch call, a translation language, an evaluator, global "
        "map/reduce, the DRIFT primer, the auto router). Items whose own LLM "
        "call fails are always skipped and reported, whatever this says.",
    )
    deduplicate: bool = Field(
        default=False, description="Enable document deduplication"
    )
    resolution_method: ResolutionMethod = Field(
        default=ResolutionMethod.MINHASH,
        description="Entity/relationship resolution method",
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
        description="Similarity threshold for fuzzy entity resolution "
        "(fuzzy_entity_resolution) and for matching claim subjects and "
        "objects to entity names",
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
        description="Leiden modularity resolution: higher gives more, smaller "
        "communities; lower gives fewer, larger ones.",
    )
    random_state: int = Field(
        default=42,
        ge=1,
        description="Leiden random seed (a positive integer; the native Leiden "
        "rejects 0). The same graph and seed give the same communities.",
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
        description="Minimum entities per level-0 community (smaller ones are merged "
        "into the neighbor they share the most edge weight with)",
    )
    auto_resolution: bool = Field(
        default=False,
        description="Sweep auto_resolution_candidates at every hierarchy level and "
        "keep the most modular resolution. Off by default: maximizing modularity "
        "on the coarse upper-level graphs picks low resolutions that merge "
        "unrelated communities, which cost global search accuracy in testing.",
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


def _coerce_visualization_embeddings(value: dict[str, Any]) -> dict[str, Any]:
    """Validate ``bedrock_model_id`` as an ``EmbeddingModelId`` (YAML gives a str)."""
    if value.get("bedrock_model_id") is None:
        return value
    try:
        model_id = EmbeddingModelId(value["bedrock_model_id"])
    except ValueError as e:
        raise ValueError(
            f"bedrock_model_id: {e}; expected one of "
            f"{[m.value for m in EmbeddingModelId]}"
        ) from e
    return {**value, "bedrock_model_id": model_id}


class VisualizationConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Render visualizations during ingestion. Off by default: it "
        "adds an embedding call per node (embedding_method) and a "
        "dimensionality reduction to every ingest; render on demand with "
        "run-visualization from the data export_data writes instead.",
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
        "ingestion writes to '<cache.local_directory>/<pipeline_id>/visualization' "
        "so visualization_data.json is synced to S3 with the cache; other callers "
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
        description="Method for dimensionality reduction ('umap', 'tsne', 'pca').",
    )
    embeddings: Annotated[
        dict[str, Any], AfterValidator(_coerce_visualization_embeddings)
    ] = Field(
        default_factory=dict,
        description="Embedding parameters for node embedding. Only "
        "'bedrock_model_id' is honored (node embeddings come from the "
        "providers.embedding model factory; the embedding dimensionality is "
        "determined by the chosen model). When unset, the OpenSearch embedding "
        "model is reused.",
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
            "Number of Neptune write batches (and relationship edges, written "
            "one per request) to submit concurrently. 1 (default) "
            "preserves sequential indexing; >1 fans batches over a thread pool, "
            "multiplexed across the Gremlin connection pool (sized to at least "
            "this value). Each batch accumulates its own stats, merged "
            "on completion. NOTE (real-AWS finding): >1 can trigger Neptune "
            "ConcurrentModificationException when batches touch overlapping "
            "vertices/properties; the retry (exponential backoff) recovers but "
            "the conflict churn can make a large run SLOWER than sequential. "
            "Prefer 1 unless you have measured a net win on your data/cluster."
        ),
    )
    max_attempts: int = Field(
        default=4,
        ge=1,
        description="Total attempts per Neptune indexer request (writes, "
        "read-backs, drops and label clears), including the first (1 disables "
        "the retry). Every error is retried except one a retry cannot fix, "
        "which fails on the first attempt: a Neptune error code in "
        "PERMANENT_NEPTUNE_ERROR_CODES (malformed query, access denied, bad "
        "parameter, read-only violation, ...) or a client-side "
        "ValueError/TypeError.",
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
        description="Maximum number of neighbours one node contributes per "
        "traversal hop, so a hub entity cannot fan the expansion out unbounded",
    )
    entity_importance_source: Literal["rank", "degree", "none"] = Field(
        default="rank",
        description=(
            "Graph-expansion relevance orders entities by distance from the "
            "seeds first (seeds always first) and by this importance among "
            "entities at the same distance. 'rank' uses the indexed entity `rank`, "
            "'degree' counts the entity's edges at query time (MS GraphRAG ranks "
            "entities by degree), and each is normalized by the largest value in "
            "the same result. 'none' gives every entity a neutral 0.5, the "
            "previous behaviour (it read an `importance` property that entity "
            "vertices never store), so expansion ranks by distance alone."
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
        description="Build the OpenSearch relationship VECTOR index (embeds "
        "Relationship.description). LightRAG mix/hybrid query it for high-level "
        "keyword retrieval and incident edges; GraphRAG local for its "
        "relationship section and bridge relationships; simple and drift search "
        "it with the other indexes (auto routes to these strategies). Only "
        "global does without it. Set false to skip the relationship embeddings: "
        "mix/hybrid then lose high-level retrieval and local its relationship "
        "section. The Neptune relationship EDGES are always indexed regardless — "
        "both methodologies traverse them for graph expansion; this toggles only "
        "the vector index. Symmetric with graph.community_detection.enabled.",
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
    default_analyzer: str = Field(
        default="standard",
        min_length=1,
        description="OpenSearch text analyzer used when the language has no specific mapping",
    )
    language_analyzers: dict[str, str] = Field(
        default_factory=lambda: {"en": "english", "ko": "nori"},
        description="Maps a language code to its OpenSearch text analyzer; extend without code changes",
    )
    embedding_model_id: EmbeddingModelRef = Field(
        default=EmbeddingModelId.TITAN_EMBED_V2,
        description=(
            "Embedding model for vector generation. With the bedrock embedding "
            "provider, one of the EmbeddingModelId values (other ids are "
            "rejected because the vector dimension must be known for the index "
            "mapping); with another provider, any model name, defaulting to "
            "providers.embedding.model_id."
        ),
        json_schema_extra={_MODEL_ROLE_KEY: "embedding"},
    )
    embedding_dimension: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Vector embedding dimension. Must be one the model supports (Titan "
            "Embed V2: 256, 512 or 1024); None uses the model's largest."
        ),
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
        "weight) before upsert, instead of overwriting. Requires a graph "
        "adapter that supports read-back (an adapter without it degrades to "
        "overwrite). On by default: overwriting replaces an entity shared with "
        "unchanged documents by its delta-only description and text_unit_ids. "
        "Delta entities match stored ones by the full build's rule "
        "(processing.fuzzy_entity_resolution), which costs an entity-name "
        "projection per delta run.",
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
        "would otherwise be reported as a successful run. The same rate bounds "
        "the share of communities left without a report before the community "
        "detection stage fails; below it their documents are recorded FAILED "
        "and retried. Set to 1.0 to disable the partial-failure gate (only "
        "total write failures fail the indexing stage).",
    )
    max_document_failures: int = Field(
        default=3,
        ge=1,
        description="Incremental runs: a document recorded FAILED this many "
        "consecutive runs with unchanged content is no longer retried. It stays "
        "FAILED and is skipped as unchanged (with a WARNING) until its content "
        "changes or this value is raised.",
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
        description="Enable reranking of search results for improved relevance",
    )
    rerank_model_id: RerankModelRef = Field(
        default=RerankModelId.COHERE_RERANK_V3_5,
        description=(
            "Reranking model. With the bedrock rerank provider, one of the "
            "RerankModelId values (other ids are rejected); with another "
            "provider, any model name, defaulting to providers.rerank.model_id."
        ),
        json_schema_extra={_MODEL_ROLE_KEY: "rerank"},
    )
    top_k: int = Field(
        default=100,
        ge=1,
        description="Maximum number of top results to rerank",
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
            "community_relevance_model_id, descend into the children of the "
            "relevant ones (dropping a parent once a child is relevant), and "
            "rate the whole next depth when a depth yields none, down to "
            "community_depth. One LLM call per rated report, "
            "processing.max_concurrency at a time; the relevant reports are "
            "then mapped. Off by default, as in MS GraphRAG."
        ),
    )
    community_relevance_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model that rates report relevance for dynamic "
        "community selection",
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
    map_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model used for the map step of MS GraphRAG "
        "map-reduce (rates community-report key points 0-100). Rating is cheap, "
        "so a fast/cheap model (e.g. Haiku) is the sensible default.",
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
    map_reduce_model_id: BedrockModelId = role_model_field(
        "fast",
        description="Language model used for the reduce step (reduce_with_llm) "
        "and the concat-and-reduce fallback",
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
        description="Entity names passed to keyword expansion, from the "
        "newest iteration's results first",
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
        description=(
            "Minimum improvement ratio (unseen / retrieved results of an "
            "iteration) required to continue iterations. Independently, the "
            "loop also stops once two consecutive iterations each add fewer "
            "than 2 unseen results."
        ),
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


def _reject_auto_strategy(strategies: list[SearchStrategy]) -> list[SearchStrategy]:
    if SearchStrategy.AUTO in strategies:
        raise ValueError("auto_routable_strategies cannot include 'auto'")
    return strategies


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
            "Strategies the AUTO router may pick. Default: local, mix, global, "
            "drift. simple is left out because the graph strategies already "
            "cover its direct lookups with graph context; mix (LightRAG) is "
            "included for multi-hop fact chains, which it serves from the same "
            "index. global and drift are skipped when "
            "graph.community_detection.enabled is false (no community reports); "
            "local is used when nothing is left. Every strategy stays "
            "selectable explicitly."
        ),
    )
    query_rewrite_model_id: BedrockModelId = role_model_field(
        "fast",
        description=(
            "Language model identifier used for rewriting a follow-up message "
            "as a standalone question from the conversation history, before "
            "retrieval (memory queries with history only)"
        ),
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
    store_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Seconds a query waits for one OpenSearch or Neptune request (and "
            "for opening a Neptune connection) before giving up on it. A store "
            "that does not answer fails the query instead of answering from "
            "no context. Ingestion keeps its own, longer limits."
        ),
    )
    store_max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Attempts of one OpenSearch query request that timed out or got a "
            "429/502/503/504, including the first. Neptune query requests are "
            "not retried."
        ),
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
        description="Per-logger levels for chatty third-party libraries "
        "(logger name -> level)",
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
        description="Emit OpenTelemetry spans for ingestion and query runs",
    )
    backend: ObservabilityBackend = Field(
        default=ObservabilityBackend.NONE,
        description="'none' uses the process's global tracer provider; 'otlp' "
        "installs an OTLP exporter configured by OTEL_EXPORTER_OTLP_* variables; "
        "'langfuse' and 'phoenix' export over OTLP to that vendor with its "
        "attribute names added",
    )
    capture_content: bool = Field(
        default=False,
        description="Record prompt and completion text on model spans (after "
        "redact_patterns). Off by default: spans carry counts, ids and token "
        "usage only",
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
        description="OpenTelemetry service.name (OTEL_SERVICE_NAME wins when set)",
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
    "drift_primer": frozenset({"query", "community_reports"}),
    "entity_extraction": frozenset({"query"}),
    "global_map": frozenset({"query", "reports"}),
    "graph_extraction": frozenset({"input_text"}),
    "graph_refinement": frozenset({"text", "entities", "relationships"}),
    "keyword_expansion": frozenset({"query"}),
    "keywords_extraction": frozenset({"query"}),
    "query_refinement": frozenset({"original_query", "results_summary"}),
    "query_rewrite": frozenset({"query", "conversation_history"}),
    "strategy_selection": frozenset({"query", "strategies"}),
}


def _warn_missing_output_tags(
    key: str, templates: str, required_tags: tuple[str, ...]
) -> None:
    """Warn when an override no longer teaches the XML tags its parser reads.

    A ``*_system`` override replaces the whole built-in system prompt, which is
    where the output schema lives for the XML-parsed prompts; without it the
    model answers in a free format the parser drops. Some users teach the
    format in prose or a human override, so this warns rather than fails.
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
        description="Directory to save evaluation results",
    )
    embedding_model_id: EmbeddingModelRef = Field(
        default=EmbeddingModelId.TITAN_EMBED_V2,
        description=(
            "Embedding model for evaluation. One of the EmbeddingModelId values "
            "with the bedrock embedding provider; with another provider, any "
            "model name, defaulting to providers.embedding.model_id."
        ),
        json_schema_extra={_MODEL_ROLE_KEY: "embedding"},
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

    @model_validator(mode="before")
    @classmethod
    def _apply_backend_preset(cls, data: Any) -> Any:
        """Fill in the ``backend`` preset under the keys the config sets."""
        if not isinstance(data, dict):
            return data
        return apply_backend_preset(data)

    @model_validator(mode="after")
    def _inherit_tier_models(self) -> "Config":
        """Point every role-model field the user did not set at its tier model.

        The tier models are ``aws.bedrock.default_model_id`` /
        ``fast_model_id``, or ``providers.llm.default_model_id`` /
        ``fast_model_id`` when the language models do not come from Bedrock.
        """
        llm = self.providers.llm
        if llm.kind == BEDROCK:
            tiers = {
                "default": self.aws.bedrock.default_model_id,
                "fast": self.aws.bedrock.fast_model_id,
            }
        else:
            if llm.default_model_id is None:
                raise ValueError(
                    "providers.llm.default_model_id is required when "
                    f"providers.llm.kind is '{llm.kind}'"
                )
            tiers = {
                "default": llm.default_model_id,
                "fast": llm.fast_model_id or llm.default_model_id,
            }
        for name in type(self).model_fields:
            section = getattr(self, name)
            if isinstance(section, BaseModel):
                resolved = _apply_model_tiers(section, tiers)
                if resolved is not section:
                    self.__dict__[name] = resolved
        return self

    @model_validator(mode="after")
    def _resolve_provider_models(self) -> "Config":
        """Default and validate every embedding/rerank model id by provider.

        A Bedrock provider takes only catalog members (the vector dimension
        and rerank limits come from the catalog); another provider takes any
        name, and an id the config leaves unset follows the provider's
        ``model_id``.
        """
        providers = self.providers
        errors: list[str] = []
        roles: dict[str, tuple[str | None, type[Enum] | None]] = {}
        if providers.embedding.kind == BEDROCK:
            roles["embedding"] = (None, EmbeddingModelId)
        elif providers.embedding.model_id is None or (
            providers.embedding.dimension is None
        ):
            errors.append(
                "providers.embedding.model_id and providers.embedding.dimension "
                "are required when providers.embedding.kind is "
                f"'{providers.embedding.kind}'"
            )
            roles["embedding"] = (None, None)
        else:
            roles["embedding"] = (providers.embedding.model_id, None)
        if providers.rerank.kind == BEDROCK:
            roles["rerank"] = (None, RerankModelId)
        else:
            roles["rerank"] = (providers.rerank.model_id, None)
        if providers.graph.kind == GREMLIN and self.aws.neptune.use_iam:
            errors.append(
                "providers.graph.kind 'gremlin' needs aws.neptune.use_iam: false "
                "(SigV4 signing is Neptune-only)"
            )
        for name in type(self).model_fields:
            section = getattr(self, name)
            if isinstance(section, BaseModel):
                resolved = _resolve_role_models(section, roles, errors, f"{name}.")
                if resolved is not section:
                    self.__dict__[name] = resolved
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @property
    def doc_status_enabled(self) -> bool:
        """Whether incremental indexing (the doc-status registry) is on.

        ``providers.doc_status.enabled`` when set, else ``aws.dynamodb.enabled``
        (the original switch, so existing configs keep their behaviour).
        """
        enabled = self.providers.doc_status.enabled
        return self.aws.dynamodb.enabled if enabled is None else enabled


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
