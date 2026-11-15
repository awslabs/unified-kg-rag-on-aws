# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``aws`` section: Bedrock, guardrails, retries, Neptune, OpenSearch, S3, DynamoDB."""

from typing import Any, Literal

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from unified_kg_rag.domain.models.evidence import evidence

from ._common import ConfigSection, S3EncryptionType
from ._models import EffortLevel, ModelPurpose, ModelTier


class ContextualGroundingConfig(ConfigSection):
    """Bedrock Guardrails contextual grounding and relevance check on answers.

    Off by default. When on, the answer's ``ApplyGuardrail`` call (source
    OUTPUT) also carries the retrieved context the answer model was given
    (qualifier ``grounding_source``) and the question it answered
    (``query``), so the guardrail scores whether the answer is grounded in
    that context and relevant to the question, and blocks it below the
    thresholds configured on the guardrail itself (its
    ``contextualGroundingPolicyConfig``; ``iac`` adds one with
    ``-c guardrail_contextual_grounding=true``). A blocked answer takes the
    usual refusal path, with the failed checks in its guardrail record.

    Bedrock limits the check to 100,000 characters of reference, 1,000 of
    query and 5,000 of answer, and bills it per 1,000 characters of all
    three. A longer context is cut to its highest-ranked sections and a
    longer question to its head; a longer answer is screened without the
    check (logged). The Classic tier scores English, French and Spanish
    only: other languages, Korean included, score low and are blocked.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Send the retrieved context and the question with the answer's "
            "guardrail check, so the guardrail's contextual grounding policy "
            "scores and can block an ungrounded or irrelevant answer. The "
            "guardrail must have that policy; its Classic tier scores only "
            "English, French and Spanish. Needs check_output"
        ),
    )


class GuardrailConfig(ConfigSection):
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
    Ingestion and the evaluation judges run unguarded (run-eval's answers
    come from the screened query path): a PII-anonymizing
    guardrail rewrites names to placeholders (every person merges into one
    entity) and a prompt-attack filter blocks instruction-like corpus text,
    which silently yields empty extractions.
    """

    # Settings set in code are checked too: check_output turned off under
    # an enabled grounding check is refused, as in a file.

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
    contextual_grounding: ContextualGroundingConfig = Field(
        default_factory=ContextualGroundingConfig,
        description="Contextual grounding and relevance check on each answer "
        "against the retrieved context (off by default)",
    )

    @property
    def enabled(self) -> bool:
        return bool(self.identifier)

    @model_validator(mode="after")
    def validate_contextual_grounding(self) -> "GuardrailConfig":
        self.check_contextual_grounding()
        return self

    def check_contextual_grounding(self) -> None:
        """Refuse an enabled grounding check without ``check_output``.

        The grounding check rides on the answer's check: without it, an
        enabled check would silently never run. Checked at load and on
        assignment here, and again by the query chain when it is built: a
        refused assignment stays applied, and ``contextual_grounding.enabled``
        set in code is validated by its own section, which cannot see
        ``check_output``.

        Raises:
            ValueError: The grounding check is on and ``check_output`` off.
        """
        if self.contextual_grounding.enabled and not self.check_output:
            raise ValueError(
                "contextual_grounding.enabled requires check_output: the "
                "grounding check runs on the answer's guardrail check"
            )


RetryJitter = Literal["equal", "full", "none"]


class RetryPolicy(ConfigSection):
    """One caller's retry policy: attempts and backoff (a ``retry`` profile).

    Only errors a new attempt can fix are retried, by every caller; the
    others fail on the first attempt. The wait before retry ``n`` (1 for
    the first) is drawn under a ceiling of ``base_delay_seconds * 2^(n-1)``,
    at most ``max_delay_seconds`` (:meth:`ceiling`; the draw is
    ``adapters.llm.retry.backoff_delay``). A custom LLM factory may return
    one as its ``transient_retry``.
    """

    max_attempts: int = Field(
        default=5,
        ge=1,
        description="Attempts per call, including the first. 1 disables the retry.",
        json_schema_extra=evidence("operational"),
    )
    base_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Ceiling of the wait before the first retry, in seconds; it doubles "
            "per retry. 0 retries at once."
        ),
        json_schema_extra=evidence("operational"),
    )
    max_delay_seconds: float | None = Field(
        default=16.0,
        ge=0.0,
        description="Upper bound on one wait's ceiling, in seconds. null: none.",
        json_schema_extra=evidence("operational"),
    )
    max_total_seconds: float | None = Field(
        default=60.0,
        ge=0.0,
        description=(
            "Wall-clock retry budget per call, in seconds: no retry starts "
            "once the next wait would cross it. null: none (the deadline of "
            "the query a call is made for still stops a model or query-store "
            "retry)."
        ),
        json_schema_extra=evidence("operational"),
    )
    client_max_attempts: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Requests the service's own client makes for one attempt, the "
            "first included, retrying throttling, 5xx and timeouts itself "
            "before this profile's backoff: botocore's total_max_attempts on "
            "the Bedrock clients of ingestion and evaluation calls "
            "(retry.model; query-time calls take search.model_max_attempts), "
            "opensearch-py's max_retries + 1, back to back, on the OpenSearch "
            "indexing client (retry.bulk_item). The two stack: a call can make "
            "max_attempts x client_max_attempts requests. Set only on those "
            "two profiles; null leaves the client its library default."
        ),
        json_schema_extra=evidence("operational"),
    )
    jitter: RetryJitter = Field(
        default="equal",
        description=(
            "How a wait is drawn under its ceiling: 'equal' between half the "
            "ceiling and the ceiling, 'full' between 0 and the ceiling (spreads "
            "concurrent workers the most), 'none' the ceiling itself."
        ),
    )

    def ceiling(self, retry: int) -> float:
        """The wait ceiling before retry ``retry`` (1-based)."""
        ceiling = self.base_delay_seconds * 2.0 ** (retry - 1)
        if self.max_delay_seconds is not None:
            ceiling = min(ceiling, self.max_delay_seconds)
        return ceiling


# 0.3.0's name of the policy (exported then): deprecated, the same class.
TransientRetryConfig = RetryPolicy


# The profiles of the ``retry`` section and their defaults: each caller's
# values before the section had profiles. A profile a config sets in part
# keeps its own defaults for the rest (RetryConfig._profile_defaults).
RETRY_PROFILES: dict[str, dict[str, Any]] = {
    "model": {"client_max_attempts": 6},
    "ingestion_item": {
        "base_delay_seconds": 30.0,
        "max_delay_seconds": 120.0,
        "max_total_seconds": None,
        "jitter": "full",
    },
    "model_server": {"max_attempts": 3},
    "graph_store": {
        "max_attempts": 4,
        "max_delay_seconds": None,
        "max_total_seconds": None,
        "jitter": "full",
    },
    "query_store": {
        "max_attempts": 3,
        "base_delay_seconds": 0.5,
        "max_delay_seconds": 4.0,
        "max_total_seconds": None,
    },
    "bulk_item": {
        "base_delay_seconds": 1.0,
        "max_delay_seconds": 30.0,
        "max_total_seconds": None,
        "client_max_attempts": 6,
    },
    "registry_batch": {
        "max_attempts": 8,
        "base_delay_seconds": 0.1,
        "max_delay_seconds": 2.0,
        "max_total_seconds": None,
        "jitter": "none",
    },
}


# The profiles whose caller's client retries on its own (client_max_attempts).
_CLIENT_PROFILES = frozenset({"model", "bulk_item"})


def _profile(name: str) -> Any:
    return lambda: RetryPolicy(**RETRY_PROFILES[name])


class RetryConfig(ConfigSection):
    """Retries on errors a new attempt can fix, one profile per caller (``retry``).

    Each profile is a :class:`RetryPolicy` (attempts including the first,
    the backoff and its budget). Errors a retry cannot fix (validation,
    access denied, a malformed query) fail on the first attempt everywhere.
    A profile set in part keeps its own defaults for the keys it leaves out.
    """

    model: RetryPolicy = Field(
        default_factory=_profile("model"),
        description=(
            "Bedrock embedding requests and query-time model calls: the "
            "transient errors botocore does not retry (such as HTTP 424), "
            "after botocore's own attempts."
        ),
    )
    ingestion_item: RetryPolicy = Field(
        default_factory=_profile("ingestion_item"),
        description=(
            "Each ingestion LLM item (extraction, gleaning, claims, summaries, "
            "reports, chunking, translation): transient errors, call timeouts "
            "and output that does not parse, which is asked again without a "
            "wait. The long waits let a throttled Bedrock quota recover under "
            "processing.max_concurrency items."
        ),
    )
    model_server: RetryPolicy = Field(
        default_factory=_profile("model_server"),
        description=(
            "Requests to an OpenAI-compatible or Ollama model server "
            "(providers.llm, providers.embedding) on transient errors. The "
            "OpenAI client retries itself, max_attempts times; on Ollama, "
            "whose client does not, query chains and embedding calls are "
            "retried by the framework (ingestion items keep ingestion_item)."
        ),
    )
    graph_store: RetryPolicy = Field(
        default_factory=_profile("graph_store"),
        description=(
            "Each Neptune or Gremlin indexer request (writes, read-backs, drops "
            "and label clears), over a new connection after a lost one. "
            "Errors a retry cannot fix (malformed query, access denied, bad "
            "parameter, read-only violation) fail on the first attempt."
        ),
    )
    query_store: RetryPolicy = Field(
        default_factory=_profile("query_store"),
        description=(
            "One OpenSearch, Neptune or Gremlin query request that failed "
            "leaving the store usable: an OpenSearch timeout, refused "
            "connection or 429/502/503/504, a graph-store read throttled or "
            "out of memory (one stopped at its time limit, a timeout and a "
            "lost connection are not retried), so a short throttling burst is "
            "waited out. A hung OpenSearch cluster fails a query after about "
            "max_attempts x search.store_timeout_seconds."
        ),
    )
    bulk_item: RetryPolicy = Field(
        default_factory=_profile("bulk_item"),
        description=(
            "OpenSearch bulk items the cluster rejected with 429/502/503/504, "
            "resent on their own (the bulk request itself answered 200)."
        ),
    )
    registry_batch: RetryPolicy = Field(
        default_factory=_profile("registry_batch"),
        description=(
            "DynamoDB doc-status batch reads and writes: the keys or items "
            "DynamoDB returned unprocessed (throttling, the 16 MB response "
            "limit) are sent again; any left after max_attempts requests fail "
            "the run rather than read as absent."
        ),
    )
    judge_max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Attempts per RAGAS judge call (passed as RAGAS RunConfig."
            "max_retries, which RAGAS applies as stop_after_attempt, with "
            "its own exponential backoff). Attempts count against "
            "evaluation.ragas_timeout."
        ),
        json_schema_extra=evidence("operational"),
    )

    @model_validator(mode="before")
    @classmethod
    def _profile_defaults(cls, data: Any) -> Any:
        """A profile given in part keeps its profile's defaults for the rest,
        not :class:`RetryPolicy`'s."""
        if not isinstance(data, dict):
            return data
        return {
            key: (
                {**RETRY_PROFILES[key], **value}
                if key in RETRY_PROFILES and isinstance(value, dict)
                else value
            )
            for key, value in data.items()
        }

    @model_validator(mode="after")
    def _client_attempts_where_a_client_retries(self) -> "RetryConfig":
        for name in RETRY_PROFILES.keys() - _CLIENT_PROFILES:
            if getattr(self, name).client_max_attempts is not None:
                raise ValueError(
                    f"retry.{name}.client_max_attempts: that caller's client "
                    "makes one request per attempt; only "
                    + " and ".join(f"retry.{p}" for p in sorted(_CLIENT_PROFILES))
                    + " set one"
                )
        return self


class BedrockConfig(ConfigSection):
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
    query_prompt_cache_ttl: Literal["5m", "1h"] = Field(
        default="5m",
        description=(
            "Lifetime of the system-prompt cache checkpoint on query-time "
            "chains. A chain gets a checkpoint only when its system prompt "
            "reaches the model's minimum cacheable size (512 tokens on Claude "
            "Sonnet/Opus/Haiku 5.5 and Opus 5, 1,024 or 4,096 on other Claude "
            "4.5+ models): with the shipped prompts, answer generation, query "
            "entity extraction and the global-search reduce step on the "
            "512-token models. Ingestion chains and evaluation judges keep "
            "5m. '1h' keeps a query prompt cached across gaps "
            "of more than 5 minutes between queries, at a cost: a 1-hour cache "
            "write is billed at 2x the input-token price (5m: 1.25x), while a "
            "read costs the same. It pays off only at low query rates, when "
            "most queries would otherwise rewrite the cache. Applies to models "
            "whose capability record has supports_1h_cache_ttl (Claude 4.5+) "
            "on the Converse API (every inference profile); other models and "
            "InvokeModel requests keep the 5-minute checkpoint."
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
            "extraction, gleaning, claim extraction, description summaries, "
            "community reports, summary-tree summaries, chunk boundaries, "
            "translation, prompt tuning, and the output fixer when models.roles.fixing is a "
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


class NeptuneConfig(ConfigSection):
    endpoint: str | None = Field(
        default=None,
        description=(
            "Neptune cluster endpoint. Required with the neptune"
            " graph store (the CDK stack injects it as "
            "UKR__AWS__NEPTUNE__ENDPOINT)."
        ),
    )
    port: int = Field(
        default=8182,
        ge=1,
        le=65535,
        description="Neptune database connection port",
        json_schema_extra=evidence("upstream", "Neptune's port"),
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
        json_schema_extra=evidence("operational"),
    )


class OpenSearchConfig(ConfigSection):
    endpoint: str | None = Field(
        default=None,
        description=(
            "OpenSearch domain endpoint. Required with the "
            "opensearch vector store (the CDK stack injects it "
            "as UKR__AWS__OPENSEARCH__ENDPOINT)."
        ),
    )
    port: int = Field(
        default=443,
        ge=1,
        le=65535,
        description="OpenSearch cluster connection port",
        json_schema_extra=evidence(
            "upstream", "the HTTPS port of an OpenSearch Service domain"
        ),
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
            "SigV4 signing service for IAM auth. Only 'es' (a managed "
            "OpenSearch domain) is accepted: the setting exists so a config "
            "naming OpenSearch Serverless ('aoss') fails at load with the "
            "reason, as its vector collections assign their own document ids "
            "and have no refresh API, which the indexes need."
        ),
    )

    @field_validator("sigv4_service_name")
    @classmethod
    def _managed_domain_only(cls, value: str) -> str:
        if value != "es":
            raise ValueError(
                f"sigv4_service_name {value!r} is not supported: only 'es' (a "
                "managed OpenSearch domain). OpenSearch Serverless vector "
                "collections assign their own document ids and have no "
                "refresh API, which incremental indexing (upserts and deletes "
                "by id) needs."
            )
        return value

    @model_validator(mode="after")
    def validate_anonymous_auth(self) -> "OpenSearchConfig":
        basic = bool(self.username or self.password)
        if self.allow_anonymous and (self.use_iam or basic):
            raise ValueError(
                "allow_anonymous cannot be combined with use_iam or username/password"
            )
        # IAM signing replaces basic auth: the credentials would be ignored.
        if self.use_iam and basic:
            raise ValueError(
                "use_iam cannot be combined with username/password: requests "
                "are SigV4-signed and the basic-auth credentials would be "
                "ignored (unset UKR__AWS__OPENSEARCH__USERNAME/PASSWORD, or "
                "set use_iam: false)"
            )
        return self


class S3EncryptionConfig(ConfigSection):
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


class S3Config(ConfigSection):
    bucket_name: str | None = Field(
        default=None,
        description=("Bucket for the stage-cache sync (run-ingestion --s3-sync)."),
    )
    encryption: S3EncryptionConfig = Field(
        default_factory=S3EncryptionConfig,
        description="S3 server-side encryption configuration",
    )


class DynamoDBConfig(ConfigSection):
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
            "Create the doc-status table, with on-demand capacity, on first "
            "use. Set false when the table is managed by IaC (the CDK stack "
            "does)."
        ),
    )


class AWSConfig(ConfigSection):
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
