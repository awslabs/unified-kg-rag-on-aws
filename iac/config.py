# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment configuration resolved from CDK context.

All knobs are read from ``cdk.json`` ``context`` (or ``-c key=value`` on the CLI)
so a deployment can reuse existing infrastructure (VPC, S3 cache bucket) or
provision its own, and can run the data plane fully private (no NAT, VPC
endpoints only). Resolve once in ``app.py`` and thread the typed object into the
stacks.

Context keys (all optional; sensible defaults shown). A key of this shape
(lower-case snake_case) that is none of them fails the synth, so a typo
cannot leave its setting at the default:

  env_name            "dev"            stack/resource name prefix
  bedrock_region      account region   region for Bedrock model calls

  # --- Networking: reuse vs create, public vs private ---
  vpc_id              None             reuse an existing VPC (else create one)
  network_mode        "private"        "private"  -> isolated subnets + VPC
                                       endpoints, NO NAT (data plane has no
                                       internet egress);
                                       "public"   -> private-with-NAT subnets
  max_azs             2                AZs for a newly-created VPC (>=2:
                                       Neptune needs subnets in two AZs)
  sts_endpoint        False            private mode: add the STS interface
                                       endpoint, which only an image config
                                       with aws.bedrock.assumed_role_arn
                                       calls (task role credentials come from
                                       the ECS credentials endpoint)

  # --- Storage: reuse vs create ---
  cache_bucket_name   None             reuse an existing S3 cache bucket (else
                                       create one; SSE-S3, or the CMK when
                                       use_cmk=true)
  corpus_bucket_name  None             extra S3 bucket the corpus is read from
                                       (read-only grant + S3 endpoint policy);
                                       unset = the corpus lives in the cache bucket
  neptune_instance    "db.r6g.large"   Neptune instance class (Graviton)
  neptune_instances   dev:1/else:2     Neptune instances (>=2 => Multi-AZ HA)
  neptune_serverless_ncu None          "MIN-MAX" Neptune capacity units (e.g.
                                       "1-16"; 1-128 in steps of 0.5): run every
                                       instance as Neptune Serverless instead of
                                       neptune_instance; unset = provisioned
  opensearch_instance "r6g.large.search"  OpenSearch data node type (Graviton)
  opensearch_count    dev:1/else:2     OpenSearch data node count (>1 =>
                                       zone-aware multi-node + 3 dedicated masters;
                                       even when the domain spans 2 AZs)
  opensearch_master_instance "m6g.large.search"  dedicated master type when
                                       opensearch_count > 1 (8 GiB Graviton; sized
                                       for <=10 nodes / 10K shards on OS 2.13)
  opensearch_nori_package False        associate the AWS-provided analysis-nori
                                       plugin package of the domain's engine
                                       version (a custom resource; each change
                                       is a blue/green deployment)
  opensearch_korean_analyzer "cjk" (or "nori" with opensearch_nori_package)
                                       OpenSearch analyzer of Korean text
                                       (UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO, overrides
                                       language_analyzers.ko): the domain has no
                                       nori unless its optional analysis-nori
                                       package is associated
  doc_status_table    dev:"graphrag-doc-status-v2"/else:"<env>-graphrag-doc-status-v2"
  opensearch_volume_gib 50             GP3 volume per OpenSearch data node; the
                                       free-storage alarm fires at 40% of it
  backup_retention_days 7              Neptune automated backup retention
  neptune_maintenance_window "sun:20:00-sun:21:00"  weekly Neptune maintenance
                                       window (UTC, ddd:hh24:mi-ddd:hh24:mi, at
                                       least 30 minutes)
  neptune_backup_window "18:00-19:00"  daily Neptune backup window (UTC,
                                       hh24:mi-hh24:mi, at least 30 minutes; must
                                       not overlap the maintenance window)
  opensearch_off_peak_start None       start of the daily 10-hour OpenSearch
                                       off-peak window (UTC, "HH:MM"); unset =
                                       the service default, 10:00 P.M. local
                                       time of the region

  # --- Security ---
  create_guardrail    True             GuardrailStack owns (creates + keeps) a
                                       baseline guardrail; false = bring your own
  guardrail_identifier None            guardrail id the compute task USES
                                       (UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER); does NOT
                                       disable creation
  guardrail_version   None             published guardrail version the task
                                       applies (UKR__AWS__BEDROCK__GUARDRAIL__VERSION):
                                       the GuardrailVersion output; unset = the
                                       app default, DRAFT (synth warns)
  guardrail_contextual_grounding False  add a contextual grounding policy
                                       (GROUNDING + RELEVANCE, BLOCK) to the
                                       created guardrail and have the task send
                                       the retrieved context with each answer's
                                       check (UKR__AWS__BEDROCK__GUARDRAIL__CONTEXTUAL_GROUNDING__ENABLED)
  guardrail_grounding_threshold 0.7    minimum grounding and relevance score
                                       (0-0.99)
  use_cmk             False            customer-managed KMS key for at-rest
                                       encryption (S3/Neptune/OpenSearch/DDB); the
                                       alarm topic always has its own CMK
  vpc_flow_logs       dev:False/else:True  enable VPC flow logs (created VPC only)
  flow_log_retention_days 731         flow-log log group retention (a CloudWatch
                                       Logs retention value, e.g. 30/90/365/731)
  log_retention_days  30               retention of the task, pipeline,
                                       OpenSearch application and custom-
                                       resource log groups (a CloudWatch Logs
                                       retention value)
  deletion_protection dev:False/else:True  deletion protection on Neptune + the
                                       DynamoDB table (OpenSearch has none; it
                                       is only retained via removal_destroy)
  bedrock_model_arns  None             scope Bedrock IAM to specific model ARNs
                                       (list, or a comma-separated string with
                                       -c); None => every foundation model +
                                       inference profile. Replaces the
                                       foundation-model grant; the profile
                                       grants stay unless a profile ARN is
                                       listed
  alarm_email         None             subscribe an email to the alarm topic
  query_latency_alarm_seconds 60       p90 run-rag query latency the
                                       QueryLatencyHigh alarm fires above
                                       (run-rag --metrics-sink cloudwatch)
  observability_secret_arn None        complete ARN of a Secrets Manager secret
                                       whose JSON keys are injected into both
                                       task definitions as environment
                                       variables (ECS task secrets); in
                                       private mode, of the deploy region
  observability_secret_keys "LANGFUSE_PUBLIC_KEY,LANGFUSE_SECRET_KEY"
                                       JSON keys of that secret to inject, each
                                       as the variable of the same name (e.g.
                                       OTEL_EXPORTER_OTLP_HEADERS,
                                       PHOENIX_API_KEY)
  enable_cdk_nag      False            run cdk-nag AwsSolutions checks at synth
  enable_cloudwatch_tracing False      ADOT collector sidecar that sends the app's
                                       OpenTelemetry traces to CloudWatch
                                       (X-Ray OTLP endpoint); needs Transaction
                                       Search on in the account/region and an
                                       image built with the `otel` extra

  # --- Compute sizing (Fargate) ---
  fargate_cpu         2048             task vCPU units (in-task ProcessPool
                                       extractors scale with vCPU count)
  fargate_memory      8192             task memory (MiB)
  image_tag           latest           container image tag (pin a version for immutable ECR tags)
  fargate_cpu_architecture "x86_64"    CPU architecture of both task definitions:
                                       "x86_64", or "arm64" (Graviton) for an
                                       image built with --platform linux/arm64
  ingestion_ephemeral_storage_gib 40   ingestion task disk (GiB, 21-200): holds the
                                       synced corpus, the parsed documents and
                                       the run's stage cache
  sfn_timeout_hours   6                ingestion state machine timeout
  ingestion_fargate_spot False         run the ingestion task on Fargate Spot;
                                       a Spot interruption re-runs it with the
                                       same pipeline_id, which resumes from its
                                       S3 checkpoint
  spot_interruption_retries 2          re-runs after Spot interruptions (0-10;
                                       ingestion_fargate_spot only)
  spot_retry_wait_seconds 600          wait before each re-run (0-86400;
                                       ingestion_fargate_spot only): at least
                                       indexing.lease_seconds, so the lease of
                                       a task killed before it gave it up has
                                       expired

  # --- Governance (propagated as cost-allocation / ownership tags) ---
  owner               None             `owner` tag on every resource; unset =
                                       no owner tag
  cost_center         "unified-kg-rag-on-aws"   `cost-center` tag on every resource

  # --- Lifecycle ---
  removal_destroy     True (dev)       DESTROY vs RETAIN on stack deletion
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, fields
from typing import Any

_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))

# Fargate task sizes: CPU units -> valid memory (MiB) values.
_FARGATE_MEMORY: dict[int, frozenset[int]] = {
    256: frozenset((512, 1024, 2048)),
    512: frozenset(range(1024, 4097, 1024)),
    1024: frozenset(range(2048, 8193, 1024)),
    2048: frozenset(range(4096, 16385, 1024)),
    4096: frozenset(range(8192, 30721, 1024)),
    8192: frozenset(range(16384, 61441, 4096)),
    16384: frozenset(range(32768, 122881, 8192)),
}


def _as_bool(key: str, value: Any, default: bool) -> bool:
    """A boolean context value: a JSON boolean, or a ``-c key=...`` string.

    Anything else is rejected rather than read as false, so a typo such as
    ``-c use_cmk=ture`` cannot silently turn a security setting off.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise ValueError(
        f"{key} must be a boolean (true/false, yes/no, on/off, 1/0), got {value!r}"
    )


def _as_int(
    key: str, value: Any, minimum: int | None = None, maximum: int | None = None
) -> int:
    """An integer context value within ``[minimum, maximum]``."""
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer, got {value!r}") from None
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} must be at least {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{key} must be at most {maximum}, got {number}")
    return number


def _as_arn_list(key: str, value: Any) -> list[str] | None:
    """A list of ARNs: a JSON list, or a comma-separated ``-c key=...`` string."""
    if value is None:
        return None
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(
            f"{key} must be a list of ARNs or a comma-separated string, got {value!r}"
        )
    arns = [str(item).strip() for item in items if str(item).strip()]
    bad = [arn for arn in arns if not arn.startswith("arn:")]
    if bad:
        raise ValueError(f"{key}: not an ARN: {', '.join(bad)}")
    return arns or None


# Neptune Serverless capacity: 1-128 NCU in half-step increments.
_NCU_MIN, _NCU_MAX = 1.0, 128.0


def _as_ncu_range(key: str, value: Any) -> tuple[float, float] | None:
    """A ``(min, max)`` NCU range: ``"MIN-MAX"`` with ``-c``, or a JSON list."""
    if value is None:
        return None
    if isinstance(value, str):
        parts: list[Any] = value.split("-")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = []
    try:
        low, high = (float(part) for part in parts)
    except (TypeError, ValueError):
        raise ValueError(
            f'{key} must be "MIN-MAX" capacity units (e.g. "1-16"), got {value!r}'
        ) from None
    for ncu in (low, high):
        if not _NCU_MIN <= ncu <= _NCU_MAX or (ncu * 2) % 1:
            raise ValueError(
                f"{key}: capacity must be {_NCU_MIN:g}-{_NCU_MAX:g} NCU in steps "
                f"of 0.5, got {ncu:g}"
            )
    if low > high:
        raise ValueError(f"{key}: MIN ({low:g}) is above MAX ({high:g})")
    return low, high


def _as_threshold(key: str, value: Any, default: float) -> float:
    """A contextual grounding threshold: Bedrock accepts 0 to 0.99."""
    if value is None:
        return default
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {value!r}") from None
    if not 0 <= threshold <= 0.99:
        raise ValueError(f"{key} must be between 0 and 0.99, got {threshold:g}")
    return threshold


# Neptune windows (UTC): weekly "ddd:hh24:mi-ddd:hh24:mi", daily "hh24:mi-hh24:mi".
_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WEEKLY_WINDOW = re.compile(
    r"(mon|tue|wed|thu|fri|sat|sun):([01]\d|2[0-3]):([0-5]\d)-"
    r"(mon|tue|wed|thu|fri|sat|sun):([01]\d|2[0-3]):([0-5]\d)"
)
_DAILY_WINDOW = re.compile(r"([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)")
_WEEK_MINUTES = 7 * 24 * 60
_DAY_MINUTES = 24 * 60


def _window_minutes(start: int, end: int, cycle: int) -> set[int]:
    """The minutes a window covers, one cycle long, wrapping past its end."""
    length = (end - start) % cycle
    return {(start + minute) % cycle for minute in range(length)}


def _check_neptune_windows(maintenance: str, backup: str) -> None:
    """Neptune refuses a window under 30 minutes and overlapping windows at
    deploy time; refuse them at synth instead."""
    weekly = _WEEKLY_WINDOW.fullmatch(maintenance.lower())
    if not weekly:
        raise ValueError(
            "neptune_maintenance_window must be ddd:hh24:mi-ddd:hh24:mi in UTC "
            f"(e.g. sun:20:00-sun:21:00), got {maintenance!r}"
        )
    daily = _DAILY_WINDOW.fullmatch(backup)
    if not daily:
        raise ValueError(
            "neptune_backup_window must be hh24:mi-hh24:mi in UTC "
            f"(e.g. 18:00-19:00), got {backup!r}"
        )
    day1, h1, m1, day2, h2, m2 = weekly.groups()
    start = _DAYS.index(day1) * _DAY_MINUTES + int(h1) * 60 + int(m1)
    end = _DAYS.index(day2) * _DAY_MINUTES + int(h2) * 60 + int(m2)
    maintenance_minutes = _window_minutes(start, end, _WEEK_MINUTES)
    b1, n1, b2, n2 = (int(part) for part in daily.groups())
    backup_minutes = _window_minutes(b1 * 60 + n1, b2 * 60 + n2, _DAY_MINUTES)
    for key, minutes in (
        ("neptune_maintenance_window", maintenance_minutes),
        ("neptune_backup_window", backup_minutes),
    ):
        if len(minutes) < 30:
            raise ValueError(f"{key} must be at least 30 minutes long")
    daily_maintenance = {minute % _DAY_MINUTES for minute in maintenance_minutes}
    if daily_maintenance & backup_minutes:
        raise ValueError(
            f"neptune_backup_window {backup} overlaps neptune_maintenance_window "
            f"{maintenance}: Neptune needs them apart"
        )


def _as_window_start(key: str, value: Any) -> tuple[int, int] | None:
    """An "HH:MM" UTC start time."""
    if value is None:
        return None
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", str(value).strip())
    if not match:
        raise ValueError(f'{key} must be "HH:MM" in UTC (e.g. "02:00"), got {value!r}')
    return int(match.group(1)), int(match.group(2))


# A Secrets Manager secret's complete ARN ends in "-" and six random
# characters; ECS needs the complete ARN to grant and resolve it exactly.
_SECRET_ARN = re.compile(
    r"arn:aws[a-z-]*:secretsmanager:[a-z0-9-]+:\d{12}:secret:[\w/+=.@-]+-[A-Za-z0-9]{6}"
)
_ENV_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _as_secret_keys(key: str, value: Any) -> tuple[str, ...]:
    """Environment variable names: a JSON list, or a comma-separated string."""
    items = value.split(",") if isinstance(value, str) else list(value)
    names = tuple(str(item).strip() for item in items if str(item).strip())
    bad = [name for name in names if not _ENV_VAR_NAME.fullmatch(name)]
    if bad or not names:
        raise ValueError(
            f"{key} must name environment variables (e.g. "
            f"LANGFUSE_PUBLIC_KEY,LANGFUSE_SECRET_KEY), got {value!r}"
        )
    return names


def _check_fargate_size(cpu: int, memory: int) -> None:
    if cpu not in _FARGATE_MEMORY:
        raise ValueError(
            f"fargate_cpu must be one of {sorted(_FARGATE_MEMORY)}, got {cpu}"
        )
    if memory not in _FARGATE_MEMORY[cpu]:
        allowed = sorted(_FARGATE_MEMORY[cpu])
        raise ValueError(
            f"fargate_memory {memory} is not valid with fargate_cpu {cpu}: use "
            f"{allowed[0]}-{allowed[-1]} MiB in steps of "
            f"{allowed[1] - allowed[0]} MiB"
        )


# An OpenSearch analyzer name (no index-name characters such as "/" or ",").
_ANALYZER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


# Retention values CloudWatch Logs accepts (AWS::Logs::LogGroup RetentionInDays).
_LOG_RETENTION_DAYS = frozenset(
    (1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1096, 1827)
    + (2192, 2557, 2922, 3288, 3653)
)


# The shape of this app's context keys. CDK's own keys (feature flags such as
# "@aws-cdk/...", "aws:cdk:..." settings, cached lookups such as
# "availability-zones:...", "acknowledged-issue-numbers") and "_doc" notes
# never take it, so a key of this shape that is not a setting is a typo.
_SETTING_KEY = re.compile(r"[a-z][a-z0-9_]*")


def _check_context_keys(keys: list[str], known: frozenset[str]) -> None:
    """Refuse a context key that looks like a setting but is none: a
    misspelled key would otherwise leave its setting at the default."""
    unknown = sorted(
        key
        for key in keys
        if key not in known
        and (_SETTING_KEY.fullmatch(key) or key.replace("-", "_") in known)
    )
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key.replace("-", "_"), known, n=1)
        hints.append(f"{key} (did you mean {close[0]}?)" if close else key)
    raise ValueError(
        "Unknown context keys: " + ", ".join(hints) + " (see iac/config.py)"
    )


@dataclass(frozen=True)
class DeploymentConfig:
    env_name: str
    bedrock_region: str | None
    # networking
    vpc_id: str | None
    network_mode: str  # "private" | "public"
    max_azs: int
    # storage
    cache_bucket_name: str | None
    corpus_bucket_name: str | None
    neptune_instance: str
    neptune_instances: int
    opensearch_instance: str
    opensearch_count: int
    opensearch_master_instance: str
    opensearch_volume_gib: int
    doc_status_table: str
    backup_retention_days: int
    fargate_cpu: int
    fargate_memory: int
    ingestion_ephemeral_storage_gib: int
    # container image tag the Fargate task pulls. Default "latest" (mutable, for
    # local/dev iteration); set to an immutable version tag for non-dev, which
    # also flips the ECR repo to immutable tags (provenance). A tag, not a
    # digest: the ECR lifecycle rule keeps the deployed image by tag.
    image_tag: str
    sfn_timeout_hours: int
    # security
    # "create" and "use" are separate so the two-step deploy (create the
    # guardrail, then pass its id) never drops the guardrail from the template.
    create_guardrail: bool
    guardrail_identifier: str | None
    guardrail_version: str | None
    use_cmk: bool
    vpc_flow_logs: bool
    flow_log_retention_days: int
    deletion_protection: bool
    bedrock_model_arns: list[str] | None
    alarm_email: str | None
    enable_cdk_nag: bool
    # governance (propagated as cost-allocation / ownership tags)
    owner: str | None
    cost_center: str
    # lifecycle
    removal_destroy: bool
    # Whether `create_guardrail` was passed explicitly (vs. defaulted). Lets the
    # guardrail stack warn bring-your-own users who set only
    # `guardrail_identifier` that a baseline guardrail is now created as well.
    create_guardrail_explicit: bool = False
    # Opt-in ADOT collector sidecar exporting traces to CloudWatch (X-Ray OTLP
    # endpoint). Off: the templates are exactly as without the feature.
    enable_cloudwatch_tracing: bool = False
    # Analyzer the app maps Korean text to (UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO). The
    # domain has no nori unless the optional analysis-nori package is
    # associated with it; cjk (character bigrams) is built into OpenSearch.
    opensearch_korean_analyzer: str = "cjk"
    # Associate the AWS-provided analysis-nori package with the domain (a
    # custom resource). Off: the templates are exactly as without the feature.
    opensearch_nori_package: bool = False
    # Neptune Serverless (min, max) NCU; None = provisioned neptune_instance.
    neptune_serverless_ncu: tuple[float, float] | None = None
    # The STS interface endpoint in private mode (for aws.bedrock.
    # assumed_role_arn only). Off: no endpoint, nothing else changes.
    sts_endpoint: bool = False
    # Retention of the task, pipeline, OpenSearch and custom-resource logs.
    log_retention_days: int = 30
    # Fargate Spot for the ingestion task, and the re-runs of a task Spot
    # interrupted. Off: the templates are exactly as without the feature.
    ingestion_fargate_spot: bool = False
    spot_interruption_retries: int = 2
    spot_retry_wait_seconds: int = 600
    # CPU architecture of the Fargate tasks ("x86_64" | "arm64"). x86_64
    # sets no runtime platform, so the templates are as without the key.
    fargate_cpu_architecture: str = "x86_64"
    # Opt-in contextual grounding check on answers. Off: the templates are
    # exactly as without the feature.
    guardrail_contextual_grounding: bool = False
    guardrail_grounding_threshold: float = 0.7
    # Neptune maintenance and backup windows (UTC) and the start of the
    # OpenSearch off-peak window (UTC hour, minute; None = the region's
    # 10:00 P.M. local default).
    neptune_maintenance_window: str = "sun:20:00-sun:21:00"
    neptune_backup_window: str = "18:00-19:00"
    opensearch_off_peak_start: tuple[int, int] | None = None
    # p90 latency of run-rag queries (seconds) the QueryLatencyHigh alarm
    # fires above.
    query_latency_alarm_seconds: int = 60
    # Secrets Manager secret whose JSON keys become task environment
    # variables (Langfuse/OTLP credentials). None: no secrets in the tasks.
    observability_secret_arn: str | None = None
    observability_secret_keys: tuple[str, ...] = (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    )

    @property
    def collector_cache_prefix(self) -> str:
        """ECR pull-through cache prefix the collector image is pulled through."""
        return f"{self.prefix}-ecr-public"

    @property
    def is_dev(self) -> bool:
        return self.env_name.lower() == "dev"

    @property
    def stack_prefix(self) -> str:
        # PascalCase project name for CloudFormation stack ids. dev keeps the bare
        # "GraphRag" (backward-compatible); a non-dev env is suffixed (e.g.
        # "GraphRagProd") so two environments can coexist in one account/region
        # without stack-id collision.
        return "GraphRag" if self.is_dev else f"GraphRag{self.env_name.capitalize()}"

    @property
    def prefix(self) -> str:
        # Lowercase prefix for *physical resource* names (S3/ECR/DDB require
        # lowercase). dev keeps the bare "graphrag"; a non-dev env is prefixed
        # (e.g. "prod-graphrag") so physical names (S3 bucket, DDB table, ECR
        # repo, cluster, SFN) don't collide across environments in one
        # account/region — the prior code assumed one-env-per-account/region but
        # never enforced it.
        return "graphrag" if self.is_dev else f"{self.env_name.lower()}-graphrag"

    @property
    def is_private(self) -> bool:
        return self.network_mode == "private"

    @property
    def create_vpc(self) -> bool:
        return not self.vpc_id

    def check_deploy_region(self, deploy_region: str | None) -> None:
        """Fail the synth when private mode cannot reach a regional service
        the tasks call: its VPC interface endpoints (and no NAT) are in the
        deploy region only, so a call to another region has no route and
        hangs (Bedrock) or fails the task's start (the secret)."""
        if not self.is_private or not deploy_region:
            return
        if self.bedrock_region and self.bedrock_region != deploy_region:
            raise ValueError(
                f"network_mode=private but bedrock_region ('{self.bedrock_region}') "
                f"differs from the deploy region ('{deploy_region}'). In private mode "
                "there is no route to Bedrock in another region (VPC endpoints are "
                "created in the deploy region only), so runtime Bedrock calls would "
                "hang. Either set bedrock_region to the deploy region, or use "
                "network_mode=public (NAT) for cross-region Bedrock."
            )
        if self.observability_secret_arn:
            secret_region = self.observability_secret_arn.split(":")[3]
            if secret_region != deploy_region:
                raise ValueError(
                    "network_mode=private but observability_secret_arn is in "
                    f"'{secret_region}', not the deploy region ('{deploy_region}'). "
                    "The Secrets Manager endpoint is created in the deploy region "
                    "only, so no task could read the secret and none would start. "
                    "Replicate the secret to the deploy region and give that ARN, "
                    "or use network_mode=public (NAT)."
                )

    def cache_bucket(self, account: str, region: str) -> str:
        """Cache bucket name: the reused bucket, else the one storage creates.

        Known before the storage stack exists, so the networking stack can put
        it in the S3 endpoint policy without a cyclic cross-stack reference.
        """
        return self.cache_bucket_name or f"{self.prefix}-cache-{account}-{region}"

    @classmethod
    def from_context(cls, scope: Any) -> DeploymentConfig:
        _check_context_keys(
            list(scope.node.get_all_context()),
            frozenset(f.name for f in fields(cls)) - {"create_guardrail_explicit"},
        )

        def ctx(key: str, default: Any = None) -> Any:
            value = scope.node.try_get_context(key)
            return default if value is None else value

        def as_bool(key: str, default: bool) -> bool:
            return _as_bool(key, ctx(key), default)

        def as_int(
            key: str, default: int, minimum: int = 1, maximum: int | None = None
        ) -> int:
            return _as_int(key, ctx(key, default), minimum, maximum)

        env_name = str(ctx("env_name", "dev"))
        # Destructive defaults are env-aware: a dev stack stays convenient
        # (auto-destroy, no deletion protection), but any non-dev env defaults to
        # PROD-SAFE (retain stateful stores on stack delete + deletion protection)
        # so a stray `cdk destroy` cannot silently wipe the knowledge graph or the
        # doc-status registry. Explicit `-c removal_destroy=`/`-c
        # deletion_protection=` always override.
        is_dev = env_name.lower() == "dev"
        network_mode = str(ctx("network_mode", "private")).lower()
        if network_mode not in ("private", "public"):
            raise ValueError(
                f"network_mode must be 'private' or 'public', got '{network_mode}'"
            )
        flow_log_retention_days = as_int("flow_log_retention_days", 731)
        log_retention_days = as_int("log_retention_days", 30)
        for key, days in (
            ("flow_log_retention_days", flow_log_retention_days),
            ("log_retention_days", log_retention_days),
        ):
            if days not in _LOG_RETENTION_DAYS:
                raise ValueError(
                    f"{key} must be a CloudWatch Logs retention value "
                    f"{sorted(_LOG_RETENTION_DAYS)}, got {days}"
                )
        opensearch_nori_package = as_bool("opensearch_nori_package", default=False)
        # With the package associated the domain has nori, so Korean maps to
        # it unless the analyzer is set (cjk, say, until the ko indices are
        # rebuilt).
        opensearch_korean_analyzer = str(
            ctx(
                "opensearch_korean_analyzer",
                "nori" if opensearch_nori_package else "cjk",
            )
        )
        if not _ANALYZER_NAME.fullmatch(opensearch_korean_analyzer):
            raise ValueError(
                "opensearch_korean_analyzer must be an OpenSearch analyzer name "
                f"(e.g. cjk or nori), got {opensearch_korean_analyzer!r}"
            )
        neptune_serverless_ncu = _as_ncu_range(
            "neptune_serverless_ncu", ctx("neptune_serverless_ncu")
        )
        if neptune_serverless_ncu and ctx("neptune_instance") is not None:
            raise ValueError(
                "neptune_instance and neptune_serverless_ncu are exclusive: a "
                "serverless cluster has no instance class"
            )
        ingestion_fargate_spot = as_bool("ingestion_fargate_spot", default=False)
        spot_keys = [
            key
            for key in ("spot_interruption_retries", "spot_retry_wait_seconds")
            if ctx(key) is not None
        ]
        if spot_keys and not ingestion_fargate_spot:
            raise ValueError(
                f"{', '.join(spot_keys)} only apply with ingestion_fargate_spot=true"
            )
        fargate_cpu_architecture = str(ctx("fargate_cpu_architecture", "x86_64"))
        fargate_cpu_architecture = fargate_cpu_architecture.lower()
        if fargate_cpu_architecture not in ("x86_64", "arm64"):
            raise ValueError(
                "fargate_cpu_architecture must be 'x86_64' or 'arm64', got "
                f"{fargate_cpu_architecture!r}"
            )
        guardrail_contextual_grounding = as_bool(
            "guardrail_contextual_grounding", default=False
        )
        if (
            ctx("guardrail_grounding_threshold") is not None
            and not guardrail_contextual_grounding
        ):
            raise ValueError(
                "guardrail_grounding_threshold only applies with "
                "guardrail_contextual_grounding=true"
            )
        fargate_cpu = as_int("fargate_cpu", 2048)
        fargate_memory = as_int("fargate_memory", 8192)
        _check_fargate_size(fargate_cpu, fargate_memory)
        neptune_maintenance_window = str(
            ctx("neptune_maintenance_window", "sun:20:00-sun:21:00")
        ).lower()
        neptune_backup_window = str(ctx("neptune_backup_window", "18:00-19:00"))
        _check_neptune_windows(neptune_maintenance_window, neptune_backup_window)
        observability_secret_arn = ctx("observability_secret_arn")
        if observability_secret_arn is not None and not _SECRET_ARN.fullmatch(
            str(observability_secret_arn)
        ):
            raise ValueError(
                "observability_secret_arn must be the complete ARN of a Secrets "
                "Manager secret (ending in its 6-character suffix), got "
                f"{observability_secret_arn!r}"
            )
        if ctx("observability_secret_keys") is not None and not (
            observability_secret_arn
        ):
            raise ValueError(
                "observability_secret_keys only applies with observability_secret_arn"
            )
        config = cls(
            env_name=env_name,
            bedrock_region=ctx("bedrock_region"),
            vpc_id=ctx("vpc_id"),
            network_mode=network_mode,
            # The Neptune cluster needs subnets in at least two AZs, so a
            # one-AZ VPC can never synth.
            max_azs=as_int("max_azs", 2, minimum=2),
            cache_bucket_name=ctx("cache_bucket_name"),
            corpus_bucket_name=ctx("corpus_bucket_name"),
            neptune_instance=str(ctx("neptune_instance", "db.r6g.large")),
            # HA vs cost is env-driven, mirroring the destructive-default pattern:
            # dev defaults to a single instance/node (cheap, no failover), while a
            # non-dev env defaults to 2 (Neptune reader in another AZ; OpenSearch
            # zone-aware multi-node with dedicated masters). Explicit
            # `-c neptune_instances=`/`-c opensearch_count=` always override.
            # Neptune: one writer plus up to 15 read replicas.
            neptune_instances=as_int(
                "neptune_instances", 1 if is_dev else 2, maximum=16
            ),
            opensearch_instance=str(ctx("opensearch_instance", "r6g.large.search")),
            opensearch_count=as_int("opensearch_count", 1 if is_dev else 2),
            opensearch_master_instance=str(
                ctx("opensearch_master_instance", "m6g.large.search")
            ),
            opensearch_volume_gib=as_int("opensearch_volume_gib", 50, minimum=10),
            # Default DDB table name follows the same env-scoped prefix as the
            # other physical names (dev: "graphrag-doc-status-v2";
            # non-dev: "<env>-graphrag-doc-status-v2") so environments don't share a
            # registry table in one account/region.
            doc_status_table=str(
                ctx(
                    "doc_status_table",
                    (
                        "graphrag-doc-status-v2"
                        if is_dev
                        else f"{env_name.lower()}-graphrag-doc-status-v2"
                    ),
                )
            ),
            # Neptune keeps 1-35 days of automated backups.
            backup_retention_days=as_int("backup_retention_days", 7, maximum=35),
            fargate_cpu=fargate_cpu,
            fargate_memory=fargate_memory,
            # Fargate's range; 20 GiB is what a task gets without the setting.
            ingestion_ephemeral_storage_gib=as_int(
                "ingestion_ephemeral_storage_gib", 40, minimum=21, maximum=200
            ),
            image_tag=ctx("image_tag", "latest") or "latest",
            # A Standard state machine runs for up to one year.
            sfn_timeout_hours=as_int("sfn_timeout_hours", 6, maximum=8760),
            create_guardrail=as_bool("create_guardrail", default=True),
            guardrail_identifier=ctx("guardrail_identifier"),
            guardrail_version=ctx("guardrail_version"),
            use_cmk=as_bool("use_cmk", default=False),
            vpc_flow_logs=as_bool("vpc_flow_logs", default=not is_dev),
            flow_log_retention_days=flow_log_retention_days,
            deletion_protection=as_bool("deletion_protection", default=not is_dev),
            bedrock_model_arns=_as_arn_list(
                "bedrock_model_arns", ctx("bedrock_model_arns")
            ),
            alarm_email=ctx("alarm_email"),
            enable_cdk_nag=as_bool("enable_cdk_nag", default=False),
            owner=str(owner) if (owner := ctx("owner")) else None,
            cost_center=str(ctx("cost_center", "unified-kg-rag-on-aws")),
            removal_destroy=as_bool("removal_destroy", default=is_dev),
            create_guardrail_explicit=ctx("create_guardrail") is not None,
            enable_cloudwatch_tracing=as_bool(
                "enable_cloudwatch_tracing", default=False
            ),
            opensearch_korean_analyzer=opensearch_korean_analyzer,
            opensearch_nori_package=opensearch_nori_package,
            neptune_serverless_ncu=neptune_serverless_ncu,
            ingestion_fargate_spot=ingestion_fargate_spot,
            sts_endpoint=as_bool("sts_endpoint", default=False),
            log_retention_days=log_retention_days,
            spot_interruption_retries=as_int(
                "spot_interruption_retries", 2, minimum=0, maximum=10
            ),
            spot_retry_wait_seconds=as_int(
                "spot_retry_wait_seconds", 600, minimum=0, maximum=86400
            ),
            fargate_cpu_architecture=fargate_cpu_architecture,
            guardrail_contextual_grounding=guardrail_contextual_grounding,
            guardrail_grounding_threshold=_as_threshold(
                "guardrail_grounding_threshold",
                ctx("guardrail_grounding_threshold"),
                0.7,
            ),
            neptune_maintenance_window=neptune_maintenance_window,
            neptune_backup_window=neptune_backup_window,
            opensearch_off_peak_start=_as_window_start(
                "opensearch_off_peak_start", ctx("opensearch_off_peak_start")
            ),
            query_latency_alarm_seconds=as_int(
                "query_latency_alarm_seconds", 60, maximum=3600
            ),
            observability_secret_arn=observability_secret_arn,
            observability_secret_keys=_as_secret_keys(
                "observability_secret_keys",
                ctx(
                    "observability_secret_keys",
                    "LANGFUSE_PUBLIC_KEY,LANGFUSE_SECRET_KEY",
                ),
            ),
        )
        # ECR limits a pull-through cache repository prefix to 30 characters.
        if config.enable_cloudwatch_tracing and len(config.collector_cache_prefix) > 30:
            raise ValueError(
                "enable_cloudwatch_tracing: env_name is too long for the "
                f"collector's ECR pull-through cache prefix "
                f"'{config.collector_cache_prefix}' (at most 30 characters)"
            )
        return config
