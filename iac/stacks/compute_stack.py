# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute: ECR image, ECS cluster, and a Fargate task definition for the data
plane (run-ingestion / run-rag), plus a least-privilege task role.

The same task definition runs each Step Functions phase; the command and
``--resume-from-stage`` are supplied as container overrides by the orchestration
stack. Store endpoints are injected as the env vars the app reads
(NEPTUNE_ENDPOINT / OPENSEARCH_ENDPOINT / BEDROCK_REGION / S3_BUCKET_NAME).

With ``enable_cloudwatch_tracing`` the task also runs an AWS Distro for
OpenTelemetry (ADOT) collector sidecar: the app exports OTLP/HTTP to it on
localhost, and it signs (SigV4, task role) and forwards the spans to the
CloudWatch OTLP traces endpoint (``https://xray.<region>.amazonaws.com/v1/traces``).
"""

from __future__ import annotations

import json

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from constructs import Construct

from iac.config import DeploymentConfig
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.storage_stack import StorageStack

# ADOT collector v0.50.0, pinned by the multi-arch index digest so a re-pushed
# tag cannot change the sidecar. Bump both together (`docker buildx imagetools
# inspect public.ecr.aws/aws-observability/aws-otel-collector:<tag>`).
COLLECTOR_REPOSITORY = "aws-observability/aws-otel-collector"
COLLECTOR_VERSION = "v0.50.0"
COLLECTOR_DIGEST = (
    "sha256:7968fb60db6a2390a47ba6a2df029745638486e285c9b2487da1b722d0855a3e"
)
# Where the app's OTLP exporter sends spans: the collector in the same task
# (awsvpc tasks share one network namespace).
COLLECTOR_OTLP_HTTP = "127.0.0.1:4318"


class ComputeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: DeploymentConfig,
        networking: NetworkingStack,
        storage: StorageStack,
        kms_key=None,
        guardrail_identifier: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        self.kms_key = kms_key
        self.guardrail_identifier = guardrail_identifier

        # Immutable tags (a pushed image can't silently replace an existing tag)
        # preserve provenance/rollback and block a supply-chain tag-swap of the
        # running artifact. This is only compatible with a pinned version tag —
        # the mutable "latest" default (dev iteration) must stay MUTABLE so it
        # can be re-pushed. So mutability tracks whether a real tag was pinned.
        pinned_image = config.image_tag != "latest"
        # Fixed physical names (repo, log group) must go with the stack when
        # removal_destroy is set, or the next deploy collides with the
        # retained copy. Outside dev they are kept (image provenance, logs).
        removal_policy = (
            RemovalPolicy.DESTROY if config.removal_destroy else RemovalPolicy.RETAIN
        )
        self.repository = ecr.Repository(
            self,
            "Repo",
            repository_name=f"{config.prefix}-app",
            image_scan_on_push=True,
            # With use_cmk, image layers are encrypted with the shared CMK. ECR
            # adds its own KMS grant on the key, so pulls need no task-side KMS
            # permission. ECR cannot change a repository's encryption in place,
            # so without a CMK the property stays unset (default AES-256) to
            # keep existing repositories from being replaced.
            encryption=(ecr.RepositoryEncryption.KMS if self.kms_key else None),
            encryption_key=self.kms_key,
            image_tag_mutability=(
                ecr.TagMutability.IMMUTABLE
                if pinned_image
                else ecr.TagMutability.MUTABLE
            ),
            # Keep the 10 newest images, but never expire the one the task
            # definition runs: an image matched by a higher-priority rule is
            # exempt from lower-priority rules, and the deployed tag holds a
            # single image.
            lifecycle_rules=[
                ecr.LifecycleRule(
                    rule_priority=1,
                    description="Keep the deployed image_tag",
                    tag_status=ecr.TagStatus.TAGGED,
                    tag_pattern_list=[config.image_tag],
                    max_image_count=1,
                ),
                ecr.LifecycleRule(rule_priority=2, max_image_count=10),
            ],
            removal_policy=removal_policy,
            # A non-empty repository cannot be deleted.
            empty_on_delete=config.removal_destroy,
        )

        self.cluster = ecs.Cluster(
            self,
            "Cluster",
            cluster_name=f"{config.prefix}-cluster",
            vpc=networking.vpc,
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )

        self.log_group = logs.LogGroup(
            self,
            "TaskLogs",
            log_group_name=f"/{config.prefix}/tasks",
            encryption_key=self.kms_key,
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=removal_policy,
        )

        self.task_role = self._build_task_role(storage)
        self.task_definition = self._build_task_definition(networking, storage)

        # Let the task role use the shared CMK for the resources encrypted with
        # it (S3 cache, DynamoDB), so reads/writes can decrypt.
        if self.kms_key is not None:
            self.kms_key.grant_encrypt_decrypt(self.task_role)

    # ------------------------------------------------------------ IAM role
    def _build_task_role(self, storage: StorageStack) -> iam.Role:
        role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="unified-kg-rag-on-aws Fargate task role (least privilege)",
        )
        # Bedrock model invocation (LLM/embedding/rerank). Cross-region and
        # global inference profiles fan a single InvokeModel out to foundation
        # models in *several* regions, so the foundation-model resource must span
        # regions (`:*:`) for the call to authorize. Inference-profile ARNs come
        # in two shapes: account-scoped application profiles
        # (`:<account>:inference-profile/*`) and account-less system-defined
        # profiles (`::inference-profile/global.*`), so allow both. Explicit ARNs
        # replace the foundation-model grant. When they name no inference
        # profile, the profile grants are kept: models that are offered only
        # through a profile (resolved at runtime) would otherwise be denied,
        # and a profile call still needs the foundation-model ARNs listed, so
        # this grants no model beyond them. Listing any inference-profile ARN
        # scopes profiles to exactly the ARNs given.
        partition = self.partition
        profile_resources = [
            f"arn:{partition}:bedrock:*:{self.account}:inference-profile/*",
            f"arn:{partition}:bedrock:*::inference-profile/*",
        ]
        explicit = self.config.bedrock_model_arns
        if not explicit:
            bedrock_resources = [
                f"arn:{partition}:bedrock:*::foundation-model/*",
                *profile_resources,
            ]
        elif any("inference-profile/" in arn for arn in explicit):
            bedrock_resources = list(explicit)
        else:
            bedrock_resources = [*explicit, *profile_resources]
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=bedrock_resources,
            )
        )
        # The hybrid scorer's reranking step calls the Bedrock Rerank API
        # (cohere.rerank / amazon.rerank) — a distinct action from InvokeModel.
        # Real-AWS verification showed Rerank is denied when scoped to the
        # foundation-model/inference-profile ARNs above (it authorizes against a
        # different resource shape), so it needs its own statement; "*" is
        # acceptable for this read-only action. Without it, reranking silently
        # degrades to RRF-only (AccessDeniedException, caught by the scorer).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:Rerank"],
                resources=["*"],
            )
        )
        # The cross-region model resolver lists inference profiles to pick the
        # right global/regional profile id at runtime; this is an account-level
        # read action (no resource scoping available).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:ListInferenceProfiles"],
                resources=["*"],
            )
        )
        # Applying a Bedrock Guardrail on Converse/InvokeModel needs its own
        # action, scoped to guardrails in this account (the guardrail lives in
        # the Bedrock runtime region, which may differ from the deploy region).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:ApplyGuardrail"],
                resources=[f"arn:{partition}:bedrock:*:{self.account}:guardrail/*"],
            )
        )
        # Neptune data-plane IAM auth. `connect` alone authorizes opening the
        # WebSocket but NOT running Gremlin traversals — reads/writes need the
        # explicit data-access actions, else queries return HTTP 403. Scoped to
        # this cluster's resource id.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "neptune-db:connect",
                    "neptune-db:ReadDataViaQuery",
                    "neptune-db:WriteDataViaQuery",
                    "neptune-db:DeleteDataViaQuery",
                ],
                resources=[
                    f"arn:{partition}:neptune-db:{self.region}:{self.account}:"
                    f"{storage.neptune_cluster.cluster_resource_identifier}/*"
                ],
            )
        )
        # OpenSearch HTTP(S) operations on the domain.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["es:ESHttp*"],
                resources=[f"{storage.opensearch_domain.domain_arn}/*"],
            )
        )
        # DynamoDB doc-status registry + S3 cache (scoped to our resources).
        # The app only reads and writes cache objects (stage checkpoints,
        # embedding cache); expiry is the bucket lifecycle's job, so the role
        # gets no s3:DeleteObject*.
        storage.doc_status_table.grant_read_write_data(role)
        storage.cache_bucket.grant_read(role)
        storage.cache_bucket.grant_put(role)
        if self.config.corpus_bucket_name:
            # A corpus kept outside the cache bucket is only read (entrypoint
            # `aws s3 sync`); the S3 endpoint policy allows it too.
            s3.Bucket.from_bucket_name(
                self, "CorpusBucket", self.config.corpus_bucket_name
            ).grant_read(role)
        if self.config.enable_cloudwatch_tracing:
            # The collector sidecar signs span uploads to the CloudWatch OTLP
            # traces endpoint with this role. Write-only: PutTraceSegments is
            # what the CloudWatch guide's managed policies grant for it, and
            # PutSpans / PutSpansForIndexing are the X-Ray actions for
            # OpenTelemetry span uploads. None has resource-level scoping.
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=[
                        "xray:PutTraceSegments",
                        "xray:PutSpans",
                        "xray:PutSpansForIndexing",
                    ],
                    resources=["*"],
                )
            )
        return role

    # ----------------------------------------------------- task definition
    def _build_task_definition(
        self, networking: NetworkingStack, storage: StorageStack
    ) -> ecs.FargateTaskDefinition:
        # CPU/memory are config-driven: the in-task ProcessPool extractors scale
        # with vCPU count, so a larger corpus benefits from more CPU.
        task_def = ecs.FargateTaskDefinition(
            self,
            "IngestionTask",
            cpu=self.config.fargate_cpu,
            memory_limit_mib=self.config.fargate_memory,
            task_role=self.task_role,
            ephemeral_storage_gib=40,
        )
        environment = {
            "AWS_REGION": self.region,
            "BEDROCK_REGION": self.config.bedrock_region or self.region,
            "NEPTUNE_ENDPOINT": storage.neptune_cluster.cluster_endpoint.hostname,
            # Bare hostname — the OpenSearch adapter prepends the scheme itself
            # via use_ssl; passing https:// here yields https://[https://…].
            "OPENSEARCH_ENDPOINT": storage.opensearch_domain.domain_endpoint,
            "S3_BUCKET_NAME": storage.cache_bucket.bucket_name,
            # One replica per index on a multi-node domain, so losing a node or
            # an AZ keeps every shard available; a single node has nowhere to
            # place one (the index would stay yellow).
            "OPENSEARCH_NUMBER_OF_REPLICAS": str(
                min(self.config.opensearch_count - 1, 1)
            ),
            # Maps to aws.dynamodb.table_name. The table is IaC-managed (KMS,
            # PITR, alarms) and the task role has no dynamodb:CreateTable, so
            # disable app-side auto-creation. With aws.dynamodb.enabled, a
            # missing table (or any other registry error) fails the
            # document_loading stage with DocStatusRegistryError naming the
            # table and the ResourceNotFoundException, rather than an
            # AccessDenied on CreateTable or a silent full rebuild.
            # aws.dynamodb.enabled stays a config.yaml decision.
            "GRAPHRAG_DOC_STATUS_TABLE": self.config.doc_status_table,
            "GRAPHRAG_DOC_STATUS_CREATE_TABLE": "false",
            # "structured" => JSON-structured logs (CloudWatch-friendly); the
            # config model only accepts "structured" | "plain".
            "LOG_FORMAT": "structured",
        }
        if self.guardrail_identifier:
            environment["BEDROCK_GUARDRAIL_IDENTIFIER"] = self.guardrail_identifier
        if self.config.guardrail_version:
            environment["BEDROCK_GUARDRAIL_VERSION"] = self.config.guardrail_version
        if self.config.enable_cloudwatch_tracing:
            # Tracing on, backend otlp, pointed at the sidecar. Prompt and
            # completion capture stays at the config default (off).
            environment.update(
                {
                    "GRAPHRAG_OBSERVABILITY_ENABLED": "true",
                    "GRAPHRAG_OBSERVABILITY_BACKEND": "otlp",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://{COLLECTOR_OTLP_HTTP}",
                    "OTEL_RESOURCE_ATTRIBUTES": (
                        f"deployment.environment={self.config.env_name}"
                    ),
                }
            )
        app = task_def.add_container(
            "app",
            image=ecs.ContainerImage.from_ecr_repository(
                self.repository, self.config.image_tag
            ),
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="app", log_group=self.log_group
            ),
            environment=environment,
        )
        if self.config.enable_cloudwatch_tracing:
            collector = self._add_trace_collector(task_def)
            # Started first so the app's first spans find it; START, not
            # HEALTHY, so a collector that never turns healthy cannot hold the
            # app back. On stop, ECS stops the app first, then the collector,
            # which flushes what the app sent on exit.
            app.add_container_dependencies(
                ecs.ContainerDependency(
                    container=collector,
                    condition=ecs.ContainerDependencyCondition.START,
                )
            )
        return task_def

    # ------------------------------------------------ trace collector sidecar
    def _add_trace_collector(
        self, task_def: ecs.FargateTaskDefinition
    ) -> ecs.ContainerDefinition:
        """The ADOT collector: OTLP/HTTP on localhost in, CloudWatch out."""
        # Pulled through an ECR pull-through cache of ECR Public: the private
        # (no-NAT) data plane has no route to public.ecr.aws, but it reaches
        # this account's ECR through the VPC endpoints, and ECR fetches the
        # image upstream itself. The cached repository is created on the first
        # pull and is not removed with the stack. That first pull may itself
        # need an internet route when ECR is reached through an interface
        # endpoint, and ECS fails a task whose image (essential or not) it
        # cannot pull, so the README has the operator warm the cache from a
        # workstation after deploy: CollectorImageUri is the image to pull.
        prefix = self.config.collector_cache_prefix
        cache_rule = ecr.CfnPullThroughCacheRule(
            self,
            "CollectorImageCache",
            ecr_repository_prefix=prefix,
            upstream_registry="ecr-public",
            upstream_registry_url="public.ecr.aws",
        )
        repository = ecr.Repository.from_repository_name(
            self, "CollectorRepo", f"{prefix}/{COLLECTOR_REPOSITORY}"
        )
        # The first pull through the cache creates the repository and imports
        # the image, on this one repository only.
        task_def.add_to_execution_role_policy(
            iam.PolicyStatement(
                actions=["ecr:CreateRepository", "ecr:BatchImportUpstreamImage"],
                resources=[repository.repository_arn],
            )
        )
        task_def.node.add_dependency(cache_rule)
        CfnOutput(
            self,
            "CollectorImageUri",
            value=repository.repository_uri_for_digest(COLLECTOR_DIGEST),
            description="Pull once after deploy to fill the collector image cache",
        )

        region, suffix = self.region, self.url_suffix
        collector_config = {
            "extensions": {
                "health_check": {"endpoint": "127.0.0.1:13133"},
                "sigv4auth/traces": {"region": region, "service": "xray"},
            },
            "receivers": {
                "otlp": {"protocols": {"http": {"endpoint": COLLECTOR_OTLP_HTTP}}}
            },
            "processors": {
                # Hard container limit 512 MiB: refuse data before an OOM kill.
                "memory_limiter": {
                    "check_interval": "1s",
                    "limit_mib": 400,
                    "spike_limit_mib": 100,
                },
                # The endpoint takes at most 10,000 spans / 5 MB per request.
                "batch": {
                    "send_batch_size": 512,
                    "send_batch_max_size": 1000,
                    "timeout": "5s",
                },
            },
            "exporters": {
                "otlphttp/traces": {
                    "traces_endpoint": f"https://xray.{region}.{suffix}/v1/traces",
                    "compression": "gzip",
                    "auth": {"authenticator": "sigv4auth/traces"},
                    # Bounded so a shutdown flush against an unreachable
                    # endpoint ends inside ECS's 30 s stop timeout: the
                    # collector then exits 0 instead of being killed. No
                    # queue and no retry: the batch processor exports
                    # synchronously, so at shutdown only the in-flight export
                    # and its final batch remain (2 x 10 s), where a queue
                    # would drain every queued batch, each up to timeout plus
                    # retries. A failed batch is dropped; the app keeps
                    # running either way.
                    "timeout": "10s",
                    "sending_queue": {"enabled": False},
                    "retry_on_failure": {"enabled": False},
                }
            },
            "service": {
                "extensions": ["health_check", "sigv4auth/traces"],
                "pipelines": {
                    "traces": {
                        "receivers": ["otlp"],
                        "processors": ["memory_limiter", "batch"],
                        "exporters": ["otlphttp/traces"],
                    }
                },
                # No internal metrics endpoint (:8888) in the task.
                "telemetry": {"metrics": {"level": "none"}},
            },
        }
        return task_def.add_container(
            "otel-collector",
            image=ecs.ContainerImage.from_ecr_repository(repository, COLLECTOR_DIGEST),
            # Losing the collector loses traces, never the run.
            essential=False,
            cpu=256,
            memory_limit_mib=512,
            # ADOT reads its YAML (JSON is YAML) from this variable.
            environment={"AOT_CONFIG_CONTENT": json.dumps(collector_config)},
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="otel-collector", log_group=self.log_group
            ),
            health_check=ecs.HealthCheck(
                command=["CMD", "/healthcheck"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(10),
            ),
        )
