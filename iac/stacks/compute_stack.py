# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute: ECR image, ECS cluster, and a Fargate task definition for the data
plane (run-ingestion / run-rag), plus a least-privilege task role.

The same task definition runs each Step Functions phase; the command and
``--resume-from-stage`` are supplied as container overrides by the orchestration
stack. Store endpoints are injected as the env vars the app reads
(NEPTUNE_ENDPOINT / OPENSEARCH_ENDPOINT / BEDROCK_REGION / S3_BUCKET_NAME).
"""

from __future__ import annotations

from aws_cdk import Stack
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from constructs import Construct

from iac.config import DeploymentConfig
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.storage_stack import StorageStack


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

        self.repository = ecr.Repository(
            self,
            "Repo",
            repository_name=f"{config.prefix}-app",
            image_scan_on_push=True,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=10)],
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
            retention=logs.RetentionDays.ONE_MONTH,
        )

        self.task_role = self._build_task_role(storage)
        self.task_definition = self._build_task_definition(networking, storage)
        self.security_groups = [networking.service_sg]
        self.subnets = networking.app_subnets

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
            description="aws-graphrag Fargate task role (least privilege)",
        )
        # Bedrock model invocation (LLM/embedding/rerank). Cross-region and
        # global inference profiles fan a single InvokeModel out to foundation
        # models in *several* regions, so the foundation-model resource must span
        # regions (`:*:`) for the call to authorize. Inference-profile ARNs come
        # in two shapes: account-scoped application profiles
        # (`:<account>:inference-profile/*`) and account-less system-defined
        # profiles (`::inference-profile/global.*`), so allow both. Explicit ARNs
        # override this when provided.
        bedrock_resources = self.config.bedrock_model_arns or [
            "arn:aws:bedrock:*::foundation-model/*",
            f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
            "arn:aws:bedrock:*::inference-profile/*",
        ]
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
        # The cross-region model resolver lists/inspects inference profiles to
        # pick the right global/regional profile id at runtime; these are
        # account-level read actions (no resource scoping available).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:ListInferenceProfiles",
                    "bedrock:GetInferenceProfile",
                ],
                resources=["*"],
            )
        )
        # Applying a Bedrock Guardrail on Converse/InvokeModel needs its own
        # action, scoped to guardrails in this account (the guardrail lives in
        # the Bedrock runtime region, which may differ from the deploy region).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:ApplyGuardrail"],
                resources=[f"arn:aws:bedrock:*:{self.account}:guardrail/*"],
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
                    "neptune-db:GetEngineStatus",
                    "neptune-db:GetQueryStatus",
                    "neptune-db:CancelQuery",
                ],
                resources=[
                    f"arn:aws:neptune-db:{self.region}:{self.account}:"
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
        storage.doc_status_table.grant_read_write_data(role)
        storage.cache_bucket.grant_read_write(role)
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
            "GRAPHRAG_DOC_STATUS_TABLE": self.config.doc_status_table,
            # "structured" => JSON-structured logs (CloudWatch-friendly); the
            # config model only accepts "structured" | "plain".
            "LOG_FORMAT": "structured",
        }
        if self.guardrail_identifier:
            environment["BEDROCK_GUARDRAIL_IDENTIFIER"] = self.guardrail_identifier
        task_def.add_container(
            "app",
            image=ecs.ContainerImage.from_ecr_repository(self.repository, "latest"),
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="app", log_group=self.log_group
            ),
            environment=environment,
        )
        return task_def
