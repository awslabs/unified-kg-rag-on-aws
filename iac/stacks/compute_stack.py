# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config

        self.repository = ecr.Repository(
            self,
            "Repo",
            repository_name=f"{config.prefix}",
            image_scan_on_push=True,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=10)],
        )

        self.cluster = ecs.Cluster(
            self,
            "Cluster",
            vpc=networking.vpc,
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )

        self.log_group = logs.LogGroup(
            self,
            "TaskLogs",
            log_group_name=f"/aws-graphrag/{config.env_name}/tasks",
            retention=logs.RetentionDays.ONE_MONTH,
        )

        self.task_role = self._build_task_role(storage)
        self.task_definition = self._build_task_definition(networking, storage)
        self.security_groups = [networking.service_sg]
        self.subnets = networking.app_subnets

    # ------------------------------------------------------------ IAM role
    def _build_task_role(self, storage: StorageStack) -> iam.Role:
        role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="aws-graphrag Fargate task role (least privilege)",
        )
        # Bedrock model invocation (LLM/embedding/rerank).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=["*"],  # model ARNs are account/region scoped at call time
            )
        )
        # Neptune data-plane IAM auth.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["neptune-db:*"],
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
        task_def = ecs.FargateTaskDefinition(
            self,
            "IngestionTask",
            cpu=2048,
            memory_limit_mib=8192,
            task_role=self.task_role,
            ephemeral_storage_gib=40,
        )
        task_def.add_container(
            "app",
            image=ecs.ContainerImage.from_ecr_repository(self.repository, "latest"),
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="app", log_group=self.log_group
            ),
            environment={
                "AWS_REGION": self.region,
                "BEDROCK_REGION": self.config.bedrock_region or self.region,
                "NEPTUNE_ENDPOINT": storage.neptune_cluster.cluster_endpoint.hostname,
                "OPENSEARCH_ENDPOINT": (
                    f"https://{storage.opensearch_domain.domain_endpoint}"
                ),
                "S3_BUCKET_NAME": storage.cache_bucket.bucket_name,
                "GRAPHRAG_DOC_STATUS_TABLE": self.config.doc_status_table,
                "LOG_FORMAT": "json",
            },
        )
        return task_def
