# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment configuration resolved from CDK context.

All knobs are read from ``cdk.json`` ``context`` (or ``-c key=value`` on the CLI)
so a deployment can reuse existing infrastructure (VPC, S3 cache bucket) or
provision its own, and can run the data plane fully private (no NAT, VPC
endpoints only). Resolve once in ``app.py`` and thread the typed object into the
stacks.

Context keys (all optional; sensible defaults shown):

  env_name            "dev"            stack/resource name prefix
  bedrock_region      account region   region for Bedrock model calls

  # --- Networking: reuse vs create, public vs private ---
  vpc_id              None             reuse an existing VPC (else create one)
  network_mode        "private"        "private"  -> isolated subnets + VPC
                                       endpoints, NO NAT (data plane has no
                                       internet egress);
                                       "public"   -> private-with-NAT subnets
  max_azs             2                AZs for a newly-created VPC

  # --- Storage: reuse vs create ---
  cache_bucket_name   None             reuse an existing S3 cache bucket (else
                                       create one, KMS-encrypted)
  neptune_instance    "db.r6g.large"   Neptune instance class (Graviton)
  neptune_instances   1                Neptune instances (>=2 => Multi-AZ HA)
  opensearch_instance "r6g.large.search"  OpenSearch data node type (Graviton)
  opensearch_count    2                OpenSearch data node count
  doc_status_table    "<env>-graphrag-doc-status"  DynamoDB table name
  backup_retention_days 7              Neptune automated backup retention

  # --- Security ---
  guardrail_identifier None            attach an existing Bedrock guardrail
  use_cmk             False            customer-managed KMS key for at-rest
                                       encryption (S3/Neptune/OpenSearch/SNS/DDB)
  vpc_flow_logs       False            enable VPC flow logs (created VPC only)
  deletion_protection False            protect Neptune/OpenSearch from deletion
  bedrock_model_arns  None             scope Bedrock IAM to specific model ARNs
                                       (list); None => account/region foundation
                                       + inference-profile ARNs
  alarm_email         None             subscribe an email to the alarm topic
  enable_cdk_nag      False            run cdk-nag AwsSolutions checks at synth

  # --- Compute sizing (Fargate) ---
  fargate_cpu         2048             task vCPU units (in-task ProcessPool
                                       extractors scale with vCPU count)
  fargate_memory      8192             task memory (MiB)

  # --- Governance (propagated as cost-allocation / ownership tags) ---
  owner               "aws-proserve"   `owner` tag on every resource
  cost_center         "aws-graphrag"   `cost-center` tag on every resource

  # --- Lifecycle ---
  removal_destroy     True (dev)       DESTROY vs RETAIN on stack deletion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    neptune_instance: str
    neptune_instances: int
    opensearch_instance: str
    opensearch_count: int
    doc_status_table: str
    backup_retention_days: int
    fargate_cpu: int
    fargate_memory: int
    # security
    guardrail_identifier: str | None
    use_cmk: bool
    vpc_flow_logs: bool
    deletion_protection: bool
    bedrock_model_arns: list[str] | None
    alarm_email: str | None
    enable_cdk_nag: bool
    # governance (propagated as cost-allocation / ownership tags)
    owner: str
    cost_center: str
    # lifecycle
    removal_destroy: bool

    @property
    def stack_prefix(self) -> str:
        # PascalCase project name for CloudFormation stack ids; carries no env
        # prefix (environments are separated by account/region).
        return "GraphRag"

    @property
    def prefix(self) -> str:
        # Lowercase prefix for *physical resource* names (S3/ECR require
        # lowercase). No env segment, to match the account's naming convention.
        return "graphrag"

    @property
    def is_private(self) -> bool:
        return self.network_mode == "private"

    @property
    def create_vpc(self) -> bool:
        return not self.vpc_id

    @property
    def create_cache_bucket(self) -> bool:
        return not self.cache_bucket_name

    @classmethod
    def from_context(cls, scope: Any) -> DeploymentConfig:
        def ctx(key: str, default: Any = None) -> Any:
            value = scope.node.try_get_context(key)
            return default if value is None else value

        def as_bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return default

        env_name = str(ctx("env_name", "dev"))
        network_mode = str(ctx("network_mode", "private")).lower()
        if network_mode not in ("private", "public"):
            raise ValueError(
                f"network_mode must be 'private' or 'public', got '{network_mode}'"
            )
        return cls(
            env_name=env_name,
            bedrock_region=ctx("bedrock_region"),
            vpc_id=ctx("vpc_id"),
            network_mode=network_mode,
            max_azs=int(ctx("max_azs", 2)),
            cache_bucket_name=ctx("cache_bucket_name"),
            neptune_instance=str(ctx("neptune_instance", "db.r6g.large")),
            neptune_instances=int(ctx("neptune_instances", 1)),
            opensearch_instance=str(ctx("opensearch_instance", "r6g.large.search")),
            opensearch_count=int(ctx("opensearch_count", 2)),
            doc_status_table=str(ctx("doc_status_table", "graphrag-doc-status")),
            backup_retention_days=int(ctx("backup_retention_days", 7)),
            fargate_cpu=int(ctx("fargate_cpu", 2048)),
            fargate_memory=int(ctx("fargate_memory", 8192)),
            guardrail_identifier=ctx("guardrail_identifier"),
            use_cmk=as_bool(ctx("use_cmk"), default=False),
            vpc_flow_logs=as_bool(ctx("vpc_flow_logs"), default=False),
            deletion_protection=as_bool(ctx("deletion_protection"), default=False),
            bedrock_model_arns=ctx("bedrock_model_arns"),
            alarm_email=ctx("alarm_email"),
            enable_cdk_nag=as_bool(ctx("enable_cdk_nag"), default=False),
            owner=str(ctx("owner", "aws-proserve")),
            cost_center=str(ctx("cost_center", "aws-graphrag")),
            removal_destroy=as_bool(ctx("removal_destroy"), default=True),
        )
