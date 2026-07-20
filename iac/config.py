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
  neptune_instances   dev:1/else:2     Neptune instances (>=2 => Multi-AZ HA)
  opensearch_instance "r6g.large.search"  OpenSearch data node type (Graviton)
  opensearch_count    dev:1/else:2     OpenSearch data node count (>1 =>
                                       zone-aware multi-node + 3 dedicated masters)
  doc_status_table    dev:"graphrag-doc-status"/else:"<env>-graphrag-doc-status"
  backup_retention_days 7              Neptune automated backup retention

  # --- Security ---
  guardrail_identifier None            attach an existing Bedrock guardrail
  use_cmk             False            customer-managed KMS key for at-rest
                                       encryption (S3/Neptune/OpenSearch/SNS/DDB)
  vpc_flow_logs       dev:False/else:True  enable VPC flow logs (created VPC only)
  deletion_protection dev:False/else:True  protect Neptune/OpenSearch from deletion
  bedrock_model_arns  None             scope Bedrock IAM to specific model ARNs
                                       (list); None => account/region foundation
                                       + inference-profile ARNs
  alarm_email         None             subscribe an email to the alarm topic
  enable_cdk_nag      False            run cdk-nag AwsSolutions checks at synth

  # --- Compute sizing (Fargate) ---
  fargate_cpu         2048             task vCPU units (in-task ProcessPool
                                       extractors scale with vCPU count)
  fargate_memory      8192             task memory (MiB)
  image_tag           latest           container image tag (pin a version for immutable ECR tags)

  # --- Governance (propagated as cost-allocation / ownership tags) ---
  owner               "aws-proserve"   `owner` tag on every resource
  cost_center         "unified-kg-rag-on-aws"   `cost-center` tag on every resource

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
    # container image tag the Fargate task pulls. Default "latest" (mutable, for
    # local/dev iteration); set to an immutable version tag or digest for
    # non-dev, which also flips the ECR repo to immutable tags (provenance).
    image_tag: str
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
        return cls(
            env_name=env_name,
            bedrock_region=ctx("bedrock_region"),
            vpc_id=ctx("vpc_id"),
            network_mode=network_mode,
            max_azs=int(ctx("max_azs", 2)),
            cache_bucket_name=ctx("cache_bucket_name"),
            neptune_instance=str(ctx("neptune_instance", "db.r6g.large")),
            # HA vs cost is env-driven, mirroring the destructive-default pattern:
            # dev defaults to a single instance/node (cheap, no failover), while a
            # non-dev env defaults to 2 (Neptune reader in another AZ; OpenSearch
            # zone-aware multi-node with dedicated masters). Explicit
            # `-c neptune_instances=`/`-c opensearch_count=` always override.
            neptune_instances=int(ctx("neptune_instances", 1 if is_dev else 2)),
            opensearch_instance=str(ctx("opensearch_instance", "r6g.large.search")),
            opensearch_count=int(ctx("opensearch_count", 1 if is_dev else 2)),
            # Default DDB table name follows the same env-scoped prefix as the
            # other physical names (dev: "graphrag-doc-status";
            # non-dev: "<env>-graphrag-doc-status") so environments don't share a
            # registry table in one account/region.
            doc_status_table=str(
                ctx(
                    "doc_status_table",
                    (
                        "graphrag-doc-status"
                        if is_dev
                        else f"{env_name.lower()}-graphrag-doc-status"
                    ),
                )
            ),
            backup_retention_days=int(ctx("backup_retention_days", 7)),
            fargate_cpu=int(ctx("fargate_cpu", 2048)),
            fargate_memory=int(ctx("fargate_memory", 8192)),
            image_tag=ctx("image_tag", "latest") or "latest",
            guardrail_identifier=ctx("guardrail_identifier"),
            use_cmk=as_bool(ctx("use_cmk"), default=False),
            vpc_flow_logs=as_bool(ctx("vpc_flow_logs"), default=not is_dev),
            deletion_protection=as_bool(ctx("deletion_protection"), default=not is_dev),
            bedrock_model_arns=ctx("bedrock_model_arns"),
            alarm_email=ctx("alarm_email"),
            enable_cdk_nag=as_bool(ctx("enable_cdk_nag"), default=False),
            owner=str(ctx("owner", "aws-proserve")),
            cost_center=str(ctx("cost_center", "unified-kg-rag-on-aws")),
            removal_destroy=as_bool(ctx("removal_destroy"), default=is_dev),
        )
