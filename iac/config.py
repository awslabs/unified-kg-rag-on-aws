# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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
  neptune_instance    "db.r5.large"    Neptune instance class
  opensearch_instance "r6g.large.search"  OpenSearch data node type
  opensearch_count    2                OpenSearch data node count
  doc_status_table    "<env>-graphrag-doc-status"  DynamoDB table name

  # --- Security ---
  guardrail_identifier None            attach an existing Bedrock guardrail
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
    opensearch_instance: str
    opensearch_count: int
    doc_status_table: str
    # security / lifecycle
    guardrail_identifier: str | None
    removal_destroy: bool

    @property
    def prefix(self) -> str:
        return f"{self.env_name}-graphrag"

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
            neptune_instance=str(ctx("neptune_instance", "db.r5.large")),
            opensearch_instance=str(ctx("opensearch_instance", "r6g.large.search")),
            opensearch_count=int(ctx("opensearch_count", 2)),
            doc_status_table=str(
                ctx("doc_status_table", f"{env_name}-graphrag-doc-status")
            ),
            guardrail_identifier=ctx("guardrail_identifier"),
            removal_destroy=as_bool(ctx("removal_destroy"), default=True),
        )
