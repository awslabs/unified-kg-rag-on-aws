#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CDK app entry point for the unified-kg-rag-on-aws infrastructure.

Assembles the modular stacks (networking -> storage -> compute -> orchestration
-> observability, plus security). Deployment is parameterized via cdk.json
context — see iac/config.py and iac/README.md. Reuses an existing VPC / S3 cache
bucket when provided, and supports a fully-private (no-NAT) data plane.
"""

from __future__ import annotations

import os

import aws_cdk as cdk

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.guardrail_stack import GuardrailStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.observability_stack import ObservabilityStack
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.security_stack import SecurityStack
from iac.stacks.storage_stack import StorageStack

app = cdk.App()
config = DeploymentConfig.from_context(app)

# An explicit account/region is required because the VPC import + AZ-aware
# resources (Neptune/OpenSearch) do environment-specific lookups.
account = os.getenv("CDK_DEFAULT_ACCOUNT")
deploy_region = os.getenv("CDK_DEFAULT_REGION")
env = cdk.Environment(account=account, region=deploy_region)
# The guardrail must be created in the Bedrock runtime region, which may differ
# from the deploy region (cross-region inference). Falls back to the deploy
# region when bedrock_region is unset.
bedrock_env = cdk.Environment(
    account=account, region=config.bedrock_region or deploy_region
)
stack_prefix = config.stack_prefix


def stack_id(name: str) -> str:
    # PascalCase stack ids (GraphRagNetwork, ...) — no env prefix.
    return f"{stack_prefix}{name}"


security = SecurityStack(app, stack_id("Security"), config=config, env=env)
# The guardrail is region-pinned to bedrock_region in its own stack. We avoid a
# cross-region CloudFormation reference (those rewrite SSM/export plumbing across
# every stack and cause export-in-use churn on the deploy-region stacks).
# Instead the guardrail id flows to compute via the `guardrail_identifier`
# context value: deploy the guardrail stack first, then pass its id with
# `-c guardrail_identifier=<id>` on subsequent deploys. When unset, compute
# simply injects no guardrail env var (guardrails disabled).
GuardrailStack(app, stack_id("Guardrail"), config=config, env=bedrock_env)
networking = NetworkingStack(app, stack_id("Network"), config=config, env=env)
storage = StorageStack(
    app,
    stack_id("Storage"),
    config=config,
    networking=networking,
    kms_key=security.kms_key,
    env=env,
)
compute = ComputeStack(
    app,
    stack_id("Compute"),
    config=config,
    networking=networking,
    storage=storage,
    kms_key=security.kms_key,
    guardrail_identifier=config.guardrail_identifier,
    env=env,
)
orchestration = OrchestrationStack(
    app,
    stack_id("Orchestration"),
    config=config,
    networking=networking,
    compute=compute,
    cache_bucket_name=storage.cache_bucket.bucket_name,
    kms_key=security.kms_key,
    env=env,
)
ObservabilityStack(
    app,
    stack_id("Observability"),
    config=config,
    orchestration=orchestration,
    storage=storage,
    env=env,
)

# Tag everything for cost allocation / ownership / governance. These propagate
# to every taggable resource in every stack (cost-allocation reports group on
# `project` + `env` + `cost-center`).
for tag_key, tag_value in {
    "project": "unified-kg-rag-on-aws",
    "env": config.env_name,
    "managed-by": "cdk",
    "owner": config.owner,
    "cost-center": config.cost_center,
}.items():
    cdk.Tags.of(app).add(tag_key, tag_value)

# Well-Architected checks at synth time (opt-in: -c enable_cdk_nag=true).
if config.enable_cdk_nag:
    from cdk_nag import AwsSolutionsChecks

    from iac import nag_suppressions

    nag_suppressions.apply(
        {
            "networking": networking,
            "storage": storage,
            "compute": compute,
            "orchestration": orchestration,
        },
        config,
    )
    cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))

app.synth()
