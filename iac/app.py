#!/usr/bin/env python3
# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""CDK app entry point for the aws-graphrag infrastructure.

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
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.observability_stack import ObservabilityStack
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.security_stack import SecurityStack
from iac.stacks.storage_stack import StorageStack

app = cdk.App()
config = DeploymentConfig.from_context(app)

# An explicit account/region is required because the VPC import + AZ-aware
# resources (Neptune/OpenSearch) do environment-specific lookups.
env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION"),
)
prefix = config.prefix


def stack_id(name: str) -> str:
    return f"{prefix}-{name}"


security = SecurityStack(app, stack_id("security"), config=config, env=env)
networking = NetworkingStack(app, stack_id("networking"), config=config, env=env)
storage = StorageStack(
    app,
    stack_id("storage"),
    config=config,
    networking=networking,
    kms_key=security.kms_key,
    env=env,
)
compute = ComputeStack(
    app,
    stack_id("compute"),
    config=config,
    networking=networking,
    storage=storage,
    kms_key=security.kms_key,
    guardrail_identifier=security.guardrail_identifier,
    env=env,
)
orchestration = OrchestrationStack(
    app,
    stack_id("orchestration"),
    config=config,
    networking=networking,
    compute=compute,
    cache_bucket_name=storage.cache_bucket.bucket_name,
    kms_key=security.kms_key,
    env=env,
)
ObservabilityStack(
    app,
    stack_id("observability"),
    config=config,
    orchestration=orchestration,
    env=env,
)

# Tag everything for cost allocation / ownership / governance. These propagate
# to every taggable resource in every stack (cost-allocation reports group on
# `project` + `env` + `cost-center`).
for tag_key, tag_value in {
    "project": "aws-graphrag",
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
