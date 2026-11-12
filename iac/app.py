#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CDK app entry point for the unified-kg-rag-on-aws infrastructure.

Assembles the modular stacks (networking -> storage -> compute -> orchestration
-> observability, plus security with use_cmk). Deployment is parameterized via cdk.json
context — see iac/config.py and iac/README.md. Reuses an existing VPC / S3 cache
bucket when provided, and supports a fully-private (no-NAT) data plane.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import aws_cdk as cdk

# `cdk synth` runs this file from iac/ (cdk.json "app"), where the `iac` package
# is not importable. Put the repo root on sys.path so a fresh checkout works
# without PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iac.config import DeploymentConfig  # noqa: E402
from iac.stacks.compute_stack import ComputeStack  # noqa: E402
from iac.stacks.guardrail_stack import GuardrailStack  # noqa: E402
from iac.stacks.networking_stack import NetworkingStack  # noqa: E402
from iac.stacks.observability_stack import ObservabilityStack  # noqa: E402
from iac.stacks.orchestration_stack import OrchestrationStack  # noqa: E402
from iac.stacks.security_stack import SecurityStack  # noqa: E402
from iac.stacks.storage_stack import StorageStack  # noqa: E402

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

# Guard: in private (no-NAT) mode the VPC interface endpoints are created in
# the DEPLOY region only, but the runtime clients call config.bedrock_region
# and the tasks read the observability secret in its own region. If those
# differ, the data plane has no network route: every Bedrock call hangs, or no
# task starts. Fail fast at synth with a clear message instead.
config.check_deploy_region(deploy_region)
# Physical names that embed env_name and the region have length limits that
# CloudFormation would only report mid-deploy.
config.check_physical_names(deploy_region, config.bedrock_region)

stack_prefix = config.stack_prefix


def stack_id(name: str) -> str:
    # PascalCase stack ids: GraphRag<Name> in dev, GraphRag<Env><Name> otherwise
    # (see DeploymentConfig.stack_prefix).
    return f"{stack_prefix}{name}"


# Only with use_cmk: without a key the stack would be empty (CloudFormation
# refuses a template with no resources, so cdk deploy skipped it anyway), and
# no other stack refers to it.
kms_key = (
    SecurityStack(app, stack_id("Security"), config=config, env=env).kms_key
    if config.use_cmk
    else None
)
# The guardrail is region-pinned to bedrock_region in its own stack. In the
# deploy region (bedrock_region unset or equal, which private mode requires)
# compute attaches the guardrail it owns through SSM parameters read at deploy
# time. Across regions we avoid a cross-region CloudFormation reference (those
# rewrite SSM/export plumbing across every stack and cause export-in-use churn
# on the deploy-region stacks): the id flows to compute via the
# `guardrail_identifier` context value, which always wins when given. Deploy the
# guardrail stack first, then pass its id with `-c guardrail_identifier=<id>` on
# subsequent deploys. Passing the id does NOT stop the guardrail stack from
# owning its guardrail (`create_guardrail` controls that), so step 2 never
# deletes what step 1 created. See iac/stacks/guardrail_stack.py.
guardrail = GuardrailStack(
    app,
    stack_id("Guardrail"),
    config=config,
    deploy_region=deploy_region,
    env=bedrock_env,
)
networking = NetworkingStack(
    app, stack_id("Network"), config=config, kms_key=kms_key, env=env
)
storage = StorageStack(
    app,
    stack_id("Storage"),
    config=config,
    networking=networking,
    kms_key=kms_key,
    env=env,
)
compute = ComputeStack(
    app,
    stack_id("Compute"),
    config=config,
    networking=networking,
    storage=storage,
    kms_key=kms_key,
    guardrail=guardrail.attachment,
    env=env,
)
orchestration = OrchestrationStack(
    app,
    stack_id("Orchestration"),
    config=config,
    networking=networking,
    compute=compute,
    cache_bucket_name=storage.cache_bucket.bucket_name,
    kms_key=kms_key,
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
    if tag_value:  # no owner tag unless one is set
        cdk.Tags.of(app).add(tag_key, tag_value)

# Well-Architected checks at synth time (opt-in: -c enable_cdk_nag=true).
# NOTE: cdk-nag reports unsuppressed violations as error-level annotations that
# cause a NON-ZERO exit only when synthesized via the `cdk` CLI (`cdk synth`);
# a plain `python app.py` does NOT fail on them. CI therefore runs `cdk synth`
# (see .github/workflows/quality.yml) so the nag check is an enforcing gate.
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
