# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""use_cmk encryption assertions beyond the stateful stores (offline synth)."""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.security_stack import SecurityStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")


def _stacks(use_cmk: bool) -> dict[str, Any]:
    app = cdk.App(context={"use_cmk": str(use_cmk).lower(), "vpc_flow_logs": "true"})
    config = DeploymentConfig.from_context(app)
    security = SecurityStack(app, "Sec", config=config, env=_ENV)
    networking = NetworkingStack(
        app, "Net", config=config, kms_key=security.kms_key, env=_ENV
    )
    storage = StorageStack(
        app,
        "Store",
        config=config,
        networking=networking,
        kms_key=security.kms_key,
        env=_ENV,
    )
    compute = ComputeStack(
        app,
        "Compute",
        config=config,
        networking=networking,
        storage=storage,
        kms_key=security.kms_key,
        env=_ENV,
    )
    orchestration = OrchestrationStack(
        app,
        "Orch",
        config=config,
        networking=networking,
        compute=compute,
        cache_bucket_name=storage.cache_bucket.bucket_name,
        kms_key=security.kms_key,
        env=_ENV,
    )
    return {
        "security": security,
        "networking": networking,
        "storage": storage,
        "compute": compute,
        "orchestration": orchestration,
    }


@pytest.mark.parametrize("use_cmk", [True, False], ids=["cmk", "default"])
def test_ecr_repository_uses_cmk_only_when_enabled(use_cmk: bool) -> None:
    stacks = _stacks(use_cmk)
    (repo,) = (
        Template.from_stack(stacks["compute"])
        .find_resources("AWS::ECR::Repository")
        .values()
    )
    encryption = repo["Properties"].get("EncryptionConfiguration")
    if not use_cmk:
        # Unset, not AES256: ECR replaces a repository whose encryption
        # configuration changes, and the fixed repository name blocks that.
        assert encryption is None
        return
    assert encryption["EncryptionType"] == "KMS"
    assert "DataKey" in str(encryption["KmsKey"])


@pytest.mark.parametrize("use_cmk", [True, False], ids=["cmk", "default"])
def test_every_log_group_uses_cmk_only_when_enabled(use_cmk: bool) -> None:
    stacks = _stacks(use_cmk)
    log_groups = {
        f"{name}/{logical_id}": resource
        for name, stack in stacks.items()
        for logical_id, resource in Template.from_stack(stack)
        .find_resources("AWS::Logs::LogGroup")
        .items()
    }
    # Flow logs, the task and pipeline groups, and three OpenSearch groups.
    assert len(log_groups) == 6, sorted(log_groups)
    for name, resource in log_groups.items():
        key = resource["Properties"].get("KmsKeyId")
        if use_cmk:
            assert "DataKey" in str(key), name
        else:
            assert key is None, name


def test_cmk_policy_lets_cloudwatch_logs_use_the_key() -> None:
    (key,) = (
        Template.from_stack(_stacks(True)["security"])
        .find_resources("AWS::KMS::Key")
        .values()
    )
    statements = {s.get("Sid"): s for s in key["Properties"]["KeyPolicy"]["Statement"]}
    service = statements["AllowCloudWatchLogs"]
    assert service["Principal"] == {"Service": "logs.us-west-2.amazonaws.com"}
    assert "kms:GenerateDataKey*" in service["Action"]
    # Only for log groups in this account and region.
    (arn,) = service["Condition"]["ArnLike"].values()
    assert ":logs:us-west-2:111111111111:log-group:*" in str(arn)
    callers = statements["AllowAccountUseViaCloudWatchLogs"]
    assert callers["Condition"]["StringEquals"] == {
        "kms:CallerAccount": "111111111111",
        "kms:ViaService": "logs.us-west-2.amazonaws.com",
    }
