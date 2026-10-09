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
from iac.stacks.security_stack import SecurityStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")


def _stacks(use_cmk: bool) -> dict[str, Any]:
    app = cdk.App(context={"use_cmk": str(use_cmk).lower()})
    config = DeploymentConfig.from_context(app)
    security = SecurityStack(app, "Sec", config=config, env=_ENV)
    networking = NetworkingStack(app, "Net", config=config, env=_ENV)
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
    return {"security": security, "compute": compute}


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
