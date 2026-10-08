# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Removal-policy assertions for fixed-name, non-storage resources.

The ECR repository and the two log groups have fixed physical names. With
removal_destroy (dev default) `cdk destroy --all` must delete them, or the next
deploy fails on the retained copies; outside dev they are retained.
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")


def _templates(context: dict[str, Any]) -> list[Template]:
    app = cdk.App(context=context)
    config = DeploymentConfig.from_context(app)
    networking = NetworkingStack(app, "Net", config=config, env=_ENV)
    storage = StorageStack(app, "Store", config=config, networking=networking, env=_ENV)
    compute = ComputeStack(
        app, "Compute", config=config, networking=networking, storage=storage, env=_ENV
    )
    orchestration = OrchestrationStack(
        app,
        "Orch",
        config=config,
        networking=networking,
        compute=compute,
        cache_bucket_name=storage.cache_bucket.bucket_name,
        env=_ENV,
    )
    return [Template.from_stack(compute), Template.from_stack(orchestration)]


def _fixed_name_resources(context: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for template in _templates(context):
        for kind in ("AWS::ECR::Repository", "AWS::Logs::LogGroup"):
            found += template.find_resources(kind).values()
    assert len(found) == 3, "expected the ECR repo and two log groups"
    return found


@pytest.mark.parametrize(
    ("context", "policy"),
    [({}, "Delete"), ({"env_name": "prod"}, "Retain")],
    ids=["dev", "prod"],
)
def test_fixed_name_resources_follow_removal_destroy(
    context: dict[str, Any], policy: str
) -> None:
    for resource in _fixed_name_resources(context):
        assert resource["DeletionPolicy"] == policy, resource
        if resource["Type"] == "AWS::ECR::Repository":
            # A repository with images cannot be deleted unless emptied.
            assert resource["Properties"].get("EmptyOnDelete", False) is (
                policy == "Delete"
            )
