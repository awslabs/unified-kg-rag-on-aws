# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ComputeStack task-environment assertions (offline synth, no lookups)."""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")


def _task_environment(context: dict[str, Any] | None = None) -> dict[str, Any]:
    app = cdk.App(context=context or {})
    config = DeploymentConfig.from_context(app)
    networking = NetworkingStack(app, "Net", config=config, env=_ENV)
    storage = StorageStack(app, "Store", config=config, networking=networking, env=_ENV)
    compute = ComputeStack(
        app,
        "Compute",
        config=config,
        networking=networking,
        storage=storage,
        env=_ENV,
    )
    task_defs = Template.from_stack(compute).find_resources("AWS::ECS::TaskDefinition")
    (task_def,) = task_defs.values()
    (container,) = task_def["Properties"]["ContainerDefinitions"]
    return {e["Name"]: e["Value"] for e in container["Environment"]}


def test_doc_status_env_targets_iac_table_and_disables_auto_create() -> None:
    env = _task_environment()
    # Both names must match unified_kg_rag.shared.config env overrides.
    assert env["GRAPHRAG_DOC_STATUS_TABLE"] == "graphrag-doc-status"
    assert env["GRAPHRAG_DOC_STATUS_CREATE_TABLE"] == "false"


def test_doc_status_env_follows_env_scoped_table_name() -> None:
    env = _task_environment({"env_name": "prod"})
    assert env["GRAPHRAG_DOC_STATUS_TABLE"] == "prod-graphrag-doc-status"


def test_corpus_bucket_is_granted_read_only() -> None:
    app = cdk.App(context={"corpus_bucket_name": "my-corpus"})
    config = DeploymentConfig.from_context(app)
    networking = NetworkingStack(app, "Net", config=config, env=_ENV)
    storage = StorageStack(app, "Store", config=config, networking=networking, env=_ENV)
    compute = ComputeStack(
        app, "Compute", config=config, networking=networking, storage=storage, env=_ENV
    )
    policies = Template.from_stack(compute).find_resources("AWS::IAM::Policy")
    (statement,) = [
        s
        for p in policies.values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
        if "my-corpus" in str(s["Resource"])
    ]
    assert "s3:GetObject*" in statement["Action"]
    assert not [a for a in statement["Action"] if a.startswith(("s3:Put", "s3:Del"))]
