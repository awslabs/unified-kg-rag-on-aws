# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ComputeStack task-environment assertions (offline synth, no lookups)."""

from __future__ import annotations

import json
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")


def _compute_template(context: dict[str, Any] | None = None) -> Template:
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
    return Template.from_stack(compute)


def _task_environment(context: dict[str, Any] | None = None) -> dict[str, Any]:
    task_defs = _compute_template(context).find_resources("AWS::ECS::TaskDefinition")
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


@pytest.mark.parametrize(
    ("context", "replicas"),
    [({}, "0"), ({"env_name": "prod"}, "1"), ({"opensearch_count": 3}, "1")],
    ids=["dev-single-node", "prod-multi-node", "three-nodes"],
)
def test_opensearch_replicas_follow_node_count(
    context: dict[str, Any], replicas: str
) -> None:
    # Name must match the unified_kg_rag.shared.config env override.
    assert _task_environment(context)["OPENSEARCH_NUMBER_OF_REPLICAS"] == replicas


@pytest.mark.parametrize(
    ("context", "analyzer"),
    [({}, "cjk"), ({"opensearch_korean_analyzer": "nori"}, "nori")],
    ids=["default-built-in", "nori-plugin-associated"],
)
def test_korean_analyzer_is_one_the_domain_has(
    context: dict[str, Any], analyzer: str
) -> None:
    # The domain has no nori unless its optional package is associated, so the
    # default maps Korean to the built-in cjk analyzer. Name must match the
    # unified_kg_rag.shared.config env override.
    assert _task_environment(context)["OPENSEARCH_KOREAN_ANALYZER"] == analyzer


def test_guardrail_version_is_injected_only_when_set() -> None:
    assert "BEDROCK_GUARDRAIL_VERSION" not in _task_environment()
    env = _task_environment({"guardrail_identifier": "gr-1", "guardrail_version": "2"})
    assert env["BEDROCK_GUARDRAIL_VERSION"] == "2"


def _task_role_actions(template: Template) -> set[str]:
    actions: set[str] = set()
    for policy in template.find_resources("AWS::IAM::Policy").values():
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            action = statement["Action"]
            actions.update([action] if isinstance(action, str) else action)
    return actions


def test_task_role_has_no_unused_delete_or_admin_actions() -> None:
    actions = _task_role_actions(_compute_template())
    assert "s3:PutObject" in actions
    assert "neptune-db:WriteDataViaQuery" in actions
    # The app never deletes cache objects or manages Neptune queries.
    assert not [a for a in actions if a.startswith("s3:Delete")]
    assert not actions & {
        "neptune-db:GetEngineStatus",
        "neptune-db:GetQueryStatus",
        "neptune-db:CancelQuery",
    }


def test_ecr_lifecycle_never_expires_the_deployed_tag() -> None:
    (repo,) = (
        _compute_template({"image_tag": "v1.2.3"})
        .find_resources("AWS::ECR::Repository")
        .values()
    )
    rules = json.loads(repo["Properties"]["LifecyclePolicy"]["LifecyclePolicyText"])
    first, *_ = sorted(rules["rules"], key=lambda r: r["rulePriority"])
    assert first["selection"]["tagPatternList"] == ["v1.2.3"]


def _invoke_resources(context: dict[str, Any]) -> list[str]:
    """The InvokeModel statement's resources, tokens rendered as JSON."""
    template = _compute_template(context)
    (statement,) = [
        s
        for p in template.find_resources("AWS::IAM::Policy").values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
        if "bedrock:InvokeModel" in s["Action"]
    ]
    return [r if isinstance(r, str) else json.dumps(r) for r in statement["Resource"]]


_MODEL = "arn:aws:bedrock:*::foundation-model/vendor.model-a"
_PROFILE = "arn:aws:bedrock:us-west-2:111111111111:inference-profile/us.vendor.model-a"


def test_foundation_model_arns_keep_the_inference_profile_grants() -> None:
    # Profile-only models resolve to a profile id at runtime; a profile call
    # still needs the listed foundation-model ARN, so no other model is granted.
    resources = _invoke_resources({"bedrock_model_arns": _MODEL})
    assert resources[0] == _MODEL
    assert not [r for r in resources if "foundation-model/*" in r]
    assert sum("inference-profile/*" in r for r in resources) == 2


def test_listed_inference_profile_arns_are_used_exactly() -> None:
    resources = _invoke_resources({"bedrock_model_arns": f"{_MODEL},{_PROFILE}"})
    assert resources == [_MODEL, _PROFILE]


def test_count_tokens_is_granted_on_the_model_resources() -> None:
    # The app counts tokens with CountTokens for models that accept it; the
    # action authorizes against foundation-model ARNs, so it sits in the
    # InvokeModel statement. Without it every count silently degrades to an
    # estimate (AccessDenied is caught by the counter).
    (statement,) = [
        s
        for p in _compute_template().find_resources("AWS::IAM::Policy").values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
        if "bedrock:InvokeModel" in s["Action"]
    ]
    assert "bedrock:CountTokens" in statement["Action"]
