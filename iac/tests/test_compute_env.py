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


def _compute_template(
    context: dict[str, Any] | None = None, guardrail_identifier: str | None = None
) -> Template:
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
        guardrail_identifier=guardrail_identifier,
        env=_ENV,
    )
    return Template.from_stack(compute)


def _task_environment(context: dict[str, Any] | None = None) -> dict[str, Any]:
    task_defs = _compute_template(context).find_resources("AWS::ECS::TaskDefinition")
    (task_def,) = [v for k, v in task_defs.items() if k.startswith("IngestionTask")]
    (container,) = task_def["Properties"]["ContainerDefinitions"]
    return {e["Name"]: e["Value"] for e in container["Environment"]}


def _role_statements(template: Template, role_id: str) -> list[dict[str, Any]]:
    """The statements of the inline policies attached to ``role_id``'s role."""
    return [
        s
        for policy in template.find_resources("AWS::IAM::Policy").values()
        if any(
            r.get("Ref", "").startswith(role_id) for r in policy["Properties"]["Roles"]
        )
        for s in policy["Properties"]["PolicyDocument"]["Statement"]
    ]


def _actions(statements: list[dict[str, Any]]) -> set[str]:
    return {
        a
        for s in statements
        for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])
    }


def test_doc_status_env_targets_iac_table_and_disables_auto_create() -> None:
    env = _task_environment()
    # Both names must match unified_kg_rag.shared.config env overrides.
    assert env["GRAPHRAG_DOC_STATUS_TABLE"] == "graphrag-doc-status-v2"
    assert env["GRAPHRAG_DOC_STATUS_CREATE_TABLE"] == "false"


def test_doc_status_env_follows_env_scoped_table_name() -> None:
    env = _task_environment({"env_name": "prod"})
    assert env["GRAPHRAG_DOC_STATUS_TABLE"] == "prod-graphrag-doc-status-v2"


def test_corpus_bucket_is_granted_read_only() -> None:
    app = cdk.App(context={"corpus_bucket_name": "my-corpus"})
    config = DeploymentConfig.from_context(app)
    networking = NetworkingStack(app, "Net", config=config, env=_ENV)
    storage = StorageStack(app, "Store", config=config, networking=networking, env=_ENV)
    compute = ComputeStack(
        app, "Compute", config=config, networking=networking, storage=storage, env=_ENV
    )
    policies = Template.from_stack(compute).find_resources("AWS::IAM::Policy")
    statements = [
        s
        for p in policies.values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
        if "my-corpus" in str(s["Resource"])
    ]
    # One read grant per task role: ingestion reads the corpus, run-eval may
    # read a dataset kept there.
    assert len(statements) == 2
    for statement in statements:
        assert "s3:GetObject*" in statement["Action"]
        assert not [
            a for a in statement["Action"] if a.startswith(("s3:Put", "s3:Del"))
        ]


@pytest.mark.parametrize(
    ("context", "replicas"),
    [({}, "0"), ({"env_name": "prod"}, "1"), ({"opensearch_count": 4}, "1")],
    ids=["dev-single-node", "prod-multi-node", "four-nodes"],
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


def test_task_role_has_no_unused_delete_or_admin_actions() -> None:
    actions = _actions(_role_statements(_compute_template(), "TaskRole"))
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
        for s in _role_statements(template, "TaskRole")
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
        for s in _role_statements(_compute_template(), "TaskRole")
        if "bedrock:InvokeModel" in s["Action"]
    ]
    assert "bedrock:CountTokens" in statement["Action"]


def _apply_guardrail_resources(template: Template) -> list[Any]:
    return [
        resource
        for s in _role_statements(template, "TaskRole")
        if "bedrock:ApplyGuardrail" in s["Action"]
        for resource in (
            s["Resource"] if isinstance(s["Resource"], list) else [s["Resource"]]
        )
    ]


def test_apply_guardrail_is_granted_on_the_one_guardrail_only() -> None:
    # No guardrail given, no grant (previously guardrail/* in every region).
    assert _apply_guardrail_resources(_compute_template()) == []
    resources = _apply_guardrail_resources(
        _compute_template({"bedrock_region": "us-east-1"}, guardrail_identifier="gr-1")
    )
    assert len(resources) == 1
    rendered = json.dumps(resources[0])
    assert ":bedrock:us-east-1:111111111111:guardrail/gr-1" in rendered
    assert "*" not in rendered


def test_apply_guardrail_accepts_a_guardrail_arn() -> None:
    arn = "arn:aws:bedrock:us-west-2:111111111111:guardrail/gr-2"
    template = _compute_template(guardrail_identifier=arn)
    assert _apply_guardrail_resources(template) == [arn]


def test_cluster_keeps_container_insights_on() -> None:
    # The only per-task CPU/memory source (README "Cost drivers"); cdk-nag
    # AwsSolutions-ECS4 requires it.
    (cluster,) = _compute_template().find_resources("AWS::ECS::Cluster").values()
    assert cluster["Properties"]["ClusterSettings"] == [
        {"Name": "containerInsights", "Value": "enabled"}
    ]


def test_app_container_gets_the_longest_stop_timeout() -> None:
    # run-ingestion checkpoints and unwinds on SIGTERM; the default 30 s
    # (StopTask, a Step Functions timeout) can cut that short.
    task_defs = _compute_template().find_resources("AWS::ECS::TaskDefinition")
    task_def = next(v for k, v in task_defs.items() if k.startswith("IngestionTask"))
    (container,) = task_def["Properties"]["ContainerDefinitions"]
    assert container["StopTimeout"] == 120


def test_query_role_cannot_write_or_delete_any_store() -> None:
    statements = _role_statements(
        _compute_template(guardrail_identifier="gr-1"), "QueryTaskRole"
    )
    actions = _actions(statements)
    assert {"neptune-db:connect", "neptune-db:ReadDataViaQuery"} <= actions
    assert {
        "bedrock:InvokeModel",
        "bedrock:Rerank",
        "bedrock:ApplyGuardrail",
    } <= actions
    assert not [a for a in actions if a.startswith("dynamodb:")]
    assert not actions & {
        "neptune-db:WriteDataViaQuery",
        "neptune-db:DeleteDataViaQuery",
        "es:ESHttp*",
        "es:ESHttpPut",
        "es:ESHttpDelete",
        "es:ESHttpPatch",
    }
    # POST (also the verb of _bulk, _doc and _delete_by_query) only on _search.
    (post,) = [s for s in statements if "es:ESHttpPost" in _actions([s])]
    assert json.dumps(post["Resource"]).endswith('"/*/_search"]]}')
    # S3 only under eval/ (run-eval's dataset and results).
    s3_resources = json.dumps(
        [
            s["Resource"]
            for s in statements
            if any(a.startswith("s3:") for a in _actions([s]))
        ]
    )
    assert "/eval/*" in s3_resources
    assert '"/*"' not in s3_resources


def test_query_task_definition_runs_the_same_app_with_the_query_role() -> None:
    template = _compute_template()
    task_defs = template.find_resources("AWS::ECS::TaskDefinition")
    query = next(v for k, v in task_defs.items() if k.startswith("QueryTask"))
    ingestion = next(v for k, v in task_defs.items() if k.startswith("IngestionTask"))
    assert query["Properties"]["TaskRoleArn"]["Fn::GetAtt"][0].startswith(
        "QueryTaskRole"
    )
    (q_app,) = query["Properties"]["ContainerDefinitions"]
    (i_app,) = ingestion["Properties"]["ContainerDefinitions"]
    assert q_app["Name"] == i_app["Name"] == "app"
    assert q_app["Environment"] == i_app["Environment"]
    assert "EphemeralStorage" not in query["Properties"]
    outputs = template.find_outputs("*")
    assert {"QueryTaskDefinitionArn", "QueryTaskRoleArn"} <= set(outputs)


def test_tracing_adds_the_collector_to_both_task_definitions() -> None:
    template = _compute_template({"enable_cloudwatch_tracing": "true"})
    for task_def in template.find_resources("AWS::ECS::TaskDefinition").values():
        names = {c["Name"] for c in task_def["Properties"]["ContainerDefinitions"]}
        assert names == {"app", "otel-collector"}
    assert len(template.find_resources("AWS::ECR::PullThroughCacheRule")) == 1


@pytest.mark.parametrize(
    ("context", "gib"), [({}, 40), ({"ingestion_ephemeral_storage_gib": 150}, 150)]
)
def test_ingestion_disk_is_configurable(context: dict[str, Any], gib: int) -> None:
    task_defs = _compute_template(context).find_resources("AWS::ECS::TaskDefinition")
    sizes = {
        t["Properties"].get("Family", name): t["Properties"].get("EphemeralStorage")
        for name, t in task_defs.items()
    }
    assert {"SizeInGiB": gib} in sizes.values()


@pytest.mark.parametrize("gib", [20, 201])
def test_ingestion_disk_outside_fargate_range_is_rejected(gib: int) -> None:
    with pytest.raises(ValueError, match="ingestion_ephemeral_storage_gib"):
        DeploymentConfig.from_context(
            cdk.App(context={"ingestion_ephemeral_storage_gib": gib})
        )
