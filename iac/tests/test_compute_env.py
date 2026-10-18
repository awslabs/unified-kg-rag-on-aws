# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ComputeStack task-environment assertions (offline synth, no lookups)."""

from __future__ import annotations

import json
import re
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
    assert env["UKR__AWS__DYNAMODB__TABLE_NAME"] == "graphrag-doc-status-v2"
    assert env["UKR__AWS__DYNAMODB__CREATE_TABLE_IF_MISSING"] == "false"


def test_doc_status_env_follows_env_scoped_table_name() -> None:
    env = _task_environment({"env_name": "prod"})
    assert env["UKR__AWS__DYNAMODB__TABLE_NAME"] == "prod-graphrag-doc-status-v2"


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
    assert (
        _task_environment(context)[
            "UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS"
        ]
        == replicas
    )


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
    assert (
        _task_environment(context)["UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO"]
        == analyzer
    )


def test_an_unpinned_guardrail_warns_that_the_tasks_apply_draft() -> None:
    from aws_cdk.assertions import Annotations, Match

    def compute(context: dict[str, Any]) -> ComputeStack:
        app = cdk.App(context=context)
        config = DeploymentConfig.from_context(app)
        net = NetworkingStack(app, "Net", config=config, env=_ENV)
        store = StorageStack(app, "Store", config=config, networking=net, env=_ENV)
        return ComputeStack(
            app,
            "Compute",
            config=config,
            networking=net,
            storage=store,
            guardrail_identifier=config.guardrail_identifier,
            env=_ENV,
        )

    draft = Match.string_like_regexp("without guardrail_version")
    unpinned = compute({"guardrail_identifier": "gr-1"})
    Annotations.from_stack(unpinned).has_warning("*", draft)
    pinned = compute({"guardrail_identifier": "gr-1", "guardrail_version": "3"})
    Annotations.from_stack(pinned).has_no_warning("*", draft)


def test_guardrail_version_is_injected_only_when_set() -> None:
    assert "UKR__AWS__BEDROCK__GUARDRAIL__VERSION" not in _task_environment()
    env = _task_environment({"guardrail_identifier": "gr-1", "guardrail_version": "2"})
    assert env["UKR__AWS__BEDROCK__GUARDRAIL__VERSION"] == "2"


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
    actions = _actions([s for s in statements if s["Effect"] == "Allow"])
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


# Stand-ins for the cross-stack ARNs a policy names, so a statement resolves
# to the strings IAM would match.
_DOMAIN = "arn:aws:es:us-west-2:111111111111:domain/graphrag"
_BUCKET = "arn:aws:s3:::cache-bucket"


def _resolve(value: Any) -> str:
    if isinstance(value, str):
        return value
    if "Fn::Join" in value:
        separator, parts = value["Fn::Join"]
        return str(separator).join(_resolve(part) for part in parts)
    if "Ref" in value:
        return {"AWS::Partition": "aws"}.get(value["Ref"], value["Ref"])
    name = json.dumps(value)
    if "OpenSearch" in name:
        return _DOMAIN
    if "CacheBucket" in name:
        return _BUCKET
    return name


def _glob(pattern: str, value: str) -> bool:
    """IAM wildcard matching: ``*`` is any run of characters, ``/`` included,
    and ``?`` any one character."""
    regex = "".join(
        ".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern
    )
    return re.fullmatch(regex, value) is not None


def _listed(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _matches(
    statement: dict[str, Any], action: str, resource: str, context: dict[str, str]
) -> bool:
    if not any(_glob(a, action) for a in _listed(statement["Action"])):
        return False
    if not any(_glob(_resolve(r), resource) for r in _listed(statement["Resource"])):
        return False
    for operator, keys in statement.get("Condition", {}).items():
        assert operator == "StringLike", operator
        for key, patterns in keys.items():
            if key not in context or not any(
                _glob(p, context[key]) for p in _listed(patterns)
            ):
                return False
    return True


def _allowed(
    statements: list[dict[str, Any]],
    action: str,
    resource: str,
    context: dict[str, str] | None = None,
) -> bool:
    """IAM's evaluation of one request against a role's own statements: an
    explicit deny wins, else any allow, else the implicit deny."""
    hits = [s for s in statements if _matches(s, action, resource, context or {})]
    if any(s["Effect"] == "Deny" for s in hits):
        return False
    return any(s["Effect"] == "Allow" for s in hits)


@pytest.fixture(scope="module")
def query_statements() -> list[dict[str, Any]]:
    return _role_statements(
        _compute_template(guardrail_identifier="gr-1"), "QueryTaskRole"
    )


# Every request the query strategies send (recorded against OpenSearch 2.13:
# HEAD on an index, GET mappings and aliases, POST <index>/_search, also over
# several indices).
_QUERY_REQUESTS = [
    ("HEAD", "/graphrag-entities-default"),
    ("GET", "/graphrag-entities-default/_mapping"),
    ("GET", "/_alias/graphrag-entities-default"),
    ("POST", "/graphrag-entities-default/_search"),
    ("POST", "/graphrag-text-units-default,graphrag-entities-default/_search"),
]
# Writes, including those OpenSearch serves on a path ending in `_search`
# (a document id `_search`, also percent-encoded) and admin requests.
_WRITE_REQUESTS = [
    ("POST", "/idx/_doc/_search"),
    ("POST", "/idx/_doc/%5Fsearch"),
    ("PUT", "/idx/_doc/_search"),
    ("POST", "/idx/_create/_search"),
    ("POST", "/idx/_update/_search"),
    ("POST", "/idx/_doc"),
    ("PUT", "/idx/_doc/1"),
    ("DELETE", "/idx/_doc/1"),
    ("POST", "/_bulk"),
    ("POST", "/idx/_bulk"),
    ("POST", "/idx/_delete_by_query"),
    ("POST", "/idx/_update_by_query"),
    ("POST", "/_reindex"),
    ("POST", "/_aliases"),
    ("PUT", "/idx"),
    ("PUT", "/idx/_mapping"),
    ("DELETE", "/idx"),
]


@pytest.mark.parametrize(("method", "path"), _QUERY_REQUESTS)
def test_query_role_may_send_the_query_requests(
    query_statements: list[dict[str, Any]], method: str, path: str
) -> None:
    action = f"es:ESHttp{method.title()}"
    assert _allowed(query_statements, action, f"{_DOMAIN}{path}")


@pytest.mark.parametrize(("method", "path"), _WRITE_REQUESTS)
def test_query_role_may_not_write_to_opensearch(
    query_statements: list[dict[str, Any]], method: str, path: str
) -> None:
    action = f"es:ESHttp{method.title()}"
    assert not _allowed(query_statements, action, f"{_DOMAIN}{path}")


def test_query_role_reads_and_lists_only_the_eval_prefix(
    query_statements: list[dict[str, Any]],
) -> None:
    s = query_statements
    assert _allowed(s, "s3:GetObject", f"{_BUCKET}/eval/data.json")
    assert _allowed(s, "s3:PutObject", f"{_BUCKET}/eval/out/result.json")
    assert _allowed(s, "s3:ListBucket", _BUCKET, {"s3:prefix": "eval/data.json"})
    assert not _allowed(s, "s3:GetObject", f"{_BUCKET}/pipeline-runs/p/stage.json")
    assert not _allowed(s, "s3:PutObject", f"{_BUCKET}/pipeline-runs/p/stage.json")
    assert not _allowed(s, "s3:ListBucket", _BUCKET, {"s3:prefix": "pipeline-runs/"})
    assert not _allowed(s, "s3:ListBucket", _BUCKET)
    assert not _allowed(s, "s3:DeleteObject", f"{_BUCKET}/eval/data.json")


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


@pytest.mark.parametrize("env", ["dev", "prod"])
def test_task_tags_its_metrics_with_the_environment(env: str) -> None:
    # Must match the CLI's env var and the observability stack's dimension.
    assert _task_environment({"env_name": env})["UKR_METRICS_ENVIRONMENT"] == env
