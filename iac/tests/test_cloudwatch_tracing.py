# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""enable_cloudwatch_tracing: ADOT collector sidecar (offline synth, no lookups)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import COLLECTOR_DIGEST, ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")
_ON = {"enable_cloudwatch_tracing": "true"}


def _stacks(context: dict[str, Any]) -> tuple[Template, Template]:
    app = cdk.App(context=context)
    config = DeploymentConfig.from_context(app)
    networking = NetworkingStack(app, "Net", config=config, env=_ENV)
    storage = StorageStack(app, "Store", config=config, networking=networking, env=_ENV)
    compute = ComputeStack(
        app, "Compute", config=config, networking=networking, storage=storage, env=_ENV
    )
    return Template.from_stack(networking), Template.from_stack(compute)


def _containers(compute: Template) -> dict[str, dict[str, Any]]:
    # The ingestion task definition; the query one is built the same way.
    (task_def,) = [
        v
        for k, v in compute.find_resources("AWS::ECS::TaskDefinition").items()
        if k.startswith("IngestionTask")
    ]
    return {c["Name"]: c for c in task_def["Properties"]["ContainerDefinitions"]}


def _env(container: dict[str, Any]) -> dict[str, Any]:
    return {e["Name"]: e["Value"] for e in container["Environment"]}


def _render(value: Any) -> str:
    """A string or ``Fn::Join``, each ``Ref`` as ``<Name>``, other tokens ``<?>``."""
    if isinstance(value, str):
        return value
    _, parts = value["Fn::Join"]
    return "".join(p if isinstance(p, str) else f"<{p.get('Ref', '?')}>" for p in parts)


def _statements(compute: Template, role_prefix: str) -> list[dict[str, Any]]:
    return [
        s
        for logical_id, p in compute.find_resources("AWS::IAM::Policy").items()
        if logical_id.startswith(role_prefix)
        for s in p["Properties"]["PolicyDocument"]["Statement"]
    ]


def _actions(statements: list[dict[str, Any]]) -> set[str]:
    actions: set[str] = set()
    for statement in statements:
        action = statement["Action"]
        actions.update([action] if isinstance(action, str) else action)
    return actions


def _xray_endpoints(network: Template) -> list[str]:
    return [
        logical_id
        for logical_id in network.find_resources("AWS::EC2::VPCEndpoint")
        if "XRay" in logical_id
    ]


@pytest.mark.parametrize("context", [{}, {"network_mode": "public"}])
def test_disabled_adds_nothing(context: dict[str, Any]) -> None:
    network, compute = _stacks(context)
    assert set(_containers(compute)) == {"app"}
    assert not [k for k in _env(_containers(compute)["app"]) if "OBSERVABILITY" in k]
    assert not [a for a in _actions(_statements(compute, "")) if a.startswith("xray:")]
    compute.resource_count_is("AWS::ECR::PullThroughCacheRule", 0)
    assert compute.find_outputs("CollectorImageUri") == {}
    assert _xray_endpoints(network) == []


def test_enabled_runs_a_non_essential_collector_the_app_waits_for() -> None:
    _, compute = _stacks(_ON)
    containers = _containers(compute)
    collector, app = containers["otel-collector"], containers["app"]
    # Losing the collector must never stop an ingestion run.
    assert collector["Essential"] is False
    assert app["DependsOn"] == [
        {"ContainerName": "otel-collector", "Condition": "START"}
    ]
    assert collector["HealthCheck"]["Command"] == ["CMD", "/healthcheck"]
    image = json.dumps(collector["Image"])
    assert (
        f"graphrag-ecr-public/aws-observability/aws-otel-collector@{COLLECTOR_DIGEST}"
        in (image)
    )
    compute.has_resource_properties(
        "AWS::ECR::PullThroughCacheRule",
        {
            "EcrRepositoryPrefix": "graphrag-ecr-public",
            "UpstreamRegistry": "ecr-public",
            "UpstreamRegistryUrl": "public.ecr.aws",
        },
    )
    # The warm-up pull the README runs after deploy uses this exact image.
    (output,) = compute.find_outputs("CollectorImageUri").values()
    assert _render(output["Value"]).endswith(
        f"graphrag-ecr-public/aws-observability/aws-otel-collector@{COLLECTOR_DIGEST}"
    )
    config = json.loads(_render(_env(collector)["AOT_CONFIG_CONTENT"]))
    # Listens on localhost only; exports to the region's OTLP traces endpoint.
    assert config["receivers"]["otlp"]["protocols"] == {
        "http": {"endpoint": "127.0.0.1:4318"}
    }
    # ADOT v0.50.0 deprecates the "otlphttp" alias in favour of "otlp_http".
    assert list(config["exporters"]) == ["otlp_http/traces"]
    assert config["service"]["pipelines"]["traces"]["exporters"] == ["otlp_http/traces"]
    exporter = config["exporters"]["otlp_http/traces"]
    assert exporter["traces_endpoint"] == (
        "https://xray.us-west-2.<AWS::URLSuffix>/v1/traces"
    )
    assert exporter["auth"] == {"authenticator": "sigv4auth/traces"}
    # A shutdown flush must end inside ECS's 30 s stop timeout.
    # No sending queue (drained batch by batch at shutdown) and no retry, so
    # what is left at shutdown is at most two 10 s exports.
    assert exporter["timeout"] == "10s"
    assert exporter["sending_queue"] == {"enabled": False}
    assert exporter["retry_on_failure"] == {"enabled": False}
    assert config["extensions"]["sigv4auth/traces"] == {
        "region": "us-west-2",
        "service": "xray",
    }


def test_enabled_points_the_app_at_the_sidecar_without_content_capture() -> None:
    _, compute = _stacks(_ON)
    env = _env(_containers(compute)["app"])
    # Names must match the unified_kg_rag.shared.config env overrides.
    assert env["UKR__OBSERVABILITY__ENABLED"] == "true"
    assert env["UKR__OBSERVABILITY__BACKEND"] == "otlp"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:4318"
    assert not [k for k in env if "CAPTURE" in k]


def test_enabled_grants_only_xray_writes_and_the_collector_image_pull() -> None:
    _, compute = _stacks(_ON)
    task_actions = _actions(_statements(compute, "TaskRole"))
    # The X-Ray OTLP endpoint authorizes PutTraceSegments only.
    assert {a for a in task_actions if a.startswith("xray:")} == {
        "xray:PutTraceSegments"
    }
    (import_statement,) = [
        s
        for s in _statements(compute, "IngestionTaskExecutionRole")
        if "ecr:BatchImportUpstreamImage" in s["Action"]
    ]
    assert sorted(import_statement["Action"]) == [
        "ecr:BatchImportUpstreamImage",
        "ecr:CreateRepository",
    ]
    # The one cached repository, not the registry.
    assert json.dumps(import_statement["Resource"]).endswith(
        ':repository/graphrag-ecr-public/aws-observability/aws-otel-collector"]]}'
    )


@pytest.mark.parametrize(
    ("context", "expected"),
    [(_ON, 1), ({**_ON, "network_mode": "public"}, 0)],
    ids=["private", "public"],
)
def test_xray_endpoint_only_in_private_mode(
    context: dict[str, Any], expected: int
) -> None:
    network, _ = _stacks(context)
    assert len(_xray_endpoints(network)) == expected


@pytest.mark.parametrize(("context", "listed"), [(_ON, True), ({}, False)])
def test_reused_private_vpc_warning_lists_xray_only_when_enabled(
    context: dict[str, Any], listed: bool
) -> None:
    app = cdk.App(context={**context, "vpc_id": "vpc-0example"})
    config = DeploymentConfig.from_context(app)
    stack = NetworkingStack(app, "Net", config=config, env=_ENV)
    (warning,) = [
        str(entry.data)
        for construct in stack.node.find_all()
        for entry in construct.node.metadata
        if entry.type == "aws:cdk:warning" and "Reusing VPC" in str(entry.data)
    ]
    assert ("xray" in warning.split(":")[-1]) is listed


def test_long_env_name_is_rejected_when_enabled() -> None:
    # "<env>-graphrag-ecr-public" may be at most 30 characters: env <= 10.
    DeploymentConfig.from_context(cdk.App(context={**_ON, "env_name": "x" * 10}))
    with pytest.raises(ValueError, match="enable_cloudwatch_tracing"):
        DeploymentConfig.from_context(cdk.App(context={**_ON, "env_name": "x" * 11}))
    # Unaffected when the feature is off.
    DeploymentConfig.from_context(cdk.App(context={"env_name": "x" * 11}))


def test_task_overrides_target_the_app_container() -> None:
    from iac.stacks.orchestration_stack import OrchestrationStack

    app = cdk.App(context=_ON)
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
        cache_bucket_name="example-bucket",
        env=_ENV,
    )
    (machine,) = (
        Template.from_stack(orchestration)
        .find_resources("AWS::StepFunctions::StateMachine")
        .values()
    )
    definition = _render(machine["Properties"]["DefinitionString"])
    assert '"Name":"app"' in definition
    assert '"Name":"otel-collector"' not in definition


def _yaml(node: Any, indent: str = "") -> str:
    """``node`` as block YAML in the runbook's style (no YAML library in the
    IaC environment): flow lists, lower-case booleans, plain scalars."""
    lines = []
    for key, value in node.items():
        if isinstance(value, dict):
            lines.append(f"{indent}{key}:\n{_yaml(value, indent + '  ')}")
        elif isinstance(value, list):
            lines.append(f"{indent}{key}: [{', '.join(value)}]\n")
        elif isinstance(value, bool):
            lines.append(f"{indent}{key}: {str(value).lower()}\n")
        else:
            lines.append(f"{indent}{key}: {value}\n")
    return "".join(lines)


def test_readme_shows_the_collector_config_the_stack_renders() -> None:
    _, compute = _stacks(_ON)
    rendered = _render(
        _env(_containers(compute)["otel-collector"])["AOT_CONFIG_CONTENT"]
    )
    config = json.loads(
        rendered.replace("us-west-2", "us-east-1").replace(
            "<AWS::URLSuffix>", "amazonaws.com"
        )
    )
    text = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    start = text.index("```yaml\nextensions:\n  health_check:")
    block = text[start + len("```yaml\n") : text.index("```\n", start + 7)]
    assert block == _yaml(config)
