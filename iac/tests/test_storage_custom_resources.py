# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Nori package custom resource (offline synth).

``opensearch_nori_package`` associates the AWS-provided analysis-nori package
of the domain's engine version through a provider-framework custom resource
whose roles reach only this deployment's domain and stack, and Korean then
maps to nori. Off (the default), the templates are exactly as without the
feature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.storage_stack import OPENSEARCH_VERSION, StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")
_HANDLERS = Path(__file__).resolve().parent.parent / "custom_resources"
_NORI = "Custom::OpenSearchNoriPackage"


def _stacks(context: dict[str, Any]) -> tuple[Template, Template]:
    app = cdk.App(context=context)
    config = DeploymentConfig.from_context(app)
    networking = NetworkingStack(app, "Net", config=config, env=_ENV)
    storage = StorageStack(app, "Store", config=config, networking=networking, env=_ENV)
    compute = ComputeStack(
        app, "Compute", config=config, networking=networking, storage=storage, env=_ENV
    )
    return Template.from_stack(storage), Template.from_stack(compute)


def _korean_analyzer(compute: Template) -> str:
    task_defs = compute.find_resources("AWS::ECS::TaskDefinition")
    (task_def,) = [v for k, v in task_defs.items() if k.startswith("IngestionTask")]
    (container,) = task_def["Properties"]["ContainerDefinitions"]
    env = {e["Name"]: e["Value"] for e in container["Environment"]}
    return str(env["UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO"])


def _policy_statements(storage: Template, role_hint: str) -> list[dict[str, Any]]:
    statements: list[dict[str, Any]] = []
    for logical_id, policy in storage.find_resources("AWS::IAM::Policy").items():
        if role_hint in logical_id:
            statements += policy["Properties"]["PolicyDocument"]["Statement"]
    return statements


def _only(storage: Template, resource_type: str) -> tuple[str, dict[str, Any]]:
    ((logical_id, resource),) = storage.find_resources(resource_type).items()
    return logical_id, resource


def _logical_id(storage: Template, resource_type: str) -> str:
    return _only(storage, resource_type)[0]


@pytest.fixture(scope="module")
def default() -> tuple[Template, Template]:
    return _stacks({})


@pytest.fixture(scope="module")
def enabled() -> tuple[Template, Template]:
    return _stacks({"opensearch_nori_package": "true"})


# -------------------------------------------------------------------- off
def test_off_by_default_adds_nothing(default: tuple[Template, Template]) -> None:
    storage, compute = default
    assert not storage.find_resources(_NORI)
    assert not storage.find_resources("AWS::StepFunctions::StateMachine")
    assert _korean_analyzer(compute) == "cjk"


def test_explicit_defaults_render_the_same_templates(
    default: tuple[Template, Template],
) -> None:
    explicit = _stacks({"opensearch_nori_package": "false"})
    for before, after in zip(default, explicit, strict=True):
        assert json.dumps(after.to_json(), sort_keys=True) == json.dumps(
            before.to_json(), sort_keys=True
        )


# ------------------------------------------------------------------- nori
def test_nori_package_names_the_domain_and_its_engine_version(
    enabled: tuple[Template, Template],
) -> None:
    storage, _ = enabled
    domain_id = _logical_id(storage, "AWS::OpenSearchService::Domain")
    _, resource = _only(storage, _NORI)
    props = resource["Properties"]
    assert props["DomainName"] == {"Ref": domain_id}
    assert props["EngineVersion"] == OPENSEARCH_VERSION.version == "OpenSearch_2.13"
    assert props["PackageName"] == "analysis-nori"
    # After the domain and everything it configures: an engine upgrade
    # finishes before the new version's package is looked up.
    assert domain_id in resource["DependsOn"]
    domain = storage.find_resources("AWS::OpenSearchService::Domain")[domain_id]
    assert domain["UpdatePolicy"] == {"EnableVersionUpgrade": True}


def test_nori_maps_korean_to_nori_unless_set(
    enabled: tuple[Template, Template],
) -> None:
    _, compute = enabled
    assert _korean_analyzer(compute) == "nori"
    _, staged = _stacks(
        {"opensearch_nori_package": "true", "opensearch_korean_analyzer": "cjk"}
    )
    assert _korean_analyzer(staged) == "cjk"


def test_nori_roles_reach_only_the_domain_and_the_stack(
    enabled: tuple[Template, Template],
) -> None:
    storage, _ = enabled
    domain_id = _logical_id(storage, "AWS::OpenSearchService::Domain")
    assert _policy_statements(storage, "NoriPackageHandlersOnEvent") == [
        {"Action": "es:DescribePackages", "Effect": "Allow", "Resource": "*"}
    ]
    assert _policy_statements(storage, "NoriPackageHandlersIsComplete") == [
        {
            "Action": [
                "es:AssociatePackage",
                "es:DescribeDomain",
                "es:DissociatePackage",
                "es:ListPackagesForDomain",
            ],
            "Effect": "Allow",
            "Resource": {"Fn::GetAtt": [domain_id, "Arn"]},
        },
        {
            "Action": "cloudformation:DescribeStacks",
            "Effect": "Allow",
            "Resource": {"Ref": "AWS::StackId"},
        },
    ]


# ---------------------------------------------------------------- handlers
def test_every_handler_names_a_function_of_the_asset(
    enabled: tuple[Template, Template],
) -> None:
    storage, _ = enabled
    handlers = [
        function["Properties"]["Handler"]
        for function in storage.find_resources("AWS::Lambda::Function").values()
        if function["Properties"]["Runtime"].startswith("python")
    ]
    assert sorted(handlers) == [
        "nori_package.is_complete",
        "nori_package.on_event",
    ]
    for handler in handlers:
        module, function = handler.split(".")
        source = (_HANDLERS / f"{module}.py").read_text()
        assert f"\ndef {function}(" in source


def test_the_waiters_log_every_event(enabled: tuple[Template, Template]) -> None:
    storage, _ = enabled
    machines = storage.find_resources("AWS::StepFunctions::StateMachine")
    assert len(machines) == 1
    for machine in machines.values():
        assert machine["Properties"]["LoggingConfiguration"]["Level"] == "ALL"
