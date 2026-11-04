# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The state machine runs the ingestion task-definition revision of its own
deployment, and a deploy that replaces the task definition updates it
(offline synth, no lookups).

CloudFormation can neither change nor remove an export another stack
imports. A task-definition ARN exported by one stack and imported by the
state machine's would therefore fail every deploy that replaces the task
definition (a new image tag, guardrail version, size or Korean analyzer),
and a deploy of the exporting stack alone would leave the importing
template unchanged, so the state machine would keep the old revision.
"""

from __future__ import annotations

import json
from typing import Any

import aws_cdk as cdk
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.observability_stack import ObservabilityStack
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")


def _templates(context: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    app = cdk.App(context=context or {})
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
    observability = ObservabilityStack(
        app,
        "Obs",
        config=config,
        orchestration=orchestration,
        storage=storage,
        env=_ENV,
    )
    stacks = (networking, storage, compute, orchestration, observability)
    return {s.stack_name: Template.from_stack(s).to_json() for s in stacks}


def _imports(node: Any) -> list[str]:
    if isinstance(node, dict):
        if set(node) == {"Fn::ImportValue"}:
            return [node["Fn::ImportValue"]]
        return [name for value in node.values() for name in _imports(value)]
    if isinstance(node, list):
        return [name for value in node for name in _imports(value)]
    return []


def _exported_resource_types(
    templates: dict[str, dict[str, Any]],
) -> dict[str, str | None]:
    """Each export name -> the type of the resource whose Ref or GetAtt it
    exports (None for any other value)."""
    types: dict[str, str | None] = {}
    for template in templates.values():
        resources = template.get("Resources", {})
        for output in template.get("Outputs", {}).values():
            if "Export" not in output:
                continue
            value = output["Value"]
            logical_id = value.get("Ref") or (value.get("Fn::GetAtt") or [None])[0]
            resource = resources.get(logical_id) if logical_id else None
            types[output["Export"]["Name"]] = resource and resource["Type"]
    return types


def _machine_and_task(template: dict[str, Any]) -> tuple[str, str]:
    """The orchestration template's state-machine definition (rendered with
    each Ref as ``<LogicalId>``) and the logical id of its task definition."""
    resources = template["Resources"]
    (machine,) = [
        r for r in resources.values() if r["Type"] == "AWS::StepFunctions::StateMachine"
    ]
    (task_id,) = [
        k
        for k, r in resources.items()
        if r["Type"] == "AWS::ECS::TaskDefinition" and k.startswith("IngestionTask")
    ]
    _, parts = machine["Properties"]["DefinitionString"]["Fn::Join"]
    rendered = "".join(
        part if isinstance(part, str) else f"<{json.dumps(part, sort_keys=True)}>"
        for part in parts
    )
    return rendered, task_id


def test_no_stack_imports_a_task_definition() -> None:
    templates = _templates()
    exported = _exported_resource_types(templates)
    imported = {
        (stack, name): exported.get(name)
        for stack, template in templates.items()
        for name in _imports(template)
    }
    assert imported  # the stacks do share values
    assert not [
        key for key, kind in imported.items() if kind == "AWS::ECS::TaskDefinition"
    ]


def test_the_state_machine_pins_the_revision_of_its_own_task_definition() -> None:
    orchestration = _templates()["Orch"]
    definition, task_id = _machine_and_task(orchestration)
    revision = {"Fn::Select": [6, {"Fn::Split": [":", {"Ref": task_id}]}]}
    assert f"<{json.dumps(revision, sort_keys=True)}>" in definition
    assert (
        "ImportValue"
        not in definition.split('"TaskDefinition":', 1)[1].split(
            '"NetworkConfiguration"', 1
        )[0]
    )


def test_a_new_image_tag_changes_the_template_that_pins_the_revision() -> None:
    before = _templates({"image_tag": "v1.0.0"})
    after = _templates({"image_tag": "v1.0.1"})
    # The template that owns the task definition and the state machine
    # changes, so `cdk deploy --all` updates it; the replaced task
    # definition's Ref is the new revision's ARN, which the definition reads.
    definition_before, task_id = _machine_and_task(before["Orch"])
    definition_after, task_id_after = _machine_and_task(after["Orch"])
    assert task_id_after == task_id
    assert definition_after == definition_before  # the same Ref, re-resolved

    def image(template: dict[str, Any]) -> str:
        (app,) = [
            c
            for c in template["Resources"][task_id]["Properties"][
                "ContainerDefinitions"
            ]
            if c["Name"] == "app"
        ]
        return json.dumps(app["Image"])

    assert "v1.0.0" in image(before["Orch"])
    assert "v1.0.1" in image(after["Orch"])
    # No other stack holds an ingestion task definition.
    assert not [
        logical_id
        for name, template in after.items()
        if name != "Orch"
        for logical_id, r in template["Resources"].items()
        if r["Type"] == "AWS::ECS::TaskDefinition"
        and logical_id.startswith("IngestionTask")
    ]
