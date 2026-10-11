# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Alarm-topic encryption assertions (offline synth, no lookups).

CloudWatch alarms and the Step Functions failure notifications both publish to
the KMS-encrypted alarm topic. Delivery only works when the topic key lets
cloudwatch.amazonaws.com use it and the state machine role holds the key.
"""

from __future__ import annotations

import json
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.security_stack import SecurityStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")


def _orchestration_template(context: dict[str, Any]) -> Template:
    app = cdk.App(context=context)
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
    orchestration = OrchestrationStack(
        app,
        "Orch",
        config=config,
        networking=networking,
        compute=compute,
        cache_bucket_name=storage.cache_bucket.bucket_name,
        env=_ENV,
    )
    return Template.from_stack(orchestration)


@pytest.fixture(
    scope="module", params=[{}, {"use_cmk": "true"}], ids=["default", "cmk"]
)
def template(request: pytest.FixtureRequest) -> Template:
    return _orchestration_template(request.param)


def _topic_key_id(template: Template) -> str:
    (topic,) = template.find_resources("AWS::SNS::Topic").values()
    key_ref = topic["Properties"]["KmsMasterKeyId"]
    # A customer-managed key in this stack, never the AWS-managed alias/aws/sns.
    assert isinstance(key_ref, dict) and "Fn::GetAtt" in key_ref, key_ref
    return str(key_ref["Fn::GetAtt"][0])


def test_topic_uses_a_customer_managed_key(template: Template) -> None:
    key = template.find_resources("AWS::KMS::Key")[_topic_key_id(template)]
    assert key["Properties"]["EnableKeyRotation"] is True


def test_topic_key_policy_allows_cloudwatch(template: Template) -> None:
    key = template.find_resources("AWS::KMS::Key")[_topic_key_id(template)]
    statements = key["Properties"]["KeyPolicy"]["Statement"]
    cloudwatch = [
        s
        for s in statements
        if s.get("Principal", {}).get("Service") == "cloudwatch.amazonaws.com"
    ]
    assert len(cloudwatch) == 1
    assert set(cloudwatch[0]["Action"]) == {"kms:Decrypt", "kms:GenerateDataKey*"}


def test_state_machine_role_can_use_topic_key(template: Template) -> None:
    key_id = _topic_key_id(template)
    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Action": ["kms:Decrypt", "kms:GenerateDataKey*"],
                                "Resource": {"Fn::GetAtt": [key_id, "Arn"]},
                            }
                        )
                    ]
                )
            }
        },
    )


def _definition(template: Template) -> dict[str, Any]:
    (machine,) = template.find_resources("AWS::StepFunctions::StateMachine").values()
    parts = machine["Properties"]["DefinitionString"]["Fn::Join"][1]
    # Tokens (ARNs) are joined into string values; the rest is plain JSON.
    text = "".join(p if isinstance(p, str) else "TOKEN" for p in parts)
    return dict(json.loads(text))


def test_phases_retry_only_task_start_errors(template: Template) -> None:
    states = _definition(template)["States"]
    phases = [s for name, s in states.items() if name.endswith("Phase")]
    assert len(phases) == 4
    for state in phases:
        (retry,) = state["Retry"]
        assert retry["ErrorEquals"] == ["ECS.AmazonECSException", "AmazonECS.Unknown"]
        (catch,) = state["Catch"]
        assert catch == {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "NotifyFailure",
        }
    message = states["NotifyFailure"]["Parameters"]["Message"]
    assert message["execution_id.$"] == "$$.Execution.Id"
    # The whole error object: a States.TaskFailed or States.Timeout error may
    # carry no Cause, and a "$.error.Cause" path would fail the publish.
    assert message["error.$"] == "States.JsonToString($.error)"
    assert not [v for v in message.values() if "$.error." in str(v)]


def test_state_machine_timeout_is_configurable() -> None:
    template = _orchestration_template({"sfn_timeout_hours": 12})
    assert _definition(template)["TimeoutSeconds"] == 12 * 3600


def test_phases_auto_resume_within_their_stage_window(template: Template) -> None:
    # Each phase passes its window and no --resume-from-stage: an explicit
    # start would re-run the whole window on a retried execution instead of
    # skipping the stages a previous attempt completed.
    states = _definition(template)["States"]
    windows = []
    for name in ("PrepPhase", "GraphBuildPhase", "AnalysisPhase", "IndexPhase"):
        (override,) = states[name]["Parameters"]["Overrides"]["ContainerOverrides"]
        command = override["Command"]
        assert "--resume-from-stage" not in command
        windows.append(command[command.index("--enabled-stages") + 1].split(","))
    # Together the windows cover every stage once, in pipeline order.
    assert [stage for window in windows for stage in window] == [
        "document_parsing",
        "document_loading",
        "text_chunking",
        "translation",
        "graph_extraction",
        "gleaning",
        "graph_resolution",
        "claim_extraction",
        "claim_resolution",
        "graph_analysis",
        "community_detection",
        "indexing",
    ]
