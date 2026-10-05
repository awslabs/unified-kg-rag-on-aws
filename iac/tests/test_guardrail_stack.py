# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GuardrailStack synth assertions.

Regression guard for the documented two-step deploy: step 2 passes
``-c guardrail_identifier=<id>`` and must NOT drop the ``CfnGuardrail`` created
in step 1 from the template (CloudFormation would delete it while compute keeps
injecting the dead id).
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
from aws_cdk.assertions import Annotations, Match, Template

from iac.config import DeploymentConfig
from iac.stacks.guardrail_stack import GuardrailStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")
_GUARDRAIL = "AWS::Bedrock::Guardrail"


def _synth(context: dict[str, Any]) -> tuple[GuardrailStack, Template]:
    app = cdk.App(context=context)
    config = DeploymentConfig.from_context(app)
    stack = GuardrailStack(app, "TestGuardrail", config=config, env=_ENV)
    return stack, Template.from_stack(stack)


def test_step1_creates_guardrail_destroyed_in_dev() -> None:
    # dev defaults to removal_destroy, so deploy -> destroy -> deploy does not
    # collide on the fixed guardrail name.
    stack, template = _synth({})
    template.resource_count_is(_GUARDRAIL, 1)
    template.has_resource(
        _GUARDRAIL, {"DeletionPolicy": "Delete", "UpdateReplacePolicy": "Delete"}
    )
    template.has_output("GuardrailIdentifier", {})
    assert stack.guardrail is not None


def test_non_dev_retains_guardrail() -> None:
    _, template = _synth({"env_name": "prod"})
    template.has_resource(
        _GUARDRAIL, {"DeletionPolicy": "Retain", "UpdateReplacePolicy": "Retain"}
    )


def test_removal_destroy_false_retains_guardrail_in_dev() -> None:
    _, template = _synth({"removal_destroy": "false"})
    template.has_resource(_GUARDRAIL, {"DeletionPolicy": "Retain"})


def test_removal_destroy_true_destroys_guardrail_outside_dev() -> None:
    _, template = _synth({"env_name": "prod", "removal_destroy": "true"})
    template.has_resource(_GUARDRAIL, {"DeletionPolicy": "Delete"})


def test_step2_with_identifier_keeps_guardrail() -> None:
    """Passing the id for compute must not remove the stack-owned guardrail."""
    step1 = _synth({})[1].find_resources(_GUARDRAIL)
    stack, template = _synth({"guardrail_identifier": "gr-example123"})
    step2 = template.find_resources(_GUARDRAIL)

    template.resource_count_is(_GUARDRAIL, 1)
    # Same logical id + properties => CloudFormation sees no change to the
    # guardrail between the two deploys (no delete, no replacement).
    assert step1 == step2
    assert stack.guardrail is not None


_BYO_WARNING = Match.string_like_regexp(
    "guardrail_identifier is set but create_guardrail"
)


def test_identifier_without_explicit_create_warns() -> None:
    """Bring-your-own users who only pass the id now also get a guardrail."""
    stack, _ = _synth({"guardrail_identifier": "gr-example123"})
    Annotations.from_stack(stack).has_warning("*", _BYO_WARNING)


def test_identifier_with_explicit_create_true_does_not_warn() -> None:
    stack, template = _synth(
        {"guardrail_identifier": "gr-example123", "create_guardrail": "true"}
    )
    template.resource_count_is(_GUARDRAIL, 1)
    Annotations.from_stack(stack).has_no_warning("*", _BYO_WARNING)


def test_no_identifier_does_not_warn() -> None:
    stack, _ = _synth({})
    Annotations.from_stack(stack).has_no_warning("*", _BYO_WARNING)


def test_bring_your_own_creates_nothing() -> None:
    stack, template = _synth(
        {"create_guardrail": "false", "guardrail_identifier": "gr-external456"}
    )
    template.resource_count_is(_GUARDRAIL, 0)
    template.has_output("GuardrailIdentifier", {"Value": "gr-external456"})
    Annotations.from_stack(stack).has_no_warning("*", _BYO_WARNING)
    assert stack.guardrail is None
    assert stack.guardrail_identifier == "gr-external456"


def test_create_disabled_without_identifier_is_empty() -> None:
    stack, template = _synth({"create_guardrail": False})
    template.resource_count_is(_GUARDRAIL, 0)
    assert template.find_outputs("GuardrailIdentifier") == {}
    assert stack.guardrail_identifier is None


def test_create_guardrail_context_parsing() -> None:
    def parse(value: Any) -> bool:
        ctx = {} if value is None else {"create_guardrail": value}
        return DeploymentConfig.from_context(cdk.App(context=ctx)).create_guardrail

    assert parse(None) is True
    assert parse("true") is True
    assert parse("false") is False

    def explicit(ctx: dict[str, Any]) -> bool:
        return DeploymentConfig.from_context(
            cdk.App(context=ctx)
        ).create_guardrail_explicit

    assert explicit({}) is False
    assert explicit({"create_guardrail": "true"}) is True
    assert explicit({"create_guardrail": False}) is True
    assert parse(False) is False
