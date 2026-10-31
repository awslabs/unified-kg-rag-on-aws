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


def test_guardrail_does_not_anonymize_names() -> None:
    # The guardrail runs on the query path; NAME anonymization would put
    # "{NAME}" in answers and strip entity-search seeds.
    _, template = _synth({})
    (guardrail,) = template.find_resources(_GUARDRAIL).values()
    pii = guardrail["Properties"]["SensitiveInformationPolicyConfig"]
    types = {e["Type"] for e in pii["PiiEntitiesConfig"]}
    assert "NAME" not in types
    assert {"EMAIL", "PHONE", "CREDIT_DEBIT_CARD_NUMBER"} <= types


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


_VERSION = "AWS::Bedrock::GuardrailVersion"


def test_the_created_guardrail_is_published_as_a_retained_version() -> None:
    for context in ({}, {"env_name": "prod"}):
        _, template = _synth(context)
        template.resource_count_is(_VERSION, 1)
        # Retained even in dev: a replaced version stays usable by a task
        # still pinned to it; the versions go with the guardrail.
        template.has_resource(
            _VERSION, {"DeletionPolicy": "Retain", "UpdateReplacePolicy": "Retain"}
        )
        template.has_output("GuardrailVersion", {})


def test_a_policy_change_publishes_a_new_version(monkeypatch) -> None:  # noqa: ANN001
    from iac.stacks import guardrail_stack

    def description() -> str:
        (version,) = _synth({})[1].find_resources(_VERSION).values()
        return str(version["Properties"]["Description"])

    before = description()
    assert description() == before  # deterministic across synths
    monkeypatch.setattr(guardrail_stack, "PII_ANONYMIZED", ("EMAIL",))
    # A replaced description replaces the version resource: a new version.
    assert description() != before


def test_bring_your_own_publishes_no_version() -> None:
    _, template = _synth(
        {"create_guardrail": "false", "guardrail_identifier": "gr-external456"}
    )
    template.resource_count_is(_VERSION, 0)


_GROUNDING = {"guardrail_contextual_grounding": "true"}


def test_contextual_grounding_is_off_by_default() -> None:
    _, template = _synth({})
    (guardrail,) = template.find_resources(_GUARDRAIL).values()
    assert "ContextualGroundingPolicyConfig" not in guardrail["Properties"]


def test_contextual_grounding_adds_blocking_filters() -> None:
    _, template = _synth({**_GROUNDING, "guardrail_grounding_threshold": "0.8"})
    (guardrail,) = template.find_resources(_GUARDRAIL).values()
    policy = guardrail["Properties"]["ContextualGroundingPolicyConfig"]
    assert {
        f["Type"]: (f["Threshold"], f["Action"], f["Enabled"])
        for f in policy["FiltersConfig"]
    } == {"GROUNDING": (0.8, "BLOCK", True), "RELEVANCE": (0.8, "BLOCK", True)}
    # The rest of the baseline policy is unchanged.
    assert "SensitiveInformationPolicyConfig" in guardrail["Properties"]


def test_contextual_grounding_publishes_a_new_version() -> None:
    def description(context: dict[str, Any]) -> str:
        (version,) = _synth(context)[1].find_resources(_VERSION).values()
        return str(version["Properties"]["Description"])

    assert description(_GROUNDING) != description({})
    assert description({**_GROUNDING, "guardrail_grounding_threshold": "0.6"}) != (
        description(_GROUNDING)
    )


def test_contextual_grounding_threshold_validation() -> None:
    import pytest

    for value in ("1", "-0.1", "high"):
        with pytest.raises(ValueError, match="guardrail_grounding_threshold"):
            DeploymentConfig.from_context(
                cdk.App(context={**_GROUNDING, "guardrail_grounding_threshold": value})
            )
    with pytest.raises(ValueError, match="only applies with"):
        DeploymentConfig.from_context(
            cdk.App(context={"guardrail_grounding_threshold": "0.5"})
        )
