# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fargate Spot ingestion assertions (offline synth, no lookups).

With ``ingestion_fargate_spot`` the ingestion task runs on FARGATE_SPOT, and
the state machine runs a task Spot interrupted again with the same input (so
the same pipeline id resumes from its checkpoint), up to
``spot_interruption_retries`` times; every other failure is notified as
before. Off, the templates are exactly as without the feature.

The Choice rules are evaluated here against the error payloads Step
Functions hands the catch, with a small evaluator of the operators the
definition uses.
"""

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
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")
_SPOT = {"ingestion_fargate_spot": "true"}


def _stacks(context: dict[str, Any]) -> tuple[Template, Template]:
    app = cdk.App(context=context)
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
    return Template.from_stack(compute), Template.from_stack(orchestration)


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "Fn::Join" in value:
        return "".join(_render(part) for part in value["Fn::Join"][1])
    return "TOKEN"


def _definition(orchestration: Template) -> dict[str, Any]:
    (machine,) = orchestration.find_resources(
        "AWS::StepFunctions::StateMachine"
    ).values()
    rendered = _render(machine["Properties"]["DefinitionString"])
    # Tokens sit inside JSON strings only, so the rendered text parses.
    return json.loads(re.sub(r'"[^"]*TOKEN[^"]*"', '"TOKEN"', rendered))


# --------------------------------------------- a Choice-rule evaluator (ASL)
_MISSING = object()


def _path(state: dict[str, Any], path: str) -> Any:
    node: Any = state
    for key in path.removeprefix("$.").split("."):
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _glob(pattern: str) -> re.Pattern[str]:
    # ASL StringMatches: `*` matches any run of characters, `\*` a literal.
    parts = re.split(r"(\\\*|\*)", pattern)
    regex = "".join(
        ".*" if part == "*" else re.escape("*" if part == "\\*" else part)
        for part in parts
    )
    return re.compile(regex, re.DOTALL)


def _matches(rule: dict[str, Any], state: dict[str, Any]) -> bool:
    if "And" in rule:
        return all(_matches(sub, state) for sub in rule["And"])
    if "Or" in rule:
        return any(_matches(sub, state) for sub in rule["Or"])
    if "Not" in rule:
        return not _matches(rule["Not"], state)
    value = _path(state, rule["Variable"])
    if "IsPresent" in rule:
        return (value is not _MISSING) == rule["IsPresent"]
    if value is _MISSING:
        raise AssertionError(f"unguarded path {rule['Variable']}")
    if "IsString" in rule:
        return isinstance(value, str) == rule["IsString"]
    # A string comparison of a value that may not be one is guarded by an
    # IsString rule before it (as IsPresent guards a path that may be absent).
    if "StringEquals" in rule or "StringMatches" in rule:
        if not isinstance(value, str):
            raise AssertionError(f"unguarded string comparison of {value!r}")
        if "StringEquals" in rule:
            return value == rule["StringEquals"]
        return bool(_glob(rule["StringMatches"]).fullmatch(value))
    if "NumericLessThan" in rule:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value < rule["NumericLessThan"]
        )
    raise AssertionError(f"operator not covered by the evaluator: {rule}")


def _next_state(choice: dict[str, Any], state: dict[str, Any]) -> str:
    for rule in choice["Choices"]:
        if _matches(rule, state):
            return rule["Next"]
    return choice["Default"]


def _stopped_task(stop_code: str, reason: str, exit_code: int | None) -> str:
    """A Cause as Step Functions renders a stopped task (DescribeTasks)."""
    container: dict[str, Any] = {"Name": "app", "LastStatus": "STOPPED"}
    if exit_code is not None:
        container["ExitCode"] = exit_code
    return json.dumps(
        {
            "CapacityProviderName": "FARGATE_SPOT",
            "Containers": [container],
            "LastStatus": "STOPPED",
            "StopCode": stop_code,
            "StoppedReason": reason,
            "Overrides": {
                "ContainerOverrides": [
                    {
                        "Name": "app",
                        "Environment": [
                            {"Name": "UKR_PIPELINE_ID", "Value": "run-001"}
                        ],
                    }
                ]
            },
        },
        separators=(",", ":"),
    )


_SPOT_CAUSE = _stopped_task("SpotInterruption", "Your Spot Task was interrupted.", 143)


def _state(error: dict[str, Any], retries: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "source_directory": "s3://example-bucket/corpus/",
        "pipeline_id": "run-001",
        "spot_retries": {"count": retries},
        "error": error,
        **extra,
    }


@pytest.fixture(scope="module")
def spot() -> tuple[Template, dict[str, Any]]:
    compute, orchestration = _stacks(_SPOT)
    return compute, _definition(orchestration)


# ------------------------------------------------------------------- tests
def test_off_by_default_adds_nothing() -> None:
    compute, orchestration = _stacks({})
    definition = _definition(orchestration)
    assert definition["StartAt"] == "Ingest"
    ingest = definition["States"]["Ingest"]
    assert ingest["Parameters"]["LaunchType"] == "FARGATE"
    assert "CapacityProviderStrategy" not in ingest["Parameters"]
    assert ingest["Catch"][0]["Next"] == "NotifyFailure"
    assert not {"InitSpotRetries", "SpotInterrupted", "CountSpotRetry"} & set(
        definition["States"]
    )
    assert not compute.find_resources("AWS::ECS::ClusterCapacityProviderAssociations")


def test_spot_runs_the_ingest_task_on_fargate_spot(
    spot: tuple[Template, dict[str, Any]],
) -> None:
    compute, definition = spot
    parameters = definition["States"]["Ingest"]["Parameters"]
    # RunTask takes a launch type or a capacity provider strategy, not both.
    assert "LaunchType" not in parameters
    assert parameters["CapacityProviderStrategy"] == [
        {"CapacityProvider": "FARGATE_SPOT", "Weight": 1}
    ]
    (association,) = compute.find_resources(
        "AWS::ECS::ClusterCapacityProviderAssociations"
    ).values()
    assert set(association["Properties"]["CapacityProviders"]) == {
        "FARGATE",
        "FARGATE_SPOT",
    }


def test_spot_retry_loop_reruns_the_same_task_after_the_wait(
    spot: tuple[Template, dict[str, Any]],
) -> None:
    _, definition = spot
    states = definition["States"]
    assert definition["StartAt"] == "InitSpotRetries"
    assert states["InitSpotRetries"]["Result"] == {"count": 0}
    assert states["InitSpotRetries"]["ResultPath"] == "$.spot_retries"
    assert states["InitSpotRetries"]["Next"] == "Ingest"
    assert states["Ingest"]["Catch"] == [
        {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "SpotInterrupted",
        }
    ]
    count = states["CountSpotRetry"]
    assert count["Parameters"] == {"count.$": "States.MathAdd($.spot_retries.count, 1)"}
    assert count["ResultPath"] == "$.spot_retries"
    assert count["Next"] == "WaitBeforeSpotRetry"
    assert states["WaitBeforeSpotRetry"] == {
        "Type": "Wait",
        "Seconds": 600,
        "Next": "Ingest",
    }
    # The re-run is the same state: the same command and input overrides.
    assert states["SpotInterrupted"]["Default"] == "NotifyFailure"


@pytest.mark.parametrize(
    ("error", "retries", "expected"),
    [
        # Spot interrupted the task; it stopped cleanly on SIGTERM.
        ({"Error": "States.TaskFailed", "Cause": _SPOT_CAUSE}, 0, "CountSpotRetry"),
        ({"Error": "States.TaskFailed", "Cause": _SPOT_CAUSE}, 1, "CountSpotRetry"),
        # Killed when the stop timeout ran out.
        (
            {
                "Error": "States.TaskFailed",
                "Cause": _stopped_task(
                    "SpotInterruption", "Your Spot Task was interrupted.", 137
                ),
            },
            0,
            "CountSpotRetry",
        ),
        # Either marker alone matches, whatever the key casing.
        (
            {
                "Error": "States.TaskFailed",
                "Cause": '{"stopCode": "SpotInterruption", "containers": []}',
            },
            0,
            "CountSpotRetry",
        ),
        (
            {
                "Error": "States.TaskFailed",
                "Cause": '{"stoppedReason": "Your Spot Task was interrupted."}',
            },
            0,
            "CountSpotRetry",
        ),
        # Retries used up.
        ({"Error": "States.TaskFailed", "Cause": _SPOT_CAUSE}, 2, "NotifyFailure"),
        # A failed pipeline, a user's StopTask, an OOM kill.
        (
            {
                "Error": "States.TaskFailed",
                "Cause": _stopped_task(
                    "EssentialContainerExited", "Essential container in task exited", 1
                ),
            },
            0,
            "NotifyFailure",
        ),
        (
            {
                "Error": "States.TaskFailed",
                "Cause": _stopped_task("UserInitiated", "Task stopped by user", 143),
            },
            0,
            "NotifyFailure",
        ),
        (
            {
                "Error": "States.TaskFailed",
                "Cause": _stopped_task(
                    "EssentialContainerExited",
                    "OutOfMemoryError: Container killed",
                    137,
                ),
            },
            0,
            "NotifyFailure",
        ),
        # ECS did not start the task (after its own retries), or no cause.
        (
            {"Error": "AmazonECS.Unknown", "Cause": "RESOURCE:FARGATE capacity"},
            0,
            "NotifyFailure",
        ),
        # Another error whose cause names a Spot interruption (a timeout of
        # the state, say) is not the task's failure: not run again.
        ({"Error": "States.Timeout", "Cause": _SPOT_CAUSE}, 0, "NotifyFailure"),
        ({"Error": "States.TaskFailed"}, 0, "NotifyFailure"),
        (
            {"Error": "States.TaskFailed", "Cause": {"not": "a string"}},
            0,
            "NotifyFailure",
        ),
    ],
)
def test_only_a_spot_interruption_is_run_again(
    spot: tuple[Template, dict[str, Any]],
    error: dict[str, Any],
    retries: int,
    expected: str,
) -> None:
    _, definition = spot
    choice = definition["States"]["SpotInterrupted"]
    assert _next_state(choice, _state(error, retries)) == expected


@pytest.mark.parametrize("pipeline_id", ["", None])
def test_spot_interruption_without_a_pipeline_id_is_not_run_again(
    spot: tuple[Template, dict[str, Any]], pipeline_id: str | None
) -> None:
    # An empty pipeline_id makes the CLI generate a new one: the re-run would
    # start a new pipeline instead of resuming.
    _, definition = spot
    state = _state({"Error": "States.TaskFailed", "Cause": _SPOT_CAUSE})
    state["pipeline_id"] = pipeline_id
    choice = definition["States"]["SpotInterrupted"]
    assert _next_state(choice, state) == "NotifyFailure"


def test_retry_count_and_wait_follow_the_context() -> None:
    _, orchestration = _stacks(
        {**_SPOT, "spot_interruption_retries": "4", "spot_retry_wait_seconds": "60"}
    )
    states = _definition(orchestration)["States"]
    assert states["WaitBeforeSpotRetry"]["Seconds"] == 60
    choice = states["SpotInterrupted"]
    error = {"Error": "States.TaskFailed", "Cause": _SPOT_CAUSE}
    assert _next_state(choice, _state(error, 3)) == "CountSpotRetry"
    assert _next_state(choice, _state(error, 4)) == "NotifyFailure"


def test_zero_retries_and_no_wait_are_accepted() -> None:
    # 0 retries: Spot capacity without the re-run (an interruption fails).
    _, orchestration = _stacks(
        {**_SPOT, "spot_interruption_retries": "0", "spot_retry_wait_seconds": "0"}
    )
    states = _definition(orchestration)["States"]
    assert states["WaitBeforeSpotRetry"]["Seconds"] == 0
    error = {"Error": "States.TaskFailed", "Cause": _SPOT_CAUSE}
    assert _next_state(states["SpotInterrupted"], _state(error, 0)) == "NotifyFailure"


@pytest.mark.parametrize(
    "key", ["spot_interruption_retries", "spot_retry_wait_seconds"]
)
def test_spot_settings_without_spot_fail(key: str) -> None:
    with pytest.raises(ValueError, match=f"{key} only apply with"):
        DeploymentConfig.from_context(cdk.App(context={key: "1"}))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("spot_interruption_retries", "11"),
        ("spot_interruption_retries", "-1"),
        ("spot_retry_wait_seconds", "-1"),
        ("spot_retry_wait_seconds", "86401"),
    ],
)
def test_spot_settings_out_of_range_fail(key: str, value: str) -> None:
    with pytest.raises(ValueError, match=key):
        DeploymentConfig.from_context(cdk.App(context={**_SPOT, key: value}))
