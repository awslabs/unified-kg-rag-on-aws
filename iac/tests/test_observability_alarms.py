# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ObservabilityStack alarm assertions (offline synth, no lookups)."""

from __future__ import annotations

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


def _alarms(context: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
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
    stack = ObservabilityStack(
        app,
        "Obs",
        config=config,
        orchestration=orchestration,
        storage=storage,
        env=_ENV,
    )
    alarms = Template.from_stack(stack).find_resources("AWS::CloudWatch::Alarm")
    # Logical ids are "<construct id><hash>"; key by the construct id.
    return {logical_id[:-8]: a["Properties"] for logical_id, a in alarms.items()}


def test_timed_out_and_aborted_runs_alarm() -> None:
    alarm = _alarms()["PipelineTimedOutOrAborted"]
    metrics = {
        m["MetricStat"]["Metric"]["MetricName"]
        for m in alarm["Metrics"]
        if "MetricStat" in m
    }
    assert metrics == {"ExecutionsTimedOut", "ExecutionsAborted"}


def test_store_alarms_watch_the_paths_the_app_uses() -> None:
    alarms = _alarms()
    # Gremlin runs over WebSocket; GremlinHttp5xx counts HTTP only.
    assert alarms["NeptuneGremlinErrors"]["MetricName"] == "GremlinServerErrorsPerSec"
    # Registry writes are BatchWriteItem, not a single-operation PutItem.
    throttle = alarms["DocStatusWriteThrottled"]
    assert throttle["MetricName"] == "WriteThrottleEvents"
    assert "Operation" not in {d["Name"] for d in throttle["Dimensions"]}


def test_free_storage_threshold_follows_volume_size() -> None:
    alarm = _alarms({"opensearch_volume_gib": 100})["OpenSearchFreeStorageLow"]
    assert alarm["Threshold"] == 100 * 1024 * 0.4
