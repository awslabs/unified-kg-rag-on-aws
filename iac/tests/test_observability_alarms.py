# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ObservabilityStack alarm assertions (offline synth, no lookups)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import aws_cdk as cdk
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.observability_stack import EMF_NAMESPACE, ObservabilityStack
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")
# The IaC environment does not install the package, so read the emitted metric
# names from its source instead of importing it.
_PACKAGE = Path(__file__).resolve().parents[2] / "unified_kg_rag"


def _template(context: dict[str, Any] | None = None) -> Template:
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
    return Template.from_stack(stack)


def _alarms(context: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    alarms = _template(context).find_resources("AWS::CloudWatch::Alarm")
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


def test_failed_executions_alarm() -> None:
    # The task's Catch publishes most failures with context, but it cannot catch
    # States.DataLimitExceeded or States.Runtime and its publish can fail, so
    # failed executions also alarm (a caught failure notifies twice).
    alarm = _alarms()["PipelineFailures"]
    assert alarm["MetricName"] == "ExecutionsFailed"
    assert alarm["Namespace"] == "AWS/States"
    assert (alarm["Threshold"], alarm["EvaluationPeriods"]) == (1, 1)
    assert len(alarm["AlarmActions"]) == 1


def test_quarantined_documents_alarm() -> None:
    # A quarantined document is no longer indexed: someone has to look.
    alarm = _alarms()["DocumentsQuarantined"]
    assert alarm["MetricName"] == "total_documents_quarantined"
    assert (alarm["Threshold"], alarm["EvaluationPeriods"]) == (1, 1)
    assert alarm["TreatMissingData"] == "notBreaching"
    assert len(alarm["AlarmActions"]) == 1


def test_store_alarms_watch_the_paths_the_app_uses() -> None:
    alarms = _alarms()
    # Gremlin runs over WebSocket; GremlinHttp5xx counts HTTP only.
    gremlin = alarms["NeptuneGremlinErrors"]
    assert gremlin["MetricName"] == "GremlinServerErrorsPerSec"
    # Retried ConcurrentModificationExceptions during parallel writes come in
    # bursts: only errors in 3 consecutive 1-minute periods alarm.
    assert (gremlin["Period"], gremlin["EvaluationPeriods"]) == (60, 3)
    # Registry writes are BatchWriteItem, not a single-operation PutItem.
    throttle = alarms["DocStatusWriteThrottled"]
    assert throttle["MetricName"] == "WriteThrottleEvents"
    assert "Operation" not in {d["Name"] for d in throttle["Dimensions"]}


def test_neptune_resource_alarms_watch_the_writer() -> None:
    # With neptune_instances=2 an idle reader would halve a cluster-wide
    # average, hiding a saturated or memory-starved writer.
    alarms = _alarms({"neptune_instances": 2})
    for name in ("NeptuneHighCpu", "NeptuneLowFreeableMemory"):
        dims = {d["Name"]: d["Value"] for d in alarms[name]["Dimensions"]}
        assert dims.keys() == {"DBClusterIdentifier", "Role"}
        assert dims["Role"] == "WRITER"
    assert alarms["NeptuneHighCpu"]["Statistic"] == "Average"
    assert alarms["NeptuneLowFreeableMemory"]["Statistic"] == "Minimum"
    # Gremlin errors can come from any instance: stay cluster-wide.
    gremlin = alarms["NeptuneGremlinErrors"]
    assert [d["Name"] for d in gremlin["Dimensions"]] == ["DBClusterIdentifier"]


def test_free_storage_threshold_follows_volume_size() -> None:
    alarm = _alarms({"opensearch_volume_gib": 100})["OpenSearchFreeStorageLow"]
    assert alarm["Threshold"] == 100 * 1024 * 0.4


def test_resource_alarms_need_sustained_pressure() -> None:
    alarms = _alarms()
    # AWS's recommended OpenSearch alarm: maximum >= 95% in 3 consecutive
    # 1-minute periods.
    jvm = alarms["OpenSearchJvmPressure"]
    assert (jvm["Statistic"], jvm["Threshold"]) == ("Maximum", 95)
    assert (jvm["Period"], jvm["EvaluationPeriods"]) == (60, 3)
    # Bulk edge writes saturate the Neptune writer briefly by design.
    cpu = alarms["NeptuneHighCpu"]
    assert (cpu["Period"], cpu["EvaluationPeriods"]) == (300, 3)


def _pipeline_metric_fields() -> set[str]:
    source = _PACKAGE / "domain" / "models" / "pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    (model,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PipelineMetrics"
    )
    return {
        node.target.id
        for node in model.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _emf_metric_names(template: Template) -> set[str]:
    names: set[str] = set()
    for alarm in template.find_resources("AWS::CloudWatch::Alarm").values():
        props = alarm["Properties"]
        stats = [props] + [
            m["MetricStat"]["Metric"]
            for m in props.get("Metrics", [])
            if "MetricStat" in m
        ]
        names |= {s["MetricName"] for s in stats if s.get("Namespace") == EMF_NAMESPACE}
    for dashboard in template.find_resources("AWS::CloudWatch::Dashboard").values():
        # The body is an Fn::Join of JSON fragments; widget metrics render as
        # ["<namespace>","<name>",...] inside its string parts.
        _, parts = dashboard["Properties"]["DashboardBody"]["Fn::Join"]
        body = "".join(part for part in parts if isinstance(part, str))
        pattern = '"' + re.escape(EMF_NAMESPACE) + r'","(\w+)"'
        names |= set(re.findall(pattern, body))
    return names


def test_emf_metric_names_are_pipeline_metrics_fields() -> None:
    # A renamed PipelineMetrics field would leave an alarm on a series nobody
    # emits (never fires, TreatMissingData=notBreaching) or an empty widget.
    pipeline = (_PACKAGE / "application" / "ingestion" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    assert f'namespace="{EMF_NAMESPACE}"' in pipeline
    assert 'f"duration_{stage_name}_seconds"' in pipeline
    names = _emf_metric_names(_template())
    assert {"total_items_index_failed", "total_entities_extracted"} <= names
    durations = {f"duration_{stage}_seconds" for stage in _pipeline_stages()}
    assert names - _pipeline_metric_fields() - durations == set()


def _pipeline_stages() -> list[str]:
    source = _PACKAGE / "domain" / "models" / "config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    (enum,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PipelineStageType"
    )
    return [
        node.value.value
        for node in enum.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]


def test_the_duration_widget_lists_every_stage() -> None:
    from iac.stacks.observability_stack import PIPELINE_STAGES

    assert list(PIPELINE_STAGES) == _pipeline_stages()


def test_emf_alarms_watch_only_this_environment() -> None:
    # A dev run in the same account and region must not trip prod's alarms:
    # the task tags its EMF series with its env_name, the alarms filter on it.
    for env in ("dev", "prod"):
        template = _template({"env_name": env})
        emf = [
            stat
            for alarm in template.find_resources("AWS::CloudWatch::Alarm").values()
            for stat in [alarm["Properties"]]
            + [
                m["MetricStat"]["Metric"]
                for m in alarm["Properties"].get("Metrics", [])
                if "MetricStat" in m
            ]
            if stat.get("Namespace") == EMF_NAMESPACE
        ]
        assert emf
        for stat in emf:
            assert stat["Dimensions"] == [{"Name": "Environment", "Value": env}]


def test_sustained_bedrock_throttling_alarms() -> None:
    alarm = _alarms()["BedrockThrottling"]
    (query,) = alarm["Metrics"]
    # Summed over every model: Bedrock publishes per ModelId only.
    assert "SUM(InvocationThrottles)" in query["Expression"]
    assert 'SCHEMA("AWS/Bedrock", ModelId)' in query["Expression"]
    assert (alarm["Threshold"], alarm["EvaluationPeriods"]) == (100, 3)
    assert alarm["TreatMissingData"] == "notBreaching"
    assert len(alarm["AlarmActions"]) == 1


def test_ingestion_task_memory_alarms() -> None:
    alarm = _alarms()["IngestionTaskMemory"]
    stats = {
        m["MetricStat"]["Metric"]["MetricName"]: m["MetricStat"]["Metric"]
        for m in alarm["Metrics"]
        if "MetricStat" in m
    }
    assert set(stats) == {"MemoryUtilized", "MemoryReserved"}
    for metric in stats.values():
        assert metric["Namespace"] == "ECS/ContainerInsights"
        dims = {d["Name"] for d in metric["Dimensions"]}
        # The ingestion task's family, not the query task's.
        assert dims == {"ClusterName", "TaskDefinitionFamily"}
    assert alarm["Threshold"] == 90
