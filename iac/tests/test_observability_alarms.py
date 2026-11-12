# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ObservabilityStack alarm assertions (offline synth, no lookups)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Annotations, Match, Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.observability_stack import (
    DOCUMENT_METRICS,
    EMF_NAMESPACE,
    NOT_INDEXED_METRICS,
    QUERY_LATENCY_METRIC,
    QUERY_NAMESPACE,
    SUMMARY_TREE_METRICS,
    ObservabilityStack,
)
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")
# The IaC environment does not install the package, so read the emitted metric
# names from its source instead of importing it.
_PACKAGE = Path(__file__).resolve().parents[2] / "unified_kg_rag"


def _template(context: dict[str, Any] | None = None) -> Template:
    return Template.from_stack(_stack(context))


def _stack(context: dict[str, Any] | None = None) -> ObservabilityStack:
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
    return stack


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
    # No execution in a period is no failure (INSUFFICIENT_DATA before).
    assert alarm["TreatMissingData"] == "notBreaching"


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


def test_serverless_memory_alarm_fires_only_at_maximum_capacity() -> None:
    # At its minimum a serverless writer runs with little free memory by
    # design (it scales up on it): a fixed 512 MiB floor would always fire.
    alarms = _alarms({"neptune_serverless_ncu": "1-16"})
    memory = alarms["NeptuneLowFreeableMemory"]
    (expression,) = [m for m in memory["Metrics"] if "Expression" in m]
    assert expression["Expression"] == "IF(capacity >= 16, memory, 1000000000000000)"
    used = {
        m["Id"]: m["MetricStat"]["Metric"]["MetricName"]
        for m in memory["Metrics"]
        if "MetricStat" in m
    }
    assert used == {
        "capacity": "ServerlessDatabaseCapacity",
        "memory": "FreeableMemory",
    }
    at_max = alarms["NeptuneServerlessAtMaxCapacity"]
    assert at_max["MetricName"] == "ServerlessDatabaseCapacity"
    assert at_max["Threshold"] == 16
    assert at_max["ComparisonOperator"] == "GreaterThanOrEqualToThreshold"
    assert at_max["EvaluationPeriods"] == 3
    # Provisioned: the fixed floor, and no capacity alarm.
    provisioned = _alarms()
    assert provisioned["NeptuneLowFreeableMemory"]["Threshold"] == 536870912
    assert "NeptuneServerlessAtMaxCapacity" not in provisioned


@pytest.mark.parametrize(
    ("instance", "mib"),
    [
        ("db.r6g.large", 512),
        ("db.r6g.4xlarge", 4096),
        ("db.x2g.2xlarge", 4096),
        ("db.t4g.medium", 128),
        ("db.serverless", 512),
        ("db.r6g.metal", 512),
    ],
)
def test_neptune_memory_alarm_follows_the_instance_memory(
    instance: str, mib: int
) -> None:
    # The same share of the instance's memory as 512 MiB is of the default
    # class: a fixed 512 MiB fired only when a large writer was nearly out.
    # A class the table does not know keeps 512 MiB.
    alarm = _alarms({"neptune_instance": instance})["NeptuneLowFreeableMemory"]
    assert alarm["Threshold"] == mib * 1024**2


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


def _metric_stages() -> dict[str, set[str]]:
    # pipeline_metrics.METRIC_STAGES (metric -> the stages that emit it),
    # read from the source: this venv does not install the package.
    source = _PACKAGE / "application" / "ingestion" / "pipeline_metrics.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    (table,) = (
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "METRIC_STAGES"
    )
    assert isinstance(table, ast.Dict)
    stages: dict[str, set[str]] = {}
    for key, value in zip(table.keys, table.values, strict=True):
        assert isinstance(key, ast.Constant) and isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Name) and value.func.id == "frozenset"
        stages[key.value] = set(ast.literal_eval(value.args[0]))
    return stages


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
    pipeline = (
        _PACKAGE / "application" / "ingestion" / "pipeline_metrics.py"
    ).read_text(encoding="utf-8")
    assert f'namespace="{EMF_NAMESPACE}"' in pipeline
    assert 'f"duration_{stage_name}_seconds"' in pipeline
    for usage_name in (
        'f"llm_calls_{stage_name}"',
        'f"llm_input_tokens_{stage_name}"',
        'f"llm_output_tokens_{stage_name}"',
        'f"llm_throttles_{stage_name}"',
        '"llm_cache_hits"',
        '"llm_throttles"',
        '"llm_requests"',
    ):
        assert usage_name in pipeline
    names = _emf_metric_names(_template())
    assert {"total_items_index_failed", "total_entities_extracted"} <= names
    durations = {f"duration_{stage}_seconds" for stage in _pipeline_stages()}
    durations |= {
        f"llm_{kind}_{stage}"
        for kind in ("calls", "input_tokens", "output_tokens", "throttles")
        for stage in _pipeline_stages()
    } | {"llm_cache_hits", "llm_throttles", "llm_requests"}
    fields = _pipeline_metric_fields()
    # Every document outcome is a PipelineMetrics field (a name drifted from
    # DOCUMENT_METRICS would alarm on a series nobody emits).
    assert set(DOCUMENT_METRICS) <= fields
    assert names - fields - durations == set()


def test_documents_not_indexed_alarm_sums_the_not_indexed_outcomes() -> None:
    alarm = _alarms()["DocumentsNotIndexed"]
    used = [m for m in alarm["Metrics"] if "MetricStat" in m]
    assert {m["MetricStat"]["Metric"]["MetricName"] for m in used} == set(
        NOT_INDEXED_METRICS
    )
    (expression,) = [m for m in alarm["Metrics"] if "Expression" in m]
    assert expression["Expression"] == " + ".join(f"FILL({m['Id']}, 0)" for m in used)
    assert (alarm["Threshold"], alarm["EvaluationPeriods"]) == (1, 1)
    assert alarm["TreatMissingData"] == "notBreaching"
    assert set(NOT_INDEXED_METRICS) <= set(DOCUMENT_METRICS)
    # Failures only: a corpus holding files no loader reads would alarm on
    # every run, so skipped files are on the dashboard alone.
    assert set(NOT_INDEXED_METRICS) == {
        "total_files_failed",
        "total_documents_failed",
    }
    # Every document outcome is on the dashboard.
    assert set(DOCUMENT_METRICS) <= _emf_metric_names(_template())


def _pipeline_stages() -> list[str]:
    package = _PACKAGE / "domain" / "models" / "config"
    (enum,) = (
        node
        for source in sorted(package.glob("*.py"))
        for node in ast.parse(source.read_text(encoding="utf-8")).body
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


def test_the_token_widgets_plot_every_model_stage_and_cache_hits() -> None:
    from iac.stacks.observability_stack import LLM_STAGES

    names = _emf_metric_names(_template())
    model_stages = [
        stage
        for stage in _pipeline_stages()
        if stage not in ("document_parsing", "document_loading")
    ]
    assert list(LLM_STAGES) == model_stages
    for stage in model_stages:
        assert {
            f"llm_calls_{stage}",
            f"llm_input_tokens_{stage}",
            f"llm_output_tokens_{stage}",
        } <= names
    assert "llm_cache_hits" in names
    body = json.dumps(
        next(iter(_template().find_resources("AWS::CloudWatch::Dashboard").values()))[
            "Properties"
        ]["DashboardBody"]
    )
    assert "LLM tokens by stage (EMF)" in body
    assert "LLM calls by stage and response-cache hits (EMF)" in body


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


def test_bedrock_throttling_alarms_on_this_deployments_share() -> None:
    # Old behaviour: an absolute count of InvocationThrottles over every
    # caller in the account and region, which other users tripped and which
    # could not fire with another bedrock_region.
    for context in ({}, {"bedrock_region": "us-east-1", "network_mode": "public"}):
        alarm = _alarms(context)["BedrockThrottling"]
        (expression,) = [m for m in alarm["Metrics"] if "Expression" in m]
        assert expression["Expression"] == "100 * FILL(throttled, 0) / requests"
        stats = {
            m["Id"]: m["MetricStat"]["Metric"]
            for m in alarm["Metrics"]
            if "MetricStat" in m
        }
        assert stats["throttled"]["MetricName"] == "llm_throttles"
        assert stats["requests"]["MetricName"] == "llm_requests"
        for metric in stats.values():
            assert metric["Namespace"] == EMF_NAMESPACE
            assert metric["Dimensions"] == [{"Name": "Environment", "Value": "dev"}]
        assert (alarm["Threshold"], alarm["EvaluationPeriods"]) == (10, 1)
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


def _metric_names(alarm: dict[str, Any]) -> set[str]:
    stats = [alarm] + [
        m["MetricStat"]["Metric"] for m in alarm.get("Metrics", []) if "MetricStat" in m
    ]
    return {stat["MetricName"] for stat in stats if "MetricName" in stat}


def test_recommended_opensearch_alarms() -> None:
    alarms = _alarms()
    for name, metric, period, periods in (
        ("OpenSearchAutomatedSnapshotFailure", "AutomatedSnapshotFailure", 60, 1),
        ("OpenSearchIndexWritesBlocked", "ClusterIndexWritesBlocked", 300, 1),
    ):
        alarm = alarms[name]
        assert alarm["MetricName"] == metric
        assert alarm["Namespace"] == "AWS/ES"
        assert alarm["Statistic"] == "Maximum"
        assert (alarm["Period"], alarm["EvaluationPeriods"]) == (period, periods)
        assert alarm["Threshold"] == 1
        assert len(alarm["AlarmActions"]) == 1
    # Rejection counters are cumulative: the alarm reads their increase. On
    # 7.1+ engines bulk 429s come mostly from indexing pressure, which the
    # write thread-pool counter does not count.
    for name, metrics, periods in (
        ("OpenSearchWriteRejections", {"ThreadpoolWriteRejected"}, 3),
        (
            "OpenSearchIndexingPressureRejections",
            {
                "CoordinatingWriteRejected",
                "PrimaryWriteRejected",
                "ReplicaWriteRejected",
            },
            3,
        ),
        ("OpenSearchSearchRejections", {"ThreadpoolSearchRejected"}, 1),
    ):
        alarm = alarms[name]
        (expression,) = [m for m in alarm["Metrics"] if "Expression" in m]
        ids = sorted(m["Id"] for m in alarm["Metrics"] if "MetricStat" in m)
        # DIFF per counter: a FILL inside DIFF would read a gap as a reset.
        assert expression["Expression"] == " + ".join(
            f"FILL(DIFF({i}), 0)" for i in ids
        )
        assert _metric_names(alarm) == metrics
        for m in alarm["Metrics"]:
            if "MetricStat" in m:
                assert (m["MetricStat"]["Stat"], m["MetricStat"]["Period"]) == (
                    "Sum",
                    60,
                )
        assert alarm["EvaluationPeriods"] == periods
    # One data node holds no replicas, and without a CMK no key can fail.
    assert (
        not {
            "OpenSearchClusterYellow",
            "OpenSearchKmsKeyError",
            "OpenSearchKmsKeyInaccessible",
        }
        & alarms.keys()
    )


# AWS's "Recommended CloudWatch alarms for Amazon OpenSearch Service":
# name -> (metric, statistic, period, evaluation periods, comparison, threshold).
_RECOMMENDED = {
    "OpenSearchHighCpu": ("CPUUtilization", "Maximum", 900, 3, ">=", 80),
    "OpenSearchOldGenJvmPressure": (
        "OldGenJVMMemoryPressure",
        "Maximum",
        60,
        3,
        ">=",
        80,
    ),
    "OpenSearchWriteQueue": ("ThreadpoolWriteQueue", "Average", 60, 1, ">=", 100),
    "OpenSearchSearchQueue": ("ThreadpoolSearchQueue", "Average", 60, 1, ">=", 500),
}
_MASTER = {
    "OpenSearchMasterHighCpu": ("MasterCPUUtilization", "Maximum", 900, 3, ">=", 50),
    "OpenSearchMasterJvmPressure": (
        "MasterJVMMemoryPressure",
        "Maximum",
        60,
        3,
        ">=",
        95,
    ),
    "OpenSearchMasterOldGenJvmPressure": (
        "MasterOldGenJVMMemoryPressure",
        "Maximum",
        60,
        3,
        ">=",
        80,
    ),
    "OpenSearchMasterUnreachable": (
        "MasterReachableFromNode",
        "Maximum",
        300,
        1,
        "<",
        1,
    ),
}
_OPERATORS = {
    ">=": "GreaterThanOrEqualToThreshold",
    "<": "LessThanThreshold",
}


def _assert_alarm(alarm: dict[str, Any], expected: tuple[Any, ...]) -> None:
    metric, statistic, period, periods, comparison, threshold = expected
    assert alarm["Namespace"] == "AWS/ES"
    assert alarm["MetricName"] == metric
    assert (alarm["Statistic"], alarm["Period"]) == (statistic, period)
    assert alarm["EvaluationPeriods"] == periods
    assert alarm["ComparisonOperator"] == _OPERATORS[comparison]
    assert alarm["Threshold"] == threshold
    assert alarm["AlarmActions"]


@pytest.mark.parametrize("count", [1, 2])
def test_aws_recommended_opensearch_capacity_alarms(count: int) -> None:
    # Old behaviour: no CPU, old-generation heap, node, shard, 5xx or queue
    # alarm, and none on the dedicated masters of a multi-node domain.
    alarms = _alarms({"opensearch_count": count})
    for name, expected in _RECOMMENDED.items():
        _assert_alarm(alarms[name], expected)
    # Masters only exist on a multi-node domain (3 of them).
    if count > 1:
        for name, expected in _MASTER.items():
            _assert_alarm(alarms[name], expected)
    else:
        assert not _MASTER.keys() & alarms.keys()
    nodes = alarms["OpenSearchNodesMissing"]
    assert (nodes["MetricName"], nodes["Statistic"]) == ("Nodes", "Minimum")
    assert (nodes["Period"], nodes["ComparisonOperator"]) == (
        86400,
        "LessThanThreshold",
    )
    assert nodes["Threshold"] == (count + 3 if count > 1 else 1)
    # 25 shards per GiB of heap per data node: r6g.large has 16 GiB, 8 of
    # them heap.
    shards = alarms["OpenSearchShardsHigh"]
    assert shards["MetricName"] == "Shards.active"
    assert shards["Threshold"] == 25 * 8 * count
    share = alarms["OpenSearch5xxShare"]
    (expression,) = [m for m in share["Metrics"] if "Expression" in m]
    assert expression["Expression"] == "100 * FILL(errors, 0) / requests"
    assert _metric_names(share) == {"5xx", "OpenSearchRequests"}
    assert share["Threshold"] == 10


@pytest.mark.parametrize(
    ("instance", "heap_gib"),
    [
        ("r6g.large.search", 8),
        ("m6g.large.search", 4),
        ("c6g.xlarge.search", 4),
        ("r6g.4xlarge.search", 32),
        ("r6g.16xlarge.search", 32),  # the heap stops at 32 GiB
        ("or1.2xlarge.search", 32),
        ("t3.medium.search", 2),
        ("i3.large.search", None),
    ],
)
def test_opensearch_heap_follows_the_instance(
    instance: str, heap_gib: int | None
) -> None:
    from iac.stacks.observability_stack import opensearch_heap_bytes

    heap = opensearch_heap_bytes(instance)
    assert heap == (heap_gib * 1024**3 if heap_gib else None)


def test_an_unknown_instance_has_no_shard_alarm_and_says_so() -> None:
    stack = _stack({"opensearch_instance": "m4.large.search"})
    assert "OpenSearchShardsHigh" not in {
        logical_id[:-8]
        for logical_id in Template.from_stack(stack).find_resources(
            "AWS::CloudWatch::Alarm"
        )
    }
    Annotations.from_stack(stack).has_warning(
        "*", Match.string_like_regexp("no OpenSearchShardsHigh alarm")
    )


def test_yellow_alarm_on_multi_node_domains_only() -> None:
    alarm = _alarms({"opensearch_count": 2})["OpenSearchClusterYellow"]
    assert alarm["MetricName"] == "ClusterStatus.yellow"
    assert (alarm["Period"], alarm["EvaluationPeriods"]) == (60, 5)


def test_kms_key_alarms_with_a_customer_managed_key() -> None:
    alarms = _alarms({"use_cmk": True})
    assert alarms["OpenSearchKmsKeyError"]["MetricName"] == "KMSKeyError"
    inaccessible = alarms["OpenSearchKmsKeyInaccessible"]
    assert inaccessible["MetricName"] == "KMSKeyInaccessible"
    assert (inaccessible["Period"], inaccessible["Threshold"]) == (60, 1)


def test_ingestion_task_ephemeral_storage_alarm() -> None:
    alarm = _alarms()["IngestionTaskEphemeralStorage"]
    stats = {
        m["MetricStat"]["Metric"]["MetricName"]: m["MetricStat"]["Metric"]
        for m in alarm["Metrics"]
        if "MetricStat" in m
    }
    assert set(stats) == {"EphemeralStorageUtilized", "EphemeralStorageReserved"}
    for metric in stats.values():
        assert metric["Namespace"] == "ECS/ContainerInsights"
        dims = {d["Name"] for d in metric["Dimensions"]}
        assert dims == {"ClusterName", "TaskDefinitionFamily"}
    assert (alarm["Threshold"], alarm["EvaluationPeriods"]) == (90, 5)


def test_query_latency_alarm_reads_the_run_rag_metric() -> None:
    # The names the alarm reads are the ones the chain emits (run-rag
    # --metrics-sink cloudwatch hands it the sink).
    source = (_PACKAGE / "shared" / "metrics.py").read_text(encoding="utf-8")
    assert f'QUERY_METRICS_NAMESPACE = "{QUERY_NAMESPACE}"' in source
    assert f'QUERY_LATENCY_METRIC = "{QUERY_LATENCY_METRIC}"' in source
    for env, seconds in (("dev", 60), ("prod", 30)):
        context: dict[str, Any] = {"env_name": env}
        if seconds != 60:
            context["query_latency_alarm_seconds"] = seconds
        alarm = _alarms(context)["QueryLatencyHigh"]
        assert alarm["Namespace"] == QUERY_NAMESPACE
        assert alarm["MetricName"] == QUERY_LATENCY_METRIC
        assert alarm["ExtendedStatistic"] == "p90"
        assert alarm["Dimensions"] == [{"Name": "Environment", "Value": env}]
        assert alarm["Threshold"] == seconds * 1000
        assert (alarm["EvaluationPeriods"], alarm["DatapointsToAlarm"]) == (3, 2)
        assert alarm["TreatMissingData"] == "notBreaching"


def test_each_widget_holds_at_most_one_metrics_insights_query() -> None:
    # GetMetricWidgetImage renders a widget only with at most one Metrics
    # Insights query ("Maximum number of queries (1) exceeded").
    (dashboard,) = _template().find_resources("AWS::CloudWatch::Dashboard").values()
    _, parts = dashboard["Properties"]["DashboardBody"]["Fn::Join"]
    # Tokens (names, ARNs) are joined into JSON string values.
    body = json.loads("".join(p if isinstance(p, str) else "TOKEN" for p in parts))
    queries = {
        widget["properties"].get("title", ""): [
            item
            for metric in widget["properties"].get("metrics", [])
            for item in metric
            if isinstance(item, dict)
            and str(item.get("expression", "")).lstrip().upper().startswith("SELECT")
        ]
        for widget in body["widgets"]
    }
    assert any(queries.values())
    assert {title: len(q) for title, q in queries.items() if len(q) > 1} == {}


def test_the_runbook_lists_the_incremental_health_widget_exactly() -> None:
    # The runbook names the incremental degradations the dashboard plots (and
    # no alarm watches); each one must be a series of that widget.
    runbook = (_PACKAGE.parent / "docs" / "operations.md").read_text(encoding="utf-8")
    (listed,) = re.findall(
        r"the incremental degradations no alarm watches\s*\(([^)]*)\)", runbook
    )
    documented = set(re.findall(r"`(\w+)`", listed))
    (dashboard,) = _template().find_resources("AWS::CloudWatch::Dashboard").values()
    _, parts = dashboard["Properties"]["DashboardBody"]["Fn::Join"]
    body = json.loads("".join(p if isinstance(p, str) else "TOKEN" for p in parts))
    (widget,) = [
        w
        for w in body["widgets"]
        if w["properties"].get("title") == "Incremental run health (EMF)"
    ]
    plotted = {
        item
        for metric in widget["properties"]["metrics"]
        for item in metric[:2]
        if item != EMF_NAMESPACE
    }
    assert documented and plotted == documented


def test_the_dashboard_plots_a_partial_summary_tree() -> None:
    # Old behaviour: a tree whose summaries failed (or an on_reset tree that
    # covers less and less) degraded tree retrieval with no metric to show it.
    (dashboard,) = _template().find_resources("AWS::CloudWatch::Dashboard").values()
    _, parts = dashboard["Properties"]["DashboardBody"]["Fn::Join"]
    body = json.loads("".join(p if isinstance(p, str) else "TOKEN" for p in parts))
    (widget,) = [
        w
        for w in body["widgets"]
        if w["properties"].get("title") == "Summary tree (EMF)"
    ]
    plotted = {
        metric[1]
        for metric in widget["properties"]["metrics"]
        if metric[0] == EMF_NAMESPACE
    }
    assert plotted == set(SUMMARY_TREE_METRICS)
    assert plotted <= _pipeline_metric_fields()
    stages = _metric_stages()
    for name in SUMMARY_TREE_METRICS:
        # Emitted as the summary_tree stage finishes.
        assert stages[name] == {"summary_tree"}


def _widget_regions(template: Template) -> dict[str, str]:
    (dashboard,) = template.find_resources("AWS::CloudWatch::Dashboard").values()
    _, parts = dashboard["Properties"]["DashboardBody"]["Fn::Join"]
    body = json.loads("".join(p if isinstance(p, str) else "TOKEN" for p in parts))
    return {
        w["properties"]["title"]: w["properties"].get("region")
        for w in body["widgets"]
        # Metric widgets: the alarm status widget lists alarms, of no region.
        if "title" in w["properties"] and w["type"] == "metric"
    }


@pytest.mark.parametrize(
    ("context", "bedrock"),
    [
        ({}, "TOKEN"),  # the stack's own region (AWS::Region)
        ({"bedrock_region": "us-west-2"}, "us-west-2"),
        ({"bedrock_region": "us-east-1", "network_mode": "public"}, "us-east-1"),
    ],
    ids=["unset", "deploy-region", "other-region"],
)
def test_the_bedrock_widgets_read_the_bedrock_region(
    context: dict[str, Any], bedrock: str
) -> None:
    # Bedrock publishes its metrics where the calls go; the widgets watched
    # the deploy region, where a cross-region deployment's calls publish none.
    stack = _stack(context)
    regions = _widget_regions(Template.from_stack(stack))
    bedrock_widgets = {t: r for t, r in regions.items() if t.startswith("Bedrock")}
    assert len(bedrock_widgets) == 3
    assert set(bedrock_widgets.values()) == {bedrock}
    assert {r for t, r in regions.items() if t not in bedrock_widgets} <= {
        "us-west-2",
        "TOKEN",
    }
    # The throttling alarm reads the app's own metrics, published in the
    # deploy region, so another bedrock_region needs no warning.
    # has_no_warning asserts on the jsii side: find_warning returns
    # MetadataEntry objects, which a process that has not imported
    # aws_cdk.cloud_assembly_schema cannot deserialize.
    Annotations.from_stack(stack).has_no_warning(
        "*", Match.string_like_regexp("BedrockThrottling")
    )


def _dashboard_body(template: Template) -> dict[str, Any]:
    (dashboard,) = template.find_resources("AWS::CloudWatch::Dashboard").values()
    _, parts = dashboard["Properties"]["DashboardBody"]["Fn::Join"]
    return json.loads("".join(p if isinstance(p, str) else "TOKEN" for p in parts))


def test_every_alarm_notifies_when_it_fires_and_when_it_recovers() -> None:
    # Old behaviour: no OK action, so the subscriber never heard that an
    # alarm was over.
    for context in ({}, {"opensearch_count": 2, "neptune_instances": 2}):
        alarms = _alarms(context)
        assert alarms
        for name, alarm in alarms.items():
            assert len(alarm["AlarmActions"]) == 1, name
            assert alarm["OKActions"] == alarm["AlarmActions"], name


@pytest.mark.parametrize("readers", [False, True])
def test_the_dashboard_shows_every_store_and_the_alarm_status(readers: bool) -> None:
    # Old behaviour: no AWS/ES, AWS/Neptune or AWS/DynamoDB series and no
    # alarm widget, so store signals and app metrics could not be read side
    # by side.
    context = {"neptune_instances": 2} if readers else {}
    template = _template(context)
    body = _dashboard_body(template)
    namespaces: dict[str, set[str]] = {}
    neptune_roles: set[str] = set()
    for widget in body["widgets"]:
        for metric in widget["properties"].get("metrics", []):
            if isinstance(metric[0], str) and metric[0].startswith("AWS/"):
                namespaces.setdefault(metric[0], set()).add(
                    widget["properties"]["title"]
                )
                if metric[0] == "AWS/Neptune" and "Role" in metric:
                    neptune_roles.add(metric[metric.index("Role") + 1])
    for namespace in ("AWS/ES", "AWS/Neptune", "AWS/DynamoDB"):
        assert namespaces.get(namespace), namespace
    assert neptune_roles == ({"WRITER", "READER"} if readers else {"WRITER"})
    (status,) = [w for w in body["widgets"] if w["type"] == "alarm"]
    # Every alarm of the stack is listed.
    assert len(status["properties"]["alarms"]) == len(_alarms(context))


def test_neptune_reader_alarms_with_a_reader_only() -> None:
    # Old behaviour: the reader serves every query in non-dev but had no CPU
    # alarm, and replica lag was not watched.
    alarms = _alarms({"neptune_instances": 2})
    cpu = alarms["NeptuneReaderHighCpu"]
    dims = {d["Name"]: d["Value"] for d in cpu["Dimensions"]}
    assert dims["Role"] == "READER"
    assert (cpu["MetricName"], cpu["Threshold"], cpu["EvaluationPeriods"]) == (
        "CPUUtilization",
        90,
        3,
    )
    lag = alarms["NeptuneReplicaLag"]
    assert (lag["MetricName"], lag["Statistic"]) == (
        "ClusterReplicaLagMaximum",
        "Maximum",
    )
    assert lag["Threshold"] == 1000
    assert not {"NeptuneReaderHighCpu", "NeptuneReplicaLag"} & _alarms().keys()


def test_doc_status_read_throttles_and_system_errors_alarm() -> None:
    from iac.stacks.observability_stack import DOC_STATUS_OPERATIONS

    alarms = _alarms()
    read = alarms["DocStatusReadThrottled"]
    assert (read["MetricName"], read["Statistic"], read["Threshold"]) == (
        "ReadThrottleEvents",
        "Sum",
        1,
    )
    errors = alarms["DocStatusSystemErrors"]
    operations = {
        {d["Name"]: d["Value"] for d in m["MetricStat"]["Metric"]["Dimensions"]}[
            "Operation"
        ]
        for m in errors["Metrics"]
        if "MetricStat" in m
    }
    # SystemErrors exists per operation only: one series per operation the
    # registry calls (at most 10 metrics fit one alarm).
    assert operations == set(DOC_STATUS_OPERATIONS)
    assert len(operations) <= 10
    source = (_PACKAGE / "adapters" / "doc_status" / "dynamodb.py").read_text(
        encoding="utf-8"
    )
    names = re.findall(r"client\.(\w+)\(", source)
    names += re.findall(r"get_paginator\(\"(\w+)\"\)", source)
    called = {"".join(part.title() for part in name.split("_")) for name in names}
    # Set-up and helper calls, not registry requests.
    called -= {
        "CreateTable",
        "DescribeTable",
        "GetPaginator",
        "GetWaiter",
        "UpdateTimeToLive",
    }
    assert called == set(DOC_STATUS_OPERATIONS)


def test_failed_queries_alarm_on_their_share() -> None:
    # Old behaviour: only latency alarmed, and a query that fails fast lowers
    # the p90 instead of raising it.
    from iac.stacks.observability_stack import (
        QUERY_FAILED_METRIC,
        QUERY_OUTCOME_METRICS,
    )

    for env in ("dev", "prod"):
        alarm = _alarms({"env_name": env})["QueryFailures"]
        (expression,) = [m for m in alarm["Metrics"] if "Expression" in m]
        assert expression["Expression"] == "100 * FILL(failed, 0) / queries"
        stats = {
            m["Id"]: m["MetricStat"] for m in alarm["Metrics"] if "MetricStat" in m
        }
        assert stats["failed"]["Metric"]["MetricName"] == QUERY_FAILED_METRIC
        assert stats["failed"]["Stat"] == "Sum"
        assert stats["queries"]["Metric"]["MetricName"] == QUERY_LATENCY_METRIC
        assert stats["queries"]["Stat"] == "SampleCount"
        for stat in stats.values():
            assert stat["Metric"]["Namespace"] == QUERY_NAMESPACE
            assert stat["Metric"]["Dimensions"] == [
                {"Name": "Environment", "Value": env}
            ]
        assert alarm["Threshold"] == 10
        assert (alarm["EvaluationPeriods"], alarm["DatapointsToAlarm"]) == (3, 2)
        assert alarm["TreatMissingData"] == "notBreaching"
    body = json.dumps(_dashboard_body(_template()))
    for name in QUERY_OUTCOME_METRICS:
        assert f'"{QUERY_NAMESPACE}", "{name}"' in body


def test_the_query_outcome_metrics_are_emitted() -> None:
    # The alarm and widget read names the package must emit: a drifted name
    # would leave the alarm on a series nobody publishes.
    from iac.stacks.observability_stack import QUERY_OUTCOME_METRICS

    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(_PACKAGE.rglob("*.py"))
    )
    for name in QUERY_OUTCOME_METRICS:
        assert f'"{name}"' in source, name


def test_the_query_task_has_its_memory_alarm_and_widget() -> None:
    # Old behaviour: only the ingestion task's family was watched; a run-eval
    # task killed for memory went unnoticed.
    stack = _stack()
    template = Template.from_stack(stack)
    alarms = {
        logical_id[:-8]: a["Properties"]
        for logical_id, a in template.find_resources("AWS::CloudWatch::Alarm").items()
    }
    families = {}
    for name in ("IngestionTaskMemory", "QueryTaskMemory"):
        metrics = [
            m["MetricStat"]["Metric"]
            for m in alarms[name]["Metrics"]
            if "MetricStat" in m
        ]
        assert {m["MetricName"] for m in metrics} == {
            "MemoryUtilized",
            "MemoryReserved",
        }
        families[name] = {
            json.dumps(d["Value"])
            for m in metrics
            for d in m["Dimensions"]
            if d["Name"] == "TaskDefinitionFamily"
        }
        assert (alarms[name]["Threshold"], alarms[name]["EvaluationPeriods"]) == (90, 3)
    assert len(families["QueryTaskMemory"]) == 1
    assert families["QueryTaskMemory"] != families["IngestionTaskMemory"]
    titles = {
        w["properties"].get("title") for w in _dashboard_body(template)["widgets"]
    }
    assert "Query task memory and CPU (% of reserved)" in titles
