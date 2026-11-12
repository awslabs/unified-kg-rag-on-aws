# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observability: CloudWatch dashboard + alarms over the pipeline.

The application emits Embedded Metric Format (EMF) records under the
``unified_kg_rag/ingestion`` namespace when run with ``--metrics-sink cloudwatch``
(see unified_kg_rag/shared/metrics.py), each stage's as the stage finishes. The
task gets ``UKR_METRICS_ENVIRONMENT=<env_name>``, so the series carry an
``Environment`` dimension and the alarms and widgets here query only this
deployment's runs: a dev run in the same account and region does not trip prod's
alarms. This stack surfaces those plus Step Functions execution health, Bedrock
throttling and latency and the ingestion task's memory on a dashboard, and alarms
on failed, timed-out or aborted runs, dropped artifacts, sustained Bedrock
throttling, ingestion task memory pressure and store health -> the orchestration
SNS topic.
"""

from __future__ import annotations

from aws_cdk import Annotations, Duration, Stack, Token
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from constructs import Construct

from iac.config import DeploymentConfig
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.storage_stack import OPENSEARCH_MASTER_NODES, StorageStack

# Memory (GiB) per vCPU of the Neptune and OpenSearch instance families: the
# memory-optimized r families (and OpenSearch's or1/or2), x2g, the general
# purpose m and compute-optimized c families; a burstable t3/t4g.medium has
# 4 GiB.
_GIB_PER_VCPU = {
    **dict.fromkeys(
        ("r5", "r5d", "r6g", "r6gd", "r6i", "r7g", "r7gd", "r7i", "r8g", "or1", "or2"),
        8,
    ),
    **dict.fromkeys(("m5", "m6g", "m7g", "m7i", "m8g"), 4),
    **dict.fromkeys(("c5", "c6g", "c7g", "c7i", "c8g"), 2),
    "x2g": 16,
}
# Low-memory threshold: this share of the instance's memory (512 MiB of the
# default db.r6g.large's 16 GiB), and 512 MiB for a class not listed.
_LOW_MEMORY_SHARE = 32
_LOW_MEMORY_DEFAULT = 512 * 1024**2


def _instance_memory_bytes(family: str, size: str) -> int | None:
    """The memory of an instance of ``family`` and ``size`` (``r6g``,
    ``large``), or ``None`` for one this table does not know."""
    if family in ("t3", "t4g"):
        return 4 * 1024**3 if size == "medium" else None
    if family not in _GIB_PER_VCPU:
        return None
    if size == "large":
        vcpus = 2
    elif size == "xlarge":
        vcpus = 4
    elif size.endswith("xlarge") and size[: -len("xlarge")].isdigit():
        vcpus = 4 * int(size[: -len("xlarge")])
    else:
        return None
    return _GIB_PER_VCPU[family] * vcpus * 1024**3


def neptune_instance_memory_bytes(instance_class: str) -> int | None:
    """The memory of Neptune ``instance_class`` (``db.r6g.large``), or
    ``None`` for a class this table does not know."""
    parts = instance_class.split(".")
    if len(parts) != 3 or parts[0] != "db":
        return None
    return _instance_memory_bytes(parts[1], parts[2])


# OpenSearch gives the Java heap half of a node's memory, up to 32 GiB, and
# AWS recommends at most 25 shards per GiB of heap ("Recommended CloudWatch
# alarms for Amazon OpenSearch Service", shards.active).
_HEAP_SHARE = 2
_MAX_HEAP = 32 * 1024**3
SHARDS_PER_HEAP_GIB = 25


def opensearch_heap_bytes(instance_type: str) -> int | None:
    """The Java heap of an OpenSearch ``instance_type`` node
    (``r6g.large.search``), or ``None`` for a type this table does not
    know."""
    parts = instance_type.split(".")
    if len(parts) != 3 or parts[2] != "search":
        return None
    memory = _instance_memory_bytes(parts[0], parts[1])
    return min(memory // _HEAP_SHARE, _MAX_HEAP) if memory else None


EMF_NAMESPACE = "unified_kg_rag/ingestion"

# run-rag --metrics-sink cloudwatch: one query_latency_ms value per query,
# per Environment and per Environment and Strategy (the search strategy).
QUERY_NAMESPACE = "unified_kg_rag/query"
QUERY_LATENCY_METRIC = "query_latency_ms"

# Document- and file-level outcomes of an ingestion run, as run-ingestion
# emits them (one value per run, read as Sum; the file counts after
# document_parsing / document_loading, the deletions after indexing, the
# document counts at run end). The one place these names live: the
# DocumentsNotIndexed alarm and the "Document outcomes" widget read them.
FILES_FAILED = "total_files_failed"  # a source file that failed to parse
FILES_SKIPPED = "total_files_skipped"  # a source file no loader reads
DOCUMENTS_EXCLUDED = "total_documents_excluded"  # filtered out by configuration
DOCUMENTS_DELETED = "total_documents_deleted"  # removed from the stores
DOCUMENTS_FAILED = "total_documents_failed"  # recorded FAILED in the registry
DOCUMENTS_RETRYING = "total_documents_retrying"  # FAILED before, processed again
DOCUMENTS_WITHOUT_CLAIMS = "total_documents_without_claims"
DOCUMENT_METRICS: tuple[str, ...] = (
    FILES_FAILED,
    FILES_SKIPPED,
    DOCUMENTS_EXCLUDED,
    DOCUMENTS_DELETED,
    DOCUMENTS_FAILED,
    DOCUMENTS_RETRYING,
    DOCUMENTS_WITHOUT_CLAIMS,
)
# The outcomes that leave a source document out of the stores although the
# run SUCCEEDED: the DocumentsNotIndexed alarm sums them. Skipped files are
# left out: a corpus that holds files no loader reads (images, archives) would
# alarm on every run, so their count is on the dashboard only.
NOT_INDEXED_METRICS: tuple[str, ...] = (FILES_FAILED, DOCUMENTS_FAILED)
# The summary_tree stage's degradations (PipelineMetrics fields), one value
# per run that builds a RAPTOR tree.
SUMMARY_TREE_METRICS: tuple[str, ...] = (
    "total_tree_summaries_failed",
    "total_tree_uncovered_leaves",
)

# The pipeline stages, in order: each emits duration_<stage>_seconds when it
# finishes (pinned against PipelineStageType by the tests).
PIPELINE_STAGES: tuple[str, ...] = (
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
    "summary_tree",
)

# The stages that may call a model: each emits llm_calls_<stage>,
# llm_input_tokens_<stage> and llm_output_tokens_<stage> when it did (the
# app's usage accounting, tracing or not).
LLM_STAGES: tuple[str, ...] = tuple(
    stage
    for stage in PIPELINE_STAGES
    if stage not in ("document_parsing", "document_loading")
)

# Bedrock publishes its runtime metrics per ModelId only, and the model ids
# the app resolves (inference profiles included) are app config, not known
# here: these Metrics Insights queries sum over every model in the account and
# region, so other Bedrock users there count too.
BEDROCK_THROTTLES = (
    'SELECT SUM(InvocationThrottles) FROM SCHEMA("AWS/Bedrock", ModelId)'
)

# OpenSearch 7.1+ indexing-pressure rejections ("Monitoring OpenSearch
# cluster metrics with Amazon CloudWatch": totals since the node's process
# started), where a 2.x domain's bulk 429s are counted.
OPENSEARCH_INDEXING_PRESSURE_REJECTIONS: tuple[str, ...] = (
    "CoordinatingWriteRejected",
    "PrimaryWriteRejected",
    "ReplicaWriteRejected",
)


class ObservabilityStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: DeploymentConfig,
        orchestration: OrchestrationStack,
        storage: StorageStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        self.storage = storage
        self.compute = orchestration.compute
        self.ingestion_task_definition = orchestration.task_definition
        self._alarm_action = cw_actions.SnsAction(orchestration.alarm_topic)
        sm = orchestration.state_machine

        # Alarm: any failed Step Functions execution -> SNS. A failed task is
        # also published, with its error, by the state machine's Catch, so such
        # a failure notifies twice (once with context, once without). The alarm
        # stays because the Catch cannot see every failure: States.ALL does not
        # catch States.DataLimitExceeded or States.Runtime (e.g. a start input
        # without the fields the task reads), and the publish itself can fail.
        failed_alarm = cw.Alarm(
            self,
            "PipelineFailures",
            metric=sm.metric_failed(),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            # No execution in a period publishes no ExecutionsFailed datum:
            # that is no failure, not an unknown state.
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="unified-kg-rag-on-aws ingestion pipeline had a failed execution",
        )
        failed_alarm.add_alarm_action(cw_actions.SnsAction(orchestration.alarm_topic))

        # Alarm: a run that hit the state machine timeout or was stopped never
        # reaches the task's Catch and does not count as failed, so neither the
        # failure notification nor the alarm above sees it. FILL keeps a
        # missing series from blanking the sum.
        stopped_alarm = cw.Alarm(
            self,
            "PipelineTimedOutOrAborted",
            metric=cw.MathExpression(
                expression="FILL(t, 0) + FILL(a, 0)",
                using_metrics={"t": sm.metric_timed_out(), "a": sm.metric_aborted()},
                label="Timed-out or aborted executions",
                period=Duration.minutes(5),
            ),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="unified-kg-rag-on-aws ingestion pipeline had a "
            "timed-out or aborted execution",
        )
        stopped_alarm.add_alarm_action(cw_actions.SnsAction(orchestration.alarm_topic))

        # Alarm: any indexing failures -> SNS. A SUCCEEDED Step Functions run can
        # still have silently dropped artifacts (extracted > 0 but indexed == 0,
        # e.g. a backend write that erred per-item while the run continued). The
        # failure notification does not catch that; this EMF metric does.
        index_failed_alarm = cw.Alarm(
            self,
            "IndexingFailures",
            metric=self._emf("total_items_index_failed", period=Duration.minutes(5)),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            # Don't page when the metric simply isn't reported (no run in window).
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="unified-kg-rag-on-aws indexing reported failed items "
            "(possible silent artifact drop despite a SUCCEEDED run)",
        )
        index_failed_alarm.add_alarm_action(
            cw_actions.SnsAction(orchestration.alarm_topic)
        )

        # Alarm: a document quarantined -> SNS. Its processing killed the task
        # indexing.max_document_attempts runs in a row, the last one alone; it
        # is no longer indexed until someone looks.
        quarantined_alarm = cw.Alarm(
            self,
            "DocumentsQuarantined",
            metric=self._emf("total_documents_quarantined", period=Duration.minutes(5)),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="unified-kg-rag-on-aws quarantined documents whose "
            "processing kept killing the run (skipped until their content "
            "changes)",
        )
        quarantined_alarm.add_alarm_action(
            cw_actions.SnsAction(orchestration.alarm_topic)
        )

        # Alarm: per-chunk extraction failures -> SNS. A chunk whose LLM call
        # fails every retry is dropped and the run still SUCCEEDS with a partial
        # graph; nothing above sees that. Sums the three EMF failure
        # counters (FILL keeps a missing metric from blanking the sum).
        extraction_failed_alarm = cw.Alarm(
            self,
            "ExtractionFailures",
            metric=cw.MathExpression(
                expression="FILL(e, 0) + FILL(g, 0) + FILL(c, 0)",
                using_metrics={
                    key: self._emf(name, period=Duration.minutes(5))
                    for key, name in (
                        ("e", "total_extraction_failures"),
                        ("g", "total_gleaning_failures"),
                        ("c", "total_claim_extraction_failures"),
                    )
                },
                label="Failed extraction units",
                period=Duration.minutes(5),
            ),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="unified-kg-rag-on-aws extraction/gleaning/claim "
            "extraction dropped text units (partial graph despite a SUCCEEDED run)",
        )
        extraction_failed_alarm.add_alarm_action(
            cw_actions.SnsAction(orchestration.alarm_topic)
        )

        # Alarm: source documents the run left out of the stores -> SNS. A
        # file that failed to parse and a document recorded FAILED are
        # missing from the graph of a SUCCEEDED run.
        not_indexed = {
            f"m{i}": self._emf(name, period=Duration.minutes(5))
            for i, name in enumerate(NOT_INDEXED_METRICS)
        }
        not_indexed_alarm = cw.Alarm(
            self,
            "DocumentsNotIndexed",
            metric=cw.MathExpression(
                expression=" + ".join(f"FILL({key}, 0)" for key in not_indexed),
                using_metrics=not_indexed,
                label="Documents not indexed",
                period=Duration.minutes(5),
            ),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="unified-kg-rag-on-aws left source documents out of "
            "the stores (parse failures, failed documents) despite a SUCCEEDED "
            "run",
        )
        not_indexed_alarm.add_alarm_action(
            cw_actions.SnsAction(orchestration.alarm_topic)
        )

        # Store-health alarms: the run alarms above only fire when a pipeline
        # RUN goes wrong. These surface data-store degradation (red cluster,
        # disk pressure, memory pressure, DDB throttling) that would otherwise be
        # invisible until the next run breaks. All route to the same SNS topic.
        self._add_store_health_alarms()
        self._add_capacity_alarms(orchestration)
        self._add_query_alarms(orchestration)

        # All alarms route to the orchestration SNS topic, but with no subscriber
        # they fire into the void. Warn at synth (rather than silently) when no
        # email is wired so the operator knows to add one (`-c alarm_email=...`).
        if not config.alarm_email:
            Annotations.of(self).add_warning(
                "No alarm_email configured: pipeline-failure and store-health "
                "alarms publish to SNS with no subscriber. Pass "
                "-c alarm_email=<addr> to receive them."
            )

        # Dashboard: execution health + a couple of EMF pipeline metrics.
        dashboard = cw.Dashboard(
            self, "Dashboard", dashboard_name=f"{config.prefix}-dashboard"
        )
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Pipeline executions",
                left=[
                    sm.metric_succeeded(),
                    sm.metric_failed(),
                    sm.metric_timed_out(),
                    sm.metric_aborted(),
                ],
            ),
            cw.GraphWidget(
                title="Pipeline duration",
                left=[sm.metric_time()],
            ),
        )
        # Metric names must match what pipeline._emit_metrics actually emits
        # (the PipelineMetrics scalar field names), or the widgets render empty.
        extracted_metrics = [
            self._emf(name)
            for name in (
                "total_entities_extracted",
                "total_relationships_extracted",
                "total_claims_extracted",
                "total_extraction_failures",
            )
        ]
        indexed_metrics = [
            self._emf(name)
            for name in (
                "total_items_indexed",
                "relationships_indexed",
                "total_items_index_failed",
            )
        ]
        dashboard.add_widgets(
            cw.GraphWidget(title="Extracted artifacts (EMF)", left=extracted_metrics),
            cw.GraphWidget(title="Indexed artifacts (EMF)", left=indexed_metrics),
        )
        # Degradations of an incremental run that succeeded: documents left
        # for a later run, a recommended community rebuild, extraction records
        # not stored and stored items whose derived attributes are stale. On
        # the right, the rebuild signals that grow run by run (a ratio and a
        # run count, one value per run: their Maximum, not a Sum).
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Incremental run health (EMF)",
                left=[
                    self._emf(name)
                    for name in (
                        "total_documents_deferred",
                        "community_rebuild_recommended",
                        "total_records_not_stored",
                        "underived_items",
                    )
                ],
                right=[
                    self._emf(name, statistic="Maximum")
                    for name in ("community_growth_ratio", "runs_since_clustering")
                ],
            ),
            cw.GraphWidget(
                title="Document outcomes (EMF)",
                left=[self._emf(name) for name in DOCUMENT_METRICS],
            ),
        )
        # A partial RAPTOR summary tree (summaries that failed: their
        # children moved up a level) and, with refresh: on_reset, the text
        # units no summary covers: tree retrieval degrades, the run succeeds.
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Summary tree (EMF)",
                left=[self._emf(name) for name in SUMMARY_TREE_METRICS],
            ),
        )
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Stage durations (EMF, seconds)",
                left=[
                    self._emf(f"duration_{stage}_seconds", statistic="Maximum")
                    for stage in PIPELINE_STAGES
                ],
                stacked=True,
            ),
            cw.GraphWidget(
                title="Ingestion task memory, CPU and disk (% of reserved)",
                left=[
                    self._task_utilization("Memory"),
                    self._task_utilization("Cpu"),
                    self._task_utilization("EphemeralStorage"),
                ],
            ),
        )
        # Every Strategy series of this deployment's query latency.
        query_series = (
            f"{{{QUERY_NAMESPACE},Environment,Strategy}} "
            f'MetricName="{QUERY_LATENCY_METRIC}" Environment="{config.env_name}"'
        )
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Query latency by strategy (ms, p90)",
                left=[
                    cw.MathExpression(
                        expression=f"SEARCH('{query_series}', 'p90', 300)",
                        label="",
                        period=Duration.minutes(5),
                    )
                ],
            ),
            cw.GraphWidget(
                title="Query latency (ms)",
                left=[
                    self._query_latency(statistic)
                    for statistic in ("p50", "p90", "p99")
                ],
            ),
        )
        # This deployment's own model usage, per stage (the Bedrock widgets
        # below count every caller in the account and region).
        dashboard.add_widgets(
            cw.GraphWidget(
                title="LLM tokens by stage (EMF)",
                left=[
                    self._emf(f"llm_{kind}_tokens_{stage}")
                    for kind in ("input", "output")
                    for stage in LLM_STAGES
                ],
                stacked=True,
            ),
            cw.GraphWidget(
                title="LLM calls by stage and response-cache hits (EMF)",
                left=[self._emf(f"llm_calls_{stage}") for stage in LLM_STAGES],
                right=[self._emf("llm_cache_hits")],
            ),
        )
        # One Metrics Insights query per widget: GetMetricWidgetImage (the
        # API that renders a widget to a PNG, e.g. for a report) refuses a
        # widget holding two ("Maximum number of queries (1) exceeded").
        # Bedrock publishes its metrics in the region the calls go to
        # (bedrock_region), which a dashboard widget can name.
        bedrock_region = config.bedrock_region
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Bedrock invocations by model",
                region=bedrock_region,
                left=[
                    self._insights(
                        'SELECT SUM(Invocations) FROM SCHEMA("AWS/Bedrock", ModelId)'
                        " GROUP BY ModelId",
                        "Invocations",
                    )
                ],
            ),
            cw.GraphWidget(
                title="Bedrock throttles (all models)",
                region=bedrock_region,
                left=[self._insights(BEDROCK_THROTTLES, "Throttles")],
            ),
            cw.GraphWidget(
                title="Bedrock invocation latency by model (ms, average)",
                region=bedrock_region,
                left=[
                    self._insights(
                        'SELECT AVG(InvocationLatency) FROM SCHEMA("AWS/Bedrock",'
                        " ModelId) GROUP BY ModelId",
                        "Latency",
                    )
                ],
            ),
        )
        self.dashboard = dashboard

    def _emf(
        self,
        name: str,
        *,
        statistic: str = "Sum",
        period: Duration | None = None,
    ) -> cw.Metric:
        """An EMF metric of this deployment's runs (its Environment series)."""
        return cw.Metric(
            namespace=EMF_NAMESPACE,
            metric_name=name,
            statistic=statistic,
            dimensions_map={"Environment": self.config.env_name},
            period=period or Duration.minutes(5),
        )

    def _query_latency(self, statistic: str) -> cw.Metric:
        """Latency of this deployment's run-rag queries, all strategies."""
        return cw.Metric(
            namespace=QUERY_NAMESPACE,
            metric_name=QUERY_LATENCY_METRIC,
            statistic=statistic,
            dimensions_map={"Environment": self.config.env_name},
            period=Duration.minutes(5),
        )

    @property
    def bedrock_region(self) -> str | None:
        """The region the model calls go to, when it is not this stack's
        (``None`` too when the deploy region is not known at synth)."""
        region = self.config.bedrock_region
        if not region or Token.is_unresolved(self.region) or region == self.region:
            return None
        return region

    @staticmethod
    def _insights(query: str, label: str) -> cw.MathExpression:
        """A Metrics Insights query (alarms and widgets take it as math)."""
        return cw.MathExpression(
            expression=query, label=label, period=Duration.minutes(5)
        )

    def _task_utilization(self, resource: str) -> cw.MathExpression:
        """Peak use of the ingestion task's CPU or memory, in % of its size.

        From Container Insights (the cluster has it on), per task definition
        family: the ingestion task's family, not the query task's.
        """
        dims = {
            "ClusterName": self.compute.cluster.cluster_name,
            "TaskDefinitionFamily": self.ingestion_task_definition.family,
        }

        def metric(name: str) -> cw.Metric:
            return cw.Metric(
                namespace="ECS/ContainerInsights",
                metric_name=name,
                dimensions_map=dims,
                statistic="Maximum",
                period=Duration.minutes(1),
            )

        # Ids unique per resource: one widget plots memory and CPU together.
        used, reserved = f"{resource.lower()}_used", f"{resource.lower()}_reserved"
        return cw.MathExpression(
            expression=f"100 * {used} / {reserved}",
            using_metrics={
                used: metric(f"{resource}Utilized"),
                reserved: metric(f"{resource}Reserved"),
            },
            label=f"{resource} %",
            period=Duration.minutes(1),
        )

    def _add_capacity_alarms(self, orchestration: OrchestrationStack) -> None:
        action = cw_actions.SnsAction(orchestration.alarm_topic)
        # Sustained throttling: the batch retries each throttled call, so a
        # burst is absorbed, but 15 minutes of it means the run's concurrency
        # (processing.max_concurrency) exceeds the account's Bedrock quota and
        # the run slows to the retry backoff.
        throttles = cw.Alarm(
            self,
            "BedrockThrottling",
            metric=self._insights(BEDROCK_THROTTLES, "Bedrock throttles"),
            threshold=100,
            evaluation_periods=3,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="Bedrock throttled at least 100 calls in each of "
            "3 consecutive 5-minute periods (all models in the account and "
            "region): lower processing.max_concurrency or raise the quota",
        )
        throttles.add_alarm_action(action)
        if self.bedrock_region:
            # An alarm reads metrics of its own region only, and Bedrock
            # publishes them where the calls go.
            Annotations.of(self).add_warning(
                f"bedrock_region ({self.bedrock_region}) is not the deploy "
                f"region ({self.region}): the BedrockThrottling alarm watches "
                "the deploy region, where this deployment's model calls publish "
                "no metrics, so it cannot fire. The dashboard's Bedrock widgets "
                f"read {self.bedrock_region}; create a throttling alarm there to "
                "be notified (docs/operations.md, Alarms)."
            )
        # Memory pressure: Fargate kills a task that exceeds its memory (exit
        # 137, OutOfMemoryError), which loses the running stage.
        memory = cw.Alarm(
            self,
            "IngestionTaskMemory",
            metric=self._task_utilization("Memory"),
            threshold=90,
            evaluation_periods=3,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="The ingestion task used at least 90% of its "
            "memory for 3 minutes: raise fargate_memory before it is killed",
        )
        memory.add_alarm_action(action)
        # Disk: the ingestion task's ephemeral storage holds the synced
        # corpus, the parsed documents and the stage cache; a full disk fails
        # the running stage. AWS's recommended Container Insights alarm: 90%
        # of the size in 5 consecutive minutes. The reserved size includes the
        # container image.
        disk = cw.Alarm(
            self,
            "IngestionTaskEphemeralStorage",
            metric=self._task_utilization("EphemeralStorage"),
            threshold=90,
            evaluation_periods=5,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="The ingestion task used at least 90% of its "
            "ephemeral storage for 5 minutes: raise "
            "ingestion_ephemeral_storage_gib before the disk fills",
        )
        disk.add_alarm_action(action)

    def _add_query_alarms(self, orchestration: OrchestrationStack) -> None:
        # Slow queries: the p90 latency of run-rag queries (--metrics-sink
        # cloudwatch) above query_latency_alarm_seconds in 2 of 3 5-minute
        # periods. Periods without a query are missing, not breaching.
        threshold_ms = self.config.query_latency_alarm_seconds * 1000
        latency = cw.Alarm(
            self,
            "QueryLatencyHigh",
            metric=self._query_latency("p90"),
            threshold=threshold_ms,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="p90 run-rag query latency above "
            f"{self.config.query_latency_alarm_seconds} s in 2 of 3 5-minute "
            "periods (all search strategies)",
        )
        latency.add_alarm_action(cw_actions.SnsAction(orchestration.alarm_topic))

    def _alarm(
        self,
        ident: str,
        metric: cw.IMetric,
        threshold: float,
        description: str,
        comparison: cw.ComparisonOperator = (
            cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD
        ),
        evaluation_periods: int = 1,
    ) -> cw.Alarm:
        """An alarm on ``metric`` that notifies the alarm topic; a period
        without data is not breaching."""
        alarm = cw.Alarm(
            self,
            ident,
            metric=metric,
            threshold=threshold,
            evaluation_periods=evaluation_periods,
            comparison_operator=comparison,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description=description,
        )
        alarm.add_alarm_action(self._alarm_action)
        return alarm

    def _add_store_health_alarms(self) -> None:
        domain = self.storage.opensearch_domain
        table = self.storage.doc_status_table_for(self)
        _alarm = self._alarm

        # OpenSearch: a red cluster means at least one primary shard is
        # unallocated (data unavailable); free storage / JVM pressure precede
        # write rejections.
        _alarm(
            "OpenSearchClusterRed",
            domain.metric_cluster_status_red(period=Duration.minutes(5)),
            1,
            "OpenSearch cluster status is RED (unallocated primary shard)",
        )
        # MiB on the fullest node; alarm at 40% free, well before the ~20%
        # block-write watermark, so there is time to react.
        free_storage_mib = self.config.opensearch_volume_gib * 1024 * 0.4
        _alarm(
            "OpenSearchFreeStorageLow",
            domain.metric_free_storage_space(period=Duration.minutes(5)),
            free_storage_mib,
            "OpenSearch free storage space is low "
            f"(<{free_storage_mib / 1024:g} GiB on a node)",
            comparison=cw.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
        )
        # AWS's recommended alarm ("Recommended CloudWatch alarms for Amazon
        # OpenSearch Service"): JVMMemoryPressure maximum >= 95% for 1 minute,
        # 3 consecutive times. Heap pressure swings with garbage collection
        # during bulk indexing, so one high 5-minute maximum is not a fault.
        _alarm(
            "OpenSearchJvmPressure",
            domain.metric_jvm_memory_pressure(
                period=Duration.minutes(1), statistic="Maximum"
            ),
            95,
            "OpenSearch JVM memory pressure is at least 95% for 3 consecutive "
            "minutes (out-of-memory risk)",
            evaluation_periods=3,
        )
        # More of AWS's recommended OpenSearch alarms ("Recommended
        # CloudWatch alarms for Amazon OpenSearch Service"; the others are in
        # _add_opensearch_capacity_alarms). A failed automated snapshot
        # leaves the hourly restore point missing (Backup and restore);
        # blocked writes fail every ingestion bulk.
        _alarm(
            "OpenSearchAutomatedSnapshotFailure",
            domain.metric_automated_snapshot_failure(
                period=Duration.minutes(1), statistic="Maximum"
            ),
            1,
            "OpenSearch automated snapshot failed (often a red cluster): the "
            "hourly restore point is missing",
        )
        _alarm(
            "OpenSearchIndexWritesBlocked",
            domain.metric_cluster_index_writes_blocked(
                period=Duration.minutes(5), statistic="Maximum"
            ),
            1,
            "OpenSearch is blocking index writes (ClusterBlockException): "
            "ingestion writes fail until storage or memory is freed",
        )
        if self.config.use_cmk:
            # Only a customer-managed key can be disabled or deleted; the
            # metrics appear only once the domain has a problem with it.
            _alarm(
                "OpenSearchKmsKeyError",
                domain.metric_kms_key_error(
                    period=Duration.minutes(1), statistic="Maximum"
                ),
                1,
                "The OpenSearch data key (use_cmk) is disabled: re-enable it",
            )
            _alarm(
                "OpenSearchKmsKeyInaccessible",
                domain.metric_kms_key_inaccessible(
                    period=Duration.minutes(1), statistic="Maximum"
                ),
                1,
                "The OpenSearch data key (use_cmk) was deleted or its grants "
                "revoked: the domain cannot be recovered, restore a manual "
                "snapshot to a new domain",
            )
        if self.config.opensearch_count > 1:
            # A single node has nowhere to place a replica (the task sets
            # none there), so only a multi-node domain alarms on yellow.
            _alarm(
                "OpenSearchClusterYellow",
                domain.metric_cluster_status_yellow(
                    period=Duration.minutes(1), statistic="Maximum"
                ),
                1,
                "OpenSearch cluster status is YELLOW (an unallocated replica "
                "shard) for 5 minutes: losing another node loses data",
                evaluation_periods=5,
            )

        # Rejection counters are cumulative (totals since the node's process
        # started): alarm on their increase (AWS's DIFF recommendation). On
        # OpenSearch 2.x most bulk 429s come from indexing pressure, counted
        # per coordinating node, primary and replica shard, not in the write
        # thread pool (whose queue holds 10,000 since 7.1): a write alarm
        # sums the three. The bulk writer retries a rejected write, so only
        # rejections in 3 consecutive minutes alarm; a rejected search fails
        # a query at once.
        def _rejections(names: tuple[str, ...], label: str) -> cw.MathExpression:
            using = {
                f"r{i}": domain.metric(
                    name, statistic="Sum", period=Duration.minutes(1)
                )
                for i, name in enumerate(names)
            }
            # DIFF per counter, then the sum: a gap in one series (FILL
            # inside DIFF would read it as a reset to 0, then a jump) or a
            # series a domain does not publish leaves the others' increase.
            return cw.MathExpression(
                expression=" + ".join(f"FILL(DIFF({key}), 0)" for key in using),
                using_metrics=using,
                label=label,
                period=Duration.minutes(1),
            )

        for ident, names, what, periods in (
            (
                "OpenSearchWriteRejections",
                ("ThreadpoolWriteRejected",),
                "write requests (thread pool queue full)",
                3,
            ),
            (
                "OpenSearchIndexingPressureRejections",
                OPENSEARCH_INDEXING_PRESSURE_REJECTIONS,
                "writes for indexing pressure",
                3,
            ),
            (
                "OpenSearchSearchRejections",
                ("ThreadpoolSearchRejected",),
                "search requests (thread pool queue full)",
                1,
            ),
        ):
            _alarm(
                ident,
                _rejections(names, f"New rejections ({', '.join(names)})"),
                1,
                f"OpenSearch rejected {what} in {periods} consecutive minute(s)",
                evaluation_periods=periods,
            )
        self._add_opensearch_capacity_alarms()
        # DynamoDB doc-status registry: throttled writes mean delta/lineage
        # updates are being rejected, which silently corrupts incremental-indexing
        # state. The registry writes mostly through BatchWriteItem, so alarm on
        # the table-level write-throttle count rather than a single operation.
        _alarm(
            "DocStatusWriteThrottled",
            table.metric(
                "WriteThrottleEvents", statistic="Sum", period=Duration.minutes(5)
            ),
            1,
            "DynamoDB doc-status table is throttling writes",
        )

        # Neptune: the primary graph store and the write target of every
        # ingestion run. Without these, Neptune degradation (CPU saturation
        # during large edge writes, memory pressure, throttled Gremlin) is
        # invisible until a pipeline run breaks. Metrics are read from the
        # AWS/Neptune namespace keyed by the cluster id (stable across the
        # neptune_alpha construct's API surface). CPU and memory are read for
        # the WRITER role only: the cluster-wide series averages in idle
        # readers (neptune_instances=2 halves a saturated writer's CPU), and
        # every ingestion write lands on the writer.
        neptune_cluster = self.storage.neptune_cluster
        cluster_dims = {"DBClusterIdentifier": neptune_cluster.cluster_identifier}
        writer_dims = {**cluster_dims, "Role": "WRITER"}

        def _neptune_metric(
            metric_name: str,
            minutes: int = 5,
            statistic: str = "Average",
            writer: bool = False,
        ) -> cw.Metric:
            return cw.Metric(
                namespace="AWS/Neptune",
                metric_name=metric_name,
                dimensions_map=writer_dims if writer else cluster_dims,
                statistic=statistic,
                period=Duration.minutes(minutes),
            )

        # AWS publishes no recommended Neptune CPU alarm, and Neptune's metric
        # best practices note that high CPU can be appropriate: a bulk edge
        # write saturates the writer by design. Alarm only on saturation that
        # lasts three consecutive 5-minute periods.
        _alarm(
            "NeptuneHighCpu",
            _neptune_metric("CPUUtilization", writer=True),
            90,
            "Neptune writer CPU utilization is at least 90% for 15 minutes — "
            "ingestion writes and queries may stall",
            evaluation_periods=3,
        )
        # Minimum: the lowest point in the window, not a mean that hides it.
        freeable = _neptune_metric("FreeableMemory", statistic="Minimum", writer=True)
        # Low freeable memory precedes query/write failures.
        serverless = self.config.neptune_serverless_ncu
        if serverless is None:
            memory = neptune_instance_memory_bytes(self.config.neptune_instance)
            low_memory = memory // _LOW_MEMORY_SHARE if memory else _LOW_MEMORY_DEFAULT
            _alarm(
                "NeptuneLowFreeableMemory",
                freeable,
                low_memory,
                "Neptune writer freeable memory is low "
                f"(<{low_memory // 1024**2} MiB)",
                comparison=cw.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
            )
        else:
            # Neptune Serverless adds capacity when memory runs low, so at its
            # minimum (1 NCU is about 2 GiB) little free memory is the normal
            # state, not a fault. Two conditions matter instead: memory low
            # once the writer is at its maximum capacity (it cannot grow), and
            # the writer pinned at that maximum (the NCU range is too small).
            max_ncu = serverless[1]
            capacity = _neptune_metric(
                "ServerlessDatabaseCapacity", statistic="Maximum", writer=True
            )
            _alarm(
                "NeptuneLowFreeableMemory",
                cw.MathExpression(
                    # A huge constant below the maximum: never "low".
                    expression=f"IF(capacity >= {max_ncu:g}, memory, 1000000000000000)",
                    using_metrics={"capacity": capacity, "memory": freeable},
                    period=Duration.minutes(5),
                    label="Writer freeable memory at maximum capacity",
                ),
                _LOW_MEMORY_DEFAULT,
                "Neptune Serverless writer is at its maximum capacity "
                f"({max_ncu:g} NCU) with low freeable memory (<512 MiB)",
                comparison=cw.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
            )
            _alarm(
                "NeptuneServerlessAtMaxCapacity",
                capacity,
                max_ncu,
                "Neptune Serverless writer has run at its maximum capacity "
                f"({max_ncu:g} NCU) for 15 minutes: raise the maximum of "
                "neptune_serverless_ncu",
                evaluation_periods=3,
            )
        # The app talks Gremlin over WebSocket, which the deprecated HTTP-only
        # GremlinHttp5xx never counts; this per-second rate covers both.
        # Parallel edge writes hit ConcurrentModificationException, which the
        # writer retries, so a burst of errors is expected: alarm only when
        # errors persist in each of three consecutive minutes.
        _alarm(
            "NeptuneGremlinErrors",
            _neptune_metric("GremlinServerErrorsPerSec", minutes=1),
            0,
            "Neptune has returned Gremlin server errors for 3 consecutive "
            "minutes (sustained write/query failures, not retried conflicts)",
            comparison=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=3,
        )

    def _add_opensearch_capacity_alarms(self) -> None:
        """The rest of AWS's recommended OpenSearch alarms ("Recommended
        CloudWatch alarms for Amazon OpenSearch Service"), at its thresholds
        (the free-storage alarm fires earlier than its 25%, at 40%):
        data node CPU and old-generation heap, missing nodes, the shard count
        against the heap, the 5xx share of requests, the write and search
        queues, and the dedicated masters of a multi-node domain."""
        domain = self.storage.opensearch_domain
        config = self.config

        def metric(name: str, statistic: str, period: Duration) -> cw.Metric:
            return domain.metric(name, statistic=statistic, period=period)

        one_minute, fifteen = Duration.minutes(1), Duration.minutes(15)
        # Sustained use only: 100% CPU for a moment is normal under load.
        self._alarm(
            "OpenSearchHighCpu",
            metric("CPUUtilization", "Maximum", fifteen),
            80,
            "OpenSearch data node CPU at least 80% for 45 minutes (3 "
            "15-minute periods): a larger opensearch_instance or more nodes",
            evaluation_periods=3,
        )
        self._alarm(
            "OpenSearchOldGenJvmPressure",
            metric("OldGenJVMMemoryPressure", "Maximum", one_minute),
            80,
            "OpenSearch old-generation heap at least 80% for 3 consecutive "
            "minutes (garbage collection cannot free it: out-of-memory risk)",
            evaluation_periods=3,
        )
        # Nodes counts the dedicated masters too; a minimum below the
        # domain's size in a day means a node was unreachable at some point.
        masters = OPENSEARCH_MASTER_NODES if config.opensearch_count > 1 else 0
        nodes = config.opensearch_count + masters
        self._alarm(
            "OpenSearchNodesMissing",
            metric("Nodes", "Minimum", Duration.days(1)),
            nodes,
            f"OpenSearch had fewer than its {nodes} nodes at some point in a day "
            "(a node was unreachable or replaced)",
            comparison=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
        )
        heap = opensearch_heap_bytes(config.opensearch_instance)
        if heap is None:
            Annotations.of(self).add_warning(
                f"opensearch_instance {config.opensearch_instance!r} is not in "
                "the instance memory table: no OpenSearchShardsHigh alarm "
                "(25 shards per GiB of heap per data node)."
            )
        else:
            shards = SHARDS_PER_HEAP_GIB * heap // 1024**3 * config.opensearch_count
            self._alarm(
                "OpenSearchShardsHigh",
                metric("Shards.active", "Maximum", one_minute),
                shards,
                f"OpenSearch holds at least {shards} active shards (25 per GiB "
                "of heap per data node): delete unused namespaces or add nodes",
            )
        self._alarm(
            "OpenSearch5xxShare",
            cw.MathExpression(
                expression="100 * FILL(errors, 0) / requests",
                using_metrics={
                    "errors": metric("5xx", "Sum", Duration.minutes(5)),
                    "requests": metric(
                        "OpenSearchRequests", "Sum", Duration.minutes(5)
                    ),
                },
                label="5xx share of requests (%)",
                period=Duration.minutes(5),
            ),
            10,
            "At least 10% of OpenSearch requests failed with a 5xx in 5 "
            "minutes (overloaded data nodes or requests over the idle timeout)",
        )
        for ident, name, threshold in (
            ("OpenSearchWriteQueue", "ThreadpoolWriteQueue", 100),
            ("OpenSearchSearchQueue", "ThreadpoolSearchQueue", 500),
        ):
            # AWS also lists a search-queue maximum >= 5,000, which a node's
            # 1,000-task queue never reaches: left out.
            self._alarm(
                ident,
                metric(name, "Average", one_minute),
                threshold,
                f"OpenSearch {name} averaged at least {threshold} tasks in a "
                "minute (high concurrency: fewer concurrent requests or more "
                "resources)",
            )
        if not masters:
            return
        for ident, name, threshold, period, periods in (
            ("OpenSearchMasterHighCpu", "MasterCPUUtilization", 50, fifteen, 3),
            (
                "OpenSearchMasterJvmPressure",
                "MasterJVMMemoryPressure",
                95,
                one_minute,
                3,
            ),
            (
                "OpenSearchMasterOldGenJvmPressure",
                "MasterOldGenJVMMemoryPressure",
                80,
                one_minute,
                3,
            ),
        ):
            self._alarm(
                ident,
                metric(name, "Maximum", period),
                threshold,
                f"OpenSearch dedicated master {name} at least {threshold}% in "
                f"{periods} consecutive periods: a larger "
                "opensearch_master_instance",
                evaluation_periods=periods,
            )
        self._alarm(
            "OpenSearchMasterUnreachable",
            metric("MasterReachableFromNode", "Maximum", Duration.minutes(5)),
            1,
            "The OpenSearch master node was unreachable for 5 minutes "
            "(MasterNotDiscovered)",
            comparison=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
        )
