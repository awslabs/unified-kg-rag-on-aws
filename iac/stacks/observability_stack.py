# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observability: CloudWatch dashboard + alarms over the pipeline.

The application emits Embedded Metric Format (EMF) records under the
``unified_kg_rag/ingestion`` namespace when run with ``--metrics-sink cloudwatch``
(see unified_kg_rag/shared/metrics.py), each stage's as the stage finishes. The
task gets ``GRAPHRAG_METRICS_ENVIRONMENT=<env_name>``, so the series carry an
``Environment`` dimension and the alarms and widgets here query only this
deployment's runs: a dev run in the same account and region does not trip prod's
alarms. This stack surfaces those plus Step Functions execution health, Bedrock
throttling and latency and the ingestion task's memory on a dashboard, and alarms
on failed, timed-out or aborted runs, dropped artifacts, sustained Bedrock
throttling, ingestion task memory pressure and store health -> the orchestration
SNS topic.
"""

from __future__ import annotations

from aws_cdk import Duration, Stack
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from constructs import Construct

from iac.config import DeploymentConfig
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.storage_stack import StorageStack

EMF_NAMESPACE = "unified_kg_rag/ingestion"

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
)

# Bedrock publishes its runtime metrics per ModelId only, and the model ids
# the app resolves (inference profiles included) are app config, not known
# here: these Metrics Insights queries sum over every model in the account and
# region, so other Bedrock users there count too.
BEDROCK_THROTTLES = (
    'SELECT SUM(InvocationThrottles) FROM SCHEMA("AWS/Bedrock", ModelId)'
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

        # Store-health alarms: the run alarms above only fire when a pipeline
        # RUN goes wrong. These surface data-store degradation (red cluster,
        # disk pressure, memory pressure, DDB throttling) that would otherwise be
        # invisible until the next run breaks. All route to the same SNS topic.
        self._add_store_health_alarms(orchestration)
        self._add_capacity_alarms(orchestration)

        # All alarms route to the orchestration SNS topic, but with no subscriber
        # they fire into the void. Warn at synth (rather than silently) when no
        # email is wired so the operator knows to add one (`-c alarm_email=...`).
        if not config.alarm_email:
            from aws_cdk import Annotations

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
                title="Ingestion task memory and CPU (% of reserved)",
                left=[
                    self._task_utilization("Memory"),
                    self._task_utilization("Cpu"),
                ],
            ),
        )
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Bedrock invocations and throttles (all models)",
                left=[
                    self._insights(
                        'SELECT SUM(Invocations) FROM SCHEMA("AWS/Bedrock", ModelId)'
                        " GROUP BY ModelId",
                        "Invocations",
                    ),
                    self._insights(BEDROCK_THROTTLES, "Throttles"),
                ],
            ),
            cw.GraphWidget(
                title="Bedrock invocation latency by model (ms, average)",
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
            "TaskDefinitionFamily": self.compute.task_definition.family,
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

    def _add_store_health_alarms(self, orchestration: OrchestrationStack) -> None:
        topic = orchestration.alarm_topic
        action = cw_actions.SnsAction(topic)
        domain = self.storage.opensearch_domain
        table = self.storage.doc_status_table

        def _alarm(
            ident: str,
            metric: cw.IMetric,
            threshold: float,
            description: str,
            comparison: cw.ComparisonOperator = (
                cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD
            ),
            evaluation_periods: int = 1,
        ) -> None:
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
            alarm.add_alarm_action(action)

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
        _alarm(
            "NeptuneLowFreeableMemory",
            # Minimum: the lowest point in the window, not a mean that hides it.
            _neptune_metric("FreeableMemory", statistic="Minimum", writer=True),
            # bytes; ~512 MiB. Low freeable memory precedes query/write failures.
            536870912,
            "Neptune writer freeable memory is low (<512 MiB)",
            comparison=cw.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
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
