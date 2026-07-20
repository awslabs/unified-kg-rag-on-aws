# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observability: CloudWatch dashboard + alarms over the pipeline.

The application emits Embedded Metric Format (EMF) records under the
``unified_kg_rag/ingestion`` namespace when run with ``--metrics-sink cloudwatch``
(see core/metrics.py). This stack surfaces those plus Step Functions execution
health on a dashboard, and alarms on pipeline failures -> the orchestration SNS
topic.
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
        sm = orchestration.state_machine

        # Alarm: any Step Functions execution failure -> SNS.
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

        # Alarm: any indexing failures -> SNS. A SUCCEEDED Step Functions run can
        # still have silently dropped artifacts (extracted > 0 but indexed == 0,
        # e.g. a backend write that erred per-item while the run continued). The
        # SFN-failure alarm above does not catch that; this EMF metric does.
        index_failed_alarm = cw.Alarm(
            self,
            "IndexingFailures",
            metric=cw.Metric(
                namespace=EMF_NAMESPACE,
                metric_name="total_items_index_failed",
                statistic="Sum",
                period=Duration.minutes(5),
            ),
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

        # Store-health alarms: the SFN/indexing alarms above only fire when a
        # pipeline RUN fails. These surface data-store degradation (red cluster,
        # disk pressure, memory pressure, DDB throttling) that would otherwise be
        # invisible until the next run breaks. All route to the same SNS topic.
        self._add_store_health_alarms(orchestration)

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
                left=[sm.metric_succeeded(), sm.metric_failed(), sm.metric_aborted()],
            ),
            cw.GraphWidget(
                title="Pipeline duration",
                left=[sm.metric_time()],
            ),
        )
        # Metric names must match what pipeline._emit_metrics actually emits
        # (the PipelineMetrics scalar field names), or the widgets render empty.
        extracted_metrics = [
            cw.Metric(namespace=EMF_NAMESPACE, metric_name=name, statistic="Sum")
            for name in (
                "total_entities_extracted",
                "total_relationships_extracted",
                "total_claims_extracted",
            )
        ]
        indexed_metrics = [
            cw.Metric(namespace=EMF_NAMESPACE, metric_name=name, statistic="Sum")
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
        self.dashboard = dashboard

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
        ) -> None:
            alarm = cw.Alarm(
                self,
                ident,
                metric=metric,
                threshold=threshold,
                evaluation_periods=1,
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
        _alarm(
            "OpenSearchFreeStorageLow",
            domain.metric_free_storage_space(period=Duration.minutes(5)),
            # MiB; alarm well before the ~20% block-write watermark on the 50 GiB
            # GP3 volume so there is time to react.
            20480,
            "OpenSearch free storage space is low (<20 GiB)",
            comparison=cw.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
        )
        _alarm(
            "OpenSearchJvmPressure",
            domain.metric_jvm_memory_pressure(period=Duration.minutes(5)),
            85,
            "OpenSearch JVM memory pressure is high (>85%)",
        )
        # DynamoDB doc-status registry: throttled writes mean delta/lineage
        # updates are being rejected, which silently corrupts incremental-indexing
        # state. Alarm on the write path specifically (a single operation, so the
        # metric stays a plain metric, not a >10-metric math expression).
        _alarm(
            "DocStatusWriteThrottled",
            table.metric_throttled_requests_for_operation(
                "PutItem", period=Duration.minutes(5)
            ),
            1,
            "DynamoDB doc-status table is throttling PutItem (write) requests",
        )

        # Neptune: the primary graph store and the write target of every
        # ingestion phase. Without these, Neptune degradation (CPU saturation
        # during large edge writes, memory pressure, throttled Gremlin) is
        # invisible until a pipeline run breaks. Metrics are read from the
        # AWS/Neptune namespace keyed by the cluster id (stable across the
        # neptune_alpha construct's API surface).
        neptune_cluster = self.storage.neptune_cluster
        neptune_dims = {
            "DBClusterIdentifier": neptune_cluster.cluster_identifier,
        }

        def _neptune_metric(metric_name: str) -> cw.Metric:
            return cw.Metric(
                namespace="AWS/Neptune",
                metric_name=metric_name,
                dimensions_map=neptune_dims,
                statistic="Average",
                period=Duration.minutes(5),
            )

        _alarm(
            "NeptuneHighCpu",
            _neptune_metric("CPUUtilization"),
            90,
            "Neptune CPU utilization is high (>90%) — ingestion writes may stall",
        )
        _alarm(
            "NeptuneLowFreeableMemory",
            _neptune_metric("FreeableMemory"),
            # bytes; ~512 MiB. Low freeable memory precedes query/write failures.
            536870912,
            "Neptune freeable memory is low (<512 MiB)",
            comparison=cw.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
        )
        _alarm(
            "NeptuneGremlinErrors",
            cw.Metric(
                namespace="AWS/Neptune",
                metric_name="GremlinHttp5xx",
                dimensions_map=neptune_dims,
                statistic="Sum",
                period=Duration.minutes(5),
            ),
            1,
            "Neptune is returning Gremlin 5xx errors (write/query failures)",
        )
