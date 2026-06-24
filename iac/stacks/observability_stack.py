# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observability: CloudWatch dashboard + alarms over the pipeline.

The application emits Embedded Metric Format (EMF) records under the
``aws_graphrag/ingestion`` namespace when run with ``--metrics-sink cloudwatch``
(see core/metrics.py). This stack surfaces those plus Step Functions execution
health on a dashboard, and alarms on pipeline failures -> the orchestration SNS
topic.
"""

from __future__ import annotations

from aws_cdk import Stack
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from constructs import Construct

from iac.config import DeploymentConfig
from iac.stacks.orchestration_stack import OrchestrationStack

EMF_NAMESPACE = "aws_graphrag/ingestion"


class ObservabilityStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: DeploymentConfig,
        orchestration: OrchestrationStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        sm = orchestration.state_machine

        # Alarm: any Step Functions execution failure -> SNS.
        failed_alarm = cw.Alarm(
            self,
            "PipelineFailures",
            metric=sm.metric_failed(),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            alarm_description="aws-graphrag ingestion pipeline had a failed execution",
        )
        failed_alarm.add_alarm_action(cw_actions.SnsAction(orchestration.alarm_topic))

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
