# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Indexed artifacts (EMF)",
                left=[
                    cw.Metric(
                        namespace=EMF_NAMESPACE,
                        metric_name=name,
                        statistic="Sum",
                    )
                    for name in ("entities", "relationships", "claims")
                ],
            ),
        )
        self.dashboard = dashboard
