# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestration: a Step Functions state machine that runs the ingestion
pipeline as four resumable phases on Fargate.

Each phase is the SAME container image invoked with ``run-ingestion`` and a
different ``--resume-from-stage`` + ``--enabled-stages`` window, sharing one
``--pipeline-id`` so the app's S3-backed stage checkpoints hand off between
phases:

  prep         document_parsing, document_loading, text_chunking, translation
  graph-build  graph_extraction, gleaning, graph_resolution, claim_*
  analysis     graph_analysis, community_detection
  index        indexing

Each phase has bounded retries; any failure is caught and published to an SNS
topic. The state machine input selects a full vs incremental run (the
incremental path simply enables aws.dynamodb via config; the same phases run).
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_ecs as ecs  # noqa: F401  (FargatePlatformVersion)
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subscriptions
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack

# Phase -> ordered pipeline stages (the resume window). Phase N resumes from its
# first stage; the app runs that stage onward but --enabled-stages bounds it to
# this phase's stages so the next phase picks up the rest.
PHASES: list[tuple[str, str, list[str]]] = [
    (
        "Prep",
        "document_parsing",
        ["document_parsing", "document_loading", "text_chunking", "translation"],
    ),
    (
        "GraphBuild",
        "graph_extraction",
        [
            "graph_extraction",
            "gleaning",
            "graph_resolution",
            "claim_extraction",
            "claim_resolution",
        ],
    ),
    ("Analysis", "graph_analysis", ["graph_analysis", "community_detection"]),
    ("Index", "indexing", ["indexing"]),
]


class OrchestrationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: DeploymentConfig,
        networking: NetworkingStack,
        compute: ComputeStack,
        cache_bucket_name: str,
        kms_key=None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        self.networking = networking
        self.compute = compute
        self.cache_bucket_name = cache_bucket_name

        self.alarm_topic = sns.Topic(
            self,
            "PipelineAlarms",
            topic_name=f"{config.prefix}-pipeline-alarms",
            master_key=kms_key,  # SSE for the topic when a CMK is provided
        )
        # Require TLS for all publishers (defense in depth).
        self.alarm_topic.add_to_resource_policy(
            iam.PolicyStatement(
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=["sns:Publish"],
                resources=[self.alarm_topic.topic_arn],
                conditions={"Bool": {"aws:SecureTransport": "false"}},
            )
        )
        if config.alarm_email:
            self.alarm_topic.add_subscription(
                subscriptions.EmailSubscription(config.alarm_email)
            )

        # Execution logging for audit/debug (operational excellence).
        self.log_group = logs.LogGroup(
            self,
            "PipelineLogs",
            log_group_name=f"/{config.prefix}/pipeline",
            retention=logs.RetentionDays.ONE_MONTH,
        )

        definition = self._build_definition()
        self.state_machine = sfn.StateMachine(
            self,
            "IngestionPipeline",
            state_machine_name=f"{config.prefix}-ingestion",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.hours(6),
            tracing_enabled=True,  # X-Ray
            logs=sfn.LogOptions(destination=self.log_group, level=sfn.LogLevel.ALL),
        )
        CfnOutput(self, "StateMachineArn", value=self.state_machine.state_machine_arn)

    # --------------------------------------------------------- phase task
    def _phase_task(
        self, phase_name: str, resume_from: str, stages: list[str]
    ) -> tasks.EcsRunTask:
        container = self.compute.task_definition.default_container
        assert container is not None

        run = tasks.EcsRunTask(
            self,
            f"{phase_name}Phase",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,  # sync: wait for exit
            cluster=self.compute.cluster,
            task_definition=self.compute.task_definition,
            launch_target=tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST
            ),
            assign_public_ip=False,
            subnets=self.networking.app_subnets,
            security_groups=[self.networking.service_sg],
            container_overrides=[
                tasks.ContainerOverride(
                    container_definition=container,
                    # `command` must be a static array (ECS RunTask cannot take a
                    # fully-dynamic command), so per-phase flags are static here
                    # and per-run values come from environment overrides below
                    # (the CLI reads GRAPHRAG_SOURCE_DIRECTORY / GRAPHRAG_PIPELINE_ID;
                    # --config-path is a fixed in-image path baked into the image).
                    command=[
                        "run-ingestion",
                        "--config-path",
                        "/app/config.yaml",
                        "--resume-from-stage",
                        resume_from,
                        "--enabled-stages",
                        ",".join(stages),
                        "--metrics-sink",
                        "cloudwatch",
                        # Each phase is a separate Fargate task with its own local
                        # disk, so the stage checkpoints/cache must round-trip
                        # through S3 for the resume handoff between phases to work.
                        "--s3-sync",
                        "--s3-bucket-name",
                        self.cache_bucket_name,
                    ],
                    environment=[
                        tasks.TaskEnvironmentVariable(
                            name="GRAPHRAG_SOURCE_DIRECTORY",
                            value=sfn.JsonPath.string_at("$.source_directory"),
                        ),
                        tasks.TaskEnvironmentVariable(
                            name="GRAPHRAG_PIPELINE_ID",
                            value=sfn.JsonPath.string_at("$.pipeline_id"),
                        ),
                    ],
                )
            ],
            result_path="$.last_phase",
        )
        # Bounded retry on transient task failures.
        run.add_retry(
            errors=["States.TaskFailed", "States.Timeout"],
            interval=Duration.seconds(30),
            max_attempts=2,
            backoff_rate=2.0,
        )
        # Any unrecovered failure -> notify and fail the run.
        notify = tasks.SnsPublish(
            self,
            f"{phase_name}Failed",
            topic=self.alarm_topic,
            message=sfn.TaskInput.from_text(
                f"aws-graphrag ingestion failed at phase {phase_name}"
            ),
        ).next(sfn.Fail(self, f"{phase_name}Abort", cause=f"{phase_name} failed"))
        run.add_catch(notify, errors=["States.ALL"])
        return run

    # ---------------------------------------------------------- definition
    def _build_definition(self) -> sfn.IChainable:
        phase_tasks = [self._phase_task(name, rf, st) for name, rf, st in PHASES]
        chain: sfn.IChainable = phase_tasks[0]
        cursor = phase_tasks[0]
        for nxt in phase_tasks[1:]:
            cursor = cursor.next(nxt)  # type: ignore[union-attr]
        phase_tasks[-1].next(sfn.Succeed(self, "PipelineComplete"))
        return chain
