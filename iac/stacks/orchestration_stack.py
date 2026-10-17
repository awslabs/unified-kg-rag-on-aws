# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestration: a Step Functions state machine that runs the ingestion
pipeline as one resumable Fargate task.

The task runs ``run-ingestion`` over every stage with ``--s3-sync``: each stage's
output is uploaded to S3 as the stage finishes, and on SIGTERM (the execution
timed out or was stopped) the task uploads what it has and exits. Re-running
an execution with the same ``pipeline_id`` therefore resumes at the stage that
was running, and that stage's answered model calls come from the LLM response
cache.

The task is retried only when ECS fails to start it (a container that exits
non-zero is a pipeline failure, and rerunning it repeats the same work); its
error is caught and published, with the execution id, to an SNS topic. The
state machine input carries only ``source_directory`` (an s3:// URI the
container entrypoint syncs locally) and ``pipeline_id``. There is no
full-vs-incremental switch: a run is incremental when ``aws.dynamodb`` is
enabled in the baked-in config.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, Fn, RemovalPolicy, Stack, Token
from aws_cdk import aws_ecs as ecs  # noqa: F401  (FargatePlatformVersion)
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subscriptions
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack


class OrchestrationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: DeploymentConfig,
        networking: NetworkingStack,
        compute: ComputeStack,
        cache_bucket_name: str,
        kms_key: kms.IKey | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        self.networking = networking
        self.compute = compute
        self.cache_bucket_name = cache_bucket_name

        # The alarm topic always gets its OWN customer-managed key. CloudWatch
        # alarms can only publish to an encrypted topic whose key policy lets
        # cloudwatch.amazonaws.com use the key; the AWS-managed alias/aws/sns key
        # policy cannot be edited and the shared data CMK only trusts the account
        # root, so either one silently drops every alarm notification. See
        # "Enable compatibility between event sources from AWS services and
        # encrypted topics" in the SNS developer guide.
        self.topic_key = kms.Key(
            self,
            "AlarmTopicKey",
            description="unified-kg-rag-on-aws alarm topic key (CloudWatch alarms "
            "+ Step Functions publish)",
            enable_key_rotation=True,
            removal_policy=(
                RemovalPolicy.DESTROY
                if config.removal_destroy
                else RemovalPolicy.RETAIN
            ),
        )
        self.topic_key.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowCloudWatchAlarmsToPublish",
                principals=[iam.ServicePrincipal("cloudwatch.amazonaws.com")],
                actions=["kms:Decrypt", "kms:GenerateDataKey*"],
                resources=["*"],
                conditions={"StringEquals": {"aws:SourceAccount": self.account}},
            )
        )
        self.alarm_topic = sns.Topic(
            self,
            "PipelineAlarms",
            topic_name=f"{config.prefix}-pipeline-alarms",
            master_key=self.topic_key,
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
            # The shared CMK when use_cmk (Step Functions log delivery needs
            # only the logs service principal in the key policy).
            encryption_key=kms_key,
            retention=logs.RetentionDays.ONE_MONTH,
            # Fixed name: retained copies would collide on the next deploy.
            removal_policy=(
                RemovalPolicy.DESTROY
                if config.removal_destroy
                else RemovalPolicy.RETAIN
            ),
        )

        # The task's failure handler: the Catch keeps the state input
        # and adds the error under $.error, so the message carries the run id
        # and the error (its name and, when the error has one, its cause: the
        # stopped ECS task, with its exit code and the command). The
        # error is rendered whole: a path to an absent Cause would fail the
        # publish itself.
        publish = tasks.SnsPublish(
            self,
            "NotifyFailure",
            topic=self.alarm_topic,
            subject="unified-kg-rag-on-aws ingestion failed",
            message=sfn.TaskInput.from_object(
                {
                    "execution_id": sfn.JsonPath.execution_id,
                    "pipeline_id": sfn.JsonPath.string_at("$.pipeline_id"),
                    "error": sfn.JsonPath.json_to_string(
                        sfn.JsonPath.object_at("$.error")
                    ),
                }
            ),
            result_path=sfn.JsonPath.DISCARD,
        )
        # A throttled or transient publish error would otherwise fail the run
        # without its notification (the PipelineFailures alarm still fires).
        # States.TaskFailed matches every task error except States.Timeout.
        publish.add_retry(
            errors=["States.TaskFailed"],
            interval=Duration.seconds(2),
            max_attempts=3,
            backoff_rate=2.0,
        )
        self.notify_failure = publish.next(
            sfn.Fail(self, "PipelineFailed", cause="The ingestion task failed")
        )

        definition = self._build_definition()
        self.state_machine = sfn.StateMachine(
            self,
            "IngestionPipeline",
            state_machine_name=f"{config.prefix}-ingestion",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            # A run that hits this is TIMED_OUT, not FAILED: the observability
            # stack alarms on timed-out and aborted executions separately.
            timeout=Duration.hours(config.sfn_timeout_hours),
            tracing_enabled=True,  # X-Ray
            logs=sfn.LogOptions(destination=self.log_group, level=sfn.LogLevel.ALL),
        )
        # SnsPublish grants sns:Publish only; publishing to an encrypted topic
        # also needs the key (SNS docs: kms:GenerateDataKey* + kms:Decrypt).
        self.topic_key.grant(self.state_machine, "kms:Decrypt", "kms:GenerateDataKey*")
        CfnOutput(self, "StateMachineArn", value=self.state_machine.state_machine_arn)

    # ---------------------------------------------------------- ingest task
    def _ingest_task(self) -> tasks.EcsRunTask:
        task_definition = self.compute.task_definition
        container = task_definition.default_container
        assert container is not None

        run = tasks.EcsRunTask(
            self,
            "Ingest",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,  # sync: wait for exit
            cluster=self.compute.cluster,
            task_definition=task_definition,
            # The revision deployed with this state machine, not the family's
            # latest ACTIVE one: a revision registered outside this deployment
            # (or by a later deploy mid-execution) is not picked up. The ARN
            # ends in ":<revision>".
            revision_number=Token.as_number(
                Fn.select(6, Fn.split(":", task_definition.task_definition_arn))
            ),
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
                    # fully-dynamic command), so the flags are static here and
                    # per-run values come from environment overrides below
                    # (the CLI reads UKR_SOURCE_DIRECTORY / UKR_PIPELINE_ID;
                    # --config-path is a fixed in-image path baked into the image).
                    command=[
                        "run-ingestion",
                        "--config-path",
                        "/app/config.yaml",
                        "--metrics-sink",
                        "cloudwatch",
                        # The task's disk is gone when it ends: the stage
                        # checkpoints and LLM response cache live in S3, so a
                        # re-run with the same pipeline id resumes from them.
                        "--s3-sync",
                        "--s3-bucket-name",
                        self.cache_bucket_name,
                    ],
                    environment=[
                        tasks.TaskEnvironmentVariable(
                            name="UKR_SOURCE_DIRECTORY",
                            value=sfn.JsonPath.string_at("$.source_directory"),
                        ),
                        tasks.TaskEnvironmentVariable(
                            name="UKR_PIPELINE_ID",
                            value=sfn.JsonPath.string_at("$.pipeline_id"),
                        ),
                    ],
                )
            ],
            result_path="$.task",
        )
        # Retry only failures to start the task (RunTask API errors, or a
        # RunTask response with `failures`, e.g. no Fargate capacity, which
        # Step Functions raises as AmazonECS.Unknown). A task that ran and
        # exited non-zero is States.TaskFailed and is not retried.
        run.add_retry(
            errors=["ECS.AmazonECSException", "AmazonECS.Unknown"],
            interval=Duration.seconds(30),
            max_attempts=2,
            backoff_rate=2.0,
        )
        run.add_catch(self.notify_failure, errors=["States.ALL"], result_path="$.error")
        return run

    # ---------------------------------------------------------- definition
    def _build_definition(self) -> sfn.IChainable:
        return self._ingest_task().next(sfn.Succeed(self, "PipelineComplete"))
