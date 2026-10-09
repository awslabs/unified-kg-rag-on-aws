# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security: shared CMK (optional).

KMS: when config.use_cmk, a single customer-managed key encrypts at-rest data
across the deployment (S3 cache, Neptune, OpenSearch, DynamoDB, ECR, CloudWatch
Logs log groups). The SNS
alarm topic has its own key in the orchestration stack, because CloudWatch must be
allowed to use it. Key rotation is enabled. When use_cmk is False, services use
AWS-managed keys (cheaper; fine for dev). Exposed as ``self.kms_key`` (None if disabled).

The Bedrock Guardrail lives in its own ``GuardrailStack`` because it must be
created in the Bedrock runtime region (``bedrock_region``), which can differ
from the deploy region that hosts Neptune/OpenSearch/KMS.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from constructs import Construct

from iac.config import DeploymentConfig


class SecurityStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, config: DeploymentConfig, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config

        self.kms_key = self._build_kms_key()
        if self.kms_key is not None:
            CfnOutput(self, "KmsKeyArn", value=self.kms_key.key_arn)

    # --------------------------------------------------------------- KMS
    def _build_kms_key(self) -> kms.Key | None:
        if not self.config.use_cmk:
            return None
        key = kms.Key(
            self,
            "DataKey",
            alias=f"alias/{self.config.prefix}-data",
            description="unified-kg-rag-on-aws at-rest encryption key (S3/Neptune/OpenSearch/DDB)",
            enable_key_rotation=True,
            removal_policy=(
                RemovalPolicy.DESTROY
                if self.config.removal_destroy
                else RemovalPolicy.RETAIN
            ),
        )
        self._allow_cloudwatch_logs(key)
        return key

    def _allow_cloudwatch_logs(self, key: kms.Key) -> None:
        """Key policy for the log groups encrypted with this key.

        Per "Encrypt log data in CloudWatch Logs using AWS KMS": the regional
        logs service principal uses the key for this account's log groups
        (encryption-context condition), and the principals that write or read
        those log groups (ECS awslogs, flow logs, operators) need the key
        through CloudWatch Logs only (kms:ViaService). The second statement
        grants that to principals in this account, so no role needs its own
        KMS policy just to write or read logs.
        """
        logs_service = f"logs.{self.region}.amazonaws.com"
        key.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowCloudWatchLogs",
                principals=[iam.ServicePrincipal(logs_service)],
                actions=[
                    "kms:Encrypt*",
                    "kms:Decrypt*",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*",
                    "kms:Describe*",
                ],
                resources=["*"],
                conditions={
                    "ArnLike": {
                        "kms:EncryptionContext:aws:logs:arn": (
                            f"arn:{self.partition}:logs:{self.region}:"
                            f"{self.account}:log-group:*"
                        )
                    }
                },
            )
        )
        key.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowAccountUseViaCloudWatchLogs",
                principals=[iam.AnyPrincipal()],
                actions=[
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*",
                    "kms:Describe*",
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "kms:CallerAccount": self.account,
                        "kms:ViaService": logs_service,
                    }
                },
            )
        )
