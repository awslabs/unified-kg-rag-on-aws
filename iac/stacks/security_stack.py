# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security: shared CMK (optional).

KMS: when config.use_cmk, a single customer-managed key encrypts at-rest data
across the deployment (S3 cache, Neptune, OpenSearch, SNS, DynamoDB). Key
rotation is enabled. When use_cmk is False, services use AWS-managed keys
(cheaper; fine for dev). Exposed as ``self.kms_key`` (None if disabled).

The Bedrock Guardrail lives in its own ``GuardrailStack`` because it must be
created in the Bedrock runtime region (``bedrock_region``), which can differ
from the deploy region that hosts Neptune/OpenSearch/KMS.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack
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
        return kms.Key(
            self,
            "DataKey",
            alias=f"alias/{self.config.prefix}-data",
            description="unified-kg-rag-on-aws at-rest encryption key (S3/Neptune/OpenSearch/SNS/DDB)",
            enable_key_rotation=True,
            removal_policy=(
                RemovalPolicy.DESTROY
                if self.config.removal_destroy
                else RemovalPolicy.RETAIN
            ),
        )
