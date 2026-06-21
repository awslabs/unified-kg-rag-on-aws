# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Security: shared CMK (optional) + Bedrock Guardrail.

- KMS: when config.use_cmk, a single customer-managed key encrypts at-rest data
  across the deployment (S3 cache, Neptune, OpenSearch, SNS, DynamoDB). Key
  rotation is enabled. When use_cmk is False, services use AWS-managed keys
  (cheaper; fine for dev). Exposed as ``self.kms_key`` (None if disabled).
- Guardrail: reuse config.guardrail_identifier, else create a baseline guardrail
  (PII anonymization + prompt-attack filter).
"""

from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
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
        self.guardrail_identifier = self._build_guardrail()
        CfnOutput(self, "GuardrailIdentifier", value=self.guardrail_identifier)
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
            description="aws-graphrag at-rest encryption key (S3/Neptune/OpenSearch/SNS/DDB)",
            enable_key_rotation=True,
            removal_policy=(
                RemovalPolicy.DESTROY
                if self.config.removal_destroy
                else RemovalPolicy.RETAIN
            ),
        )

    # --------------------------------------------------------- Guardrail
    def _build_guardrail(self) -> str:
        if self.config.guardrail_identifier:
            return self.config.guardrail_identifier

        guardrail = bedrock.CfnGuardrail(
            self,
            "Guardrail",
            name=f"{self.config.prefix}-guardrail",
            blocked_input_messaging="This request was blocked by content policy.",
            blocked_outputs_messaging="This response was blocked by content policy.",
            description="Baseline guardrail for aws-graphrag (PII + prompt attack).",
            sensitive_information_policy_config=(
                bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                    pii_entities_config=[
                        bedrock.CfnGuardrail.PiiEntityConfigProperty(
                            type=t, action="ANONYMIZE"
                        )
                        for t in ("EMAIL", "PHONE", "NAME", "CREDIT_DEBIT_CARD_NUMBER")
                    ]
                )
            ),
            content_policy_config=(
                bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                    filters_config=[
                        bedrock.CfnGuardrail.ContentFilterConfigProperty(
                            type="PROMPT_ATTACK",
                            input_strength="HIGH",
                            output_strength="NONE",
                        )
                    ]
                )
            ),
        )
        return guardrail.attr_guardrail_id
