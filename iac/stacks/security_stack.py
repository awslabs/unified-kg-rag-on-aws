# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Security: an optional Amazon Bedrock Guardrail for the data plane.

If config.guardrail_identifier is set, an existing guardrail is reused (the app
attaches it via aws.bedrock.guardrail.identifier). Otherwise this stack creates
a baseline guardrail (PII + prompt-attack filters) and exposes its identifier so
the app config / Fargate env can reference it.

IAM is otherwise least-privilege and lives with the resources that need it (the
Fargate task role in compute_stack); this stack centralizes only the guardrail.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_bedrock as bedrock
from constructs import Construct

from iac.config import DeploymentConfig


class SecurityStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, config: DeploymentConfig, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config

        if config.guardrail_identifier:
            # Reuse an existing guardrail; nothing to create.
            self.guardrail_identifier = config.guardrail_identifier
            CfnOutput(
                self, "GuardrailIdentifier", value=config.guardrail_identifier
            )
            return

        guardrail = bedrock.CfnGuardrail(
            self,
            "Guardrail",
            name=f"{config.prefix}-guardrail",
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
        self.guardrail_identifier = guardrail.attr_guardrail_id
        CfnOutput(self, "GuardrailIdentifier", value=guardrail.attr_guardrail_id)
