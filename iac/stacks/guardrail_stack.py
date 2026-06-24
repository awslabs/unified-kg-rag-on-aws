# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guardrail: a baseline Bedrock Guardrail (PII anonymization + prompt-attack
filter) created in the Bedrock *runtime* region.

A guardrail must exist in the same region the InvokeModel/Converse call is made
against. Because this deployment runs Bedrock cross-region (``bedrock_region``,
e.g. us-west-2) while Neptune/OpenSearch/KMS live in the deploy region (e.g.
ap-northeast-2), the guardrail is its own region-pinned stack rather than part
of the deploy-region SecurityStack. The resolved identifier is consumed by the
compute task via cross-region references.

When ``config.guardrail_identifier`` is set, no guardrail is created and that id
is surfaced as-is (reuse path).
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_bedrock as bedrock
from constructs import Construct

from iac.config import DeploymentConfig


class GuardrailStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, config: DeploymentConfig, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        self.guardrail_identifier = self._build_guardrail()
        CfnOutput(self, "GuardrailIdentifier", value=self.guardrail_identifier)

    def _build_guardrail(self) -> str:
        if self.config.guardrail_identifier:
            return self.config.guardrail_identifier

        # Region-qualify the name: the guardrail lives in the Bedrock runtime
        # region, which may differ from the deploy region, and guardrail names
        # must be unique per account. The region suffix also avoids clashing with
        # any pre-existing same-named guardrail during a migration.
        guardrail = bedrock.CfnGuardrail(
            self,
            "Guardrail",
            name=f"{self.config.prefix}-guardrail-{self.region}",
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
