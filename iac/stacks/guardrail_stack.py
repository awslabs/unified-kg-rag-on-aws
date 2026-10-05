# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guardrail: a baseline Bedrock Guardrail (PII anonymization + prompt-attack
filter) created in the Bedrock *runtime* region.

A guardrail must exist in the same region the InvokeModel/Converse call is made
against. Because this deployment runs Bedrock cross-region (``bedrock_region``,
e.g. us-west-2) while Neptune/OpenSearch/KMS live in the deploy region (e.g.
ap-northeast-2), the guardrail is its own region-pinned stack rather than part
of the deploy-region SecurityStack. To avoid cross-region CloudFormation
references, the identifier reaches the compute task through the
``guardrail_identifier`` CDK context value (see ``iac/README.md``).

"Create" and "use" are separate knobs so the documented two-step deploy is
stable:

* ``create_guardrail`` (default ``True``) decides whether THIS stack owns a
  baseline guardrail. The ``CfnGuardrail`` is synthesized on every deploy while
  it is true, including the second ``cdk deploy --all -c
  guardrail_identifier=<id>`` step. Dropping it from the template on the second
  step (the previous behaviour) made CloudFormation delete the guardrail created
  in step 1 while compute kept injecting the now-dead id.
* ``guardrail_identifier`` is only the id the compute task USES. It never
  disables creation. To bring your own externally managed guardrail, pass
  ``-c create_guardrail=false -c guardrail_identifier=<id>``.

The guardrail is also ``RETAIN``-ed on stack deletion as a safety net: its id is
consumed out-of-band (CDK context -> task env var), so CloudFormation cannot see
that a delete would break compute.
"""

from __future__ import annotations

from aws_cdk import Annotations, CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
from constructs import Construct

from iac.config import DeploymentConfig


class GuardrailStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, config: DeploymentConfig, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        self.guardrail: bedrock.CfnGuardrail | None = None
        self.guardrail_identifier: str | None
        if config.create_guardrail:
            self.guardrail = self._build_guardrail()
            self.guardrail_identifier = self.guardrail.attr_guardrail_id
            if not config.guardrail_identifier:
                Annotations.of(self).add_info(
                    "Guardrail created but not attached to compute yet: re-run "
                    "`cdk deploy --all -c guardrail_identifier=<GuardrailIdentifier "
                    "output>` to inject BEDROCK_GUARDRAIL_IDENTIFIER."
                )
        else:
            # Bring-your-own path: nothing is created; surface the external id.
            self.guardrail_identifier = config.guardrail_identifier
        if self.guardrail_identifier:
            CfnOutput(self, "GuardrailIdentifier", value=self.guardrail_identifier)

    def _build_guardrail(self) -> bedrock.CfnGuardrail:
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
            description="Baseline guardrail for unified-kg-rag-on-aws (PII + prompt attack).",
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
        # Always RETAIN, independent of removal_destroy: the id is wired to
        # compute out-of-band, so a CloudFormation-side delete (template drift,
        # a mistaken synth, `cdk destroy`) would silently leave compute pointing
        # at a dead guardrail. A retained guardrail must be deleted manually.
        guardrail.apply_removal_policy(RemovalPolicy.RETAIN)
        return guardrail
