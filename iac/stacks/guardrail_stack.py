# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guardrail: a baseline Bedrock Guardrail (PII anonymization + prompt-attack
filter) created in the Bedrock *runtime* region.

A guardrail must exist in the same region the InvokeModel/Converse call is made
against. Because this deployment runs Bedrock cross-region (``bedrock_region``,
e.g. us-west-2) while Neptune/OpenSearch/KMS live in the deploy region (e.g.
ap-northeast-2), the guardrail is its own region-pinned stack rather than part
of the deploy-region SecurityStack.

How the tasks get the guardrail (``GuardrailStack.attachment``):

* Same region (``bedrock_region`` unset or the deploy region, which private
  mode requires): the guardrail this stack owns is attached by default. Its id
  and published version go into two SSM parameters, which the task
  definitions read as CloudFormation dynamic references when they deploy. Not
  CloudFormation exports: a policy change publishes a new version, and an
  export in use cannot change value, so the guardrail deploy would fail. The
  tasks' container carries the policy digest as a label, so a policy change
  also changes the task definitions and they read the new version.
* Another region, or an external guardrail: the identifier reaches compute
  through the ``guardrail_identifier`` CDK context value (a cross-region
  reference would rewrite every stack's export plumbing); see
  ``iac/README.md``. A ``guardrail_identifier`` given always wins.

"Create" and "use" are separate knobs so the cross-region two-step deploy is
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

The created guardrail is also published as a version (``CfnGuardrailVersion``,
output ``GuardrailVersion``): ``DRAFT`` changes with every edit, so the task
should apply a pinned version (``-c guardrail_version=<GuardrailVersion>``).
The version's description carries a digest of the policy, so a change to the
policy in this file publishes a new version on the next deploy; versions are
retained (they go with the guardrail), so a task still pinned to the previous
one keeps working until compute is redeployed with the new number.

``guardrail_contextual_grounding`` (default ``False``) adds a contextual
grounding policy to the created guardrail: GROUNDING and RELEVANCE filters
that block an answer scoring below ``guardrail_grounding_threshold``. It only scores an answer checked with the
retrieved context (the app's ``contextual_grounding.enabled``, which compute
turns on with the same flag); input checks are unaffected. Classic tier (the
default, and the only one without a cross-Region guardrail profile) scores
English, French and Spanish only.

The removal policy follows ``removal_destroy``: ``DESTROY`` in dev (so a
deploy -> destroy -> deploy cycle does not collide on the fixed guardrail name)
and ``RETAIN`` otherwise as a safety net, because the id is consumed out-of-band
(CDK context -> task env var) and CloudFormation cannot see that a delete would
break compute.
"""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

from aws_cdk import (
    Annotations,
    CfnDynamicReference,
    CfnDynamicReferenceService,
    CfnOutput,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from iac.config import DeploymentConfig

# PII types the baseline guardrail anonymizes (see _build_guardrail for why NAME
# is not one of them).
PII_ANONYMIZED = ("EMAIL", "PHONE", "CREDIT_DEBIT_CARD_NUMBER")


class GuardrailAttachment(NamedTuple):
    """The guardrail the tasks apply."""

    identifier: str
    # The published version; None applies the guardrail's DRAFT.
    version: str | None
    # Set when the values are read at deploy time from this stack's SSM
    # parameters: the stack to deploy first, and the policy digest whose
    # change makes the task definitions read them again.
    source: Stack | None = None
    revision: str | None = None


class GuardrailStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: DeploymentConfig,
        deploy_region: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        self.guardrail: bedrock.CfnGuardrail | None = None
        self.guardrail_version: bedrock.CfnGuardrailVersion | None = None
        self.guardrail_identifier: str | None
        self.policy_digest: str | None = None
        # What compute attaches: the context values when given, else the
        # owned guardrail when it lives in the deploy region.
        self.attachment: GuardrailAttachment | None = (
            GuardrailAttachment(config.guardrail_identifier, config.guardrail_version)
            if config.guardrail_identifier
            else None
        )
        if config.create_guardrail:
            self.guardrail = self._build_guardrail()
            self.guardrail_identifier = self.guardrail.attr_guardrail_id
            self.guardrail_version = self._publish_version(self.guardrail)
            CfnOutput(
                self,
                "GuardrailVersion",
                value=self.guardrail_version.attr_version,
                description="Pin it with -c guardrail_version=<this value>",
            )
            if config.guardrail_identifier and not config.create_guardrail_explicit:
                # Behaviour change for bring-your-own users: setting only
                # guardrail_identifier used to skip creation; it no longer does.
                Annotations.of(self).add_warning(
                    "guardrail_identifier is set but create_guardrail was not: this "
                    "stack also creates and owns a baseline guardrail. That is "
                    "expected when the id is this stack's GuardrailIdentifier "
                    "output; for an externally managed guardrail pass "
                    "`-c create_guardrail=false` (or `-c create_guardrail=true` to "
                    "silence this warning)."
                )
            if self.attachment is None and config.bedrock_region in (
                None,
                deploy_region,
            ):
                self.attachment = self._attach_owned()
            elif self.attachment is None:
                # A warning, not a note: across regions the id reaches
                # compute only through context, so a later deploy that omits
                # it detaches the guardrail an earlier one attached, and the
                # tasks then run with no screening.
                Annotations.of(self).add_warning(
                    "The guardrail this stack owns is not attached to compute: "
                    "bedrock_region differs from the deploy region, so the id "
                    "reaches the tasks only through context and this synth "
                    "gives them no guardrail; no query or answer is screened "
                    "(a deploy of it detaches one an earlier deploy attached). "
                    "Pass `-c guardrail_identifier=<GuardrailIdentifier output> "
                    "-c guardrail_version=<GuardrailVersion output>` on every "
                    "deploy, or keep both in the context of cdk.json; "
                    "`-c create_guardrail=false` creates none."
                )
        else:
            # Bring-your-own path: nothing is created; surface the external id.
            self.guardrail_identifier = config.guardrail_identifier
        if self.guardrail_identifier:
            CfnOutput(self, "GuardrailIdentifier", value=self.guardrail_identifier)

    def _attach_owned(self) -> GuardrailAttachment:
        """Hand the owned guardrail to compute through SSM parameters.

        Read as dynamic references when a task definition deploys, so no
        export pins the version (see the module docstring). An explicit
        ``guardrail_version`` still pins the version the tasks apply.
        """
        assert self.guardrail is not None and self.guardrail_version is not None
        values = {
            "identifier": self.guardrail.attr_guardrail_id,
            "version": self.guardrail_version.attr_version,
        }
        references = {}
        for name, value in values.items():
            # The literal name, not the parameter's Ref: a Ref read by another
            # stack becomes an export.
            parameter_name = f"/{self.config.prefix}/guardrail/{name}"
            ssm.StringParameter(
                self,
                f"Attached{name.capitalize()}",
                parameter_name=parameter_name,
                string_value=value,
                description=f"Guardrail {name} the compute tasks apply",
            )
            references[name] = CfnDynamicReference(
                CfnDynamicReferenceService.SSM, parameter_name
            ).to_string()
        return GuardrailAttachment(
            identifier=references["identifier"],
            version=self.config.guardrail_version or references["version"],
            source=self,
            revision=self.policy_digest,
        )

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
                    # NAME is deliberately absent. The app screens the user's
                    # query and the answer, where person names are usually
                    # corpus content: anonymizing them turns "Who is <name>?"
                    # into "{NAME}" before retrieval (no entity seed to look
                    # up) and answers into "{NAME}". Add it only for a corpus
                    # whose names must not reach users, accepting that
                    # retrieval by name stops working.
                    # Both directions: `action` alone masks only the
                    # answer (OUTPUT), so check_input passed a query's PII
                    # through to retrieval, memory and the model.
                    pii_entities_config=[
                        bedrock.CfnGuardrail.PiiEntityConfigProperty(
                            type=t,
                            action="ANONYMIZE",
                            input_action="ANONYMIZE",
                            input_enabled=True,
                            output_action="ANONYMIZE",
                            output_enabled=True,
                        )
                        for t in PII_ANONYMIZED
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
        if self.config.guardrail_contextual_grounding:
            guardrail.contextual_grounding_policy_config = (
                self._contextual_grounding_policy()
            )
        # RETAIN outside dev: the id is wired to compute out-of-band, so a
        # CloudFormation-side delete (template drift, a mistaken synth,
        # `cdk destroy`) would silently leave compute pointing at a dead
        # guardrail. With removal_destroy (dev default) keep DESTROY: the name is
        # fixed, so a retained guardrail would make the next deploy after
        # `cdk destroy` fail on the duplicate name.
        guardrail.apply_removal_policy(
            RemovalPolicy.DESTROY
            if self.config.removal_destroy
            else RemovalPolicy.RETAIN
        )
        return guardrail

    def _contextual_grounding_policy(
        self,
    ) -> bedrock.CfnGuardrail.ContextualGroundingPolicyConfigProperty:
        """Block answers scoring below the threshold on either check (the app
        sends the retrieved context and the question with each answer)."""
        return bedrock.CfnGuardrail.ContextualGroundingPolicyConfigProperty(
            filters_config=[
                bedrock.CfnGuardrail.ContextualGroundingFilterConfigProperty(
                    type=kind,
                    threshold=self.config.guardrail_grounding_threshold,
                    action="BLOCK",
                    enabled=True,
                )
                for kind in ("GROUNDING", "RELEVANCE")
            ]
        )

    def _publish_version(
        self, guardrail: bedrock.CfnGuardrail
    ) -> bedrock.CfnGuardrailVersion:
        """An immutable version of the guardrail's current policy.

        A version is published when the resource is created or replaced, and
        a description change replaces it: naming the policy's digest there
        publishes a new version whenever the policy changes. Retained, so
        the replacement leaves the previous version in place for a task
        still pinned to it (the versions go when the guardrail is deleted).
        """
        policy = self.resolve(
            [
                getattr(guardrail, name)
                for name in (
                    "blocked_input_messaging",
                    "blocked_outputs_messaging",
                    "content_policy_config",
                    "contextual_grounding_policy_config",
                    "sensitive_information_policy_config",
                    "topic_policy_config",
                    "word_policy_config",
                )
            ]
        )
        digest = hashlib.sha256(
            json.dumps(policy, sort_keys=True).encode()
        ).hexdigest()[:16]
        self.policy_digest = digest
        version = bedrock.CfnGuardrailVersion(
            self,
            "PublishedVersion",
            guardrail_identifier=guardrail.attr_guardrail_id,
            description=f"Policy {digest}",
        )
        version.apply_removal_policy(RemovalPolicy.RETAIN)
        return version
