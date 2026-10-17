# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Amazon Bedrock Guardrails as a :class:`ContentGuardPort`.

Screens a text with the ``ApplyGuardrail`` API, independently of any model
call: the raw user query with ``source=INPUT`` and the final answer with
``source=OUTPUT``. Only that text is evaluated (and billed), never the
retrieved context or the developer prompt around it, so a prompt-attack
filter judges what the user typed and a PII filter masks the user's data, not
the corpus.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, ClassVar

from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from unified_kg_rag.adapters.aws.bedrock import get_assumed_role_boto_session
from unified_kg_rag.ports.guardrail import GuardAction, GuardSource, GuardVerdict
from unified_kg_rag.shared import GuardrailError, get_logger, observability
from unified_kg_rag.shared.genai import (
    GUARDRAIL_ACTION,
    GUARDRAIL_ID,
    GUARDRAIL_INTERVENED,
    GUARDRAIL_POLICIES,
    GUARDRAIL_SOURCE,
)

if TYPE_CHECKING:
    import boto3

    from unified_kg_rag.domain.models import Config

logger = get_logger(__name__)

_INTERVENED = "GUARDRAIL_INTERVENED"
# Assessment entry actions that changed the text. Detect-mode entries
# ("NONE" with detected=true) did not.
_BLOCKED = "BLOCKED"
_ANONYMIZED = "ANONYMIZED"
# The answer to a blocked text when the guardrail returned no blocked message.
DEFAULT_BLOCKED_MESSAGE = "Sorry, I can't help with that request."


def _acting_policies(assessments: list[dict[str, Any]]) -> tuple[list[str], bool]:
    """``(policies, any_blocked)`` from an ApplyGuardrail ``assessments`` list.

    Each policy (``topicPolicy``, ``contentPolicy``, ...) holds lists of
    entries with an ``action``; an entry is named by its ``name`` (topics,
    regexes) or else its ``type`` (filters, PII types).
    """
    policies: list[str] = []
    blocked = False
    for assessment in assessments:
        for policy_name, policy in assessment.items():
            if not policy_name.endswith("Policy") or not isinstance(policy, dict):
                continue
            for entries in policy.values():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    action = entry.get("action") if isinstance(entry, dict) else None
                    if action not in (_BLOCKED, _ANONYMIZED):
                        continue
                    blocked = blocked or action == _BLOCKED
                    label = entry.get("name") or entry.get("type") or "unknown"
                    policies.append(f"{policy_name}:{label}")
    return sorted(set(policies)), blocked


def verdict_from_response(text: str, response: dict[str, Any]) -> GuardVerdict:
    """The verdict an ApplyGuardrail ``response`` gives on ``text``.

    An intervention counts as masking only when every acting entry
    anonymized; anything else (a block, or an intervention whose assessment
    names no entry) is a block, so an unrecognized reply fails closed.
    """
    if response.get("action") != _INTERVENED:
        return GuardVerdict(action=GuardAction.NONE, text=text)
    outputs = response.get("outputs") or []
    replaced = "".join(str(o.get("text", "")) for o in outputs if isinstance(o, dict))
    policies, blocked = _acting_policies(response.get("assessments") or [])
    if blocked or not policies:
        return GuardVerdict(
            action=GuardAction.BLOCKED,
            text=replaced or DEFAULT_BLOCKED_MESSAGE,
            policies=policies,
        )
    return GuardVerdict(
        action=GuardAction.MASKED, text=replaced or text, policies=policies
    )


class BedrockGuardrail:
    """Screens texts with one Bedrock guardrail (``aws.bedrock.guardrail``).

    Fails closed: an API error raises :class:`GuardrailError` instead of
    letting the text through. Each intervention is logged at WARNING with a
    process-wide running count, and each check is a ``kg_rag.guardrail`` span
    (guardrail id, source, action, policies) when tracing is on.
    """

    # A screening call is small and must not hold a user query for long.
    BOTO_CONNECT_TIMEOUT: ClassVar[int] = 10
    BOTO_READ_TIMEOUT: ClassVar[int] = 30

    _count: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        *,
        client: Any = None,
    ) -> None:
        guardrail = config.aws.bedrock.guardrail
        if not guardrail.identifier:
            raise GuardrailError("aws.bedrock.guardrail.identifier is not set")
        self.identifier = guardrail.identifier
        self.version = guardrail.version
        self.full_output = guardrail.trace
        if client is None:
            import boto3

            session = get_assumed_role_boto_session(
                boto_session or boto3.Session(profile_name=config.aws.profile_name),
                assumed_role_arn=config.aws.bedrock.assumed_role_arn,
            )
            # The guardrail lives in the Bedrock runtime region.
            client = session.client(
                "bedrock-runtime",
                region_name=config.aws.bedrock_region,
                config=BotoConfig(
                    connect_timeout=self.BOTO_CONNECT_TIMEOUT,
                    read_timeout=self.BOTO_READ_TIMEOUT,
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        self._client = client

    async def acheck(self, text: str, source: GuardSource) -> GuardVerdict:
        with observability.span(
            "kg_rag.guardrail",
            {GUARDRAIL_ID: self.identifier, GUARDRAIL_SOURCE: source.value},
        ) as span:
            request: dict[str, Any] = {
                "guardrailIdentifier": self.identifier,
                "guardrailVersion": self.version,
                "source": source.value.upper(),
                "content": [{"text": {"text": text}}],
            }
            if self.full_output:
                request["outputScope"] = "FULL"
            try:
                response = await asyncio.to_thread(
                    self._client.apply_guardrail, **request
                )
            except (ClientError, BotoCoreError) as e:
                raise GuardrailError(
                    f"Bedrock guardrail '{self.identifier}' could not screen the "
                    f"{source.value}: {e}"
                ) from e
            verdict = verdict_from_response(text, response)
            span.set_attributes(
                {
                    GUARDRAIL_ACTION: verdict.action.value,
                    GUARDRAIL_INTERVENED: verdict.intervened,
                    GUARDRAIL_POLICIES: verdict.policies,
                }
            )
        if verdict.intervened:
            self._record(source, verdict)
        return verdict

    def _record(self, source: GuardSource, verdict: GuardVerdict) -> None:
        with self._lock:
            type(self)._count += 1
            total = type(self)._count
        logger.warning(
            "Bedrock guardrail '%s' %s the %s (policies: %s; interventions in "
            "this process: %d)",
            self.identifier,
            verdict.action.value,
            source.value,
            ", ".join(verdict.policies) or "none reported",
            total,
        )
