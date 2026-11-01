# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Amazon Bedrock Guardrails as a :class:`ContentGuardPort`.

Screens a text with the ``ApplyGuardrail`` API, independently of any model
call: the raw user query with ``source=INPUT`` and the final answer with
``source=OUTPUT``. Only that text is evaluated (and billed), never the
retrieved context or the developer prompt around it, so a prompt-attack
filter judges what the user typed and a PII filter masks the user's data, not
the corpus.

With ``aws.bedrock.guardrail.contextual_grounding`` on, the answer's check
also sends the context the answer model was given (qualifier
``grounding_source``) and the question it answered (``query``), which only the
contextual grounding policy reads. Bedrock limits these to 100,000 and 1,000
characters (a longer context keeps its highest-ranked sections, a longer
question its head) and the answer to 5,000: a longer answer is screened
without the grounding check, which is logged and recorded as skipped.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, ClassVar

from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from unified_kg_rag.adapters.aws.bedrock import get_assumed_role_boto_session
from unified_kg_rag.ports.guardrail import (
    GroundingCheck,
    GroundingContext,
    GuardAction,
    GuardSource,
    GuardVerdict,
)
from unified_kg_rag.shared import GuardrailError, get_logger, observability
from unified_kg_rag.shared.genai import (
    GUARDRAIL_ACTION,
    GUARDRAIL_GROUNDING_CHECKED,
    GUARDRAIL_GROUNDING_FAILED,
    GUARDRAIL_GROUNDING_SCORE,
    GUARDRAIL_ID,
    GUARDRAIL_INTERVENED,
    GUARDRAIL_POLICIES,
    GUARDRAIL_RELEVANCE_SCORE,
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

# Bedrock's contextual grounding limits, in characters.
GROUNDING_SOURCE_MAX_CHARS = 100_000
GROUNDING_QUERY_MAX_CHARS = 1_000
GROUNDING_ANSWER_MAX_CHARS = 5_000
_GROUNDING_POLICY = "contextualGroundingPolicy"


def select_grounding_source(context: GroundingContext) -> str:
    """The context as the answer model saw it, or, above Bedrock's limit, its
    highest-ranked sections: whole ones in rank order while they fit, then
    the head of the next (the context's own head without ranked sections)."""
    limit = GROUNDING_SOURCE_MAX_CHARS
    if len(context.source) <= limit:
        return context.source
    parts: list[str] = []
    used = 0
    for section in context.ranked_sources:
        room = limit - used - (1 if parts else 0)
        if room <= 0:
            break
        parts.append(section[:room])  # a cut section fills the limit
        used = len("\n".join(parts))
    return "\n".join(parts) if parts else context.source[:limit]


def grounding_from_response(response: dict[str, Any]) -> GroundingCheck:
    """The contextual grounding scores in an ApplyGuardrail ``response``.

    A check fails when Bedrock detected it (score below the threshold) or
    blocked for it, so a detect-only policy still reports the failure; the
    answer was blocked for it when the guardrail intervened with a blocking
    entry. No entry at all means the guardrail has no contextual grounding
    policy.
    """
    check = GroundingCheck(checked=False, skipped_reason="not_assessed")
    intervened = response.get("action") == _INTERVENED
    for assessment in response.get("assessments") or []:
        policy = assessment.get(_GROUNDING_POLICY) or {}
        for entry in policy.get("filters") or []:
            name = str(entry.get("type", "unknown")).lower()
            check.checked, check.skipped_reason = True, None
            if "score" in entry:
                check.scores[name] = float(entry["score"])
            if entry.get("detected") or entry.get("action") == _BLOCKED:
                check.failed.append(name)
            if intervened and entry.get("action") == _BLOCKED:
                check.blocked = True
    return check


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


def verdict_from_response(
    text: str, response: dict[str, Any], *, first_output_only: bool = False
) -> GuardVerdict:
    """The verdict an ApplyGuardrail ``response`` gives on ``text``.

    An intervention counts as masking only when every acting entry
    anonymized; anything else (a block, or an intervention whose assessment
    names no entry) is a block, so an unrecognized reply fails closed.
    ``first_output_only``: ``text`` was the first of several content blocks
    (the grounding reference and query follow), so only the first output can
    be its replacement; a reply mirroring every block must not hand the
    reference back as the answer.
    """
    if response.get("action") != _INTERVENED:
        return GuardVerdict(action=GuardAction.NONE, text=text)
    outputs = [o for o in response.get("outputs") or [] if isinstance(o, dict)]
    if first_output_only:
        outputs = outputs[:1]
    replaced = "".join(str(o.get("text", "")) for o in outputs)
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
    # Answers whose grounding check did not pass, per outcome.
    _grounding_counts: ClassVar[dict[str, int]] = {}
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

    async def acheck(
        self,
        text: str,
        source: GuardSource,
        *,
        grounding: GroundingContext | None = None,
    ) -> GuardVerdict:
        # Bedrock runs the grounding check on output only; an answer over its
        # limit is screened by the other policies alone.
        if source is not GuardSource.OUTPUT:
            grounding = None
        too_long = grounding is not None and len(text) > GROUNDING_ANSWER_MAX_CHARS
        if too_long:
            grounding = None
        with observability.span(
            "kg_rag.guardrail",
            {GUARDRAIL_ID: self.identifier, GUARDRAIL_SOURCE: source.value},
        ) as span:
            content: list[dict[str, Any]] = [{"text": {"text": text}}]
            if grounding is not None:
                # The answer stays the first block: the first output is the
                # answer whether a masked reply mirrors one block or all three.
                content = [
                    {"text": {"text": text, "qualifiers": ["guard_content"]}},
                    {
                        "text": {
                            "text": select_grounding_source(grounding),
                            "qualifiers": ["grounding_source"],
                        }
                    },
                    {
                        "text": {
                            "text": grounding.query[:GROUNDING_QUERY_MAX_CHARS],
                            "qualifiers": ["query"],
                        }
                    },
                ]
            request: dict[str, Any] = {
                "guardrailIdentifier": self.identifier,
                "guardrailVersion": self.version,
                "source": source.value.upper(),
                "content": content,
            }
            # A passing grounding score is only reported in the full output.
            if self.full_output or grounding is not None:
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
            verdict = verdict_from_response(
                text, response, first_output_only=grounding is not None
            )
            if grounding is not None:
                verdict.grounding = grounding_from_response(response)
            elif too_long:
                verdict.grounding = GroundingCheck(
                    checked=False, skipped_reason="answer_too_long"
                )
            span.set_attributes(
                {
                    GUARDRAIL_ACTION: verdict.action.value,
                    GUARDRAIL_INTERVENED: verdict.intervened,
                    GUARDRAIL_POLICIES: verdict.policies,
                    **_grounding_attributes(verdict.grounding),
                }
            )
        if verdict.grounding is not None:
            self._record_grounding(verdict, len(text))
        if verdict.intervened:
            self._record(source, verdict)
        return verdict

    def _record_grounding(self, verdict: GuardVerdict, answer_chars: int) -> None:
        """WARNING, with a process-wide count, for an answer the grounding
        check blocked, failed without blocking (detect-only) or skipped."""
        check = verdict.grounding
        assert check is not None
        if not check.checked:
            outcome = (
                f"skipped: {answer_chars} characters exceed the "
                f"{GROUNDING_ANSWER_MAX_CHARS}-character limit"
                if check.skipped_reason == "answer_too_long"
                else "skipped: the guardrail has no contextual grounding policy"
            )
        elif not check.failed:
            return
        elif check.blocked:
            outcome = "blocked"
        else:
            outcome = "failed but not blocked (detect-only policy)"
        key = outcome.split(":")[0].split(" ")[0]
        with self._lock:
            counts = type(self)._grounding_counts
            counts[key] = counts.get(key, 0) + 1
            total = counts[key]
        logger.warning(
            "Contextual grounding check of guardrail '%s' on an answer %s "
            "(failed: %s; scores: %s; %s answers in this process: %d)",
            self.identifier,
            outcome,
            ", ".join(check.failed) or "none",
            ", ".join(f"{k} {v:.2f}" for k, v in sorted(check.scores.items()))
            or "none reported",
            key,
            total,
        )

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


def _grounding_attributes(check: GroundingCheck | None) -> dict[str, Any]:
    """The span attributes of a contextual grounding check (none without one)."""
    if check is None:
        return {}
    attributes: dict[str, Any] = {
        GUARDRAIL_GROUNDING_CHECKED: check.checked,
        GUARDRAIL_GROUNDING_FAILED: check.failed,
    }
    for name, key in (
        ("grounding", GUARDRAIL_GROUNDING_SCORE),
        ("relevance", GUARDRAIL_RELEVANCE_SCORE),
    ):
        if name in check.scores:
            attributes[key] = check.scores[name]
    return attributes
