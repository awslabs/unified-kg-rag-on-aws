# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-guard port — screening of the user's query and the final answer.

The query chain checks the raw user query once, before memory, rewrite,
routing and retrieval, and the final answer once, after generation. It never
guards the internal model calls in between: their prompts carry retrieved
corpus text and developer instructions, which a prompt-attack or PII filter
would judge as if the user had typed them. ``adapters.guardrail.bedrock``
implements the port with the Amazon Bedrock ``ApplyGuardrail`` API.

With a contextual grounding check on, the answer's check also receives a
:class:`GroundingContext`: the question the answer model answered and the
context it was given, so the guard can judge whether the answer is grounded
in that context and relevant to the question.

The verdict records belong to the contract, so they live here.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class GuardSource(str, Enum):
    """Which side of the conversation a text comes from."""

    INPUT = "input"
    OUTPUT = "output"


class GuardAction(str, Enum):
    """What the guard did with a text."""

    NONE = "none"
    # Allowed, with sensitive spans replaced (e.g. PII anonymized).
    MASKED = "masked"
    # Refused; ``GuardVerdict.text`` is the configured refusal message.
    BLOCKED = "blocked"


class GroundingContext(BaseModel):
    """What the answer model was given: the reference an answer is checked
    against by a contextual grounding check."""

    query: str = Field(description="The question the answer model answered")
    source: str = Field(
        description="The retrieved context exactly as the answer model saw it"
    )
    ranked_sources: list[str] = Field(
        default_factory=list,
        description="The same context as one entry per section, highest-ranked "
        "first; a guard whose reference has a size limit selects from these "
        "when source exceeds it",
    )


class GroundingCheck(BaseModel):
    """The outcome of a contextual grounding check on one answer."""

    checked: bool = Field(
        description="Whether the check ran; False when it was skipped "
        "(skipped_reason says why)"
    )
    skipped_reason: str | None = Field(
        default=None,
        description="Why the check did not run: 'answer_too_long' or "
        "'not_assessed' (the guardrail has no contextual grounding policy)",
    )
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Score per check, e.g. {'grounding': 0.91, 'relevance': 0.42}",
    )
    failed: list[str] = Field(
        default_factory=list,
        description="Checks whose score fell below their threshold, whether "
        "or not the guard blocked the answer for it",
    )
    blocked: bool = Field(
        default=False,
        description="Whether the guard blocked the answer for a failed check "
        "(False for a detect-only failure)",
    )


class GuardVerdict(BaseModel):
    """The guard's decision on one text."""

    action: GuardAction
    text: str = Field(
        description="The text to continue with: the input unchanged (none), "
        "the masked text (masked) or the refusal message (blocked)"
    )
    policies: list[str] = Field(
        default_factory=list,
        description="Policies that acted, e.g. 'contentPolicy:PROMPT_ATTACK' "
        "or 'sensitiveInformationPolicy:EMAIL'",
    )
    grounding: GroundingCheck | None = Field(
        default=None,
        description="The contextual grounding check on an answer; None when "
        "none was requested",
    )

    @property
    def intervened(self) -> bool:
        return self.action is not GuardAction.NONE


@runtime_checkable
class ContentGuardPort(Protocol):
    """Screens one text from the user (input) or for the user (output)."""

    async def acheck(
        self,
        text: str,
        source: GuardSource,
        *,
        grounding: GroundingContext | None = None,
    ) -> GuardVerdict:
        """The verdict on ``text``; raises if the guard could not decide.

        ``grounding`` is passed only for an answer (``OUTPUT``) and only while
        a contextual grounding check is configured
        (``aws.bedrock.guardrail.contextual_grounding.enabled``), so a guard
        that implements no such check is never handed one by default.
        """
        ...
