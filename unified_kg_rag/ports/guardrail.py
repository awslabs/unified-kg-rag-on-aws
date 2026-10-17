# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-guard port — screening of the user's query and the final answer.

The query chain checks the raw user query once, before memory, rewrite,
routing and retrieval, and the final answer once, after generation. It never
guards the internal model calls in between: their prompts carry retrieved
corpus text and developer instructions, which a prompt-attack or PII filter
would judge as if the user had typed them. ``adapters.guardrail.bedrock``
implements the port with the Amazon Bedrock ``ApplyGuardrail`` API.

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

    @property
    def intervened(self) -> bool:
        return self.action is not GuardAction.NONE


@runtime_checkable
class ContentGuardPort(Protocol):
    """Screens one text from the user (input) or for the user (output)."""

    async def acheck(self, text: str, source: GuardSource) -> GuardVerdict:
        """The verdict on ``text``; raises if the guard could not decide."""
        ...
