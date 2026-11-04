# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM usage records (what ``shared.usage`` scopes count)."""

from pydantic import BaseModel, Field


class LLMUsageCounts(BaseModel):
    llm_calls: int = Field(
        default=0,
        description="Model calls made: those that returned and those cancelled in flight.",
    )
    input_tokens: int = Field(
        default=0,
        description=(
            "Input tokens the calls reported; a cancelled call's are estimated "
            "from its prompt."
        ),
    )
    output_tokens: int = Field(
        default=0,
        description="Output tokens they reported (none for a cancelled call).",
    )
    cache_hits: int = Field(
        default=0,
        description=(
            "Responses the ingestion LLM response cache served instead of a "
            "call (no tokens spent)."
        ),
    )
    estimated_calls: int = Field(
        default=0,
        description=(
            "Of llm_calls, those that had reported no usage when this was "
            "read: still running in the background (an auto query's unread "
            "speculative extraction) or cancelled in flight (a query "
            "deadline). The request was sent, so it is billed; their input "
            "tokens are estimated from the prompt and their output tokens 0."
        ),
    )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMUsage(LLMUsageCounts):
    """The model calls of one query, stage or run, with their breakdowns."""

    by_role: dict[str, LLMUsageCounts] = Field(
        default_factory=dict,
        description=(
            "Per model role (the purpose the model was built for: query, "
            "ingestion, evaluation)."
        ),
    )
    by_model: dict[str, LLMUsageCounts] = Field(
        default_factory=dict,
        description="Per model id (as the model factory resolved it).",
    )
