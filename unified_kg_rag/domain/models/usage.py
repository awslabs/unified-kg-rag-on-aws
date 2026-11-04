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
            "Input tokens the calls reported, prompt-cache reads and writes "
            "included; a cancelled call's are estimated from its prompt."
        ),
    )
    output_tokens: int = Field(
        default=0,
        description="Output tokens they reported (none for a cancelled call).",
    )
    cache_read_input_tokens: int = Field(
        default=0,
        description=(
            "Of input_tokens, those read from the provider's prompt cache "
            "(Bedrock cacheReadInputTokens), billed at a lower rate."
        ),
    )
    cache_write_input_tokens: int = Field(
        default=0,
        description=(
            "Of input_tokens, those written to the provider's prompt cache "
            "(Bedrock cacheWriteInputTokens), billed at a higher rate."
        ),
    )
    request_retries: int = Field(
        default=0,
        description=(
            "Requests the SDK resent within the calls (botocore's "
            "ResponseMetadata.RetryAttempts, failed calls included): each "
            "call counts once in llm_calls however many requests it took. A "
            "resent throttled request is not a Bedrock invocation; one "
            "resent after a read timeout or a server error is, and may be "
            "billed."
        ),
    )
    failed_calls: int = Field(
        default=0,
        description=(
            "Calls that ended in an error and reported no usage; not in "
            "llm_calls. A request Bedrock accepted and then failed still "
            "counts as an invocation in CloudWatch."
        ),
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
