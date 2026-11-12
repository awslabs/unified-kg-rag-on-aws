# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Count the Bedrock runtime requests that were throttled.

botocore retries a throttled request inside one call (``adaptive`` mode,
``BaseBedrockModelFactory.BOTO_RETRY_MODE``), so the call that returns at
last reports only how many requests it resent (``RetryAttempts``), not why.
:func:`count_throttled_requests` attaches two botocore event handlers to a
runtime client: one remembers the call's ``modelId``, the other counts every
attempt that came back throttled (HTTP 429 or a throttling error code) in the
open ``shared.usage`` scopes, under the client's role and that model, as
``throttled_requests``. The handlers run in the calling thread, whose context
holds the scopes (as the usage callback's do). A throttling error raised
inside an event stream, after the response started, is not a request botocore
retries and is not counted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unified_kg_rag.shared import usage

__all__ = ["THROTTLING_ERROR_CODES", "count_throttled_requests"]

# The error codes Bedrock runtime throttles with (HTTP 429).
THROTTLING_ERROR_CODES: frozenset[str] = frozenset(
    {"ThrottlingException", "TooManyRequestsException"}
)

# The request-context key the call's model id is kept under, from the
# parameters to the attempts.
_MODEL_ID_KEY = "kg_rag_model_id"
_UNKNOWN_MODEL = "unknown"


def _remember_model_id(params: Any = None, context: Any = None, **_: Any) -> None:
    """``before-parameter-build``: keep the call's ``modelId`` in its request
    context, which botocore hands to each attempt's ``needs-retry``."""
    if isinstance(params, Mapping) and isinstance(context, dict):
        model_id = params.get("modelId")
        if model_id:
            context[_MODEL_ID_KEY] = str(model_id)


def _throttled(response: Any) -> bool:
    """Whether an attempt's ``(http_response, parsed)`` was throttled."""
    if not isinstance(response, tuple) or len(response) != 2:
        return False
    http_response, parsed = response
    if getattr(http_response, "status_code", None) == 429:
        return True
    error = parsed.get("Error") if isinstance(parsed, Mapping) else None
    return isinstance(error, Mapping) and error.get("Code") in THROTTLING_ERROR_CODES


def count_throttled_requests(client: Any, service: str, role: str) -> None:
    """Count the throttled attempts of ``client`` (of botocore ``service``,
    ``bedrock-runtime``) in the open usage scopes, under ``role`` (the
    purpose its calls are made for) and the call's model id."""

    def on_attempt(response: Any = None, request_dict: Any = None, **_: Any) -> None:
        if not _throttled(response):
            return
        context = (
            request_dict.get("context") if isinstance(request_dict, Mapping) else None
        )
        model_id = (
            context.get(_MODEL_ID_KEY, _UNKNOWN_MODEL)
            if isinstance(context, Mapping)
            else _UNKNOWN_MODEL
        )
        usage.record_throttled_request(role, model_id)

    client.meta.events.register(f"before-parameter-build.{service}", _remember_model_id)
    client.meta.events.register(f"needs-retry.{service}", on_attempt)
