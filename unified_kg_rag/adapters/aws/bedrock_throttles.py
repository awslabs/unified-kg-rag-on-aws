# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Count the Bedrock runtime requests sent, and those throttled.

botocore retries a throttled request inside one call (``adaptive`` mode,
``BaseBedrockModelFactory.BOTO_RETRY_MODE``), so the call that returns at
last reports only how many requests it resent (``RetryAttempts``), not why.
:func:`count_requests` attaches two botocore event handlers to a runtime
client: one remembers the call's model, the other counts every attempt
(``needs-retry`` fires once per attempt, the last one included) in the open
``shared.usage`` scopes, under the client's role and that model, as
``sent_requests``, and the attempts that came back throttled (HTTP 429 or a
throttling error code) as ``throttled_requests`` too. Every Bedrock runtime
client is counted (the language, embedding and rerank models'), so the
throttled share has one denominator whatever the model. The handlers run in
the calling thread, whose context holds the scopes (as the usage callback's
do). A throttling error raised inside an event stream, after the response
started, is not a request botocore retries and is not counted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unified_kg_rag.adapters.aws.bedrock_retry import THROTTLING_ERROR_CODES
from unified_kg_rag.shared import usage

__all__ = ["count_requests"]

# The request-context key the call's model id is kept under, from the
# parameters to the attempts.
_MODEL_ID_KEY = "kg_rag_model_id"
_UNKNOWN_MODEL = "unknown"


def _model_id(params: Mapping[str, Any]) -> str | None:
    """The model a call's parameters name: ``modelId`` (InvokeModel,
    Converse), else the rerank model's ARN, by its last path segment (the
    model id the rerank usage is recorded under)."""
    if model_id := params.get("modelId"):
        return str(model_id)
    configuration = params.get("rerankingConfiguration")
    if isinstance(configuration, Mapping):
        model = (
            (configuration.get("bedrockRerankingConfiguration") or {}).get(
                "modelConfiguration"
            )
            or {}
        ).get("modelArn")
        if model:
            return str(model).rsplit("/", 1)[-1]
    return None


def _remember_model_id(params: Any = None, context: Any = None, **_: Any) -> None:
    """``before-parameter-build``: keep the call's model id in its request
    context, which botocore hands to each attempt's ``needs-retry``."""
    if isinstance(params, Mapping) and isinstance(context, dict):
        if model_id := _model_id(params):
            context[_MODEL_ID_KEY] = model_id


def _throttled(response: Any) -> bool:
    """Whether an attempt's ``(http_response, parsed)`` was throttled."""
    if not isinstance(response, tuple) or len(response) != 2:
        return False
    http_response, parsed = response
    if getattr(http_response, "status_code", None) == 429:
        return True
    error = parsed.get("Error") if isinstance(parsed, Mapping) else None
    return isinstance(error, Mapping) and error.get("Code") in THROTTLING_ERROR_CODES


def count_requests(client: Any, service: str, role: str) -> None:
    """Count the attempts of ``client`` (of botocore ``service``:
    ``bedrock-runtime``, ``bedrock-agent-runtime``), and the throttled ones,
    in the open usage scopes, under ``role`` (the purpose its calls are made
    for) and the call's model id."""

    def on_attempt(response: Any = None, request_dict: Any = None, **_: Any) -> None:
        context = (
            request_dict.get("context") if isinstance(request_dict, Mapping) else None
        )
        model_id = (
            context.get(_MODEL_ID_KEY, _UNKNOWN_MODEL)
            if isinstance(context, Mapping)
            else _UNKNOWN_MODEL
        )
        usage.record_request(role, model_id, throttled=_throttled(response))

    client.meta.events.register(f"before-parameter-build.{service}", _remember_model_id)
    client.meta.events.register(f"needs-retry.{service}", on_attempt)
