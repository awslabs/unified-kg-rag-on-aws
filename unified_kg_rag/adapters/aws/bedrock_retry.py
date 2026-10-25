# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which Bedrock runtime errors are transient, and which are an item's own.

botocore's ``standard``/``adaptive`` retry modes only retry throttling, a fixed
set of 5xx codes, and connection-level errors. Bedrock surfaces several
transient model-side failures outside that set — most notably
``ModelErrorException`` (HTTP 424, "The system encountered an unexpected
error"), ``ModelNotReadyException`` and ``ModelTimeoutException`` (HTTP 408,
the model took too long) — which botocore raises on the first occurrence. :func:`is_transient_bedrock_error` names exactly those failures, for
the bounded backoff of ``adapters.llm.retry`` and the batch processor's
retries, leaving validation, access-denied and other client errors to fail
fast. :func:`is_bedrock_item_error` names the errors an ingestion item causes
itself (a ``ValidationException`` of its request), which count against its
document.
"""

from __future__ import annotations

import re

from botocore import exceptions as boto_exc

# Error codes Bedrock runtime returns for conditions that resolve on their own.
# ThrottlingException / ServiceUnavailableException / InternalServerException
# are normally absorbed by botocore's own retries; they are listed so a burst
# that outlasts botocore's attempts still gets the backoff.
TRANSIENT_BEDROCK_ERROR_CODES: frozenset[str] = frozenset(
    {
        "ModelErrorException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ThrottlingException",
    }
)


def is_transient_bedrock_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a Bedrock/botocore failure worth retrying."""
    if isinstance(exc, boto_exc.ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in TRANSIENT_BEDROCK_ERROR_CODES
    # Connect/read timeouts, endpoint-unreachable and dropped connections.
    return isinstance(exc, boto_exc.ConnectionError | boto_exc.HTTPClientError)


# ValidationExceptions of the model or the request settings, not of the
# item's text: they fail every item alike until the configuration is fixed.
_CONFIGURATION_VALIDATION = re.compile(
    r"model identifier is invalid|on-demand throughput|inference profile"
    r"|not permitted|not supported|isn't supported|unsupported"
    r"|does(n't| not) support|malformed|guardrail|end of (its )?life|legacy"
    r"|access|maximum tokens|max_tokens\W{0,3}\d+ ?>|cannot both|extraneous key",
    re.IGNORECASE,
)


def is_bedrock_item_error(exc: BaseException) -> bool:
    """Return True if Bedrock rejected the item's own request.

    A ``ValidationException`` (such as an input too long for the model) other
    than one of the model or the request settings (an invalid model id, a model
    that needs an inference profile, an unsupported parameter). Nothing else
    counts: ``ServiceQuotaExceededException`` (HTTP 400 on InvokeModel),
    throttling, stream and model errors, access denied, and botocore's
    credential, region and parameter errors say nothing about the item.
    """
    if not isinstance(exc, boto_exc.ClientError):
        return False
    error = exc.response.get("Error", {})
    return error.get("Code") == "ValidationException" and not (
        _CONFIGURATION_VALIDATION.search(str(error.get("Message", "")))
    )
