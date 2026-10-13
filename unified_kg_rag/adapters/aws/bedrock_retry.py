# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which Bedrock runtime errors are transient.

botocore's ``standard``/``adaptive`` retry modes only retry throttling, a fixed
set of 5xx codes, and connection-level errors. Bedrock surfaces several
transient model-side failures outside that set — most notably
``ModelErrorException`` (HTTP 424, "The system encountered an unexpected
error"), ``ModelNotReadyException`` and ``ModelTimeoutException`` (HTTP 408,
the model took too long) — which botocore raises on the first occurrence. :func:`is_transient_bedrock_error` names exactly those failures, for
the bounded backoff of ``adapters.llm.retry`` and the batch processor's
retries, leaving validation, access-denied and other client errors to fail
fast.
"""

from __future__ import annotations

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
