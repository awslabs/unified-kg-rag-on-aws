# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which retrieval failures fail the query and which degrade it to no results.

One policy for every store and model call a query makes. The store clients
translate their failures into typed errors at the boundary (``aws.opensearch``,
``aws.neptune``): ``StoreConnectionError`` for a store that cannot be reached
or keeps failing after the client's retries (``StoreTimeoutError`` when it did
not answer in time) and ``StoreConfigurationError`` for one that refuses the
client the same way on every attempt (``StoreAccessDeniedError`` for HTTP
401/403, ``StoreRequestRejectedError`` for a 400). The classification here is
then by type alone; no error message is matched.
"""

import sys
from collections.abc import Iterator

from unified_kg_rag.shared import (
    ConfigurationError,
    IndexNotFoundError,
    InvalidFilterError,
)

# Errors that fail every later query the same way: a store that cannot be
# reached or keeps failing, a missing suffix, a configuration the store or the
# query refuses, a filter key no index declares.
_FATAL_TYPES = (
    ConnectionError,
    ConfigurationError,
    IndexNotFoundError,
    InvalidFilterError,
)


def causes(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and the exceptions it was raised from, innermost last."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_fatal_retrieval_error(exc: BaseException) -> bool:
    """Whether ``exc`` must fail the query rather than degrade it to ``[]``.

    Returning no results for a failure that every later query repeats turns a
    broken deployment into a plausible "no information found". Fatal:

    - ``ConnectionError`` (``StoreConnectionError``, ``StoreTimeoutError``): a
      store that cannot be reached, lost the connection, did not answer within
      ``search.store_timeout_seconds`` or kept answering 429/5xx, after the
      client's retries.
    - ``ConfigurationError`` (``StoreConfigurationError`` and its denied and
      rejected kinds, an embedding model the index was not built with),
      ``IndexNotFoundError`` and ``InvalidFilterError``.
    - A botocore ``ClientError`` (a Bedrock call) unless its code is one of the
      transient ones (``is_transient_bedrock_error``): an invalid or
      inaccessible model id, a model in another region, denied access.
    - Any other botocore error: no credentials, no region, an endpoint that
      could not be reached or timed out after the model wrapper's retries.

    Anything else (a single request the store failed, a throttled model call
    after its retries) degrades only the section that made it.
    """
    if isinstance(exc, _FATAL_TYPES):
        return True
    # A botocore error exists only once botocore is loaded; a local-store,
    # local-model deployment never imports the AWS SDK.
    boto_exc = sys.modules.get("botocore.exceptions")
    if boto_exc is None:
        return False
    if isinstance(exc, boto_exc.ClientError):
        from unified_kg_rag.adapters.aws.bedrock_retry import (
            is_transient_bedrock_error,
        )

        return not is_transient_bedrock_error(exc)
    return isinstance(exc, boto_exc.BotoCoreError)
