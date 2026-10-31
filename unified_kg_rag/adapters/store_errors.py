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

from unified_kg_rag.shared import (
    ConfigurationError,
    EmbeddingModelError,
    IndexNotFoundError,
    InvalidFilterError,
)

# Errors that fail every later query the same way: a store that cannot be
# reached or keeps failing, a missing suffix, a configuration the store or the
# query refuses, a filter key no index declares, and a query embedding that
# failed after the client's retries (every vector search of the query needs
# it).
_FATAL_TYPES = (
    ConnectionError,
    ConfigurationError,
    EmbeddingModelError,
    IndexNotFoundError,
    InvalidFilterError,
)


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
    - ``EmbeddingModelError``: the query embedding failed, whatever the cause
      (a 5xx after the client's retries too). The query's vector searches all
      share it, so no partial answer is left to degrade to.
    - A botocore ``ClientError`` (a Bedrock call) unless its code is one of the
      transient ones (``is_transient_bedrock_error``): an invalid or
      inaccessible model id, a model in another region, denied access.
    - Any other botocore error: no credentials, no region, an endpoint that
      could not be reached or timed out after the model wrapper's retries.
    - A local model server's error (``_is_fatal_provider_error``): an HTTP
      4xx other than 408/429 (a bad key, an unknown model), or a server that
      could not be reached or did not answer after the client's retries.

    Not fatal: a prompt longer than the model's context
    (``ContextOverflowError``, whichever provider raised it), which only that
    prompt hits, and anything else (a single request the store failed, a
    throttled model call after its retries). It degrades only the section
    that made it.
    """
    # Before the provider rules: Bedrock raises it as a ValidationException
    # ClientError and OpenAI-compatible servers as a 400.
    overflow = sys.modules.get("langchain_core.exceptions")
    if overflow is not None and isinstance(exc, overflow.ContextOverflowError):
        return False
    if isinstance(exc, _FATAL_TYPES) or _is_fatal_provider_error(exc):
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


def is_bad_request(exc: BaseException) -> bool:
    """Whether ``exc`` is an OpenAI-compatible server's HTTP 400.

    vLLM and llama.cpp report a prompt longer than the model's context as a
    plain 400, which LangChain does not map to ``ContextOverflowError``: the
    global-search map step leaves a batch that got one unrated, as an
    overflow, unless no batch was rated at all. A typed overflow
    (langchain-openai's ``OpenAIContextOverflowError`` is a ``BadRequestError``
    too) is never one: only that prompt is too long, however many batches hit
    it.
    """
    openai = sys.modules.get("openai")
    if openai is None or not isinstance(exc, openai.BadRequestError):
        return False
    overflow = sys.modules.get("langchain_core.exceptions")
    return overflow is None or not isinstance(exc, overflow.ContextOverflowError)


def _is_fatal_status(status: object) -> bool:
    # 408 and 429 resolve on their own, as does a 5xx; a 4xx is the same on
    # every call. A status-less error (Ollama's mid-stream ones: -1) is not.
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 429)


def _is_fatal_provider_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a local model server's error every call repeats.

    The OpenAI and Ollama clients (and httpx under them) are looked up in
    ``sys.modules``: a deployment that never imported one cannot have raised
    its errors. Matched by type, so opensearch-py's and gremlinpython's
    errors of similar names are not.
    """
    openai = sys.modules.get("openai")
    if openai is not None:
        if isinstance(exc, openai.APIStatusError):
            return _is_fatal_status(exc.status_code)
        # The server could not be reached or did not answer (APITimeoutError)
        # after the client's own retries.
        if isinstance(exc, openai.APIConnectionError):
            return True
    ollama = sys.modules.get("ollama")
    if ollama is not None and isinstance(exc, ollama.ResponseError):
        return _is_fatal_status(exc.status_code)
    httpx = sys.modules.get("httpx")
    return httpx is not None and isinstance(exc, httpx.TransportError)
