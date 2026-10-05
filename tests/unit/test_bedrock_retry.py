# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the transient-error retry on the Bedrock embedding path.

AWS-free: the embedding wrapper is built with a fake ``bedrock-runtime`` client
whose ``invoke_model`` raises scripted botocore errors, and ``time.sleep`` in
the retry module is patched so backoff is observable without waiting.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from unified_kg_rag.adapters.aws import bedrock_retry
from unified_kg_rag.adapters.aws.bedrock import BedrockEmbeddingsWrapper
from unified_kg_rag.adapters.aws.bedrock_retry import (
    backoff_delay,
    call_with_transient_retry,
    is_transient_bedrock_error,
)
from unified_kg_rag.domain.models import Config, EmbeddingModelId
from unified_kg_rag.domain.models.config import TransientRetryConfig

pytestmark = pytest.mark.unit


def _client_error(code: str, status: int = 400) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "synthetic failure"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "InvokeModel",
    )


class _FakeRuntimeClient:
    """Fake bedrock-runtime client: raises ``errors`` in order, then succeeds."""

    def __init__(self, errors: list[BaseException] | None = None) -> None:
        self.errors = list(errors or [])
        self.calls = 0

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        body = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        return {"body": io.BytesIO(body)}


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(bedrock_retry, "_sleep", recorded.append)
    return recorded


def _wrapper(client: _FakeRuntimeClient, **kwargs: Any) -> BedrockEmbeddingsWrapper:
    return BedrockEmbeddingsWrapper(
        client=client, model_id="amazon.titan-embed-text-v2:0", **kwargs
    )


# --- classification -------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "ModelErrorException",
        "ModelNotReadyException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ThrottlingException",
    ],
)
def test_transient_codes_are_retryable(code: str) -> None:
    assert is_transient_bedrock_error(_client_error(code))


@pytest.mark.parametrize(
    "code", ["ValidationException", "AccessDeniedException", "ResourceNotFound"]
)
def test_client_errors_are_not_retryable(code: str) -> None:
    assert not is_transient_bedrock_error(_client_error(code))


def test_connection_errors_are_retryable() -> None:
    assert is_transient_bedrock_error(EndpointConnectionError(endpoint_url="x"))
    assert is_transient_bedrock_error(ReadTimeoutError(endpoint_url="x"))
    assert not is_transient_bedrock_error(ValueError("bad"))


def test_backoff_delay_is_bounded_and_grows() -> None:
    for attempt in range(1, 10):
        ceiling = min(16.0, 2.0 * 2 ** (attempt - 1))
        d = backoff_delay(attempt, base_delay=2.0, max_delay=16.0)
        assert ceiling / 2 <= d <= ceiling


# --- embedding wrapper ----------------------------------------------------


def test_model_error_retried_then_succeeds(sleeps: list[float]) -> None:
    client = _FakeRuntimeClient(
        [_client_error("ModelErrorException", 424)] * 2,
    )
    emb = _wrapper(client).embed_documents(["synthetic text"])
    assert emb == [[0.1, 0.2, 0.3]]
    assert client.calls == 3
    assert len(sleeps) == 2
    assert sleeps[1] >= sleeps[0] / 2  # exponential ceiling, jittered


def test_non_transient_error_not_retried(sleeps: list[float]) -> None:
    client = _FakeRuntimeClient([_client_error("ValidationException")])
    with pytest.raises(ClientError):
        _wrapper(client).embed_query("synthetic text")
    assert client.calls == 1
    assert sleeps == []


def test_exhausted_retries_reraise_original(sleeps: list[float]) -> None:
    client = _FakeRuntimeClient([_client_error("ModelErrorException", 424)] * 10)
    with pytest.raises(ClientError) as info:
        _wrapper(
            client, transient_retry=TransientRetryConfig(max_attempts=3)
        ).embed_query("synthetic text")
    assert info.value.response["Error"]["Code"] == "ModelErrorException"
    assert client.calls == 3
    assert len(sleeps) == 2


def test_retry_respects_wall_clock_budget(sleeps: list[float]) -> None:
    client = _FakeRuntimeClient([_client_error("ModelErrorException", 424)] * 10)
    # Budget smaller than the first backoff: give up without sleeping.
    with pytest.raises(ClientError):
        _wrapper(
            client, transient_retry=TransientRetryConfig(max_total_seconds=0.1)
        ).embed_query("x")
    assert client.calls == 1
    assert sleeps == []


def test_mid_batch_failure_does_not_restart_batch(sleeps: list[float]) -> None:
    # The second text's InvokeModel fails once; only that call is retried.
    client = _FakeRuntimeClient()
    original = client.invoke_model
    state = {"n": 0}

    def flaky(**kwargs: Any) -> dict[str, Any]:
        state["n"] += 1
        if state["n"] == 2:
            client.calls += 1
            raise _client_error("ModelErrorException", 424)
        return original(**kwargs)

    client.invoke_model = flaky  # type: ignore[method-assign]
    out = _wrapper(client).embed_documents(["a", "b", "c"])
    assert len(out) == 3
    assert client.calls == 4  # 3 texts + 1 retry
    assert len(sleeps) == 1


async def test_async_aembed_query_retries(sleeps: list[float]) -> None:
    client = _FakeRuntimeClient([_client_error("ModelNotReadyException", 429)])
    emb = await _wrapper(client).aembed_query("synthetic text")
    assert emb == [0.1, 0.2, 0.3]
    assert client.calls == 2
    assert len(sleeps) == 1


def test_call_with_transient_retry_passthrough(sleeps: list[float]) -> None:
    policy = TransientRetryConfig()
    assert call_with_transient_retry(lambda: 42, operation="noop", policy=policy) == 42
    assert sleeps == []


# --- one policy from config -------------------------------------------------


def test_embedding_factory_applies_configured_policy(mocker) -> None:
    from unified_kg_rag.adapters.aws.bedrock import BedrockEmbeddingModelFactory

    config = Config()
    config.aws.bedrock.transient_retry = TransientRetryConfig(max_attempts=2)
    factory = BedrockEmbeddingModelFactory(
        config=config, boto_session=mocker.MagicMock()
    )
    model = factory.get_model(EmbeddingModelId.TITAN_EMBED_V2)
    assert model.transient_retry.max_attempts == 2


def test_legacy_search_llm_retry_key_still_configures_the_policy() -> None:
    legacy = Config.model_validate({"search": {"llm_retry": {"max_attempts": 2}}})
    assert legacy.aws.bedrock.transient_retry.max_attempts == 2

    both = Config.model_validate(
        {
            "search": {"llm_retry": {"max_attempts": 2}},
            "aws": {"bedrock": {"transient_retry": {"max_attempts": 4}}},
        }
    )
    assert both.aws.bedrock.transient_retry.max_attempts == 4
