# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which errors the ingestion batch retries treat as transient, per provider.

The ingestion stages take the classifier from the configured LLM factory: the
Bedrock factory names botocore's transient errors, the local factories (and
any factory without a classifier) network failures and HTTP 408/429/5xx of
the httpx, OpenAI and Ollama clients. The errors are real client exception
types built without a network.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.bedrock_retry import is_transient_bedrock_error
from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
)
from unified_kg_rag.adapters.llm.langchain import BatchProcessor
from unified_kg_rag.adapters.llm.retry import (
    looks_transient,
    transient_error_predicate,
)
from unified_kg_rag.adapters.local.models import (
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config

openai = pytest.importorskip("openai")
ollama = pytest.importorskip("ollama")

pytestmark = pytest.mark.unit

_REQUEST = httpx.Request("POST", "http://model-server.test/v1/chat/completions")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST, json={"error": "synthetic"})


def _openai_status(cls: type[Exception], status: int) -> Exception:
    return cls("synthetic", response=_response(status), body=None)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "Converse")


_TRANSIENT_LOCAL: list[BaseException] = [
    httpx.ConnectError("refused", request=_REQUEST),
    httpx.ReadTimeout("read timed out", request=_REQUEST),
    httpx.RemoteProtocolError("dropped", request=_REQUEST),
    httpx.HTTPStatusError("503", request=_REQUEST, response=_response(503)),
    openai.APIConnectionError(request=_REQUEST),
    openai.APITimeoutError(request=_REQUEST),
    _openai_status(openai.InternalServerError, 500),
    _openai_status(openai.RateLimitError, 429),
    ollama.ResponseError("model runner crashed", status_code=500),
    ollama.ResponseError("busy", status_code=503),
    ollama.ResponseError("too many requests", status_code=429),
    ConnectionResetError("reset"),
    TimeoutError("timed out"),
]

_PERMANENT_LOCAL: list[BaseException] = [
    _openai_status(openai.BadRequestError, 400),
    _openai_status(openai.AuthenticationError, 401),
    _openai_status(openai.NotFoundError, 404),
    ollama.ResponseError("model 'synthetic-chat' not found", status_code=404),
    # Raised mid-stream with no HTTP status.
    ollama.ResponseError("bad json"),
    httpx.HTTPStatusError("400", request=_REQUEST, response=_response(400)),
    ValueError("not a network error"),
    _client_error("ThrottlingException"),  # botocore: the Bedrock classifier's
]


@pytest.mark.parametrize("exc", _TRANSIENT_LOCAL, ids=repr)
def test_local_network_and_server_errors_are_transient(exc: BaseException) -> None:
    assert looks_transient(exc)


@pytest.mark.parametrize("exc", _PERMANENT_LOCAL, ids=repr)
def test_local_client_errors_fail_fast(exc: BaseException) -> None:
    assert not looks_transient(exc)


@pytest.mark.parametrize(
    "factory_cls",
    [OllamaLanguageModelFactory, OpenAICompatibleLanguageModelFactory],
)
def test_local_factories_classify_their_client_errors(factory_cls: Any) -> None:
    predicate = transient_error_predicate(factory_cls(Config()))
    assert all(predicate(e) for e in _TRANSIENT_LOCAL)
    assert not any(predicate(e) for e in _PERMANENT_LOCAL)


def test_bedrock_factory_keeps_the_bedrock_classifier() -> None:
    predicate = transient_error_predicate(
        BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)
    )
    assert predicate is is_transient_bedrock_error
    assert predicate(_client_error("ModelErrorException"))
    assert predicate(EndpointConnectionError(endpoint_url="x"))
    assert not predicate(_client_error("ValidationException"))


def test_factory_without_a_classifier_retries_transient_looking_errors() -> None:
    predicate = transient_error_predicate(object())
    assert predicate is looks_transient
    assert predicate(httpx.ConnectError("refused", request=_REQUEST))
    assert not predicate(ValueError("permanent"))


class _FlakyFactory:
    """A custom LLM factory (no ``is_transient_error``); never called here."""

    def get_model(self, model_id: str, **kwargs: Any) -> Any:
        raise AssertionError("not called")

    def get_model_info(self, model_id: str) -> Any:
        raise AssertionError("not called")


@pytest.mark.parametrize(
    ("factory", "error", "attempts_made"),
    [
        (OllamaLanguageModelFactory(Config()), httpx.ConnectError("refused"), 3),
        (
            OllamaLanguageModelFactory(Config()),
            ollama.ResponseError("crashed", status_code=500),
            3,
        ),
        (
            OpenAICompatibleLanguageModelFactory(Config()),
            httpx.ReadTimeout("slow"),
            3,
        ),
        (
            OllamaLanguageModelFactory(Config()),
            ollama.ResponseError("not found", status_code=404),
            1,
        ),
        (_FlakyFactory(), httpx.ConnectError("refused"), 3),
        (_FlakyFactory(), ValueError("permanent"), 1),
    ],
    ids=[
        "ollama-connect",
        "ollama-500",
        "openai-timeout",
        "ollama-404",
        "custom-connect",
        "custom-permanent",
    ],
)
def test_stage_batch_retries_follow_the_llm_factory(
    mocker: Any, factory: Any, error: BaseException, attempts_made: int
) -> None:
    # The stage's batch processor carries the factory's classifier, so a local
    # run retries its own network failures (and not a missing model).
    mocker.patch("unified_kg_rag.adapters.ingestion.description_summarizer.setup_chain")
    stage = DescriptionSummarizer(
        Config(), providers=Providers(Config(), llm_factory=factory)
    )
    processor: BatchProcessor = stage.batch_processor.model_copy(
        update={"max_attempts": 3, "retry_multiplier": 1.0, "retry_max_wait": 0}
    )
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    def call() -> None:
        calls["n"] += 1
        raise error

    with pytest.raises(type(error)):
        call()
    assert calls["n"] == attempts_made
