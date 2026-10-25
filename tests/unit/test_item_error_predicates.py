# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which batch-item errors count against a document, per provider.

``indexing.max_document_failures`` excludes a document after that many
counted failures, so only errors the item causes count: output that does not
parse or was cut off, and a request the model rejects for its content. Any
other error (an outage, credentials, a quota, a setting, a bug) fails every
document alike and counts for none. The errors are real client exception
types built without a network.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from botocore import exceptions as boto_exc
from langchain_core.exceptions import OutputParserException

from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.adapters.llm.retry import item_error_predicate
from unified_kg_rag.adapters.local.models import (
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
)
from unified_kg_rag.domain.models import Config, TextUnit
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError

openai = pytest.importorskip("openai")
ollama = pytest.importorskip("ollama")

pytestmark = pytest.mark.unit

_REQUEST = httpx.Request("POST", "http://model-server.test/v1/chat/completions")


def _openai(cls: type[Exception], status: int, message: str) -> Exception:
    return cls(message, response=httpx.Response(status, request=_REQUEST), body=None)


def _bedrock(code: str, status: int, message: str = "x") -> boto_exc.ClientError:
    return boto_exc.ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "InvokeModel",
    )


_OUTPUT_ERRORS = [OutputParserException("bad"), LLMOutputTruncatedError("cut")]
_NOT_ITEM_ERRORS: list[BaseException] = [
    TimeoutError(),
    RuntimeError("model down"),
    TypeError("unexpected keyword argument"),
    KeyError("x"),
    KeyboardInterrupt(),
    boto_exc.NoCredentialsError(),
    boto_exc.NoRegionError(),
    boto_exc.TokenRetrievalError(provider="sso", error_msg="expired"),
    boto_exc.CredentialRetrievalError(provider="container-role", error_msg="x"),
    boto_exc.ParamValidationError(report="bad"),
]

_BEDROCK_ITEM = [
    _bedrock("ValidationException", 400, "Input is too long for requested model."),
    _bedrock(
        "ValidationException",
        400,
        "The model returned the following errors: prompt is too long: "
        "210000 tokens > 200000 maximum",
    ),
]
_BEDROCK_NOT_ITEM = [
    _bedrock("ServiceQuotaExceededException", 400, "exceeds the service quota"),
    _bedrock("ModelStreamErrorException", 424),
    _bedrock("ThrottlingException", 429),
    _bedrock("AccessDeniedException", 403),
    _bedrock("ExpiredTokenException", 403),
    _bedrock("ResourceNotFoundException", 404),
    _bedrock("ValidationException", 400, "The provided model identifier is invalid."),
    _bedrock(
        "ValidationException",
        400,
        "Invocation of model ID x with on-demand throughput isn't supported. "
        "Retry your request with the ID or ARN of an inference profile",
    ),
    _bedrock("ValidationException", 400, "temperature: Extra inputs are not permitted"),
    _bedrock("ValidationException", 400, "This model doesn't support tool use."),
]

_LOCAL_ITEM = [
    _openai(
        openai.BadRequestError,
        400,
        "This model's maximum context length is 8192 tokens. However, your "
        "messages resulted in 9000 tokens",
    ),
    _openai(openai.UnprocessableEntityError, 422, "bad input"),
    ollama.ResponseError("input rejected", 400),
]
_LOCAL_NOT_ITEM = [
    _openai(openai.BadRequestError, 400, "Invalid model name"),
    _openai(openai.BadRequestError, 400, "max_tokens is too large"),
    _openai(openai.BadRequestError, 400, "Unsupported parameter: 'top_k'"),
    _openai(openai.NotFoundError, 404, "no such model"),
    _openai(openai.AuthenticationError, 401, "bad key"),
    _openai(openai.RateLimitError, 429, "slow down"),
    _openai(openai.InternalServerError, 500, "crashed"),
    ollama.ResponseError("boom", -1),
    httpx.ConnectError("refused", request=_REQUEST),
]


def _ids(errors: list[BaseException]) -> list[str]:
    return [f"{type(e).__name__}:{str(e)[:30]}" for e in errors]


@pytest.mark.parametrize(
    "error",
    [*_OUTPUT_ERRORS, *_BEDROCK_ITEM],
    ids=_ids([*_OUTPUT_ERRORS, *_BEDROCK_ITEM]),
)
def test_bedrock_counts_its_item_errors(error: BaseException) -> None:
    factory = BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)
    assert item_error_predicate(factory)(error)


@pytest.mark.parametrize(
    "error",
    [*_NOT_ITEM_ERRORS, *_BEDROCK_NOT_ITEM],
    ids=_ids([*_NOT_ITEM_ERRORS, *_BEDROCK_NOT_ITEM]),
)
def test_bedrock_counts_no_other_error(error: BaseException) -> None:
    factory = BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)
    assert not item_error_predicate(factory)(error)


@pytest.mark.parametrize(
    "factory_cls", [OllamaLanguageModelFactory, OpenAICompatibleLanguageModelFactory]
)
def test_local_factories_count_their_item_errors_only(factory_cls: Any) -> None:
    predicate = item_error_predicate(factory_cls(Config()))
    assert all(predicate(e) for e in [*_OUTPUT_ERRORS, *_LOCAL_ITEM])
    counted = [e for e in [*_NOT_ITEM_ERRORS, *_LOCAL_NOT_ITEM] if predicate(e)]
    assert counted == []


def test_a_factory_without_a_classifier_counts_output_errors_only() -> None:
    predicate = item_error_predicate(object())
    assert all(predicate(e) for e in _OUTPUT_ERRORS)
    assert not any(predicate(e) for e in [*_BEDROCK_ITEM, *_LOCAL_ITEM])


def _units(documents: list[str]) -> list[TextUnit]:
    return [
        TextUnit(id=f"u{i}", text="t", document_ids=[doc])
        for i, doc in enumerate(documents)
    ]


@pytest.mark.parametrize(
    ("documents", "errors", "one_cause"),
    [
        # Every unit of two documents, one error: a setting, counted for none.
        (["a", "b"], {0: _BEDROCK_ITEM[0], 1: _BEDROCK_ITEM[0]}, True),
        # One document that always fails is still counted.
        (["a", "a"], {0: _BEDROCK_ITEM[0], 1: _BEDROCK_ITEM[0]}, False),
        # Not every unit failed, or not with the same error.
        (["a", "b"], {0: _BEDROCK_ITEM[0]}, False),
        (["a", "b"], {0: _BEDROCK_ITEM[0], 1: OutputParserException("bad")}, False),
        # Output that does not parse is each unit's own, however alike.
        (["a", "b"], {0: _OUTPUT_ERRORS[0], 1: _OUTPUT_ERRORS[0]}, False),
    ],
)
def test_one_error_on_every_unit_of_several_documents_counts_none(
    documents: list[str], errors: dict[int, BaseException], one_cause: bool
) -> None:
    assert (
        GraphExtractor._one_cause_failed_documents(_units(documents), errors)
        is one_cause
    )
