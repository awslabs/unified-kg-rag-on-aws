# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedding input truncation per catalog model, on the wire (AWS-free).

A real ``bedrock-runtime`` client is short-circuited at ``before-call``; the
request body each embed call would send is captured at
``before-parameter-build``. Cohere Embed v3 (512-token window) once lost every
text: the wrapper's 512-token estimate buffer left a limit of 0 tokens.
"""

from __future__ import annotations

import json
from typing import Any

import boto3
import pytest

from unified_kg_rag.adapters.aws.bedrock import (
    BedrockCrossRegionModelHelper,
    BedrockEmbeddingModelFactory,
)
from unified_kg_rag.adapters.aws.bedrock_models import _EMBEDDING_MODEL_INFO
from unified_kg_rag.domain.models import Config, EmbeddingModelId

pytestmark = pytest.mark.unit

REGION = "us-east-1"


class _Body:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw


class _CapturingSession:
    """A session whose ``bedrock-runtime`` client records each InvokeModel
    body and answers it locally, without a network call."""

    profile_name = "default"

    def __init__(self) -> None:
        self._session = boto3.Session(
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name=REGION,
        )
        self.bodies: list[dict[str, Any]] = []

    def _capture(self, params: dict[str, Any], **_: Any) -> None:
        self.bodies.append(json.loads(params["body"]))

    def _answer(self, **_: Any) -> tuple[Any, dict[str, Any]]:
        body = self.bodies[-1]
        if "texts" in body:
            vectors = [[0.1, 0.2]] * len(body["texts"])
            payload: dict[str, Any] = {
                "embeddings": (
                    {"float": vectors} if "embedding_types" in body else vectors
                )
            }
        else:
            payload = {"embedding": [0.1, 0.2], "inputTextTokenCount": 1}

        class _Http:
            status_code = 200
            headers: dict[str, str] = {}

        return _Http(), {
            "body": _Body(payload),
            "contentType": "application/json",
            "ResponseMetadata": {},
        }

    def client(self, service_name: str, **kwargs: Any) -> Any:
        kwargs.pop("region_name", None)
        client = self._session.client(service_name, region_name=REGION, **kwargs)
        client.meta.events.register(
            "before-parameter-build.bedrock-runtime.InvokeModel", self._capture
        )
        client.meta.events.register(
            "before-call.bedrock-runtime.InvokeModel", self._answer
        )
        return client

    def get_credentials(self) -> Any:
        return None


def _texts_sent(body: dict[str, Any]) -> list[str]:
    return body["texts"] if "texts" in body else [body["inputText"]]


@pytest.fixture
def embed(mocker) -> Any:
    # On-demand ids everywhere: profile resolution is covered elsewhere.
    mocker.patch.object(
        BedrockCrossRegionModelHelper,
        "get_cross_region_model_id",
        side_effect=lambda _s, model_id, *_a, **_k: str(model_id),
    )

    def _embed(model_id: EmbeddingModelId, texts: list[str]) -> list[str]:
        session = _CapturingSession()
        factory = BedrockEmbeddingModelFactory(
            Config(),
            boto_session=session,  # type: ignore[arg-type]
            region_name=REGION,
        )
        model = factory.get_model(model_id)
        vectors = model.embed_documents(texts)
        assert len(vectors) == len(texts)
        return [text for body in session.bodies for text in _texts_sent(body)]

    return _embed


@pytest.mark.parametrize("model_id", list(EmbeddingModelId))
def test_short_text_is_sent_intact(model_id: EmbeddingModelId, embed) -> None:
    texts = ["Vendor partners with Buyer.", "Depot stores gadgets."]
    assert embed(model_id, texts) == texts


@pytest.mark.parametrize("model_id", list(EmbeddingModelId))
def test_text_below_the_limit_is_sent_intact(model_id: EmbeddingModelId, embed) -> None:
    info = _EMBEDDING_MODEL_INFO[model_id]
    assert info.max_sequence_tokens is not None
    assert info.max_sequence_length is not None
    # Well under both the token and the character limit.
    words = min(info.max_sequence_tokens // 4, info.max_sequence_length // 16)
    text = " ".join(["vendor"] * words)
    assert embed(model_id, [text]) == [text]


@pytest.mark.parametrize("model_id", list(EmbeddingModelId))
def test_text_above_the_limit_is_cut_not_emptied(
    model_id: EmbeddingModelId, embed
) -> None:
    info = _EMBEDDING_MODEL_INFO[model_id]
    assert info.max_sequence_tokens is not None
    assert info.max_sequence_length is not None
    text = " ".join(["vendor"] * (info.max_sequence_tokens * 2))
    [sent] = embed(model_id, [text])
    assert text.startswith(sent)
    assert len(sent) <= info.max_sequence_length
    # Most of the window survives: the buffer takes at most an eighth of it.
    assert len(sent.split()) >= min(
        info.max_sequence_tokens // 2, info.max_sequence_length // 16
    )


@pytest.mark.parametrize(
    ("model_id", "limit"),
    [
        # 512-token window: the 512-token default buffer is capped at 512 // 8.
        (EmbeddingModelId("cohere.embed-multilingual-v3"), 512 - 64),
        # 8,192-token window: an eighth (1,024) is above the default, kept.
        (EmbeddingModelId("amazon.titan-embed-text-v2:0"), 8192 - 512),
    ],
)
def test_buffer_is_capped_only_for_small_windows(
    model_id: EmbeddingModelId, limit: int, embed, mocker
) -> None:
    from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter

    limits: list[int] = []
    truncate = BedrockTokenCounter.truncate_to_token_limit

    def spy(self: BedrockTokenCounter, text: str, max_tokens: int) -> Any:
        limits.append(max_tokens)
        return truncate(self, text, max_tokens)

    mocker.patch.object(BedrockTokenCounter, "truncate_to_token_limit", spy)
    embed(model_id, ["Vendor ships parts."])
    assert limits and set(limits) == {limit}
