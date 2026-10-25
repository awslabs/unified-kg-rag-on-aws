# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fake Ollama server for the local-mode adapters (``httpx`` handler).

Pass it to ``httpx.MockTransport`` and the transport to the local factories
(``OllamaLanguageModelFactory(config, transport=...)``): ``/api/chat`` answers
with ``respond(system_prompt, user_prompt)`` (falling back to ``answer`` when
the responder rejects the prompt with ``AssertionError``, i.e. the query's
answer generation, and with HTTP ``status`` when it raises
:class:`ModelRejects`), ``/api/embed`` with hashing vectors of ``dimensions``.
No network and no model.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from tests.fixtures.fakes.embeddings import HashingEmbeddings

Responder = Callable[[str, str], str]


class ModelRejects(Exception):
    """Raised by a responder: the server rejects the request with HTTP
    ``status`` (400: a request the model rejects for its content)."""

    def __init__(self, message: str = "input rejected", status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class FakeOllama:
    """Records the models each endpoint was asked for."""

    def __init__(self, respond: Responder, answer: str, dimensions: int) -> None:
        self.respond = respond
        self.answer = answer
        self.embedder = HashingEmbeddings(dimensions)
        self.chat_models: list[str] = []
        self.embed_models: list[str] = []
        self.answers = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/embed":
            self.embed_models.append(body["model"])
            inputs = body["input"]
            inputs = inputs if isinstance(inputs, list) else [inputs]
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "embeddings": self.embedder.embed_documents(inputs),
                },
            )
        if request.url.path != "/api/chat":
            return httpx.Response(404, json={"error": request.url.path})
        self.chat_models.append(body["model"])
        messages = body["messages"]
        system = "".join(m["content"] for m in messages if m["role"] == "system")
        human = "".join(m["content"] for m in messages if m["role"] == "user")
        try:
            reply = self.respond(system, human)
        except AssertionError:
            self.answers += 1
            reply = self.answer
        except ModelRejects as e:
            return httpx.Response(e.status, json={"error": str(e)})
        chunk = {
            "model": body["model"],
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": reply},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 10,
        }
        return httpx.Response(200, content=(json.dumps(chunk) + "\n").encode())
