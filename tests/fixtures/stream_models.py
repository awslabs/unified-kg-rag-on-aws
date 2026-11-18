# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A loopback model server that streams its answer as each provider's wire
does: Ollama (``/api/chat``, NDJSON), an OpenAI-compatible server
(``/v1/chat/completions``, SSE) and Bedrock (``ConverseStream``, the AWS
event stream). A streamed answer reasons first (``reasoning_seconds`` of
reasoning chunks, no answer text), then writes ``chunks`` answer chunks,
optionally stalling after one of them (``stall_after``, ``stall_seconds``)
or before its first byte (``first_seconds``). With ``throttled`` set, every
chat call is refused as each provider throttles (HTTP 429: Bedrock's
``ThrottlingException``, an error body for the others) and counted
(``throttled``).

It counts the streamed responses it is writing (a client that closes its
connection is seen at once, while the answer waits too) and those whose
client left before the end, so a test sees whether a model is still
generating for a caller that stopped reading. Binds 127.0.0.1 only.
"""

from __future__ import annotations

import binascii
import json
import select
import socket
import struct
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ANSWER_WORDS = "Vendor ships parts to Buyer every quarter under the agreement".split()


@dataclass
class StreamKnobs:
    """How the next streamed answers are written."""

    chunk_delay: float = 0.05
    chunks: int = 20
    reasoning_seconds: float = 0.0
    # The answer chunk after which the stream stalls (None: it does not),
    # for stall_seconds or until ``unstall`` is set (a test's teardown).
    stall_after: int | None = None
    stall_seconds: float = 0.0
    unstall: threading.Event = field(default_factory=threading.Event)
    # Seconds the stream waits before its first piece (a slow first byte),
    # a stall as well.
    first_seconds: float = 0.0
    # Refuse every chat call with HTTP 429 (a throttled model).
    throttled: bool = False


@dataclass
class StreamModels:
    """The server's knobs and counters."""

    knobs: StreamKnobs = field(default_factory=StreamKnobs)
    writing: int = 0
    completed: int = 0
    left: int = 0
    # Chat calls refused as throttled (StreamKnobs.throttled).
    throttled: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def count(self, name: str, delta: int = 1) -> None:
        with self.lock:
            setattr(self, name, getattr(self, name) + delta)

    def settled(self, timeout: float = 60.0) -> int:
        """The responses still being written after up to ``timeout`` s."""
        end = time.monotonic() + timeout
        while self.writing and time.monotonic() < end:
            time.sleep(0.05)
        return self.writing


def _words(n: int) -> list[str]:
    return [ANSWER_WORDS[i % len(ANSWER_WORDS)] + " " for i in range(n)]


def _event_header(name: str, value: str) -> bytes:
    key, text = name.encode(), value.encode()
    return (
        struct.pack("B", len(key)) + key + b"\x07" + struct.pack(">H", len(text)) + text
    )


def _bedrock_event(event_type: str, payload: dict[str, Any]) -> bytes:
    """One AWS event-stream message (``vnd.amazon.eventstream``)."""
    headers = b"".join(
        _event_header(k, v)
        for k, v in {
            ":event-type": event_type,
            ":content-type": "application/json",
            ":message-type": "event",
        }.items()
    )
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 12 + len(headers) + len(body) + 4, len(headers))
    prelude += struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + headers + body
    return message + struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)


def _sse(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n".encode()


class _Wire:
    """One provider's framing of a streamed answer: its reasoning piece,
    answer pieces and the pieces around them."""

    content_type: str

    def __init__(self, body: dict[str, Any]) -> None:
        self.model = body.get("model") or "fake"

    def start(self) -> list[bytes]:
        return []

    def reasoning(self) -> bytes:
        raise NotImplementedError

    def after_reasoning(self) -> list[bytes]:
        return []

    def answer(self, text: str) -> bytes:
        raise NotImplementedError

    def end(self, n: int) -> list[bytes]:
        raise NotImplementedError

    def whole(self, text: str) -> dict[str, Any]:
        raise NotImplementedError


class _Ollama(_Wire):
    content_type = "application/x-ndjson"

    def _line(self, message: dict[str, Any], **extra: Any) -> bytes:
        return (
            json.dumps({"model": self.model, "message": message, **extra}) + "\n"
        ).encode()

    def reasoning(self) -> bytes:
        return self._line(
            {"role": "assistant", "content": "", "thinking": "hm "}, done=False
        )

    def answer(self, text: str) -> bytes:
        return self._line({"role": "assistant", "content": text}, done=False)

    def end(self, n: int) -> list[bytes]:
        return [
            self._line(
                {"role": "assistant", "content": ""},
                done=True,
                done_reason="stop",
                prompt_eval_count=5,
                eval_count=n,
            )
        ]

    def whole(self, text: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "message": {"role": "assistant", "content": text},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 5,
        }


class _OpenAI(_Wire):
    content_type = "text/event-stream"

    def _delta(self, delta: dict[str, Any], finish: str | None = None) -> bytes:
        return _sse(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    def start(self) -> list[bytes]:
        return [self._delta({"role": "assistant", "content": ""})]

    def reasoning(self) -> bytes:
        return self._delta({"reasoning_content": "hm "})

    def answer(self, text: str) -> bytes:
        return self._delta({"content": text})

    def end(self, n: int) -> list[bytes]:
        return [self._delta({}, "stop"), _sse("[DONE]")]

    def whole(self, text: str) -> dict[str, Any]:
        return {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }


class _Bedrock(_Wire):
    content_type = "application/vnd.amazon.eventstream"

    def __init__(self, body: dict[str, Any]) -> None:
        super().__init__(body)
        self.block = 0

    def start(self) -> list[bytes]:
        return [_bedrock_event("messageStart", {"role": "assistant"})]

    def _delta(self, delta: dict[str, Any]) -> bytes:
        return _bedrock_event(
            "contentBlockDelta", {"contentBlockIndex": self.block, "delta": delta}
        )

    def reasoning(self) -> bytes:
        return self._delta({"reasoningContent": {"text": "hm "}})

    def after_reasoning(self) -> list[bytes]:
        pieces = [
            self._delta({"reasoningContent": {"signature": "sig"}}),
            _bedrock_event("contentBlockStop", {"contentBlockIndex": self.block}),
        ]
        self.block += 1
        return pieces

    def answer(self, text: str) -> bytes:
        return self._delta({"text": text})

    def end(self, n: int) -> list[bytes]:
        return [
            _bedrock_event("contentBlockStop", {"contentBlockIndex": self.block}),
            _bedrock_event("messageStop", {"stopReason": "end_turn"}),
            _bedrock_event(
                "metadata",
                {
                    "usage": {
                        "inputTokens": 5,
                        "outputTokens": n,
                        "totalTokens": 5 + n,
                    },
                    "metrics": {"latencyMs": 1},
                },
            ),
        ]

    def whole(self, text: str) -> dict[str, Any]:
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 5, "totalTokens": 10},
            "metrics": {"latencyMs": 1},
        }


def _pieces(wire: _Wire, knobs: StreamKnobs) -> Iterator[tuple[bytes, float, float]]:
    """The streamed answer's pieces, each with the seconds to wait after it
    and the seconds it stalls then."""
    for piece in wire.start():
        yield piece, 0.0, 0.0
    reasoning = int(knobs.reasoning_seconds / knobs.chunk_delay)
    for _ in range(reasoning):
        yield wire.reasoning(), knobs.chunk_delay, 0.0
    if reasoning:
        for piece in wire.after_reasoning():
            yield piece, 0.0, 0.0
    words = _words(knobs.chunks)
    for i, word in enumerate(words):
        stall = knobs.stall_seconds if i == knobs.stall_after else 0.0
        yield wire.answer(word), knobs.chunk_delay, stall
    for piece in wire.end(len(words)):
        yield piece, 0.0, 0.0


def _handler(models: StreamModels) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return None

        def _json(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.loads(raw or b"{}")
            if self.path == "/api/embed":
                inputs = body.get("input") or []
                inputs = inputs if isinstance(inputs, list) else [inputs]
                self._json({"embeddings": [[0.1] * 64 for _ in inputs]})
                return
            wire: _Wire
            if self.path == "/api/chat":
                wire, streamed = _Ollama(body), body.get("stream", True)
            elif self.path.endswith("/chat/completions"):
                wire, streamed = _OpenAI(body), bool(body.get("stream"))
            elif self.path.endswith("/converse-stream"):
                wire, streamed = _Bedrock(body), True
            elif self.path.endswith("/converse"):
                wire, streamed = _Bedrock(body), False
            else:
                self.send_error(404)
                return
            knobs = models.knobs
            if knobs.throttled:
                models.count("throttled")
                self._throttled(isinstance(wire, _Bedrock))
                return
            if not streamed:
                self._json(wire.whole("".join(_words(knobs.chunks))))
                return
            self._stream(wire, knobs)

        def _throttled(self, bedrock: bool) -> None:
            data = json.dumps(
                {"message": "Too many requests"}
                if bedrock
                else {"error": {"message": "Too many requests", "type": "rate_limit"}}
            ).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            if bedrock:
                self.send_header("x-amzn-ErrorType", "ThrottlingException")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _stream(self, wire: _Wire, knobs: StreamKnobs) -> None:
            self.send_response(200)
            self.send_header("Content-Type", wire.content_type)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("x-amzn-RequestId", "fake-request")
            self.end_headers()
            models.count("writing")
            try:
                self._wait(knobs.first_seconds, knobs.unstall)
                for piece, wait, stall in _pieces(wire, knobs):
                    self.wfile.write(f"{len(piece):X}\r\n".encode() + piece + b"\r\n")
                    self.wfile.flush()
                    self._wait(wait)
                    self._wait(stall, knobs.unstall)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                models.count("completed")
            except OSError:
                # The client left mid-answer.
                models.count("left")
                self.close_connection = True
            finally:
                models.count("writing", -1)

        def _wait(self, seconds: float, unstall: threading.Event | None = None) -> None:
            """Wait ``seconds`` (a stall: until ``unstall`` is set, too)
            before the next piece; raise ``OSError`` as soon as the client
            closes its connection, so ``writing`` counts the connections
            still open, not the waits left."""
            end = time.monotonic() + seconds
            while (left := end - time.monotonic()) > 0:
                if unstall is not None and unstall.is_set():
                    return
                if not select.select([self.connection], [], [], min(left, 0.05))[0]:
                    continue
                try:
                    data = self.connection.recv(1, socket.MSG_PEEK)
                except OSError:
                    data = b""
                if not data:
                    raise ConnectionResetError("the client closed its connection")
                time.sleep(min(left, 0.05))

    return Handler


def serve(
    models: StreamModels | None = None,
) -> tuple[ThreadingHTTPServer, StreamModels]:
    """Start the server on a free port on a daemon thread."""
    models = models or StreamModels()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(models))
    server.daemon_threads = True
    threading.Thread(
        target=server.serve_forever, name="stream-models", daemon=True
    ).start()
    return server, models
