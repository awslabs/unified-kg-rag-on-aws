# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console demo's fake model server (``console/demo/fake_models.py``)
with fault injection, for load and chaos runs (docs/guides/load-testing.md).

It binds 127.0.0.1 only. Each ``/api/chat`` and ``/api/embed`` request, in
this order, may wait (``delay`` plus up to ``jitter`` seconds), hang
(``hang_rate``, for ``hang_seconds``), lose its connection with no response
(``drop_rate``) or be answered with ``error_status`` (``error_rate``: 429
throttling, 500, 503). A streamed ``/api/chat`` answer may start with
``reasoning_chunks`` reasoning chunks (Ollama's ``thinking`` field, no
answer text) and lose its connection mid-stream: after its reasoning,
before any answer text (``reasoning_drop_rate``), or after its first
answer chunk (``answer_drop_rate``). The knobs change at run time:
``POST /chaos {"error_rate": 0.3}`` answers the knobs now in force, and
``GET /stats`` the request counts, the faults injected and the requests in
flight.

Usage::

    uv run python scripts/loadtest/chaos_models.py --port 11613 \
        --error-rate 0.3 --error-status 429
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "console" / "demo"))

import fake_models  # type: ignore[import-not-found]  # noqa: E402

KNOBS: dict[str, float] = {
    "error_rate": 0.0,
    "error_status": 429,
    "drop_rate": 0.0,
    "hang_rate": 0.0,
    "hang_seconds": 30.0,
    "delay": 0.0,
    "jitter": 0.0,
    "reasoning_chunks": 0.0,
    "reasoning_drop_rate": 0.0,
    "answer_drop_rate": 0.0,
}
# The mid-stream faults of a streamed chat answer, in draw order.
STREAM_FAULTS = ("reasoning_drop", "answer_drop")


class Chaos:
    """The knobs, the counters and the seeded draws of one server."""

    def __init__(self, seed: int = 7, **knobs: float) -> None:
        unknown = set(knobs) - set(KNOBS)
        if unknown:
            raise ValueError(f"unknown chaos knobs: {sorted(unknown)}")
        self.knobs: dict[str, float] = {**KNOBS, **knobs}
        self.stats: dict[str, int] = dict.fromkeys(
            (
                "chat",
                "embed",
                "hangs",
                "drops",
                "errors",
                "reasoning_drops",
                "answer_drops",
                "inflight",
                "max_inflight",
            ),
            0,
        )
        self.lock = threading.Lock()
        self.rng = random.Random(seed)

    def update(self, changes: dict[str, Any]) -> dict[str, float]:
        """Apply the known knobs of ``changes``; an unknown one is refused."""
        unknown = set(changes) - set(self.knobs)
        if unknown:
            raise ValueError(f"unknown chaos knobs: {sorted(unknown)}")
        with self.lock:
            self.knobs.update({k: float(v) for k, v in changes.items()})
            return dict(self.knobs)

    def begin(self, kind: str) -> tuple[float, str | None]:
        """Count a request and draw its fate: the seconds it waits and the
        fault (``"hang"``, ``"drop"``, ``"error"`` or None) it gets."""
        with self.lock:
            self.stats[kind] += 1
            self.stats["inflight"] += 1
            self.stats["max_inflight"] = max(
                self.stats["max_inflight"], self.stats["inflight"]
            )
            k = self.knobs
            wait = k["delay"] + self.rng.random() * k["jitter"]
            draws = [self.rng.random() for _ in range(3)]
            fault = next(
                (
                    name
                    for name, draw in zip(("hang", "drop", "error"), draws, strict=True)
                    if draw < k[f"{name}_rate"]
                ),
                None,
            )
            if fault is not None:
                self.stats[f"{fault}s"] += 1
            if fault == "hang":
                wait += k["hang_seconds"]
            return wait, fault

    def stream_fault(self) -> str | None:
        """Draw a streamed answer's mid-stream fault (``"reasoning_drop"``,
        ``"answer_drop"`` or None); no draw while both rates are 0, so the
        other faults' draws stay as without these knobs."""
        with self.lock:
            rates = [self.knobs[f"{name}_rate"] for name in STREAM_FAULTS]
            if not any(rates):
                return None
            draw = self.rng.random()
            for name, rate in zip(STREAM_FAULTS, rates, strict=True):
                if draw < rate:
                    return name
                draw -= rate
            return None

    def count(self, fault: str) -> None:
        """Count a mid-stream ``fault`` once it is injected."""
        with self.lock:
            self.stats[f"{fault}s"] += 1

    def end(self) -> None:
        with self.lock:
            self.stats["inflight"] -= 1


def make_server(
    chaos: Chaos, port: int = 0, dimensions: int = 64, chunk_delay: float = 0.0
) -> ThreadingHTTPServer:
    """A server on 127.0.0.1:``port`` (0: a free one) answering as the fake
    models, faults drawn from ``chaos``."""
    models = fake_models.FakeModels(dimensions, chunk_delay, 0.0, None)
    base: Any = fake_models.make_handler(models)

    class Handler(base):  # type: ignore[misc, valid-type]
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/stats":
                with chaos.lock:
                    payload = {**chaos.stats, "knobs": chaos.knobs}
                self._json(200, {**payload, "chat_calls": models.calls})
                return
            super().do_GET()

        def _drain(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def _drop(self) -> None:
            self.close_connection = True
            try:
                self.connection.shutdown(2)
            except OSError:
                pass

        def _chunk(self, chunk: dict[str, Any]) -> None:
            line = (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
            self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()

        def _chat(self) -> None:
            """A chat answer as the fake models stream it, with the
            reasoning chunks and the mid-stream fault the knobs draw."""
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            streamed = body.get("stream", True)
            fault = chaos.stream_fault() if streamed else None
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            if streamed:
                for i in range(int(chaos.knobs["reasoning_chunks"])):
                    self._chunk(
                        {
                            "model": body.get("model"),
                            "created_at": "2026-01-01T00:00:00Z",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "thinking": f"step {i} ",
                            },
                            "done": False,
                        }
                    )
            if fault == "reasoning_drop":
                chaos.count(fault)
                self._drop()
                return
            for n, chunk in enumerate(models.chat(body)):
                # After the first answer chunk, if there is a next one.
                if n == 1 and fault == "answer_drop":
                    chaos.count(fault)
                    self._drop()
                    return
                self._chunk(chunk)
            self.wfile.write(b"0\r\n\r\n")

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/chaos":
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    knobs = chaos.update(json.loads(self.rfile.read(length) or b"{}"))
                except (ValueError, TypeError) as e:
                    self._json(400, {"error": str(e)})
                    return
                self._json(200, knobs)
                return
            if self.path not in ("/api/chat", "/api/embed"):
                super().do_POST()
                return
            wait, fault = chaos.begin("embed" if self.path == "/api/embed" else "chat")
            try:
                if wait:
                    time.sleep(wait)
                if fault == "drop":
                    self._drain()
                    self._drop()
                elif fault == "error":
                    self._drain()
                    self._json(
                        int(chaos.knobs["error_status"]), {"error": "injected fault"}
                    )
                elif self.path == "/api/chat":
                    self._chat()
                else:
                    super().do_POST()
            except OSError:
                pass  # the client went away mid-answer
            finally:
                chaos.end()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def call(
    port: int, path: str, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    """A request to the server on 127.0.0.1:``port`` (POST with a body,
    else GET): the status and the JSON answer."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        data = None if body is None else json.dumps(body).encode()
        connection.request("GET" if data is None else "POST", path, body=data)
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=0, help="0: a free port")
    parser.add_argument("--port-file", type=Path, help="write the port here")
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument(
        "--chunk-delay", type=float, default=0.0, help="seconds between streamed chunks"
    )
    parser.add_argument("--seed", type=int, default=7)
    for knob, default in KNOBS.items():
        parser.add_argument(f"--{knob.replace('_', '-')}", type=float, default=default)
    args = parser.parse_args(argv)
    chaos = Chaos(args.seed, **{k: getattr(args, k) for k in KNOBS})
    server = make_server(chaos, args.port, args.dimensions, args.chunk_delay)
    port = server.server_address[1]
    if args.port_file:
        args.port_file.write_text(str(port), encoding="utf-8")
    print(f"chaos models on http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
