# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console demo's fake model server (``console/demo/fake_models.py``)
with fault injection, for load and chaos runs (docs/guides/load-testing.md).

It binds 127.0.0.1 only. Each ``/api/chat`` and ``/api/embed`` request, in
this order, may wait (``delay`` plus up to ``jitter`` seconds), hang
(``hang_rate``, for ``hang_seconds``), lose its connection with no response
(``drop_rate``) or be answered with ``error_status`` (``error_rate``: 429
throttling, 500, 503). The knobs change at run time:
``POST /chaos {"error_rate": 0.3}`` answers the knobs now in force, and
``GET /stats`` the request counts, the faults injected and the requests in
flight.

Usage::

    uv run python scripts/loadtest/chaos_models.py --port 11613 \
        --error-rate 0.3 --error-status 429
"""

from __future__ import annotations

import argparse
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
}


class Chaos:
    """The knobs, the counters and the seeded draws of one server."""

    def __init__(self, seed: int = 7, **knobs: float) -> None:
        unknown = set(knobs) - set(KNOBS)
        if unknown:
            raise ValueError(f"unknown chaos knobs: {sorted(unknown)}")
        self.knobs: dict[str, float] = {**KNOBS, **knobs}
        self.stats: dict[str, int] = dict.fromkeys(
            ("chat", "embed", "hangs", "drops", "errors", "inflight", "max_inflight"), 0
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
                    self.close_connection = True
                    try:
                        self.connection.shutdown(2)
                    except OSError:
                        pass
                elif fault == "error":
                    self._drain()
                    self._json(
                        int(chaos.knobs["error_status"]), {"error": "injected fault"}
                    )
                else:
                    super().do_POST()
            finally:
                chaos.end()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


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
