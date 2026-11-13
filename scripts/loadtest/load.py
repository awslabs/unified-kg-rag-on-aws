# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive queries at a ``GraphRAGChain`` or the console and report latency,
throughput, errors and resources (docs/guides/load-testing.md).

Local only: it unsets every ``AWS_*`` variable, points the shared AWS config
and credentials files at the null device, turns the instance metadata lookup
off and refuses a config whose models or stores are not of a local kind on a
loopback address, or that screens with the Bedrock guardrail.

Modes:

- ``sweep``: after one warm-up pass, ``--queries`` queries at each
  concurrency of ``--levels``, in process. Per level: p50/p95/p99,
  throughput, errors by ``ErrorClass`` (and by strategy) with the latency
  of the failed queries, the answers without evidence (no sources,
  abstained, empty), degraded or with unresolved citations (``flags``), and
  the event-loop lag (a 10 ms ticker's overshoot: the
  time the loop spent in code that does not yield).
- ``soak``: ``--concurrency`` workers for ``--duration`` seconds, sampling
  RSS, file descriptors (SQLite files and sockets apart), threads, gc
  objects, asyncio tasks and conversations every ``--sample`` seconds
  (``--csv``), and the per-hour slope of each over the second half.
- ``http``: ``sweep`` through a running console's ``/api/ask`` (server-sent
  events), with the time to the first event; its loop lag is this
  client's (``client_loop_lag_*``), not the console's.
- ``gate``: check a report against the nightly gates (harness.Gates); exit 1
  naming each failure.

Usage::

    uv run --extra local python scripts/loadtest/load.py sweep \
        --config WS/config.yaml --levels 1,4,16 --queries 140
    uv run --extra local python scripts/loadtest/load.py soak \
        --config WS/config.yaml --concurrency 8 --duration 600 --memory-share 0.3
    uv run --extra local python scripts/loadtest/load.py http \
        --console http://127.0.0.1:8501 --levels 1,8 --queries 80   # token: UKR_CONSOLE_TOKEN
    uv run python scripts/loadtest/load.py gate report.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
import sys
import threading
import time
from array import array
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from harness import (
    Gates,
    Histogram,
    check_gates,
    local_only_env,
    percentile,
    second_half_rise,
    summarize,
    trends_per_hour,
)

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
# The provider kinds that run on this host (provider_config's built-in
# kinds): a store or model kind not named here (dynamodb, neptune, bedrock,
# a plugin's) is refused, whatever its endpoint.
LOCAL_KINDS: dict[str, frozenset[str]] = {
    "llm": frozenset({"ollama", "openai_compatible"}),
    "embedding": frozenset({"ollama", "openai_compatible"}),
    "rerank": frozenset({"cross_encoder", "none"}),
    "graph": frozenset({"sqlite", "gremlin"}),
    "vector": frozenset({"sqlite", "opensearch"}),
    "doc_status": frozenset({"sqlite"}),
}

# A query per strategy, in English, Korean and Japanese, each answerable from
# the demo corpus (console/demo/corpus.py).
QUERIES: tuple[tuple[str, str], ...] = (
    ("Who carries the lidar units to Kestrel Depot?", "mix"),
    ("What is the late-delivery penalty in the supply agreement?", "local"),
    ("How many lidar units did Orion Labs reject?", "ppr"),
    ("How long does Globex Sensors warrant each lidar unit?", "hybrid"),
    ("Who approved the lidar unit specification at Acme Robotics?", "simple"),
    ("엄브라 프레이트는 어디로 배송하나요?", "mix"),
    ("애크미 로보틱스와 글로벡스 센서스의 관계는?", "local"),
    ("オリオン・ラボは何を検査しましたか?", "naive"),
    ("What are the main themes of the corpus?", "global"),
    ("Summarize the supply chain.", "tree"),
    ("How are Acme Robotics and Umbra Freight connected?", "iterative"),
    ("How many organizations are there?", "structured"),
    ("What does Ravi Patel do?", "drift"),
    ("Tell me about the lidar unit", "auto"),
)


def query_mix(strategies: str = "") -> list[tuple[str, str]]:
    """The queries of the comma-separated ``strategies`` (all for none); a
    strategy without its own query asks each query with it."""
    if not strategies:
        return list(QUERIES)
    wanted = [s for s in strategies.split(",") if s]
    own = [(q, s) for q, s in QUERIES if s in wanted]
    missing = [s for s in wanted if s not in {s for _, s in own}]
    return own + [(q, s) for s in missing for q, _ in QUERIES]


def _host(endpoint: str) -> str | None:
    """The host of a URL or of a bare ``host[:port]`` endpoint."""
    return urlparse(endpoint if "//" in endpoint else f"//{endpoint}").hostname


def refuse_remote(config: Any) -> None:
    """Exit unless the config is ``backend: local``, every model and store
    is of a :data:`LOCAL_KINDS` kind (the registry only when it is on), no
    Bedrock guardrail screens queries and every endpoint it reaches is on a
    loopback address."""
    providers = config.providers
    if config.backend != "local":
        raise SystemExit("refusing AWS models or stores: use backend: local")
    for role, kinds in LOCAL_KINDS.items():
        section = getattr(providers, role)
        if role == "doc_status" and not section.enabled:
            continue
        if section.kind not in kinds:
            raise SystemExit(
                f"refusing providers.{role}.kind {section.kind!r}: not one of "
                f"{', '.join(sorted(kinds))}"
            )
    if config.aws.bedrock.guardrail.enabled:
        raise SystemExit("refusing the Bedrock guardrail: turn it off")
    endpoints = [
        getattr(section, "base_url", None)
        for section in (providers.llm, providers.embedding, providers.rerank)
    ]
    if providers.graph.kind == "gremlin":
        endpoints.append(config.aws.neptune.endpoint or "")
    if providers.vector.kind == "opensearch":
        endpoints.append(config.aws.opensearch.endpoint or "")
    for endpoint in endpoints:
        if endpoint is not None and _host(str(endpoint)) not in LOOPBACK:
            raise SystemExit(f"refusing the non-loopback endpoint {endpoint!r}")


@dataclass
class Tally:
    """What the workers of one level or soak recorded."""

    # 8 bytes a value (a list of floats takes 32): a soak keeps every one.
    latencies: array[float] = field(default_factory=lambda: array("d"))
    # When each failed query ended (a deadline shows here).
    error_latencies: array[float] = field(default_factory=lambda: array("d"))
    errors: Counter[str] = field(default_factory=Counter)
    flags: Counter[str] = field(default_factory=Counter)
    lags: Histogram = field(default_factory=Histogram)

    def report(self, wall: float) -> dict[str, Any]:
        return summarize(
            self.latencies,
            self.errors,
            wall,
            self.flags,
            self.lags,
            self.error_latencies,
        )

    def failed(self, strategy: str, error_class: str, started: float) -> None:
        self.errors[f"{strategy}:{error_class}"] += 1
        self.error_latencies.append(time.perf_counter() - started)

    def answered(
        self, strategy: str, answer: Mapping[str, Any], started: float
    ) -> None:
        """An answer's latency and its flags, from its ``AnswerView`` fields
        (the in-process view, or the console's result event): without
        evidence (``no_sources``, ``abstained``, ``empty_answer``), degraded,
        or with unresolved citations."""
        self.latencies.append(time.perf_counter() - started)
        flags = {
            "no_sources": not answer.get("sources"),
            "abstained": bool(answer.get("abstained")),
            "empty_answer": not str(answer.get("answer") or "").strip(),
            "degraded": bool(answer.get("degraded")),
            "citations_unresolved": bool(answer.get("citations_unresolved")),
        }
        for kind, raised in flags.items():
            if raised:
                self.flags[f"{kind}:{strategy}"] += 1


async def ask(
    chain: Any,
    query: str,
    strategy: str,
    tally: Tally,
    query_seconds: float,
    conversation: str | None = None,
) -> None:
    """One query: its latency on success (and its flags, :meth:`Tally.answered`),
    else its ``strategy:error_class`` and latency (a raised error, or the
    error answer of ``processing.ignore_errors``)."""
    from unified_kg_rag import RAGInput, classify_error
    from unified_kg_rag.application.presentation.answer_view import AnswerView

    started = time.perf_counter()
    try:
        output = await asyncio.wait_for(
            chain.ainvoke(
                RAGInput(
                    query=query,
                    search_strategy=strategy,
                    use_memory=conversation is not None,
                    conversation_id=conversation,
                )
            ),
            query_seconds,
        )
    except Exception as e:  # noqa: BLE001 - counted by class
        tally.failed(strategy, classify_error(e).value, started)
        return
    view = AnswerView.from_output(output)
    if view.error:
        tally.failed(strategy, view.error_class or "failure", started)
        return
    # The fields the flags read, not a dump of every source's text.
    fields = ("answer", "sources", "abstained", "degraded", "citations_unresolved")
    tally.answered(strategy, {k: getattr(view, k) for k in fields}, started)


async def with_loop_lag(
    tally: Tally, work: Awaitable[Any], tick: float = 0.01
) -> float:
    """Run ``work`` while a ticker records how late each ``tick`` sleep
    wakes (the event-loop lag); the wall time of ``work``."""
    done = asyncio.Event()

    async def ticker() -> None:
        while not done.is_set():
            before = time.perf_counter()
            await asyncio.sleep(tick)
            tally.lags.add(max(0.0, time.perf_counter() - before - tick))

    task = asyncio.create_task(ticker())
    started = time.perf_counter()
    try:
        await work
    finally:
        wall = time.perf_counter() - started
        done.set()
        await task
    return wall


async def run_level(
    chain: Any,
    concurrency: int,
    total: int,
    query_seconds: float,
    mix: list[tuple[str, str]],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    for _ in range(total):
        queue.put_nowait(rng.choice(mix))
    tally = Tally()

    async def worker() -> None:
        while not queue.empty():
            query, strategy = queue.get_nowait()
            await ask(chain, query, strategy, tally, query_seconds)

    wall = await with_loop_lag(
        tally, asyncio.gather(*(worker() for _ in range(concurrency)))
    )
    return {"concurrency": concurrency, **tally.report(wall)}


def process_sample(chain: Any) -> dict[str, Any]:
    """This process' resources now (Linux ``/proc``; elsewhere RSS and file
    descriptors are left out)."""
    sample: dict[str, Any] = {
        "threads": threading.active_count(),
        "gc_objects": len(gc.get_objects()),
        "asyncio_tasks": len(asyncio.all_tasks()),
    }
    proc = Path("/proc/self")
    if proc.exists():
        status = (proc / "status").read_text(encoding="utf-8")
        sample["rss_kb"] = int(
            next(
                line for line in status.splitlines() if line.startswith("VmRSS")
            ).split()[1]
        )
        targets = []
        for fd in os.listdir(proc / "fd"):
            try:
                targets.append(os.readlink(proc / "fd" / fd))
            except OSError:
                pass
        sample["fds"] = len(targets)
        sample["sqlite_fds"] = sum(".sqlite3" in t for t in targets)
        sample["sockets"] = sum(t.startswith("socket:") for t in targets)
        sample["os_threads"] = len(os.listdir(proc / "task"))
    memories = getattr(getattr(chain, "memory_manager", None), "_memories", None)
    if memories is not None:
        sample["conversations"] = len(memories)
    return sample


async def soak(
    chain: Any,
    concurrency: int,
    duration: float,
    every: float,
    query_seconds: float,
    mix: list[tuple[str, str]],
    memory_share: float,
    csv: Path | None,
) -> dict[str, Any]:
    stop = time.monotonic() + duration
    rng = random.Random(1)
    tally = Tally()
    window: list[float] = []
    samples: list[dict[str, Any]] = []
    conversations = 0

    async def worker() -> None:
        nonlocal conversations
        while time.monotonic() < stop:
            query, strategy = rng.choice(mix)
            conversation = None
            if rng.random() < memory_share:
                conversations += 1
                conversation = f"soak-{conversations}"
            done = len(tally.latencies)
            await ask(chain, query, strategy, tally, query_seconds, conversation)
            window.extend(tally.latencies[done:])

    async def sampler() -> None:
        started = time.monotonic()
        while time.monotonic() < stop:
            await asyncio.sleep(min(every, max(0.0, stop - time.monotonic())))
            gc.collect()
            sample = process_sample(chain)
            sample.update(
                t=round(time.monotonic() - started, 1),
                done=len(tally.latencies),
                errors=sum(tally.errors.values()),
                window_p50=round(percentile(window, 0.5), 3) if window else None,
                window_p95=round(percentile(window, 0.95), 3) if window else None,
            )
            window.clear()
            samples.append(sample)
            print(json.dumps(sample), file=sys.stderr, flush=True)

    wall = await with_loop_lag(
        tally, asyncio.gather(sampler(), *(worker() for _ in range(concurrency)))
    )
    if csv is not None:
        keys = sorted({k for s in samples for k in s})
        rows = [",".join(keys)]
        rows += [",".join(str(s.get(k, "")) for k in keys) for s in samples]
        await asyncio.to_thread(
            csv.write_text, "\n".join(rows) + "\n", encoding="utf-8"
        )
    return {
        "concurrency": concurrency,
        "duration_seconds": duration,
        **tally.report(wall),
        "trends_per_hour": trends_per_hour(samples),
        "second_half_rise": second_half_rise(samples),
        "first_sample": samples[0] if samples else None,
        "last_sample": samples[-1] if samples else None,
    }


async def http_level(
    base: str,
    token: str,
    concurrency: int,
    total: int,
    query_seconds: float,
    mix: list[tuple[str, str]],
    seed: int,
) -> dict[str, Any]:
    import httpx

    if urlparse(base).hostname not in LOOPBACK:
        raise SystemExit(f"refusing the non-loopback console {base}")
    rng = random.Random(seed)
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    for _ in range(total):
        queue.put_nowait(rng.choice(mix))
    tally = Tally()
    first_events: list[float] = []
    headers = {"X-Console-Token": token, "X-Console-Request": "1"}
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(
        base_url=base, headers=headers, timeout=query_seconds, limits=limits
    ) as client:

        async def one(query: str, strategy: str) -> None:
            started = time.perf_counter()
            first = None
            result: dict[str, Any] | None = None
            body = {"query": query, "search_strategy": strategy}
            async with client.stream("POST", "/api/ask", json=body) as response:
                if response.status_code != 200:
                    tally.failed(strategy, f"http_{response.status_code}", started)
                    await response.aread()
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    first = (
                        first if first is not None else time.perf_counter() - started
                    )
                    event = json.loads(line[6:])
                    if event.get("kind") in ("result", "error"):
                        result = event
            if result is None:
                tally.failed(strategy, "no_result", started)
            elif result["kind"] == "error" or result.get("error"):
                kind = (
                    result.get("error_kind") or result.get("error_class") or "failure"
                )
                tally.failed(strategy, kind, started)
            else:
                tally.answered(strategy, result, started)
                first_events.append(first or 0.0)

        async def worker() -> None:
            while not queue.empty():
                query, strategy = queue.get_nowait()
                started = time.perf_counter()
                try:
                    await one(query, strategy)
                except httpx.HTTPError as e:
                    tally.failed(strategy, type(e).__name__, started)

        wall = await with_loop_lag(
            tally, asyncio.gather(*(worker() for _ in range(concurrency)))
        )
    report = {"concurrency": concurrency, **tally.report(wall)}
    # The lag of this client's loop, not the console's: named apart.
    for key in ("loop_lag_p99_ms", "loop_lag_max_ms"):
        report[f"client_{key}"] = report.pop(key)
    report["first_event_p50"] = (
        round(percentile(first_events, 0.5), 3) if first_events else None
    )
    report["first_event_p95"] = (
        round(percentile(first_events, 0.95), 3) if first_events else None
    )
    return report


async def in_process(config_path: str, run: Callable[[Any], Awaitable[Any]]) -> Any:
    from unified_kg_rag import GraphRAGChain, get_config

    config = get_config(config_path)
    refuse_remote(config)
    chain = GraphRAGChain(config)
    try:
        return await run(chain)
    finally:
        await chain.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["sweep", "soak", "http", "gate"])
    parser.add_argument(
        "report", nargs="?", type=Path, help="gate: the report to check"
    )
    parser.add_argument("--config", help="sweep, soak: the workspace's config.yaml")
    parser.add_argument("--levels", default="1,4,16", help="sweep, http: concurrencies")
    parser.add_argument("--queries", type=int, default=100, help="per level")
    parser.add_argument("--concurrency", type=int, default=8, help="soak")
    parser.add_argument("--duration", type=float, default=600.0, help="soak seconds")
    parser.add_argument(
        "--sample", type=float, default=30.0, help="soak: seconds per sample"
    )
    parser.add_argument(
        "--memory-share",
        type=float,
        default=0.0,
        help="soak: share of queries that start a conversation",
    )
    parser.add_argument("--csv", type=Path, help="soak: write the samples here")
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="seconds per query"
    )
    parser.add_argument(
        "--strategies", default="", help="comma-separated (default: all)"
    )
    parser.add_argument("--console", help="http: the console's URL")
    parser.add_argument("--output", type=Path, help="also write the report here")
    parser.add_argument("--max-errors", type=int, default=Gates.max_errors)
    parser.add_argument("--max-loop-lag-ms", type=float, default=Gates.loop_lag_p99_ms)
    parser.add_argument(
        "--max-rss-mb-per-hour", type=float, default=Gates.rss_slope_mb_per_hour
    )
    parser.add_argument(
        "--rss-rise-floor-mb",
        type=float,
        default=Gates.rss_rise_floor_mb,
        help="gate: RSS fails only past both the rate and this rise",
    )
    parser.add_argument(
        "--max-flags",
        type=int,
        default=Gates.max_flags,
        help="gate: flagged answers per level (no sources, abstained, ...)",
    )
    args = parser.parse_args(argv)
    mix = query_mix(args.strategies)
    levels = [int(c) for c in args.levels.split(",") if c]
    report: dict[str, Any]

    if args.mode == "gate":
        if args.report is None:
            parser.error("gate needs the report path")
        gates = Gates(
            max_errors=args.max_errors,
            loop_lag_p99_ms=args.max_loop_lag_ms,
            rss_slope_mb_per_hour=args.max_rss_mb_per_hour,
            rss_rise_floor_mb=args.rss_rise_floor_mb,
            max_flags=args.max_flags,
        )
        failures = check_gates(
            json.loads(args.report.read_text(encoding="utf-8")), gates
        )
        for failure in failures:
            print(f"GATE FAILED: {failure}")
        print("gates passed" if not failures else f"{len(failures)} gate(s) failed")
        return 1 if failures else 0
    # Before anything builds a model or store client (gate reads a file).
    local = local_only_env(os.environ)
    os.environ.clear()
    os.environ.update(local)
    if args.mode == "http":
        token = os.environ.get("UKR_CONSOLE_TOKEN", "")
        if not args.console or not token:
            parser.error("http needs --console and the UKR_CONSOLE_TOKEN variable")
        report = {
            "http": [
                asyncio.run(
                    http_level(
                        args.console, token, c, args.queries, args.timeout, mix, c
                    )
                )
                for c in levels
            ]
        }
    else:
        if not args.config:
            parser.error(f"{args.mode} needs --config")

        async def run(chain: Any) -> dict[str, Any]:
            if args.mode == "soak":
                return {
                    "soak": await soak(
                        chain,
                        args.concurrency,
                        args.duration,
                        args.sample,
                        args.timeout,
                        mix,
                        args.memory_share,
                        args.csv,
                    )
                }
            warm = await run_level(chain, 1, len(mix), args.timeout, mix, 0)
            return {
                "warm_up": warm,
                "sweep": [
                    await run_level(chain, c, args.queries, args.timeout, mix, c)
                    for c in levels
                ],
            }

        report = asyncio.run(in_process(args.config, run))
    text = json.dumps(report, indent=1, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
