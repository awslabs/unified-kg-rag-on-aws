# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The load harness's arithmetic: latency summaries, resource trends and the
gates a nightly run must pass (docs/guides/load-testing.md).

Pure functions over plain lists and dicts (no package import), so
``tests/unit/test_loadtest_harness.py`` checks them without a model, a store
or a network.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# The resource samples a soak takes whose per-hour trend is reported; a
# steady rise in one of them over the second half is the leak signal.
TREND_KEYS: tuple[str, ...] = (
    "rss_kb",
    "fds",
    "sqlite_fds",
    "sockets",
    "os_threads",
    "gc_objects",
    "asyncio_tasks",
    "window_p95",
)


def percentile(values: Sequence[float], p: float) -> float:
    """The ``p`` quantile (0..1) of ``values``, interpolated linearly
    between the closest ranks; NaN for none."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _rounded(value: float, digits: int = 3) -> float | None:
    return None if math.isnan(value) else round(value, digits)


class Histogram:
    """Counts of values in fixed-width buckets: the same memory however long
    a run is. The event-loop ticker records a value every 10 ms, which a
    list would keep for the whole run and a soak would count as the
    process' own RSS growth."""

    def __init__(self, width: float = 0.001, buckets: int = 10_000) -> None:
        self.width = width
        self.counts = [0] * (buckets + 1)  # the last one holds every larger value
        self.count = 0
        self.max = 0.0

    def add(self, value: float) -> None:
        self.counts[min(int(value / self.width), len(self.counts) - 1)] += 1
        self.count += 1
        self.max = max(self.max, value)

    def percentile(self, p: float) -> float:
        """The upper edge of the bucket holding the ``p`` quantile (0..1),
        nearest rank, capped at the largest value (in the last bucket, the
        largest value: an upper bound); NaN for none."""
        if not self.count:
            return math.nan
        rank = max(1, math.ceil(p * self.count))
        seen = 0
        for index, n in enumerate(self.counts[:-1]):
            seen += n
            if seen >= rank:
                return min((index + 1) * self.width, self.max)
        return self.max


def summarize(
    latencies: Sequence[float],
    errors: Mapping[str, int],
    wall_seconds: float,
    flags: Mapping[str, int] | None = None,
    loop_lag: Histogram | None = None,
) -> dict[str, Any]:
    """One level's report: requests, throughput, latency quantiles, the
    errors by ``strategy:error_class`` and by class alone, the flags
    (degraded or empty answers) and the event-loop lag in milliseconds."""
    failed = sum(errors.values())
    requests = len(latencies) + failed
    by_class: Counter[str] = Counter()
    for key, count in errors.items():
        by_class[key.rsplit(":", 1)[-1]] += count
    lag = loop_lag or Histogram()
    return {
        "requests": requests,
        "ok": len(latencies),
        "errors": failed,
        "error_rate": round(failed / requests, 4) if requests else 0.0,
        "errors_by_class": dict(sorted(by_class.items())),
        "errors_by_strategy": dict(sorted(errors.items())),
        "flags": dict(sorted((flags or {}).items())),
        "throughput_rps": round(requests / wall_seconds, 2) if wall_seconds else 0.0,
        "p50": _rounded(percentile(latencies, 0.5)),
        "p95": _rounded(percentile(latencies, 0.95)),
        "p99": _rounded(percentile(latencies, 0.99)),
        "max": round(max(latencies), 3) if latencies else None,
        "loop_lag_p99_ms": _rounded(lag.percentile(0.99) * 1000, 1),
        "loop_lag_max_ms": round(lag.max * 1000, 1) if lag.count else None,
    }


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """The least-squares slope of ``ys`` over ``xs`` (0 for fewer than 3
    points or no spread in ``xs``)."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys differ in length")
    if len(xs) < 3:
        return 0.0
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    spread = sum((x - mean_x) ** 2 for x in xs)
    if not spread:
        return 0.0
    return (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / spread
    )


def trends_per_hour(
    samples: Sequence[Mapping[str, Any]], keys: Iterable[str] = TREND_KEYS
) -> dict[str, float]:
    """Per-hour slope of each numeric sample key over the second half of
    ``samples`` (the first half is warm-up: caches and pools fill). Each
    sample holds ``t``, its time in seconds; a key missing from a sample in
    that half is not reported."""
    half = samples[len(samples) // 2 :]
    trends: dict[str, float] = {}
    for key in keys:
        points = [
            (float(s["t"]), float(s[key]))
            for s in half
            if isinstance(s.get(key), (int, float)) and not isinstance(s[key], bool)
        ]
        if half and len(points) == len(half):
            xs, ys = zip(*points, strict=True)
            trends[key] = round(slope(xs, ys) * 3600, 2)
    return trends


@dataclass(frozen=True)
class Gates:
    """The thresholds a nightly run passes. Each is for a fixed corpus and
    the fake models: not a statement about real model latency."""

    max_errors: int = 0
    loop_lag_p99_ms: float = 100.0
    rss_slope_mb_per_hour: float = 10.0
    require_convergence: bool = True


def check_gates(report: Mapping[str, Any], gates: Gates = Gates()) -> list[str]:
    """The gate failures of a run report (``nightly.py``'s ``report.json``:
    ``sweep``, a list of level reports; ``soak``, a soak report; ``chaos``,
    scenario results with ``converged``); an empty list passes. A section
    the report does not hold is not checked; a section that holds no
    requests fails, so a run that measured nothing does not pass."""
    failures: list[str] = []
    levels = [*report.get("sweep", [])]
    soak = report.get("soak")
    if soak is not None:
        levels.append({"concurrency": f"soak {soak.get('concurrency')}", **soak})
    for level in levels:
        name = f"concurrency {level.get('concurrency')}"
        if not level.get("requests"):
            failures.append(f"{name}: no requests ran")
            continue
        if level.get("errors", 0) > gates.max_errors:
            failures.append(
                f"{name}: {level['errors']} errors (max {gates.max_errors}): "
                f"{level.get('errors_by_strategy')}"
            )
        lag = level.get("loop_lag_p99_ms")
        if lag is None:
            failures.append(f"{name}: no event-loop lag measured")
        elif lag >= gates.loop_lag_p99_ms:
            failures.append(
                f"{name}: event-loop lag p99 {lag} ms (limit {gates.loop_lag_p99_ms} ms)"
            )
    if soak is not None:
        rss = (soak.get("trends_per_hour") or {}).get("rss_kb")
        if rss is None:
            failures.append("soak: no RSS trend (too few samples)")
        elif rss / 1024 >= gates.rss_slope_mb_per_hour:
            failures.append(
                f"soak: RSS grows {rss / 1024:.1f} MB/h "
                f"(limit {gates.rss_slope_mb_per_hour} MB/h)"
            )
    if gates.require_convergence:
        for scenario in report.get("chaos", []):
            if not scenario.get("converged"):
                failures.append(
                    f"chaos {scenario.get('name')}: did not converge with a full "
                    f"build: {scenario.get('detail')}"
                )
    return failures
