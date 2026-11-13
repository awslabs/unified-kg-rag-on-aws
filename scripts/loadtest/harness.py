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
import os
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
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


def local_only_env(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` with nothing that reaches AWS: no ``AWS_*`` variable (keys,
    profile, region), no shared config or credentials file (an SSO cache or a
    profile in ``~/.aws`` would otherwise still sign requests) and no
    instance metadata lookup."""
    env = {k: v for k, v in environ.items() if not k.startswith("AWS_")}
    env.update(
        AWS_EC2_METADATA_DISABLED="true",
        AWS_CONFIG_FILE=os.devnull,
        AWS_SHARED_CREDENTIALS_FILE=os.devnull,
    )
    return env


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
    error_latencies: Sequence[float] = (),
) -> dict[str, Any]:
    """One level's report: requests, throughput, latency quantiles of the
    answers and of the errors (when each failed: a deadline shows there),
    the errors by ``strategy:error_class`` and by class alone, the flags
    (answers without evidence or degraded, ``kind:strategy``) and the
    event-loop lag in milliseconds."""
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
        "error_p99": _rounded(percentile(error_latencies, 0.99)),
        "error_max": round(max(error_latencies), 3) if error_latencies else None,
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


def _second_half(
    samples: Sequence[Mapping[str, Any]], keys: Iterable[str]
) -> Iterator[tuple[str, list[float], list[float]]]:
    """Each key's ``(times, values)`` over the second half of ``samples``
    (the first half is warm-up: caches and pools fill); a key missing from
    a sample in that half is left out."""
    half = samples[len(samples) // 2 :]
    for key in keys:
        points = [
            (float(s["t"]), float(s[key]))
            for s in half
            if isinstance(s.get(key), (int, float)) and not isinstance(s[key], bool)
        ]
        if half and len(points) == len(half):
            xs, ys = zip(*points, strict=True)
            yield key, list(xs), list(ys)


def trends_per_hour(
    samples: Sequence[Mapping[str, Any]], keys: Iterable[str] = TREND_KEYS
) -> dict[str, float]:
    """Per-hour slope of each numeric sample key over the second half of
    ``samples``. Each sample holds ``t``, its time in seconds; a key missing
    from a sample in that half is not reported."""
    return {
        key: round(slope(xs, ys) * 3600, 2)
        for key, xs, ys in _second_half(samples, keys)
    }


def second_half_rise(
    samples: Sequence[Mapping[str, Any]], keys: Iterable[str] = TREND_KEYS
) -> dict[str, float]:
    """How much each key rose over the second half of ``samples``: the
    least-squares slope times the half's span (a fit, so one noisy sample
    moves it little). With :func:`trends_per_hour` it tells a fast but small
    rise in a short run (allocator noise) from growth."""
    return {
        key: round(slope(xs, ys) * (xs[-1] - xs[0]), 2)
        for key, xs, ys in _second_half(samples, keys)
    }


@dataclass(frozen=True)
class Gates:
    """The thresholds a nightly run passes. Each is for a fixed corpus and
    the fake models: not a statement about real model latency.

    RSS fails only when it grows both faster than ``rss_slope_mb_per_hour``
    and by at least ``rss_rise_floor_mb`` over the soak's second half: a
    short soak's slope extrapolates a megabyte of allocator noise into tens
    of MB/h (a clean 10-minute soak measured 8 to 13.5 MB/h from about 1 MB
    over its second half), so the 10-minute nightly soak catches gross
    leaks only; a slow one needs a soak of 30 minutes or more, where the
    rate decides. The floor sits well above that noise, and a leak of
    about 1.2 KB a query passes it over the nightly soak's second half
    (about 6,900 queries at concurrency 8). Calibrating it on the runner
    (the slope distribution of repeated clean soaks) is still to do."""

    max_errors: int = 0
    loop_lag_p99_ms: float = 100.0
    rss_slope_mb_per_hour: float = 10.0
    rss_rise_floor_mb: float = 8.0
    # Answers flagged as without evidence (no sources, abstained, empty),
    # degraded or with unresolved citations: on the fixed corpus and the
    # deterministic fake models every query is answerable, so any is a
    # regression (an empty context answers "could not find" with no error).
    max_flags: int = 0
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
        flags: Counter[str] = Counter()
        for key, count in (level.get("flags") or {}).items():
            flags[key.split(":", 1)[0]] += count
        if flags.total() > gates.max_flags:
            failures.append(
                f"{name}: {flags.total()} flagged answers (max {gates.max_flags}): "
                + ", ".join(f"{kind} {n}" for kind, n in sorted(flags.items()))
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
        rise = (soak.get("second_half_rise") or {}).get("rss_kb")
        if rss is None or rise is None:
            failures.append("soak: no RSS trend (too few samples)")
        elif (
            rss / 1024 >= gates.rss_slope_mb_per_hour
            and rise / 1024 >= gates.rss_rise_floor_mb
        ):
            failures.append(
                f"soak: RSS grows {rss / 1024:.1f} MB/h, {rise / 1024:.1f} MB over "
                f"the second half (limits {gates.rss_slope_mb_per_hour} MB/h and "
                f"{gates.rss_rise_floor_mb} MB)"
            )
    if gates.require_convergence:
        for scenario in report.get("chaos", []):
            if not scenario.get("converged"):
                failures.append(
                    f"chaos {scenario.get('name')}: did not converge with a full "
                    f"build: {scenario.get('detail')}"
                )
    return failures
