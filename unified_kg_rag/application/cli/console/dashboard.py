# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dashboard: usage, cost and latency of the questions asked through
the console and of the jobs it ran, per strategy, model and role.

Costs are priced with ``evaluation.prices`` (``answer_cost_usd``); a model
without a price leaves the cost out (``unpriced_models``) rather than
counting it as free.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.usage import LLMUsage, LLMUsageCounts
from unified_kg_rag.evaluation.comparison import answer_cost_usd
from unified_kg_rag.shared.utils.stats import quantile

_COUNTERS = (
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "cache_hits",
    "failed_calls",
    "rerank_calls",
)


def _add(target: dict[str, float], counts: Mapping[str, Any]) -> None:
    for name in _COUNTERS:
        target[name] = target.get(name, 0) + int(counts.get(name) or 0)


def _model_cost(model: str, counts: Mapping[str, Any], config: Config) -> float | None:
    if model not in config.evaluation.prices:
        return None
    usage = LLMUsage(by_model={model: LLMUsageCounts.model_validate(dict(counts))})
    usd, _ = answer_cost_usd(usage, config.evaluation.prices)
    return usd


def summarize(
    queries: list[dict[str, Any]],
    jobs: Iterable[dict[str, Any]],
    config: Config,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """The dashboard's figures over ``queries`` (store rows) and ``jobs``."""
    now = now or time.time()
    latencies = [
        q["latency_seconds"]
        for q in queries
        if q.get("latency_seconds") is not None and q["outcome"] != "cancelled"
    ]
    by_strategy: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "queries": 0,
            "abstained": 0,
            "errors": 0,
            "latencies": [],
            "cost_usd": 0.0,
            "priced": 0,
            "tokens": 0,
        }
    )
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    by_role: dict[str, dict[str, float]] = defaultdict(dict)
    days: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"queries": 0, "cost_usd": 0.0, "latencies": []}
    )
    unpriced: set[str] = set()
    total_cost = 0.0
    priced_queries = 0
    for query in queries:
        strategy = query.get("strategy") or query.get("requested_strategy") or "unknown"
        row = by_strategy[strategy]
        row["queries"] += 1
        row["abstained"] += int(bool(query.get("abstained")))
        row["errors"] += int(query.get("outcome") in ("failed", "error"))
        if query.get("latency_seconds") is not None:
            row["latencies"].append(query["latency_seconds"])
        usage = query.get("usage") or {}
        row["tokens"] += int(usage.get("input_tokens") or 0) + int(
            usage.get("output_tokens") or 0
        )
        if query.get("cost_usd") is not None:
            row["cost_usd"] += query["cost_usd"]
            row["priced"] += 1
            total_cost += query["cost_usd"]
            priced_queries += 1
        for model, counts in (usage.get("by_model") or {}).items():
            _add(by_model[model], counts)
            if (
                counts.get("llm_calls") or counts.get("rerank_calls")
            ) and model not in config.evaluation.prices:
                unpriced.add(model)
        for role, counts in (usage.get("by_role") or {}).items():
            _add(by_role[role], counts)
        day = datetime.fromtimestamp(query["created"]).strftime("%Y-%m-%d")
        days[day]["queries"] += 1
        days[day]["cost_usd"] += query.get("cost_usd") or 0.0
        if query.get("latency_seconds") is not None:
            days[day]["latencies"].append(query["latency_seconds"])

    strategies = [
        {
            "strategy": name,
            "queries": row["queries"],
            "abstained": row["abstained"],
            "errors": row["errors"],
            "latency_p50": quantile(row["latencies"], 0.5),
            "latency_p95": quantile(row["latencies"], 0.95),
            "tokens_per_query": row["tokens"] / row["queries"] if row["queries"] else 0,
            "cost_usd": row["cost_usd"] if row["priced"] else None,
            "cost_per_query": (
                row["cost_usd"] / row["priced"] if row["priced"] else None
            ),
        }
        for name, row in sorted(by_strategy.items(), key=lambda kv: -kv[1]["queries"])
    ]
    models = [
        {"model": model, **counts, "cost_usd": _model_cost(model, counts, config)}
        for model, counts in sorted(by_model.items())
    ]
    roles = [{"role": role, **counts} for role, counts in sorted(by_role.items())]
    timeline = [
        {
            "day": day,
            "queries": row["queries"],
            "cost_usd": row["cost_usd"],
            "latency_p50": quantile(row["latencies"], 0.5),
        }
        for day, row in sorted(days.items())
    ]
    job_rows = []
    for job in jobs:
        usage = _job_usage(job)
        duration = (
            (job.get("ended") or now) - job["started"] if job.get("started") else None
        )
        job_rows.append(
            {
                "id": job["id"],
                "kind": job["kind"],
                "status": job["status"],
                "created": job["created"],
                "duration_seconds": duration,
                "llm_calls": usage.get("llm_calls"),
                "tokens": (
                    (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                    if usage
                    else None
                ),
                "cost_usd": _job_cost(job),
            }
        )
    return {
        "totals": {
            "queries": len(queries),
            "answered": sum(q["outcome"] == "answered" for q in queries),
            "abstained": sum(bool(q.get("abstained")) for q in queries),
            "errors": sum(q["outcome"] in ("failed", "error") for q in queries),
            # Questions refused as asked (a namespace without an index, a
            # filter to change): the caller's to fix, not errors.
            "refused": sum(q["outcome"] == "refused" for q in queries),
            "latency_p50": quantile(latencies, 0.5),
            "latency_p95": quantile(latencies, 0.95),
            "latency_mean": statistics.fmean(latencies) if latencies else None,
            "cost_usd": total_cost if priced_queries else None,
            "priced_queries": priced_queries,
        },
        "strategies": strategies,
        "models": models,
        "roles": roles,
        "timeline": timeline,
        "jobs": job_rows,
        "unpriced_models": sorted(unpriced),
        "prices_configured": bool(config.evaluation.prices),
    }


def _job_usage(job: Mapping[str, Any]) -> dict[str, Any]:
    result = job.get("result") or {}
    if job.get("kind") == "ingest":
        return dict(result.get("token_usage") or {})
    total: dict[str, float] = {}
    for run in result.get("runs") or []:
        usage = run.get("usage") or {}
        # Per phase (answer_generation, scoring), each with the counters.
        phases = [usage] if "llm_calls" in usage else list(usage.values())
        for phase in phases:
            if isinstance(phase, dict):
                _add(total, phase)
    return total


def _job_cost(job: Mapping[str, Any]) -> float | None:
    """An evaluation job's USD from its comparison's cost block, when priced."""
    comparison = (job.get("result") or {}).get("comparison") or {}
    strategies = (comparison.get("cost") or {}).get("strategies") or {}
    totals = [
        s.get("usd_total")
        for s in strategies.values()
        if s.get("usd_total") is not None
    ]
    return float(sum(totals)) if totals else None
