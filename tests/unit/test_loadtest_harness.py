# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/loadtest: the load harness's statistics, its nightly gates, the
export comparison and the chaos model server.

The nightly workflow (.github/workflows/load-test.yml) runs the harness
itself; these keep its arithmetic and its verdicts right in every pull
request, without a model, a store or a network beyond loopback.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit

LOADTEST = Path(__file__).resolve().parents[2] / "scripts" / "loadtest"


def _import(name: str) -> ModuleType:
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(LOADTEST))
        return importlib.import_module(name)


harness = _import("harness")
compare_exports = _import("compare_exports")


# -- statistics -------------------------------------------------------------


def test_percentile_interpolates_between_ranks() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert harness.percentile(values, 0.0) == 1.0
    assert harness.percentile(values, 1.0) == 4.0
    assert harness.percentile(values, 0.5) == pytest.approx(2.5)
    assert harness.percentile(list(range(101)), 0.99) == pytest.approx(99.0)
    assert math.isnan(harness.percentile([], 0.5))


def _histogram(values: list[float]) -> Any:
    histogram = harness.Histogram()
    for value in values:
        histogram.add(value)
    return histogram


def test_the_histogram_reads_quantiles_to_its_bucket_width() -> None:
    lags = [i / 1000 for i in range(1, 1001)]  # 1 ms .. 1 s
    histogram = _histogram(lags)
    assert histogram.count == 1000 and histogram.max == 1.0
    width = histogram.width * 1.01  # to its bucket width
    assert histogram.percentile(0.5) == pytest.approx(0.5, abs=width)
    assert histogram.percentile(0.99) == pytest.approx(0.99, abs=width)
    assert histogram.percentile(1.0) == 1.0
    assert math.isnan(harness.Histogram().percentile(0.5))
    # a value past the last bucket counts there and reads as the largest one
    long = _histogram([0.001] * 98 + [30.0, 60.0])
    assert long.percentile(0.5) == pytest.approx(0.002)  # its bucket's upper edge
    assert long.percentile(0.99) == 60.0


def test_the_histogram_keeps_its_size() -> None:
    histogram = harness.Histogram()
    size = len(histogram.counts)
    for i in range(100_000):
        histogram.add((i % 500) / 1000)
    assert len(histogram.counts) == size and histogram.count == 100_000


def test_summarize_counts_errors_by_class_and_strategy() -> None:
    report = harness.summarize(
        [0.1, 0.2, 0.3],
        {"mix:timeout": 2, "local:timeout": 1, "local:store_unreachable": 1},
        wall_seconds=2.0,
        flags={"degraded:mix": 1},
        loop_lag=_histogram([0.001, 0.002, 0.050]),
    )
    assert report["requests"] == 7
    assert report["ok"] == 3
    assert report["errors"] == 4
    assert report["error_rate"] == pytest.approx(4 / 7, abs=1e-4)
    assert report["errors_by_class"] == {"store_unreachable": 1, "timeout": 3}
    assert report["errors_by_strategy"]["mix:timeout"] == 2
    assert report["throughput_rps"] == 3.5
    assert report["p50"] == 0.2
    assert report["max"] == 0.3
    assert report["flags"] == {"degraded:mix": 1}
    assert report["loop_lag_max_ms"] == 50.0
    assert 40 < report["loop_lag_p99_ms"] <= 50


def test_summarize_of_nothing_reports_no_latency() -> None:
    report = harness.summarize([], {}, wall_seconds=0.0)
    assert report["requests"] == 0
    assert report["p50"] is None and report["loop_lag_p99_ms"] is None
    json.dumps(report, allow_nan=False)  # a report stays valid JSON


def test_slope_is_the_least_squares_fit() -> None:
    assert harness.slope([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx(2.0)
    assert harness.slope([0, 1], [0, 5]) == 0.0  # too few points
    assert harness.slope([1, 1, 1], [0, 1, 2]) == 0.0  # no spread
    with pytest.raises(ValueError):
        harness.slope([0, 1, 2], [0, 1])


def test_trends_use_the_second_half_per_hour() -> None:
    # RSS jumps during warm-up, then grows 1 KB a second (3,600 KB an hour).
    samples = [{"t": t, "rss_kb": 1000 + t * 50} for t in range(0, 10)]
    samples += [{"t": t, "rss_kb": 5000 + t, "conversations": 3} for t in range(10, 20)]
    trends = harness.trends_per_hour(samples)
    assert trends["rss_kb"] == pytest.approx(3600.0)
    # a key not in every sample of the second half is not reported
    assert "fds" not in trends
    assert harness.trends_per_hour([]) == {}
    # the rise is the fit over the second half's span: 9 s at 1 KB a second
    assert harness.second_half_rise(samples)["rss_kb"] == pytest.approx(9.0)
    assert harness.second_half_rise([]) == {}


# -- gates ------------------------------------------------------------------


def _level(**overrides: Any) -> dict[str, Any]:
    return {
        "concurrency": 4,
        "requests": 100,
        "errors": 0,
        "errors_by_strategy": {},
        "flags": {},
        "loop_lag_p99_ms": 12.0,
        **overrides,
    }


def _soak(mb_per_hour: float, rise_mb: float, **overrides: Any) -> dict[str, Any]:
    return {
        **_level(concurrency=8),
        "trends_per_hour": {"rss_kb": mb_per_hour * 1024},
        "second_half_rise": {"rss_kb": rise_mb * 1024},
        **overrides,
    }


def _report(**overrides: Any) -> dict[str, Any]:
    return {
        "sweep": [_level(concurrency=1), _level(concurrency=16)],
        "soak": _soak(4.5, 1.5),
        "chaos": [{"name": "kill", "converged": True, "detail": ""}],
        **overrides,
    }


def test_a_clean_report_passes_the_gates() -> None:
    assert harness.check_gates(_report()) == []


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (
            _report(sweep=[_level(errors=1, errors_by_strategy={"mix:timeout": 1})]),
            "1 errors",
        ),
        (_report(sweep=[_level(loop_lag_p99_ms=100.0)]), "event-loop lag p99 100.0 ms"),
        (_report(sweep=[_level(loop_lag_p99_ms=None)]), "no event-loop lag"),
        (_report(sweep=[_level(requests=0)]), "no requests ran"),
        (_report(soak=_soak(10, 8)), "RSS grows 10.0 MB/h, 8.0 MB over"),
        (_report(soak=_soak(96, 8)), "RSS grows 96.0 MB/h"),
        (_report(soak={**_level(), "trends_per_hour": {}}), "no RSS trend"),
        (
            _report(soak={**_level(), "trends_per_hour": {"rss_kb": 1.0}}),
            "no RSS trend",
        ),
        (
            _report(sweep=[_level(flags={"no_sources:mix": 2, "abstained:mix": 1})]),
            "3 flagged answers (max 0)",
        ),
        (
            _report(chaos=[{"name": "throttle", "converged": False, "detail": "x"}]),
            "chaos throttle: did not converge",
        ),
    ],
)
def test_each_gate_fails_its_report(report: dict[str, Any], expected: str) -> None:
    failures = harness.check_gates(report)
    assert len(failures) == 1, failures
    assert expected in failures[0]


def test_a_short_soak_fails_on_rss_only_past_the_floor() -> None:
    # The clean 10-minute soak that failed the rate alone: about 1 MB over
    # its 300-second second half is 12 MB/h, and allocator noise.
    samples = [{"t": float(t), "rss_kb": 150_000.0} for t in range(0, 300, 10)]
    samples += [
        {"t": float(t), "rss_kb": 150_000 + (t - 300) * 1024 / 290}
        for t in range(300, 600, 10)
    ]
    trend = harness.trends_per_hour(samples)["rss_kb"] / 1024
    rise = harness.second_half_rise(samples)["rss_kb"] / 1024
    assert trend > 10 and rise == pytest.approx(1.0, abs=0.05)
    soak = _soak(trend, rise)
    assert harness.check_gates(_report(soak=soak)) == []
    # ... a real leak in the same run is past both
    assert harness.check_gates(_report(soak=_soak(120, 10)))
    # ... and a long soak growing slowly is past both too
    assert harness.check_gates(
        _report(soak=_soak(12, 9)), harness.Gates(rss_rise_floor_mb=8)
    )


def test_gate_thresholds_are_parameters() -> None:
    report = _report(sweep=[_level(errors=2)])
    assert harness.check_gates(report, harness.Gates(max_errors=2)) == []
    flagged = _report(sweep=[_level(flags={"degraded:mix": 1})])
    assert harness.check_gates(flagged, harness.Gates(max_flags=1)) == []
    assert (
        harness.check_gates(
            _report(soak=_soak(12, 9)), harness.Gates(rss_rise_floor_mb=10)
        )
        == []
    )
    no_convergence = harness.Gates(require_convergence=False)
    failing = _report(chaos=[{"name": "kill", "converged": False}])
    assert harness.check_gates(failing, no_convergence) == []


# -- export comparison ------------------------------------------------------


def _export(*records: dict[str, Any]) -> list[str]:
    return [json.dumps(r) for r in records] + [""]


ENTITY = {
    "kind": "entity",
    "id": "e1",
    "name": "Acme Robotics",
    "attributes": {
        "entity_keys": ["acme robotics"],
        "summarized_from": "4385",
        "summarized_with": "cf82",
    },
    "community_ids": ["c1"],
    "description": "A robot maker.",
    "text_unit_ids": ["t1", "t2"],
    "embedding_digests": {"name_embedding": "aa", "description_embedding": "bb"},
    "updated_at": "2026-01-01",
}
COMMUNITY = {"kind": "community", "id": "c1", "name": "Acme network", "rank": 3}


def _compare(
    a: list[dict[str, Any]], b: list[dict[str, Any]], tolerance: Any = None
) -> list[Any]:
    index = compare_exports.index_records
    tolerance = tolerance or compare_exports.EXACT
    return compare_exports.compare(index(_export(*a)), index(_export(*b)), tolerance)


def test_exports_converge_without_their_volatile_keys() -> None:
    later = {**ENTITY, "updated_at": "2026-02-02", "pipeline_id": "p2"}
    diffs = _compare([ENTITY, COMMUNITY], [later, COMMUNITY])
    assert compare_exports.converged(diffs)
    assert [d.kind for d in diffs] == ["community", "entity"]


def test_a_missing_record_and_a_changed_key_are_differences() -> None:
    changed = {**ENTITY, "description": "Another text."}
    diffs = _compare([ENTITY, COMMUNITY], [changed])
    assert not compare_exports.converged(diffs)
    by_kind = {d.kind: d for d in diffs}
    assert by_kind["community"].only_a == ["c1"]
    assert by_kind["entity"].changed == {"e1": ["description"]}
    assert "only in A: c1" in by_kind["community"].describe()


def test_exact_comparison_keeps_list_order() -> None:
    reordered = {**ENTITY, "text_unit_ids": ["t2", "t1"]}
    assert not compare_exports.converged(_compare([ENTITY], [reordered]))


def test_incremental_tolerance_is_the_documented_one() -> None:
    recovered = {
        **ENTITY,
        "text_unit_ids": ["t2", "t1"],
        "description": "A robot maker.\nIt buys lidar units.",
        "embedding_digests": {"name_embedding": "aa", "description_embedding": "cc"},
        "attributes": {"entity_keys": ["acme robotics"]},
        "community_ids": ["c2"],
    }
    other_community = {**COMMUNITY, "name": "Another cut", "rank": 5}
    incremental = compare_exports.INCREMENTAL
    assert compare_exports.converged(
        _compare([ENTITY, COMMUNITY], [recovered, other_community], incremental)
    )
    # ... but not past it: a lost text unit, a changed name embedding.
    lost = {**recovered, "text_unit_ids": ["t1"]}
    stale = {**recovered, "embedding_digests": {"name_embedding": "zz"}}
    rekeyed = {**recovered, "attributes": {"entity_keys": ["acme"]}}
    for record in (lost, stale, rekeyed):
        assert not compare_exports.converged(_compare([ENTITY], [record], incremental))


def test_nothing_built_does_not_converge() -> None:
    assert not compare_exports.converged(_compare([], []))


@pytest.mark.parametrize(
    "lines",
    [
        [json.dumps({"kind": "entity"})],
        [json.dumps(ENTITY), json.dumps(ENTITY)],
    ],
)
def test_a_record_without_an_id_or_twice_is_refused(lines: list[str]) -> None:
    with pytest.raises(ValueError):
        compare_exports.index_records(lines)


def test_the_cli_exits_by_the_verdict(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(_export(ENTITY)), encoding="utf-8")
    b.write_text(
        "\n".join(_export({**ENTITY, "text_unit_ids": ["t2", "t1"]})), encoding="utf-8"
    )
    assert compare_exports.main([str(a), str(a)]) == 0
    assert compare_exports.main([str(a), str(b)]) == 1
    assert compare_exports.main([str(a), str(b), "--incremental"]) == 0
    assert compare_exports.main([str(a), str(tmp_path / "missing.jsonl")]) == 2


# -- the load driver's guards -----------------------------------------------


@pytest.fixture(scope="module")
def load() -> ModuleType:
    return _import("load")


def _config(
    backend: str = "local", guardrail: bool = False, **providers: Any
) -> SimpleNamespace:
    section = SimpleNamespace(kind="ollama", base_url="http://127.0.0.1:11500")
    defaults = {
        "llm": section,
        "embedding": section,
        "rerank": SimpleNamespace(kind="none"),
        "graph": SimpleNamespace(kind="sqlite"),
        "vector": SimpleNamespace(kind="sqlite"),
        "doc_status": SimpleNamespace(kind="sqlite", enabled=True),
    }
    aws = SimpleNamespace(
        neptune=SimpleNamespace(endpoint="localhost"),
        opensearch=SimpleNamespace(
            endpoint="https://search-x.us-west-2.es.amazonaws.com"
        ),
        bedrock=SimpleNamespace(guardrail=SimpleNamespace(enabled=guardrail)),
    )
    return SimpleNamespace(
        backend=backend, aws=aws, providers=SimpleNamespace(**{**defaults, **providers})
    )


def test_the_driver_refuses_anything_not_local(load: ModuleType) -> None:
    load.refuse_remote(_config())
    # the compose stores are on localhost
    load.refuse_remote(_config(graph=SimpleNamespace(kind="gremlin")))
    # a registry that is off is never reached, whatever its kind
    load.refuse_remote(
        _config(doc_status=SimpleNamespace(kind="dynamodb", enabled=False))
    )
    remote = SimpleNamespace(kind="ollama", base_url="http://models.example.com:11434")
    for config in (
        _config(doc_status=SimpleNamespace(kind="dynamodb", enabled=True)),
        _config(graph=SimpleNamespace(kind="neptune")),  # under backend: local
        _config(vector=SimpleNamespace(kind="my_plugin_store")),
        _config(guardrail=True),
        _config(llm=remote),
        _config(embedding=SimpleNamespace(kind="bedrock")),
        _config(rerank=SimpleNamespace(kind="bedrock")),
        _config(vector=SimpleNamespace(kind="opensearch")),  # an AWS domain
        _config(graph=SimpleNamespace(kind="neptune"), backend="aws"),
        _config(backend="aws"),
    ):
        with pytest.raises(SystemExit):
            load.refuse_remote(config)


def test_the_environment_reaches_no_aws_account() -> None:
    env = harness.local_only_env(
        {
            "AWS_PROFILE": "prod",
            "AWS_REGION": "us-east-1",
            "AWS_SECRET_ACCESS_KEY": "x",
            "HOME": "/home/u",
        }
    )
    assert env["HOME"] == "/home/u" and "AWS_PROFILE" not in env
    assert "AWS_REGION" not in env and "AWS_SECRET_ACCESS_KEY" not in env
    # the profiles and SSO cache of ~/.aws are not read
    assert env["AWS_CONFIG_FILE"] == env["AWS_SHARED_CREDENTIALS_FILE"] == os.devnull
    assert env["AWS_EC2_METADATA_DISABLED"] == "true"


def test_the_query_mix_covers_every_strategy_and_language(load: ModuleType) -> None:
    from unified_kg_rag.domain.models import SearchStrategy

    strategies = {s for _, s in load.QUERIES}
    assert strategies == {s.value for s in SearchStrategy}
    text = " ".join(q for q, _ in load.QUERIES)
    assert any("가" <= c <= "힣" for c in text)  # Korean
    assert any("゠" <= c <= "ヿ" for c in text)  # Japanese katakana
    assert {s for _, s in load.query_mix("ppr,mix")} == {"ppr", "mix"}
    # a strategy without a query of its own asks every query
    assert len(load.query_mix("ppr")) == 1


def test_an_answer_without_evidence_is_flagged(load: ModuleType) -> None:
    tally = load.Tally()
    started = time.perf_counter()
    source = {"kind": "text", "text": "Umbra Freight delivers"}
    tally.answered("mix", {"answer": "Umbra Freight.", "sources": [source]}, started)
    assert not tally.flags
    # the answer of an empty context: no error, no source, abstained
    abstained = {"answer": "I could not find it.", "sources": [], "abstained": True}
    tally.answered("local", abstained, started)
    tally.answered(
        "ppr", {"answer": " ", "sources": [source], "degraded": [{}]}, started
    )
    assert dict(tally.flags) == {
        "no_sources:local": 1,
        "abstained:local": 1,
        "empty_answer:ppr": 1,
        "degraded:ppr": 1,
    }
    tally.failed("tree", "timeout", started)
    report = tally.report(1.0)
    assert report["ok"] == 3 and report["errors_by_strategy"] == {"tree:timeout": 1}
    assert report["error_max"] is not None and report["error_max"] >= 0
    failures = harness.check_gates({"sweep": [{**report, "concurrency": 1}]})
    assert any(
        "4 flagged answers (max 0): abstained 1, degraded 1" in f for f in failures
    )


def test_the_gate_mode_exits_one_on_a_failed_gate(
    load: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    assert load.main(["gate", str(path)]) == 0
    path.write_text(json.dumps(_report(sweep=[_level(errors=3)])), encoding="utf-8")
    assert load.main(["gate", str(path)]) == 1
    assert load.main(["gate", str(path), "--max-errors", "3"]) == 0
    path.write_text(json.dumps(_report(soak=_soak(30, 9))), encoding="utf-8")
    assert load.main(["gate", str(path)]) == 1
    assert load.main(["gate", str(path), "--rss-rise-floor-mb", "10"]) == 0


# -- the chaos model server -------------------------------------------------


@pytest.fixture(scope="module")
def chaos_models() -> ModuleType:
    return _import("chaos_models")


@pytest.fixture
def server(chaos_models: ModuleType) -> Iterator[tuple[Any, int]]:
    chaos = chaos_models.Chaos(seed=1)
    httpd = chaos_models.make_server(chaos)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield chaos, httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_server_binds_loopback_and_answers_as_the_fake_models(
    chaos_models: ModuleType, server: tuple[Any, int]
) -> None:
    chaos, port = server
    status, body = chaos_models.call(port, "/api/embed", {"input": ["lidar unit"]})
    assert status == 200 and len(body["embeddings"][0]) == 64
    assert chaos.stats["embed"] == 1


def test_the_server_listens_on_loopback_only(chaos_models: ModuleType) -> None:
    httpd = chaos_models.make_server(chaos_models.Chaos())
    try:
        assert httpd.server_address[0] == "127.0.0.1"
    finally:
        httpd.server_close()


def test_injected_errors_change_at_run_time(
    chaos_models: ModuleType, server: tuple[Any, int]
) -> None:
    chaos, port = server
    call = chaos_models.call
    status, knobs = call(port, "/chaos", {"error_rate": 1.0, "error_status": 503})
    assert status == 200 and knobs["error_rate"] == 1.0
    assert call(port, "/api/embed", {"input": ["x"]})[0] == 503
    call(port, "/chaos", {"error_rate": 0.0})
    assert call(port, "/api/embed", {"input": ["x"]})[0] == 200
    assert chaos.stats["embed"] == 2 and chaos.stats["errors"] == 1
    # the handler counts a request done after its response is sent
    deadline = time.monotonic() + 5
    while chaos.stats["inflight"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert chaos.stats["inflight"] == 0
    assert call(port, "/stats")[1]["errors"] == 1


def test_an_unknown_knob_is_refused(
    chaos_models: ModuleType, server: tuple[Any, int]
) -> None:
    _, port = server
    status, body = chaos_models.call(port, "/chaos", {"eror_rate": 1.0})
    assert status == 400 and "eror_rate" in body["error"]


def test_a_dropped_connection_gets_no_response(
    chaos_models: ModuleType, server: tuple[Any, int]
) -> None:
    chaos, port = server
    chaos.update({"drop_rate": 1.0})
    with pytest.raises((ConnectionError, OSError)):
        chaos_models.call(port, "/api/embed", {"input": ["x"]})
    assert chaos.stats["drops"] == 1


def test_the_fault_draws_follow_the_knobs(chaos_models: ModuleType) -> None:
    chaos = chaos_models.Chaos(seed=3, hang_rate=1.0, hang_seconds=5.0, delay=0.5)
    wait, fault = chaos.begin("chat")
    assert fault == "hang" and wait == pytest.approx(5.5)
    chaos.end()
    never = chaos_models.Chaos(seed=3)
    assert all(never.begin("chat")[1] is None for _ in range(50))
    with pytest.raises(ValueError):
        chaos_models.Chaos(typo=1.0)


# -- the nightly run's pieces -----------------------------------------------


@pytest.mark.timeout(240)
def test_the_nightly_workspace_ingests_and_exports(tmp_path: Path) -> None:
    """What the nightly composes, end to end on a small corpus: the model
    server process, a workspace (every methodology), one ingestion run and
    its export, which converges with itself and holds every methodology's
    artifacts; a reset keeps the corpus and config only. A signature or
    config change that breaks the nightly fails here, in the pull request."""
    nightly = _import("nightly")
    models = nightly.Models(tmp_path)
    try:
        ws = tmp_path / "ws"
        config = nightly.write_workspace(ws, models.port, docs=3, lang="mixed")
        from unified_kg_rag import get_config

        assert get_config(str(config)).backend == "local"
        build = nightly.ingest(ws, "full")
        assert build["exit"] == 0 and build["status"] == "completed", build
        assert models.chat_calls() > 0
        a = nightly.export(ws, tmp_path / "a.jsonl")
        b = nightly.export(ws, tmp_path / "b.jsonl")
    finally:
        models.close()
    diffs = compare_exports.compare_files(a, b)
    assert compare_exports.converged(diffs)
    kinds = {d.kind for d in diffs}
    assert {"text_unit", "entity", "relationship", "community"} <= kinds
    assert "summary_node" in kinds  # raptor's tree is compared too
    nightly.reset(ws)
    assert {p.name for p in ws.iterdir()} == set(nightly.KEEP)
    report = {
        "sweep": [{**_level(), "throughput_rps": 1.0, "p50": 0.1, "p99": 0.2}],
        "soak": {
            **_soak(1, 0.5),
            "duration_seconds": 60,
            "throughput_rps": 1.0,
            "p50": 0.1,
            "p99": 0.2,
        },
        "chaos": [{"name": "kill", "converged": True, "detail": ""}],
        "gate_failures": [],
    }
    assert "Gates: passed" in nightly.markdown(report)
