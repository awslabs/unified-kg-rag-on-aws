# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The nightly load, soak and chaos run (.github/workflows/load-test.yml),
on the embedded SQLite stores and the fake models; nothing leaves the host.

1. Ingest the demo corpus with every methodology; ``load.py sweep`` and
   ``load.py soak`` query it.
2. Build a synthetic corpus once in full, as the reference.
3. Chaos, on that corpus in the same workspace, reset between scenarios
   (record ids name the source path):
   ``throttle`` (a share of model calls answered 429) and ``faults`` (503s
   and dropped connections), each followed by a clean run that must
   converge with the reference within what incremental indexing documents
   (``compare_exports.INCREMENTAL``); ``kill`` (SIGKILL in the middle of a
   run, then at once a resume under the same pipeline id, which takes the
   dead run's lease over: the SQLite registry hands on the lease of a holder
   of this host whose process has exited), which must converge exactly.
4. Check the gates (``harness.Gates``) and write ``report.json`` and
   ``report.md`` into the output directory; exit 1 when a gate fails.

Usage::

    uv run --extra local python scripts/loadtest/nightly.py OUT \
        [--soak-seconds 600] [--chaos-docs 60]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess  # nosec B404 - runs this repository's scripts and console scripts
import sys
import time
from pathlib import Path
from typing import Any

from chaos_models import call
from compare_exports import EXACT, INCREMENTAL, Tolerance, compare_files, converged
from harness import Gates, check_gates, local_only_env
from workspace import write_workspace

HERE = Path(__file__).resolve().parent
# What a workspace reset keeps.
KEEP = frozenset({"corpus", "config.yaml", "eval.json"})


def _env() -> dict[str, str]:
    return {**local_only_env(os.environ), "UKR__LOGGING__LEVEL": "WARNING"}


def _script(name: str) -> str:
    found = Path(sys.executable).with_name(name)
    if not found.exists():
        raise SystemExit(
            f"{name} not found next to {sys.executable}: run `uv sync` first"
        )
    return str(found)


class Models:
    """The chaos model server, in a process of its own."""

    def __init__(self, out: Path) -> None:
        port_file = out / "models.port"
        port_file.unlink(missing_ok=True)
        self.log = (out / "models.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(  # nosec B603 - fixed argv
            [
                sys.executable,
                str(HERE / "chaos_models.py"),
                "--port-file",
                str(port_file),
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 30
        while not port_file.exists() or not port_file.read_text(encoding="utf-8"):
            if time.monotonic() > deadline or self.process.poll() is not None:
                raise SystemExit("the chaos model server did not start (models.log)")
            time.sleep(0.1)
        self.port = int(port_file.read_text(encoding="utf-8"))

    def _call(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        status, answer = call(self.port, path, body)
        if status != 200:
            raise SystemExit(
                f"the chaos model server answered {status} to {path}: {answer}"
            )
        return answer

    def chaos(self, **knobs: float) -> None:
        reset = {"error_rate": 0.0, "drop_rate": 0.0, "hang_rate": 0.0, "delay": 0.0}
        self._call("/chaos", {**reset, **knobs})

    def chat_calls(self) -> int:
        return int(self._call("/stats")["chat"])

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)
        self.log.close()


def ingest(workspace: Path, tag: str, *extra: str) -> dict[str, Any]:
    """``run-ingestion`` in ``workspace``: its exit code, status and failure
    counts (its stdout JSON and its log beside the workspace, in
    ``<workspace>-ingest-<tag>.json`` and ``.log``, which a reset keeps)."""
    command = [
        _script("run-ingestion"),
        "--config-path",
        "config.yaml",
        "--source-directory",
        "corpus",
        "--output-format",
        "json",
        "--yes",
        *extra,
    ]
    stem = workspace.parent / f"{workspace.name}-ingest-{tag}"
    started = time.monotonic()
    with (
        stem.with_suffix(".json").open("w", encoding="utf-8") as out,
        stem.with_suffix(".log").open("w", encoding="utf-8") as log,
    ):
        code = subprocess.run(  # nosec B603 - fixed argv
            command, cwd=workspace, env=_env(), stdout=out, stderr=log, check=False
        ).returncode
    return {
        "tag": tag,
        "exit": code,
        "seconds": round(time.monotonic() - started, 1),
        **_outcome(stem.with_suffix(".json")),
    }


def _outcome(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": None}
    return {
        k: data.get(k)
        for k in ("status", "error_type", "failed_documents", "units_failed")
    }


def export(workspace: Path, path: Path) -> Path:
    """``run-describe --include-text`` of the workspace's namespace."""
    with (path.with_suffix(".log")).open("w", encoding="utf-8") as log:
        subprocess.run(  # nosec B603 - fixed argv
            [
                _script("run-describe"),
                "--config-path",
                "config.yaml",
                "--output-path",
                str(path),
                "--include-text",
                "--output-format",
                "json",
            ],
            cwd=workspace,
            env=_env(),
            stdout=subprocess.DEVNULL,
            stderr=log,
            check=True,
        )
    return path


def reset(workspace: Path) -> None:
    """Remove what runs wrote into the workspace (stores, caches, logs): the
    corpus and the config stay, so record ids, which name the source path,
    stay the same."""
    for entry in workspace.iterdir():
        if entry.name not in KEEP:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()


def load(mode: str, config: Path, out: Path, *args: str) -> dict[str, Any]:
    report = out / f"{mode}.json"
    with (out / f"{mode}.log").open("w", encoding="utf-8") as log:
        subprocess.run(  # nosec B603 - fixed argv
            [
                sys.executable,
                str(HERE / "load.py"),
                mode,
                "--config",
                str(config),
                "--output",
                str(report),
                *args,
            ],
            env=_env(),
            stdout=subprocess.DEVNULL,
            stderr=log,
            check=True,
        )
    result: dict[str, Any] = json.loads(report.read_text(encoding="utf-8"))
    return result


def killed_run(models: Models, ws: Path, after_calls: int) -> tuple[bool, int]:
    """Start an ingestion run (pipeline id ``kill``) and SIGKILL it once it
    made ``after_calls`` chat calls: whether the kill landed before the run
    ended, and the calls it made."""
    command = [
        _script("run-ingestion"),
        "--config-path",
        "config.yaml",
        "--source-directory",
        "corpus",
        "--output-format",
        "json",
        "--yes",
        "--pipeline-id",
        "kill",
    ]
    start = models.chat_calls()
    with (ws.parent / f"{ws.name}-ingest-killed.log").open(
        "w", encoding="utf-8"
    ) as log:
        process = subprocess.Popen(  # nosec B603 - fixed argv
            command, cwd=ws, env=_env(), stdout=subprocess.DEVNULL, stderr=log
        )
        while process.poll() is None and models.chat_calls() - start < after_calls:
            time.sleep(0.02)
        process.send_signal(signal.SIGKILL)
        killed = process.wait() == -signal.SIGKILL
    return killed, models.chat_calls() - start


def chaos_scenarios(models: Models, out: Path, docs: int) -> list[dict[str, Any]]:
    """Each scenario on the same corpus in the same workspace, reset between
    them, against a full build without faults."""
    ws = out / "chaos"
    write_workspace(
        ws,
        models.port,
        docs=docs,
        chars=2000,
        lang="mixed",
    )
    models.chaos()
    before = models.chat_calls()
    build = ingest(ws, "reference")
    full_calls = models.chat_calls() - before
    if build["exit"] != 0 or build["status"] != "completed":
        raise SystemExit(f"the reference build failed: {build}")
    reference = export(ws, out / "chaos-reference.jsonl")
    results = []

    def result(
        name: str, runs: list[dict[str, Any]], tolerance: Tolerance, problem: str = ""
    ) -> dict[str, Any]:
        diffs = compare_files(
            reference, export(ws, out / f"chaos-{name}.jsonl"), tolerance
        )
        detail = "; ".join(
            d.describe(limit=2).replace("\n", " |") for d in diffs if not d.same
        )
        return {
            "name": name,
            "runs": runs,
            "compared": "exact" if tolerance == EXACT else "incremental",
            "converged": converged(diffs) and not problem,
            "detail": "; ".join(p for p in (problem, detail) if p),
        }

    for name, knobs in (
        ("throttle", {"error_rate": 0.3, "error_status": 429}),
        ("faults", {"error_rate": 0.15, "error_status": 503, "drop_rate": 0.05}),
    ):
        reset(ws)
        models.chaos(**knobs)
        runs = [{**ingest(ws, f"{name}-faults"), "knobs": knobs}]
        models.chaos()
        runs.append(ingest(ws, f"{name}-recovery"))
        # The faulted run may fail (a stage over its failure budget exits 1);
        # the clean run after it must not, and must leave no document failed.
        recovery = runs[-1]
        problem = (
            f"the clean run after it exited {recovery['exit']}"
            if recovery["exit"]
            else (
                f"{recovery['failed_documents']} documents still failed"
                if recovery["failed_documents"]
                else ""
            )
        )
        results.append(result(name, runs, INCREMENTAL, problem))

    reset(ws)
    # Slowed down so the kill lands in the middle of a stage.
    models.chaos(delay=0.02)
    killed, calls = killed_run(models, ws, full_calls // 2)
    models.chaos()
    # At once: the killed run's lease is handed on, not waited out.
    resume = ingest(ws, "kill-resume", "--pipeline-id", "kill")
    resume.update(chat_calls_before_kill=calls, chat_calls_full_build=full_calls)
    if not killed:
        problem = "the run ended before the kill: raise --chaos-docs"
    elif resume["error_type"] == "NamespaceLeaseError":
        problem = "the resume found the killed run's lease held (NamespaceLeaseError)"
    elif resume["exit"]:
        problem = f"the resume exited {resume['exit']}"
    else:
        problem = ""
    results.append(result("kill", [resume], EXACT, problem))
    return results


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Nightly load test (embedded stores, fake models)",
        "",
        "Local numbers with fake models on one host: a regression signal, not "
        "product performance.",
        "",
        "| Run | Requests | Errors | Flagged | Throughput (rps) | p50 (s) | p99 (s) "
        "| Loop lag p99 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for level in [*report["sweep"], report["soak"]]:
        name = (
            "soak" if "duration_seconds" in level else "sweep"
        ) + f" c={level['concurrency']}"
        lines.append(
            f"| {name} | {level['requests']} | {level['errors']} "
            f"| {sum((level.get('flags') or {}).values())} | {level['throughput_rps']} "
            f"| {level['p50']} | {level['p99']} | {level['loop_lag_p99_ms']} |"
        )
    lines += [
        "",
        f"Soak trends per hour: `{json.dumps(report['soak']['trends_per_hour'])}`",
        "",
        "Soak rise over its second half: "
        f"`{json.dumps(report['soak'].get('second_half_rise'))}`",
        "",
    ]
    lines += ["| Chaos scenario | Converged | Detail |", "|---|---|---|"]
    lines += [
        f"| {c['name']} | {c['converged']} | {c['detail'] or ''} |"
        for c in report["chaos"]
    ]
    lines += [
        "",
        "Gates: " + ("passed" if not report["gate_failures"] else "FAILED"),
        "",
    ]
    lines += [f"- {failure}" for failure in report["gate_failures"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--sweep-levels", default="1,4,16")
    parser.add_argument("--sweep-queries", type=int, default=70)
    parser.add_argument("--soak-seconds", type=float, default=600.0)
    parser.add_argument("--soak-concurrency", type=int, default=8)
    parser.add_argument("--chaos-docs", type=int, default=60)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    models = Models(out)
    started = time.monotonic()
    try:
        demo = out / "queries"
        config = write_workspace(demo, models.port)
        build = ingest(demo, "full")
        if build["exit"] != 0:
            raise SystemExit(f"ingesting the demo corpus failed: {build}")
        sweep = load(
            "sweep",
            config,
            out,
            "--levels",
            args.sweep_levels,
            "--queries",
            str(args.sweep_queries),
        )
        soak = load(
            "soak",
            config,
            out,
            "--concurrency",
            str(args.soak_concurrency),
            "--duration",
            str(args.soak_seconds),
            "--sample",
            "10",
            "--memory-share",
            "0.3",
            "--csv",
            str(out / "soak.csv"),
        )
        chaos = chaos_scenarios(models, out, args.chaos_docs)
    finally:
        models.close()
    report: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
        },
        "seconds": round(time.monotonic() - started, 1),
        "ingest": build,
        "sweep": sweep["sweep"],
        "soak": soak["soak"],
        "chaos": chaos,
    }
    report["gate_failures"] = check_gates(report, Gates())
    (out / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
