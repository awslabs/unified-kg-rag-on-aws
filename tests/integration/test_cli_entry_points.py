# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every console script runs in a process of its own, as a user runs it.

An in-process test shares the test session's imports: a CLI that reads a
registry its own imports never fill (``run-describe`` and the strategy
registry) passed every such test and failed for every user. Here each
script of ``[project.scripts]`` runs from this environment's ``bin``
directory in a clean environment (no ``PYTHONPATH``, no ``UKR__``
variables), once with ``--help`` and once for real, against the console
demo's workspace (``console/demo``: its synthetic corpus and dataset) on
the embedded SQLite stores, with the demo's fake model server (no model,
no network, no AWS). A script without its extra installed must exit 2
naming the extra instead.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess  # nosec B404 - runs this environment's console scripts
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.console_demo import CONSOLE_INSTALLED, load
from tests.fixtures.extras import MCP_INSTALLED

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(300)]

ROOT = Path(__file__).parents[2]
TOKEN = "test-token-0123456789abcdef"
# What run-prompt-tuning's corpus profile reads from the model.
PROFILE = {
    "domain": "procurement",
    "language": "English",
    "persona": "You are a procurement analyst.",
    "entity_types": ["ORGANIZATION", "PERSON"],
}


def _scripts() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as file:
        scripts: dict[str, str] = tomllib.load(file)["project"]["scripts"]
    return scripts


def _executable(name: str) -> str:
    found = Path(sys.executable).with_name(name)
    assert found.is_file(), f"{name} is not installed next to {sys.executable}"
    return str(found)


@pytest.fixture(scope="module")
def models() -> Iterator[str]:
    """The demo's fake model server, which also profiles a corpus (the
    prompt-tuning prompt); its base URL."""
    fake = load("fake_models")

    class Models(fake.FakeModels):  # type: ignore[name-defined,misc]
        def chat(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
            system = "".join(
                m.get("content", "")
                for m in body.get("messages") or []
                if m.get("role") == "system"
            )
            if "expert corpus analyst" not in system:
                yield from super().chat(body)
                return
            yield {
                "model": body.get("model"),
                "message": {"role": "assistant", "content": json.dumps(PROFILE)},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            }

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), fake.make_handler(Models(chunk_delay=0.0))
    )
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _environment(home: Path) -> dict[str, str]:
    """A clean environment: no PYTHONPATH, no UKR__ setting, no AWS."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "AWS_EC2_METADATA_DISABLED": "true",
        "UKR_CONSOLE_TOKEN": TOKEN,
    }


def _run(
    workspace: Path, name: str, *args: str, timeout: float = 180
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 - this environment's console script
        [_executable(name), *args],
        cwd=workspace,
        env=_environment(workspace),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _ok(result: subprocess.CompletedProcess[str]) -> subprocess.CompletedProcess[str]:
    assert result.returncode == 0, (
        result.args,
        result.stdout[-2000:],
        result.stderr[-4000:],
    )
    return result


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory, models: str) -> Path:
    """The demo workspace, ingested by run-ingestion (its real invocation)."""
    demo = load("demo")
    root = tmp_path_factory.mktemp("cli")
    demo.write_corpus(root)
    config = demo.demo_config("fake", model_base_url=models, region=None)
    config["logging"] = {"log_to_file": False}
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    ingested = _ok(
        _run(
            root,
            "run-ingestion",
            "--config-path",
            "config.yaml",
            "--source-directory",
            "corpus",
            "--cache-directory",
            "cache",
            "--yes",
        )
    )
    # The run names the graph it exported for run-visualization.
    assert "Graph for run-visualization:" in ingested.stdout
    return root


def _json(result: subprocess.CompletedProcess[str]) -> Any:
    return json.loads(_ok(result).stdout)


def test_every_script_prints_its_help(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(
            zip(
                _scripts(),
                pool.map(lambda name: _run(tmp_path, name, "--help"), _scripts()),
                strict=True,
            )
        )
    for name, result in results.items():
        _ok(result)
        assert result.stdout.startswith(f"usage: {name}"), (name, result.stdout[:200])


def test_every_script_has_a_real_invocation_here() -> None:
    # A new console script adds its invocation to this module.
    assert set(_scripts()) == set(INVOCATIONS)


def _rag(workspace: Path) -> None:
    answer = _json(
        _run(
            workspace,
            "run-rag",
            "--config-path",
            "config.yaml",
            "--query",
            "Who carries the lidar units to Kestrel Depot?",
            "--search-strategy",
            "local",
            "--output-format",
            "json",
        )
    )
    assert answer["success"] and "Umbra Freight" in answer["answer"]
    assert answer["sources"]


def _describe(workspace: Path) -> None:
    described = _json(
        _run(
            workspace,
            "run-describe",
            "--config-path",
            "config.yaml",
            "--output-path",
            "graph.jsonl",
            "--output-format",
            "json",
        )
    )
    assert {"local", "mix", "global"} <= set(described["strategies"])
    assert described["auto_routes"]
    assert described["registry"] == {"processed": len(load("corpus").documents())}
    assert described["export"]["counts"]["entity"] > 0
    assert (workspace / "graph.jsonl").is_file()


def _eval(workspace: Path) -> None:
    summary = _json(
        _run(
            workspace,
            "run-eval",
            "--config-path",
            "config.yaml",
            "--data-path",
            "datasets/demo-eval.json",
            "--search-strategy",
            "local",
            "--output-format",
            "json",
        )
    )
    (run,) = summary["runs"]
    assert run["strategy"] == "local"
    assert run["total_queries"] == len(load("corpus").DATASET)
    assert run["failed_evaluations"] == 0
    assert (workspace / summary["output_directory"]).is_dir()


def _visualization(workspace: Path) -> None:
    assert list((workspace / "cache").glob("*/visualization/visualization_data.json"))
    # Without --data-path: the export of the namespace in the cache shows
    # the build its stores hold.
    shown = _json(
        _run(
            workspace,
            "run-visualization",
            "--config-path",
            "config.yaml",
            "--suffix",
            "default",
            "--output-directory",
            "viz",
            "--renderers",
            "static",
            "--output-format",
            "json",
        )
    )
    assert list((workspace / "viz").glob("*.html"))
    assert Path(shown["data_path"]).parent.parent.parent.name == "cache"
    # Without a config naming the stores: the newest export, with a warning.
    _ok(_run(workspace, "run-visualization", "--output-directory", "viz2"))
    missing = _run(
        workspace,
        "run-visualization",
        "--config-path",
        "config.yaml",
        "--suffix",
        "nope",
    )
    assert missing.returncode == 2 and "'nope' holds no graph" in missing.stderr


def _prompt_tuning(workspace: Path) -> None:
    tuned = _json(
        _run(
            workspace,
            "run-prompt-tuning",
            "--config-path",
            "config.yaml",
            "--source-directory",
            "corpus",
            "--output-path",
            "tuned.yaml",
            "--max-docs",
            "3",
            "--output-format",
            "json",
        )
    )
    assert tuned["profile"]["domain"] == PROFILE["domain"]
    assert yaml.safe_load((workspace / "tuned.yaml").read_text(encoding="utf-8"))


def _benchmark(workspace: Path) -> None:
    prepared = _json(
        _run(
            workspace,
            "prepare-benchmark",
            "structural",
            "--output-directory",
            "bench",
            "--output-format",
            "json",
        )
    )
    assert prepared["questions"] > 0
    assert (workspace / "bench" / "eval_data.json").is_file()


def _missing_extra(workspace: Path, name: str, extra: str) -> None:
    result = _run(workspace, name, "--config-path", "config.yaml")
    assert result.returncode == 2, (result.stdout[-2000:], result.stderr[-2000:])
    assert f"--extra {extra}" in result.stdout + result.stderr


_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}


def _mcp_server(workspace: Path) -> None:
    if not MCP_INSTALLED:
        _missing_extra(workspace, "run-mcp-server", "mcp")
        return
    messages = [
        _INITIALIZE,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "kg_describe_schema", "arguments": {}},
        },
    ]
    process = subprocess.Popen(  # nosec B603 - this environment's console script
        [_executable("run-mcp-server"), "--config-path", "config.yaml"],
        cwd=workspace,
        env=_environment(workspace),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for message in messages:
            process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()
        replies: dict[Any, Any] = {}
        while 2 not in replies:
            line = process.stdout.readline()
            assert line, "the server closed stdout"
            reply = json.loads(line)
            replies[reply.get("id")] = reply
        process.stdin.close()
        code = process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    stderr = process.stderr.read() if process.stderr else ""
    assert code == 0, stderr[-4000:]
    described = replies[2]["result"]
    assert not described.get("isError"), described
    assert "local" in described["structuredContent"]["strategies"]


def _get(url: str) -> Any:
    request = urllib.request.Request(url, headers={"X-Console-Token": TOKEN})
    with urllib.request.urlopen(
        request, timeout=60
    ) as response:  # nosec B310 - loopback
        return json.loads(response.read())


def _when_listening(url: str, process: subprocess.Popen[str]) -> Any:
    """``_get(url)`` once the server listens (it prints its address on
    binding, before it accepts)."""
    deadline = time.monotonic() + 60
    while True:
        try:
            return _get(url)
        except urllib.error.URLError:
            assert process.poll() is None, "run-console exited"
            assert time.monotonic() < deadline, "run-console never listened"
            time.sleep(0.2)


def _console(workspace: Path) -> None:
    if not CONSOLE_INSTALLED:
        _missing_extra(workspace, "run-console", "console")
        return
    process = subprocess.Popen(  # nosec B603 - this environment's console script
        [
            _executable("run-console"),
            "--config-path",
            "config.yaml",
            "--source-directory",
            ".",
            "--port",
            "0",
            "--output-format",
            "json",
        ],
        cwd=workspace,
        env=_environment(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    try:
        lines = []
        while True:
            line = process.stdout.readline()
            assert line, "run-console closed stdout before serving"
            lines.append(line)
            if line.startswith("}"):
                break
        status = json.loads("".join(lines))
        assert status["status"] == "serving"
        base = status["url"].split("/?", 1)[0]
        strategies = _when_listening(f"{base}/api/strategies", process)
        listed = {s["name"]: s for s in strategies["strategies"]}
        assert listed["local"]["available"] and listed["mix"]["available"]
        assert _get(f"{base}/api/status")["namespace"]
        process.send_signal(signal.SIGTERM)
        code = process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    stderr = process.stderr.read() if process.stderr else ""
    assert code in (0, 128 + signal.SIGTERM), (code, stderr[-4000:])


# Each console script's real invocation (run-ingestion's is the workspace
# fixture's).
INVOCATIONS = {
    "run-ingestion": lambda workspace: None,
    "run-rag": _rag,
    "run-eval": _eval,
    "run-visualization": _visualization,
    "run-prompt-tuning": _prompt_tuning,
    "run-describe": _describe,
    "prepare-benchmark": _benchmark,
    "run-mcp-server": _mcp_server,
    "run-console": _console,
}


@pytest.mark.parametrize("name", sorted(INVOCATIONS))
def test_each_script_runs_in_a_process_of_its_own(workspace: Path, name: str) -> None:
    INVOCATIONS[name](workspace)
