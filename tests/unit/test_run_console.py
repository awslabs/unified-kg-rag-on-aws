# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-console``: loopback by default, a remote address only when asked,
the console extra checked first, the address printed once it is bound."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.console_demo import requires_console
from unified_kg_rag.application.cli import run_console

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("127.0.0.2", True),
        ("0.0.0.0", False),
        ("192.168.1.5", False),
        ("example.org", False),
    ],  # nosec B104
)
def test_loopback_addresses(host: str, loopback: bool) -> None:
    assert run_console._is_loopback(host) is loopback


def test_the_default_host_is_loopback() -> None:
    args = run_console._build_parser().parse_args([])
    assert args.host == "127.0.0.1" and not args.allow_remote and args.port == 8501


def test_another_address_needs_allow_remote(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run_console.main(["--host", "0.0.0.0"])  # nosec B104
    assert exc.value.code == 2
    assert "--allow-remote" in capsys.readouterr().err


def test_a_missing_extra_is_named(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_console, "missing_extra", lambda: ["fastapi"])
    with pytest.raises(SystemExit) as exc:
        run_console.main(["--output-format", "json"])
    assert exc.value.code == 2
    status = json.loads(capsys.readouterr().out)
    assert status["exit_code"] == 2 and "uv sync --extra console" in status["error"]


def test_a_missing_workspace_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        run_console.main(["--source-directory", str(tmp_path / "nope")])
    assert exc.value.code == 2


def test_the_web_directory_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(run_console.WEB_DIRECTORY_ENV, str(tmp_path))
    assert run_console.web_directory() == tmp_path.resolve()
    monkeypatch.delenv(run_console.WEB_DIRECTORY_ENV)
    assert run_console.web_directory().parts[-3:] == ("console", "web", "dist")


@requires_console
def test_it_binds_then_prints_where_it_serves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    served: dict[str, Any] = {}

    def run(self: Any, sockets: list[socket.socket]) -> None:
        served["address"] = sockets[0].getsockname()
        served["app"] = self.config.app
        sockets[0].close()

    monkeypatch.setattr(uvicorn.Server, "run", run)
    (tmp_path / "sample-questions.json").write_text('["Who ships?"]')
    code = run_console.main(
        [
            "--source-directory",
            str(tmp_path),
            "--cache-directory",
            str(tmp_path / "cache"),
            "--port",
            "0",
            "--output-format",
            "json",
        ]
    )
    assert code == 0
    status = json.loads(capsys.readouterr().out)
    host, port = served["address"]
    assert host == "127.0.0.1" and status["url"] == f"http://127.0.0.1:{port}/"
    assert status["status"] == "serving"
    options = served["app"].state.console.options
    assert options.loopback and options.sample_questions == ["Who ships?"]
    assert (tmp_path / "cache" / "console" / "console.sqlite3").exists()


@requires_console
def test_a_port_in_use_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    try:
        code = run_console.main(
            [
                "--source-directory",
                str(tmp_path),
                "--cache-directory",
                str(tmp_path / "c"),
                "--port",
                str(busy.getsockname()[1]),
            ]
        )
    finally:
        busy.close()
    assert code == 1
    assert "Cannot listen" in capsys.readouterr().out


@requires_console
def test_a_config_that_does_not_load_still_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets: sockets[0].close())
    config = tmp_path / "config.yaml"
    config.write_text("search:\n  no_such_key: 1\n")
    code = run_console.main(
        [
            "--config-path",
            str(config),
            "--source-directory",
            str(tmp_path),
            "--cache-directory",
            str(tmp_path / "c"),
            "--port",
            "0",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "does not load" in out and "Console" in out


def test_the_console_server_loads_no_store_or_model_backend() -> None:
    """The server and its services import the application and evaluation
    layers and the public extension registries only: no vendor binding, no
    store adapter (they come through the chain and the CLIs)."""
    import ast

    package = Path(run_console.__file__).parent / "console"
    allowed_adapters = {
        "unified_kg_rag.adapters.retrieval.strategy_registry",
        "unified_kg_rag.adapters.ingestion.parser",
    }
    for path in [*package.glob("*.py"), Path(run_console.__file__)]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                if name.startswith("unified_kg_rag.adapters"):
                    assert name in allowed_adapters, f"{path.name} imports {name}"


def test_the_cli_loads_no_web_package_before_it_checks_the_extra() -> None:
    import subprocess

    code = (
        "import sys, unified_kg_rag.application.cli.run_console; "
        "sys.exit('fastapi' in sys.modules or 'uvicorn' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
