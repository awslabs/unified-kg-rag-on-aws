# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-mcp-server`` and ``build_mcp_server``: the transport's address
rules, the strategy narrowing, the start-up checks and the missing ``mcp``
extra (these run on a core install too)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.fixtures.extras import requires_mcp
from unified_kg_rag.application.cli import mcp_server, run_mcp_server
from unified_kg_rag.application.cli.mcp_server import (
    MCP_EXTRA_HINT,
    build_mcp_server,
    mcp_extra_error,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


@pytest.fixture
def without_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """An install without the mcp extra: importing the SDK fails."""
    for name in list(sys.modules):
        if name == "mcp" or name.startswith("mcp."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mcp", None)


def test_building_a_server_without_the_extra_names_it(without_mcp: None) -> None:
    assert mcp_extra_error() == MCP_EXTRA_HINT
    with pytest.raises(ConfigurationError, match=r"mcp extra.*--extra mcp"):
        build_mcp_server(SimpleNamespace(config=Config()))  # type: ignore[arg-type]


def _config_file(tmp_path: Path, **settings: object) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"logging": {"log_to_file": False}, **settings}),
        encoding="utf-8",
    )
    return str(path)


def test_run_mcp_server_without_the_extra_exits_2(
    without_mcp: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    built: list[object] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_mcp_server, "GraphRAGChain", lambda **kw: built.append(kw))
        with pytest.raises(SystemExit) as exited:
            run_mcp_server.main(["--config-path", _config_file(tmp_path)])
    assert exited.value.code == 2
    assert "--extra mcp" in capsys.readouterr().err
    # Nothing was built, and stdout stayed empty (it is the protocol's).
    assert built == []
    assert capsys.readouterr().out == ""


@requires_mcp
def test_run_mcp_server_without_store_endpoints_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("UKR__AWS__NEPTUNE__ENDPOINT", "UKR__AWS__OPENSEARCH__ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as exited:
        run_mcp_server.main(["--config-path", _config_file(tmp_path)])
    assert exited.value.code == 2
    err = capsys.readouterr()
    assert "Missing endpoint configuration for the MCP server" in err.err
    assert err.out == ""


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.3.2.1", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("LOCALHOST", True),
        ("0.0.0.0", False),  # nosec B104 - the address under test
        ("::", False),
        ("10.0.0.5", False),
        ("mcp.example.com", False),
    ],
)
def test_loopback_addresses(host: str, loopback: bool) -> None:
    assert run_mcp_server.is_loopback(host) is loopback


def test_a_remote_bind_needs_allow_remote(capsys: pytest.CaptureFixture[str]) -> None:
    http = ["--transport", "streamable-http"]
    with pytest.raises(SystemExit) as exited:
        run_mcp_server.parse_args([*http, "--host", "0.0.0.0"])  # nosec B104
    assert exited.value.code == 2
    assert "authenticates no caller" in capsys.readouterr().err
    args = run_mcp_server.parse_args(
        [*http, "--host", "0.0.0.0", "--allow-remote"]  # nosec B104
    )
    expected = ("0.0.0.0", 8000, True)  # nosec B104 - the address under test
    assert (args.host, args.port, args.allow_remote) == expected
    assert run_mcp_server.parse_args(http).host == "127.0.0.1"


@pytest.mark.parametrize(
    "flags", [["--host", "127.0.0.2"], ["--port", "9000"], ["--allow-remote"]]
)
def test_http_flags_are_refused_with_stdio(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        run_mcp_server.parse_args(flags)
    assert exited.value.code == 2


def test_the_port_and_suffix_are_validated() -> None:
    for flags in (["--port", "0"], ["--port", "70000"], ["--suffix", "a,b"]):
        with pytest.raises(SystemExit):
            run_mcp_server.parse_args(flags)
    assert run_mcp_server.parse_args(["--suffix", "tenant_a"]).suffix == "tenant_a"


def test_mcp_settings_are_validated() -> None:
    with pytest.raises(ValueError):
        Config.model_validate({"mcp": {"tools": ["kg_delete"]}})
    with pytest.raises(ValueError, match="more than once"):
        Config.model_validate({"mcp": {"tools": ["kg_search", "kg_search"]}})
    with pytest.raises(ValueError):
        Config.model_validate({"mcp": {"tools": []}})
    with pytest.raises(ValueError):
        Config.model_validate({"mcp": {"max_hops": 4}})


@requires_mcp
def test_the_server_lists_read_only_tools_only() -> None:
    import asyncio

    from mcp import Client

    server = build_mcp_server(SimpleNamespace(config=Config()))  # type: ignore[arg-type]

    async def listed() -> list[str]:
        async with Client(server) as client:
            tools = (await client.list_tools()).tools
            assert all(t.annotations and t.annotations.read_only_hint for t in tools)
            return [t.name for t in tools]

    names = asyncio.run(listed())
    assert names == list(Config().mcp.tools)
    assert not any(
        word in name
        for name in names
        for word in ("ingest", "delete", "write", "update", "reset")
    )
    assert mcp_server.SERVER_NAME == "unified-kg-rag"


@requires_mcp
def test_kg_answer_takes_a_bounded_history_over_the_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from mcp import Client

    from tests.unit.test_mcp_tools import FakeChain
    from unified_kg_rag.application.retrieval import graph_lookup

    async def names(chain: object, suffix: object, units: list[str]) -> dict:
        return {}

    monkeypatch.setattr(graph_lookup, "text_unit_file_names", names)
    chain = FakeChain()
    server = build_mcp_server(chain)  # type: ignore[arg-type]
    turns = [
        {"role": "user", "content": "창고는 어디에 있나요?"},
        {"role": "assistant", "content": "倉庫は港の近くにあります。"},
    ]

    async def run() -> tuple[dict, dict, bool]:
        async with Client(server, mode="legacy") as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            answered = await client.call_tool(
                "kg_answer", {"query": "And who runs it?", "history": turns}
            )
            assert not answered.is_error, answered.content
            refused = await client.call_tool(
                "kg_answer",
                {"query": "q", "history": [{"role": "user", "content": "q"}] * 21},
            )
            return (
                tools["kg_answer"].input_schema["properties"],
                tools["kg_search"].input_schema["properties"],
                refused.is_error,
            )

    answer_schema, search_schema, refused = asyncio.run(run())
    assert "history" in answer_schema and "history" not in search_schema
    assert refused
    (rag_input,) = chain.inputs
    assert rag_input.history is not None
    assert [t.model_dump() for t in rag_input.history] == turns
