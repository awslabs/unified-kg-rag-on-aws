# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``docs/getting-started/local.md`` (and its Korean edition) runs as written.

Every ``bash`` block of the page is read and run in order in an empty
working directory: the shell commands that write the sample files run in
``bash``, and every ``uv run <console script>`` runs that script's entry
point in-process with the block's arguments and environment variables. The
installs (``uv sync``, ``uv pip``, ``ollama pull``) are skipped. The models
are the local adapters against a fake Ollama (``FakeOllama``, no network),
and the reranker a fake cross-encoder (no download).

A renamed flag, environment variable, script or output file the page names
fails here: each command must exit 0, and the outputs the page promises
(the stores in ``.unified-kg-rag/``, the comparison file in
``outputs/evaluation/``) must be where it says.
"""

from __future__ import annotations

import json
import re
import shlex
import signal
import subprocess  # nosec B404 - runs the page's own file-writing commands
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_ingestion_stages import _extraction, _source_text
from unified_kg_rag.adapters.local import rerank as rerank_module
from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.application import config as config_module

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).parents[2]
# The page and its Korean edition run the same commands.
PAGES = [
    ROOT / "docs" / "getting-started" / name for name in ("local.md", "local.ko.md")
]
# The sample corpus's entities (the page's made-up companies and people).
NAMES = ("Acme Robotics", "Alice Chen", "Globex Sensors", "Bob Martin")
ANSWER = "Alice Chen is the chief engineer at Acme Robotics."
# The default local embedding model's dimension (bge-m3).
DIMENSION = 1024
_INSTALLS = (("uv", "sync"), ("uv", "pip"), ("ollama", "pull"))


def _respond(system: str, human: str) -> str:
    if "knowledge graph extraction expert" in system:
        return _extraction(_source_text(human), NAMES)
    if "search strategy selector" in system:
        return "local"
    if "specialized entity extraction expert" in system:
        return "Acme Robotics, Alice Chen"
    if "expert keyword extractor" in system:
        return json.dumps(
            {
                "high_level_keywords": ["supply agreement"],
                "low_level_keywords": ["Acme Robotics", "Globex Sensors"],
            }
        )
    # Gleaning, claims and reports as in the stage tests; the answers (an
    # AssertionError there) get the fake's ANSWER.
    return stages._respond(system, human)


def _commands(text: str) -> list[str]:
    """The page's ``bash`` blocks, as commands: a heredoc stays with the
    command that opens it, a backslash-continued line with its successor."""
    commands: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        lines = block.replace("\\\n", " ").splitlines()
        index = 0
        while index < len(lines):
            line = lines[index]
            index += 1
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            heredoc = re.search(r"<<'?(\w+)'?", line)
            if heredoc:
                end = lines.index(heredoc.group(1), index)
                line = "\n".join([line, *lines[index : end + 1]])
                index = end + 1
            commands.append(line)
    return commands


def _scripts() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["scripts"]


def _run_script(entry: str, argv: list[str]) -> int:
    """Run the console script ``entry`` (``module:function``) as the shell
    would, in-process: its exit code. The entry point installs signal
    handlers; the test's are restored after."""
    module, _, function = entry.partition(":")
    __import__(module)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    saved = sys.argv
    sys.argv = argv
    try:
        result = getattr(sys.modules[module], function)()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = saved
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return int(result or 0)


def _with_transport(init: Any, default: Any) -> Any:
    def patched(self: Any, config: Any, *, transport: Any = None) -> None:
        init(self, config, transport=transport or default)

    return patched


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    """The CLIs' Ollama clients answer from a ``FakeOllama``, and their
    cross-encoder is a fake (no network, no download)."""
    import httpx

    server = FakeOllama(_respond, ANSWER, DIMENSION)
    transport = httpx.MockTransport(server)
    for factory in (OllamaLanguageModelFactory, OllamaEmbeddingFactory):
        monkeypatch.setattr(
            factory, "__init__", _with_transport(factory.__init__, transport)
        )
    init = rerank_module.CrossEncoderRerankFactory.__init__

    def rerank(self: Any, config: Any, *, loader: Any = None) -> None:
        init(
            self,
            config,
            loader=loader or (lambda *_, **__: FakeCrossEncoder("x", relevant=None)),
        )

    monkeypatch.setattr(rerank_module.CrossEncoderRerankFactory, "__init__", rerank)
    return server


@pytest.mark.timeout(300)
@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_the_local_quickstart_runs_as_written(
    page: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_models: FakeOllama,
) -> None:
    text = page.read_text(encoding="utf-8")
    commands = _commands(text)
    scripts = _scripts()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    # A prompt the fake cannot answer fails at once instead of backing off.
    monkeypatch.setenv("UKR__RETRY__MAX_ATTEMPTS", "1")
    ran: list[str] = []
    for command in commands:
        words = shlex.split(command.split("\n", 1)[0])
        env: dict[str, str] = {}
        while words and re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", words[0]):
            name, _, value = words.pop(0).partition("=")
            env[name] = value
        if tuple(words[:2]) in _INSTALLS:
            continue
        if words[:2] != ["uv", "run"]:
            # Writes the sample files: plain shell, in the working directory.
            subprocess.run(  # nosec B603 B607 - the page's own commands
                ["bash", "-euc", command], cwd=tmp_path, check=True
            )
            continue
        script, argv = words[2], words[3:]
        assert script in scripts, f"the page runs '{script}', not a console script"
        with monkeypatch.context() as scoped:
            for name, value in env.items():
                scoped.setenv(name, value)
            # A new process: the config without a path is loaded again, with
            # this command's environment (get_config loads it once).
            scoped.setattr(
                config_module, "_config_loader", config_module.ConfigLoader()
            )
            code = _run_script(scripts[script], [script, *argv])
        assert code == 0, f"{command!r} exited {code}"
        ran.append(script)

    # The page's flow: ingest, two queries, one comparison.
    assert ran == ["run-ingestion", "run-rag", "run-rag", "run-eval"]
    assert fake_models.chat_models and fake_models.embed_models
    # The stores where the page says they are.
    assert "`.unified-kg-rag/`" in text
    stores = tmp_path / ".unified-kg-rag"
    assert {"graph.sqlite3", "vector.sqlite3"} <= {p.name for p in stores.iterdir()}
    # Each strategy's results, reports and summary, and the comparison file
    # the page names, pairing mix against local.
    assert "`outputs/evaluation/`" in text
    outputs = tmp_path / "outputs" / "evaluation"
    for kind in ("results", "reports", "summary"):
        for strategy in ("local", "mix"):
            assert list(outputs.glob(f"evaluation_{kind}_{strategy}_*.json")), (
                kind,
                strategy,
            )
    assert "evaluation_comparison_<timestamp>.json" in text
    (comparison,) = outputs.glob("evaluation_comparison_*.json")
    paired = json.loads(comparison.read_text(encoding="utf-8"))
    assert paired["baseline"] == "local"
    assert set(paired["strategies"]) == {"local", "mix"}
