# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The local quickstarts run as written: ``docs/getting-started/local.md``,
the README's five-minute quickstart, and their Korean editions.

Every ``bash`` block of the page is read and run in order in an empty
working directory: the shell commands that write the sample files run in
``bash``, and every ``uv run [--no-dev] <console script>`` runs that script's entry
point in-process with the block's arguments and environment variables. The
installs (``uv sync``, ``uv pip``, ``ollama pull``) and the checkout
(``git clone``, ``cd``) are skipped. The models are the local adapters
against a fake Ollama (``FakeOllama``, no network). The reranker is a fake
cross-encoder (no download) only when the page's ``uv sync`` installs the
``cross-encoder`` extra; otherwise loading it fails as without the extra, and
the queries must still succeed with reranking off.

A renamed flag, environment variable, script or output file the page names
fails here: each command must exit 0, and the outputs the page promises
(the stores in ``.unified-kg-rag/``, the comparison file in
``outputs/evaluation/run_<timestamp>_<id>/``) must be where it says.
"""

from __future__ import annotations

import json
import re
import shlex
import signal
import subprocess  # nosec B404 - runs the page's own file-writing commands
import sys
import tomllib
from itertools import dropwhile
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
from unified_kg_rag.shared.exceptions import ConfigurationError

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).parents[2]
# Each page and its Korean edition run the same commands: the console
# scripts each runs, in order.
GUIDE_FLOW = ["run-ingestion", "run-rag", "run-rag", "run-eval"]
README_FLOW = ["run-ingestion", "run-rag"]
PAGES = {
    ROOT / "docs" / "getting-started" / "local.md": GUIDE_FLOW,
    ROOT / "docs" / "getting-started" / "local.ko.md": GUIDE_FLOW,
    ROOT / "README.md": README_FLOW,
    ROOT / "README.ko.md": README_FLOW,
}
# The sample corpus's entities (the page's made-up companies and people).
NAMES = ("Acme Robotics", "Alice Chen", "Globex Sensors", "Bob Martin")
ANSWER = "Alice Chen is the chief engineer at Acme Robotics."
# The default local embedding model's dimension (bge-m3).
DIMENSION = 1024
_INSTALLS = (("uv", "sync"), ("uv", "pip"), ("ollama", "pull"), ("git", "clone"))
_UV_RUN_FLAGS = frozenset({"--no-dev"})


def _respond(system: str, human: str) -> str:
    if "knowledge graph extraction expert" in system:
        return _extraction(_source_text(human), NAMES)
    if "search strategy selector" in system:
        return '{"strategy": "local"}'
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
    """The CLIs' Ollama clients answer from a ``FakeOllama`` (no network)."""
    import httpx

    server = FakeOllama(_respond, ANSWER, DIMENSION)
    transport = httpx.MockTransport(server)
    for factory in (OllamaLanguageModelFactory, OllamaEmbeddingFactory):
        monkeypatch.setattr(
            factory, "__init__", _with_transport(factory.__init__, transport)
        )
    return server


def _installs_cross_encoder(commands: list[str]) -> bool:
    return any(
        command.startswith("uv sync") and "cross-encoder" in command
        for command in commands
    )


def _reranker(monkeypatch: pytest.MonkeyPatch, *, installed: bool) -> list[str]:
    """The cross-encoder a page's install gives: a fake one, or a load that
    fails as it does without the extra. The model ids it was asked to load."""
    loaded: list[str] = []

    def load(model_id: str, **_: Any) -> FakeCrossEncoder:
        loaded.append(model_id)
        if not installed:
            raise ConfigurationError(
                "sentence_transformers is not installed: uv sync --extra cross-encoder"
            )
        return FakeCrossEncoder("x", relevant=None)

    init = rerank_module.CrossEncoderRerankFactory.__init__

    def rerank(self: Any, config: Any, *, loader: Any = None) -> None:
        init(self, config, loader=loader or load)

    monkeypatch.setattr(rerank_module.CrossEncoderRerankFactory, "__init__", rerank)
    return loaded


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "page", PAGES, ids=lambda page: page.relative_to(ROOT).as_posix()
)
def test_the_local_quickstart_runs_as_written(
    page: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_models: FakeOllama,
) -> None:
    text = page.read_text(encoding="utf-8")
    commands = _commands(text)
    scripts = _scripts()
    cross_encoder = _installs_cross_encoder(commands)
    loaded = _reranker(monkeypatch, installed=cross_encoder)
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
        if tuple(words[:2]) in _INSTALLS or words[:1] == ["cd"]:
            continue
        if words[:2] != ["uv", "run"]:
            # Writes the sample files: plain shell, in the working directory.
            subprocess.run(  # nosec B603 B607 - the page's own commands
                ["bash", "-euc", command], cwd=tmp_path, check=True
            )
            continue
        # uv's own flags (`uv run --no-dev <script>`) take no value here.
        words = words[:2] + list(dropwhile(lambda w: w in _UV_RUN_FLAGS, words[2:]))
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

    assert ran == PAGES[page]
    assert fake_models.chat_models and fake_models.embed_models
    # The queries asked for the preset's cross-encoder, which loaded only
    # where the page installs it (else they ran with reranking off).
    assert loaded
    stores = tmp_path / ".unified-kg-rag"
    assert {"graph.sqlite3", "vector.sqlite3"} <= {p.name for p in stores.iterdir()}
    if not cross_encoder:
        # The page says its queries then rerank nothing.
        assert re.search(
            r"reranking (?:is )?off|fused order|재순위화 없이|융합 순서", text
        )
    if PAGES[page] is README_FLOW:
        return
    # The guide's flow: ingest, two queries, one comparison; the stores
    # where it says they are.
    assert "`.unified-kg-rag/`" in text
    # Each strategy's results, reports and summary, and the comparison file
    # the page names, pairing mix against local.
    # One run-eval invocation writes into its own run subdirectory.
    assert "`outputs/evaluation/run_<timestamp>_<id>/`" in text
    (outputs,) = (tmp_path / "outputs" / "evaluation").glob("run_*")
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
