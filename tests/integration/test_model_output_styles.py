# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every strategy over model output as other model families write it.

The fixtures answered as Claude does: bare XML, strict JSON, an English
"None". A reasoning model's ``<think>`` block began every answer, a Python
dict or a Llama trailing comma failed mix, hybrid and iterative and left
DRIFT with no intermediate answer, and a Korean "없음" section failed its
text unit on every run. Here the console demo's fake model server
(``console/demo/fake_models.py``) answers in each style
(``FakeModels.restyle``), the demo corpus is ingested into the embedded
stores (``backend: local``, no AWS) and every strategy answers.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.console_demo import load
from unified_kg_rag.application.config import get_config
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import SearchStrategy

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_THINK = (
    "<think>\nThe user wants output in the format they gave, e.g. "
    '{"key": "value"}. Draft: <entities> ... I will answer now. [1]\n</think>\n\n'
)


def _json_reply(reply: str) -> Any:
    try:
        return json.loads(reply)
    except ValueError:
        return None


def _pydict(reply: str) -> str:
    """JSON answers as Python's ``repr``: single quotes, True/None."""
    parsed = _json_reply(reply)
    return reply if parsed is None else repr(parsed)


def _trailing_comma(reply: str) -> str:
    """JSON answers with a comma before every closing bracket (Llama)."""
    parsed = _json_reply(reply)
    if parsed is None:
        return reply
    text = json.dumps(parsed, indent=1, ensure_ascii=False)
    return re.sub(r"(\S)\n(\s*)([}\]])", r"\1,\n\2\3", text)


def _prose(reply: str) -> str:
    """Structured answers inside chat prose and a code fence."""
    if not reply.lstrip().startswith(("{", "<")):
        return reply
    return f"Sure! Here is the output you requested:\n\n```\n{reply}\n```\n\nAnything else?"


def _ko_none(reply: str) -> str:
    """Empty sections saying "none" in Korean."""
    return reply.replace(
        "<relationships></relationships>", "<relationships>없음</relationships>"
    ).replace("<claims></claims>", "<claims>없음</claims>")


STYLES: dict[str, Callable[[str], str]] = {
    "plain": lambda reply: reply,
    "think": lambda reply: _THINK + reply,
    "pydict": _pydict,
    "trailing_comma": _trailing_comma,
    "prose": _prose,
    "ko_none": _ko_none,
}

QUESTION = "Who does Acme Robotics contract with?"


@pytest.fixture(scope="module", params=list(STYLES))
def styled(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[str, Path, dict[str, Any]]]:
    """A demo workspace ingested with the fake models answering in one
    style, and ``run-ingestion``'s JSON output."""
    style = request.param
    restyle = STYLES[style]
    fake = load("fake_models")
    demo = load("demo")

    def styled_reply(system: str, reply: str) -> str:
        # The output fixer's prompt has no system message: it answers plain.
        return restyle(reply) if system else reply

    server, _ = fake.serve(chunk_delay=0.0, restyle=styled_reply)
    root = tmp_path_factory.mktemp(f"style-{style}")
    try:
        source, _ = demo.write_corpus(root)
        url = f"http://127.0.0.1:{server.server_address[1]}"
        config = demo.demo_config("fake", model_base_url=url, region=None)
        # Claims too, so their "none" section is read in every style, and
        # every methodology, so every strategy runs.
        config["processing"]["claim_extraction"] = {"enabled": True}
        config["indexing"] = {
            "methodologies": ["graphrag", "lightrag", "hipporag", "raptor"]
        }
        (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        env = {**os.environ, "NO_PROXY": "127.0.0.1,localhost"}
        run = subprocess.run(
            [
                str(Path(sys.executable).with_name("run-ingestion")),
                "--config-path",
                str(root / "config.yaml"),
                "--source-directory",
                str(source),
                "--cache-directory",
                str(root / "cache"),
                "--pipeline-id",
                "styles",
                "--yes",
                "--output-format",
                "json",
                "--log-level",
                "WARNING",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert run.returncode == 0, run.stderr[-3000:]
        yield style, root, json.loads(run.stdout)
    finally:
        server.shutdown()


def test_ingestion_fails_no_unit_in_any_style(
    styled: tuple[str, Path, dict[str, Any]],
) -> None:
    style, _, output = styled
    # ko_none: a section saying 없음 failed its text unit, on every run.
    assert output["units_failed"] == 0, (style, output)


@pytest.mark.parametrize("strategy", list(SearchStrategy), ids=lambda s: s.value)
def test_every_strategy_answers_in_every_style(
    styled: tuple[str, Path, dict[str, Any]],
    strategy: SearchStrategy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    style, root, _ = styled
    monkeypatch.chdir(root)  # the demo config's store paths are relative
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    chain = GraphRAGChain(get_config(root / "config.yaml"))
    try:
        out = chain.invoke(RAGInput(query=QUESTION, search_strategy=strategy))
    finally:
        chain.close()
    case = (style, strategy.value)
    metadata = out.metadata or {}
    # think: the reasoning block began every answer.
    assert out.answer.strip() and not out.answer.lstrip().startswith("<"), case
    # pydict, trailing_comma, prose: mix, hybrid and iterative failed, DRIFT
    # answered from nothing, all without a degraded mark.
    assert metadata.get("degraded_error") is None, (case, metadata)
    assert out.sources, case
    if strategy is SearchStrategy.DRIFT:
        assert metadata["drift_answers"] > 0, case
    if strategy is SearchStrategy.ITERATIVE:
        assert metadata["iterative"]["stop_reason"] == "sufficient", case
    if strategy is SearchStrategy.STRUCTURED:
        assert metadata["structured"]["reason"] not in {
            "refused_plan",
            "unreadable_plan",
        }, (case, metadata)
    if strategy is SearchStrategy.AUTO:
        # think: the router read the first strategy its reasoning named.
        assert metadata["routing"]["decided_by"] == "router", case


def test_dynamic_community_selection_rates_in_every_style(
    styled: tuple[str, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (rv8 retrieval L9): the fake's rating prompt never
    # matched (the prompt wraps), so every rating was unreadable and the
    # dynamic selection was never exercised.
    style, root, _ = styled
    monkeypatch.chdir(root)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    config = get_config(root / "config.yaml")
    config.search.global_search.use_dynamic_selection = True
    chain = GraphRAGChain(config)
    try:
        out = chain.invoke(
            RAGInput(query=QUESTION, search_strategy=SearchStrategy.GLOBAL)
        )
    finally:
        chain.close()
    metadata = out.metadata or {}
    assert metadata.get("degraded") is None, (style, metadata.get("degraded"))
    assert metadata["global_dynamic_selection"] is True, style
    assert metadata["global_rated_report_count"] > 0, style
    assert metadata["global_unrated_report_count"] == 0, style
    assert metadata["global_selected_report_count"] > 0, style
