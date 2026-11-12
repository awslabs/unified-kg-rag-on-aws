# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console demo: its configs for the three model choices load as a run
loads them (no AWS call: nothing is built), its corpus and dataset agree,
and the fake model server answers each prompt kind deterministically."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.console_demo import load
from unified_kg_rag.domain.models import Config, ModelRole

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("models", ["fake", "ollama", "bedrock"])
def test_each_demo_config_loads_with_local_stores(models: str) -> None:
    demo = load("demo")
    data = demo.demo_config(
        models, model_base_url="http://127.0.0.1:1", region="eu-west-1"
    )
    config = Config.model_validate(data)
    assert config.providers.graph.kind == "sqlite"
    assert config.providers.vector.kind == "sqlite"
    assert config.providers.doc_status.enabled
    expected = {"fake": "ollama", "ollama": "ollama", "bedrock": "bedrock"}[models]
    assert config.providers.llm.kind == expected
    if models == "bedrock":
        assert config.aws.region_name == "eu-west-1"
        assert "anthropic" in config.model_id(ModelRole.ANSWER_GENERATION)
    if models == "fake":
        assert set(config.evaluation.prices) == {"demo-model", "demo-model-fast"}
        assert config.providers.rerank.kind == "none"


def test_the_demo_writes_a_corpus_its_dataset_answers(tmp_path: Path) -> None:
    demo = load("demo")
    corpus = load("corpus")
    source, dataset = demo.write_corpus(tmp_path)
    files = {p.name for p in source.iterdir()}
    assert files == set(corpus.documents())
    # Every format the core install parses without an extra, PDFs included.
    assert {Path(name).suffix for name in files} == {
        ".pdf",
        ".md",
        ".html",
        ".csv",
        ".json",
        ".txt",
    }
    items = json.loads(dataset.read_text(encoding="utf-8"))
    for item in items:
        for reference in item.get("reference_sources", []):
            assert reference in files
        if item.get("answer"):
            name = item["reference_sources"][0]
            text = (
                "\n".join(corpus.page_texts(name))
                if name in corpus.PDF_DOCUMENTS
                else corpus.TEXT_DOCUMENTS[name]
            )
            assert item["answer"].split(",")[0] in text
    assert (
        sum(1 for i in items if (i.get("metadata") or {}).get("answerable") is False)
        == 1
    )
    assert json.loads((tmp_path / "sample-questions.json").read_text(encoding="utf-8"))


def test_a_rewrite_removes_the_retired_corpus_files(tmp_path: Path) -> None:
    demo = load("demo")
    corpus = load("corpus")
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / corpus.RETIRED_FILES[0]).write_text("old copy")
    source, _ = demo.write_corpus(tmp_path)
    assert {p.name for p in source.iterdir()} == set(corpus.documents())


def test_the_corpus_pdfs_parse_page_by_page_as_laid_out(tmp_path: Path) -> None:
    from unified_kg_rag.adapters.ingestion.file_loaders import PdfFileLoader

    corpus = load("corpus")
    for name in corpus.PDF_DOCUMENTS:
        path = tmp_path / name
        path.write_bytes(corpus.documents()[name])
        pages = PdfFileLoader(path).load()
        assert [p.page_content for p in pages] == corpus.page_texts(name)
        assert [p.metadata["page"] for p in pages] == list(range(len(pages)))
        assert len(pages) >= 2 and path.stat().st_size < 16_000


def test_the_fake_models_answer_every_prompt_kind() -> None:
    fake = load("fake_models")
    extraction, streamed = fake.respond(
        "You are a knowledge graph extraction expert",
        "<input_text>\nUmbra Freight delivers to Kestrel Depot.\n</input_text>",
    )
    assert not streamed
    assert "<name>Umbra Freight</name>" in extraction and "DELIVERS_TO" in extraction
    answer, streamed = fake.respond(
        "You are an expert AI assistant that answers questions from knowledge graph context",
        'Query: "Who carries the units?"\n<context>Umbra Freight carries each lidar unit. '
        "Orion Labs inspects.</context>",
    )
    assert streamed and answer.startswith("Umbra Freight carries")
    # RAPTOR's tree summary is the passages' text, so the tree is built
    # (an empty reply left every cluster unsummarized, retried each run).
    summary, streamed = fake.respond(
        "You write the summaries of a summary tree that a retrieval system searches",
        "Summarize the following passages in English, in 200 words at most.\n\n"
        "<passages>\nUmbra Freight carries each lidar unit.\n\n"
        "케스트럴 창고가 부품을 보관한다.\n</passages>\n",
    )
    assert not streamed and "Umbra Freight carries each lidar unit." in summary
    assert "케스트럴 창고" in summary
    assert fake.route("What are the overall themes?") == "global"
    assert fake.route("How are Acme Robotics and Globex Sensors related?") == "mix"
    # The router answers the name alone, as its prompt asks; the rewrite,
    # the translation and the rating answer JSON.
    routed, _ = fake.respond(
        "You are an expert search strategy selector",
        'Query: "What are the overall themes?"',
    )
    assert routed == "global"
    rating, _ = fake.respond(
        "You are a helpful assistant responsible for deciding whether the provided "
        "information is useful",
        "x",
    )
    assert json.loads(rating)["rating"] == 4
    verdict = json.loads(fake.judge("Answer 1:\nlong long answer\n\nAnswer 2:\nshort"))
    assert verdict["comprehensiveness"]["winner"] == "1"
    assert verdict["directness"]["winner"] == "2"
    assert fake.tokens("케스트럴 창고") == ["케스", "스트", "트럴", "창고"]
    vector = fake.embed("Kestrel Depot", 16)
    assert len(vector) == 16 and abs(sum(v * v for v in vector) - 1) < 1e-9
    assert fake.respond("unknown system prompt", "x") == ("", False)


def test_the_fake_reports_and_key_points_cite_their_evidence() -> None:
    # The demo shows citations as a model writes them: each report finding
    # cites its numbered relationship, each map point the report it is from.
    fake = load("fake_models")
    report = fake.community_report(
        "<entity_data>\n- [1] Acme Robotics: robots\n</entity_data>\n"
        "<relationship_data>\n- [1] Acme Robotics -> lidar unit (BUYS): Acme "
        "Robotics buys every lidar unit.\n- [2] 애크미 로보틱스 -> 글로벡스 센서스 "
        "(CONTRACTS_WITH): 애크미 로보틱스는 계약을 맺었다.\n</relationship_data>"
    )
    assert "buys every lidar unit. [Data: Relationships (1)]" in report
    assert "계약을 맺었다. [Data: Relationships (2)]" in report
    human = (
        'User Question: "Who buys the lidar unit?"\n<community_reports>\n'
        "--- Report (id: 1) ---\nGlobex Sensors makes sensors.\n\n"
        "--- Report (id: 2) ---\nAcme Robotics buys every lidar unit.\n"
        "</community_reports>"
    )
    reply, _ = fake.respond("Extract the KEY POINTS from the reports", human)
    assert json.loads(reply)["points"][0] == {
        "description": "Acme Robotics buys every lidar unit.",
        "score": 80,
        "report_ids": ["2"],
    }


def test_a_restart_keeps_the_config_settings_saved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    demo = load("demo")
    args = ["--models", "fake", "--workspace", str(tmp_path), "--no-ingest"]
    assert demo.main([*args, "--prepare-only"]) == 0
    path = tmp_path / "config.yaml"
    written = yaml.safe_load(path.read_text())
    port = demo.kept_model_port(path)
    assert port and written["providers"]["llm"]["base_url"].endswith(f":{port}")
    # What Settings saves (here: a price) survives the next start.
    written["evaluation"]["prices"]["demo-model"]["input_per_million_tokens"] = 9.5
    path.write_text(yaml.safe_dump(written))
    assert demo.main([*args, "--prepare-only"]) == 0
    kept = yaml.safe_load(path.read_text())
    assert kept["evaluation"]["prices"]["demo-model"]["input_per_million_tokens"] == 9.5
    # The fake models answer where the kept config looks.
    assert f"Fake models on http://127.0.0.1:{port}" in capsys.readouterr().out
    assert demo.main([*args, "--prepare-only", "--reset-config"]) == 0
    reset = yaml.safe_load(path.read_text())
    assert (
        reset["evaluation"]["prices"]["demo-model"]["input_per_million_tokens"] == 3.0
    )
    # A kept config that does not load stops the demo, naming the way out.
    path.write_text("search:\n  no_such_key: 1\n")
    with pytest.raises(SystemExit, match="--reset-config"):
        demo.main([*args, "--prepare-only"])


def test_without_the_console_extra_the_demo_stops_before_ingesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It ingested the corpus and printed the console's address, then
    # run-console failed on the missing extra.
    demo = load("demo")
    monkeypatch.setattr(demo, "missing_extra", lambda: ["fastapi"])
    monkeypatch.setattr(demo, "ingest", lambda *a: pytest.fail("ingested"))
    workspace = tmp_path / "ws"
    with pytest.raises(SystemExit, match="--extra console"):
        demo.main(["--models", "fake", "--workspace", str(workspace)])
    assert not workspace.exists()
    # --prepare-only starts no console: it needs no extra.
    assert (
        demo.main(["--workspace", str(workspace), "--prepare-only", "--no-ingest"]) == 0
    )


def test_a_token_run_console_refuses_stops_the_demo_before_ingesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It ingested and printed an address with a token run-console then
    # refused (under 16 characters), so the address never opened.
    demo = load("demo")
    monkeypatch.setattr(demo, "missing_extra", lambda: [])
    monkeypatch.setattr(demo, "missing_web_build", lambda: False)
    monkeypatch.setattr(demo, "ingest", lambda *a: pytest.fail("ingested"))
    monkeypatch.setenv("UKR_CONSOLE_TOKEN", "x" * 15)
    workspace = tmp_path / "ws"
    with pytest.raises(SystemExit, match="at least 16 characters"):
        demo.main(["--models", "fake", "--workspace", str(workspace)])
    assert not workspace.exists()
    # --prepare-only starts no console: the token is not used.
    assert (
        demo.main(["--workspace", str(workspace), "--prepare-only", "--no-ingest"]) == 0
    )


@pytest.mark.parametrize("models", ["fake", "ollama"])
def test_the_local_model_demos_name_the_local_extra_before_any_work(
    models: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = load("demo")
    from unified_kg_rag.application.cli import preflight

    monkeypatch.setattr(preflight, "module_installed", lambda module: False)
    with pytest.raises(SystemExit) as raised:
        demo.main(["--models", models, "--workspace", str(tmp_path), "--prepare-only"])
    assert "--extra local" in str(raised.value)
    assert not (tmp_path / "corpus").exists()


def test_the_bedrock_demo_needs_no_local_extra() -> None:
    assert load("demo").missing_local_extra("bedrock") is None


def test_without_the_built_web_client_the_demo_stops_before_ingesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It ingested and printed an address that served no page.
    demo = load("demo")
    monkeypatch.setattr(demo, "missing_extra", lambda: [])
    monkeypatch.setattr(demo, "web_directory", lambda: tmp_path / "no-dist")
    monkeypatch.setattr(demo, "ingest", lambda *a: pytest.fail("ingested"))
    workspace = tmp_path / "ws"
    with pytest.raises(SystemExit, match="npm run build"):
        demo.main(["--models", "fake", "--workspace", str(workspace)])
    assert not workspace.exists()


def test_a_sigterm_stops_the_console_the_demo_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import signal
    import threading

    demo = load("demo")
    signals: list[int] = []

    class _Console:
        def __init__(self, *_: object, **__: object) -> None:
            self.stopped = threading.Event()

        def send_signal(self, signum: int) -> None:
            signals.append(signum)
            self.stopped.set()

        def wait(self) -> int:
            # The demo's own SIGTERM, while it waits on the console.
            os.kill(os.getpid(), signal.SIGTERM)
            assert self.stopped.wait(5)
            return 143

    monkeypatch.setattr(demo, "missing_extra", lambda: [])
    monkeypatch.setattr(demo, "missing_web_build", lambda: False)
    monkeypatch.setattr(demo.subprocess, "Popen", _Console)
    before = signal.getsignal(signal.SIGTERM)
    code = demo.main(["--models", "fake", "--workspace", str(tmp_path), "--no-ingest"])
    assert code == 143 and signals == [signal.SIGTERM]
    assert signal.getsignal(signal.SIGTERM) is before  # restored
