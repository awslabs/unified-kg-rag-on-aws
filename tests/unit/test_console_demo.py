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
    items = json.loads(dataset.read_text(encoding="utf-8"))
    for item in items:
        for reference in item.get("reference_sources", []):
            assert reference in files
        if item.get("answer"):
            text = corpus.DOCUMENTS[item["reference_sources"][0]]
            assert item["answer"].split(",")[0] in text
    assert (
        sum(1 for i in items if (i.get("metadata") or {}).get("answerable") is False)
        == 1
    )
    assert json.loads((tmp_path / "sample-questions.json").read_text(encoding="utf-8"))


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
    assert fake.route("What are the overall themes?") == "global"
    assert fake.route("How are Acme Robotics and Globex Sensors related?") == "mix"
    verdict = json.loads(fake.judge("Answer 1:\nlong long answer\n\nAnswer 2:\nshort"))
    assert verdict["comprehensiveness"]["winner"] == "1"
    assert verdict["directness"]["winner"] == "2"
    assert fake.tokens("케스트럴 창고") == ["케스", "스트", "트럴", "창고"]
    vector = fake.embed("Kestrel Depot", 16)
    assert len(vector) == 16 and abs(sum(v * v for v in vector) - 1) < 1e-9
    assert fake.respond("unknown system prompt", "x") == ("", False)


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
