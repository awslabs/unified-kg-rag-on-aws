# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-eval --arm NAME=overlay.yaml`` end to end on the embedded stores.

Configuration arms (an A/A copy and citations off) answer one dataset next to
the base configuration, each with its own chain and evaluation manager,
interleaved with the base runs; each arm run is compared with the same
strategy's base run, the arms are Holm-adjusted in families of their own
apart from the strategies, the A/A arms not at all (their deltas are the
noise floor), the manifests record the overlays, and ``--from-results``
rebuilds the same comparison. The models are the Ollama adapters against a fake server, the
stores the SQLite ones: no container, no AWS.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.application.cli import run_evaluation
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Config,
    EvaluatorType,
    PipelineStageStatus,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

no_aws = e2e.no_aws

_LABELS = ["local", "mix", "local@aa", "mix@aa", "local@uncited", "mix@uncited"]


def _run_eval(
    argv: list[str], chain: Any, config: Config, factory: Any
) -> tuple[int, dict[str, Any]]:
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [*argv, "--output-format", "json"]
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = asyncio.run(
            run_evaluation.EvaluationRunner(
                args, chain, config, chain_factory=factory
            ).run()
        )
    out = buffer.getvalue()
    return code, json.loads(out[out.find("{") :])


def test_configuration_arms_are_answered_paired_and_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_aws
) -> None:
    monkeypatch.setattr(run_evaluation, "display_ascii_art", lambda **_: None)
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.evaluation.enabled_evaluators = [
        EvaluatorType.ANSWER_MATCH,
        EvaluatorType.RETRIEVAL,
    ]
    server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
    encoder = e2e.FakeCrossEncoder("synthetic-reranker")
    providers = e2e._providers(config, server, encoder)
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    dataset = tmp_path / "eval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "query_id": f"q{i}",
                    "question": question,
                    "answer": "Depot",
                    "reference_sources": ["logistics.txt"],
                }
                for i, question in enumerate(
                    [
                        "Who stores the gadgets for Vendor?",
                        "Which company stores widgets?",
                        "Where are Vendor's widgets kept?",
                    ]
                )
            ]
        )
    )
    aa = tmp_path / "aa.yaml"
    aa.write_text("")  # no change: an A/A arm
    uncited = tmp_path / "uncited.yaml"
    uncited.write_text(yaml.safe_dump({"search": {"citations": {"enabled": False}}}))
    built: list[Config] = []

    def factory(arm_config: Config) -> GraphRAGChain:
        built.append(arm_config)
        return GraphRAGChain(
            arm_config, providers=e2e._providers(arm_config, server, encoder)
        )

    chain = GraphRAGChain(config, providers=providers)
    out_dir = tmp_path / "out"
    answers_before = server.answers
    try:
        code, out = _run_eval(
            [
                "--data-path",
                str(dataset),
                "--search-strategy",
                "local,mix",
                "--arm",
                f"aa={aa}",
                "--arm",
                f"uncited={uncited}",
                "--output-directory",
                str(out_dir),
            ],
            chain,
            config,
            factory,
        )
    finally:
        chain.close()
    assert code == 0, out
    # One chain per arm, its overlay over the run's configuration.
    assert [c.search.citations.enabled for c in built] == [True, False]
    # Every run answered every query (plus one warm-up per manager and
    # strategy).
    assert server.answers - answers_before >= 3 * len(_LABELS)
    comparison = out["comparison"]
    assert list(comparison["strategies"]) == _LABELS
    # Each arm run against the same strategy's base run.
    assert comparison["baselines"] == {
        "mix": "local",
        "local@aa": "local",
        "mix@aa": "mix",
        "local@uncited": "local",
        "mix@uncited": "mix",
    }
    # The strategies against the first, the arms against their base runs:
    # separate Holm families; the A/A arms are the noise floor, unadjusted.
    families = comparison["multiplicity"]["families"]
    assert families["primary"]["tests"] == 1
    assert families["arm_primary"]["tests"] == 2
    assert comparison["aa_arms"] == ["local@aa", "mix@aa"]
    assert comparison["multiplicity"]["unadjusted"] == ["local@aa", "mix@aa"]
    aa_delta = comparison["deltas_vs_baseline"]["local@aa"][
        "answer_match/answer_contains"
    ]
    assert aa_delta["mean_delta"] == 0.0 and "family" not in aa_delta
    uncited_delta = comparison["deltas_vs_baseline"]["mix@uncited"][
        "answer_match/answer_contains"
    ]
    assert uncited_delta["family"] == "arm_primary"
    # The citations arm differs from the base in search.citations.enabled.
    assert any("whether their answers cite" in w for w in comparison["warnings"])
    # Each arm run's manifest names its overlay; its files carry its label.
    summaries = {
        p.name: json.loads(p.read_text())
        for p in out_dir.rglob("evaluation_summary_*.json")
    }
    arm_manifests = {
        name: s["run_manifest"]
        for name, s in summaries.items()
        if name.startswith("evaluation_summary_mix@uncited_")
    }
    (manifest,) = arm_manifests.values()
    assert manifest["arm"] == {
        "name": "uncited",
        "overlay": str(uncited),
        "overlay_sha256": hashlib.sha256(uncited.read_bytes()).hexdigest(),
        "suffix": None,
        "aa": False,
    }
    assert manifest["citations"]["enabled"] is False
    assert manifest["run_order"] == "interleaved" and manifest["repeat"] == 1
    assert manifest["compared_runs"][:2] == ["local", "mix"]

    # Rebuilt from the files: the same runs, pairs and deltas.
    code, rebuilt = _run_eval(
        ["--from-results", str(out_dir), "--output-directory", str(tmp_path / "re")],
        None,
        config,
        None,
    )
    assert code == 0
    again = rebuilt["comparison"]
    # In the order the run compared them (the manifests' compared_runs: the
    # interleaved runs share a start time and their stamps are random).
    assert list(again["strategies"]) == _LABELS
    assert again["baselines"] == comparison["baselines"]
    assert again["aa_arms"] == comparison["aa_arms"]
    assert again["multiplicity"]["families"] == families
    assert (
        again["deltas_vs_baseline"]["mix@uncited"]["answer_match/answer_contains"][
            "mean_delta"
        ]
        == comparison["deltas_vs_baseline"]["mix@uncited"][
            "answer_match/answer_contains"
        ]["mean_delta"]
    )


def test_an_invalid_overlay_is_a_usage_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_evaluation, "display_ascii_art", lambda **_: None)
    overlay = tmp_path / "bad.yaml"
    overlay.write_text(yaml.safe_dump({"search": {"no_such_key": 1}}))
    dataset = tmp_path / "eval.json"
    dataset.write_text(json.dumps([{"question": "Who?", "answer": "Depot"}]))

    class _Chain:
        async def ainvoke(self, inputs: Any, config: Any = None) -> Any:
            return {"answer": "Depot", "sources": [], "metadata": {}}

    args = run_evaluation.CommandLineInterface().parser.parse_args(
        ["--data-path", str(dataset), "--arm", f"bad={overlay}"]
    )
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    runner = run_evaluation.EvaluationRunner(args, _Chain(), config)
    with pytest.raises(run_evaluation.ConfigurationError, match="no_such_key"):
        asyncio.run(runner.run())
