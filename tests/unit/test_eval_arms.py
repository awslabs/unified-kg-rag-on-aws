# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-eval --arm NAME=overlay.yaml``: parsing, the arm's configuration
and how the comparison pairs arm runs (the end-to-end run is in
tests/integration/test_eval_arms_e2e.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from unified_kg_rag.application.cli import run_evaluation
from unified_kg_rag.application.cli.arm_args import arm_config, arm_spec
from unified_kg_rag.domain.models import Config, ModelRole
from unified_kg_rag.evaluation.arms import arm_label, arm_pairs, label_strategy
from unified_kg_rag.shared import ConfigurationError

pytestmark = pytest.mark.unit


def test_an_arm_is_name_overlay_and_an_optional_suffix(tmp_path: Path) -> None:
    overlay = tmp_path / "o.yaml"
    overlay.write_text("{}")
    assert arm_spec(f"aa={overlay}") == {
        "name": "aa",
        "overlay": str(overlay),
        "suffix": None,
    }
    assert arm_spec(f"b={overlay},suffix=v2")["suffix"] == "v2"
    for bad in (
        "aa",
        f"a b={overlay}",
        f"closed_book={overlay}",
        f"x@y={overlay}",
        f"aa={overlay},top_k=3",
        f"aa={tmp_path / 'missing.yaml'}",
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            arm_spec(bad)


def test_the_overlay_merges_over_the_run_config_but_not_its_scoring(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "o.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "search": {"citations": {"enabled": False}},
                "evaluation": {"answer_span": "full"},
            }
        )
    )
    base = Config()
    base.search.query_extraction_cache_size = 7
    config, warnings = arm_config(base, arm_spec(f"a={overlay}"))
    assert config.search.citations.enabled is False
    assert config.search.query_extraction_cache_size == 7  # the run's own setting
    # Every arm is scored as the base runs are.
    assert config.evaluation.answer_span == base.evaluation.answer_span
    assert any("'evaluation' section is ignored" in w for w in warnings)
    overlay.write_text(yaml.safe_dump({"search": {"no_such_key": 1}}))
    with pytest.raises(ConfigurationError, match="no_such_key"):
        arm_config(base, arm_spec(f"a={overlay}"))


def test_arm_runs_pair_with_their_strategy_base_run() -> None:
    assert arm_label("local", "aa") == "local@aa" and arm_label("mix", None) == "mix"
    assert label_strategy("local@aa") == "local"
    assert arm_pairs(["local", "mix", "closed_book"]) == {}
    assert arm_pairs(["local", "mix", "local@aa", "mix@aa", "closed_book"]) == {
        "mix": "local",
        "local@aa": "local",
        "mix@aa": "mix",
        "closed_book": "local",
    }


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--from-results", "d", "--arm", "a={o}"], "--arm answers the questions"),
        (["--data-path", "x.json", "--arm", "a={o}", "--arm", "a={o}"], "must differ"),
    ],
)
def test_arm_usage_errors(argv: list[str], message: str, tmp_path, capsys) -> None:
    overlay = tmp_path / "o.yaml"
    overlay.write_text("{}")
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [a.format(o=overlay) for a in argv]
    )
    with pytest.raises(SystemExit) as exc:
        run_evaluation.EvaluationRunner(args, None, Config())
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


class _Chain:
    """A chain with an extraction cache of its own."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._extractions = object()


@pytest.mark.parametrize(
    ("overlay", "shared"),
    [
        ({}, True),  # A/A: the same extraction
        ({"search": {"citations": {"enabled": False}}}, True),
        # The extraction runs on the default tier: the fast one is not its.
        ({"aws": {"bedrock": {"fast_effort": "high"}}}, True),
        ({"aws": {"bedrock": {"default_effort": "low"}}}, False),
        ({"search": {"output_ceilings": {"query_extraction": 64}}}, False),
        ({"aws": {"bedrock": {"region_name": "eu-west-1"}}}, False),
        ({"models": {"roles": {"entity_extraction": "fast"}}}, False),
        ({"providers": {"llm": {"temperature": 0.7}}}, False),
        # A cache of 0 in the overlay is the arm's: nothing kept or shared.
        ({"search": {"query_extraction_cache_size": 0}}, False),
    ],
)
async def test_an_arm_shares_extractions_only_when_they_would_be_the_same(
    overlay: dict, shared: bool, tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "o.yaml"
    path.write_text(yaml.safe_dump(overlay))
    data = tmp_path / "x.json"
    data.write_text("[]")
    monkeypatch.setattr(
        run_evaluation, "EvaluationManager", lambda **kw: kw, raising=False
    )
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        ["--data-path", str(data), "--arm", f"a={path}"]
    )
    base = _Chain(Config())
    runner = run_evaluation.EvaluationRunner(args, base, Config(), chain_factory=_Chain)
    manager = await runner._arm_manager(args.arm[0], [], [])
    chain = manager["rag_chain"]
    assert (chain._extractions is base._extractions) is shared


def test_every_arm_is_judged_by_the_judge_chosen_over_all_answer_models(
    tmp_path: Path,
) -> None:
    # The base answers with the fast model, the arm with the default judge:
    # each picked its own judge (the default judge for the base, another
    # for the arm), so the arm's judged deltas measured the judge too.
    overlay = tmp_path / "opus.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "roles": {
                        "answer_generation": "anthropic.claude-opus-5-5",
                        "evaluation": "anthropic.claude-haiku-5-5",
                        "answer_span": "default",
                    }
                }
            }
        )
    )
    data = tmp_path / "x.json"
    data.write_text("[]")
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        ["--data-path", str(data), "--arm", f"opus={overlay}", "--evaluators", "ragas"]
    )
    base = Config()
    base.models.roles.answer_generation = "fast"
    runner = run_evaluation.EvaluationRunner(args, None, base)
    judge = runner.config.model_id(ModelRole.EVALUATION)
    # Neither answer model: Sonnet 5.5, the default tier's.
    assert judge == "anthropic.claude-sonnet-5-5"
    assert runner.config.models.roles.evaluation == judge  # pinned
    arm, warnings = arm_config(runner.config, args.arm[0])
    assert arm.model_id(ModelRole.ANSWER_GENERATION) == "anthropic.claude-opus-5-5"
    assert arm.model_id(ModelRole.EVALUATION) == judge
    assert arm.model_id(ModelRole.ANSWER_SPAN) == base.model_id(ModelRole.ANSWER_SPAN)
    assert sum("is ignored; every arm is scored" in w for w in warnings) == 2


def test_runs_judged_by_different_models_are_warned_about() -> None:
    from unified_kg_rag.evaluation.saved_results import judge_warnings

    def manifest(judge: str | None, embedding: str | None) -> dict:
        return {
            "models": {
                "evaluation_judge": judge,
                "judge_effort": "low" if judge else None,
                "evaluation_embedding": embedding,
            }
        }

    same = [("local", manifest("j1", "e1")), ("mix", manifest("j1", "e1"))]
    assert judge_warnings(same) == []
    # A run that judged nothing records no judge: not compared.
    assert judge_warnings([*same, ("naive", manifest(None, None))]) == []
    warnings = judge_warnings([*same, ("local@b", manifest("j2", "e2"))])
    assert len(warnings) == 2
    assert "LLM judge" in warnings[0] and "local@b" in warnings[0]
    assert "embedding" in warnings[1]
