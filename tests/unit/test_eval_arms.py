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
from unified_kg_rag.domain.models import Config
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
