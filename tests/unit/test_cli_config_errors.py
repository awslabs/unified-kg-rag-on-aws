# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bad configuration file is a usage error in every CLI.

A value of the wrong type, invalid YAML or a missing ``--config-path`` file
prints one error line and exits 2, as argparse does for a bad option, instead
of "An unexpected error occurred" with a traceback.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from unified_kg_rag.application.cli import (
    run_evaluation,
    run_ingestion_pipeline,
    run_prompt_tuning,
    run_rag_chain,
    run_visualization,
)
from unified_kg_rag.shared.config import ConfigLoader
from unified_kg_rag.shared.exceptions import ConfigurationError

pytestmark = pytest.mark.unit

_CLIS: dict[str, tuple[Callable[[], object], list[str]]] = {
    "run-ingestion": (run_ingestion_pipeline.main, ["--source-directory", "."]),
    "run-rag": (run_rag_chain.main, ["--query", "q"]),
    "run-eval": (run_evaluation.main, ["--eval-data-path", "eval.json"]),
    "run-prompt-tuning": (run_prompt_tuning.main, ["--source-directory", "."]),
    "run-visualization": (run_visualization.main, ["--data-path", "graph.json"]),
}

_BAD_FILES = {
    "wrong type": "processing:\n  max_attempts: abc\n",
    "invalid yaml": "processing: [1\n",
    "not a mapping": "- a\n- b\n",
}


def _run(
    cli: str, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> int | str | None:
    main, args = _CLIS[cli]
    monkeypatch.setattr("sys.argv", [cli, *args, "--config-path", str(config_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


@pytest.mark.parametrize("content", list(_BAD_FILES.values()), ids=list(_BAD_FILES))
@pytest.mark.parametrize("cli", list(_CLIS))
def test_invalid_config_file_is_one_line_and_exit_2(
    cli: str,
    content: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")

    assert _run(cli, config_path, monkeypatch) == 2

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert lines[0].startswith(f"{cli}: error: ")
    assert "Traceback" not in captured.err + captured.out
    assert "unexpected error" not in captured.err + captured.out


@pytest.mark.parametrize("cli", list(_CLIS))
def test_missing_config_file_is_a_usage_error(
    cli: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(cli, tmp_path / "nope.yaml", monkeypatch) == 2
    assert "configuration file not found" in capsys.readouterr().err


def test_validation_error_names_the_key_but_not_the_value(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "processing:\n  max_attempts: s3cr3t-value\n", encoding="utf-8"
    )
    with pytest.raises(ConfigurationError) as exc:
        ConfigLoader(config_path).load_config()
    message = str(exc.value)
    assert message.startswith("Configuration validation error: ")
    assert "processing.max_attempts" in message
    assert "s3cr3t-value" not in message
    assert "\n" not in message
