# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``.env`` files: never loaded at import, only from the CLI's directory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from unified_kg_rag.application.config import load_cli_dotenv

pytestmark = pytest.mark.unit

_PROBE = """
import os
import dotenv

calls = []
dotenv.load_dotenv = lambda *args, **kwargs: calls.append((args, kwargs))
before = dict(os.environ)

import unified_kg_rag
import unified_kg_rag.shared
import unified_kg_rag.application.ingestion.pipeline
import unified_kg_rag.application.retrieval.rag_chain
import unified_kg_rag.application.cli.run_rag_chain
import unified_kg_rag.application.cli.run_evaluation
import unified_kg_rag.application.cli.run_visualization
import unified_kg_rag.application.cli.run_prompt_tuning
from unified_kg_rag.application.config import get_config

get_config()
assert calls == [], calls
assert dict(os.environ) == before, set(os.environ) ^ set(before)
"""


def test_importing_the_package_loads_no_dotenv(tmp_path: Path) -> None:
    # A .env in the working directory and in a parent: a library user's
    # environment must not change by importing the package.
    (tmp_path / ".env").write_text("UKG_TEST_PARENT_ENV=1\n")
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / ".env").write_text("UKG_TEST_CWD_ENV=1\n")
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=100,
        env={k: v for k, v in os.environ.items() if not k.startswith("UKG_TEST_")},
    )
    assert result.returncode == 0, result.stderr


def test_cli_loads_only_the_working_directorys_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("UKG_TEST_PARENT_ENV=parent\n")
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("UKG_TEST_PARENT_ENV", raising=False)

    # No .env here: the parent's is not searched for.
    assert load_cli_dotenv() is None
    assert "UKG_TEST_PARENT_ENV" not in os.environ

    (cwd / ".env").write_text("UKG_TEST_CWD_ENV=file\nUKG_TEST_SET_ENV=file\n")
    monkeypatch.setenv("UKG_TEST_SET_ENV", "environment")
    monkeypatch.delenv("UKG_TEST_CWD_ENV", raising=False)
    assert load_cli_dotenv() == cwd / ".env"
    assert os.environ["UKG_TEST_CWD_ENV"] == "file"
    # A variable already set wins over the file.
    assert os.environ["UKG_TEST_SET_ENV"] == "environment"
    monkeypatch.delenv("UKG_TEST_CWD_ENV")
