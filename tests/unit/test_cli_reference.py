# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""docs/cli.md is generated from the CLI parsers.

The reference lists every flag with its default and help text; this test fails
when a flag changed without the file being regenerated.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import entry_points
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "gen_cli_reference.py"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_cli_reference", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_reference_is_up_to_date() -> None:
    generator = _generator()
    committed = generator.OUTPUT.read_text(encoding="utf-8")
    assert (
        committed == generator.render()
    ), "docs/cli.md is out of date: run `uv run python scripts/gen_cli_reference.py`"


def test_reference_covers_every_script() -> None:
    scripts = {
        script.name
        for script in entry_points(group="console_scripts")
        if script.value.startswith("unified_kg_rag.application.cli.")
    }
    documented = {command for command, _, _ in _generator().COMMANDS}
    assert documented == scripts


def test_reference_lists_every_flag() -> None:
    generator = _generator()
    text = generator.render()
    for command, parser in generator.build_parsers().items():
        for action in generator.documented_actions(parser):
            names = action.option_strings or [action.metavar or action.dest]
            assert f"`{names[0]}`" in text, (command, names[0])


def test_provider_flags_are_the_same_on_every_cli() -> None:
    generator = _generator()
    groups = [
        [
            (action.option_strings, action.help)
            for action in generator.provider_group(parser)._group_actions
        ]
        for parser in generator.build_parsers().values()
        if generator.provider_group(parser) is not None
    ]
    assert len(groups) >= 2
    assert all(group == groups[0] for group in groups)


def test_environment_defaults_do_not_leak_into_the_reference(monkeypatch) -> None:
    generator = _generator()
    expected = generator.render()
    monkeypatch.setenv("UKR_SOURCE_DIRECTORY", "/somewhere/else")
    monkeypatch.setenv("UKR_PIPELINE_ID", "run-from-env")
    assert generator.render() == expected
