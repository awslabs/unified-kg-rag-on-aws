# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""docs/configuration.md is generated from the configuration models.

The reference lists every key with its type, default and description; this test
fails when a config field changed without the file being regenerated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "gen_config_reference.py"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_config_reference", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configuration_reference_is_up_to_date() -> None:
    generator = _generator()
    committed = generator.OUTPUT.read_text(encoding="utf-8")
    assert committed == generator.render(), (
        "docs/configuration.md is out of date: run "
        "`uv run python scripts/gen_config_reference.py`"
    )


def test_reference_lists_every_config_leaf() -> None:
    from pydantic import BaseModel

    from unified_kg_rag.domain.models.config import Config

    generator = _generator()
    text = generator.render()

    def leaves(cls: type[BaseModel]):
        for name, info in cls.model_fields.items():
            sub = generator._model_in(info.annotation)
            if sub is None:
                yield name
            else:
                yield from leaves(sub)

    for name in leaves(Config):
        assert f"| `{name}` |" in text, name


def test_reference_lists_every_moved_and_removed_key() -> None:
    from unified_kg_rag.domain.models.config import (
        MOVED_KEY_NOTES,
        MOVED_KEYS,
        REMOVED_KEYS,
    )

    text = _generator().render()
    moved = text.split("## Moved keys", 1)[1].split("## Removed keys", 1)[0]
    removed = text.split("## Removed keys", 1)[1]
    for old, new in MOVED_KEYS.items():
        assert f"| `{old}` | `{new}` |" in moved, old
    for old, note in MOVED_KEY_NOTES.items():
        assert old in MOVED_KEYS
        assert f"- `{old}`: {note}." in moved, old
    for old in REMOVED_KEYS:
        assert f"| `{old}` |" in removed, old
