# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every recovery note the extraction store keeps (the note keys of
``application/storage/stored_items.py``) is named wherever a list of them
tells an operator when the index is safe to delete: the field's description
(so docs/reference/configuration.md), the runbook's row and the CHANGELOG
bullet. The lists were kept by hand and fell behind twice (R2D-3, R3D-4)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from unified_kg_rag.application.storage import stored_items
from unified_kg_rag.domain.models.config import OpenSearchIndexingConfig

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


_NOTE_KEY = re.compile(r"[a-z]+(?:-[a-z]+)+")
_PACKAGE = Path(stored_items.__file__).resolve().parents[2]


def _note_constants() -> dict[str, str]:
    """The module's note-key constants (``NAME = "a-key"``, annotated or
    not), by name."""
    tree = ast.parse(Path(stored_items.__file__).read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id.isupper()
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and _NOTE_KEY.fullmatch(value.value)
        ):
            constants[target.id] = value.value
    return constants


def _note_keys() -> set[str]:
    keys = set(_note_constants().values())
    assert keys >= {stored_items.PENDING_STRIP, stored_items.REMOVED_RECORDS}
    return keys


def _listing(text: str, marker: str) -> str:
    """The paragraph, bullet or table row of ``text`` that holds ``marker``
    (its words, however the lines wrap)."""
    found = re.search(r"\s+".join(map(re.escape, marker.split())), text)
    assert found, marker
    start = found.start()
    begin = max(text.rfind("\n\n", 0, start), text.rfind("\n- ", 0, start))
    end = min(
        i for i in (text.find("\n\n", start), text.find("\n- ", start)) if i != -1
    )
    return text[begin:end]


def test_every_note_is_listed_where_its_index_is_described() -> None:
    keys = _note_keys()
    field = OpenSearchIndexingConfig.model_fields["extraction_records_index_prefix"]
    operations = (REPO_ROOT / "docs" / "operations.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    listings = {
        "extraction_records_index_prefix": field.description or "",
        "docs/operations.md": _listing(operations, "| Extraction store |"),
        "CHANGELOG.md": _listing(
            changelog, "so a run stopped mid-removal or mid-commit cannot resume them"
        ),
    }
    for where, text in listings.items():
        missing = sorted(k for k in keys if k not in text)
        assert not missing, f"{where} does not name {missing}"


def _derives_from(function: ast.AST, constants: set[str]) -> bool:
    """Whether ``function`` builds a key from a note constant
    (``f"{PENDING_WRITES}-..."``: the parts of a split note)."""
    return any(
        isinstance(node, ast.JoinedStr)
        and node.values
        and isinstance(node.values[0], ast.FormattedValue)
        and isinstance(node.values[0].value, ast.Name)
        and node.values[0].value.id in constants
        for node in ast.walk(function)
    )


def test_every_note_written_is_a_listed_note_key() -> None:
    # A note key defined anywhere but stored_items (or written as a literal)
    # would be missing from the lists the test above checks.
    constants = set(_note_constants())
    stray = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for call in ast.walk(function):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in {"_write_note", "_delete_note"}
                    and call.args
                ):
                    continue
                key = call.args[0]
                if isinstance(key, ast.Name) and (
                    key.id in constants or _derives_from(function, constants)
                ):
                    continue
                stray.append(f"{path.relative_to(_PACKAGE)}:{call.lineno}")
    assert not stray, f"note keys not defined in stored_items: {stray}"
