# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every recovery note the extraction store keeps (the note keys of
``application/storage/stored_items.py``) is named wherever a list of them
tells an operator when the index is safe to delete: the field's description
(so docs/reference/configuration.md), the runbook's row and the CHANGELOG
bullet. The lists were kept by hand and fell behind twice (R2D-3, R3D-4)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from unified_kg_rag.application.storage import stored_items
from unified_kg_rag.domain.models.config import OpenSearchIndexingConfig

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _note_keys() -> set[str]:
    """The module's note-key constants (``NAME = "a-key"``)."""
    source = Path(stored_items.__file__).read_text(encoding="utf-8")
    keys = set(re.findall(r'^[A-Z_]+ = "([a-z]+(?:-[a-z]+)+)"$', source, re.M))
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
