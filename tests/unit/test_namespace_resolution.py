# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One rule names the index namespace a query or a run addresses.

``resolve_namespace(config, suffix)`` (the suffix, else the config's
``index_value``, else ``default``) and ``namespace_suffix`` (a store's own
default) live in ``domain/retrieval/store_names.py``. Copies of the rule
disagreed: a query without a suffix read ``default`` where ingestion with the
same config wrote ``index_value``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.retrieval.store_names import (
    namespace_suffix,
    resolve_namespace,
)

pytestmark = pytest.mark.unit

PACKAGE = Path(__file__).resolve().parents[2] / "unified_kg_rag"
OWNER = PACKAGE / "domain" / "retrieval" / "store_names.py"

# An ad-hoc fallback to the default suffix or to the config's index_value.
_COPIES = re.compile(
    r"\bor\s*\(?\s*(Constants\.DEFAULT_SUFFIX|DEFAULT_NAMESPACE|[\"']default[\"'])"
    r"|document_parsing\.index_value\s*\)?\s*or\b"
    r"|\bor\s+[\w.]*document_parsing\.index_value"
)


def _config(index_value: str | None) -> Config:
    config = Config()
    config.processing.document_parsing.index_value = index_value
    return config


@pytest.mark.parametrize(
    ("index_value", "suffix", "expected"),
    [
        (None, None, "default"),
        ("tenant", None, "tenant"),
        ("tenant", "other", "other"),
        (None, "other", "other"),
        ("tenant", "", "tenant"),
    ],
)
def test_resolve_namespace(
    index_value: str | None, suffix: str | None, expected: str
) -> None:
    assert resolve_namespace(_config(index_value), suffix) == expected


def test_namespace_suffix_defaults_without_a_config() -> None:
    assert namespace_suffix(None) == namespace_suffix("") == "default"
    assert namespace_suffix("tenant") == "tenant"


def test_no_copy_of_the_namespace_rule_remains() -> None:
    found = [
        f"{path.relative_to(PACKAGE)}:{text[: m.start()].count(chr(10)) + 1}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path != OWNER
        for text in [path.read_text(encoding="utf-8")]
        for m in _COPIES.finditer(text)
    ]
    assert not found, f"use resolve_namespace/namespace_suffix: {found}"


def test_the_copy_pattern_catches_the_old_forms() -> None:
    # The pin of the grep: each form the copies took is matched.
    for form in (
        "suffix or Constants.DEFAULT_SUFFIX.value",
        "x = (\n    suffix\n    or config.processing.document_parsing.index_value\n)",
        "config.processing.document_parsing.index_value\n    or default",
        'item.get("suffix") or "default"',
        "suffix or DEFAULT_NAMESPACE",
    ):
        assert _COPIES.search(form), form
