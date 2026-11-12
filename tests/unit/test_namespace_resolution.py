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

import ast
import re

import pytest

from tests.fixtures.source_guards import SourceGuard, matches, nodes, text_matching
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.retrieval.store_names import (
    namespace_suffix,
    resolve_namespace,
)

pytestmark = pytest.mark.unit


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


# One rule, one owner (tests/fixtures/source_guards, as the other
# single-owner guards in test_single_owner_guards.py).
GUARD = SourceGuard(
    name="namespace rule",
    use="resolve_namespace/namespace_suffix",
    owners=("domain/retrieval/store_names.py",),
    find=text_matching(_COPIES),
)


@pytest.mark.slow
def test_no_copy_of_the_namespace_rule_remains() -> None:
    found = [str(hit) for hit in GUARD.copies()]
    assert not found, f"use {GUARD.use}: {found}"
    assert GUARD.hits(), "the owner no longer matches the pattern it guards"


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


def _alias(node: ast.AST) -> bool:
    """A function that only returns ``resolve_namespace(...)``: the rule
    under another name (``namespace_of``, ``namespace_name``), which drifts
    from it as soon as one of them changes."""
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return False
    body = [
        b
        for b in node.body
        if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
    ]
    return (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Call)
        and ast.unparse(body[0].value.func).endswith("resolve_namespace")
    )


ALIASES = SourceGuard(
    name="namespace rule aliases",
    use="resolve_namespace itself",
    owners=(),
    find=nodes(_alias),
)


def test_no_function_renames_the_namespace_rule() -> None:
    assert not ALIASES.copies(), [str(hit) for hit in ALIASES.copies()]
    assert matches(
        nodes(_alias),
        "def namespace_of(config, value=None):\n"
        "    return resolve_namespace(config, value)\n",
    )
