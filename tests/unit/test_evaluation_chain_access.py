# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture guard: evaluation drives the RAG chain through its public
methods only (``aread_build_record``, ``document_retriever``,
``check_filter_*``...), never a private attribute (``chain._get_retriever``),
so the chain can change its internals without breaking ``run-eval``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import unified_kg_rag.evaluation

pytestmark = pytest.mark.unit

_EVALUATION = Path(unified_kg_rag.evaluation.__file__).parent


def _private_chain_accesses(source: str) -> list[str]:
    """``<name>._x`` where the receiver's name ends in ``chain``
    (``chain``, ``rag_chain``, ``self.rag_chain``)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute) or not node.attr.startswith("_"):
            continue
        if node.attr.startswith("__"):
            continue
        receiver = node.value
        name = (
            receiver.id
            if isinstance(receiver, ast.Name)
            else receiver.attr if isinstance(receiver, ast.Attribute) else ""
        )
        if name.endswith("chain"):
            found.append(f"{name}.{node.attr} (line {node.lineno})")
    return found


def test_the_guard_finds_a_private_chain_access() -> None:
    assert _private_chain_accesses("self.rag_chain._get_retriever(1)") == [
        "rag_chain._get_retriever (line 1)"
    ]
    assert _private_chain_accesses("chain.aread_build_record(s)") == []


def test_evaluation_reads_no_private_chain_attribute() -> None:
    offenders = {
        str(path.relative_to(_EVALUATION)): accesses
        for path in sorted(_EVALUATION.rglob("*.py"))
        if (accesses := _private_chain_accesses(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}
