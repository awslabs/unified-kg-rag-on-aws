# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions from the cycle-18 review (AWS-free)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from tests.fixtures.run_rag import patch_chain
from unified_kg_rag.application.cli import run_rag_chain
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.evaluation.text_matching import normalize_answer, phrase_in_text

pytestmark = pytest.mark.unit


async def test_run_rag_passes_only_chain_keywords(mocker) -> None:
    """run-rag's chain kwargs must bind to GraphRAGChain (it raised TypeError)."""
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(aclose=None)

    patch_chain(mocker, side_effect=fake_create)
    cli = object.__new__(run_rag_chain.RAGChainRunner)
    cli.config = object()
    cli.args = SimpleNamespace(mode="rag", enable_thinking=True)
    cli.metrics_sink = None
    await cli._initialize_chain()
    # bind() of an empty dict passes, so first prove the chain was created.
    assert captured["config"] is cli.config
    inspect.signature(GraphRAGChain.__init__).bind(None, **captured)


@pytest.mark.parametrize(
    ("gold", "answer", "expected"),
    [
        ("Type-A", "Type-B is used", False),
        ("A-1", "Plan B-1 applies", False),
        ("Vitamin-A", "Vitamin B", False),
        ("Type-A", "type-a applies", True),
        ("5-year", "a 5 year term", True),
        ("1995", "from 1990-1995", True),
    ],
)
def test_dashes_split_words_but_not_lone_letters(gold, answer, expected) -> None:
    assert phrase_in_text(gold, answer) is expected


def test_lone_letter_dash_keeps_one_token() -> None:
    assert normalize_answer("Type-A") == "typea"
    assert normalize_answer("5-year") == "5 year"


def test_chunkers_reject_unknown_keywords(config) -> None:
    from unified_kg_rag.adapters.ingestion.chunker import IntelligentTextChunker

    with pytest.raises(TypeError):
        IntelligentTextChunker(config, boto_session=object())


def test_entry_points_take_the_session_on_the_providers_only(config) -> None:
    # The session is the Providers bundle's (Providers(config, boto_session=)):
    # the constructors' own boto_session duplicated it.
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import PipelineConfig

    with pytest.raises(TypeError, match="boto_session"):
        DataIngestionPipeline(config, PipelineConfig(), boto_session=object())
    with pytest.raises(TypeError, match="boto_session"):
        GraphRAGChain(config, boto_session=object())
