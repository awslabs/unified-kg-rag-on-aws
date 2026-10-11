# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions from the cycle-18 review (AWS-free)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from unified_kg_rag.application.cli import run_rag_chain
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.evaluation.text_matching import normalize_answer, phrase_in_text

pytestmark = pytest.mark.unit


async def test_run_rag_passes_only_chain_keywords(mocker) -> None:
    """run-rag's chain kwargs must bind to GraphRAGChain (it raised TypeError)."""
    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(aclose=None)

    mocker.patch.object(run_rag_chain, "create_rag_chain", side_effect=fake_create)
    cli = object.__new__(run_rag_chain.RAGChainRunner)
    cli.config = object()
    cli.args = SimpleNamespace(mode="rag", enable_thinking=True)
    await cli._initialize_chain()
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


@pytest.mark.parametrize(
    ("rewrite", "usable"),
    [
        (
            "I can't find the termination clause in the Vendor agreement; where is it?",
            True,
        ),
        ("Sorry, which payment terms apply to the Buyer?", True),
        ("I'm sorry, I can't help with that.", False),
        ("I\u2019m sorry, I can\u2019t help with that.", False),
        ("What are the Vendor's payment terms?", True),
        # An apology that asks the user to clarify is a meta-reply.
        ("I'm sorry, could you clarify which contract?", False),
        ("Sorry, which agreement do you mean?", False),
        ("I'm sorry, please specify the contract.", False),
        ("I can't tell; are you referring to the Vendor agreement?", False),
        # Without a refusal opening the same words are a real question.
        ("Could you clarify which clause of the Vendor agreement applies?", True),
    ],
)
def test_rewrite_refusal_guard(rewrite: str, usable: bool) -> None:
    assert (GraphRAGChain._usable_rewrite(rewrite) is not None) is usable


def test_chunkers_reject_unknown_keywords(config) -> None:
    from unified_kg_rag.adapters.ingestion.chunker import IntelligentTextChunker

    with pytest.raises(TypeError):
        IntelligentTextChunker(config, boto_session=object())


def test_pipeline_rejects_session_alongside_providers(config) -> None:
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import PipelineConfig

    with pytest.raises(ValueError, match="not both"):
        DataIngestionPipeline(
            config,
            PipelineConfig(),
            providers=Providers(config),
            boto_session=object(),
        )
