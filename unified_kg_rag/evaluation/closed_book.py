# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The closed-book arm of ``run-eval --closed-book``: answers without retrieval.

``closed_book_chain`` is a runnable ``EvaluationManager`` uses in place of
the RAG chain. It asks the answer model (``models.roles.answer_generation``,
``ClosedBookAnswerPrompt``) each question with no context: no retriever,
store or index is touched. Its scores bound what the answer model knows
without the corpus, so a strategy that does not beat the arm on a question
set measures prior knowledge, not retrieval.

Each answer reports no source, ``metadata.retrieval: "none"`` and
``metadata.closed_book: true``, and the arm's label (``CLOSED_BOOK_ARM``) as
its strategy; its usage and processing time are recorded like a strategy's.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, ModelPurpose, ModelRole
from unified_kg_rag.domain.prompts import ClosedBookAnswerPrompt
from unified_kg_rag.shared.usage import usage_scope

# The arm's label: its strategy in the results, file names and comparison.
CLOSED_BOOK_ARM = "closed_book"


def closed_book_chain(
    config: Config, providers: Providers | None = None
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """A runnable answering ``{"query": ...}`` from the answer model alone,
    as ``{"answer", "sources": [], "metadata"}`` (the shape
    ``EvaluationManager`` reads from the RAG chain)."""
    providers = Providers.resolve(config, providers)
    answer = setup_chain(
        providers.llm_factory,
        config.model_id(ModelRole.ANSWER_GENERATION),
        ClosedBookAnswerPrompt,
        StrOutputParser(),
        custom_prompts=config.custom_prompts,
        model_purpose=ModelPurpose.QUERY,
    )

    async def _answer(inputs: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        with usage_scope() as meter:
            text = await answer.ainvoke({"query": inputs["query"]})
        return {
            "answer": str(text).strip(),
            "sources": [],
            "metadata": {
                "search_strategy": CLOSED_BOOK_ARM,
                "retrieval": "none",
                "closed_book": True,
                "processing_time": time.monotonic() - start,
                "usage": meter.snapshot(),
            },
        }

    return RunnableLambda(_answer, name="ClosedBookAnswer")


def closed_book_arms(names: Iterable[str]) -> list[str]:
    """The closed-book arms among compared run labels: ``closed_book`` and an
    earlier run's ``baseline:closed_book``."""
    return [
        name
        for name in names
        if name == CLOSED_BOOK_ARM or name.endswith(f":{CLOSED_BOOK_ARM}")
    ]
