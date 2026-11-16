# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The closed-book arm of ``run-eval --closed-book``: answers without retrieval.

``closed_book_chain`` is a runnable ``EvaluationManager`` uses in place of
the RAG chain. It asks the answer model (``models.roles.answer_generation``,
``ClosedBookAnswerPrompt``) each question with no context: no retriever,
store or index is touched. Its scores bound what the answer model knows
without the corpus, so a strategy that does not beat the arm on a question
set measures prior knowledge, not retrieval.

It answers at each query's effort, as the strategies do (an effort the
answer model refuses fails the run before any answer). Each answer reports
no source, ``metadata.retrieval: "none"`` and
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
from unified_kg_rag.application.retrieval.rag_chain import (
    answer_reasoning,
    check_answer_effort,
)
from unified_kg_rag.domain.models import Config, ModelPurpose, ModelRole
from unified_kg_rag.domain.models.config import EffortLevel
from unified_kg_rag.domain.prompts import ClosedBookAnswerPrompt
from unified_kg_rag.shared.usage import usage_scope

# The arm's label: its strategy in the results, file names and comparison.
CLOSED_BOOK_ARM = "closed_book"


class ClosedBookChain(RunnableLambda[dict[str, Any], dict[str, Any]]):
    """The closed-book runnable, with the RAG chain's ``check_effort``: the
    arm answers at each query's effort, as the strategies do, so ``run-eval``
    checks the efforts it asks for before answering any."""

    def __init__(self, config: Config, providers: Providers) -> None:
        self._config = config
        self._factory = providers.llm_factory
        # One answer chain per reasoning setting, built on first use.
        self._chains: dict[tuple[Any, ...], Runnable[Any, Any]] = {}
        super().__init__(self._answer, name="ClosedBookAnswer")

    def check_effort(self, effort: EffortLevel | None) -> None:
        """Refuse an ``effort`` the answer model cannot apply
        (:func:`check_answer_effort`, as the RAG chain does).

        Raises:
            InvalidArgumentError: Naming the model and what it takes.
        """
        check_answer_effort(self._config, self._factory, effort)

    def _chain(self, reasoning: dict[str, Any]) -> Runnable[Any, Any]:
        key = tuple(sorted(reasoning.items()))
        if (chain := self._chains.get(key)) is None:
            chain = self._chains[key] = setup_chain(
                self._factory,
                self._config.model_id(ModelRole.ANSWER_GENERATION),
                ClosedBookAnswerPrompt,
                StrOutputParser(),
                custom_prompts=self._config.custom_prompts,
                model_purpose=ModelPurpose.QUERY,
                **reasoning,
            )
        return chain

    async def _answer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.check_effort(inputs.get("effort"))
        answer = self._chain(answer_reasoning(inputs))
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


def closed_book_chain(
    config: Config, providers: Providers | None = None
) -> ClosedBookChain:
    """A runnable answering ``{"query": ..., "effort": ...}`` from the
    answer model alone, at the query's effort (``RAGInput.effort``; unset:
    the model tier's), as ``{"answer", "sources": [], "metadata"}`` (the
    shape ``EvaluationManager`` reads from the RAG chain)."""
    return ClosedBookChain(config, Providers.resolve(config, providers))


def closed_book_arms(names: Iterable[str]) -> list[str]:
    """The closed-book arms among compared run labels: ``closed_book`` and an
    earlier run's ``baseline:closed_book``."""
    return [
        name
        for name in names
        if name == CLOSED_BOOK_ARM or name.endswith(f":{CLOSED_BOOK_ARM}")
    ]
