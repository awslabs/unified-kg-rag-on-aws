# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation manager and the LLM-free evaluators.

The evaluator base classes and the langchain/ragas evaluators are adapters:
import them from ``unified_kg_rag.adapters.evaluators.*``.

The names below load on first use: the manager imports the RAG chain and
every search strategy (seconds), which a submodule such as ``gates`` or
``comparison`` (``run-eval --help``, ``--from-results``) never needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .answer_match_evaluator import AnswerMatchEvaluator
    from .evaluation_manager import EvaluationManager
    from .graph_aware_evaluator import GraphAwareEvaluator
    from .retrieval_evaluator import RetrievalEvaluator

__all__ = [
    "AnswerMatchEvaluator",
    "EvaluationManager",
    "GraphAwareEvaluator",
    "RetrievalEvaluator",
]


def __getattr__(name: str) -> Any:
    value: Any
    if name == "AnswerMatchEvaluator":
        from .answer_match_evaluator import AnswerMatchEvaluator as value
    elif name == "EvaluationManager":
        from .evaluation_manager import EvaluationManager as value
    elif name == "GraphAwareEvaluator":
        from .graph_aware_evaluator import GraphAwareEvaluator as value
    elif name == "RetrievalEvaluator":
        from .retrieval_evaluator import RetrievalEvaluator as value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
