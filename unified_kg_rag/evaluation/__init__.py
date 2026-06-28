# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Any

from .base import BaseEvaluator, BaseGraphRAGEvaluator
from .evaluation_manager import EvaluationManager
from .graph_aware_evaluator import GraphAwareEvaluator

if TYPE_CHECKING:
    from unified_kg_rag.adapters.evaluators.langchain_evaluator import (
        LangChainEvaluator,
    )
    from unified_kg_rag.adapters.evaluators.ragas_evaluator import RagasEvaluator

__all__ = [
    "BaseEvaluator",
    "BaseGraphRAGEvaluator",
    "EvaluationManager",
    "GraphAwareEvaluator",
    "LangChainEvaluator",
    "RagasEvaluator",
]

# The langchain/ragas adapter evaluators import `unified_kg_rag.evaluation.base`,
# so eagerly importing them here would re-enter this partially-initialized
# package (circular import, e.g. when an adapter is the import entry point under
# coverage). Expose them LAZILY (PEP 562) so the public import path
# `from unified_kg_rag.evaluation import LangChainEvaluator` still works, but the
# adapter is only loaded on first access — after this package is fully built.
_LAZY_EXPORTS = {
    "LangChainEvaluator": "unified_kg_rag.adapters.evaluators.langchain_evaluator",
    "RagasEvaluator": "unified_kg_rag.adapters.evaluators.ragas_evaluator",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
