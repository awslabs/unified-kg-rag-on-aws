# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for the chain ``run-rag`` builds."""

from __future__ import annotations

from typing import Any

from unified_kg_rag.application.cli import run_rag_chain
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain


def patch_chain(mocker: Any, **kwargs: Any) -> Any:
    """Patch the ``GraphRAGChain`` ``run-rag`` constructs (``kwargs`` go to
    ``mocker.patch.object``: ``return_value=``, ``side_effect=``); the CLI's
    filter conversion (``GraphRAGChain.typed_filter_values``) stays real.
    Returns the mock class."""
    chain_cls = mocker.patch.object(run_rag_chain, "GraphRAGChain", **kwargs)
    chain_cls.typed_filter_values = GraphRAGChain.typed_filter_values
    return chain_cls
