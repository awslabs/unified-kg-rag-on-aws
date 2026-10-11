# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-selection flags shared by run-ingestion, run-rag and run-eval.

Each flag overrides one ``backend`` / ``providers`` key of the loaded config
(see ``application.bootstrap.with_provider_overrides``); omitted flags leave
the config as it is. ``--backend local`` is the same as ``backend: local`` in
the file: it fills in the local providers and stores for every key the file
leaves unset.
"""

from __future__ import annotations

import argparse

from unified_kg_rag.application.bootstrap import with_provider_overrides
from unified_kg_rag.domain.models import Backend, Config


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the provider flags to ``parser`` (one argument group)."""
    group = parser.add_argument_group(
        "providers",
        "Override the config's provider selection (see the user guide's "
        "'Local mode').",
    )
    group.add_argument(
        "--backend",
        choices=[b.value for b in Backend],
        help="Provider preset: 'aws' (Bedrock, Neptune, OpenSearch, DynamoDB) or "
        "'local' (Ollama, cross-encoder, Gremlin Server, local OpenSearch, SQLite)",
    )
    group.add_argument(
        "--llm-provider",
        metavar="KIND",
        help="providers.llm.kind: bedrock, openai_compatible or ollama",
    )
    group.add_argument(
        "--llm-model",
        metavar="MODEL",
        help="providers.llm.default_model_id (non-Bedrock language models)",
    )
    group.add_argument(
        "--embedding-provider",
        metavar="KIND",
        help="providers.embedding.kind: bedrock, openai_compatible or ollama",
    )
    group.add_argument(
        "--embedding-model",
        metavar="MODEL",
        help="providers.embedding.model_id (non-Bedrock embeddings)",
    )
    group.add_argument(
        "--embedding-dimension",
        type=int,
        metavar="N",
        help="providers.embedding.dimension (non-Bedrock embeddings)",
    )
    group.add_argument(
        "--rerank-provider",
        metavar="KIND",
        help="providers.rerank.kind: bedrock, cross_encoder or none",
    )
    group.add_argument(
        "--model-base-url",
        metavar="URL",
        help="providers.llm.base_url and providers.embedding.base_url (the "
        "Ollama or OpenAI-compatible server)",
    )


def apply_provider_arguments(config: Config, args: argparse.Namespace) -> Config:
    """``config`` with the provider flags in ``args`` applied."""
    return with_provider_overrides(
        config,
        backend=getattr(args, "backend", None),
        llm_kind=getattr(args, "llm_provider", None),
        llm_model_id=getattr(args, "llm_model", None),
        embedding_kind=getattr(args, "embedding_provider", None),
        embedding_model_id=getattr(args, "embedding_model", None),
        embedding_dimension=getattr(args, "embedding_dimension", None),
        rerank_kind=getattr(args, "rerank_provider", None),
        base_url=getattr(args, "model_base_url", None),
    )
