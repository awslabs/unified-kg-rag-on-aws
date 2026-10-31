# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from .common import (
    EMBEDDING_FIELD_SUFFIX,
    clean_display_name,
    compute_hash,
    default_max_workers,
    ensure_list,
    entity_key,
    generate_stable_id,
    normalize_name,
    parse_llm_json,
    safe_float_parse,
    strip_embedding_fields,
    text_digest,
)
from .concurrency import ContextThreadPoolExecutor
from .event_loop import configure_event_loop

# NOTE: the shared kernel imports no other layer, and this package root
# re-exports only dependency-light helpers because the domain layer imports
# it. The LangChain-coupled helpers (`BatchProcessor`, `RobustXMLOutputParser`)
# live in `unified_kg_rag.adapters.llm.langchain`, the stage cache keys and
# document identity in `unified_kg_rag.domain.ingestion`, the rich console
# helpers in `unified_kg_rag.application.cli.display`.

__all__ = [
    "EMBEDDING_FIELD_SUFFIX",
    "ContextThreadPoolExecutor",
    "clean_display_name",
    "compute_hash",
    "configure_event_loop",
    "default_max_workers",
    "ensure_list",
    "entity_key",
    "generate_stable_id",
    "normalize_name",
    "parse_llm_json",
    "safe_float_parse",
    "strip_embedding_fields",
    "text_digest",
]
