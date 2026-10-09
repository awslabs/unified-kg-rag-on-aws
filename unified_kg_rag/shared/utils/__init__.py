# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from .cache_keys import (
    corpus_manifest_fingerprint,
    stage_cache_key,
    stage_input_fingerprint,
)
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

# NOTE: this package root re-exports only dependency-light helpers, because the
# domain layer imports it and must stay free of LangChain/lxml/rich at import
# time. Import the LangChain-coupled helpers (`BatchProcessor`,
# `BATCH_ITEM_FAILED`, `RobustXMLOutputParser`) from `.langchain`,
# `convert_langchain_to_document` from `.document_converter`, and the rich
# console helpers from `.display`.
#
# `setup_chain` / `create_robust_xml_output_parser` are Bedrock-coupled and
# now live in `unified_kg_rag.adapters.aws.chain_factory` (the shared kernel must
# not depend on adapters). Import them from there.

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
    "corpus_manifest_fingerprint",
    "stage_cache_key",
    "stage_input_fingerprint",
    "strip_embedding_fields",
    "text_digest",
]
