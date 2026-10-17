# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The embedding settings an OpenSearch index holds, and the configured ones.

``OpenSearchIndexer`` records the embedding model and dimension each index is
built with in its mapping's ``_meta`` (keys below). The retriever reads them
back to refuse a query embedded with another model or dimension, whose kNN
search would either fail on the dimension or rank by unrelated vectors.

Ingestion also records what built the index (``META_BUILD``: an
``IndexBuildRecord``), which ``run-eval`` copies into its run manifest.
"""

from collections.abc import Mapping
from typing import Any

from unified_kg_rag.domain.models.config import (
    OpenSearchIndexingConfig,
    model_id_value,
)
from unified_kg_rag.ports.model_factory import EmbeddingFactoryPort
from unified_kg_rag.shared.exceptions import ConfigurationError

# Keys of the embedding settings an index records in its mapping's _meta.
META_MODEL = "embedding_model_id"
META_DIMENSION = "embedding_dimension"
# The latest ingestion run that wrote the index (an IndexBuildRecord dump).
META_BUILD = "build"


def configured_embedding_model(opensearch_config: OpenSearchIndexingConfig) -> str:
    """The configured embedding model id, as an index records it."""
    return model_id_value(opensearch_config.embedding_model_id)


def resolve_embedding_dimension(
    embedding_factory: EmbeddingFactoryPort, opensearch_config: OpenSearchIndexingConfig
) -> int:
    """The vector dimension the configured embedding model produces.

    ``indexing.opensearch.embedding_dimension`` when set (it must be one the
    model supports, if the model reports its dimensions), otherwise the
    model's largest supported dimension.

    Raises:
        ConfigurationError: The model is unknown, does not support the
            configured dimension, or reports no dimension while none is
            configured.
    """
    model = configured_embedding_model(opensearch_config)
    model_info = embedding_factory.get_model_info(opensearch_config.embedding_model_id)
    if not model_info:
        raise ConfigurationError(f"Unsupported model: '{model}'")
    supported = model_info.dimensions
    if (dim := opensearch_config.embedding_dimension) is not None:
        is_supported = (
            supported is None
            or (isinstance(supported, list) and dim in supported)
            or (isinstance(supported, int) and dim == supported)
        )
        if not is_supported:
            raise ConfigurationError(
                f"Dimension {dim} not supported by '{model}'. "
                f"Supported: '{supported}'"
            )
        return dim
    if isinstance(supported, list) and supported:
        return int(supported[-1])
    if isinstance(supported, int):
        return supported
    # Guessing a dimension would build an index the model's vectors do not fit.
    raise ConfigurationError(
        f"Embedding model '{model}' reports no vector dimension; set "
        "indexing.opensearch.embedding_dimension."
    )


def stored_embedding_settings(
    mappings: Mapping[str, Any],
) -> tuple[str | None, set[int]]:
    """The model id (``None`` if unrecorded) and dimensions an index holds.

    The dimensions are those of the index's ``knn_vector`` fields, plus the one
    its ``_meta`` records.
    """
    meta = mappings.get("_meta") or {}
    dimensions = {
        field["dimension"]
        for field in (mappings.get("properties") or {}).values()
        if isinstance(field, dict)
        and field.get("type") == "knn_vector"
        and isinstance(field.get("dimension"), int)
    }
    if isinstance(meta.get(META_DIMENSION), int):
        dimensions.add(meta[META_DIMENSION])
    model = meta.get(META_MODEL)
    return (model if isinstance(model, str) else None), dimensions


def stored_build_record(mappings: Mapping[str, Any]) -> dict[str, Any] | None:
    """The build record an index's ``_meta`` holds (``None`` if unrecorded)."""
    record = (mappings.get("_meta") or {}).get(META_BUILD)
    return record if isinstance(record, dict) else None
