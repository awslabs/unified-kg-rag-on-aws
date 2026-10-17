# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The embedding dimension an OpenSearch index is built with (AWS-free)."""

from __future__ import annotations

from typing import Any

import pytest

from unified_kg_rag.adapters.storage.embedding_settings import (
    resolve_embedding_dimension,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.ports.model_factory import EmbeddingModelInfo
from unified_kg_rag.shared import ConfigurationError

pytestmark = pytest.mark.unit


class _Factory:
    def __init__(self, info: EmbeddingModelInfo | None) -> None:
        self.info = info

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_model_info(self, model_id: Any) -> EmbeddingModelInfo | None:
        return self.info


def _resolve(info: EmbeddingModelInfo | None, dimension: int | None = None) -> int:
    config = Config.model_validate({"models": {"embedding": {"dimension": dimension}}})
    return resolve_embedding_dimension(_Factory(info), config)  # type: ignore[arg-type]


def test_largest_supported_dimension_is_the_default() -> None:
    assert _resolve(EmbeddingModelInfo(dimensions=[256, 512, 1024])) == 1024
    assert _resolve(EmbeddingModelInfo(dimensions=768)) == 768


def test_unknown_model_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported model"):
        _resolve(None)


def test_unsupported_configured_dimension_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="Dimension 300 not supported"):
        _resolve(EmbeddingModelInfo(dimensions=[256, 512]), dimension=300)


def test_model_without_dimensions_needs_a_configured_one() -> None:
    # It used to fall back to 1024 silently, building an index the model's
    # vectors may not fit.
    with pytest.raises(ConfigurationError, match="models.embedding.dimension"):
        _resolve(EmbeddingModelInfo())


def test_model_without_dimensions_uses_the_configured_one() -> None:
    assert _resolve(EmbeddingModelInfo(), dimension=384) == 384
