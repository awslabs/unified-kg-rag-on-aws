# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, model-free embedding provider (``EmbeddingFactoryPort``).

Texts that share words get similar vectors (feature hashing over lowercased
word tokens), so vector search returns meaningful neighbours without calling
an embedding model.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.embeddings import Embeddings


class HashingEmbeddings(Embeddings):
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        # The extra constant component keeps a text without words off the zero
        # vector, which cosine similarity rejects.
        vector = [0.0] * self.dimensions
        vector[0] = 1.0
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[1 + int.from_bytes(digest[:4], "big") % (self.dimensions - 1)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class _ModelInfo:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions


class HashingEmbeddingFactory:
    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def get_model(self, model_id: Any, **kwargs: Any) -> HashingEmbeddings:
        return HashingEmbeddings(self.dimensions)

    def get_model_info(self, model_id: Any) -> _ModelInfo:
        return _ModelInfo(self.dimensions)
