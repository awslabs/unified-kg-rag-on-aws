# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping
from typing import Any

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from unified_kg_rag.domain.models.config import LayoutSettings
from unified_kg_rag.shared import get_logger

from .node_embedder import NodeEmbeddings

logger = get_logger(__name__)


class DimensionalityReducer:
    def __init__(self, config: LayoutSettings | Mapping[str, Any]):
        settings = LayoutSettings.model_validate(config)
        self.umap_config = settings.umap
        self.tsne_config = settings.tsne
        self.pca_config = settings.pca

    def reduce_dimensions(
        self, embeddings: NodeEmbeddings, method: str = "umap"
    ) -> dict[str, tuple[float, float]]:
        if not embeddings.embeddings:
            logger.warning("No embeddings to reduce.")
            return {}

        logger.info(
            "Reducing dimensions of %s nodes using %s...",
            len(embeddings.embeddings),
            method.upper(),
        )

        nodes = embeddings.nodes
        embedding_matrix = np.array([embeddings.embeddings[node] for node in nodes])

        # UMAP's spectral init needs more than n_components + 1 points.
        if embedding_matrix.shape[0] <= self.umap_config.n_components + 1:
            logger.warning(
                "Not enough data points (%s) for dimensionality reduction, falling back to PCA.",
                embedding_matrix.shape[0],
            )
            method = "pca"

        if embedding_matrix.shape[0] < 2:
            logger.warning(
                "Cannot perform dimensionality reduction with less than 2 nodes."
            )
            return {nodes[0]: (0.0, 0.0)} if nodes else {}

        # A failure propagates: the caller falls back to a seeded topology
        # layout and marks it degraded, instead of a random layout that looks
        # valid.
        if method.lower() == "umap":
            reduced_matrix = self._apply_umap(embedding_matrix)
        elif method.lower() == "tsne":
            reduced_matrix = self._apply_tsne(embedding_matrix)
        else:
            reduced_matrix = self._apply_pca(embedding_matrix)

        layout = {
            node: (float(reduced_matrix[i, 0]), float(reduced_matrix[i, 1]))
            for i, node in enumerate(nodes)
        }
        logger.info("Dimensionality reduction completed.")
        return layout

    def _apply_umap(self, embeddings: np.ndarray) -> np.ndarray:
        reducer = umap.UMAP(
            n_neighbors=min(self.umap_config.n_neighbors, embeddings.shape[0] - 1),
            min_dist=self.umap_config.min_dist,
            n_components=self.umap_config.n_components,
            metric=self.umap_config.metric,
            random_state=self.umap_config.random_state,
        )
        return np.asarray(reducer.fit_transform(embeddings))

    def _apply_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        perplexity = min(self.tsne_config.perplexity, embeddings.shape[0] - 1)
        if perplexity <= 0:
            logger.warning(
                "Cannot apply t-SNE with perplexity <= 0. Falling back to PCA."
            )
            return self._apply_pca(embeddings)
        tsne = TSNE(
            n_components=self.tsne_config.n_components,
            perplexity=perplexity,
            max_iter=self.tsne_config.max_iter,
            random_state=self.tsne_config.random_state,
            init="pca",
            learning_rate="auto",
        )
        return np.asarray(tsne.fit_transform(embeddings))

    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        n_components = min(
            self.pca_config.n_components, embeddings.shape[1], embeddings.shape[0]
        )
        pca = PCA(n_components=n_components, random_state=self.pca_config.random_state)
        return np.asarray(pca.fit_transform(embeddings))
