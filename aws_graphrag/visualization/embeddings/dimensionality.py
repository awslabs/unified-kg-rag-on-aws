# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from typing import Any

import numpy as np
import umap
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from aws_graphrag.shared import get_logger

from .node2vec import NodeEmbeddings

logger = get_logger(__name__)


class ReducerConfig(BaseModel):
    n_components: int = Field(2, description="Number of dimensions for the output.")
    metric: str = Field(
        "euclidean", description="Metric used for distance calculation."
    )
    random_state: int | None = Field(
        default=None, description="Random seed for reproducible results"
    )


class UMAPConfig(ReducerConfig):
    n_neighbors: int = Field(
        default=15,
        description="Number of neighboring points used in local approximation",
    )
    min_dist: float = Field(
        default=0.1,
        description="Minimum distance between points in the low-dimensional representation",
    )


class TSNEConfig(ReducerConfig):
    perplexity: int = Field(
        default=30, description="Balance between local and global aspects of the data"
    )
    max_iter: int = Field(
        default=1000, description="Maximum number of iterations for optimization"
    )


class DimensionalityReducer:
    def __init__(self, config: dict[str, Any]):
        self.umap_config = UMAPConfig(**config.get("umap", {}))
        self.tsne_config = TSNEConfig(**config.get("tsne", {}))
        self.pca_config = ReducerConfig(**config.get("pca", {}))

    def reduce_dimensions(
        self, embeddings: NodeEmbeddings, method: str = "umap"
    ) -> dict[str, tuple[float, float]]:
        if not embeddings.embeddings:
            logger.warning("No embeddings to reduce.")
            return {}

        logger.info(
            f"Reducing dimensions of {len(embeddings.embeddings)} nodes using {method.upper()}..."
        )

        nodes = embeddings.nodes
        embedding_matrix = np.array([embeddings.embeddings[node] for node in nodes])

        if embedding_matrix.shape[0] < self.umap_config.n_components + 1:
            logger.warning(
                f"Not enough data points ({embedding_matrix.shape[0]}) for dimensionality reduction, falling back to PCA."
            )
            method = "pca"

        if embedding_matrix.shape[0] < 2:
            logger.warning(
                "Cannot perform dimensionality reduction with less than 2 nodes."
            )
            return {nodes[0]: (0.0, 0.0)} if nodes else {}

        try:
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

        except Exception as e:
            logger.error(
                f"Dimensionality reduction failed with method {method.upper()}: {e}",
                exc_info=True,
            )
            return self._generate_random_layout(nodes)

    def _apply_umap(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            reducer = umap.UMAP(
                n_neighbors=min(self.umap_config.n_neighbors, embeddings.shape[0] - 1),
                min_dist=self.umap_config.min_dist,
                n_components=self.umap_config.n_components,
                metric=self.umap_config.metric,
                random_state=self.umap_config.random_state,
            )
            result = reducer.fit_transform(embeddings)
            return np.asarray(result)
        except ImportError:
            logger.warning("UMAP is not installed. Falling back to PCA.")
            return self._apply_pca(embeddings)
        except Exception as e:
            logger.warning(f"UMAP failed: {e}. Falling back to PCA.")
            return self._apply_pca(embeddings)

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

    @staticmethod
    def _generate_random_layout(nodes: list[str]) -> dict[str, tuple[float, float]]:
        logger.warning("Generating random 2D layout as a fallback.")
        return {
            node: (np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for node in nodes
        }
