# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import time

import boto3
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from aws_graphrag.adapters.aws import BedrockEmbeddingModelFactory
from aws_graphrag.domain.models import Config
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


class NodeEmbeddings(BaseModel):
    nodes: list[str] = Field(
        description="List of node identifiers in the graph for which embeddings were generated"
    )
    embeddings: dict[str, np.ndarray] = Field(
        description="Mapping of node identifiers to their corresponding high-dimensional embedding vectors"
    )

    class Config:
        arbitrary_types_allowed = True


class BedrockNodeEmbedder:
    def __init__(
        self, config: Config, boto_session: boto3.Session | None = None
    ) -> None:
        self.config = config
        self.viz_config = self.config.graph.visualization
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )

        embedding_factory = BedrockEmbeddingModelFactory(
            config=self.config,
            boto_session=self.boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )

        embedding_model_id = self.viz_config.embeddings.get(
            "bedrock_model_id", self.config.indexing.opensearch.embedding_model_id
        )

        model_info = embedding_factory.get_model_info(embedding_model_id)
        if not model_info:
            raise ValueError(
                f"Unsupported Bedrock model for visualization: '{embedding_model_id}'"
            )

        if isinstance(model_info.dimensions, list):
            self.dimensions = model_info.dimensions[-1]
        else:
            self.dimensions = model_info.dimensions or 1024

        self.embedding_model = embedding_factory.get_model(
            model_id=embedding_model_id, dimensions=self.dimensions
        )
        logger.info(
            "Initialized BedrockNodeEmbedder with model '%s' and dimension %s",
            embedding_model_id.value,
            self.dimensions,
        )

    def generate_embeddings(self, graph: nx.Graph) -> NodeEmbeddings:
        if not graph or graph.number_of_nodes() == 0:
            logger.warning("Cannot generate embeddings for an empty or invalid graph.")
            return NodeEmbeddings(nodes=[], embeddings={})

        logger.info(
            "Generating Bedrock embeddings for %s nodes...", graph.number_of_nodes()
        )
        start_time = time.time()

        try:
            nodes = [str(n) for n in graph.nodes()]
            texts_to_embed = []
            for node_id in nodes:
                node_data = graph.nodes[node_id]
                name = node_data.get("name", node_id)
                description = node_data.get("description", "")
                text = f"{name}: {description}".strip()
                texts_to_embed.append(text)

            if not texts_to_embed:
                logger.warning("No text content found in nodes to generate embeddings.")
                return self._generate_random_embeddings(graph)

            embeddings_list = self.embedding_model.embed_documents(texts_to_embed)

            embeddings_dict = {
                node_id: np.array(embedding)
                for node_id, embedding in zip(nodes, embeddings_list, strict=True)
                if embedding is not None
            }

            if len(embeddings_dict) != len(nodes):
                logger.warning(
                    "Failed to generate embeddings for %s nodes.",
                    len(nodes) - len(embeddings_dict),
                )

            logger.info(
                f"Generated embeddings for {len(embeddings_dict)} nodes in {time.time() - start_time:.2f}s."
            )
            return NodeEmbeddings(
                nodes=list(embeddings_dict.keys()), embeddings=embeddings_dict
            )

        except Exception as e:
            logger.error(
                "Failed to generate Bedrock embeddings: %s. Falling back to random embeddings.",
                e,
                exc_info=True,
            )
            return self._generate_random_embeddings(graph)

    def _generate_random_embeddings(self, graph: nx.Graph) -> NodeEmbeddings:
        logger.warning("Using random embeddings as a fallback.")
        nodes = [str(node) for node in graph.nodes()]
        embeddings = {node: np.random.normal(0, 1, self.dimensions) for node in nodes}
        return NodeEmbeddings(nodes=nodes, embeddings=embeddings)
