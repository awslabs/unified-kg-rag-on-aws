# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from aws_graphrag.adapters.aws import BedrockRerankModelFactory
from aws_graphrag.domain.models import Config, FusionMethod, RetrievalResult
from aws_graphrag.domain.retrieval.mixins import MetricsMixin
from aws_graphrag.shared import get_logger
from aws_graphrag.shared.utils import compute_hash

logger = get_logger(__name__)


class HybridScorer(MetricsMixin):
    def __init__(
        self, config: Config, boto_session: Any | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.boto_session = boto_session
        self.fusion_config = config.search.fusion
        self.rerank_factory: BedrockRerankModelFactory | None = None
        self.rerank_model: Any = None
        self._initialize_reranking()

    def _initialize_reranking(self) -> None:
        try:
            rerank_config = self.config.search.reranking
            if not rerank_config or not rerank_config.enabled:
                logger.debug("Reranking is disabled in configuration")
                return

            self.rerank_factory = BedrockRerankModelFactory(
                config=self.config,
                boto_session=self.boto_session,
                region_name=self.config.aws.bedrock.region_name,
            )

            self.rerank_model = self.rerank_factory.get_model(
                model_id=rerank_config.rerank_model_id,
                top_k=rerank_config.top_k,
            )

        except Exception as e:
            logger.warning("Reranking initialization failed: %s", e)
            self.rerank_factory = None
            self.rerank_model = None

    def fuse_and_rerank_results(
        self,
        results_dict: dict[str, list[RetrievalResult]],
        top_k: int,
        retrieval_multiplier: int = 1,
        query: str | None = None,
    ) -> list[RetrievalResult]:
        start_time = time.time()
        method = self.fusion_config.method

        fusion_methods: dict[
            FusionMethod,
            Callable[[dict[str, list[RetrievalResult]]], list[RetrievalResult]],
        ] = {
            FusionMethod.RRF: self._reciprocal_rank_fusion,
            FusionMethod.WEIGHTED: self._weighted_fusion,
        }

        fusion_func = fusion_methods.get(method)
        if not fusion_func:
            raise ValueError(f"Unknown fusion method: '{method}'")

        # RRF uses only rank order and overwrites scores, so per-bucket min-max
        # normalization is wasted work (and would be discarded) there; only the
        # weighted fusion consumes the normalized scores.
        if method == FusionMethod.WEIGHTED:
            fusion_input = {
                name: self._normalize_scores(res_list)
                for name, res_list in results_dict.items()
            }
        else:
            fusion_input = results_dict

        combined_results = fusion_func(fusion_input)

        if self.fusion_config.diversity_lambda < 1.0:
            combined_results = self._apply_diversity_filtering(
                combined_results, top_k=top_k, retrieval_multiplier=retrieval_multiplier
            )

        if self.rerank_model is not None and query is not None:
            combined_results = self._apply_bedrock_reranking(combined_results, query)

        combined_results.sort(key=lambda x: x.score or 0.0, reverse=True)
        final_results = combined_results[:top_k]

        processing_time = time.time() - start_time
        self._record_timing("processing_time", processing_time)
        self._record_metric("initial_fused_count", len(combined_results))
        self._record_metric("final_fused_count", len(final_results))

        logger.info(
            "Fusion completed: %s -> %s results in %.3fs",
            len(combined_results),
            len(final_results),
            processing_time,
        )

        return final_results

    @staticmethod
    def _normalize_scores(results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []

        min_score = float("inf")
        max_score = float("-inf")
        has_score = False

        for r in results:
            if r.score is not None:
                has_score = True
                min_score = min(min_score, r.score)
                max_score = max(max_score, r.score)

        if not has_score:
            for r in results:
                r.score = 0.5
            return results

        score_range = max_score - min_score

        for result in results:
            result.score = (
                0.5 if score_range == 0 else (result.score - min_score) / score_range
            )

        return results

    def _reciprocal_rank_fusion(
        self, result_map: dict[str, list[RetrievalResult]]
    ) -> list[RetrievalResult]:
        k = self.fusion_config.rrf_k
        scores: dict[str, float] = defaultdict(float)
        objects: dict[str, RetrievalResult] = {}

        for results in result_map.values():
            for rank, result in enumerate(results, 1):
                key = self._get_result_key(result)
                scores[key] += 1.0 / (k + rank)
                if key not in objects:
                    objects[key] = result.model_copy()

        for key, score in scores.items():
            if key in objects:
                objects[key].score = score

        return list(objects.values())

    def _weighted_fusion(
        self, result_map: dict[str, list[RetrievalResult]]
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = defaultdict(float)
        objects: dict[str, RetrievalResult] = {}
        weights = self.fusion_config.fusion_weights

        for name, results in result_map.items():
            if name not in weights:
                logger.warning(
                    "Weighted fusion: source bucket '%s' has no configured weight "
                    "in fusion_weights (%s); defaulting to 1.0. Configure a weight "
                    "for this bucket or use RRF fusion to avoid silent equal weighting.",
                    name,
                    sorted(weights),
                )
            weight = weights.get(name, 1.0)
            for result in results:
                key = self._get_result_key(result)
                scores[key] += (result.score or 0.0) * weight
                if key not in objects:
                    objects[key] = result.model_copy()

        for key, score in scores.items():
            if key in objects:
                objects[key].score = score

        return list(objects.values())

    @staticmethod
    def _get_result_key(result: RetrievalResult) -> str:
        content_hash = compute_hash(result.content, length=16)
        return f"{result.source or 'unknown'}-{content_hash}"

    def _apply_diversity_filtering(
        self,
        results: list[RetrievalResult],
        top_k: int,
        retrieval_multiplier: int = 1,
    ) -> list[RetrievalResult]:
        lambda_val = self.fusion_config.diversity_lambda
        # MMR penalty term (1 - lambda) is only active when lambda < 1.0; at 1.0
        # the result degenerates to pure relevance ordering, so skip the work.
        if not results or lambda_val >= 1.0 or len(results) < 2:
            return results

        target_count = top_k * retrieval_multiplier
        results.sort(key=lambda x: x.score or 0.0, reverse=True)

        word_sets: dict[int, set[str]] = {}
        for i, result in enumerate(results):
            if result.content:
                word_sets[i] = set(result.content.lower().split())
            else:
                word_sets[i] = set()

        selected_indices: list[int] = [0]
        remaining_indices = set(range(1, len(results)))

        def calculate_mmr(candidate_idx: int) -> float:
            candidate = results[candidate_idx]
            relevance = candidate.score or 0.0

            candidate_words = word_sets[candidate_idx]
            max_similarity = max(
                (
                    self._jaccard(candidate_words, word_sets[selected_idx])
                    for selected_idx in selected_indices
                ),
                default=0.0,
            )

            return lambda_val * relevance - (1 - lambda_val) * max_similarity

        while remaining_indices and len(selected_indices) < target_count:
            best_idx = max(remaining_indices, key=calculate_mmr)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        selected = [results[i] for i in selected_indices]

        filtered_count = len(results) - len(selected)
        if filtered_count > 0:
            logger.debug(
                "Diversity filtering removed %s similar results", filtered_count
            )

        self._record_metric("diversity_filtered_count", filtered_count)
        return selected

    def _apply_bedrock_reranking(
        self, results: list[RetrievalResult], query: str
    ) -> list[RetrievalResult]:
        if not self.rerank_model or not results:
            return results

        try:
            start_time = time.time()

            result_map = {self._get_result_key(res): res for res in results}
            documents = []
            for key, result in result_map.items():
                doc = Document(
                    page_content=result.content or "",
                    metadata={
                        "key": key,
                        "source": result.source or "",
                        "original_score": result.score or 0.0,
                    },
                )
                documents.append(doc)

            original_top_n = self.rerank_model.top_n
            adjusted_top_n = (
                min(len(documents), original_top_n)
                if original_top_n
                else len(documents)
            )

            if adjusted_top_n != original_top_n:
                logger.debug(
                    "Adjusting rerank 'top_n' from %s to %s to match document count (%s)",
                    original_top_n,
                    adjusted_top_n,
                    len(documents),
                )
                self.rerank_model.top_n = adjusted_top_n

            try:
                reranked_docs = self.rerank_model.compress_documents(
                    documents=documents, query=query
                )
            finally:
                if adjusted_top_n != original_top_n:
                    self.rerank_model.top_n = original_top_n
            reranked_results = []
            for i, doc in enumerate(reranked_docs):
                key_value = doc.metadata.get("key")
                if key_value is not None and isinstance(key_value, str):
                    original_result = result_map.get(key_value)

                    if original_result:
                        reranked_result = original_result.model_copy()
                        new_score = doc.metadata.get(
                            "relevance_score", 1.0 - (i * 0.01)
                        )
                        reranked_result.score = new_score
                        reranked_result.metadata = (
                            dict(reranked_result.metadata)
                            if reranked_result.metadata
                            else {}
                        )
                        reranked_result.metadata.update(
                            {
                                "reranked": True,
                                "rerank_position": i + 1,
                                "original_score": doc.metadata.get(
                                    "original_score", 0.0
                                ),
                            }
                        )
                        reranked_results.append(reranked_result)

            processing_time = time.time() - start_time
            self._record_timing("processing_time", processing_time)
            self._record_metric("reranked_count", len(reranked_results))

            logger.info(
                "Reranking completed: %s -> %s results in %.3fs",
                len(results),
                len(reranked_results),
                processing_time,
            )

            return reranked_results

        except Exception as e:
            logger.error("Reranking failed: %s", e)
            return results

    @staticmethod
    def _jaccard(words1: set[str], words2: set[str]) -> float:
        """Jaccard similarity of two word sets (shared by MMR diversity)."""
        union = len(words1 | words2)
        return len(words1 & words2) / union if union > 0 else 0.0
