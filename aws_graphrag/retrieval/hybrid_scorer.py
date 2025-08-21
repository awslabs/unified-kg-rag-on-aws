import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from aws_graphrag.aws import BedrockRerankModelFactory
from aws_graphrag.core import get_logger
from aws_graphrag.models import Config, FusionMethod, RetrievalResult
from aws_graphrag.utils import compute_hash

from .mixins import MetricsMixin

logger = get_logger(__name__)


class HybridScorer(MetricsMixin):
    def __init__(self, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.fusion_config = config.search.fusion
        self.rerank_factory: BedrockRerankModelFactory | None = None
        self.rerank_model = None
        self._initialize_reranking()

    def _initialize_reranking(self) -> None:
        try:
            rerank_config = self.config.search.reranking
            if not rerank_config or not rerank_config.enabled:
                logger.debug("Reranking is disabled in configuration")
                return

            self.rerank_factory = BedrockRerankModelFactory(
                config=self.config,
                region_name=self.config.aws.bedrock.region_name,
            )

            self.rerank_model = self.rerank_factory.get_model(
                model_id=rerank_config.rerank_model_id,
                top_k=rerank_config.top_k,
            )

        except Exception as e:
            logger.warning(f"Reranking initialization failed: {e}")
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

        normalized_map = {
            name: self._normalize_scores(res_list)
            for name, res_list in results_dict.items()
        }

        fusion_methods: dict[FusionMethod, Callable] = {
            FusionMethod.RRF: self._reciprocal_rank_fusion,
            FusionMethod.WEIGHTED: self._weighted_fusion,
        }

        fusion_func = fusion_methods.get(method)
        if not fusion_func:
            raise ValueError(f"Unknown fusion method: '{method}'")

        combined_results = fusion_func(normalized_map)

        if self.fusion_config.diversity_lambda > 0.0:
            combined_results = self._apply_diversity_filtering(
                combined_results, top_k=top_k, retrieval_multiplier=retrieval_multiplier
            )

        if self.rerank_model and query:
            combined_results = self._apply_bedrock_reranking(combined_results, query)

        combined_results.sort(key=lambda x: x.score or 0.0, reverse=True)
        final_results = combined_results[:top_k]

        processing_time = time.time() - start_time
        self._record_timing("processing_time", processing_time)
        self._record_metric("initial_fused_count", len(combined_results))
        self._record_metric("final_fused_count", len(final_results))

        logger.info(
            f"Fusion completed: {len(combined_results)} -> {len(final_results)} "
            f"results in {processing_time:.3f}s"
        )

        return final_results

    @staticmethod
    def _normalize_scores(results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []

        scores = [r.score for r in results if r.score is not None]
        if not scores:
            for r in results:
                r.score = 0.5
            return results

        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score

        for result in results:
            if result.score is not None:
                result.score = (
                    0.5
                    if score_range == 0
                    else (result.score - min_score) / score_range
                )
            else:
                result.score = 0.5

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
                    objects[key] = result.model_copy(deep=True)

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
            weight = weights.get(name, 1.0)
            for result in results:
                key = self._get_result_key(result)
                scores[key] += (result.score or 0.0) * weight
                if key not in objects:
                    objects[key] = result.model_copy(deep=True)

        for key, score in scores.items():
            if key in objects:
                objects[key].score = score

        return list(objects.values())

    @staticmethod
    def _get_result_key(result: RetrievalResult) -> str:
        content_hash = compute_hash(result.content, algorithm="md5", length=16)
        return f"{result.source or 'unknown'}-{content_hash}"

    def _apply_diversity_filtering(
        self,
        results: list[RetrievalResult],
        top_k: int,
        retrieval_multiplier: int = 1,
    ) -> list[RetrievalResult]:
        lambda_val = self.fusion_config.diversity_lambda
        if not results or lambda_val <= 0 or len(results) < 2:
            return results

        target_count = top_k * retrieval_multiplier
        results.sort(key=lambda x: x.score or 0.0, reverse=True)
        selected: list[RetrievalResult] = [results[0]]
        remaining = results[1:]

        def calculate_mmr(candidate: RetrievalResult) -> float:
            relevance = candidate.score or 0.0
            max_similarity = max(
                self._calculate_jaccard_similarity(candidate, selected_result)
                for selected_result in selected
            )
            return lambda_val * relevance - (1 - lambda_val) * max_similarity

        while remaining and len(selected) < target_count:
            best_candidate = max(remaining, key=calculate_mmr)
            selected.append(best_candidate)
            remaining.remove(best_candidate)

        filtered_count = len(results) - len(selected)
        if filtered_count > 0:
            logger.debug(
                f"Diversity filtering removed {filtered_count} similar results"
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
                    f"Adjusting rerank 'top_n' from {original_top_n} to {adjusted_top_n} "
                    f"to match document count ({len(documents)})"
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
                key = doc.metadata.get("key")
                if key is not None:
                    original_result = result_map.get(key)

                    if original_result:
                        reranked_result = original_result.model_copy(deep=True)
                        new_score = doc.metadata.get(
                            "relevance_score", 1.0 - (i * 0.01)
                        )
                        reranked_result.score = new_score
                        reranked_result.metadata = reranked_result.metadata or {}
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
                f"Reranking completed: {len(results)} -> {len(reranked_results)} "
                f"results in {processing_time:.3f}s"
            )

            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    @staticmethod
    def _calculate_jaccard_similarity(
        r1: RetrievalResult, r2: RetrievalResult
    ) -> float:
        if not r1.content or not r2.content:
            return 0.0

        words1 = set(r1.content.lower().split())
        words2 = set(r2.content.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def score_and_sort_results(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        if not results:
            return []

        normalized = self._normalize_scores(results)
        normalized.sort(key=lambda x: x.score or 0.0, reverse=True)
        self._record_metric("last_scored_results_count", len(normalized))

        return normalized
