# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from unified_kg_rag.adapters.aws import BedrockRerankModelFactory
from unified_kg_rag.domain.models import Config, FusionMethod, RetrievalResult
from unified_kg_rag.domain.retrieval.mixins import MetricsMixin
from unified_kg_rag.ports.model_factory import RerankFactoryPort
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import compute_hash

logger = get_logger(__name__)


class HybridScorer(MetricsMixin):
    def __init__(
        self,
        config: Config,
        boto_session: Any | None = None,
        rerank_factory: RerankFactoryPort | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.boto_session = boto_session
        self.fusion_config = config.search.fusion
        # Injected rerank provider (port); defaults to Bedrock when reranking is
        # enabled and none is supplied.
        self.rerank_factory: RerankFactoryPort | None = rerank_factory
        self.rerank_model: Any = None
        self._initialize_reranking()

    def _initialize_reranking(self) -> None:
        try:
            rerank_config = self.config.search.reranking
            if not rerank_config or not rerank_config.enabled:
                logger.debug("Reranking is disabled in configuration")
                return

            if self.rerank_factory is None:
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
        per_type_quota: dict[str, int] | None = None,
        rerank_only_types: set[str] | None = None,
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
            # Diversity filtering cuts to top_k*multiplier — which, with top_k=10,
            # pre-starves the fused set to 10 BEFORE rerank/quota, so a per-type quota
            # (sum ~50) has nothing left to protect and a KG item at native rank ~10 is
            # already gone. When a quota is set, keep at least the quota budget so
            # diversity de-dups WITHIN that budget instead of collapsing it.
            div_top_k = top_k
            if per_type_quota:
                div_top_k = max(top_k, sum(per_type_quota.values()))
            combined_results = self._apply_diversity_filtering(
                combined_results,
                top_k=div_top_k,
                retrieval_multiplier=retrieval_multiplier,
            )

        if self.rerank_model is not None and query is not None:
            if rerank_only_types is None:
                combined_results = self._apply_bedrock_reranking(
                    combined_results, query
                )
            else:
                # Rerank ONLY the given types (e.g. {"text"}), leaving KG streams
                # (entity/relationship/community) on their native degree/cosine order.
                # Content-vs-query reranking buries multi-hop BRIDGE entities/relations
                # whose description lacks the question's surface terms, and upstream
                # LightRAG reranks chunks only.
                to_rerank = [
                    r for r in combined_results if r.retriever_type in rerank_only_types
                ]
                kept = [
                    r
                    for r in combined_results
                    if r.retriever_type not in rerank_only_types
                ]
                if to_rerank:
                    to_rerank = self._apply_bedrock_reranking(to_rerank, query)
                combined_results = kept + to_rerank

        combined_results.sort(key=lambda x: x.score or 0.0, reverse=True)
        # A single cross-type [:top_k] cut lets one type (usually entities, with many
        # near-tie RRF scores) crowd out text chunks — which is where LightRAG/GraphRAG
        # multi-hop answers actually live. When per_type_quota is given, guarantee each
        # retriever_type its own slots (mirrors LightRAG's separate 40-entity /
        # 20-chunk streams) before back-filling the remainder by score. No quota ->
        # the original flat behavior (other strategies unaffected).
        if per_type_quota:
            final_results = self._select_with_type_quota(
                combined_results, top_k, per_type_quota
            )
        else:
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
    def _select_with_type_quota(
        ranked: list[RetrievalResult],
        top_k: int,
        per_type_quota: dict[str, int],
    ) -> list[RetrievalResult]:
        """Pick up to quota[type] of each retriever_type (in score order), then
        back-fill any remaining top_k budget from types that carry NO quota.

        Guarantees text chunks (and each other type) their reserved slots instead of
        losing every slot to a single dominant type under flat RRF ties.

        A quota is a CAP as well as a floor. The back-fill
        used to re-offer over-quota leftovers to any type, so `text`'s 20-slot
        chunk quota silently became 40 (the two chunk streams' full width) — while
        upstream LightRAG truncates the MERGED chunk pool to `chunk_top_k` (20)
        before it ever reaches the context (`utils.process_chunks_unified` steps
        1+3, called from `operate._build_context_str`). The surplus chunks consumed
        window that upstream spends on the KG sections. Types with no quota entry still
        back-fill, so non-mix strategies (which pass no quota at all) are unchanged.

        ``top_k`` bounds only that back-fill of unquotaed types; the quotaed types
        are bounded by their own entries. A quota is therefore an explicit, larger
        request than ``top_k`` — the caller's ``top_k`` cannot shrink slots the
        strategy deliberately reserved — and the effective ceiling is logged when
        the two disagree so an unexpectedly wide result set is traceable.
        """
        selected: list[RetrievalResult] = []
        used_per_type: dict[str, int] = {}
        leftovers: list[RetrievalResult] = []
        for r in ranked:  # already score-sorted desc
            rtype = r.retriever_type
            quota = per_type_quota.get(rtype, 0)
            if used_per_type.get(rtype, 0) < quota:
                selected.append(r)
                used_per_type[rtype] = used_per_type.get(rtype, 0) + 1
            elif rtype not in per_type_quota:
                leftovers.append(r)
        # back-fill remaining budget by score (leftovers are already in score order)
        quota_total = sum(per_type_quota.values())
        budget = max(top_k, quota_total)
        if quota_total > top_k:
            logger.debug(
                "Per-type quota (%s slots across %s types) exceeds top_k=%s; "
                "the quota governs the result width",
                quota_total,
                len(per_type_quota),
                top_k,
            )
        for r in leftovers:
            if len(selected) >= budget:
                break
            selected.append(r)
        selected.sort(key=lambda x: x.score or 0.0, reverse=True)
        return selected

    @staticmethod
    def _normalize_scores(results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []

        # Operate on COPIES: callers (e.g. drift_search) keep the original
        # result objects in SearchResult/metrics and may reuse them after
        # fusion. Mutating `result.score` in place would clobber those originals
        # with normalized values. RRF/weighted fusion downstream re-copies too,
        # but the normalization itself must not leak into the caller's list.
        copies = [r.model_copy() for r in results]

        min_score = float("inf")
        max_score = float("-inf")
        has_score = False

        for r in copies:
            if r.score is not None:
                has_score = True
                min_score = min(min_score, r.score)
                max_score = max(max_score, r.score)

        if not has_score:
            for r in copies:
                r.score = 0.5
            return copies

        score_range = max_score - min_score

        for result in copies:
            result.score = (
                0.5 if score_range == 0 else (result.score - min_score) / score_range
            )

        return copies

    def _reciprocal_rank_fusion(
        self, result_map: dict[str, list[RetrievalResult]]
    ) -> list[RetrievalResult]:
        k = self.fusion_config.rrf_k
        weights = self.fusion_config.fusion_weights
        scores: dict[str, float] = defaultdict(float)
        objects: dict[str, RetrievalResult] = {}

        # Plain RRF overwrites each item's score with a flat
        # weight/(k+rank) (~0.0164 at rank1 vs 0.0143 at rank10) — so WITHIN a stream
        # the gold item is indistinguishable from noise, and downstream token_manager
        # (priority = score*multiplier) fills quota slots with the wrong item (the
        # "Medavoy retrieved but dropped from context" bug). Upstream LightRAG keeps
        # the native order (entities by cosine, relations by degree). We preserve that
        # by BLENDING a per-stream min-max-normalized native score into the RRF score:
        # cross-stream fusion still comes from RRF rank; within-stream discrimination
        # is restored by the native component (scaled small so it breaks ties/orders
        # within a rank neighborhood without dominating the cross-stream RRF signal).
        native_norm: dict[str, float] = {}
        for results in result_map.values():
            for r in self._normalize_scores(results):
                native_norm[self._get_result_key(r)] = r.score or 0.0

        for name, results in result_map.items():
            weight = weights.get(name, 1.0)
            for rank, result in enumerate(results, 1):
                key = self._get_result_key(result)
                scores[key] += weight / (k + rank)
                if key not in objects:
                    objects[key] = result.model_copy()

        # The native component must order items WITHIN an RRF-rank tie without
        # flipping genuine cross-stream rank gaps, so scale it to the granularity of
        # one RRF rank step. Adjacent-rank steps are 1/(k+r) - 1/(k+r+1), largest at
        # r=1; derive the scale from that identity rather than fixing a constant, so
        # it tracks a reconfigured `rrf_k` (at the default k=60 this is ~2.7e-4).
        # Native scores are min-max normalized to [0, 1], so the blended component
        # can never exceed a single rank step.
        native_scale = 1.0 / (k + 1) - 1.0 / (k + 2)
        for key, score in scores.items():
            if key in objects:
                objects[key].score = score + native_scale * native_norm.get(key, 0.0)

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
        # Sort a copy: the caller still holds `results` (the post-fusion list),
        # and an in-place sort here would reorder it as a side effect.
        results = sorted(results, key=lambda x: x.score or 0.0, reverse=True)

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

            if not reranked_results:
                # The rerank model returned nothing usable (empty output, or all
                # keys failed the guard above). Returning [] here would silently
                # drop the entire candidate set and produce an empty answer, so
                # degrade to the original (already fused/scored) results instead.
                logger.warning(
                    "Reranking produced no usable results; "
                    "falling back to the %s pre-rerank results",
                    len(results),
                )
                return results

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
