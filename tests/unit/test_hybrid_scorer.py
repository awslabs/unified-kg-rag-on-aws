# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HybridScorer — the shared fusion/rerank/diversity core used by
every GraphRAG and LightRAG search strategy (AWS-free)."""

from __future__ import annotations

import copy
import logging

import pytest

from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.domain.models import Config, RetrievalResult
from unified_kg_rag.domain.models import config as config_module

pytestmark = pytest.mark.unit


def _scorer(config: Config | None = None) -> HybridScorer:
    config = config or Config()
    # Disable reranking so no Bedrock client is constructed.
    config.providers.rerank.kind = "none"
    return HybridScorer(config)


def _r(content: str, score: float | None, source: str) -> RetrievalResult:
    return RetrievalResult(
        content=content, score=score, source=source, retriever_type="test"
    )


class TestReciprocalRankFusion:
    def test_higher_rank_contributes_more(self) -> None:
        scorer = _scorer()
        k = scorer.fusion_config.rrf_k
        result_map = {"src": [_r("first", 0.9, "1"), _r("second", 0.8, "2")]}
        fused = scorer._reciprocal_rank_fusion(result_map)
        by_content = {r.content: r.score for r in fused}
        assert by_content["first"] == pytest.approx(1.0 / (k + 1))
        assert by_content["second"] == pytest.approx(1.0 / (k + 2))

    def test_one_bucket_keeps_its_own_order(self) -> None:
        # Its bucket's order is the only thing RRF reads: a native score blended
        # in (scaled to the rank-1 step) outweighed the smaller steps deeper in
        # a list and inverted positions of a DRIFT bucket that mixes
        # graph-distance and fused index scores.
        scorer = _scorer()
        items = [
            _r(f"c{p}", 0.1 if p % 2 == 0 else (1.0 if p > 30 else 0.0), f"s{p}")
            for p in range(60)
        ]
        fused = scorer._reciprocal_rank_fusion({"results": items})
        fused.sort(key=lambda r: r.score or 0.0, reverse=True)
        assert [r.source for r in fused] == [f"s{p}" for p in range(60)]

    def test_cross_source_contributions_accumulate(self) -> None:
        scorer = _scorer()
        k = scorer.fusion_config.rrf_k
        # Same content+source in two lists, both at rank 1.
        shared = _r("shared", 0.9, "x")
        result_map = {
            "a": [shared.model_copy()],
            "b": [shared.model_copy()],
        }
        fused = scorer._reciprocal_rank_fusion(result_map)
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(2.0 / (k + 1))

    def test_fused_object_is_a_copy(self) -> None:
        scorer = _scorer()
        original = _r("c", 0.9, "x")
        fused = scorer._reciprocal_rank_fusion({"a": [original]})
        assert fused[0] is not original

    def test_default_weights_reproduce_plain_rrf(self) -> None:
        # No fusion_weights by default, and an unconfigured bucket weighs 1.0.
        scorer = _scorer()
        assert scorer.fusion_config.fusion_weights == {}
        k = scorer.fusion_config.rrf_k
        fused = scorer._reciprocal_rank_fusion(
            {"unconfigured_bucket": [_r("only", 0.9, "1")]}
        )
        assert fused[0].score == pytest.approx(1.0 / (k + 1))

    def test_per_bucket_weight_scales_rrf_contribution(self) -> None:
        # A configured per-bucket weight applies under RRF too.
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.fusion_weights = {"boosted": 3.0}  # "plain" defaults 1.0
        scorer = HybridScorer(config)
        k = scorer.fusion_config.rrf_k
        result_map = {
            "boosted": [_r("b", 0.5, "1")],
            "plain": [_r("p", 0.9, "2")],
        }
        fused = scorer._reciprocal_rank_fusion(result_map)
        by_content = {r.content: r.score for r in fused}
        assert by_content["b"] == pytest.approx(3.0 / (k + 1))
        assert by_content["p"] == pytest.approx(1.0 / (k + 1))


class TestRenamedFusionBuckets:
    @pytest.fixture(autouse=True)
    def _fresh_warnings(self) -> None:
        config_module._warn_renamed_bucket.cache_clear()
        config_module._warn_retired_bucket.cache_clear()

    def test_a_weight_on_a_renamed_bucket_applies_to_its_new_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            config = Config.model_validate(
                {"search": {"fusion": {"fusion_weights": {"opensearch_all": 2.0}}}}
            )
        assert config.search.fusion.fusion_weights == {"vector_all": 2.0}
        assert "'opensearch_all' is renamed" in caplog.text
        assert "'vector_all'" in caplog.text

    @pytest.mark.parametrize("weight", [-1.0, -0.01])
    def test_a_negative_weight_is_refused(self, weight: float) -> None:
        # It would subtract a bucket's ranks from the fused score.
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            Config.model_validate(
                {"search": {"fusion": {"fusion_weights": {"text_units": weight}}}}
            )
        # 0 turns the bucket off.
        Config.model_validate(
            {"search": {"fusion": {"fusion_weights": {"text_units": 0.0}}}}
        )

    def test_the_new_name_wins_over_the_old_one(self) -> None:
        weights = {"opensearch_all": 2.0, "vector_all": 0.5}
        config = Config.model_validate(
            {"search": {"fusion": {"fusion_weights": weights}}}
        )
        assert config.search.fusion.fusion_weights == {"vector_all": 0.5}

    @pytest.mark.parametrize(
        "retired",
        [
            "opensearch_community_reports",
            # The buckets 0.2.3's global search emitted.
            "opensearch_candidate_community_reports",
            "opensearch_expanded_community_reports",
        ],
    )
    def test_a_weight_on_a_retired_bucket_is_dropped_with_a_warning(
        self, caplog: pytest.LogCaptureFixture, retired: str
    ) -> None:
        weights = {retired: 3.0, "text_units": 2.0}
        with caplog.at_level(logging.WARNING):
            config = Config.model_validate(
                {"search": {"fusion": {"fusion_weights": weights}}}
            )
        assert config.search.fusion.fusion_weights == {"text_units": 2.0}
        assert f"'{retired}'" in caplog.text


class TestResultKey:
    def test_same_content_and_source_collapse(self) -> None:
        scorer = _scorer()
        assert scorer._get_result_key(_r("same", 0.1, "s")) == scorer._get_result_key(
            _r("same", 0.9, "s")
        )

    def test_different_source_distinct(self) -> None:
        scorer = _scorer()
        assert scorer._get_result_key(_r("same", 0.1, "s1")) != scorer._get_result_key(
            _r("same", 0.1, "s2")
        )

    def test_same_id_different_artifact_kind_stays_distinct(self) -> None:
        # The id is only unique within an artifact kind, so the key is qualified
        # by `retriever_type` — an entity and a text unit that share an id are
        # two artifacts, not one.
        scorer = _scorer()
        entity = RetrievalResult(
            content="x", score=1.0, source="shared-id", retriever_type="entity"
        )
        text_unit = RetrievalResult(
            content="x", score=1.0, source="shared-id", retriever_type="text"
        )
        assert scorer._get_result_key(entity) != scorer._get_result_key(text_unit)

    @pytest.mark.parametrize("missing_id", [None, "", "None"])
    def test_results_without_an_id_fall_back_to_content_hash(
        self, missing_id: str | None
    ) -> None:
        # Without an id there is nothing stable to key on, so distinct content
        # must stay distinct rather than collapsing into one "unknown" bucket.
        # `NeptuneRetriever` stringifies an absent node id to "None", so that
        # spelling counts as missing too.
        scorer = _scorer()
        first = RetrievalResult(
            content="one", score=1.0, source=missing_id, retriever_type="entity"
        )
        second = RetrievalResult(
            content="two", score=1.0, source=missing_id, retriever_type="entity"
        )
        assert scorer._get_result_key(first) != scorer._get_result_key(second)


class TestCrossStoreRankReinforcement:
    """RRF must reward graph/vector AGREEMENT on the same artifact.

    The two stores render one entity differently, so keying fusion on a hash of
    the rendered content split it into two items and the reinforcement RRF
    exists to provide never happened between the graph and vector streams.
    """

    @staticmethod
    def _graph_hit(entity_id: str, name: str, score: float) -> RetrievalResult:
        # Rendering shape of NeptuneRetriever._build_content.
        return RetrievalResult(
            content=f"Entity: {name}\nDescription: {name} description\nPath: {name} -> Other",
            score=score,
            source=entity_id,
            retriever_type="entity",
        )

    @staticmethod
    def _vector_hit(entity_id: str, name: str, score: float) -> RetrievalResult:
        # Rendering shape of OpenSearchRetriever._extract_content.
        return RetrievalResult(
            content=f"Description: {name} description\n\nTitle: {name}",
            score=score,
            source=entity_id,
            retriever_type="entity",
        )

    def test_entity_found_by_both_stores_outscores_entity_found_by_one(self) -> None:
        scorer = _scorer()
        fused = scorer._reciprocal_rank_fusion(
            {
                # The graph stream ranks the shared entity first, while the
                # vector stream ranks it BELOW an entity only that store
                # returned — so cross-store agreement is the only signal that
                # can lift the shared entity to the top.
                "graph": [self._graph_hit("e-shared", "Shared", 0.9)],
                "vector": [
                    self._vector_hit("e-vector-only", "VectorOnly", 0.9),
                    self._vector_hit("e-shared", "Shared", 0.8),
                ],
            }
        )

        # Two artifacts went in, two come out: the graph and vector renderings
        # of "e-shared" are one entity, not two results.
        assert len(fused) == 2
        by_source = {r.source: r.score or 0.0 for r in fused}
        assert by_source["e-shared"] > by_source["e-vector-only"]

    def test_distinct_entities_are_not_merged(self) -> None:
        scorer = _scorer()
        fused = scorer._reciprocal_rank_fusion(
            {
                "graph": [self._graph_hit("e-1", "One", 0.9)],
                "vector": [self._vector_hit("e-2", "Two", 0.9)],
            }
        )
        assert {r.source for r in fused} == {"e-1", "e-2"}


class TestOneContributionPerBucket:
    """An artifact counts once per fusion bucket, however often it appears there.

    A retriever leg is one query over one index, so an id cannot repeat within
    it and per-leg dedupe is implicit. DRIFT is different: it concatenates every
    iteration into ONE bucket and dedupes by rendered content, so the same
    entity reached over two different paths renders two different `Path:` lines,
    survives `_filter_unique_results` twice, and reaches fusion twice in the
    same bucket. Under the id key that is one identity, and summing both
    occurrences let repetition outrank rank. Reinforcement across buckets is
    the point of RRF and must stay; repetition within a bucket is not a signal.
    """

    @staticmethod
    def _graph_hit(
        entity_id: str, name: str, score: float, path: list[str]
    ) -> RetrievalResult:
        # Mirrors NeptuneRetriever._create_retrieval_result: the real renderer
        # for the content, the node id in `source`, and the entity section type.
        node_data = {"id": entity_id, "name": name, "description": f"{name} desc"}
        path_data = [{"name": [p]} for p in path]
        return RetrievalResult(
            content=NeptuneRetriever._build_content(node_data, path_data),
            score=score,
            source=entity_id,
            retriever_type="entity",
        )

    def _drift_bucket(self) -> list[RetrievalResult]:
        """One bucket holding one artifact twice, rendered with two paths.

        DRIFT fuses the evidence of several local searches in one bucket, so
        an artifact can arrive more than once: rank 1 `best`, ranks 2 and 3
        `repeat`.
        """
        best = self._graph_hit("e-best", "Best", 1.0, ["Seed", "Best"])
        repeat_a = self._graph_hit("e-repeat", "Repeat", 0.5, ["Seed", "Repeat"])
        repeat_b = self._graph_hit("e-repeat", "Repeat", 0.5, ["Other", "Repeat"])
        bucket = [best, repeat_a, repeat_b]
        assert bucket[1].content != bucket[2].content
        return bucket

    def test_drift_repeat_does_not_outrank_best_under_rrf(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        scorer = HybridScorer(config)
        k = config.search.fusion.rrf_k

        out = scorer.fuse_and_rerank_results(
            {"results": self._drift_bucket()}, top_k=10
        )

        assert [r.source for r in out] == ["e-best", "e-repeat"]
        by_source = {r.source: r.score or 0.0 for r in out}
        assert by_source["e-best"] == pytest.approx(1.0 / (k + 1))
        # `repeat`: its best rank in the bucket, once — not 1/(k+2) + 1/(k+3).
        assert by_source["e-repeat"] == pytest.approx(1.0 / (k + 2))

    def test_same_artifact_in_two_buckets_still_reinforces(self) -> None:
        # The dedupe is per bucket: the cross-store case that keying on the id
        # exists for keeps accumulating one contribution from each bucket.
        scorer = _scorer()
        k = scorer.fusion_config.rrf_k
        hit = self._graph_hit("e-shared", "Shared", 0.9, ["Seed", "Shared"])
        fused = scorer._reciprocal_rank_fusion(
            {"graph": [hit.model_copy()], "vector": [hit.model_copy()]}
        )
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(2.0 / (k + 1))


class TestFuseAndRerank:
    def test_sorted_desc_and_truncated_to_top_k(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 1.0  # skip diversity filtering
        scorer = HybridScorer(config)
        result_map = {"a": [_r("x", 0.5, "1"), _r("y", 0.4, "2"), _r("z", 0.3, "3")]}
        out = scorer.fuse_and_rerank_results(result_map, top_k=2)
        assert len(out) == 2
        assert out[0].score >= out[1].score

    def test_single_result_survives_fusion(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 1.0
        scorer = HybridScorer(config)
        out = scorer.fuse_and_rerank_results({"a": [_r("x", 1.0, "1")]}, top_k=5)
        assert [r.source for r in out] == ["1"]

    def test_fusion_leaves_scorer_state_untouched(self) -> None:
        # One scorer serves every concurrent query of a strategy, so a query
        # must not write per-query numbers onto it.
        config = Config()
        config.providers.rerank.kind = "none"
        scorer = HybridScorer(config)
        factory = scorer.rerank_factory
        before = copy.deepcopy(
            {k: v for k, v in vars(scorer).items() if k != "rerank_factory"}
        )
        scorer.fuse_and_rerank_results(
            {"a": [_r("x", 1.0, "1"), _r("y", 0.5, "2")]}, top_k=1
        )
        assert scorer.rerank_factory is factory
        assert {k: v for k, v in vars(scorer).items() if k != "rerank_factory"} == (
            before
        )


class TestTypeQuota:
    """A per-type quota is a cap as well as a floor.

    Upstream LightRAG truncates the MERGED chunk pool to `chunk_top_k` before it
    reaches the context, so a `text` quota of 20 must not grow to 40 just because
    the two chunk streams together offered 40 candidates.
    """

    @staticmethod
    def _typed(rtype: str, n: int, base: float) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                content=f"{rtype}-{i}",
                score=base - i * 0.001,
                source="s",
                retriever_type=rtype,
            )
            for i in range(n)
        ]

    def test_over_quota_leftovers_do_not_backfill_their_own_type(self) -> None:
        ranked = self._typed("text", 40, 0.9) + self._typed("entity", 60, 0.5)
        out = HybridScorer._select_with_type_quota(
            ranked, top_k=10, per_type_quota={"text": 20, "entity": 40}
        )
        counts: dict[str, int] = {}
        for r in out:
            counts[r.retriever_type] = counts.get(r.retriever_type, 0) + 1
        assert counts == {"text": 20, "entity": 40}

    def test_the_highest_scoring_items_of_each_type_are_the_ones_kept(self) -> None:
        ranked = self._typed("text", 5, 0.9)
        out = HybridScorer._select_with_type_quota(
            ranked, top_k=2, per_type_quota={"text": 2}
        )
        assert [r.content for r in out] == ["text-0", "text-1"]

    def test_types_without_a_quota_entry_still_backfill(self) -> None:
        # A quota dict names the streams it bounds; anything else keeps the old
        # behavior of filling the remaining budget by score.
        ranked = self._typed("text", 2, 0.9) + self._typed("claim", 5, 0.5)
        out = HybridScorer._select_with_type_quota(
            ranked, top_k=6, per_type_quota={"text": 2}
        )
        counts: dict[str, int] = {}
        for r in out:
            counts[r.retriever_type] = counts.get(r.retriever_type, 0) + 1
        assert counts["text"] == 2
        assert counts["claim"] == 4  # budget max(top_k=6, quota sum=2) - 2 text

    def test_diversity_cut_keeps_every_type_its_reserved_slots(self) -> None:
        # MMR runs before the quota. A single cross-type cut to the quota total
        # let over-quota entities (higher fused scores) take slots the quota
        # then dropped anyway, so text kept 16 of its 20. Each type must be
        # diversified within its own quota instead.
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 0.5

        def stream(rtype: str, n: int, base: float) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    content=" ".join(f"{rtype}{i}w{j}" for j in range(10)),
                    score=base - i * 0.001,
                    source=f"{rtype}{i}",
                    retriever_type=rtype,
                )
                for i in range(n)
            ]

        quota = {"text": 20, "entity": 10}
        out = HybridScorer(config).fuse_and_rerank_results(
            {
                "entities": stream("entity", 30, 0.9),
                "text_units": stream("text", 20, 0.5),
            },
            top_k=10,
            per_type_quota=quota,
            rerank_only_types={"text"},
        )

        counts: dict[str, int] = {}
        for r in out:
            counts[r.retriever_type] = counts.get(r.retriever_type, 0) + 1
        assert counts == quota


class TestDiversityFiltering:
    def test_skipped_at_lambda_one(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 1.0
        scorer = HybridScorer(config)
        results = [_r("a b c", 0.9, "1"), _r("a b c", 0.8, "2")]
        out = scorer._apply_diversity_filtering(list(results), top_k=10)
        assert len(out) == 2  # no filtering

    def test_penalizes_redundant_at_low_lambda(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 0.1  # strong diversity
        scorer = HybridScorer(config)
        # Two near-duplicates + one distinct; MMR should prefer the distinct one
        # over the redundant duplicate for the 2nd slot.
        results = [
            _r("alpha beta gamma", 1.0, "1"),
            _r("alpha beta gamma delta", 0.95, "2"),
            _r("completely different content here", 0.5, "3"),
        ]
        out = scorer._apply_diversity_filtering(list(results), top_k=2)
        assert out[0].content == "alpha beta gamma"
        assert out[1].content == "completely different content here"


class TestDiversityRelevanceScale:
    """MMR must compare relevance and similarity on the same [0, 1] scale.

    RRF scores are ~1/(k+rank) (~0.009-0.016 at k=60), so comparing them raw
    against a 0..1 Jaccard penalty lets the penalty dominate and keeps the
    results LEAST similar to rank 1 (off-topic) instead of the relevant ones.
    """

    _ON_TOPIC_WORDS = "vendor invoice"
    _OFF_TOPIC_WORDS = (
        "lunar orbit telescope",
        "garden tomato soil",
        "jazz piano chord",
        "mountain trail weather",
        "ocean coral reef",
        "chess opening gambit",
    )

    def _scenario(self) -> list[RetrievalResult]:
        # Six on-topic chunks sharing the query vocabulary (Jaccard ~0.14 between
        # any two) and six off-topic chunks with disjoint vocabulary, ranked
        # on-topic first.
        on_topic = [
            _r(
                f"{self._ON_TOPIC_WORDS} payment{i} terms{i} net{i} days{i} "
                f"fee{i} clause{i}",
                0.9 - i * 0.01,
                f"t{i}",
            )
            for i in range(6)
        ]
        off_topic = [
            _r(words, 0.5 - i * 0.01, f"o{i}")
            for i, words in enumerate(self._OFF_TOPIC_WORDS)
        ]
        return on_topic + off_topic

    def test_rrf_scores_keep_on_topic_results(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 0.5  # the shipped default
        scorer = HybridScorer(config)

        out = scorer.fuse_and_rerank_results({"text_units": self._scenario()}, top_k=4)

        assert [r.source for r in out] == ["t0", "t1", "t2", "t3"]

    def test_mmr_on_raw_rrf_scale_scores(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 0.5
        scorer = HybridScorer(config)
        results = self._scenario()
        for rank, result in enumerate(results, start=1):
            result.score = 1.0 / (60 + rank)

        out = scorer._apply_diversity_filtering(results, top_k=4)

        assert [r.source for r in out] == ["t0", "t1", "t2", "t3"]

    def test_scale_invariant(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 0.3
        scorer = HybridScorer(config)
        small = self._scenario()
        large = self._scenario()
        for rank, (a, b) in enumerate(zip(small, large, strict=True), start=1):
            a.score = 1.0 / (60 + rank)
            b.score = 1000.0 / (60 + rank)

        out_small = scorer._apply_diversity_filtering(small, top_k=5)
        out_large = scorer._apply_diversity_filtering(large, top_k=5)

        assert [r.source for r in out_small] == [r.source for r in out_large]

    def test_single_item_returned_unchanged(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 0.5
        scorer = HybridScorer(config)
        only = [_r("alpha beta", 0.01, "1")]

        assert scorer._apply_diversity_filtering(only, top_k=4) == only

    def test_equal_scores_fall_back_to_diversity(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 0.5
        scorer = HybridScorer(config)
        results = [
            _r("alpha beta gamma", 0.5, "1"),
            _r("alpha beta gamma delta", 0.5, "2"),
            _r("completely different content here", 0.5, "3"),
        ]

        out = scorer._apply_diversity_filtering(results, top_k=2)

        # No relevance signal: the first item seeds the set, the most dissimilar
        # candidate wins the next slot, and nothing raises on a zero score range.
        assert [r.source for r in out] == ["1", "3"]

    def test_lambda_one_preserves_relevance_order(self) -> None:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 1.0
        scorer = HybridScorer(config)

        out = scorer.fuse_and_rerank_results({"text_units": self._scenario()}, top_k=4)

        assert [r.source for r in out] == ["t0", "t1", "t2", "t3"]


def _reference_mmr(
    results: list[RetrievalResult], target_count: int, lambda_val: float
) -> list[str]:
    """The original MMR loop (rescans the selected set per candidate per round).

    Kept as the oracle the incremental implementation must match exactly.
    """
    ranked = sorted(results, key=lambda x: x.score or 0.0, reverse=True)
    words = [set((r.content or "").lower().split()) for r in ranked]
    scores = [r.score or 0.0 for r in ranked]
    span = scores[0] - scores[-1]
    relevances = (
        [(s - scores[-1]) / span for s in scores] if span > 0 else [1.0] * len(scores)
    )
    selected = [0]
    remaining = set(range(1, len(ranked)))

    def mmr(idx: int) -> float:
        similarity = max(
            (
                (
                    len(words[idx] & words[j]) / len(words[idx] | words[j])
                    if words[idx] | words[j]
                    else 0.0
                )
                for j in selected
            ),
            default=0.0,
        )
        return lambda_val * relevances[idx] - (1 - lambda_val) * similarity

    while remaining and len(selected) < target_count:
        best = max(sorted(remaining), key=mmr)
        selected.append(best)
        remaining.remove(best)
    return [str(ranked[i].source) for i in selected]


class TestDiversityFilteringScaling:
    """The MMR selection is incremental (O(n * target)) and skipped when it
    cannot remove anything; neither may change which results are kept."""

    @staticmethod
    def _candidates(n: int, seed: int, words: int = 40) -> list[RetrievalResult]:
        import random

        rng = random.Random(seed)
        vocab = [f"term{i}" for i in range(300)]
        return [
            _r(
                " ".join(rng.sample(vocab, rng.randint(0, words))),
                round(rng.random(), 2),  # coarse scores -> exercises ties
                f"s{i}",
            )
            for i in range(n)
        ]

    @staticmethod
    def _scorer_at(lambda_val: float) -> HybridScorer:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = lambda_val
        return HybridScorer(config)

    @pytest.mark.parametrize("seed", range(12))
    def test_matches_reference_selection(self, seed: int) -> None:
        lambda_val = (0.0, 0.3, 0.5, 0.8)[seed % 4]
        results = self._candidates(60, seed)
        target = (1, 5, 17, 59)[seed % 4]

        out = self._scorer_at(lambda_val)._apply_diversity_filtering(
            list(results), top_k=target
        )

        assert [r.source for r in out] == _reference_mmr(results, target, lambda_val)

    def test_cut_covering_every_candidate_keeps_all_in_score_order(self) -> None:
        results = self._candidates(30, seed=3)
        scorer = self._scorer_at(0.5)

        out = scorer._diversify(
            list(results), top_k=10, retrieval_multiplier=3, per_type_quota=None
        )

        assert sorted(r.source for r in out) == sorted(r.source for r in results)
        assert [r.score for r in out] == sorted(
            (r.score for r in results), reverse=True
        )

    def test_scales_quadratically_not_cubically(self) -> None:
        # Mix-shaped candidate counts used to take 2-19 s with the cubic loop.
        # Compare 4x the candidates against the same machine instead of a
        # wall-clock bound: quadratic work grows ~16x, cubic ~64x. CPU time
        # (not wall-clock time, which counts the time other processes hold
        # the CPU: the ratio failed on a loaded host) and the best of five
        # runs each keep scheduler noise out of the ratio.
        import time

        scorer = self._scorer_at(0.5)

        def best_time(n: int) -> float:
            results = self._candidates(n, seed=7, words=60)
            times = []
            for _ in range(5):
                start = time.process_time()
                out = scorer._apply_diversity_filtering(list(results), top_k=n - 1)
                times.append(time.process_time() - start)
                assert len(out) == n - 1
            return min(times)

        assert best_time(400) < 32 * best_time(100)


class TestRerankDegradation:
    def test_returns_original_when_rerank_model_raises(self) -> None:
        scorer = _scorer()

        class _BoomModel:
            top_n = 5

            def compress_documents(self, documents, query):  # noqa: ANN001
                raise RuntimeError("rerank backend down")

        scorer.rerank_model = _BoomModel()
        results = [_r("a", 0.9, "1"), _r("b", 0.8, "2")]
        out, stage = scorer._apply_bedrock_reranking(results, query="q", scope="all")
        assert out == results  # unchanged on failure
        assert stage.status == "failed"
        assert _BoomModel.top_n == 5  # restored in finally

    def test_a_reranker_without_scores_keeps_positive_rank_scores(self) -> None:
        # A reranker that orders without relevance_score: the fallback score
        # stays positive and decreasing past the 100th document.
        scorer = _scorer()

        class _OrderOnly:
            top_n = 150

            def compress_documents(self, documents, query):  # noqa: ANN001
                return list(reversed(documents))

        scorer.rerank_model = _OrderOnly()
        results = [_r(f"doc {n}", 1.0 - n / 1000, str(n)) for n in range(150)]
        out, stage = scorer._apply_bedrock_reranking(results, query="q", scope="all")
        scores = [r.score for r in out]
        assert len(scores) == 150 and all(score > 0 for score in scores)
        assert scores == sorted(scores, reverse=True) and len(set(scores)) == 150
        assert scores[0] == 1.0 and scores[1] == 0.5


class TestRerankBeforeDiversity:
    """Fusion runs fuse -> rerank -> MMR -> quota.

    MMR used to cut each type to its quota on the fused (rank-only) scores
    before the rerank, so the item the reranker ranks first could be cut
    before the reranker ever saw it.
    """

    class _ScoreModel:
        """A rerank model that scores by a fixed table and records its input."""

        top_n = 100

        def __init__(self, scores: dict[str, float]) -> None:
            self.scores = scores
            self.seen: list[str] = []

        def compress_documents(self, documents, query):  # noqa: ANN001
            # A per-call copy runs this (top_n is never set on the shared
            # model), so record into the list the copy shares.
            self.seen[:] = [d.metadata["source"] for d in documents]
            ranked = sorted(documents, key=lambda d: -self.scores[d.metadata["source"]])
            for doc in ranked:
                doc.metadata["relevance_score"] = self.scores[doc.metadata["source"]]
            return ranked[: self.top_n]

    @staticmethod
    def _chunks() -> list[RetrievalResult]:
        # Fused order a > b > c > d; c and d repeat a's words, b does not.
        return [
            RetrievalResult(
                content=text, score=score, source=src, retriever_type="text"
            )
            for src, text, score in (
                ("a", "x y", 0.9),
                ("b", "p q", 0.8),
                ("c", "x y z", 0.7),
                ("d", "x y w", 0.6),
            )
        ]

    def _scorer(self, scores: dict[str, float], top_n: int = 100) -> HybridScorer:
        config = Config()
        config.providers.rerank.kind = "none"
        config.search.fusion.diversity_lambda = 0.5
        scorer = HybridScorer(config)
        scorer.rerank_model = self._ScoreModel(scores)
        scorer.rerank_model.top_n = top_n
        return scorer

    def test_the_rerankers_best_item_is_not_cut_by_diversity(self) -> None:
        scorer = self._scorer({"a": 0.6, "b": 0.7, "c": 0.8, "d": 0.9})
        out = scorer.fuse_and_rerank_results(
            {"chunks": self._chunks()},
            top_k=2,
            query="q",
            per_type_quota={"text": 2},
            rerank_only_types={"text"},
        )
        # Every candidate reached the reranker, and its best one survives.
        assert sorted(scorer.rerank_model.seen) == ["a", "b", "c", "d"]
        assert [r.source for r in out] == ["d", "b"]

    def test_the_reranker_sees_at_most_its_top_n_best_fused(self) -> None:
        scorer = self._scorer({"a": 0.6, "b": 0.7, "c": 0.8, "d": 0.9}, top_n=3)
        scorer.fuse_and_rerank_results({"chunks": self._chunks()}, top_k=4, query="q")
        assert scorer.rerank_model.seen == ["a", "b", "c"]


def _diverse(texts: list[str]) -> list[str]:
    config = Config()
    config.providers.rerank.kind = "none"
    scorer = HybridScorer(config)
    results = [
        RetrievalResult(
            content=text,
            score=1.0 - i * 0.01,
            source=f"c{i}",
            retriever_type="text",
            metadata={},
        )
        for i, text in enumerate(texts)
    ]
    return [r.source for r in scorer._apply_diversity_filtering(results, top_k=2)]


def test_mmr_drops_a_japanese_near_duplicate() -> None:
    # Regression: redundancy was measured over whitespace words, and an
    # unspaced Japanese sentence is one: a copy with one word appended
    # shared nothing with its original and was kept over a distinct passage.
    original = "アクメ社は二〇二〇年にベータ社を一千万ドルで買収した。"
    other = "ガンマ社はオハイオ州で輸出用の部品を製造している。"
    assert _diverse([original, original + "今日", other]) == ["c0", "c2"]


def test_mmr_on_spaced_text_is_unchanged() -> None:
    original = "Acme Corp acquired Beta Inc in 2020 for ten million dollars"
    other = "Gamma Ltd makes widgets in Ohio for export"
    assert _diverse([original, original + " today", other]) == ["c0", "c2"]
