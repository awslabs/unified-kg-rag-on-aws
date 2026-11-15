# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity-resolution grouping must not lose same-name distinct entities.

Regression: _group_similar_entities keyed a dict on entity.name
({name: entity}), so two distinct entities sharing a surface name collapsed to
one (last-writer-wins) BEFORE resolution ran — silently dropping the others'
type/description/text_unit_ids. Grouping must carry every entity through.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.graph_resolver import EntityResolver
from unified_kg_rag.domain.models import Config, Entity, ResolutionMethod

pytestmark = pytest.mark.unit


def _resolver() -> EntityResolver:
    # Threads (not processes) so the test runs in-process; small input.
    return EntityResolver(Config(), max_workers=2, use_process_pool=False)


def test_same_name_entities_are_not_dropped() -> None:
    resolver = _resolver()
    resolver.show_progress = False
    entities = [
        Entity(id="e1", name="Mercury", type="planet", text_unit_ids=["t1"]),
        Entity(id="e2", name="Mercury", type="element", text_unit_ids=["t2"]),
        Entity(id="e3", name="Venus", type="planet", text_unit_ids=["t3"]),
    ]
    groups = resolver._group_similar_entities(entities)
    grouped_ids = {e.id for group in groups for e in group}
    # All three survive grouping — none silently dropped by name collision.
    assert grouped_ids == {"e1", "e2", "e3"}


def test_distinct_names_each_present() -> None:
    resolver = _resolver()
    resolver.show_progress = False
    entities = [
        Entity(id="a", name="Alpha", type="x"),
        Entity(id="b", name="Beta", type="y"),
    ]
    groups = resolver._group_similar_entities(entities)
    assert {e.id for g in groups for e in g} == {"a", "b"}


def test_matcher_passed_to_pool_via_initializer_not_per_task(mocker) -> None:
    # Regression (perf): the FuzzyMatcher (tens of MB at scale) must be shipped
    # to workers ONCE via the pool initializer, not pickled per submitted task.
    # Assert the executor is constructed with initializer/initargs and that
    # submit() is called with ONLY the entity name (no matcher argument).
    import unified_kg_rag.domain.ingestion.graph_resolver as gr

    config = Config()
    config.processing.fuzzy_entity_resolution = True
    resolver = EntityResolver(config, max_workers=2, use_process_pool=False)
    resolver.show_progress = False

    captured: dict = {}

    class FakeExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=None):
            captured["initializer"] = initializer
            captured["initargs"] = initargs

        def shutdown(self, wait=True, cancel_futures=False):
            pass

        def submit(self, fn, *args):
            captured.setdefault("submit_args", []).append(args)

            class _F:
                def result(self_inner):
                    return []

            return _F()

    mocker.patch.object(gr, "ContextThreadPoolExecutor", FakeExecutor)
    mocker.patch.object(gr, "as_completed", lambda futs: list(futs))

    resolver._group_similar_entities(
        [Entity(id="a", name="Alpha"), Entity(id="b", name="Beta")]
    )

    # The matcher is handed to the pool initializer exactly once...
    assert captured["initializer"] is gr._init_worker_fuzzy_matcher
    assert captured["initargs"] is not None and len(captured["initargs"]) == 1
    # ...and each submit carries only the (tiny) name, never the matcher.
    assert all(
        len(args) == 1 and isinstance(args[0], str) for args in captured["submit_args"]
    )


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.6, 0.75, 0.85, 0.95])
@pytest.mark.parametrize("num_perm", [64, 128])
def test_the_lsh_layout_proposes_the_pairs_at_the_threshold_at_a_bounded_cost(
    threshold: float, num_perm: int
) -> None:
    # datasketch's own layout (8 bands of 16 rows at 0.85 and 128) proposed
    # a pair at the threshold 46% of the time; the rest never matched. The
    # recall-tuned one proposed most pairs at half a low threshold too (42
    # bands of 3 rows at 0.6: 68% at 0.3, 118 candidates a query on a
    # repetitive name set), so its cost is bounded, and it never proposes
    # fewer pairs at the threshold than datasketch's.
    from datasketch import MinHashLSH

    from unified_kg_rag.domain.ingestion.base_resolver import (
        LSH_MAX_PROPOSED_AT_HALF,
        LSH_RECALL_AT_THRESHOLD,
        _candidate_probability,
        _lsh_params,
    )

    bands, rows = _lsh_params(threshold, num_perm)
    assert bands * rows <= num_perm
    own = MinHashLSH(threshold=threshold, num_perm=num_perm)
    at = _candidate_probability(threshold, bands, rows)
    assert at >= _candidate_probability(threshold, own.b, own.r)
    if (bands, rows) != (own.b, own.r):
        assert (
            _candidate_probability(threshold / 2, bands, rows)
            <= LSH_MAX_PROPOSED_AT_HALF
        )
    if threshold >= 0.85 and num_perm == 128:  # the package's permutations
        assert at >= LSH_RECALL_AT_THRESHOLD


def test_matchers_derive_the_lsh_layout_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # The layout search is cached per (threshold, permutations), not repeated
    # by every matcher.
    from unified_kg_rag.domain.ingestion import base_resolver

    calls: list[tuple] = []
    real = base_resolver._candidate_probability

    def counted(*args):  # type: ignore[no-untyped-def]
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(base_resolver, "_candidate_probability", counted)
    base_resolver._lsh_params.cache_clear()
    try:
        for _ in range(3):
            base_resolver.FuzzyMatcher(
                ["Acme", "Acme Inc"],
                resolution_method=base_resolver.ResolutionMethod.MINHASH,
                similarity_threshold=0.42,
            )
        searched = len(calls)
        for _ in range(3):
            base_resolver.FuzzyMatcher(
                ["Acme", "Acme Inc"],
                resolution_method=base_resolver.ResolutionMethod.MINHASH,
                similarity_threshold=0.42,
            )
    finally:
        base_resolver._lsh_params.cache_clear()
    assert searched > 0
    assert len(calls) == searched


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        # Exact 3-gram Jaccard 0.90-0.91: datasketch's layout never proposed
        # these pairs, so the variant never matched its entity.
        ("northwind traders incorporated", "northwind tradders incorporated"),
        ("contoso pharmaceuticals limited", "contoso pharmaceuticals limiteed"),
        ("tailspin toys entertainment company", "tailspin toys entertainment companny"),
    ],
)
def test_minhash_finds_a_pair_above_the_threshold(name: str, variant: str) -> None:
    from unified_kg_rag.domain.ingestion import base_resolver

    matcher = base_resolver.FuzzyMatcher(
        [name], resolution_method=ResolutionMethod.MINHASH, similarity_threshold=0.85
    )
    assert [candidate for candidate, _ in matcher.find_all_matches(variant)] == [name]


def test_minhash_matches_by_the_exact_jaccard_similarity() -> None:
    from unified_kg_rag.domain.ingestion.base_resolver import (
        FuzzyMatcher,
        name_shingles,
    )

    name, variant = "northwind traders incorporated", "northwind tradders incorporated"
    exact = len(name_shingles(name) & name_shingles(variant)) / len(
        name_shingles(name) | name_shingles(variant)
    )
    at = FuzzyMatcher(
        [name], resolution_method=ResolutionMethod.MINHASH, similarity_threshold=exact
    )
    above = FuzzyMatcher(
        [name],
        resolution_method=ResolutionMethod.MINHASH,
        similarity_threshold=exact + 0.001,
    )
    assert at.find_all_matches(variant) == [(name, exact)]
    assert above.find_all_matches(variant) == []


def test_the_minhash_threshold_is_its_own_once_set() -> None:
    processing = Config().processing
    assert processing.resolution_threshold == processing.similarity_threshold == 0.85
    processing.minhash_similarity_threshold = 0.7
    assert processing.resolution_threshold == 0.7
    processing.resolution_method = ResolutionMethod.SEQUENCE_MATCHER
    assert processing.resolution_threshold == 0.85
