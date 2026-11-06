# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Algorithm-level parity with Microsoft BenchmarkQED (optional).

``evaluation.benchmarks.autoq`` and the ``assertion`` evaluator reimplement
BenchmarkQED's decision steps (https://github.com/microsoft/benchmark-qed,
MIT). These tests feed the same synthetic inputs and the same scripted model
replies to both and compare the decisions: claim parsing, assertion parsing
and ranking, the validation rule, the per-trial assertion aggregation and the
clustered text sampling. They run only where ``benchmark_qed`` is importable
(it is not a dependency: install it in a scratch environment, never its
datasets, which are under the Microsoft Research License); the intentional
deviations they pin are listed in docs/design.md#92-length-controlled-judging-assertions-and-the-closed-book-arm.
"""

from __future__ import annotations

import asyncio
import json
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unified_kg_rag.evaluation import assertion_evaluator
from unified_kg_rag.evaluation.benchmarks import autoq

bqed = pytest.importorskip("benchmark_qed")

pd = pytest.importorskip("pandas")

from benchmark_qed.autod.data_model.text_unit import TextUnit  # noqa: E402
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster  # noqa: E402
from benchmark_qed.autod.sampler.clustering.kmeans import (  # noqa: E402
    KmeansClustering,
)
from benchmark_qed.autod.sampler.sampling import kmeans_sampler  # noqa: E402
from benchmark_qed.autoe.assertion.aggregation import (  # noqa: E402
    summarize_standard_scores,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen import (  # noqa: E402
    local_claim_assertion_gen,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.validator import (  # noqa: E402
    ValidationScores,
)
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor import (  # noqa: E402
    local_claim_extractor,
)

pytestmark = pytest.mark.unit

# A fixed synthetic sample (neutral entities, made-up figures).
_TEXTS = [
    "Vendor supplies widgets to Buyer at a fixed price of 40 credits.",
    "Depot stores gadgets for Vendor in a bonded warehouse.",
    "Carrier moves gadgets for Depot every week by rail.",
    "Buyer pays a penalty of 100 credits per late day.",
    "벤더는 구매자에게 위젯을 공급한다.",
    "倉庫は商品を保管する。",
]

_CLAIM_REPLY = {
    "claims": [
        {"statement": "Vendor supplies widgets.", "sources": ["1", 2], "score": 8},
        {"statement": "Penalty is 100 credits.", "sources": ["4"], "score": 9},
        {"statement": "Invented.", "sources": ["9"], "score": 9},
        {"statement": "Irrelevant.", "sources": ["1"], "score": 0},
        {"statement": "", "sources": ["1"], "score": 5},
        {"statement": "Score as text.", "sources": ["1"], "score": "7"},
        {"statement": "벤더는 위젯을 공급한다.", "sources": ["5"], "score": 6},
        {"statement": "No list.", "sources": "1", "score": 5},
    ]
}


def _their_units() -> list[TextUnit]:
    return [
        TextUnit(id=f"u{n}", short_id=str(n), text=text)
        for n, text in enumerate(_TEXTS, start=1)
    ]


def test_claim_parsing_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    async def chat(llm: Any, messages: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(content=json.dumps(_CLAIM_REPLY))

    monkeypatch.setattr(local_claim_extractor, "chat", chat)
    extractor = local_claim_extractor.DataLocalClaimExtractor(llm=object())
    theirs = asyncio.run(extractor.aextract_claims("Who supplies?", _their_units()))
    ours = autoq.parse_claims(
        _CLAIM_REPLY, {str(n): f"u{n}" for n in range(1, len(_TEXTS) + 1)}
    )
    # Deliberate deviation: a claim citing only ids the prompt never gave
    # keeps its raw sources there (the extractor's ``continue`` skips the
    # source mapping, and the later filter checks only that sources are
    # non-empty); here it is dropped, since it cites no given text.
    kept = [c for c in theirs.claims if "source_ids" in c]
    assert [c["statement"] for c in theirs.claims if "source_ids" not in c] == [
        "Invented."
    ]
    assert [(c["statement"], set(c["source_ids"])) for c in kept] == [
        (c.statement, {s.removeprefix("u") for s in c.sources}) for c in ours
    ]


def _claims() -> list[autoq.Claim]:
    return [
        autoq.Claim(statement="A", sources=["u1"], score=5),
        autoq.Claim(statement="B", sources=["u2", "u3"], score=7),
        autoq.Claim(statement="C", sources=["u4"], score=3),
        autoq.Claim(statement="D", sources=["u1", "u4", "u5"], score=9),
    ]


def _their_claims(claims: list[autoq.Claim]) -> list[dict[str, Any]]:
    texts = {f"u{n}": t for n, t in enumerate(_TEXTS, start=1)}
    return [
        {
            "statement": c.statement,
            "score": c.score,
            "sources": [{"source_id": s, "text": texts[s]} for s in c.sources],
        }
        for c in claims
    ]


@pytest.mark.parametrize("seed", range(5))
def test_assertion_parsing_and_ranking_match(seed: int) -> None:
    rng = random.Random(seed)
    entries = []
    for n in range(12):
        sources = rng.sample(
            ["claim_1", "claim_2", "claim_3", "claim_4", "claim_9"], rng.randint(0, 3)
        )
        entries.append(
            {
                "statement": f"Assertion {n}",
                "sources": sources,
                "score": rng.choice([0, 1, 4, 5, 7, 10, 11]),
            }
        )
    reply = {"assertions": entries}
    claims = _claims()
    generator = local_claim_assertion_gen.LocalClaimAssertionGenerator(
        llm=object(), max_assertions=5
    )
    theirs = generator._rank_and_limit_assertions(
        generator._parse_assertions(entries, claims=_their_claims(claims)), 5
    )
    ours = autoq.rank_assertions(autoq.parse_assertions(reply, claims), 5)
    # The same assertions, in the same rank order, with the same sources.
    assert [a.statement for a in theirs] == [a.statement for a in ours]
    texts = {f"u{n}": t for n, t in enumerate(_TEXTS, start=1)}
    assert [set(a.sources) for a in theirs] == [
        {texts[u] for u in a.sources} for a in ours
    ]


@pytest.mark.parametrize(
    "scores",
    [
        {"grounding": 5, "relevance": 3, "verifiability": 4},
        {"grounding": 2, "relevance": 5, "verifiability": 5},
        {"grounding": 9, "relevance": 3, "verifiability": 3},  # clamped to 5
        {"grounding": 3, "relevance": 3, "verifiability": 0},  # clamped to 1
        {"grounding": "4", "relevance": 4, "verifiability": 4},
    ],
)
def test_the_validation_rule_matches_on_complete_scores(scores: dict) -> None:
    theirs = ValidationScores.from_llm_response(scores).is_valid(3)
    ours = autoq.parse_validation({"scores": [{"id": 1, **scores}]}, 1, 3)
    assert ours == [theirs]


def test_a_missing_score_is_a_deliberate_deviation() -> None:
    # BenchmarkQED defaults a missing score to 3 (a pass); a reply that does
    # not score an assertion fails it here.
    partial = {"grounding": 5, "relevance": 5}
    assert ValidationScores.from_llm_response(partial).is_valid(3) is True
    assert autoq.parse_validation({"scores": [{"id": 1, **partial}]}, 1, 3) == [False]


@pytest.mark.parametrize("seed", range(5))
def test_trial_aggregation_matches_the_pass_rate(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    ours: dict[str, float] = {}
    for q in range(6):
        count, trials = int(rng.integers(1, 6)), int(rng.integers(1, 7))
        labels = [
            [
                (
                    "supported"
                    if rng.random() < 0.6
                    else rng.choice(["absent", "contradicted"])
                )
                for _ in range(count)
            ]
            for _ in range(trials)
        ]
        for t, trial in enumerate(labels):
            for a, label in enumerate(trial):
                rows.append(
                    {
                        "question": f"q{q}",
                        "assertion": f"a{a}",
                        "trial": t,
                        "score": int(label == "supported"),
                    }
                )
        aggregate = assertion_evaluator.aggregate_trials(["positive"] * count, labels)
        ours[f"q{q}"] = aggregate["assertion_recall"]
    _, by_question, _ = summarize_standard_scores(pd.DataFrame(rows))
    theirs = dict(zip(by_question["question"], by_question["pass_rate"], strict=True))
    assert theirs == pytest.approx(ours)


def test_cluster_sampling_matches_on_fixed_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(1)
    vectors = np.vstack(
        [rng.normal(c, 0.3, size=(n, 8)) for c, n in ((0, 9), (3, 7), (-3, 5))]
    )
    labels = np.repeat([0, 1, 2], [9, 7, 5])
    units = [
        TextUnit(id=f"u{i}", short_id=str(i), text="t", text_embedding=list(v))
        for i, v in enumerate(vectors)
    ]

    def fixed(self: Any, text_units: Any, num_clusters: Any = None, **kw: Any) -> Any:
        groups: dict[int, list[TextUnit]] = {}
        for label, unit in zip(labels.tolist(), text_units, strict=True):
            groups.setdefault(label, []).append(unit)
        return [TextCluster(id=str(k), text_units=v) for k, v in groups.items()]

    monkeypatch.setattr(KmeansClustering, "cluster", fixed)
    sampler = kmeans_sampler.KmeansTextSampler(random_seed=0)
    theirs = sampler.sample(
        text_units=units, sample_size=None, num_clusters=3, num_samples_per_cluster=4
    )
    by_cluster: dict[str, list[str]] = {}
    for unit in theirs:
        by_cluster.setdefault(unit.cluster_id, []).append(unit.id)
    ours = autoq.cluster_samples(vectors, labels, per_cluster=4)
    assert [by_cluster[str(n)] for n in range(3)] == [
        [f"u{i}" for i in idx] for _, idx in ours
    ]


def test_kmeans_finds_the_same_partition_on_separated_groups() -> None:
    rng = np.random.default_rng(2)
    vectors = np.vstack([rng.normal(c, 0.2, size=(12, 4)) for c in (0, 4, 8, -4)])
    units = [
        TextUnit(id=str(i), short_id=str(i), text="t", text_embedding=list(v))
        for i, v in enumerate(vectors)
    ]
    theirs = KmeansClustering(random_seed=0).cluster(units, num_clusters=4)
    their_groups = {frozenset(int(u.id) for u in c.text_units) for c in theirs}
    labels = autoq.kmeans(vectors, 4, seed=0)
    our_groups = {
        frozenset(int(i) for i in np.where(labels == k)[0]) for k in set(labels)
    }
    assert their_groups == our_groups
