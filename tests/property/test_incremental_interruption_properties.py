# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An interrupted incremental run never leaves the index unrepairable.

Random corpora: one is indexed, the next run over another is interrupted at a
random write step, and the run after that sees a follow-up corpus: the
interrupted run's corpus again, that corpus with the documents it changed
restored to their indexed content, or that corpus without the documents it
added. The follow-up run must leave exactly what a fresh full build of the
follow-up corpus stores, with every registry row PROCESSED and its text
units matching the stores.

A random case can name a point its run never reaches (no removal, fewer
writes than ``k``), so each case records whether the interruption fired
(``hypothesis.event``), and ``test_each_interruption_point_fires`` pins cases
that reach every point (also run as explicit examples of the property).

The seed variant puts a seed graph's shard documents beside the documents
of each corpus (its nodes name document entities): the interrupted run
adds, edits, re-shards or removes the seed, and the follow-up converges the
same way. A seed may spell a name twice (merged or qualified; one that
refuses the collision fails its run at start-up, so it is not drawn). The
strict variant runs it all under a strict extraction schema: the documents'
statements and the seeds are checked as extraction and the conversion check
them, and the off-schema ones dropped.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    POINTS,
    Corpus,
    IncrementalRun,
    expected_state,
)
from tests.fixtures.seed_graphs import EX, Seed, random_seeds, seeded_corpus
from unified_kg_rag.domain.ingestion.names import compact_name_key
from unified_kg_rag.domain.models import Config, ExtractionSchema
from unified_kg_rag.domain.models.seed_graph import SeedEdge, SeedNode
from unified_kg_rag.shared.exceptions import SeedGraphError

pytestmark = pytest.mark.property

_PATHS = ["/corpus/a.txt", "/corpus/b.txt", "/corpus/c.txt", "/corpus/d.txt"]
_NAMES = ["Vendor", "Depot", "Carrier", "Buyer"]
# Spellings of one entity each (legal forms the full build folds, in
# English, Korean and Japanese): which one an entity is stored under
# follows the corpus, so an interrupted commit may be re-keying it.
_SPELLINGS = [
    "Acme Inc",
    "Acme Corp",
    "주식회사 한빛",
    "(주)한빛",
    "한빛 주식회사",
    "株式会社トヨダ",
    "トヨダ",
]

_edge = st.tuples(
    st.sampled_from(_NAMES + _SPELLINGS),
    st.sampled_from(_NAMES + _SPELLINGS),
    st.integers(1, 3),
).filter(lambda e: compact_name_key(e[0]) != compact_name_key(e[1]))
_unit = st.lists(_edge, min_size=1, max_size=3, unique_by=lambda e: (e[0], e[1]))
_content = st.lists(_unit, min_size=1, max_size=2)
_corpus = st.fixed_dictionaries({path: st.none() | _content for path in _PATHS}).map(
    lambda state: {path: content for path, content in state.items() if content}
)


_V, _D, _C, _B = _NAMES
# Four indexed documents; the interrupted run changes two (one into two text
# units) and removes two, so it reaches every interruption point.
_BEFORE: Corpus = {
    "/corpus/a.txt": [[(_V, _D, 1)]],
    "/corpus/b.txt": [[(_C, _B, 1)]],
    "/corpus/c.txt": [[(_V, _B, 2)]],
    "/corpus/d.txt": [[(_D, _C, 1)]],
}
_CHANGED_AND_REMOVED: Corpus = {
    "/corpus/a.txt": [[(_V, _D, 2), (_D, _C, 1)]],
    "/corpus/c.txt": [[(_V, _C, 3)], [(_B, _D, 1)]],
}
# Changes a and removes the other three: three record deletes.
_ONE_CHANGED: Corpus = {"/corpus/a.txt": [[(_V, _D, 2), (_D, _C, 1)]]}


def _case(interrupted: Corpus, point: str, k: int) -> dict:
    return {
        "before": _BEFORE,
        "interrupted": interrupted,
        "point": point,
        "k": k,
        "follow": "same",
    }


_FIRING_CASES = [
    *(_case(_CHANGED_AND_REMOVED, point, 1) for point in POINTS),
    _case(_CHANGED_AND_REMOVED, "mid_record_deletes", 2),
    _case(_ONE_CHANGED, "mid_record_deletes", 3),
    _case(_ONE_CHANGED, "after_removal", 1),
    # A new document's delta interrupted at its first store write leaves a
    # vector copy without its graph copy; removing the document finds it.
    {
        "before": {},
        "interrupted": {"/corpus/d.txt": [[(_V, _D, 1)]]},
        "point": "mid_index_delta",
        "k": 1,
        "follow": "new_docs_removed",
    },
    # Stripping a shared edge interrupted after its endpoints were stripped:
    # the retry still finds the edge through an endpoint citing the unit.
    {
        "before": {
            "/corpus/c.txt": [[(_V, _D, 1)]],
            "/corpus/d.txt": [[(_C, _V, 2), (_V, _C, 1)], [(_V, _D, 1), (_D, _V, 1)]],
        },
        "interrupted": {
            "/corpus/d.txt": [[(_C, _V, 2), (_V, _C, 1)], [(_V, _D, 1), (_D, _V, 1)]]
        },
        "point": "mid_index_delta",
        "k": 2,
        "follow": "same",
    },
    # Removing a.txt merges "주식회사 한빛" into "한빛 주식회사" and splits
    # c.txt's statement off to "(주)한빛": the strip re-points the edge to
    # Vendor there and folds (주)한빛's own edge to Vendor into it. Killed
    # before the folded id is deleted, the retry no longer reached the
    # re-pointed edge, and the duplicate stayed.
    {
        "before": {
            "/corpus/a.txt": [
                [(_D, _B, 2), ("한빛 주식회사", _D, 3)],
                [(_D, "주식회사 한빛", 3), (_D, _B, 2)],
            ],
            "/corpus/c.txt": [[("주식회사 한빛", _V, 2), ("(주)한빛", _V, 2)]],
            "/corpus/d.txt": [[("한빛 주식회사", _D, 1), (_D, _B, 2), (_B, _D, 3)]],
        },
        "interrupted": {
            "/corpus/c.txt": [[("주식회사 한빛", _V, 2), ("(주)한빛", _V, 2)]],
            "/corpus/d.txt": [[("한빛 주식회사", _D, 1), (_D, _B, 2), (_B, _D, 3)]],
        },
        "point": "before_retired_deletes",
        "k": 1,
        "follow": "same",
    },
]


def _with_firing_examples(test):  # noqa: ANN001, ANN202
    for case in _FIRING_CASES:
        test = example(**case)(test)
    return test


def _follow_up(kind: str, before: Corpus, interrupted: Corpus) -> Corpus:
    if kind == "same":
        return interrupted
    if kind == "reverted":
        return {
            path: before.get(path, content) for path, content in interrupted.items()
        }
    return {path: c for path, c in interrupted.items() if path in before}


@pytest.mark.slow
@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    before=_corpus,
    interrupted=_corpus,
    point=st.sampled_from(POINTS),
    k=st.integers(1, 3),
    follow=st.sampled_from(["same", "reverted", "new_docs_removed"]),
)
@_with_firing_examples
def test_follow_up_run_after_an_interruption_matches_a_full_build(
    before: Corpus,
    interrupted: Corpus,
    point: str,
    k: int,
    follow: str,
) -> None:
    fired = _check_follow_up(before, interrupted, point, k, follow)
    event(f"{point}: {'interrupted' if fired else 'not reached'}")


def _check_follow_up(
    before: Corpus,
    interrupted: Corpus,
    point: str,
    k: int,
    follow: str,
    config: Config | None = None,
) -> bool:
    """Run the three runs (under ``config``, the default when ``None``) and
    check the follow-up; return whether the interruption fired."""
    corpus = _follow_up(follow, before, interrupted)
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(before)
    fired = harness.run(interrupted, interrupt=point, k=k)

    harness.run(corpus)

    assert harness.state() == expected_state(corpus, config)
    assert harness.registry_problems(corpus) == []
    harness.run(corpus)
    assert harness.extracted == []
    return fired


@pytest.mark.parametrize(
    "case",
    _FIRING_CASES,
    ids=[f"{c['point']}-k{c['k']}" for c in _FIRING_CASES],
)
def test_each_interruption_point_fires(case: dict) -> None:
    # The examples above reach their point, so every point is exercised
    # whatever the random cases draw.
    assert _check_follow_up(**case)


# One claim (Vendor rates Depot) stated by units whose statuses disagree
# (incremental_runs._claim_status): c.txt's first unit TRUE, its second
# "true." (one status), a.txt's UNKNOWN (adds nothing), b.txt's second
# FALSE (disputes it). Each check compares the claims' statuses too.
_STATUS_BEFORE: Corpus = {
    "/corpus/c.txt": [[(_V, _D, 3)], [(_V, _D, 3), (_C, _B, 1)]],
    "/corpus/a.txt": [[(_V, _D, 3)]],
}
_STATUS_DISPUTED: Corpus = {
    **_STATUS_BEFORE,
    "/corpus/b.txt": [[(_C, _B, 1)], [(_V, _D, 3)]],
}


@pytest.mark.parametrize("follow", ["same", "new_docs_removed"])
@pytest.mark.parametrize("point", POINTS)
def test_an_interrupted_run_converges_on_the_claims_status(
    point: str, follow: str
) -> None:
    corpus = _follow_up(follow, _STATUS_BEFORE, _STATUS_DISPUTED)
    (status,) = {state[1] for state in expected_state(corpus)["claims"].values()}
    assert status == ("disputed" if follow == "same" else "true")
    _check_follow_up(_STATUS_BEFORE, _STATUS_DISPUTED, point, 1, follow)


# --- seed graphs ------------------------------------------------------------------

_SCHEMA = Config().processing.graph_extraction.extraction_schema()


def _seeded(
    seed: st.SearchStrategy[Seed], schema: ExtractionSchema
) -> st.SearchStrategy[Corpus]:
    """Corpora with ``seed`` (or none) beside the documents, converted
    against ``schema``; a seed that does not convert is not drawn."""

    def convert(drawn: tuple[Corpus, Seed | None]) -> Corpus | None:
        try:
            return seeded_corpus(drawn[0], {"alpha": drawn[1]}, schema)
        except SeedGraphError:
            return None

    return (
        st.tuples(_corpus, st.none() | seed)
        .map(convert)
        .filter(lambda corpus: corpus is not None)
    )


_seeded_corpus = _seeded(
    random_seeds(
        [*_NAMES, "Harbor"],
        (None, "ORG", "PRODUCT"),
        ("RELATED_TO", "SUPPLIES"),
        collisions=("merge", "qualify"),
    ),
    _SCHEMA,
)


def _with_seed(corpus: Corpus, seed: Seed | None) -> Corpus:
    return seeded_corpus(corpus, {"alpha": seed}, _SCHEMA)


_SEED_NODES = (
    SeedNode(id="vendor", name=_V, type="PRODUCT", iri=EX + "vendor"),
    SeedNode(id="harbor", name="Harbor", description="Harbor seed fact"),
    SeedNode(id="depot", name=_D),
)
_SEED_EDGES = (
    SeedEdge(source="vendor", target="harbor", type="RELATED_TO", weight=2.5),
    SeedEdge(source="vendor", target="depot", type="RELATED_TO", description="seed"),
)
_SEED_V1: Seed = (1, _SEED_NODES, _SEED_EDGES)
# Re-sharded, Vendor's card edited.
_SEED_V2: Seed = (
    3,
    (_SEED_NODES[0].model_copy(update={"description": "edited"}), *_SEED_NODES[1:]),
    _SEED_EDGES,
)
_SEED_CASES = [
    {
        "before": _with_seed(_BEFORE, _SEED_V1),
        "interrupted": _with_seed(_CHANGED_AND_REMOVED, _SEED_V2),
        "point": point,
        "k": k,
        "follow": follow,
    }
    for point, k in [
        ("after_write_ahead", 1),
        ("mid_removal", 2),
        ("in_merge", 1),
        ("mid_index_delta", 2),
        ("mid_commit", 2),
    ]
    for follow in ("same", "reverted")
] + [
    # The interrupted run removes the seed (its shards are deleted).
    {
        "before": _with_seed(_BEFORE, _SEED_V1),
        "interrupted": _BEFORE,
        "point": "mid_record_deletes",
        "k": 1,
        "follow": "same",
    },
    # It adds one: the follow-up drops it again.
    {
        "before": _BEFORE,
        "interrupted": _with_seed(_BEFORE, _SEED_V2),
        "point": "mid_index_delta",
        "k": 1,
        "follow": "new_docs_removed",
    },
]


def _with_seed_examples(test):  # noqa: ANN001, ANN202
    for case in _SEED_CASES:
        test = example(**case)(test)
    return test


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    before=_seeded_corpus,
    interrupted=_seeded_corpus,
    point=st.sampled_from(POINTS),
    k=st.integers(1, 3),
    follow=st.sampled_from(["same", "reverted", "new_docs_removed"]),
)
@_with_seed_examples
def test_an_interrupted_seed_round_converges(
    before: Corpus,
    interrupted: Corpus,
    point: str,
    k: int,
    follow: str,
) -> None:
    fired = _check_follow_up(before, interrupted, point, k, follow)
    event(f"{point}: {'interrupted' if fired else 'not reached'}")


@pytest.mark.parametrize(
    "case",
    _SEED_CASES,
    ids=[f"{c['point']}-k{c['k']}-{c['follow']}" for c in _SEED_CASES],
)
def test_each_seed_interruption_fires(case: dict) -> None:
    assert _check_follow_up(**case)


# --- under a strict schema ----------------------------------------------------------

# The documents' types are ORG (odd strengths: off the schema, dropped) and
# PERSON; their edges RELATED_TO. The seeds' types are declared, a spelling
# of a declared one, undeclared or none.
_STRICT = Config.model_validate(
    {
        "processing": {
            "graph_extraction": {
                "schema_mode": "strict",
                "entity_types": [
                    {"label": "ORGANIZATION", "iri": EX + "Organization"},
                    "PERSON",
                ],
                "relationship_types": ["RELATED_TO", "SUPPLIES"],
                "relationship_patterns": ["ORGANIZATION -SUPPLIES-> PERSON"],
            }
        }
    }
)
_strict_seeded_corpus = _seeded(
    random_seeds(
        [*_NAMES, "Harbor"],
        ("ORGANIZATION", "person", "vendor", None),
        ("RELATED_TO", "supplies", "KNOWS"),
        collisions=("merge", "qualify"),
    ),
    _STRICT.processing.graph_extraction.extraction_schema(),
)


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(
    before=_strict_seeded_corpus,
    interrupted=_strict_seeded_corpus,
    point=st.sampled_from(POINTS),
    k=st.integers(1, 3),
    follow=st.sampled_from(["same", "reverted", "new_docs_removed"]),
)
@example(
    before=seeded_corpus(
        _BEFORE,
        {"alpha": _SEED_V1},
        _STRICT.processing.graph_extraction.extraction_schema(),
    ),
    interrupted=seeded_corpus(
        _CHANGED_AND_REMOVED,
        {"alpha": _SEED_V2},
        _STRICT.processing.graph_extraction.extraction_schema(),
    ),
    point="mid_index_delta",
    k=2,
    follow="same",
)
def test_an_interrupted_strict_seed_round_converges(
    before: Corpus,
    interrupted: Corpus,
    point: str,
    k: int,
    follow: str,
) -> None:
    fired = _check_follow_up(before, interrupted, point, k, follow, _STRICT)
    event(f"{point}: {'interrupted' if fired else 'not reached'}")


def test_the_strict_variant_drops_off_schema_statements() -> None:
    # What makes the strict variant one: the full build differs.
    corpus = seeded_corpus(
        _BEFORE,
        {"alpha": _SEED_V1},
        _STRICT.processing.graph_extraction.extraction_schema(),
    )
    assert expected_state(corpus, _STRICT) != expected_state(
        _with_seed(_BEFORE, _SEED_V1)
    )
    assert _check_follow_up(
        corpus,
        seeded_corpus(
            _CHANGED_AND_REMOVED,
            {"alpha": _SEED_V2},
            _STRICT.processing.graph_extraction.extraction_schema(),
        ),
        "mid_index_delta",
        2,
        "same",
        _STRICT,
    )
