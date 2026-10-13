# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An interrupted incremental run never leaves the index unrepairable.

Random corpora: one is indexed, the next run over another is interrupted at a
random write step, and the run after that sees a follow-up corpus: the
interrupted run's corpus again, that corpus with the documents it changed
restored to their indexed content, or that corpus without the documents it
added. The follow-up run must leave exactly what a fresh full build of the
follow-up corpus stores, with every registry row PROCESSED and its lineage
matching the stores.

A random case can name a point its run never reaches (no removal, fewer
writes than ``k``), so each case records whether the interruption fired
(``hypothesis.event``), and ``test_each_interruption_point_fires`` pins cases
that reach every point (also run as explicit examples of the property).
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
    largest_lineage,
)

pytestmark = pytest.mark.property

_PATHS = ["/corpus/a.txt", "/corpus/b.txt", "/corpus/c.txt", "/corpus/d.txt"]
_NAMES = ["Vendor", "Depot", "Carrier", "Buyer"]

_edge = st.tuples(
    st.sampled_from(_NAMES), st.sampled_from(_NAMES), st.integers(1, 3)
).filter(lambda e: e[0] != e[1])
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
# Changes a and removes the other three: three record deletes and, under the
# tight item limit, a write-ahead record spilled into lineage overflow.
_ONE_CHANGED: Corpus = {"/corpus/a.txt": [[(_V, _D, 2), (_D, _C, 1)]]}


def _case(interrupted: Corpus, point: str, k: int, tight: bool) -> dict:
    return {
        "before": _BEFORE,
        "interrupted": interrupted,
        "point": point,
        "k": k,
        "follow": "same",
        "tight": tight,
    }


_FIRING_CASES = [
    *(_case(_CHANGED_AND_REMOVED, point, 1, False) for point in POINTS),
    _case(_CHANGED_AND_REMOVED, "mid_commit", 1, True),
    _case(_CHANGED_AND_REMOVED, "mid_record_deletes", 2, False),
    _case(_ONE_CHANGED, "mid_record_deletes", 3, True),
    _case(_ONE_CHANGED, "after_removal", 1, True),
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
    tight=st.booleans(),
)
@_with_firing_examples
def test_follow_up_run_after_an_interruption_matches_a_full_build(
    before: Corpus,
    interrupted: Corpus,
    point: str,
    k: int,
    follow: str,
    tight: bool,
) -> None:
    fired, _ = _check_follow_up(before, interrupted, point, k, follow, tight)
    event(f"{point}: {'interrupted' if fired else 'not reached'}")


def _check_follow_up(
    before: Corpus,
    interrupted: Corpus,
    point: str,
    k: int,
    follow: str,
    tight: bool,
) -> tuple[bool, bool]:
    """Run the three runs and check the follow-up.

    Returns whether the interruption fired and whether the interrupted run
    left lineage overflow.
    """
    corpus = _follow_up(follow, before, interrupted)
    # A tight registry item limit: every record a commit writes fits, but a
    # write-ahead record (stored + planned ids) may spill into overflow.
    limit = largest_lineage(before, interrupted, corpus) if tight else None
    store = FakeDocStatusStore(max_record_ids=limit)
    harness = IncrementalRun(store)
    harness.run(before)
    fired = harness.run(interrupted, interrupt=point, k=k)
    left_overflow = bool(store.overflow)

    harness.run(corpus)

    assert harness.state() == expected_state(corpus)
    assert harness.registry_problems(corpus) == []
    harness.run(corpus)
    assert harness.extracted == []
    return fired, left_overflow


@pytest.mark.parametrize(
    "case",
    _FIRING_CASES,
    ids=[
        f"{c['point']}-k{c['k']}{'-tight' if c['tight'] else ''}" for c in _FIRING_CASES
    ],
)
def test_each_interruption_point_fires(case: dict) -> None:
    # The examples above reach their point, so every point is exercised
    # whatever the random cases draw.
    fired, left_overflow = _check_follow_up(**case)
    assert fired
    if case["interrupted"] is _ONE_CHANGED and case["tight"]:
        assert left_overflow
