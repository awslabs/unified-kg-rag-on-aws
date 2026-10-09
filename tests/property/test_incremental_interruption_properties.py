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
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
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


def _follow_up(kind: str, before: Corpus, interrupted: Corpus) -> Corpus:
    if kind == "same":
        return interrupted
    if kind == "reverted":
        return {
            path: before.get(path, content) for path, content in interrupted.items()
        }
    return {path: c for path, c in interrupted.items() if path in before}


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
def test_follow_up_run_after_an_interruption_matches_a_full_build(
    before: Corpus,
    interrupted: Corpus,
    point: str,
    k: int,
    follow: str,
    tight: bool,
) -> None:
    corpus = _follow_up(follow, before, interrupted)
    # A tight registry item limit: every record a commit writes fits, but a
    # write-ahead record (stored + planned ids) may spill into overflow.
    limit = largest_lineage(before, interrupted, corpus) if tight else None
    harness = IncrementalRun(FakeDocStatusStore(max_record_ids=limit))
    harness.run(before)
    harness.run(interrupted, interrupt=point, k=k)

    harness.run(corpus)

    assert harness.state() == expected_state(corpus)
    assert harness.registry_problems(corpus) == []
    harness.run(corpus)
    assert harness.extracted == []
