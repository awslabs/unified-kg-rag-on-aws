# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ClaimResolver orchestration (AWS-free, pure domain).

Covers the class-level resolve pipeline, entity-name-map construction, claim
grouping/dedup, and the merge helpers — complements test_claim_resolution.py,
which already covers `resolve_single_claim_task` and the FuzzyMatcher guard.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aws_graphrag.domain.ingestion.claim_resolver import (
    ClaimResolutionStats,
    ClaimResolver,
)
from aws_graphrag.domain.models import Claim, Config, Entity

pytestmark = pytest.mark.unit


def _resolver() -> ClaimResolver:
    # Thread pool (not process pool) keeps the task picklable-free and in-process.
    return ClaimResolver(Config(), max_workers=2, use_process_pool=False)


def _claim(
    id_: str,
    subject: str,
    obj: str,
    *,
    type_: str = "PERFORMANCE",
    **kw: object,
) -> Claim:
    return Claim(
        id=id_,
        subject_id="",
        subject_name=subject,
        object_id="",
        object_name=obj,
        type=type_,
        **kw,
    )


class TestReductionRate:
    def test_zero_original_claims_avoids_div_by_zero(self) -> None:
        # Guard branch: no claims -> 0% rather than ZeroDivisionError.
        assert ClaimResolutionStats(original_claims=0).reduction_rate == 0.0

    def test_reduction_rate_percentage(self) -> None:
        # 10 -> 6 claims is a 40% reduction.
        stats = ClaimResolutionStats(original_claims=10, resolved_claims=6)
        assert stats.reduction_rate == pytest.approx(40.0)


class TestResolvePipeline:
    def test_empty_claims_yields_empty_result(self) -> None:
        out, stats = _resolver().resolve([], [Entity(id="e1", name="Alice")])
        assert out == []
        assert stats.original_claims == 0
        assert stats.resolved_claims == 0
        assert stats.claim_groups_created == 0

    def test_both_entities_resolve_end_to_end(self) -> None:
        ents = [Entity(id="e-alice", name="Alice"), Entity(id="e-acme", name="Acme")]
        out, stats = _resolver().resolve(
            [_claim("c1", "Alice", "Acme", type_="WORKS_AT")], ents
        )
        assert len(out) == 1
        assert out[0].subject_id == "e-alice"
        assert out[0].object_id == "e-acme"
        assert stats.resolved_claims == 1
        assert stats.unresolved_claims == 0

    def test_literal_object_kept_with_no_object_id(self) -> None:
        # Object is a literal value: subject anchors, object stays as text.
        ents = [Entity(id="e-alice", name="Alice")]
        out, _ = _resolver().resolve([_claim("c1", "Alice", "$100k revenue")], ents)
        assert len(out) == 1
        assert out[0].subject_id == "e-alice"
        assert out[0].object_id is None
        assert out[0].object_name == "$100k revenue"

    def test_unresolvable_subject_counts_as_unresolved(self) -> None:
        # Subject has no anchor entity -> claim dropped, counted unresolved.
        ents = [Entity(id="e-acme", name="Acme")]
        out, stats = _resolver().resolve(
            [_claim("c1", "totally-unknown-subject-zzz", "Acme")], ents
        )
        assert out == []
        assert stats.original_claims == 1
        assert stats.resolved_claims == 0
        assert stats.unresolved_claims == 1


class TestEntityNameMapConstruction:
    def test_blank_name_entity_skipped_from_map(self) -> None:
        # An entity whose name normalizes to empty contributes no anchor, so a
        # claim referencing only it cannot resolve.
        ents = [Entity(id="e-blank", name="   "), Entity(id="e-alice", name="Alice")]
        out, _ = _resolver().resolve([_claim("c1", "   ", "Alice")], ents)
        # Subject "   " (blank) has no anchor -> dropped.
        assert out == []

    def test_resolution_uses_original_entity_name_not_query(self) -> None:
        # Subject given lowercase; resolves to entity, subject_name normalized to
        # the original entity name "Alice".
        ents = [Entity(id="e-alice", name="Alice")]
        out, _ = _resolver().resolve([_claim("c1", "alice", "anything literal")], ents)
        assert len(out) == 1
        assert out[0].subject_id == "e-alice"
        assert out[0].subject_name == "Alice"


class TestGroupSimilarClaims:
    def test_empty_input_returns_empty(self) -> None:
        assert ClaimResolver._group_similar_claims([]) == []

    def test_same_subject_object_type_grouped_together(self) -> None:
        claims = [
            Claim(
                id="c1",
                subject_id="e1",
                subject_name="A",
                object_id="e2",
                object_name="B",
                type="WORKS_AT",
            ),
            Claim(
                id="c2",
                subject_id="e1",
                subject_name="A",
                object_id="e2",
                object_name="B",
                type="WORKS_AT",
            ),
        ]
        groups = ClaimResolver._group_similar_claims(claims)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_distinct_type_or_endpoints_stay_separate(self) -> None:
        # Grouping key is (subject_id, object_id, type): differing type or object
        # splits into separate groups.
        claims = [
            Claim(
                id="c1",
                subject_id="e1",
                subject_name="A",
                object_id="e2",
                object_name="B",
                type="WORKS_AT",
            ),
            Claim(
                id="c2",
                subject_id="e1",
                subject_name="A",
                object_id="e2",
                object_name="B",
                type="FOUNDED",
            ),
            Claim(
                id="c3",
                subject_id="e1",
                subject_name="A",
                object_id="e3",
                object_name="C",
                type="WORKS_AT",
            ),
        ]
        groups = ClaimResolver._group_similar_claims(claims)
        assert len(groups) == 3

    def test_literal_objects_with_none_id_group_by_none(self) -> None:
        # Two literal-object claims (object_id=None) of same subject+type collapse
        # into one group keyed on None.
        claims = [
            Claim(
                id="c1",
                subject_id="e1",
                subject_name="A",
                object_id=None,
                object_name="$100",
                type="FINANCIAL",
            ),
            Claim(
                id="c2",
                subject_id="e1",
                subject_name="A",
                object_id=None,
                object_name="$200",
                type="FINANCIAL",
            ),
        ]
        groups = ClaimResolver._group_similar_claims(claims)
        assert len(groups) == 1
        assert len(groups[0]) == 2


class TestMergeClaims:
    def test_single_claim_returned_as_is(self) -> None:
        # Singleton group is returned unchanged (same object identity).
        only = Claim(
            id="c1",
            subject_id="e1",
            subject_name="A",
            object_id="e2",
            object_name="B",
            type="WORKS_AT",
        )
        assert _resolver()._merge_claims([only]) is only

    def test_merge_picks_primary_identity_and_combines_fields(self) -> None:
        early = datetime(2020, 1, 1)
        late = datetime(2021, 6, 1)
        c1 = Claim(
            id="c-primary",
            short_id="1",
            subject_id="e1",
            subject_name="A",
            object_id="e2",
            object_name="B",
            type="WORKS_AT",
            status="TRUE",
            description="first",
            text_unit_ids=["t1"],
            source_text="src one",
            attributes={"k1": "v1"},
            created_at=late,
        )
        c2 = Claim(
            id="c-secondary",
            short_id="2",
            subject_id="e1",
            subject_name="A",
            object_id="e2",
            object_name="B",
            type="WORKS_AT",
            status="TRUE",
            description="second",
            text_unit_ids=["t2"],
            source_text="src two",
            attributes={"k2": "v2"},
            created_at=early,
        )
        merged = _resolver()._merge_claims([c1, c2])
        # Identity fields come from the primary (first) claim.
        assert merged.id == "c-primary"
        assert merged.short_id == "1"
        # Descriptions and source text concatenated; text_units unioned + sorted.
        assert "first" in merged.description and "second" in merged.description
        assert "src one" in merged.source_text and "src two" in merged.source_text
        assert merged.text_unit_ids == ["t1", "t2"]
        # Attributes merged; status is the most common value.
        assert merged.attributes == {"k1": "v1", "k2": "v2"}
        assert merged.status == "TRUE"
        # created_at is the earliest across the group.
        assert merged.created_at == early

    def test_merge_status_picks_majority_value(self) -> None:
        def _c(id_: str, status: str | None) -> Claim:
            return Claim(
                id=id_,
                subject_id="e1",
                subject_name="A",
                object_id="e2",
                object_name="B",
                type="WORKS_AT",
                status=status,
            )

        merged = _resolver()._merge_claims(
            [_c("c1", "TRUE"), _c("c2", "FALSE"), _c("c3", "TRUE")]
        )
        assert merged.status == "TRUE"

    def test_merge_with_all_none_optional_fields(self) -> None:
        # None description/status/text_units/source_text/attributes must not crash;
        # they collapse to empty defaults.
        def _c(id_: str) -> Claim:
            return Claim(
                id=id_,
                subject_id="e1",
                subject_name="A",
                object_id="e2",
                object_name="B",
                type="WORKS_AT",
            )

        merged = _resolver()._merge_claims([_c("c1"), _c("c2")])
        assert merged.id == "c1"
        assert merged.description == ""
        assert merged.source_text == ""
        assert merged.status == ""
        assert merged.text_unit_ids == []
        assert merged.attributes == {}


class TestResolveDedupIntegration:
    def test_duplicate_claims_merge_into_one(self) -> None:
        # Two identical resolvable claims collapse to a single merged claim, so
        # resolved_claims (post-merge) is 1 while original_claims is 2.
        ents = [Entity(id="e-alice", name="Alice"), Entity(id="e-acme", name="Acme")]
        dup = [
            _claim("c1", "Alice", "Acme", type_="WORKS_AT", description="d1"),
            _claim("c2", "Alice", "Acme", type_="WORKS_AT", description="d2"),
        ]
        out, stats = _resolver().resolve(dup, ents)
        assert len(out) == 1
        assert stats.original_claims == 2
        assert stats.resolved_claims == 1
        assert stats.claim_groups_created == 1
        assert stats.reduction_rate == pytest.approx(50.0)
