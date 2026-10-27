# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ClaimResolver orchestration (AWS-free, pure domain).

Covers the class-level resolve pipeline, entity-name-map construction, claim
grouping/dedup, and the merge helpers — complements test_claim_resolution.py,
which already covers `resolve_single_claim` and the FuzzyMatcher guard.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from unified_kg_rag.domain.ingestion.claim_resolver import (
    ClaimResolutionStats,
    ClaimResolver,
    resolved_claim_id,
)
from unified_kg_rag.domain.models import Claim, Config, Entity

pytestmark = pytest.mark.unit


def _resolver() -> ClaimResolver:
    return ClaimResolver(Config())


def _claim(
    id_: str,
    subject: str,
    obj: str,
    *,
    type_: str = "PERFORMANCE",
    **kw: object,
) -> Claim:
    kw.setdefault("text_unit_ids", ["u1"])
    return Claim(
        id=id_,
        subject_id="",
        subject_name=subject,
        object_id="",
        object_name=obj,
        type=type_,
        **kw,
    )


def _entity(id_: str, name: str, units: tuple[str, ...] = ("u1",)) -> Entity:
    """An entity stated in ``units`` (by default the one ``_claim`` cites)."""
    return Entity(id=id_, name=name, text_unit_ids=list(units))


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
        ents = [_entity("e-alice", "Alice"), _entity("e-acme", "Acme")]
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
        ents = [_entity("e-alice", "Alice")]
        out, _ = _resolver().resolve([_claim("c1", "Alice", "$100k revenue")], ents)
        assert len(out) == 1
        assert out[0].subject_id == "e-alice"
        assert out[0].object_id is None
        assert out[0].object_name == "$100k revenue"

    def test_an_entity_stated_only_in_another_unit_is_not_a_candidate(self) -> None:
        # A full build resolves a claim as an incremental run derives it:
        # against the entities its own text units stated. The subject's
        # entity stated elsewhere drops the claim; the object's leaves the
        # object a literal value.
        ents = [_entity("e-alice", "Alice"), _entity("e-acme", "Acme", ("u2",))]
        out, stats = _resolver().resolve(
            [
                _claim("c1", "Alice", "Acme", type_="WORKS_AT"),
                _claim("c2", "Acme", "Alice", type_="EMPLOYS"),
            ],
            ents,
        )
        assert [(c.subject_id, c.object_id, c.object_name) for c in out] == [
            ("e-alice", None, "Acme")
        ]
        assert stats.unresolved_claims == 1

    def test_unresolvable_subject_counts_as_unresolved(self) -> None:
        # Subject has no anchor entity -> claim dropped, counted unresolved.
        ents = [_entity("e-acme", "Acme")]
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
        ents = [_entity("e-blank", "   "), _entity("e-alice", "Alice")]
        out, _ = _resolver().resolve([_claim("c1", "   ", "Alice")], ents)
        # Subject "   " (blank) has no anchor -> dropped.
        assert out == []

    def test_resolution_uses_original_entity_name_not_query(self) -> None:
        # Subject given lowercase; resolves to entity, subject_name normalized to
        # the original entity name "Alice".
        ents = [_entity("e-alice", "Alice")]
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

    def test_literal_objects_group_by_value(self) -> None:
        # Literal-object claims (object_id=None) are keyed by the literal value,
        # so "$100" and "$200" stay distinct while a restatement merges.
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
            Claim(
                id="c3",
                subject_id="e1",
                subject_name="A",
                object_id=None,
                object_name="$100",
                type="FINANCIAL",
            ),
        ]
        groups = ClaimResolver._group_similar_claims(claims)
        assert sorted(len(g) for g in groups) == [1, 2]
        assert {c.object_name for c in max(groups, key=len)} == {"$100"}

    def test_distinct_dates_stay_separate(self) -> None:
        def _dated(id_: str, start: str | None) -> Claim:
            return Claim(
                id=id_,
                subject_id="e1",
                subject_name="Vendor",
                object_id="e2",
                object_name="Buyer",
                type="DELIVERY",
                start_date=start,
            )

        claims = [
            _dated("c1", "2024-01-01"),
            _dated("c2", "2024-06-01"),
            _dated("c3", "2024-01-01"),
            _dated("c4", None),
            _dated("c5", "UNKNOWN"),  # placeholder == no date
        ]
        groups = ClaimResolver._group_similar_claims(claims)
        assert sorted(sorted(c.id for c in g) for g in groups) == [
            ["c1", "c3"],
            ["c2"],
            ["c4", "c5"],
        ]


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

    @pytest.mark.parametrize("order", [("TRUE", "FALSE"), ("FALSE", "TRUE")])
    def test_merge_status_tie_is_the_smallest(self, order: tuple[str, str]) -> None:
        # Not the first claim's: the claims' order follows the text units,
        # which differs between namespaces of the same text.
        claims = [
            Claim(
                id=f"c{n}",
                subject_id="e1",
                subject_name="A",
                object_id="e2",
                object_name="B",
                type="WORKS_AT",
                status=status,
            )
            for n, status in enumerate(order)
        ]

        assert _resolver()._merge_claims(claims).status == "FALSE"

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
        assert merged.attributes is None


class TestResolveDedupIntegration:
    def test_duplicate_claims_merge_into_one(self) -> None:
        # Two identical resolvable claims collapse to a single merged claim, so
        # resolved_claims (post-merge) is 1 while original_claims is 2.
        ents = [_entity("e-alice", "Alice"), _entity("e-acme", "Acme")]
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


class TestResolvedClaimId:
    """The stored id is keyed on the resolved entities, not the raw names."""

    _UNITS = ("t-Acme Inc.", "t-Acme", "t-a", "t-b")
    _ENTITIES = [_entity("e-acme", "Acme", _UNITS), _entity("e-buyer", "Buyer", _UNITS)]

    def _resolve(self, *subjects: str) -> list[Claim]:
        claims = [
            _claim(
                f"raw-{subject}",
                subject,
                "Buyer",
                type_="OBLIGATION",
                text_unit_ids=[f"t-{subject}"],
            )
            for subject in subjects
        ]
        out, _ = _resolver().resolve(claims, self._ENTITIES)
        return out

    def test_spelling_variants_get_one_id_whatever_the_order(self) -> None:
        (first,) = self._resolve("Acme Inc.", "Acme")
        (second,) = self._resolve("Acme", "Acme Inc.")
        (alone,) = self._resolve("Acme Inc.")

        assert first.id == second.id == alone.id == resolved_claim_id(first)
        assert first.short_id == first.id[:8]
        assert first.subject_id == "e-acme"

    def test_type_case_and_unknown_dates_do_not_split_a_claim(self) -> None:
        base = Claim(
            id="x",
            subject_id="e-acme",
            subject_name="Acme",
            object_id="e-buyer",
            object_name="Buyer",
            type="OBLIGATION",
        )
        assert resolved_claim_id(base) == resolved_claim_id(
            base.model_copy(update={"type": "obligation", "start_date": "unknown"})
        )
        assert resolved_claim_id(base) != resolved_claim_id(
            base.model_copy(update={"start_date": "2024-01-01"})
        )

    def test_merge_unions_document_filters(self) -> None:
        claims = [
            _claim(
                f"c-{tenant}",
                "Acme",
                "Buyer",
                text_unit_ids=[f"t-{tenant}"],
                attributes={"index": "default", "filters": {"tenant": tenant}},
            )
            for tenant in ("a", "b")
        ]
        (merged,), _ = _resolver().resolve(claims, self._ENTITIES)

        assert merged.attributes == {
            "index": "default",
            "filters": {"tenant": ["a", "b"]},
        }


def test_unresolved_claims_are_not_warned_of_on_every_resolution(caplog) -> None:
    # A derivation resolves the stored statements again on every incremental
    # run, so the resolver's WARNING repeated run after run; the claim
    # resolution stage warns once, with the reason.
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import Entity, PipelineContext

    claims = [_claim("c1", "Ghost", "Buyer")]
    entities = [Entity(id="e-buyer", name="Buyer", text_unit_ids=["u1"])]
    caplog.set_level("WARNING")
    resolved, stats = _resolver().resolve(claims, entities)
    assert (resolved, stats.unresolved_claims) == ([], 1)
    assert caplog.records == []

    context = PipelineContext.model_construct(
        claims=claims, resolved_entities=entities, resolved_claims=[]
    )
    ps.ClaimResolutionStage(Config())._execute_core(context)
    # (Rendered by the structured formatter once a test configured it.)
    (record,) = caplog.records
    assert (
        "Dropped 1 of 1 claim statements: their subject is not an entity "
        "their own text units state" in record.getMessage()
    )
