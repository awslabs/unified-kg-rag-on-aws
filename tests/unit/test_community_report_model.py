# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the structured CommunityReport model (MS GraphRAG parity)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from unified_kg_rag.domain.models import CommunityFinding, CommunityReport

pytestmark = pytest.mark.unit


def _report(**kw) -> CommunityReport:
    base = {"id": "r1", "community_id": "c1", "name": "Acme Cluster"}
    base.update(kw)
    return CommunityReport(**base)


def test_render_full_content_includes_summary_rating_and_findings() -> None:
    report = _report(
        summary="An executive summary.",
        rating=8.0,
        rating_explanation="High impact",
        findings=[
            CommunityFinding(summary="Head one", explanation="Detail one."),
            CommunityFinding(summary="Head two", explanation="Detail two."),
        ],
    )

    body = report.render_full_content()

    assert "# Acme Cluster" in body
    assert "8.0/10" in body
    assert "High impact" in body
    assert "An executive summary." in body
    assert "## Head one" in body
    assert "Detail two." in body


def test_render_full_content_handles_finding_without_explanation() -> None:
    report = _report(findings=[CommunityFinding(summary="Only headline")])
    body = report.render_full_content()
    assert "## Only headline" in body


def test_render_full_content_empty_when_nothing_structured() -> None:
    # No summary/rating/findings -> empty body (caller substitutes a placeholder).
    report = _report(name="")
    assert report.render_full_content() == ""


def test_rating_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        _report(rating=11.0)
    with pytest.raises(ValidationError):
        _report(rating=-1.0)


def test_findings_default_empty_and_backward_compatible() -> None:
    # A report built the old way (full_content only) still validates.
    report = _report(summary="s", full_content="legacy body")
    assert report.findings == []
    assert report.rating == 0.0
    assert report.full_content == "legacy body"
