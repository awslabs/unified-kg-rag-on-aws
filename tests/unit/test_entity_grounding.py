# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure entity-grounding logic (AWS-free).

Grounds the real hallucination observed in E2E: a "Warranty Period / 24 months /
Provisional Acceptance Date" entity that was absent from the source corpus.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.entity_grounding import (
    is_grounded,
    normalize_for_grounding,
    token_overlap_ratio,
)

pytestmark = pytest.mark.unit

# The chunk the model actually read (a Schedule G liquidated-damages fragment).
CHUNK = (
    "Schedule G Liquidated Damages. ULSD Phase: US$ 405,720 per day beyond the "
    "Required ULSD Phase Commercial Operation Date. The amount due shall not "
    "exceed fifteen per cent (15%) of the Contract Price."
)


class TestNormalizeForGrounding:
    def test_casefold_and_whitespace_collapse(self) -> None:
        assert normalize_for_grounding("  ULSD\n  Phase ") == "ulsd phase"

    def test_empty(self) -> None:
        assert normalize_for_grounding(None) == ""
        assert normalize_for_grounding("") == ""

    def test_unicode_nfkc(self) -> None:
        # full-width digits fold to ascii
        assert normalize_for_grounding("２４") == "24"


class TestTokenOverlapRatio:
    def test_full_overlap(self) -> None:
        assert token_overlap_ratio("ULSD Phase", CHUNK) == 1.0

    def test_no_overlap(self) -> None:
        assert token_overlap_ratio("zzz qqq", CHUNK) == 0.0

    def test_empty_span_is_zero(self) -> None:
        assert token_overlap_ratio("", CHUNK) == 0.0

    def test_partial(self) -> None:
        # 1 of 2 tokens present
        assert token_overlap_ratio("ULSD nonexistentword", CHUNK) == pytest.approx(0.5)


class TestIsGrounded:
    def test_hallucinated_span_rejected(self) -> None:
        # The actual hallucination: none of this appears in the chunk.
        hallu = (
            "The Warranty Period from the Provisional Acceptance Date during "
            "which all supplied equipment is warranted."
        )
        assert is_grounded(hallu, CHUNK) is False

    def test_verbatim_span_grounded(self) -> None:
        real = "US$ 405,720 per day beyond the Required ULSD Phase Commercial Operation Date"
        assert is_grounded(real, CHUNK) is True

    def test_whitespace_variant_grounded(self) -> None:
        # same span, different wrapping/indentation
        real = "US$ 405,720 per day beyond\n   the Required ULSD Phase"
        assert is_grounded(real, CHUNK) is True

    def test_no_span_is_grounded_cannot_judge(self) -> None:
        assert is_grounded(None, CHUNK) is True
        assert is_grounded("", CHUNK) is True

    def test_empty_chunk_is_grounded_cannot_judge(self) -> None:
        assert is_grounded("anything at all here", "") is True

    def test_short_span_is_grounded_below_min_tokens(self) -> None:
        # 2 tokens < default min_span_tokens(4): too short to judge -> keep
        assert is_grounded("ULSD Phase", CHUNK) is True

    def test_light_paraphrase_grounded_via_overlap(self) -> None:
        # reordered/extra filler but most tokens are in the chunk
        span = "the Required ULSD Phase Commercial Operation Date amount"
        assert is_grounded(span, CHUNK) is True

    def test_overlap_threshold_is_configurable(self) -> None:
        span = "ULSD Phase Commercial nonexistent fabricated invented"  # ~50% overlap
        assert is_grounded(span, CHUNK, min_overlap_ratio=0.9) is False
        assert is_grounded(span, CHUNK, min_overlap_ratio=0.4) is True

    def test_min_span_tokens_configurable(self) -> None:
        # With min_span_tokens=1 a 2-token ungrounded span is now judged.
        assert is_grounded("fabricated clause", CHUNK, min_span_tokens=1) is False
