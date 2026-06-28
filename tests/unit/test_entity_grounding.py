# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure entity-grounding logic (AWS-free).

Exercises the hallucination guard: an entity/relationship whose verbatim
source_text span is absent from its source chunk (the model invented it from
domain priors) is rejected, while genuinely-present spans are kept.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.entity_grounding import (
    is_grounded,
    normalize_for_grounding,
    token_overlap_ratio,
)

pytestmark = pytest.mark.unit

# A neutral synthetic contract fragment (the model "read" this chunk).
CHUNK = (
    "Section 4 Penalties. The Vendor pays the Buyer USD 1,000 per day of delay "
    "beyond the Required Delivery Date. The total penalty shall not exceed ten "
    "per cent (10%) of the Order Value."
)


class TestNormalizeForGrounding:
    def test_casefold_and_whitespace_collapse(self) -> None:
        assert normalize_for_grounding("  Required\n  Delivery ") == "required delivery"

    def test_empty(self) -> None:
        assert normalize_for_grounding(None) == ""
        assert normalize_for_grounding("") == ""

    def test_unicode_nfkc(self) -> None:
        # full-width digits fold to ascii
        assert normalize_for_grounding("２４") == "24"


class TestTokenOverlapRatio:
    def test_full_overlap(self) -> None:
        assert token_overlap_ratio("Required Delivery", CHUNK) == 1.0

    def test_no_overlap(self) -> None:
        assert token_overlap_ratio("zzz qqq", CHUNK) == 0.0

    def test_empty_span_is_zero(self) -> None:
        assert token_overlap_ratio("", CHUNK) == 0.0

    def test_partial(self) -> None:
        # 1 of 2 tokens present
        assert token_overlap_ratio("Vendor nonexistentword", CHUNK) == pytest.approx(
            0.5
        )


class TestIsGrounded:
    def test_hallucinated_span_rejected(self) -> None:
        # A plausible standard-clause hallucination: none of this is in the chunk.
        hallu = (
            "The Warranty Period from the Provisional Acceptance Date during "
            "which all supplied equipment is warranted."
        )
        assert is_grounded(hallu, CHUNK) is False

    def test_verbatim_span_grounded(self) -> None:
        real = "USD 1,000 per day of delay beyond the Required Delivery Date"
        assert is_grounded(real, CHUNK) is True

    def test_whitespace_variant_grounded(self) -> None:
        # same span, different wrapping/indentation
        real = "USD 1,000 per day of delay beyond\n   the Required Delivery Date"
        assert is_grounded(real, CHUNK) is True

    def test_no_span_is_grounded_cannot_judge(self) -> None:
        assert is_grounded(None, CHUNK) is True
        assert is_grounded("", CHUNK) is True

    def test_empty_chunk_is_grounded_cannot_judge(self) -> None:
        assert is_grounded("anything at all here", "") is True

    def test_short_span_is_grounded_below_min_tokens(self) -> None:
        # 2 tokens < default min_span_tokens(4): too short to judge -> keep
        assert is_grounded("Required Delivery", CHUNK) is True

    def test_light_paraphrase_grounded_via_overlap(self) -> None:
        # reordered/extra filler but most tokens are in the chunk
        span = "the Required Delivery Date per day of delay"
        assert is_grounded(span, CHUNK) is True

    def test_overlap_threshold_is_configurable(self) -> None:
        # 3 of 6 tokens present in the chunk (Vendor, Buyer, delay) = 0.5 overlap.
        span = "Vendor Buyer delay nonexistent fabricated invented"
        assert token_overlap_ratio(span, CHUNK) == pytest.approx(0.5)
        assert is_grounded(span, CHUNK, min_overlap_ratio=0.9) is False
        assert is_grounded(span, CHUNK, min_overlap_ratio=0.4) is True

    def test_min_span_tokens_configurable(self) -> None:
        # With min_span_tokens=1 a 2-token ungrounded span is now judged.
        assert is_grounded("fabricated clause", CHUNK, min_span_tokens=1) is False
