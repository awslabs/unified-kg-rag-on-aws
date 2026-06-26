# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pure helpers in utils/common.py."""

from __future__ import annotations

import pytest

from unified_kg_rag.shared.utils.common import (
    compute_hash,
    ensure_list,
    generate_stable_id,
    normalize_name,
    safe_float_parse,
)

pytestmark = pytest.mark.unit


class TestComputeHash:
    def test_unsupported_algorithm_raises(self) -> None:
        # Only sha256 is supported; md5 (and others) are rejected.
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            compute_hash("x", algorithm="md5")

    def test_sha256_is_default(self) -> None:
        import hashlib

        expected = hashlib.sha256(b"x").hexdigest()[:16]
        assert compute_hash("x") == expected
        assert compute_hash("x", algorithm="sha256") == expected


class TestEnsureList:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([1, 2], [1, 2]),
            ("a", ["a"]),
            (None, []),
            (0, []),
        ],
    )
    def test_scalars_and_lists(self, value: object, expected: list) -> None:
        assert ensure_list(value) == expected

    def test_dict_with_inner_key(self) -> None:
        assert ensure_list({"items": [1, 2]}, inner_key="items") == [1, 2]

    def test_dict_without_inner_key_returns_dict_wrapped(self) -> None:
        # A non-empty dict with no inner_key is truthy and not a list -> wrapped.
        assert ensure_list({"a": 1}) == [{"a": 1}]


class TestNormalizeName:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, ""),
            ("", ""),
            ("Hello_World", "hello world"),
            ("Acme-Corp", "acme corp"),
            ("  Multiple   Spaces  ", "multiple spaces"),
            ("Special!@#Chars", "special chars"),
        ],
    )
    def test_normalization(self, value: str | None, expected: str) -> None:
        assert normalize_name(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # Non-Latin scripts MUST be preserved (entity ids hash this; the old
            # ASCII-only regex collapsed these to "" and zeroed the graph).
            ("東京電力", "東京電力"),
            ("서울특별시", "서울특별시"),
            ("Müller", "müller"),
            # NFKC normalization unifies full-width forms.
            ("Ａ１２３", "a123"),
        ],
    )
    def test_normalization_preserves_non_latin(self, value: str, expected: str) -> None:
        assert normalize_name(value) == expected

    def test_non_empty_input_never_collapses_to_empty(self) -> None:
        # A name of only punctuation/symbols falls back rather than vanishing
        # (an empty id would merge unrelated entities).
        assert normalize_name("...") != ""
        assert normalize_name("中文") != ""

    def test_distinct_scripts_do_not_collide(self) -> None:
        # Two different CJK names must map to different normalized forms.
        assert normalize_name("東京") != normalize_name("大阪")


class TestSafeFloatParse:
    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            ("1.5", None, 1.5),
            (2, None, 2.0),
            (None, 9.0, 9.0),
            ("not-a-number", 0.0, 0.0),
            (None, None, None),
        ],
    )
    def test_parsing(
        self, value: object, default: float | None, expected: float | None
    ) -> None:
        assert safe_float_parse(value, default) == expected


class TestGenerateStableId:
    def test_is_deterministic(self) -> None:
        assert generate_stable_id("content") == generate_stable_id("content")

    def test_differs_by_content(self) -> None:
        assert generate_stable_id("a") != generate_stable_id("b")

    def test_differs_by_namespace(self) -> None:
        assert generate_stable_id("x", "ns1") != generate_stable_id("x", "ns2")
