# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pure helpers in utils/common.py."""

from __future__ import annotations

import pytest

from unified_kg_rag.shared.utils import common as common_mod
from unified_kg_rag.shared.utils.common import (
    _cgroup_cpu_quota,
    compute_hash,
    default_max_workers,
    ensure_list,
    generate_stable_id,
    normalize_name,
    parse_llm_json,
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


class TestParseLlmJson:
    def test_plain_object(self) -> None:
        assert parse_llm_json('{"a": 1}') == {"a": 1}

    def test_strips_json_code_fence(self) -> None:
        assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_strips_bare_code_fence(self) -> None:
        assert parse_llm_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_isolates_object_from_prose(self) -> None:
        assert parse_llm_json('here: {"a": 1} done') == {"a": 1}

    def test_nested_braces_span(self) -> None:
        assert parse_llm_json('{"a": {"b": 2}}') == {"a": {"b": 2}}

    def test_malformed_returns_empty(self) -> None:
        assert parse_llm_json("not json at all") == {}

    def test_no_braces_returns_empty(self) -> None:
        assert parse_llm_json("I cannot help") == {}

    def test_non_dict_returns_empty(self) -> None:
        assert parse_llm_json("[1, 2, 3]") == {}

    def test_empty_string_returns_empty(self) -> None:
        assert parse_llm_json("") == {}


class TestGenerateStableId:
    def test_is_deterministic(self) -> None:
        assert generate_stable_id("content") == generate_stable_id("content")

    def test_differs_by_content(self) -> None:
        assert generate_stable_id("a") != generate_stable_id("b")

    def test_differs_by_namespace(self) -> None:
        assert generate_stable_id("x", "ns1") != generate_stable_id("x", "ns2")


class TestCgroupCpuQuota:
    """The cgroup CPU-quota parser (fix for the 2-vCPU Fargate resolver hang).

    A mis-parse here re-introduces the original never-finishing resolver bug, so
    the cgroup-v2 (cpu.max) and cgroup-v1 (cfs_quota_us/period_us) paths, the
    ceil-division, and the 'max'/-1 unlimited handling are all pinned.
    """

    def _patch_v2(self, monkeypatch, content: str | None) -> None:
        # Simulate cgroup v2 present (cpu.max) with `content`, or absent.
        from pathlib import Path

        real_is_file = Path.is_file
        real_read_text = Path.read_text

        def fake_is_file(self):  # noqa: ANN001
            if str(self) == "/sys/fs/cgroup/cpu.max":
                return content is not None
            return real_is_file(self)

        def fake_read_text(self, *a, **k):  # noqa: ANN001
            if str(self) == "/sys/fs/cgroup/cpu.max" and content is not None:
                return content
            return real_read_text(self, *a, **k)

        monkeypatch.setattr(Path, "is_file", fake_is_file)
        monkeypatch.setattr(Path, "read_text", fake_read_text)

    def test_v2_quota_ceils_to_whole_cpus(self, monkeypatch) -> None:
        # 250000/100000 = 2.5 -> ceil -> 3
        self._patch_v2(monkeypatch, "250000 100000")
        assert _cgroup_cpu_quota() == 3

    def test_v2_exact_two_cpus(self, monkeypatch) -> None:
        self._patch_v2(monkeypatch, "200000 100000")
        assert _cgroup_cpu_quota() == 2

    def test_v2_unlimited_max_returns_none(self, monkeypatch) -> None:
        self._patch_v2(monkeypatch, "max 100000")
        assert _cgroup_cpu_quota() is None

    def test_v2_sub_one_cpu_floors_to_at_least_one(self, monkeypatch) -> None:
        # 50000/100000 = 0.5 -> ceil -> 1 (never 0).
        self._patch_v2(monkeypatch, "50000 100000")
        assert _cgroup_cpu_quota() == 1


class TestDefaultMaxWorkers:
    def test_always_at_least_one(self, monkeypatch) -> None:
        # Even if the available CPU count resolves to 1, workers >= 1.
        monkeypatch.setattr(common_mod, "_available_cpu_count", lambda: 1)
        assert default_max_workers() == 1

    def test_scales_by_fraction(self, monkeypatch) -> None:
        # 10 CPUs * 0.8 = 8.
        monkeypatch.setattr(common_mod, "_available_cpu_count", lambda: 10)
        assert default_max_workers() == 8

    def test_two_vcpu_does_not_degenerate_to_zero(self, monkeypatch) -> None:
        # The regression: int(2 * 0.8) = 1 (a single worker), never 0.
        monkeypatch.setattr(common_mod, "_available_cpu_count", lambda: 2)
        assert default_max_workers() == 1
