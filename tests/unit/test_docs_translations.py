# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each translated page and its Korean twin keep the same structure.

The pairs are ``TRANSLATED_PAGES`` in ``scripts/check_docs.py``. A pair must
have the same headings per level, fenced code blocks, table rows and set of
repository link targets; a Korean page that carries the lag note
(CONTRIBUTING.md, Documentation) is reported as lagging instead of failing.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tests.fixtures.docs_check import check_docs, git_tree

pytestmark = pytest.mark.unit

_DOCS = check_docs()


@pytest.mark.parametrize("page", _DOCS.TRANSLATED_PAGES)
def test_translated_page_matches_its_english_page(page: str) -> None:
    report = _DOCS.check_translation(page)
    assert not report.failures, "\n".join(report.failures)
    for lagging in report.lagging:
        warnings.warn(lagging, stacklevel=1)


def test_every_korean_page_is_listed() -> None:
    report = _DOCS.check_translations()
    assert not report.failures, "\n".join(report.failures)


_EN = (
    "# Guide\n> 🇰🇷 [한국어](./guide.ko.md)\n## Install\n```sh\nrun\n```\n"
    "| a | b |\n|---|---|\n| 1 | 2 |\n"
    "See [README](../README.md#why) and [design](./design.md#x)"
    " [site](https://aws.amazon.com/blogs/x).\n\nSet `units: 2` for 1,024 rows.\n"
)
_KO = (
    "# 가이드\n> 🇬🇧 [English](./guide.md)\n## 설치\n```sh\nrun\n```\n"
    "| 가 | 나 |\n|---|---|\n| 1 | 2 |\n"
    "[README](../README.ko.md#이유)와 [설계](./design.md#x)(영문)"
    " [사이트](https://aws.amazon.com/ko/blogs/x).\n\n"
    "행 1,024개에는 `units: 2`를 설정합니다.\n"
)
_PAGES = ["README", "docs/guide"]


def _check(
    root: Path, monkeypatch: pytest.MonkeyPatch, ko: str, **extra: str
) -> object:
    monkeypatch.setattr(_DOCS, "TRANSLATED_PAGES", _PAGES)
    files = {
        "README.md": "# R\n",
        "README.ko.md": "# R\n",
        "docs/guide.md": _EN,
        "docs/guide.ko.md": ko,
        "docs/design.md": "# X\n",
        **extra,
    }
    return _DOCS.check_translations(git_tree(root, files))


def test_a_matching_pair_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _check(tmp_path, monkeypatch, _KO)
    assert (report.failures, report.lagging) == ([], [])


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (("## 설치", "### 설치"), "headings per level differ"),
        (("```sh\nrun\n```\n", ""), "fenced code blocks differ"),
        (("| 1 | 2 |\n", "| 1 | 2 |\n| 1 | 2 |\n"), "table rows differ"),
        (("[README](../README.ko.md#이유)와 ", ""), "link targets differ"),
        (("./design.md#x", "./operations.md"), "link targets differ"),
        (("`units: 2`", "`units: 3`"), "inline code differ"),
        (("1,024", "1,000"), "numbers differ: EN only ['1024']"),
        (("(영문)", ""), "docs/guide.ko.md:10: link to the English-only"),
        (
            ("../README.ko.md#이유", "../README.md#why"),
            "docs/guide.ko.md:10: links README.md, which has a Korean twin",
        ),
    ],
)
def test_a_structural_difference_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit: tuple[str, str],
    message: str,
) -> None:
    report = _check(
        tmp_path, monkeypatch, _KO.replace(*edit), **{"docs/operations.md": ""}
    )
    assert len(report.failures) == 1
    prefix = message if message.startswith("docs/") else f"docs/guide: {message}"
    assert report.failures[0].startswith(prefix)
    assert report.lagging == []


def test_a_lagging_page_is_reported_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = (
        "> 이 번역은 영문판보다 늦을 수 있습니다. 최신 내용은 영문판을 확인하세요.\n\n"
    )
    report = _check(tmp_path, monkeypatch, note + _KO.replace("## 설치", "### 설치"))
    assert report.failures == []
    assert report.lagging == ["docs/guide.ko.md lags its English page"]


def test_an_unlisted_korean_page_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _check(tmp_path, monkeypatch, _KO, **{"docs/design.ko.md": "# X\n"})
    assert report.failures == [
        "docs/design.ko.md: not in TRANSLATED_PAGES (scripts/check_docs.py)"
    ]


def test_a_missing_twin_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DOCS, "TRANSLATED_PAGES", ["docs/guide"])
    report = _DOCS.check_translations(git_tree(tmp_path, {"docs/guide.md": _EN}))
    assert report.failures == ["docs/guide: missing docs/guide.ko.md"]
