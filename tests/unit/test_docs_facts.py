# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The counts and lists the docs state match the code.

A stage, a strategy or a methodology added in code without the docs (or a
sentence edited without the code) fails here: the number of ingestion stages,
the strategy tables, the default ``auto`` routable list and the number of
methodologies, wherever the tracked Markdown states them. The stage list sits
under an explicit ``<a id="pipeline-stages">`` so its links survive a count
change.
"""

from __future__ import annotations

import re

import pytest

from tests.fixtures.docs_check import REPO_ROOT, check_docs
from unified_kg_rag.domain.models.config import PipelineStageType, SearchConfig
from unified_kg_rag.domain.models.methodology import Methodology
from unified_kg_rag.domain.models.retrieval import SearchStrategy

pytestmark = pytest.mark.unit

_DOCS = check_docs()
STAGES = [stage.value for stage in PipelineStageType]
STRATEGIES = [strategy.value for strategy in SearchStrategy]
ROUTABLE = [s.value for s in SearchConfig().auto_routable_strategies]
METHODOLOGIES = [m.value for m in Methodology]
_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_KO_WORDS = {"두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6}


def _pages() -> dict[str, str]:
    """Every tracked Markdown page but the release history, whitespace
    collapsed so a statement that wraps reads as one line."""
    return {
        path.relative_to(REPO_ROOT).as_posix(): re.sub(
            r"\s+", " ", path.read_text(encoding="utf-8")
        )
        for path in _DOCS.tracked_files(REPO_ROOT)
        if path.suffix == ".md" and path.name != "CHANGELOG.md"
    }


PAGES = _pages()


def _stage_counts(text: str) -> list[int]:
    english = re.findall(
        r"\b(\d+)(?:-| )(?:checkpointed |resumable |pipeline |ingestion )?stages?\b",
        text,
    )
    korean = re.findall(r"(\d+)단계|단계 (\d+)개", text)
    return [int(n) for n in english] + [int(a or b) for a, b in korean]


def _methodology_counts(text: str) -> list[int]:
    text = text.replace("HippoRAG 2", "HippoRAG")
    english = re.findall(
        r"\b(two|three|four|five|six|seven|\d+) (?:[\w-]+ ){0,3}?(?:methodologies|methods)\b",
        text,
        flags=re.I,
    )
    korean = re.findall(r"(두|세|네|다섯|여섯) 가지 (?:\S+ ){0,3}?방법", text)
    return [int(n) if n.isdigit() else _WORDS[n.lower()] for n in english] + [
        _KO_WORDS[n] for n in korean
    ]


_STRATEGY_NAME = re.compile(r"\b(" + "|".join(STRATEGIES) + r")\b")
# Where a page states the default routable list: after the setting's name
# and a "default", or after "the router picks".
_ROUTABLE_AT = re.compile(
    r"auto_routable_strategies`?[^.;|]{0,20}?\b(?:default|기본값?)\b"
    r"|\| `search\.auto_routable_strategies` \|"
    r"|router pick\w*|라우터가 질의마다"
)
_LIST_END = re.compile(r"[);|]|\. |(?<=`) per | among |중에서| 중 ")


def _routable_lists(text: str) -> list[list[str]]:
    out = []
    for match in _ROUTABLE_AT.finditer(text):
        window = _LIST_END.split(text[match.end() : match.end() + 200], 1)[0]
        names = _STRATEGY_NAME.findall(window)
        if len(names) >= 2:
            out.append(names)
    return out


def test_the_counts_are_read_from_the_code() -> None:
    # The scans below compare against these; an empty one would pass anything.
    assert len(STAGES) >= 13 and len(STRATEGIES) >= 11 and len(METHODOLOGIES) >= 4
    assert ROUTABLE and "auto" not in ROUTABLE


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_stated_stage_count_matches_the_pipeline(page: str) -> None:
    for count in _stage_counts(PAGES[page]):
        assert count == len(STAGES), f"{page} states {count} stages, not {len(STAGES)}"


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_stated_methodology_count_matches_the_enum(page: str) -> None:
    for count in _methodology_counts(PAGES[page]):
        assert count == len(
            METHODOLOGIES
        ), f"{page} states {count} methodologies, not {len(METHODOLOGIES)}"


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_stated_auto_routable_list_matches_the_default(page: str) -> None:
    for names in _routable_lists(PAGES[page]):
        assert names == ROUTABLE, f"{page} states auto routes to {names}"


def test_the_routable_list_is_stated_somewhere() -> None:
    # The scan must find the statements it pins (README, overview, retrieval,
    # the evidence table), or the test above checks nothing.
    stated = [page for page in PAGES if _routable_lists(PAGES[page])]
    assert len(stated) >= 5, stated


@pytest.mark.parametrize(
    ("page", "heading_id"),
    [
        ("docs/guides/ingest.md", "pipeline-stages"),
        ("docs/guides/ingest.ko.md", "파이프라인-단계"),
    ],
)
def test_the_stage_list_names_every_stage_in_order(page: str, heading_id: str) -> None:
    text = (REPO_ROOT / page).read_text(encoding="utf-8")
    assert f'<a id="{heading_id}"></a>' in text
    section = text.split(f'<a id="{heading_id}"></a>', 1)[1].split("\n## ", 2)[1]
    listed = re.findall(r"^\d+\. \*\*`(\w+)`\*\*", section, flags=re.M)
    assert listed == STAGES


def test_the_design_stage_table_has_a_row_per_stage() -> None:
    text = (REPO_ROOT / "docs/design.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d+)(?:-(\d+))? \| ", text.split("### 4.1", 1)[0], re.M)
    numbers = sorted({n for a, b in rows for n in range(int(a), int(b or a) + 1)})
    assert numbers == list(range(1, len(STAGES) + 1))


@pytest.mark.parametrize(
    "page", ["docs/concepts/retrieval.md", "docs/concepts/retrieval.ko.md"]
)
def test_the_strategy_tables_list_every_strategy(page: str) -> None:
    text = (REPO_ROOT / page).read_text(encoding="utf-8")
    rows = re.findall(r"^\| `(\w+)` \|", text, flags=re.M)
    assert set(rows) >= set(STRATEGIES), set(STRATEGIES) - set(rows)
    assert set(rows) <= set(STRATEGIES), set(rows) - set(STRATEGIES)


@pytest.mark.parametrize("page", ["README.md", "README.ko.md"])
def test_the_readme_method_table_covers_every_methodology_strategy(page: str) -> None:
    text = (REPO_ROOT / page).read_text(encoding="utf-8")
    table = re.findall(r"^\| [^|`]+ \| ((?:`\w+`(?:, )?)+) \|", text, flags=re.M)
    assert len(table) == len(METHODOLOGIES)
    named = {name for row in table for name in re.findall(r"`(\w+)`", row)}
    # iterative wraps another strategy and auto routes; neither is a method's.
    assert named == set(STRATEGIES) - {"iterative", "auto"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the 13 stages and a 12-stage pipeline", [13, 12]),
        ("13단계 파이프라인, 파이프라인 단계 12개", [13, 12]),
        ("stages 5-7 and two stages", []),
    ],
)
def test_the_stage_count_scan(text: str, expected: list[int]) -> None:
    assert _stage_counts(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("four graph RAG methodologies, three published methods", [4, 3]),
        ("HippoRAG 2 and RAPTOR methodologies", []),
        ("네 가지 방법, 두 전략을 평가하는 방법", [4]),
    ],
)
def test_the_methodology_count_scan(text: str, expected: list[int]) -> None:
    assert _methodology_counts(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "picks one of `search.auto_routable_strategies` per query (default "
            "`local`, `mix`, `global`, `iterative`). Next",
            [["local", "mix", "global", "iterative"]],
        ),
        (
            "an LLM router pick `local`, `mix` or `drift` (for x)",
            [["local", "mix", "drift"]],
        ),
        (
            "| `search.auto_routable_strategies` | `local`, `mix` | evidence |",
            [["local", "mix"]],
        ),
        ("a router picks a strategy per query (`auto`)", []),
    ],
)
def test_the_routable_list_scan(text: str, expected: list[list[str]]) -> None:
    assert _routable_lists(text) == expected


def test_a_page_is_read_whole() -> None:
    # A statement wrapped over two lines is still one statement.
    assert _stage_counts(re.sub(r"\s+", " ", "a 13\nstages run")) == [13]
