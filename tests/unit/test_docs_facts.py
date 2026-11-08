# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The counts and lists the docs state match the code.

A stage, a strategy or a methodology added in code without the docs (or a
sentence edited without the code) fails here: the number of ingestion stages,
the strategy tables, the default ``auto`` routable list, the number of
methodologies, of strategies and of CLIs, wherever the tracked Markdown states
them. The stage list sits
under an explicit ``<a id="pipeline-stages">`` so its links survive a count
change.
"""

from __future__ import annotations

import re
import tomllib

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
# The strategies a caller picks besides the auto router.
PICKABLE = [s for s in STRATEGIES if s != SearchStrategy.AUTO.value]
with (REPO_ROOT / "pyproject.toml").open("rb") as _file:
    SCRIPTS = sorted(tomllib.load(_file)["project"]["scripts"])
# Strategies no paper's methodology is: a wrapper, the router, and the graph
# query the README names after its method table.
_NOT_A_METHOD = {"iterative", "auto", "structured"}
_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
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


def _number(word: str) -> int:
    return int(word) if word.isdigit() else _WORDS[word.lower()]


_COUNT_WORD = r"(\d+|" + "|".join(_WORDS) + r")"


def _strategy_counts(text: str) -> list[int]:
    """Where a page counts the strategies one picks among: "one of eleven
    strategies", "전략 11개 중" (not "3 strategies x 4 gates")."""
    english = re.findall(
        r"\b(?:one of|all|any of) "
        + _COUNT_WORD
        + r" (?:search |retrieval )?strategies\b",
        text,
        flags=re.I,
    )
    korean = re.findall(r"전략 (\d+)개 중", text)
    return [_number(n) for n in english] + [int(n) for n in korean]


def _cli_counts(text: str) -> list[int]:
    english = re.findall(
        r"\b" + _COUNT_WORD + r" (?:CLIs|console scripts|command-line tools)\b(?!')",
        text,
        flags=re.I,
    )
    korean = re.findall(r"(?:CLI|콘솔 스크립트) (\d+)개", text)
    return [_number(n) for n in english] + [int(n) for n in korean]


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
def test_every_stated_strategy_count_matches_the_enum(page: str) -> None:
    for count in _strategy_counts(PAGES[page]):
        assert count == len(
            PICKABLE
        ), f"{page} states {count} strategies besides auto, not {len(PICKABLE)}"


def test_the_strategy_count_is_stated_somewhere() -> None:
    # The overview and its Korean edition count them; the scan must see both.
    stated = {page for page in PAGES if _strategy_counts(PAGES[page])}
    assert {"docs/concepts/overview.md", "docs/concepts/overview.ko.md"} <= stated


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_stated_cli_count_matches_the_scripts(page: str) -> None:
    for count in _cli_counts(PAGES[page]):
        assert count == len(SCRIPTS), f"{page} states {count} CLIs, not {len(SCRIPTS)}"


@pytest.mark.parametrize(
    "page", ["docs/reference/exit-codes-and-json.md", "docs/reference/cli.md"]
)
def test_the_cli_pages_name_every_console_script(page: str) -> None:
    text = PAGES[page]
    missing = [script for script in SCRIPTS if f"`{script}`" not in text]
    assert not missing, f"{page} does not name {missing}"


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
    text = (REPO_ROOT / "docs/design/ingestion.md").read_text(encoding="utf-8")
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
    # iterative wraps another strategy, auto routes and structured runs a
    # graph query: none is a paper's method, so the table leaves them out
    # and the page names them after it.
    assert named == set(STRATEGIES) - _NOT_A_METHOD
    after = text.split(table[-1], 1)[1].split("\n## ", 1)[0]
    for name in _NOT_A_METHOD:
        assert f"`{name}`" in after, f"{page} does not name `{name}` after its table"


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("runs one of eleven strategies, or the `auto` router", [11]),
        ("runs one of 10 search strategies", [10]),
        ("전략 11개 중 하나나 `auto`", [11]),
        ("3 strategies × 4 gates; 전략 3개 × 게이트 4개", []),
    ],
)
def test_the_strategy_count_scan(text: str, expected: list[int]) -> None:
    assert _strategy_counts(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the nine CLIs and 9 console scripts", [9, 9]),
        ("CLI 9개", [9]),
        ("every CLI takes it", []),
    ],
)
def test_the_cli_count_scan(text: str, expected: list[int]) -> None:
    assert _cli_counts(text) == expected


def test_a_page_is_read_whole() -> None:
    # A statement wrapped over two lines is still one statement.
    assert _stage_counts(re.sub(r"\s+", " ", "a 13\nstages run")) == [13]


def _extras(line: str) -> list[str]:
    return sorted(re.findall(r"--extra (\w+)", line))


def test_the_contributor_setup_installs_the_extras_ci_tests_with() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text()
    # The quality job's sync installs the most extras (the others test less).
    ci = max(
        (line for line in workflow.splitlines() if "run: uv sync --locked" in line),
        key=lambda line: len(_extras(line)),
    )
    guide = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    setup = [line for line in guide.splitlines() if line.startswith("uv sync ")]
    assert len(setup) == 1, setup
    assert _extras(setup[0]) == _extras(ci)
    pip = re.search(r"pip install -e '\.\[([\w,]+)\]'", guide)
    assert pip is not None
    assert sorted(pip.group(1).split(",")) == _extras(ci)
