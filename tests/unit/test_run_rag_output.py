# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""run-rag prints model output and errors as plain text (AWS-free).

Regression: the answer went through rich markup, so ``[link=...]`` in model
output became an OSC-8 terminal hyperlink and raw ESC bytes reached the
terminal. Answers, errors and the verbose panels now print literally with
control characters (other than newline and tab) removed.
"""

from __future__ import annotations

import io
import re

import pytest
from rich.console import Console

from unified_kg_rag.application.cli import run_rag_chain
from unified_kg_rag.application.cli.run_rag_chain import RAGChainRunner

pytestmark = pytest.mark.unit

_HOSTILE = (
    "See [link=https://evil.example]here[/link] \x1b]0;pwned\x07\x1b[2J\x9b31m done"
)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    out = io.StringIO()
    monkeypatch.setattr(
        run_rag_chain,
        "console",
        Console(file=out, force_terminal=True, color_system="truecolor", width=300),
    )
    return out


def _assert_inert(rendered: str) -> None:
    # Rich's own styling (SGR "ESC[...m") is expected; strip it to see the text.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    assert "\x1b]8;" not in rendered  # no OSC-8 hyperlink
    assert "pwned" in rendered  # text kept...
    assert "\x1b]0;" not in rendered  # ...but its escape sequences removed
    assert "\x1b[2J" not in rendered
    assert "\x9b" not in rendered and "\x07" not in rendered
    assert "[link=https://evil.example]here[/link]" in plain


def test_answer_is_printed_literally(terminal: io.StringIO) -> None:
    RAGChainRunner._print_result(
        {"success": True, "answer": _HOSTILE + "\nline two\tcol"}
    )
    rendered = terminal.getvalue()
    _assert_inert(rendered)
    assert "line two" in rendered


def test_error_is_printed_literally(terminal: io.StringIO) -> None:
    RAGChainRunner._print_result({"success": False, "error": _HOSTILE})
    _assert_inert(terminal.getvalue())


def test_verbose_panels_print_model_output_literally(terminal: io.StringIO) -> None:
    RAGChainRunner._print_result(
        {
            "success": True,
            "answer": "ok",
            "processed_query": {
                "original_query": "q",
                "translated_query": _HOSTILE,
                "entities": ["[link=https://evil.example]here[/link]"],
            },
            "sources": [{"source": _HOSTILE, "score": 0.5}],
        },
        verbose=True,
    )
    rendered = terminal.getvalue()
    assert "\x1b]8;" not in rendered
    assert "\x1b]0;" not in rendered and "\x9b" not in rendered


def test_terminal_safe_keeps_newline_and_tab() -> None:
    assert run_rag_chain.terminal_safe("a\nb\tc\x1b\x00\x7f\x85d") == "a\nb\tcd"
