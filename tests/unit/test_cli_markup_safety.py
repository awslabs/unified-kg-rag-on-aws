# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data printed by the CLIs is never read as rich markup (AWS-free).

Regression: an entity name, a community title, a query or an error holding
``[/red]`` or ``[/x]`` went into a markup string, and rich raised
``MarkupError`` (a closing tag no open tag matches), crashing the CLI after
the run or while printing its error.
"""

from __future__ import annotations

import argparse

import pytest

from unified_kg_rag.application.cli import (
    run_evaluation,
    run_ingestion_pipeline,
    run_rag_chain,
)
from unified_kg_rag.domain.models import Claim, Community, Entity, Relationship
from unified_kg_rag.shared.utils import display

pytestmark = pytest.mark.unit

_NAME = "[/red]Vendor[bold]"
_QUERY = "[/x] what does Vendor ship?"


def test_sample_tables_and_communities_print_markup_in_data_literally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    entity = Entity(id="e1", name=_NAME, type="[/x]", description="[/red] desc")
    display.display_sample_entities([entity])
    display.display_sample_relationships(
        [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                source_name=_NAME,
                target_name="[/y]Buyer",
                description="[/red] ships",
            )
        ]
    )
    display.display_sample_claims(
        [
            Claim(
                id="c1",
                subject_id="e1",
                subject_name=_NAME,
                object_name="[/z]",
                type="[/red]",
                description="[/red] pays",
            )
        ]
    )
    display.display_communities(
        [
            Community(
                id="c1",
                name="[/red] Widget supply",
                level="0",
                parent="",
                children=[],
                entity_ids=["e1"],
                size=1,
            )
        ],
        None,
        [entity],
        None,
    )

    out = capsys.readouterr().out
    assert "[/red]Vendor[bold]" in out
    assert "[/y]Buyer" in out
    assert "[/red] Widget supply" in out


def test_run_rag_banner_prints_the_query_literally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.args = argparse.Namespace(
        mode="rag",
        search_strategy="local",
        search_type="hybrid",
        top_k=5,
        retrieval_multiplier=2,
        use_memory=False,
        interactive=False,
        disable_query_processing=False,
        verbose=False,
        suffix="[/x]",
        filters=["k:[/red]"],
        conversation_id="[/c]",
        query=_QUERY,
    )

    runner._display_run_info()

    out = capsys.readouterr().out
    assert _QUERY in out
    assert "k:[/red]" in out


def test_run_ingestion_prints_an_unexpected_error_literally(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"boom on {_NAME}")

    monkeypatch.setattr(run_ingestion_pipeline, "IngestionPipelineRunner", fail)
    monkeypatch.setattr("sys.argv", ["run-ingestion", "--source-directory", "."])

    with pytest.raises(SystemExit) as exit_info:
        run_ingestion_pipeline.main()

    assert exit_info.value.code == 1
    assert f"boom on {_NAME}" in capsys.readouterr().out


def test_run_eval_prints_an_unexpected_error_literally(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"boom on {_QUERY}")

    monkeypatch.setattr(run_evaluation, "load_cli_config", fail)
    monkeypatch.setattr("sys.argv", ["run-eval", "--data-path", "eval.json"])

    with pytest.raises(SystemExit) as exit_info:
        run_evaluation.main()

    assert exit_info.value.code == 1
    assert f"boom on {_QUERY}" in capsys.readouterr().out
