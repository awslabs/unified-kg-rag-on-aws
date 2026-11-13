# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the CLIs print and accept for a person at the terminal (AWS-free):
the namespace flags of run-ingestion, run-rag's search hits and sources,
the interactive commands and the failure messages."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tests.fixtures.run_rag import answer_document, search_document
from unified_kg_rag.application.cli import run_ingestion_pipeline, run_rag_chain
from unified_kg_rag.application.presentation.answer_view import SourceView
from unified_kg_rag.domain.models import Config, Methodology

pytestmark = pytest.mark.unit


def _ing_parser() -> argparse.ArgumentParser:
    return run_ingestion_pipeline.CommandLineInterface._setup_arguments()


def _ingestion_runner(
    mocker, tmp_path: Path, config: Config, *flags: str
) -> run_ingestion_pipeline.IngestionPipelineRunner:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    source = tmp_path / "src"
    source.mkdir(exist_ok=True)
    args = _ing_parser().parse_args(
        [
            "--source-directory",
            str(source),
            "--cache-directory",
            str(tmp_path / "cache"),
            *flags,
        ]
    )
    return run_ingestion_pipeline.IngestionPipelineRunner(args)


# --- run-ingestion: namespace flags -------------------------------------


def test_ingestion_namespace_flags_override_the_config_for_the_run(
    mocker, tmp_path
) -> None:
    config = Config(processing={"document_parsing": {"index_value": "from-file"}})
    runner = _ingestion_runner(
        mocker,
        tmp_path,
        config,
        "--suffix",
        "tenant-a",
        "--methodologies",
        "lightrag, raptor",
        "--reset",
    )
    assert runner.config.processing.document_parsing.index_value == "tenant-a"
    assert runner.config.indexing.methodologies == [
        Methodology.LIGHTRAG,
        Methodology.RAPTOR,
    ]
    assert runner.config.indexing.reset is True
    # The loaded config is not changed: a later run of the same file does
    # not reset again.
    assert config.indexing.reset is False


def test_ingestion_without_namespace_flags_keeps_the_config(mocker, tmp_path) -> None:
    config = Config(processing={"document_parsing": {"index_value": "from-file"}})
    runner = _ingestion_runner(mocker, tmp_path, config)
    assert runner.config.processing.document_parsing.index_value == "from-file"
    assert runner.config.indexing.reset is False


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--methodologies", "lightrg"], "unknown methodology 'lightrg'"),
        (["--methodologies", "lightrag,,raptor"], "empty methodology name"),
        (["--suffix", "Bad Name"], "--suffix"),
    ],
    ids=["unknown-methodology", "empty-name", "bad-suffix"],
)
def test_ingestion_namespace_flag_errors_are_usage_errors(
    mocker, tmp_path, capsys, flags: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as exited:
        _ingestion_runner(mocker, tmp_path, Config(), *flags)
    assert exited.value.code == 2
    assert message in capsys.readouterr().err


def test_ingestion_refuses_a_listed_stage_no_methodology_needs(
    mocker, tmp_path, capsys
) -> None:
    runner = _ingestion_runner(
        mocker, tmp_path, Config(), "--enabled-stages", "summary_tree"
    )
    with pytest.raises(SystemExit) as exited:
        runner._create_pipeline_config()
    assert exited.value.code == 2
    assert "--methodologies graphrag,lightrag,raptor" in capsys.readouterr().err
    # Selected, the same list runs.
    runner = _ingestion_runner(
        mocker,
        tmp_path,
        Config(),
        "--enabled-stages",
        "summary_tree",
        "--methodologies",
        "graphrag,lightrag,raptor",
    )
    stages = runner._create_pipeline_config().stages_enabled
    assert [s.value for s, on in stages.items() if on] == ["summary_tree"]


def test_ingestion_banner_names_the_namespace_as_run_rag_does(mocker, tmp_path) -> None:
    config = Config(indexing={"additional_suffix": "v2"})
    runner = _ingestion_runner(mocker, tmp_path, config, "--suffix", "tenant-a")
    printed = mocker.patch.object(run_ingestion_pipeline.console, "print")
    runner._display_run_info()
    (panel,), _ = printed.call_args
    text = str(panel.renderable)
    assert "Index Namespace:[/dim] 'tenant-a-v2'" in text
    assert "Methodologies:[/dim] graphrag, lightrag" in text


# --- run-rag: hits and sources ------------------------------------------

_TEXT_HIT = {
    "content": "Acme Robotics is a manufacturer\nof warehouse robots.",
    "score": 0.5,
    "source": "tu-1",
    "retriever_type": "text",
    "metadata": {
        "id": "tu-1",
        "section_type": "text",
        "attributes": {"file_name": "acme.txt", "file_path": "corpus/acme.txt"},
    },
}
_SECOND_CHUNK = {
    "content": "Second chunk.",
    "score": 0.4,
    "source": "tu-2",
    "retriever_type": "text",
    "metadata": {
        "section_type": "text",
        "attributes": {
            "file_path": "corpus/globex.txt",
            "chunk_id": 2,
            "total_chunks": 3,
        },
    },
}
_RELATIONSHIP_HIT = {
    "content": "Relationship: Acme Robotics -[SUPPLIES]-> Globex Sensors",
    "score": 0.25,
    "source": "rel-1",
    "retriever_type": "relationship",
    "metadata": {
        "section_type": "relationship",
        "source_name": "Acme Robotics",
        "target_name": "Globex Sensors",
    },
}
_ENTITY_SOURCE = {
    "content": "Entity: Acme Robotics",
    "score": 0.2,
    "source": "ent-1",
    "metadata": {"name": "Acme Robotics", "section_type": "entity"},
}


@pytest.fixture
def printed(monkeypatch: pytest.MonkeyPatch):
    """What run-rag prints, as plain text."""
    import io

    from rich.console import Console

    out = io.StringIO()
    monkeypatch.setattr(
        run_rag_chain, "console", Console(file=out, width=200, color_system=None)
    )
    return out


def test_search_mode_prints_each_hit(printed) -> None:
    # --mode search printed a header and nothing else without --verbose.
    run_rag_chain.RAGChainRunner._print_result(
        search_document([_TEXT_HIT, _SECOND_CHUNK, _RELATIONSHIP_HIT])
    )
    text = printed.getvalue()
    assert "Search Results (3)" in text
    assert "1. [text] acme.txt  (score 0.500)" in text
    # The snippet is the text on one line.
    assert "Acme Robotics is a manufacturer of warehouse robots." in text
    assert "2. [text] globex.txt (chunk 2/3)  (score 0.400)" in text
    assert "3. [relationship] Acme Robotics -> Globex Sensors  (score 0.250)" in text


def test_search_mode_shows_the_first_hits_unless_verbose(printed, monkeypatch) -> None:
    monkeypatch.setattr(run_rag_chain, "_HITS_SHOWN", 2)
    hits = search_document([_TEXT_HIT] * 3)
    run_rag_chain.RAGChainRunner._print_result(hits)
    assert "... and 1 more" in printed.getvalue()
    assert "3. [text]" not in printed.getvalue()
    run_rag_chain.RAGChainRunner._print_result(hits, verbose=True)
    assert "3. [text] acme.txt" in printed.getvalue()


def test_an_answer_prints_a_short_sources_line_by_default(printed) -> None:
    sources = [_TEXT_HIT, _SECOND_CHUNK, _TEXT_HIT, _RELATIONSHIP_HIT, _ENTITY_SOURCE]
    run_rag_chain.RAGChainRunner._print_result(
        answer_document("Acme supplies.", sources=sources)
    )
    text = printed.getvalue()
    assert (
        "Sources (5): files: acme.txt, globex.txt; 1 relationship, 1 entity "
        "(--verbose lists each)" in text
    )
    # Not the per-source list.
    assert "(score" not in text


def test_a_verbose_answer_lists_each_source_with_its_text(printed) -> None:
    run_rag_chain.RAGChainRunner._print_result(
        answer_document("Acme supplies.", sources=[_TEXT_HIT, _ENTITY_SOURCE]),
        verbose=True,
    )
    text = printed.getvalue()
    assert "Reference Sources (2)" in text
    assert "1. [text] acme.txt  (score 0.500)" in text
    assert "2. [entity] Acme Robotics  (score 0.200)" in text
    assert "Sources (2):" not in text


def test_the_sources_line_counts_many_files_and_types(printed) -> None:
    sources = [
        {**_TEXT_HIT, "metadata": {"attributes": {"file_name": f"f{i}.txt"}}}
        for i in range(7)
    ] + [_ENTITY_SOURCE, _ENTITY_SOURCE]
    views = [SourceView.of_source(i, s) for i, s in enumerate(sources, 1)]  # type: ignore[arg-type]
    line = run_rag_chain._sources_line(views)
    assert line.startswith("Sources (9): files: f0.txt, f1.txt, f2.txt, f3.txt, f4.txt")
    assert "and 2 more; 2 entities" in line


def _json_runner(*flags: str) -> run_rag_chain.RAGChainRunner:
    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.args = run_rag_chain.CommandLineInterface._setup_arguments().parse_args(
        ["-q", "hi", "--output-format", "json", *flags]
    )
    return runner


def test_the_json_answer_does_not_repeat_its_sources() -> None:
    result = {
        "success": True,
        "answer": "a",
        "sources": [_TEXT_HIT],
        "search_results": {"results": [_TEXT_HIT], "total_results": 1},
    }
    document = _json_runner()._json_document(result)
    assert document["sources"] == [_TEXT_HIT]
    assert document["search_results"] == {"total_results": 1}
    # --verbose keeps every retrieved result; the result itself is unchanged.
    assert _json_runner("--verbose")._json_document(result) == result
    assert result["search_results"]["results"] == [_TEXT_HIT]
    # --mode search has no sources: its results are the result.
    search = {"success": True, "search_results": {"results": [_TEXT_HIT]}}
    assert _json_runner("--mode", "search")._json_document(search) == search


# --- run-rag --interactive: the commands --------------------------------


def _interactive_runner(mocker, lines: list[str], *flags: str):
    """A runner whose prompt reads ``lines`` (then EOF) and whose queries
    are recorded, not run."""
    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.args = run_rag_chain.CommandLineInterface._setup_arguments().parse_args(
        ["--interactive", *flags]
    )
    runner.config = Config()
    runner.rag_chain = mocker.MagicMock()
    runner.rag_chain.acheck_methodology = mocker.AsyncMock()
    runner.interactive_commands = {
        "help": runner._handle_help,
        "new": runner._handle_new_conversation,
        "set-filter": runner._handle_set_filter,
        "clear-filters": runner._handle_clear_filters,
        "show-config": runner._handle_show_config,
    }
    mocker.patch.object(
        run_rag_chain.console, "input", side_effect=[*lines, EOFError()]
    )
    queries: list = []

    async def run_query(rag_input):
        queries.append(rag_input)
        return {"success": True, "answer": "ok", "conversation_id": "c-1"}

    runner._run_query = run_query  # type: ignore[method-assign]
    runner._print_result = mocker.MagicMock()  # type: ignore[method-assign]
    return runner, queries


async def test_interactive_help_is_a_command_only_as_the_whole_line(
    mocker, printed
) -> None:
    runner, queries = _interactive_runner(mocker, ["help", "help me find Acme"])
    await runner._interactive_mode()
    assert "set-filter <k:v>" in printed.getvalue()
    # "help me ..." is a question.
    assert [q.query for q in queries] == ["help me find Acme"]


async def test_interactive_filters_reach_the_next_questions(mocker, printed) -> None:
    runner, queries = _interactive_runner(
        mocker,
        [
            "set-filter category:contract",
            "set-filter no-colon",
            "show-config",
            "first question",
            "clear-filters",
            "second question",
        ],
    )
    await runner._interactive_mode()
    text = printed.getvalue()
    assert "Filter added: 'category' = 'contract'" in text
    assert "Invalid filter format" in text
    assert "Filters: category:contract" in text
    assert "All filters cleared." in text
    assert queries[0].filters == {"category": "contract"}
    assert not queries[1].filters
    # Every question of the session keeps memory on.
    assert all(q.use_memory for q in queries)


async def test_interactive_new_starts_another_conversation(mocker, printed) -> None:
    runner, queries = _interactive_runner(
        mocker, ["first", "new", "second"], "--conversation-id", "mine"
    )
    await runner._interactive_mode()
    assert queries[0].conversation_id == "mine"
    assert queries[1].conversation_id not in ("mine", "c-1")
    assert "Starting new conversation" in printed.getvalue()


async def test_interactive_quit_and_eof_end_the_session(mocker, printed) -> None:
    runner, queries = _interactive_runner(mocker, ["  ", "QUIT", "never asked"])
    await runner._interactive_mode()
    assert queries == []
    assert "Ending conversation." in printed.getvalue()
    # EOF (Ctrl-D, a piped stdin) ends it too instead of spinning.
    runner, queries = _interactive_runner(mocker, [])
    await runner._interactive_mode()
    assert "Conversation interrupted." in printed.getvalue()


async def test_interactive_answers_print_details_only_with_verbose(mocker) -> None:
    runner, _ = _interactive_runner(mocker, ["question"])
    await runner._interactive_mode()
    runner._print_result.assert_called_once_with(
        {"success": True, "answer": "ok", "conversation_id": "c-1"}, verbose=False
    )
    runner, _ = _interactive_runner(mocker, ["question"], "--verbose")
    await runner._interactive_mode()
    assert runner._print_result.call_args.kwargs == {"verbose": True}


async def test_interactive_a_failing_question_does_not_end_the_session(
    mocker, printed
) -> None:
    runner, queries = _interactive_runner(mocker, ["boom", "after"])
    calls = 0

    async def run_query(rag_input):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("model down")
        queries.append(rag_input)
        return {"success": True, "answer": "ok"}

    runner._run_query = run_query  # type: ignore[method-assign]
    await runner._interactive_mode()
    assert "Unexpected error: model down" in printed.getvalue()
    assert [q.query for q in queries] == ["after"]


# --- run-ingestion: failure output and the embedding preflight ----------


class _Embeddings:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dimension


class _Factory:
    """An embedding model that reports no dimensions (as a local server's)
    and returns ``returns``-long vectors whatever it is asked for."""

    def __init__(self, returns: int) -> None:
        self.returns = returns
        self.requested: list[int] = []

    def get_model(self, model_id, **kwargs):
        self.requested.append(kwargs["dimensions"])
        return _Embeddings(self.returns)

    def get_model_info(self, model_id):
        return type("Info", (), {"dimensions": None})()


def _context():
    from datetime import datetime

    from unified_kg_rag.domain.models import (
        PipelineConfig,
        PipelineContext,
        PipelineStageStatus,
    )

    return PipelineContext(
        pipeline_id="run-7",
        config=PipelineConfig(),
        status=PipelineStageStatus.RUNNING,
        start_time=datetime.now(),
        source_directory=Path("src"),
    )


def _probe(returns: int, dimension: int):
    from unittest.mock import MagicMock

    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.config = Config(
        backend="local",
        models={
            "default": "llm",
            "embedding": {"model_id": "embedder", "dimension": dimension},
        },
    )
    pipeline.providers = MagicMock(embedding_factory=_Factory(returns))
    return pipeline


def test_the_preflight_refuses_an_embedding_dimension_the_index_cannot_take() -> None:
    # It failed every vector write at the indexing stage, after the
    # extraction was paid for, without naming the setting.
    from unified_kg_rag.shared import ConfigurationError

    with pytest.raises(ConfigurationError) as refused:
        _probe(returns=64, dimension=1024)._check_embedding_dimension(_context())
    message = str(refused.value)
    assert "returns vectors of dimension 64" in message
    assert "mapped with 1024 (models.embedding.dimension)" in message
    assert "Set models.embedding.dimension (--embedding-dimension) to 64" in message
    pipeline = _probe(returns=1024, dimension=1024)
    pipeline._check_embedding_dimension(_context())
    assert pipeline.providers.embedding_factory.requested == [1024]
    # An incremental run with nothing to (re)extract embeds nothing new.
    from unified_kg_rag.domain.models import DocumentDelta

    unchanged = _context()
    unchanged.incremental_delta = DocumentDelta()
    _probe(returns=64, dimension=1024)._check_embedding_dimension(unchanged)


def test_a_failed_run_names_the_first_stage_error_and_the_next_step() -> None:
    from datetime import datetime

    from unified_kg_rag.domain.models import (
        PipelineConfig,
        PipelineContext,
        PipelineStageResult,
        PipelineStageStatus,
    )

    runner = run_ingestion_pipeline.IngestionPipelineRunner
    context = PipelineContext(
        pipeline_id="run-7",
        config=PipelineConfig(),
        status=PipelineStageStatus.FAILED,
        start_time=datetime.now(),
        source_directory=Path("src"),
        stage_results=[
            PipelineStageResult(
                stage_name="graph_extraction",
                status=PipelineStageStatus.COMPLETED,
                start_time=datetime.now(),
            ),
            PipelineStageResult(
                stage_name="indexing",
                status=PipelineStageStatus.FAILED,
                start_time=datetime.now(),
                error_message="Vector dimension mismatch. Expected: 1024, Given: 64",
            ),
        ],
    )
    assert runner._stage_failure(context) == (
        "Pipeline failed at stage(s): indexing. First error (indexing): Vector "
        "dimension mismatch. Expected: 1024, Given: 64"
    )
    step = runner._next_step(context)
    assert "set models.embedding.dimension (--embedding-dimension)" in step
    assert "rerun with --pipeline-id run-7" in step
    context.stage_results[1].error_message = "throttled"
    assert "models.embedding" not in runner._next_step(context)


# --- a suffix nothing was ingested under --------------------------------


@pytest.mark.parametrize(
    ("additional", "expected"),
    [(None, "default, tenant-a, tenant-a-v2"), ("v2", "tenant-a")],
)
async def test_a_typod_suffix_names_the_ingested_ones(
    additional: str | None, expected: str
) -> None:
    from unified_kg_rag.adapters.retrievers.opensearch_retriever import (
        OpenSearchRetriever,
    )
    from unified_kg_rag.shared import IndexNotFoundError

    config = Config(indexing={"additional_suffix": additional})
    prefix = config.indexing.opensearch.text_units_index_prefix

    class _Client:
        async def aindex_exists(self, name: str) -> bool:
            return False

        def get_aliases_by_index(self, pattern: str) -> dict[str, list[str]]:
            assert pattern == f"{prefix}-*"
            return {
                f"{prefix}-default-1": [f"{prefix}-default"],
                f"{prefix}-tenant-a-2": [f"{prefix}-tenant-a"],
                f"{prefix}-tenant-a-v2-3": [f"{prefix}-tenant-a-v2"],
                f"{prefix}-stale-4": [],
            }

    retriever = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(retriever, "_config", config)
    object.__setattr__(retriever, "_opensearch_config", config.indexing.opensearch)
    object.__setattr__(retriever, "_opensearch_client", _Client())
    with pytest.raises(IndexNotFoundError) as refused:
        await retriever._require_ingested_suffix("tenant-b", RuntimeError("404"))
    # Data, not message: the message reaches logs, error answers and callers
    # that may not read the other namespaces.
    assert refused.value.ingested == expected.split(", ")
    assert "tenant-a" not in str(refused.value)
    assert refused.value.described() == (
        f"{refused.value} Suffixes ingested on this store: {expected}."
    )
    # A surface serving some namespaces names only those.
    assert refused.value.described(["tenant-a", "other"]).endswith(
        "Namespaces served here that hold an index: tenant-a."
    )
    assert refused.value.described(["other"]).endswith(
        "No namespace served here holds an index."
    )


async def test_run_rag_exits_2_for_a_suffix_nothing_was_ingested_under(
    mocker, capsys
) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import RAGInput
    from unified_kg_rag.shared import IndexNotFoundError

    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.args = run_rag_chain.CommandLineInterface._setup_arguments().parse_args(
        ["-q", "hi", "--suffix", "tenant-b"]
    )
    runner.metrics_sink = mocker.MagicMock()
    runner.rag_chain = mocker.MagicMock()
    runner.rag_chain.ainvoke = mocker.AsyncMock(
        side_effect=IndexNotFoundError("No indices found for suffix 'tenant-b'")
    )
    with pytest.raises(SystemExit) as exited:
        await runner._run_query(RAGInput(query="hi"))
    assert exited.value.code == 2
    assert "No indices found for suffix 'tenant-b'" in capsys.readouterr().err


async def test_run_rag_names_the_ingested_suffixes_of_a_typod_one(
    mocker, capsys
) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import RAGInput
    from unified_kg_rag.shared import IndexNotFoundError

    runner = object.__new__(run_rag_chain.RAGChainRunner)
    runner.args = run_rag_chain.CommandLineInterface._setup_arguments().parse_args(
        ["-q", "hi", "--suffix", "tenant-b"]
    )
    runner.metrics_sink = mocker.MagicMock()
    runner.rag_chain = mocker.MagicMock()
    runner.rag_chain.ainvoke = mocker.AsyncMock(
        side_effect=IndexNotFoundError("No index.", ingested=["tenant-a"])
    )
    with pytest.raises(SystemExit):
        await runner._run_query(RAGInput(query="hi"))
    assert "Suffixes ingested on this store: tenant-a." in capsys.readouterr().err


# --- run-eval: tables -----------------------------------------------------


def test_comparison_and_statistics_tables_never_cut_a_metric_name(
    monkeypatch,
) -> None:
    # On an 80-column terminal "answer_match/answer_contains" printed as
    # "answer_match/answer_contai…".
    import io
    from datetime import datetime

    from rich.console import Console

    from unified_kg_rag.application.cli import run_evaluation
    from unified_kg_rag.domain.models import EvaluationSummary

    out = io.StringIO()
    monkeypatch.setattr(
        run_evaluation, "console", Console(file=out, width=80, color_system=None)
    )
    name = "answer_match/answer_contains"
    delta = {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "pairs": 2}
    comparison = {
        "baseline": "local",
        "strategies": {
            s: {"metrics": {name: {"mean": 0.5}}, "routed_strategies": {}}
            for s in ("local", "mix", "hybrid")
        },
        "deltas_vs_baseline": {"mix": {name: delta}, "hybrid": {name: delta}},
    }
    run_evaluation.EvaluationRunner._print_comparison(comparison, 1.0)
    stats = {"mean": 0.5, "median": 0.5, "stdev": 0.0, "min": 0.5, "max": 0.5}
    summary = EvaluationSummary(
        total_queries=2,
        successful_evaluations=2,
        failed_evaluations=0,
        metric_statistics={name: {**stats, "count": 2}},
        evaluation_start_time=datetime(2026, 1, 1),
        evaluation_end_time=datetime(2026, 1, 1),
    )
    run_evaluation.EvaluationRunner._print_summary(summary, 1.0)
    text = out.getvalue()
    assert text.count(name) == 2
    assert "…" not in text


# --- --help loads no chain or pipeline ---------------------------------


def test_help_loads_neither_the_chain_nor_the_pipeline() -> None:
    # run-rag and run-ingestion --help took 4 s of CPU (of 11-14 s on a busy
    # host) importing LangChain, the strategies and the stores they never
    # use; a test still patching the names keeps working (lazy.lazy_names).
    import subprocess
    import sys

    script = (
        "import sys\n"
        "from unified_kg_rag.application.cli import (\n"
        "    run_ingestion_pipeline, run_prompt_tuning, run_rag_chain)\n"
        "run_rag_chain.CommandLineInterface._setup_arguments().format_help()\n"
        "run_ingestion_pipeline.CommandLineInterface._setup_arguments().format_help()\n"
        "run_prompt_tuning._build_parser().format_help()\n"
        "heavy = ['unified_kg_rag.application.retrieval.rag_chain',\n"
        "         'unified_kg_rag.application.ingestion.pipeline',\n"
        "         'unified_kg_rag.application.prompts.tuner', 'langchain_core']\n"
        "print([m for m in heavy if m in sys.modules])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=100
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
    # The names stay attributes of the modules, loaded on first access.
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    assert run_ingestion_pipeline.DataIngestionPipeline is DataIngestionPipeline
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        run_ingestion_pipeline.nope  # noqa: B018
