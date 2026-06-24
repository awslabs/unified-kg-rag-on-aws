# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for the CLI argument parsing + input validation (AWS-free).

Covers only the parser construction, flag defaults/choices, and the local
validation paths (``_validate_args``, eval-data-path existence, output-format,
stage-name validation) of the five ``run-*`` entry points. The AWS-driven
``run()`` bodies (chain/pipeline/Bedrock) are deliberately NOT exercised here;
those are integration-level. Where a ``main()`` path is asserted, the chain /
pipeline / tuner construction is patched so nothing touches AWS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from aws_graphrag.application.cli import (
    run_evaluation,
    run_ingestion_pipeline,
    run_prompt_tuning,
    run_rag_chain,
    run_visualization,
)
from aws_graphrag.domain.models import SearchStrategy, SearchType

pytestmark = pytest.mark.unit


# --- run_rag_chain: parser ----------------------------------------------


def _rag_parser() -> argparse.ArgumentParser:
    return run_rag_chain.CommandLineInterface._setup_arguments()


def test_rag_parser_defaults() -> None:
    args = _rag_parser().parse_args(["--query", "hello"])
    assert args.query == "hello"
    assert args.interactive is False
    assert args.mode == "rag"
    assert args.search_strategy == "auto"
    assert args.search_type == "hybrid"
    assert args.top_k == 10
    assert args.retrieval_multiplier == 1
    assert args.output_format == "text"
    assert args.use_memory is False
    assert args.disable_query_processing is False


def test_rag_parser_short_flags_and_overrides() -> None:
    args = _rag_parser().parse_args(
        ["-q", "q", "-v", "--top-k", "5", "--output-format", "json"]
    )
    assert args.query == "q"
    assert args.verbose is True
    assert args.top_k == 5
    assert args.output_format == "json"


@pytest.mark.parametrize("strategy", [s.value for s in SearchStrategy])
def test_rag_parser_accepts_every_search_strategy(strategy: str) -> None:
    args = _rag_parser().parse_args(["-q", "q", "--search-strategy", strategy])
    assert args.search_strategy == strategy


@pytest.mark.parametrize("stype", [s.value for s in SearchType])
def test_rag_parser_accepts_every_search_type(stype: str) -> None:
    args = _rag_parser().parse_args(["-q", "q", "--search-type", stype])
    assert args.search_type == stype


def test_rag_parser_rejects_unknown_search_strategy() -> None:
    with pytest.raises(SystemExit):
        _rag_parser().parse_args(["-q", "q", "--search-strategy", "bogus"])


def test_rag_parser_rejects_unknown_output_format() -> None:
    with pytest.raises(SystemExit):
        _rag_parser().parse_args(["-q", "q", "--output-format", "xml"])


def test_rag_parser_filters_nargs() -> None:
    args = _rag_parser().parse_args(["-q", "q", "--filters", "a:1", "b:2"])
    assert args.filters == ["a:1", "b:2"]


# --- run_rag_chain: RAGChainRunner validation ----------------------------


def test_rag_runner_requires_query_or_interactive(config, mocker) -> None:
    # No --query and not --interactive -> _validate_args exits.
    mocker.patch.object(run_rag_chain, "get_config", return_value=config)
    args = _rag_parser().parse_args([])
    with pytest.raises(SystemExit) as exc:
        run_rag_chain.RAGChainRunner(args)
    assert exc.value.code == 1


def test_rag_runner_accepts_query(config, mocker) -> None:
    mocker.patch.object(run_rag_chain, "get_config", return_value=config)
    args = _rag_parser().parse_args(["-q", "hi"])
    runner = run_rag_chain.RAGChainRunner(args)  # no SystemExit
    assert runner.args.query == "hi"


def test_rag_runner_accepts_interactive(config, mocker) -> None:
    mocker.patch.object(run_rag_chain, "get_config", return_value=config)
    args = _rag_parser().parse_args(["--interactive"])
    runner = run_rag_chain.RAGChainRunner(args)
    assert runner.args.interactive is True


# --- run_rag_chain: _parse_filters --------------------------------------


def test_rag_parse_filters_key_value() -> None:
    out = run_rag_chain.RAGChainRunner._parse_filters(["entity_type:person", "x:y"])
    assert out == {"entity_type": "person", "x": "y"}


def test_rag_parse_filters_skips_malformed_and_handles_none() -> None:
    assert run_rag_chain.RAGChainRunner._parse_filters(None) == {}
    # "noColon" has no ':' -> skipped with a warning, not raised.
    assert run_rag_chain.RAGChainRunner._parse_filters(["noColon", "k:v"]) == {"k": "v"}


def test_rag_parse_filters_value_with_colon_splits_once() -> None:
    out = run_rag_chain.RAGChainRunner._parse_filters(["url:http://x:8080"])
    assert out == {"url": "http://x:8080"}


# --- run_evaluation: parser ----------------------------------------------


def _eval_parser() -> argparse.ArgumentParser:
    return run_evaluation.CommandLineInterface._setup_arguments()


def test_eval_parser_requires_eval_data_path() -> None:
    with pytest.raises(SystemExit):
        _eval_parser().parse_args([])


def test_eval_parser_defaults() -> None:
    args = _eval_parser().parse_args(["--eval-data-path", "data.json"])
    assert args.eval_data_path == Path("data.json")
    assert args.search_strategy == "auto"
    assert args.search_type == "hybrid"
    assert args.top_k == 10
    assert args.outputs_directory is None


def test_eval_parser_rejects_bad_strategy() -> None:
    with pytest.raises(SystemExit):
        _eval_parser().parse_args(["--eval-data-path", "d.json", "--search-strategy", "nope"])


def test_eval_runner_missing_file_exits(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_evaluation, "get_config", return_value=config)
    missing = tmp_path / "does-not-exist.json"
    args = _eval_parser().parse_args(["--eval-data-path", str(missing)])
    with pytest.raises(SystemExit) as exc:
        run_evaluation.EvaluationRunner(args, rag_chain=object())
    assert exc.value.code == 1


def test_eval_runner_existing_file_ok(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_evaluation, "get_config", return_value=config)
    data = tmp_path / "eval.json"
    data.write_text("[]", encoding="utf-8")
    args = _eval_parser().parse_args(["--eval-data-path", str(data)])
    runner = run_evaluation.EvaluationRunner(args, rag_chain=object())  # no exit
    assert runner.args.eval_data_path == data


# --- run_ingestion_pipeline: parser --------------------------------------


def _ing_parser() -> argparse.ArgumentParser:
    return run_ingestion_pipeline.CommandLineInterface._setup_arguments()


def test_ingestion_parser_defaults() -> None:
    args = _ing_parser().parse_args([])
    assert args.s3_prefix == "pipeline-runs"
    assert args.metrics_sink == "none"
    assert args.force_rebuild is False
    assert args.s3_sync is False


def test_ingestion_parser_enabled_stages_splits_csv() -> None:
    args = _ing_parser().parse_args(
        ["--enabled-stages", "DOCUMENT_PARSING, TEXT_CHUNKING"]
    )
    assert args.enabled_stages == ["DOCUMENT_PARSING", "TEXT_CHUNKING"]


def test_ingestion_parser_rejects_bad_metrics_sink() -> None:
    with pytest.raises(SystemExit):
        _ing_parser().parse_args(["--metrics-sink", "datadog"])


def test_ingestion_runner_requires_source_directory(config, mocker) -> None:
    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    args = _ing_parser().parse_args([])  # no source dir, no metadata op
    args.source_directory = None  # ensure env var fallback didn't set it
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert exc.value.code == 1


def test_ingestion_runner_missing_source_dir_exits(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    missing = tmp_path / "nope"
    args = _ing_parser().parse_args(["--source-directory", str(missing)])
    with pytest.raises(SystemExit):
        run_ingestion_pipeline.IngestionPipelineRunner(args)


def test_ingestion_runner_s3_sync_needs_bucket(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(["--source-directory", str(src), "--s3-sync"])
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert exc.value.code == 1


def test_ingestion_runner_resume_needs_pipeline_id(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(
        ["--source-directory", str(src), "--resume-from-stage", "TEXT_CHUNKING"]
    )
    with pytest.raises(SystemExit):
        run_ingestion_pipeline.IngestionPipelineRunner(args)


def test_ingestion_runner_metadata_op_needs_pipeline_id(config, mocker) -> None:
    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    args = _ing_parser().parse_args(["--verify-metadata"])
    with pytest.raises(SystemExit):
        run_ingestion_pipeline.IngestionPipelineRunner(args)


def test_ingestion_runner_valid_source_sets_cache_dir(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    cache = tmp_path / "mycache"
    args = _ing_parser().parse_args(
        ["--source-directory", str(src), "--cache-directory", str(cache)]
    )
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    assert runner.args.source_directory == src
    assert runner.args.cache_directory == cache
    assert cache.is_dir()  # created by _validate_args


def test_ingestion_create_pipeline_config_rejects_invalid_stage(
    config, mocker, tmp_path
) -> None:
    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(
        ["--source-directory", str(src), "--enabled-stages", "NOT_A_REAL_STAGE"]
    )
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    with pytest.raises(SystemExit) as exc:
        runner._create_pipeline_config()
    assert exc.value.code == 1


def test_ingestion_create_pipeline_config_enables_subset(
    config, mocker, tmp_path
) -> None:
    from aws_graphrag.domain.models import PipelineStageType

    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    # Use a real stage name from the enum.
    stage = next(iter(PipelineStageType)).name
    args = _ing_parser().parse_args(
        ["--source-directory", str(src), "--enabled-stages", stage]
    )
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    pc = runner._create_pipeline_config()
    enabled = {s for s, on in pc.stages_enabled.items() if on}
    assert PipelineStageType[stage] in enabled
    # Only the requested stage is on.
    assert len(enabled) == 1


def test_ingestion_create_pipeline_config_defaults_all_stages(
    config, mocker, tmp_path
) -> None:
    from aws_graphrag.domain.models import PipelineStageType

    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = _ing_parser().parse_args(["--source-directory", str(src)])
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    pc = runner._create_pipeline_config()
    assert all(pc.stages_enabled.values())
    assert set(pc.stages_enabled.keys()) == set(PipelineStageType)


def test_ingestion_build_metrics_sink(config, mocker, tmp_path) -> None:
    from aws_graphrag.shared import CloudWatchEMFSink, NullMetricsSink

    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()

    args_none = _ing_parser().parse_args(["--source-directory", str(src)])
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args_none)
    assert isinstance(runner._build_metrics_sink(), NullMetricsSink)

    args_cw = _ing_parser().parse_args(
        ["--source-directory", str(src), "--metrics-sink", "cloudwatch"]
    )
    runner_cw = run_ingestion_pipeline.IngestionPipelineRunner(args_cw)
    assert isinstance(runner_cw._build_metrics_sink(), CloudWatchEMFSink)


# --- run_prompt_tuning: parser + load_corpus_texts -----------------------


def test_prompt_tuning_parser_requires_source_dir() -> None:
    with pytest.raises(SystemExit):
        run_prompt_tuning._build_parser().parse_args([])


def test_prompt_tuning_parser_defaults() -> None:
    args = run_prompt_tuning._build_parser().parse_args(["--source-dir", "docs"])
    assert args.source_dir == Path("docs")
    assert args.output == Path("tuned_prompts.yaml")
    assert args.max_docs == 20
    assert args.config_path is None


def test_load_corpus_texts_filters_and_limits(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "c.markdown").write_text("gamma", encoding="utf-8")
    (tmp_path / "skip.pdf").write_text("ignored", encoding="utf-8")
    texts = run_prompt_tuning.load_corpus_texts(tmp_path, max_docs=10)
    assert set(texts) == {"alpha", "beta", "gamma"}


def test_load_corpus_texts_respects_max_docs(tmp_path) -> None:
    for i in range(5):
        (tmp_path / f"{i}.txt").write_text(str(i), encoding="utf-8")
    texts = run_prompt_tuning.load_corpus_texts(tmp_path, max_docs=2)
    assert len(texts) == 2


def test_prompt_tuning_main_no_texts_returns_1(config, mocker, tmp_path) -> None:
    mocker.patch.object(run_prompt_tuning, "get_config", return_value=config)
    empty = tmp_path / "empty"
    empty.mkdir()
    mocker.patch(
        "sys.argv", ["run-prompt-tuning", "--source-dir", str(empty)]
    )
    # No text files -> early return 1, tuner/AWS never constructed.
    assert run_prompt_tuning.main() == 1


# --- run_visualization: parser + helpers ---------------------------------


def test_visualization_parser_requires_data_path() -> None:
    with pytest.raises(SystemExit):
        run_visualization._build_parser().parse_args([])


def test_visualization_parser_defaults() -> None:
    args = run_visualization._build_parser().parse_args(["--data-path", "g.json"])
    assert args.data_path == Path("g.json")
    assert args.output_dir == Path("visualization_outputs")
    assert isinstance(args.renderers, list) and args.renderers  # registry default


def test_visualization_parser_multiple_renderers() -> None:
    args = run_visualization._build_parser().parse_args(
        ["--data-path", "g.json", "--renderers", "interactive", "static"]
    )
    assert args.renderers == ["interactive", "static"]


def test_visualization_hierarchy_entries_from_dict() -> None:
    data = {"communities": {"hierarchy": [{"community_id": "c1"}]}}
    assert run_visualization._hierarchy_entries(data) == [{"community_id": "c1"}]


def test_visualization_hierarchy_entries_from_list() -> None:
    data = {"communities": [{"community_id": "c2"}]}
    assert run_visualization._hierarchy_entries(data) == [{"community_id": "c2"}]


def test_visualization_hierarchy_entries_empty() -> None:
    assert run_visualization._hierarchy_entries({}) == []


def test_visualization_to_communities_infers_size_from_nodes() -> None:
    entries = [{"community_id": "c1", "level": 0, "nodes": ["a", "b", "c"]}]
    comms = run_visualization._to_communities(entries)
    assert len(comms) == 1
    assert comms[0].id == "c1"
    assert comms[0].size == 3


def test_visualization_to_hierarchical_communities() -> None:
    entries = [
        {
            "community_id": "c1",
            "level": 1,
            "nodes": ["a", "b"],
            "parent": "root",
            "children": ["c2"],
        }
    ]
    hcs = run_visualization._to_hierarchical_communities(entries)
    assert len(hcs) == 1
    assert hcs[0].community_id == "c1"
    assert hcs[0].level == 1
    assert hcs[0].nodes == {"a", "b"}
    assert hcs[0].parent_id == "root"
    assert hcs[0].children_ids == ["c2"]


def test_visualization_load_render_context_roundtrip(tmp_path) -> None:
    import json

    data = {
        "nodes": [{"id": "n1", "attributes": {"label": "x"}}, {"id": "n2"}],
        "edges": [{"source": "n1", "target": "n2", "attributes": {"w": 1}}],
        "communities": {"hierarchy": []},
        "centrality": {"n1": {"node_id": "n1", "degree": 0.5}},
        "layout": {"n1": [0.0, 0.0]},
    }
    path = tmp_path / "viz.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    ctx = run_visualization.load_render_context(path)
    assert ctx.graph.number_of_nodes() == 2
    assert ctx.graph.number_of_edges() == 1
    assert "n1" in ctx.centrality
    assert ctx.layout == {"n1": [0.0, 0.0]}
