# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed graphs at start-up: ``processing.seed_graphs`` (validation, paths,
the environment), the seed source registry, the built-in ``jsonl`` and
``csv`` sources on local files, the preflight check of a kind's
requirement, how converted seeds key the stage caches, and
``application.bootstrap.load_seed_graphs``. S3 is in
``test_seed_sources_s3.py``."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import unified_kg_rag.adapters.seed_sources.jsonl as jsonl
import unified_kg_rag.adapters.seed_sources.property_graph as property_graph
import unified_kg_rag.adapters.seed_sources.registry as seed_registry
from tests.fixtures.extras import requires_ontology
from tests.fixtures.seed_graphs import (
    EDGES,
    EDGES_CSV,
    EX,
    NODES,
    NODES_CSV,
    PROCURE,
    lines,
)
from unified_kg_rag.adapters.seed_sources import (
    SeedSourceContext,
    SeedSourceOptions,
    SeedTerms,
    get_seed_source,
    register_seed_source,
    registered_seed_sources,
    seed_source_requirement,
)
from unified_kg_rag.application.bootstrap import (
    load_seed_graphs,
    resolve_extraction_schema,
    seed_conversion_settings,
)
from unified_kg_rag.application.cli.preflight import missing_extras_error
from unified_kg_rag.application.config import ConfigLoader
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.ingestion.cache_keys import (
    CONFIG_KEYING,
    Unkeyed,
    classify,
    extraction_fingerprint,
    extraction_settings,
    seed_corpus_fingerprint,
    shapes_answers,
    stage_input_fingerprint,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    SEED_FILTER_KEY,
    SeedConversion,
    SeedConversionSettings,
)
from unified_kg_rag.domain.models import Config, PipelineStageType
from unified_kg_rag.domain.models.config import SEED_GRAPH_FILTER_KEY, SeedGraphConfig
from unified_kg_rag.domain.models.seed_graph import SeedGraph
from unified_kg_rag.shared.exceptions import ConfigurationError, SeedGraphError

pytestmark = pytest.mark.unit

ONTOLOGY = Path(__file__).parents[1] / "fixtures" / "ontologies" / "procure.ttl"


def _write(path: Path, text: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _seed(path: Path | str, kind: str = "jsonl", **settings: Any) -> SeedGraphConfig:
    return SeedGraphConfig(name="procure", kind=kind, path=str(path), **settings)


def _load(path: Path | str, kind: str = "jsonl", **settings: Any) -> SeedGraph:
    seed = _seed(path, kind, **settings)
    return get_seed_source(kind)(SeedSourceContext(seed, Config()))


def _shape(graph: SeedGraph) -> tuple[Any, Any]:
    """The graph without its load report."""
    return graph.nodes, graph.edges


def _text(conversion: SeedConversion) -> str:
    """Every shard document's text."""
    return "\n".join(shard.page_content for shard in conversion.documents_content)


def _config(*seeds: dict[str, Any], **extraction: Any) -> Config:
    return Config.model_validate(
        {
            "processing": {
                "seed_graphs": list(seeds),
                **({"graph_extraction": extraction} if extraction else {}),
            }
        }
    )


@pytest.fixture
def procure_jsonl(tmp_path: Path) -> Path:
    return _write(tmp_path / "procure.jsonl", lines(*NODES, *EDGES))


@pytest.fixture
def custom_kind() -> Iterator[str]:
    kind = "synthetic_service"
    yield kind
    seed_registry._REGISTRY.pop(kind, None)
    seed_registry._REQUIRES.pop(kind, None)


# --- configuration ---------------------------------------------------------------


def test_a_seed_graph_takes_its_defaults() -> None:
    seed = SeedGraphConfig(name="procure", kind="jsonl", path="seed.jsonl")
    assert (seed.on_name_collision, seed.shards, seed.max_items_per_unit) == (
        "error",
        16,
        50,
    )
    assert (seed.max_nodes, seed.max_edges, seed.max_bytes) == (
        100_000,
        500_000,
        256 * 1024 * 1024,
    )
    assert Config().processing.seed_graphs == []


def test_an_rdf_seed_defaults_to_a_smaller_byte_cap() -> None:
    # rdflib holds the whole graph, 25-110 times the file's bytes, before
    # max_nodes is checked: 256 MiB of Turtle needed some 28 GB.
    rdf = SeedGraphConfig(name="procure", kind="rdf", path="seed.ttl")
    assert rdf.max_bytes == 32 * 1024 * 1024
    assert rdf.model_copy(update={"path": "/abs/seed.ttl"}).max_bytes == rdf.max_bytes
    raised = SeedGraphConfig(
        name="procure", kind="rdf", path="seed.ttl", max_bytes=256 * 1024 * 1024
    )
    assert raised.max_bytes == 256 * 1024 * 1024
    loaded = Config.model_validate(
        {"processing": {"seed_graphs": [{"name": "p", "kind": "rdf", "path": "s"}]}}
    )
    assert loaded.processing.seed_graphs[0].max_bytes == 32 * 1024 * 1024


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ({"name": "Procure"}, "only lowercase letters"),
        ({"name": "pro/cure"}, "only lowercase letters"),
        ({"name": "-procure"}, "starts with a lowercase letter or a digit"),
        ({"name": "x" * 65}, "at most 64"),
        ({"kind": "JSONL"}, "should match pattern"),
        ({"path": ""}, "at least 1"),
        ({"filters": {SEED_GRAPH_FILTER_KEY: "other"}}, "is reserved"),
        ({"filters": {"score": float("nan")}}, "not a finite number"),
        ({"filters": {"tags": ["a"]}}, "filters.tags"),
        ({"shards": 0}, "greater than or equal to 1"),
        ({"max_description_chars": 10}, "greater than or equal to 50"),
        ({"on_name_collision": "rename"}, "'error', 'merge' or 'qualify'"),
        ({"secret_arn": "arn:aws:s3:::bucket"}, "should match pattern"),
        # A misspelled key in a list item is an error, not a default.
        ({"shard": 4}, "Extra inputs are not permitted"),
    ],
)
def test_a_seed_graph_refuses_invalid_settings(
    item: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _config({"name": "procure", "kind": "jsonl", "path": "seed.jsonl", **item})


def test_seed_graph_names_are_unique() -> None:
    seed = {"name": "procure", "kind": "jsonl", "path": "a.jsonl"}
    with pytest.raises(ValidationError, match="unique: procure appear"):
        _config(seed, {**seed, "path": "b.jsonl"})
    assert len(_config(seed, {**seed, "name": "parts"}).processing.seed_graphs) == 2


def test_the_reserved_filter_key_is_the_conversions() -> None:
    assert SEED_GRAPH_FILTER_KEY == SEED_FILTER_KEY


def test_filter_values_keep_their_type() -> None:
    seed = _seed("s.jsonl", filters={"region": "eu", "tier": 1, "open": True})
    assert seed.filters == {"region": "eu", "tier": 1, "open": True}
    assert type(seed.filters["open"]) is bool


def test_a_seed_graph_is_frozen() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        _seed("s.jsonl").shards = 2  # type: ignore[misc]


def test_seed_graphs_load_from_the_environment_as_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = _write(tmp_path / "config.yaml", "{}\n")
    monkeypatch.setenv(
        "UKR__PROCESSING__SEED_GRAPHS",
        json.dumps([{"name": "procure", "kind": "csv", "path": "seed/"}]),
    )
    seeds = ConfigLoader(config_file).load_config().processing.seed_graphs
    assert [(s.name, s.kind) for s in seeds] == [("procure", "csv")]
    # Relative to the config file, as a schema source's path.
    assert seeds[0].path == str((tmp_path / "seed").resolve())


def test_an_invalid_seed_graph_in_the_environment_names_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = _write(tmp_path / "config.yaml", "{}\n")
    monkeypatch.setenv(
        "UKR__PROCESSING__SEED_GRAPHS",
        json.dumps([{"name": "procure", "kind": "jsonl", "path": "s", "shard": 2}]),
    )
    with pytest.raises(ConfigurationError, match="UKR__PROCESSING__SEED_GRAPHS"):
        ConfigLoader(config_file).load_config()


def test_relative_seed_paths_resolve_against_the_config_file(tmp_path: Path) -> None:
    config_file = _write(
        tmp_path / "conf" / "config.yaml",
        "processing:\n  seed_graphs:\n"
        "    - {name: local, kind: jsonl, path: seed/procure.jsonl}\n"
        "    - {name: absolute, kind: jsonl, path: /srv/seed.jsonl}\n"
        "    - {name: remote, kind: jsonl, path: 's3://seed-bucket/procure.jsonl'}\n",
    )
    config = ConfigLoader(config_file).load_config()
    assert [s.path for s in config.processing.seed_graphs] == [
        str((tmp_path / "conf" / "seed" / "procure.jsonl").resolve()),
        "/srv/seed.jsonl",
        "s3://seed-bucket/procure.jsonl",
    ]
    # The resolved paths survive a rebuild of the config from its input.
    rebuilt = Config.model_validate(config.as_input())
    assert rebuilt.processing.seed_graphs == config.processing.seed_graphs


# --- keying ------------------------------------------------------------------------


def _keys(config: Config, corpus: str = "corpus") -> dict[str, str]:
    return {
        stage.value: stage_input_fingerprint(config, stage, corpus)
        for stage in PipelineStageType
    }


def test_seed_graphs_are_keyed_by_content_not_by_setting() -> None:
    rule = classify("processing.seed_graphs.path")
    assert rule is CONFIG_KEYING["processing.seed_graphs"]
    assert isinstance(rule, Unkeyed)
    assert not shapes_answers("processing.seed_graphs")


def test_configuring_a_seed_changes_no_setting_key(procure_jsonl: Path) -> None:
    configured = _config(
        {"name": "procure", "kind": "jsonl", "path": str(procure_jsonl)}
    )
    assert _keys(configured) == _keys(Config())
    # Documents never re-extract for a seed.
    assert extraction_settings(configured) == extraction_settings(Config())
    assert extraction_fingerprint(configured) == extraction_fingerprint(Config())


def test_no_seed_keeps_the_corpus_fingerprint() -> None:
    assert seed_corpus_fingerprint("abc123") == "abc123"
    assert seed_corpus_fingerprint("abc123", []) == "abc123"
    assert seed_corpus_fingerprint("abc123", ["d1"]) != "abc123"
    assert seed_corpus_fingerprint("abc123", ["d1"]) != seed_corpus_fingerprint(
        "abc124", ["d1"]
    )
    assert len(seed_corpus_fingerprint("abc123", ["d1"])) == 12


def _corpus_key(config: Config) -> str:
    """The corpus fingerprint of an empty corpus with ``config``'s seeds."""
    digests = [c.digest for c in load_seed_graphs(config)]
    return seed_corpus_fingerprint("corpus", digests)


def test_a_moved_seed_keys_nothing_and_an_edited_one_every_stage(
    tmp_path: Path,
) -> None:
    text = lines(*NODES, *EDGES)
    here = _write(tmp_path / "a" / "procure.jsonl", text)
    there = _write(tmp_path / "b" / "renamed.jsonl", text)
    edited = _write(
        tmp_path / "c" / "procure.jsonl",
        text.replace("Industrial valve vendor.", "Industrial pump vendor."),
    )

    def config(path: Path) -> Config:
        return _config({"name": "procure", "kind": "jsonl", "path": str(path)})

    first = _keys(config(here), _corpus_key(config(here)))
    assert _keys(config(there), _corpus_key(config(there))) == first
    changed = _keys(config(edited), _corpus_key(config(edited)))
    assert all(changed[stage] != first[stage] for stage in first)
    # Without seeds, the keys are the corpus's alone.
    assert _keys(Config(), _corpus_key(Config())) == _keys(Config())
    assert extraction_settings(config(edited)) == extraction_settings(config(here))


def test_the_pipeline_folds_its_seeds_into_the_corpus_fingerprint(
    procure_jsonl: Path, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    _write(corpus / "doc.txt", "Acme supplies Globex.")
    conversions = load_seed_graphs(
        _config({"name": "procure", "kind": "jsonl", "path": str(procure_jsonl)})
    )
    pipeline: Any = SimpleNamespace(
        pipeline_config=SimpleNamespace(local_directory=str(tmp_path / "cache")),
        target_directory=None,
        seed_conversions=(),
    )
    alone = DataIngestionPipeline._compute_corpus_fingerprint(pipeline, corpus)
    assert DataIngestionPipeline.seed_conversions == ()
    pipeline.seed_conversions = tuple(conversions)
    with_seed = DataIngestionPipeline._compute_corpus_fingerprint(pipeline, corpus)
    assert with_seed == seed_corpus_fingerprint(alone, [conversions[0].digest])
    assert with_seed != alone


# --- registry ------------------------------------------------------------------------


def test_the_built_in_kinds_are_registered() -> None:
    assert {"jsonl", "csv", "rdf"} <= set(registered_seed_sources())
    # rdf needs the ontology extra (test_rdf_seed_source.py).
    assert (seed_source_requirement("rdf") or ("",))[0] == "rdflib"
    assert seed_source_requirement("jsonl") is None
    assert seed_source_requirement("csv") is None


def test_a_registered_kind_is_loaded_and_converted(custom_kind: str) -> None:
    seen: list[SeedSourceContext] = []

    @register_seed_source(custom_kind)
    def _load(context: SeedSourceContext) -> SeedGraph:
        seen.append(context)
        return PROCURE

    assert get_seed_source(custom_kind) is _load
    config = _config(
        {"name": "procure", "kind": custom_kind, "path": "urn:service:procure?token=x"}
    )
    (conversion,) = load_seed_graphs(config)
    assert seen[0].source is config.processing.seed_graphs[0]
    assert seen[0].config is config
    assert seen[0].location == "urn:service:procure?token=x"
    assert conversion.name == "procure"
    assert (conversion.stats.nodes, conversion.stats.edges) == (2, 1)


def test_a_later_registration_replaces_the_earlier(custom_kind: str) -> None:
    register_seed_source(custom_kind, requires=("absent", "x"))(lambda c: PROCURE)
    replacement = register_seed_source(custom_kind)(lambda c: SeedGraph())
    assert get_seed_source(custom_kind) is replacement
    assert seed_source_requirement(custom_kind) is None


@pytest.mark.parametrize("kind", ["my-kind", "", "1st", "_x"])
def test_invalid_kinds_cannot_register(kind: str) -> None:
    with pytest.raises(ValueError, match="Cannot register a seed source"):
        register_seed_source(kind)


def test_an_unknown_kind_fails_naming_the_registered_ones() -> None:
    config = _config({"name": "procure", "kind": "owl", "path": "abox.ttl"})
    with pytest.raises(SeedGraphError, match="'owl'; registered: csv, jsonl, rdf"):
        load_seed_graphs(config)
    # A configuration error: the CLIs exit 2 on it.
    assert issubclass(SeedGraphError, ConfigurationError)


def test_the_context_reads_options_and_redacts_the_location() -> None:
    seed = _seed(
        "s3://user:secret@seed-bucket/procure.jsonl?X-Amz-Signature=abc",  # gitleaks:allow
        options={"dangling": "skip"},
    )
    context = SeedSourceContext(seed, Config())
    assert context.location == "s3://seed-bucket/procure.jsonl"
    assert context.options(SeedSourceOptions).dangling == "skip"
    assert context.terms.error is SeedGraphError
    assert str(context.terms("boom")) == "seed graph 'procure': boom"
    assert SeedTerms().error is SeedGraphError
    bad = SeedSourceContext(_seed("s.jsonl", options={"danglng": "skip"}), Config())
    with pytest.raises(SeedGraphError, match="options.danglng: Extra inputs"):
        bad.options(SeedSourceOptions)


def test_the_context_builds_clients_of_the_run_session_once() -> None:
    sessions: list[Any] = []

    class _Session:
        def client(self, service_name: str, **kwargs: Any) -> tuple[str, Any]:
            return service_name, kwargs["region_name"]

    def factory() -> Any:
        sessions.append(_Session())
        return sessions[-1]

    config = Config.model_validate(
        {"aws": {"region_name": "eu-west-1", "bedrock": {"region_name": "us-west-2"}}}
    )
    context = SeedSourceContext(_seed("s.jsonl"), config, session_factory=factory)
    assert context.client("s3") == ("s3", "eu-west-1")
    assert context.client("bedrock-runtime") == ("bedrock-runtime", "us-west-2")
    assert len(sessions) == 1


# --- jsonl -----------------------------------------------------------------------------


def test_jsonl_reads_nodes_and_edges(procure_jsonl: Path) -> None:
    graph = _load(procure_jsonl)
    assert _shape(graph) == _shape(PROCURE)
    assert graph.report.files == 1
    assert graph.report.bytes_read == procure_jsonl.stat().st_size


def test_jsonl_reads_a_directory_in_sorted_order(tmp_path: Path) -> None:
    root = tmp_path / "seed"
    _write(root / "b" / "edges.jsonl", lines(*EDGES))
    _write(root / "a-nodes.jsonl", lines(NODES[1]))
    _write(root / "a" / "nodes.jsonl", lines(NODES[0]))
    _write(root / ".hidden" / "x.jsonl", "not json\n")
    _write(root / ".skipped.jsonl", "not json\n")
    _write(root / "notes.txt", "not json\n")
    graph = _load(root)
    # a-nodes.jsonl < a/nodes.jsonl < b/edges.jsonl
    assert [node.id for node in graph.nodes] == ["b1", "v1"]
    assert graph.edges == PROCURE.edges
    assert graph.report.files == 3


def test_jsonl_skips_blank_lines_and_a_byte_order_mark(tmp_path: Path) -> None:
    text = "\n" + lines(*NODES) + "  \r\n" + lines(*EDGES).replace("\n", "\r\n")
    path = _write(tmp_path / "s.jsonl", b"\xef\xbb\xbf" + text.encode("utf-8"))
    assert _shape(_load(path)) == _shape(PROCURE)


def test_conversion_does_not_depend_on_line_order(tmp_path: Path) -> None:
    forward = _write(tmp_path / "f.jsonl", lines(*NODES, *EDGES))
    backward = _write(tmp_path / "b.jsonl", lines(*EDGES, *reversed(NODES)))
    digests = [
        load_seed_graphs(
            _config({"name": "procure", "kind": "jsonl", "path": str(path)})
        )[0].digest
        for path in (forward, backward)
    ]
    assert digests[0] == digests[1]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json\n", "line 1: not valid JSON"),
        ('{"kind": "node", "id": "x", "confidence": NaN}\n', "NaN is not a JSON"),
        ("[1, 2]\n", "line 1: not a JSON object"),
        ('{"id": "v1"}\n', '"kind" is null; expected "node" or "edge"'),
        ('{"kind": "vertex", "id": "v1"}\n', '"kind" is "vertex"'),
        ('{"kind": "node", "id": "v1", "label": "x"}\n', "label: Extra inputs"),
        ('{"kind": "node", "name": "No id"}\n', "id: Field required"),
        ('{"kind": "node", "id": ""}\n', "id: String should have at least 1"),
        ('{"kind": "node", "id": "v1", "confidence": 2}\n', "confidence: Input"),
        ('{"kind": "edge", "source": "v1"}\n', "target: Field required"),
        ('{"kind": "edge", "source": "a", "target": "b", "weight": -1}\n', "weight"),
        (
            '{"kind": "node", "id": "v1", "properties": {"tags": ["a"]}}\n',
            "property 'tags' is not a string, number, boolean or null",
        ),
        (
            '{"kind": "node", "id": "v1", "properties": {"o": {"k": 1}}}\n',
            "property 'o' is not a string",
        ),
        ('{"kind": "node", "id": "v1", "properties": [1]}\n', "properties is not an"),
        (
            '{"kind": "node", "id": "v1"}\n{"kind": "node", "id": "v1"}\n',
            r"line 2: duplicate node id 'v1' \(first at '.*s.jsonl' line 1\)",
        ),
        (b"\xff\xfe{}\n", "is not valid UTF-8 \\(byte offset 0\\)"),
        # Under the line cap, deeper than the JSON decoder recurses.
        ("[" * 100_000 + "]" * 100_000 + "\n", "line 1: not valid JSON .nested"),
        # A JSON \ud800 escape decodes to a lone surrogate: no store encodes it.
        ('{"kind": "node", "id": "v1", "name": "A\\ud800"}\n', "name: Input should"),
        (
            '{"kind": "node", "id": "v1", "description": "\\udc00"}\n',
            "description: .*lone surrogate .U\\+DC00",
        ),
        (
            '{"kind": "node", "id": "v1", "properties": {"k": "\\ud800"}}\n',
            "property 'k' holds a lone surrogate",
        ),
        (
            '{"kind": "node", "id": "v1", "properties": {"k\\udfff": 1}}\n',
            "a property key holds a lone surrogate .U\\+DFFF",
        ),
        (
            '{"kind": "node", "id": "v1", "name": "' + "n" * 1001 + '"}\n',
            "name: String should have at most 1000 characters",
        ),
        (
            '{"kind": "node", "id": "v1", "iri": "urn:' + "x" * 4000 + '"}\n',
            "iri: String should have at most 4000 characters",
        ),
        (
            '{"kind": "edge", "source": "a", "target": "b", "type": "'
            + "T" * 257
            + '"}\n',
            "type: String should have at most 256 characters",
        ),
    ],
)
def test_jsonl_refuses_a_bad_line_naming_it(
    tmp_path: Path, text: str | bytes, message: str
) -> None:
    path = _write(tmp_path / "s.jsonl", text)
    with pytest.raises(SeedGraphError, match=message) as error:
        _load(path)
    assert str(error.value).startswith("seed graph 'procure': ")


def test_jsonl_refuses_a_line_over_the_line_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert property_graph.MAX_LINE_BYTES == 1024 * 1024
    monkeypatch.setattr(property_graph, "MAX_LINE_BYTES", 200)
    long = json.dumps({"kind": "node", "id": "v2", "description": "x" * 200})
    path = _write(tmp_path / "s.jsonl", lines(NODES[1]) + long + "\n")
    with pytest.raises(SeedGraphError, match="line 2 holds more than 200 bytes"):
        _load(path)


def test_jsonl_refuses_files_over_max_bytes_before_reading(
    tmp_path: Path, mocker: Any
) -> None:
    root = tmp_path / "seed"
    _write(root / "a.jsonl", lines(*NODES))
    _write(root / "b.jsonl", lines(*EDGES))
    total = sum(p.stat().st_size for p in root.iterdir())
    read = mocker.spy(property_graph.FileReader, "read")
    with pytest.raises(
        SeedGraphError, match=f"hold {total} bytes, more than max_bytes"
    ):
        _load(root, max_bytes=total - 1)
    read.assert_not_called()
    assert len(_load(root, max_bytes=total).nodes) == 2


def test_jsonl_refuses_a_file_that_grew_past_max_bytes(
    procure_jsonl: Path, mocker: Any
) -> None:
    # Sized within the cap (the check before reading passes), then read
    # larger.
    size = procure_jsonl.stat().st_size
    mocker.patch.object(jsonl, "check_total")
    with pytest.raises(SeedGraphError, match="grew since it was sized"):
        _load(procure_jsonl, max_bytes=size - 1)
    assert len(_load(procure_jsonl, max_bytes=size).nodes) == len(NODES)


def test_jsonl_counts_every_file_read_against_max_bytes(
    tmp_path: Path, mocker: Any
) -> None:
    # Each file within the cap alone, both over it.
    root = tmp_path / "seed"
    _write(root / "a.jsonl", lines(*NODES))
    _write(root / "b.jsonl", lines(*EDGES))
    largest = max(p.stat().st_size for p in root.iterdir())
    mocker.patch.object(jsonl, "check_total")
    with pytest.raises(SeedGraphError, match="grew since it was sized"):
        _load(root, max_bytes=largest)


@pytest.mark.parametrize(
    ("cap", "kind"), [("max_nodes", "nodes"), ("max_edges", "edges")]
)
def test_jsonl_stops_at_the_node_and_edge_caps(
    tmp_path: Path, cap: str, kind: str
) -> None:
    edges = [{**EDGES[0], "target": f"b{i}"} for i in range(3)]
    path = _write(tmp_path / "s.jsonl", lines(*NODES, *edges))
    with pytest.raises(
        SeedGraphError, match=f"line .*: more {kind} than {cap} \\(1\\)"
    ):
        _load(path, **{cap: 1})


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"secret_arn": "arn:aws:secretsmanager:us-east-1:1:secret:x"}, "secret_arn"),
        ({"options": {"format": "ndjson"}}, "options.format: Extra inputs"),
        ({"options": {"dangling": "stub"}}, "options.dangling"),
    ],
)
def test_jsonl_refuses_settings_it_does_not_take(
    procure_jsonl: Path, settings: dict[str, Any], message: str
) -> None:
    with pytest.raises(SeedGraphError, match=message):
        _load(procure_jsonl, **settings)


def test_jsonl_refuses_a_missing_path_an_empty_directory_and_other_urls(
    tmp_path: Path,
) -> None:
    with pytest.raises(SeedGraphError, match="not found"):
        _load(tmp_path / "missing.jsonl")
    _write(tmp_path / "empty" / "notes.txt", "x")
    with pytest.raises(SeedGraphError, match="holds no .jsonl file"):
        _load(tmp_path / "empty")
    with pytest.raises(SeedGraphError, match="is a URL; kind 'jsonl' reads"):
        _load("https://user:pw@example.org/seed.jsonl")  # gitleaks:allow
    with pytest.raises(SeedGraphError, match="does not name an S3 bucket"):
        _load("s3://Not_A_Bucket/seed.jsonl")


# --- csv ------------------------------------------------------------------------------


@pytest.fixture
def procure_csv(tmp_path: Path) -> Path:
    root = tmp_path / "csv"
    _write(root / "nodes.csv", NODES_CSV)
    _write(root / "edges.csv", EDGES_CSV)
    return root


def test_csv_reads_the_same_graph_as_jsonl(procure_csv: Path) -> None:
    graph = _load(procure_csv, "csv")
    assert _shape(graph) == _shape(PROCURE)
    assert graph.report.files == 2


def test_csv_takes_other_file_names_and_no_edges_file(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write(root / "parts" / "vertices.csv", NODES_CSV)
    _write(root / "links.csv", EDGES_CSV)
    graph = _load(
        root, "csv", options={"nodes": "parts/vertices.csv", "edges": "links.csv"}
    )
    assert _shape(graph) == _shape(PROCURE)
    nodes_only = _load(
        root, "csv", options={"nodes": "parts/vertices.csv", "edges": None}
    )
    assert (len(nodes_only.nodes), nodes_only.edges) == (2, ())


def test_csv_keeps_only_the_property_columns_it_is_given(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write(root / "nodes.csv", "id,name,country,size\nv1,Acme,DE,large\n")
    _write(root / "edges.csv", "source,target,type,since\n")
    assert _load(root, "csv").nodes[0].properties == {"country": "DE", "size": "large"}
    kept = _load(root, "csv", options={"property_columns": ["size", "since"]})
    assert kept.nodes[0].properties == {"size": "large"}
    with pytest.raises(SeedGraphError, match="property_columns names colour, a column"):
        _load(root, "csv", options={"property_columns": ["colour"]})


def test_csv_reads_quoted_cells_a_byte_order_mark_and_crlf(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    nodes = 'id,name,description\r\nv1,"Acme, Inc.","Valves.\nPumps."\r\n\r\n'
    _write(root / "nodes.csv", b"\xef\xbb\xbf" + nodes.encode("utf-8"))
    _write(root / "edges.csv", "source,target\n")
    (node,) = _load(root, "csv").nodes
    assert (node.name, node.description) == ("Acme, Inc.", "Valves.\nPumps.")


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        ("", r"nodes.csv': is empty"),
        ("name,type\nAcme,VENDOR\n", "line 1: no id column"),
        ("id,name,name\nv1,a,b\n", "line 1: columns name repeat"),
        ("id,,name\nv1,x,a\n", "line 1: a column has no name"),
        ("id,name\nv1,Acme,extra\n", "line 2: 3 cells, more than the 2 columns"),
        ("id,confidence\nv1,high\n", "line 2: confidence: Input should be a valid"),
        ("id\nv1\nv1\n", "line 3: duplicate node id 'v1'"),
        ('id,name\nv1,"unterminated\n', "not valid CSV"),
        ("id,description\nv1," + "x" * 131_073 + "\n", "field larger than field limit"),
    ],
)
def test_csv_refuses_a_bad_file_naming_the_line(
    tmp_path: Path, nodes: str, message: str
) -> None:
    root = tmp_path / "csv"
    _write(root / "nodes.csv", nodes)
    _write(root / "edges.csv", EDGES_CSV)
    with pytest.raises(SeedGraphError, match=message):
        _load(root, "csv")


def test_csv_refuses_edges_without_endpoints(tmp_path: Path) -> None:
    root = tmp_path / "csv"
    _write(root / "nodes.csv", NODES_CSV)
    _write(root / "edges.csv", "source,type\nv1,SUPPLIES\n")
    with pytest.raises(SeedGraphError, match="edges.csv' line 1: no target column"):
        _load(root, "csv")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"nodes": "../nodes.csv"}, "not a file name relative"),
        ({"nodes": "/etc/nodes.csv"}, "not a file name relative"),
        ({"edges": "missing.csv"}, "missing.csv' not found"),
        ({"delimiter": ";"}, "options.delimiter: Extra inputs"),
    ],
)
def test_csv_refuses_bad_options(
    procure_csv: Path, options: dict[str, Any], message: str
) -> None:
    with pytest.raises(SeedGraphError, match=message):
        _load(procure_csv, "csv", options=options)


def test_csv_reads_a_directory_only(procure_csv: Path) -> None:
    with pytest.raises(SeedGraphError, match="is not a directory; kind 'csv'"):
        _load(procure_csv / "nodes.csv", "csv")


def test_csv_refuses_files_over_max_bytes(procure_csv: Path) -> None:
    total = len(NODES_CSV) + len(EDGES_CSV)
    with pytest.raises(SeedGraphError, match=f"hold {total} bytes"):
        _load(procure_csv, "csv", max_bytes=total - 1)


# --- preflight ---------------------------------------------------------------------------


def test_preflight_names_a_missing_seed_requirement(custom_kind: str) -> None:
    register_seed_source(custom_kind, requires=("synthetic_absent_module", "x y"))(
        lambda context: SeedGraph()
    )
    config = _config(
        {"name": "procure", "kind": custom_kind, "path": "a"},
        {"name": "parts", "kind": "jsonl", "path": "b"},
    )
    error = missing_extras_error(config) or ""
    assert (
        f"'synthetic_absent_module' (for processing.seed_graphs 'procure' kind "
        f"'{custom_kind}')" in error
    )
    assert "parts" not in error
    assert "Install x y." in error
    assert (
        missing_extras_error(_config({"name": "p", "kind": "jsonl", "path": "a"}))
        is None
    )


def test_preflight_passes_an_installed_seed_requirement(custom_kind: str) -> None:
    register_seed_source(custom_kind, requires=("json", "x"))(lambda c: SeedGraph())
    assert (
        missing_extras_error(_config({"name": "p", "kind": custom_kind, "path": "a"}))
        is None
    )


# --- bootstrap: load_seed_graphs -------------------------------------------------------------


def test_no_seed_graphs_load_nothing() -> None:
    assert load_seed_graphs(Config()) == []


def test_the_conversion_settings_come_from_the_config() -> None:
    seed = _seed(
        "s.jsonl",
        filters={"region": "eu"},
        on_name_collision="qualify",
        description_properties=["country"],
        confidence=0.8,
        shards=4,
        max_items_per_unit=10,
        max_nodes=5,
        max_edges=6,
        max_property_keys=7,
        max_description_chars=80,
        options={"dangling": "skip", "qualifier_property": "country", "nodes": "n.csv"},
    )
    assert seed_conversion_settings(seed) == SeedConversionSettings(
        name="procure",
        filters={"region": "eu"},
        on_name_collision="qualify",
        qualifier_property="country",
        description_properties=("country",),
        confidence=0.8,
        shards=4,
        max_items_per_unit=10,
        max_nodes=5,
        max_edges=6,
        max_property_keys=7,
        max_description_chars=80,
        dangling="skip",
    )
    with pytest.raises(SeedGraphError, match="options.dangling"):
        seed_conversion_settings(_seed("s.jsonl", options={"dangling": "stub"}))


def test_dangling_edges_fail_or_are_skipped(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "s.jsonl", lines(*NODES, *EDGES, {**EDGES[0], "target": "x9"})
    )
    seed = {"name": "procure", "kind": "jsonl", "path": str(path)}
    with pytest.raises(SeedGraphError, match="edges name unknown nodes 'x9'"):
        load_seed_graphs(_config(seed))
    (skipped,) = load_seed_graphs(_config({**seed, "options": {"dangling": "skip"}}))
    assert (skipped.stats.edges, skipped.stats.dangling_edges_skipped) == (1, 1)


def test_seeds_load_in_configuration_order(
    procure_jsonl: Path, procure_csv: Path
) -> None:
    conversions = load_seed_graphs(
        _config(
            {"name": "procure", "kind": "jsonl", "path": str(procure_jsonl)},
            {"name": "parts", "kind": "csv", "path": str(procure_csv)},
        )
    )
    assert [c.name for c in conversions] == ["procure", "parts"]
    assert conversions[1].filters == {SEED_FILTER_KEY: "parts"}


def test_a_seed_load_logs_counts_only(
    procure_jsonl: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config({"name": "procure", "kind": "jsonl", "path": str(procure_jsonl)})
    with caplog.at_level(logging.INFO, logger="unified_kg_rag.application.bootstrap"):
        (conversion,) = load_seed_graphs(config)
    (record,) = [r for r in caplog.records if r.name.endswith("bootstrap")]
    message = record.getMessage()
    # In: a structured-logging setup of another test may wrap the message.
    assert f"Seed graph 'procure' (jsonl '{procure_jsonl}')" in message
    assert "read 2 nodes and 1 edges from 1 files" in message
    assert conversion.digest[:12] in message
    for text in ("Acme", "Globex", "valve", "DE", EX):
        assert text not in message


def test_a_seed_needs_the_schema_source_resolved(
    procure_jsonl: Path, tmp_path: Path
) -> None:
    schema = _write(tmp_path / "schema.yaml", "relationship_types: [SUPPLIES]\n")
    config = _config(
        {"name": "procure", "kind": "jsonl", "path": str(procure_jsonl)},
        schema_source={"kind": "file", "path": str(schema)},
    )
    with pytest.raises(ConfigurationError, match="not loaded yet"):
        load_seed_graphs(config)
    (conversion,) = load_seed_graphs(resolve_extraction_schema(config))
    assert conversion.stats.edges == 1


def test_a_seed_maps_onto_a_file_schema(procure_jsonl: Path, tmp_path: Path) -> None:
    schema = _write(
        tmp_path / "schema.yaml",
        "entity_types:\n"
        "  - VENDOR\n"
        f"  - {{label: BUYER, iri: '{EX}Buyer'}}\n"
        "relationship_types: [SUPPLIES]\n"
        "relationship_patterns: ['VENDOR -SUPPLIES-> BUYER']\n",
    )
    config = resolve_extraction_schema(
        _config(
            {"name": "procure", "kind": "jsonl", "path": str(procure_jsonl)},
            schema_source={"kind": "file", "path": str(schema)},
            schema_mode="strict",
        )
    )
    (conversion,) = load_seed_graphs(config)
    # b1's type_iri named BUYER; nothing was off the schema.
    assert '"type":"BUYER"' in _text(conversion)
    assert conversion.stats.nodes_unmapped_type_iri == 0
    assert conversion.stats.entities_dropped_off_schema == 0
    assert conversion.stats.relationships_dropped_off_pattern == 0


@requires_ontology
def test_a_seed_maps_onto_an_owl_schema(procure_jsonl: Path) -> None:
    config = resolve_extraction_schema(
        _config(
            {"name": "procure", "kind": "jsonl", "path": str(procure_jsonl)},
            schema_source={"kind": "owl", "path": str(ONTOLOGY)},
        )
    )
    (conversion,) = load_seed_graphs(config)
    text = _text(conversion)
    assert '"type":"BUYER"' in text
    assert '"type":"VENDOR"' in text
    assert '"type":"SUPPLIES"' in text
    assert conversion.stats.nodes_unmapped_type_iri == 0


def test_a_loader_gets_the_bundle_session(custom_kind: str) -> None:
    sessions: list[Any] = []

    @register_seed_source(custom_kind)
    def _load(context: SeedSourceContext) -> SeedGraph:
        sessions.append(context.boto_session())
        return PROCURE

    config = _config({"name": "procure", "kind": custom_kind, "path": "urn:x"})
    session, bundle = object(), SimpleNamespace(boto_session=object())
    load_seed_graphs(config, boto_session=session)  # type: ignore[arg-type]
    load_seed_graphs(config, bundle)  # type: ignore[arg-type]
    assert sessions == [session, bundle.boto_session]


def test_ingestion_exits_2_on_a_seed_graph_error(
    mocker: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unified_kg_rag.application.cli import run_ingestion_pipeline

    mocker.patch.object(
        run_ingestion_pipeline, "load_cli_config", return_value=Config()
    )
    mocker.patch.object(run_ingestion_pipeline, "display_ascii_art")
    mocker.patch.object(
        run_ingestion_pipeline,
        "DataIngestionPipeline",
        side_effect=SeedGraphError("seed graph 'procure': duplicate node ids 'v1'"),
    )
    (tmp_path / "src").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-ingestion",
            "--source-directory",
            str(tmp_path / "src"),
            "--cache-directory",
            str(tmp_path),
        ],
    )
    # A SeedGraphError is a ConfigurationError: the usage exit code.
    with pytest.raises(SystemExit) as exit_:
        run_ingestion_pipeline.main()
    assert exit_.value.code == 2
