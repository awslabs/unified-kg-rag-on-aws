# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema sources: the registry, the ``file`` loader, and the start-up load
into the inline keys (``resolve_extraction_schema``)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.adapters.schema_sources import (
    SchemaSourceContext,
    get_schema_source,
    redact_location,
    register_schema_source,
    registered_schema_sources,
)
from unified_kg_rag.adapters.schema_sources import registry as schema_registry
from unified_kg_rag.application.bootstrap import (
    resolve_extraction_schema,
    with_provider_overrides,
)
from unified_kg_rag.application.config import ConfigLoader
from unified_kg_rag.domain.ingestion.cache_keys import (
    answer_settings,
    extraction_fingerprint,
    stage_input_fingerprint,
    unit_record_fingerprint,
)
from unified_kg_rag.domain.models import (
    Config,
    PipelineConfig,
    PipelineStageType,
    SchemaDocument,
)
from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.shared.exceptions import ConfigurationError, SchemaSourceError

pytestmark = pytest.mark.unit

_SCHEMA_YAML = """\
relationship_types:
  - "WORKS_FOR: Employment"
  - {label: PART_OF, iri: "urn:example:partOf"}
relationship_patterns:
  - "PERSON -WORKS_FOR-> ORGANIZATION"
  - {source_type: ORGANIZATION, relation: PART_OF, target_type: ORGANIZATION}
"""
_INLINE: dict[str, Any] = {
    "relationship_types": [
        "WORKS_FOR: Employment",
        {"label": "PART_OF", "iri": "urn:example:partOf"},
    ],
    "relationship_patterns": [
        "PERSON -WORKS_FOR-> ORGANIZATION",
        {
            "source_type": "ORGANIZATION",
            "relation": "PART_OF",
            "target_type": "ORGANIZATION",
        },
    ],
}


def _config(**extraction: Any) -> Config:
    return Config.model_validate({"processing": {"graph_extraction": extraction}})


def _file_config(path: Path, **extraction: Any) -> Config:
    return _config(schema_source={"kind": "file", "path": str(path)}, **extraction)


def _load_file(path: str, **source: Any) -> SchemaDocument:
    """The ``file`` loader's document of ``path``."""
    return get_schema_source("file")(
        SchemaSourceContext(
            SchemaSourceConfig(kind="file", path=path, **source), Config()
        )
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _keys(config: Config) -> dict[str, str]:
    return {
        stage.value: stage_input_fingerprint(config, stage, "corpus")
        for stage in PipelineStageType
    }


@pytest.fixture
def custom_kind() -> Iterator[str]:
    kind = "synthetic_service"
    yield kind
    schema_registry._REGISTRY.pop(kind, None)


# --- registry ------------------------------------------------------------------


def test_file_is_registered() -> None:
    assert "file" in registered_schema_sources()
    assert "inline" not in registered_schema_sources()


def test_a_registered_kind_fills_the_inline_keys(custom_kind: str) -> None:
    seen: list[SchemaSourceContext] = []

    @register_schema_source(custom_kind)
    def _load(context: SchemaSourceContext) -> SchemaDocument:
        seen.append(context)
        return SchemaDocument(**_INLINE)

    assert get_schema_source(custom_kind) is _load
    config = _config(schema_source={"kind": custom_kind, "path": "urn:service:x"})
    resolved = resolve_extraction_schema(config)
    assert seen and seen[0].source.path == "urn:service:x"
    assert seen[0].config is config
    extraction = resolved.processing.graph_extraction
    assert extraction.schema_source == SchemaSourceConfig()
    inline = _config(**_INLINE).processing.graph_extraction
    assert extraction.relationship_types == inline.relationship_types
    assert extraction.relationship_patterns == inline.relationship_patterns


def test_a_later_registration_replaces_the_earlier(custom_kind: str) -> None:
    register_schema_source(custom_kind)(lambda context: SchemaDocument())
    replacement = register_schema_source(custom_kind)(
        lambda context: SchemaDocument(**_INLINE)
    )
    assert get_schema_source(custom_kind) is replacement


@pytest.mark.parametrize("kind", ["inline", "File", "my-kind", "", "1st"])
def test_invalid_kinds_cannot_register(kind: str) -> None:
    with pytest.raises(ValueError, match="Cannot register"):
        register_schema_source(kind)


def test_an_unknown_kind_fails_naming_the_registered_ones() -> None:
    config = _config(schema_source={"kind": "owl", "path": "x.ttl"})
    with pytest.raises(SchemaSourceError, match="'owl'; registered: inline, file"):
        resolve_extraction_schema(config)
    # A configuration error: the CLIs exit 2 on it.
    assert issubclass(SchemaSourceError, ConfigurationError)


# --- the file loader -----------------------------------------------------------


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_file_loader_reads_yaml_and_json(tmp_path: Path, suffix: str) -> None:
    text = json.dumps(_INLINE) if suffix == ".json" else _SCHEMA_YAML
    path = _write(tmp_path / f"schema{suffix}", text)
    document = _load_file(str(path))
    assert document.entity_types is None
    assert document.model_dump(mode="json") == SchemaDocument(**_INLINE).model_dump(
        mode="json"
    )


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("schema.yaml", "- a\n- b\n", "must hold a mapping, got list"),
        ("schema.yaml", "relationship_types: [\n", "does not parse"),
        ("schema.json", "{", "does not parse"),
        ("schema.yaml", "relation_types: [R]\n", "relation_types"),
        ("schema.yaml", "schema_mode: strict\n", "schema_mode"),
        (
            "schema.yaml",
            "relationship_types:\n  - {label: R, descripton: typo}\n",
            "descripton",
        ),
        ("schema.ttl", "@prefix x: <urn:x> .\n", "must end in"),
    ],
)
def test_file_loader_refuses_bad_files(
    tmp_path: Path, name: str, text: str, message: str
) -> None:
    path = _write(tmp_path / name, text)
    with pytest.raises(SchemaSourceError, match=message):
        resolve_extraction_schema(_file_config(path))


def test_file_loader_needs_an_existing_path(tmp_path: Path) -> None:
    with pytest.raises(SchemaSourceError, match="path is required"):
        resolve_extraction_schema(_config(schema_source={"kind": "file"}))
    with pytest.raises(SchemaSourceError, match="not found"):
        resolve_extraction_schema(_file_config(tmp_path / "missing.yaml"))


def test_an_invalid_loaded_schema_names_the_source(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "schema.yaml",
        "relationship_types: [R]\nrelationship_patterns: ['GENE -R-> PERSON']\n",
    )
    with pytest.raises(SchemaSourceError, match="schema_source \\(file .*undeclared"):
        resolve_extraction_schema(_file_config(path))


# --- start-up load -------------------------------------------------------------


def test_inline_config_is_returned_as_is() -> None:
    config = Config()
    assert resolve_extraction_schema(config) is config


def test_loaded_schema_keys_as_the_same_schema_inline(tmp_path: Path) -> None:
    path = _write(tmp_path / "schema.yaml", _SCHEMA_YAML)
    resolved = resolve_extraction_schema(_file_config(path))
    inline = _config(**_INLINE)
    assert resolved.processing.graph_extraction == inline.processing.graph_extraction
    assert _keys(resolved) == _keys(inline)
    assert extraction_fingerprint(resolved) == extraction_fingerprint(inline)
    assert unit_record_fingerprint(resolved, gleaned=True) == (
        unit_record_fingerprint(inline, gleaned=True)
    )
    assert answer_settings(resolved) == answer_settings(inline)
    # Loading again changes nothing.
    assert resolve_extraction_schema(resolved) is resolved


def test_the_content_is_keyed_not_the_location(tmp_path: Path) -> None:
    here = _write(tmp_path / "a.yaml", _SCHEMA_YAML)
    (tmp_path / "moved").mkdir()
    there = _write(tmp_path / "moved" / "b.yaml", _SCHEMA_YAML)
    first = resolve_extraction_schema(_file_config(here))
    moved = resolve_extraction_schema(_file_config(there))
    assert _keys(first) == _keys(moved)
    assert extraction_fingerprint(first) == extraction_fingerprint(moved)
    edited = _write(tmp_path / "a.yaml", _SCHEMA_YAML.replace("Employment", "Jobs"))
    changed = resolve_extraction_schema(_file_config(edited))
    assert extraction_fingerprint(changed) != extraction_fingerprint(first)
    upstream = PipelineStageType.TRANSLATION.value
    assert _keys(changed)[upstream] == _keys(first)[upstream]
    assert (
        _keys(changed)[PipelineStageType.GRAPH_EXTRACTION.value]
        != _keys(first)[PipelineStageType.GRAPH_EXTRACTION.value]
    )


def test_file_entity_types_replace_the_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "schema.yaml",
        "entity_types: ['GENE: Genes', {label: DISEASE, iri: 'urn:d'}]\n"
        "relationship_types: [ASSOCIATED_WITH]\n"
        "relationship_patterns: ['GENE -ASSOCIATED_WITH-> DISEASE']\n",
    )
    resolved = resolve_extraction_schema(_file_config(path))
    schema = resolved.processing.graph_extraction.extraction_schema()
    assert [spec.label for spec in schema.entity_types] == ["GENE", "DISEASE"]
    assert schema.entity_type("disease").iri == "urn:d"  # type: ignore[union-attr]


def test_entity_types_in_both_the_file_and_the_config_fail(tmp_path: Path) -> None:
    path = _write(tmp_path / "schema.yaml", "entity_types: [GENE]\n")
    with pytest.raises(SchemaSourceError, match="Both the config and"):
        resolve_extraction_schema(_file_config(path, entity_types=["PERSON"]))
    # A file without entity_types keeps the config's.
    _write(tmp_path / "schema.yaml", "relationship_types: [R]\n")
    resolved = resolve_extraction_schema(_file_config(path, entity_types=["PERSON"]))
    assert resolved.processing.graph_extraction.entity_types == ["PERSON"]


def test_the_loaded_config_keeps_its_input(tmp_path: Path) -> None:
    # Rebuilt under provider flags, it keeps the loaded schema and the
    # caller's other settings.
    path = _write(tmp_path / "schema.yaml", _SCHEMA_YAML)
    config = _file_config(path, max_entities_per_chunk=7)
    resolved = resolve_extraction_schema(config)
    rebuilt = with_provider_overrides(resolved, backend="local", llm_model_id="m")
    extraction = rebuilt.processing.graph_extraction
    assert extraction.relationship_types == (
        resolved.processing.graph_extraction.relationship_types
    )
    assert extraction.max_entities_per_chunk == 7
    assert extraction.schema_source.kind == "inline"
    # The caller's config is left as it was.
    assert config.processing.graph_extraction.schema_source.kind == "file"


def test_a_relative_path_resolves_against_the_config_file(tmp_path: Path) -> None:
    (tmp_path / "conf").mkdir()
    _write(tmp_path / "conf" / "schema.yaml", _SCHEMA_YAML)
    config_path = _write(
        tmp_path / "conf" / "config.yaml",
        "processing:\n  graph_extraction:\n    schema_source:\n"
        "      kind: file\n      path: schema.yaml\n",
    )
    config = ConfigLoader(config_path).load_config()
    source = config.processing.graph_extraction.schema_source
    assert source.path == str((tmp_path / "conf" / "schema.yaml").resolve())
    resolve_extraction_schema(config)


def test_a_url_path_is_left_as_written(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        "processing:\n  graph_extraction:\n    schema_source:\n"
        "      kind: service\n      path: https://example.com/schema\n",
    )
    source = ConfigLoader(config_path).load_config().processing.graph_extraction
    assert source.schema_source.path == "https://example.com/schema"


def test_the_pipeline_loads_the_schema_once(tmp_path: Path, mocker: Any) -> None:
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    path = _write(tmp_path / "schema.yaml", _SCHEMA_YAML)
    calls: list[SchemaSourceContext] = []
    load_file = get_schema_source("file")

    def counting(context: SchemaSourceContext) -> SchemaDocument:
        calls.append(context)
        return load_file(context)

    mocker.patch.dict(schema_registry._REGISTRY, {"file": counting})
    config = Config.model_validate(
        {
            "backend": "local",
            "providers": {
                "rerank": {"kind": "none"},
                "graph": {"sqlite_path": str(tmp_path / "graph.sqlite3")},
                "vector": {"sqlite_path": str(tmp_path / "vector.sqlite3")},
            },
            "processing": {
                "graph_extraction": {
                    "schema_source": {"kind": "file", "path": str(path)}
                }
            },
        }
    )
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(local_directory=str(tmp_path / "cache")),
        source_directory=tmp_path,
    )
    assert len(calls) == 1
    extraction = pipeline.config.processing.graph_extraction
    assert extraction.schema_source.kind == "inline"
    assert len(extraction.relationship_types) == 2
    stage = next(
        s
        for s in pipeline.stages
        if pipeline.name_to_type_map[s.name] is PipelineStageType.GRAPH_EXTRACTION
    )
    assert stage.config is pipeline.config


# --- the loader contract -------------------------------------------------------


def test_an_unloaded_source_is_not_read_as_the_inline_schema() -> None:
    # The inline keys of an unloaded source are the defaults, not its schema.
    config = _config(schema_source={"kind": "file", "path": "schema.yaml"})
    with pytest.raises(ConfigurationError, match="resolve_extraction_schema"):
        config.processing.graph_extraction.extraction_schema()
    from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor

    with pytest.raises(ConfigurationError, match="not loaded yet"):
        BaseProcessor(config)


def test_the_loader_gets_the_options_and_the_secret_arn(custom_kind: str) -> None:
    seen: list[SchemaSourceContext] = []

    @register_schema_source(custom_kind)
    def _load(context: SchemaSourceContext) -> SchemaDocument:
        seen.append(context)
        return SchemaDocument(**_INLINE)

    arn = "arn:aws:secretsmanager:us-west-2:111122223333:secret:synthetic"
    resolved = resolve_extraction_schema(
        _config(
            schema_source={
                "kind": custom_kind,
                "options": {"graph": "synthetic", "depth": 2},
                "secret_arn": arn,
            }
        )
    )
    assert seen[0].source.options == {"graph": "synthetic", "depth": 2}
    assert seen[0].source.secret_arn == arn
    # The loaded config is inline: no option or secret survives the load.
    assert resolved.processing.graph_extraction.schema_source == SchemaSourceConfig()


def test_options_and_the_secret_key_nothing(custom_kind: str) -> None:
    plain = _config(schema_source={"kind": custom_kind})
    tuned = _config(
        schema_source={
            "kind": custom_kind,
            "options": {"graph": "other"},
            "secret_arn": "arn:aws:secretsmanager:us-west-2:111122223333:secret:x",
        }
    )
    assert _keys(plain) == _keys(tuned)
    assert extraction_fingerprint(plain) == extraction_fingerprint(tuned)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"options": {"a": 1}}, "schema_source.options is set but"),
        (
            {"secret_arn": "arn:aws:secretsmanager:us-west-2:1:secret:x"},
            "schema_source.secret_arn is set but",
        ),
        ({"kind": "svc", "secret_arn": "my-secret"}, "secret_arn"),
    ],
)
def test_source_settings_are_checked(source: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SchemaSourceConfig(**source)


class _Session:
    """Records the clients asked for; makes none."""

    def __init__(self) -> None:
        self.clients: list[tuple[str, dict[str, Any]]] = []

    def client(self, service_name: str, **kwargs: Any) -> str:
        self.clients.append((service_name, kwargs))
        return f"client:{service_name}"


def test_the_session_is_built_once_and_only_when_asked() -> None:
    built: list[_Session] = []

    def factory() -> Any:
        built.append(_Session())
        return built[-1]

    config = Config.model_validate(
        {"aws": {"region_name": "eu-west-1", "bedrock": {"region_name": "us-east-1"}}}
    )
    context = SchemaSourceContext(
        SchemaSourceConfig(kind="svc"), config, session_factory=factory
    )
    assert built == []
    assert context.client("s3") == "client:s3"
    assert context.client("bedrock-runtime") == "client:bedrock-runtime"
    assert context.client("sts", region_name="ap-south-1") == "client:sts"
    assert len(built) == 1 and context.boto_session() is built[0]
    assert built[0].clients == [
        ("s3", {"region_name": "eu-west-1"}),
        ("bedrock-runtime", {"region_name": "us-east-1"}),
        ("sts", {"region_name": "ap-south-1"}),
    ]


def test_the_run_session_reaches_the_loader(custom_kind: str) -> None:
    from unified_kg_rag.adapters.providers import Providers

    sessions: list[Any] = []

    @register_schema_source(custom_kind)
    def _load(context: SchemaSourceContext) -> SchemaDocument:
        sessions.append(context.boto_session())
        return SchemaDocument(**_INLINE)

    config = _config(schema_source={"kind": custom_kind})
    session = _Session()
    resolve_extraction_schema(config, providers=Providers(config, boto_session=session))  # type: ignore[arg-type]
    resolve_extraction_schema(config, boto_session=session)  # type: ignore[arg-type]
    assert sessions == [session, session]


@pytest.mark.parametrize(
    ("path", "shown"),
    [
        (
            "https://user:token@schemas.example.com:8443/v1/onto.ttl?sig=abc#frag",
            "https://schemas.example.com:8443/v1/onto.ttl",
        ),
        ("s3://bucket/schemas/onto.yaml", "s3://bucket/schemas/onto.yaml"),
        ("/srv/schemas/onto.yaml", "/srv/schemas/onto.yaml"),
        (None, "<no path>"),
    ],
)
def test_locations_are_shown_without_credentials(path: str | None, shown: str) -> None:
    assert redact_location(path) == shown


def test_errors_never_show_a_url_credential(custom_kind: str) -> None:
    register_schema_source(custom_kind)(
        lambda context: SchemaDocument(relationship_patterns=["GENE -R-> PERSON"])
    )
    path = "https://user:token@schemas.example.com/onto?sig=secret-sig"
    with pytest.raises(SchemaSourceError) as caught:
        resolve_extraction_schema(
            _config(schema_source={"kind": custom_kind, "path": path})
        )
    message = str(caught.value)
    assert "https://schemas.example.com/onto" in message
    assert "token" not in message and "secret-sig" not in message
    with pytest.raises(SchemaSourceError) as caught:
        resolve_extraction_schema(_config(schema_source={"kind": "file", "path": path}))
    message = str(caught.value)
    assert "local path only" in message
    assert "token" not in message and "secret-sig" not in message


@pytest.mark.parametrize(
    "path", ["s3://bucket/schema.yaml", "https://example.com/schema.yaml"]
)
def test_the_file_loader_refuses_urls(path: str) -> None:
    with pytest.raises(SchemaSourceError, match="is a URL; kind 'file' reads a local"):
        _load_file(path)


def test_the_file_loader_takes_no_options(tmp_path: Path) -> None:
    path = _write(tmp_path / "schema.yaml", _SCHEMA_YAML)
    with pytest.raises(SchemaSourceError, match="takes no schema_source.options"):
        _load_file(str(path), options={"a": 1})


def test_a_directory_is_an_unreadable_schema_file(tmp_path: Path) -> None:
    folder = tmp_path / "schema.yaml"
    folder.mkdir()
    with pytest.raises(SchemaSourceError, match="Cannot read schema file"):
        _load_file(str(folder))
