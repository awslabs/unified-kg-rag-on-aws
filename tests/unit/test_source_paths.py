# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every file-reading source kind (schema ``file``/``owl``, seed
``jsonl``/``csv``/``rdf``) resolves its ``path`` through
:func:`.source_files.locations.resolve_source_path`, so a ``file://`` URL
reads what the plain path reads and other schemes are refused alike."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.extras import requires_ontology
from tests.fixtures.seed_graphs import EDGES, EDGES_CSV, NODES, NODES_CSV, lines
from unified_kg_rag.adapters.schema_sources import (
    SchemaSourceContext,
    get_schema_source,
)
from unified_kg_rag.adapters.seed_sources import SeedSourceContext, get_seed_source
from unified_kg_rag.adapters.source_files.locations import (
    SourceTerms,
    resolve_source_path,
)
from unified_kg_rag.application.bootstrap import resolve_extraction_schema
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import SchemaSourceConfig, SeedGraphConfig
from unified_kg_rag.shared.exceptions import (
    ConfigurationError,
    SchemaSourceError,
    SeedGraphError,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[1] / "fixtures"
ONTOLOGY = FIXTURES / "ontologies" / "procure.ttl"
ABOX = FIXTURES / "seeds" / "procure-abox.ttl"
SCHEMA_YAML = 'relationship_types:\n  - "WORKS_FOR: Employment"\n'


class _Error(ConfigurationError):
    pass


TERMS = SourceTerms(noun="Seed graph", error=_Error)


# --- the resolver ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("onto.ttl", Path("onto.ttl")),
        ("/srv/onto.ttl", Path("/srv/onto.ttl")),
        ("file:///srv/onto.ttl", Path("/srv/onto.ttl")),
        ("FILE:///srv/onto.ttl", Path("/srv/onto.ttl")),
        ("file://localhost/srv/onto.ttl", Path("/srv/onto.ttl")),
        ("file:///srv/my%20onto.ttl", Path("/srv/my onto.ttl")),
        ("s3://bucket/onto.ttl", "s3://bucket/onto.ttl"),
        ("S3://bucket/prefix/", "S3://bucket/prefix/"),
    ],
)
def test_a_path_resolves_to_a_local_path_or_an_s3_url(
    path: str, expected: Path | str
) -> None:
    resolved = resolve_source_path(path, refusal="refused", terms=TERMS)
    assert resolved == expected
    assert type(resolved) is type(expected)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("https://example.org/onto.ttl", "^refused$"),
        ("ftp://example.org/onto.ttl", "^refused$"),
        ("file://host/onto.ttl", "^refused$"),
        ("s3://Not_A_Bucket/onto.ttl", "does not name an S3 bucket"),
    ],
)
def test_other_locations_are_refused_in_the_callers_terms(
    path: str, message: str
) -> None:
    with pytest.raises(_Error, match=message):
        resolve_source_path(path, refusal="refused", terms=TERMS)


# --- every kind ---------------------------------------------------------------------


def _schema(kind: str) -> Callable[[str], Any]:
    def load(path: str) -> Any:
        return get_schema_source(kind)(
            SchemaSourceContext(SchemaSourceConfig(kind=kind, path=path), Config())
        )

    return load


def _seed(kind: str, config: Callable[[], Config] = Config) -> Callable[[str], Any]:
    def load(path: str) -> Any:
        seed = SeedGraphConfig(name="procure", kind=kind, path=path)
        return get_seed_source(kind)(SeedSourceContext(seed, config()))

    return load


def _rdf_config() -> Config:
    return resolve_extraction_schema(
        Config.model_validate(
            {
                "processing": {
                    "graph_extraction": {
                        "schema_source": {"kind": "owl", "path": str(ONTOLOGY)}
                    }
                }
            }
        )
    )


def _schema_file(tmp_path: Path) -> Path:
    path = tmp_path / "schema.yaml"
    path.write_text(SCHEMA_YAML, encoding="utf-8")
    return path


def _jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "procure.jsonl"
    path.write_text(lines(*NODES, *EDGES), encoding="utf-8")
    return path


def _csv(tmp_path: Path) -> Path:
    root = tmp_path / "csv"
    root.mkdir()
    (root / "nodes.csv").write_text(NODES_CSV, encoding="utf-8")
    (root / "edges.csv").write_text(EDGES_CSV, encoding="utf-8")
    return root


KINDS = [
    pytest.param(_schema("file"), _schema_file, SchemaSourceError, id="file"),
    pytest.param(
        _schema("owl"),
        lambda _: ONTOLOGY,
        SchemaSourceError,
        id="owl",
        marks=requires_ontology,
    ),
    pytest.param(_seed("jsonl"), _jsonl, SeedGraphError, id="jsonl"),
    pytest.param(_seed("csv"), _csv, SeedGraphError, id="csv"),
    pytest.param(
        _seed("rdf", _rdf_config),
        lambda _: ABOX,
        SeedGraphError,
        id="rdf",
        marks=requires_ontology,
    ),
]


@pytest.mark.parametrize(("load", "make", "error"), KINDS)
def test_a_file_url_reads_what_the_path_reads(
    load: Callable[[str], Any],
    make: Callable[[Path], Path],
    error: type[ConfigurationError],
    tmp_path: Path,
) -> None:
    path = make(tmp_path).resolve()
    assert load(path.as_uri()) == load(str(path))
    assert load("file://localhost" + path.as_posix()) == load(str(path))


@pytest.mark.parametrize(("load", "make", "error"), KINDS)
def test_every_kind_refuses_other_urls_alike(
    load: Callable[[str], Any],
    make: Callable[[Path], Path],
    error: type[ConfigurationError],
    tmp_path: Path,
) -> None:
    path = make(tmp_path).resolve()
    for url in ("https://example.org" + path.as_posix(), "file://host" + str(path)):
        with pytest.raises(error, match=r"is a URL; kind '\w+' reads a local"):
            load(url)
        with pytest.raises(error, match=r"\(or a file:// URL\)"):
            load(url)
