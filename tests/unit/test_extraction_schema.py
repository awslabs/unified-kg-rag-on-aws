# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extraction schema: its config forms, validation, label keys, and how
it keys the extraction caches."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from unified_kg_rag.application.config import ConfigLoader
from unified_kg_rag.domain.ingestion import cache_keys
from unified_kg_rag.domain.ingestion.cache_keys import (
    answer_settings,
    changed_settings,
    extraction_fingerprint,
    extraction_settings,
    stage_input_fingerprint,
    unit_record_fingerprint,
)
from unified_kg_rag.domain.models import (
    Config,
    EntityTypeSpec,
    ExtractionSchema,
    PipelineStageType,
    RelationshipPattern,
    RelationshipTypeSpec,
    schema_key,
)
from unified_kg_rag.domain.prompts import GraphExtractionPrompt
from unified_kg_rag.shared.exceptions import ConfigurationError

pytestmark = pytest.mark.unit

_P = PipelineStageType


def _config(**extraction: Any) -> Config:
    return Config.model_validate({"processing": {"graph_extraction": extraction}})


_SCHEMA: dict[str, Any] = {
    "entity_types": [
        "PERSON: People",
        {"label": "ORGANIZATION", "description": "Companies", "iri": "urn:x:Org"},
    ],
    "relationship_types": [
        "WORKS_FOR: Employment",
        {"label": "partner of", "iri": "urn:x:partnerOf"},
    ],
    "relationship_patterns": [
        "PERSON -WORKS_FOR-> ORGANIZATION",
        {
            "source_type": "organization",
            "relation": "PARTNER_OF",
            "target_type": "Organization",
        },
    ],
}


# --- schema_key ----------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    ["WORKS_FOR", "works for", " Works_For ", "works  for", "WORKS__FOR", "Ｗorks_for"],
)
def test_schema_key_ignores_case_and_separators(spelling: str) -> None:
    assert schema_key(spelling) == "works_for"


@pytest.mark.parametrize(
    ("label", "declared"),
    [
        ("PERSONS", "PERSON"),
        ("Org", "ORGANIZATION"),
        ("works-for", "works for"),
        ("employs", "WORKS_FOR"),
        ("WORKSFOR", "WORKS_FOR"),
        ("_PERSON", "PERSON"),
    ],
)
def test_schema_key_never_matches_near_misses(label: str, declared: str) -> None:
    assert schema_key(label) != schema_key(declared)


def test_schema_key_is_idempotent() -> None:
    for label in ("Works For", " a_ b ", "ß-Type"):
        assert schema_key(schema_key(label)) == schema_key(label)


# --- config forms --------------------------------------------------------------


def test_default_schema_is_the_entity_types_alone() -> None:
    config = Config()
    extraction = config.processing.graph_extraction
    assert extraction.relationship_types == []
    assert extraction.relationship_patterns == []
    assert extraction.schema_mode == "align"
    assert extraction.schema_source.kind == "inline"
    assert extraction.schema_source.path is None
    schema = extraction.extraction_schema()
    assert [spec.label for spec in schema.entity_types] == [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "CONCEPT",
        "OBJECT",
        "EVENT",
        "TEMPORAL",
    ]
    assert schema.entity_types[0].description == "Names, individuals, roles, titles"


def test_string_and_mapping_forms_build_one_schema() -> None:
    schema = _config(**_SCHEMA).processing.graph_extraction.extraction_schema()
    assert schema.entity_types == (
        EntityTypeSpec(label="PERSON", description="People"),
        EntityTypeSpec(label="ORGANIZATION", description="Companies", iri="urn:x:Org"),
    )
    assert schema.relationship_types[1] == RelationshipTypeSpec(
        label="partner of", iri="urn:x:partnerOf"
    )
    assert schema.relationship_patterns[0] == RelationshipPattern(
        source_type="PERSON", relation="WORKS_FOR", target_type="ORGANIZATION"
    )
    assert schema.entity_type(" organization ") is schema.entity_types[1]
    assert schema.relationship_type("PARTNER_OF") is schema.relationship_types[1]
    assert schema.entity_type("ORG") is None


def test_string_items_are_kept_as_written() -> None:
    # The dump (and with it every key) of a string-form config is what it
    # was before the mapping form existed.
    extraction = _config(**_SCHEMA).processing.graph_extraction
    dumped = extraction.model_dump(mode="json")
    assert dumped["entity_types"][0] == "PERSON: People"
    assert dumped["relationship_patterns"][0] == "PERSON -WORKS_FOR-> ORGANIZATION"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PERSON -WORKS_FOR-> ORGANIZATION", ("PERSON", "WORKS_FOR", "ORGANIZATION")),
        ("  PERSON  - works for ->  ORG  ", ("PERSON", "works for", "ORG")),
        ("PERSON -CO-AUTHOR-> PERSON", ("PERSON", "CO-AUTHOR", "PERSON")),
        (
            "research lab -PART_OF-> university",
            ("research lab", "PART_OF", "university"),
        ),
    ],
)
def test_pattern_string_form(text: str, expected: tuple[str, str, str]) -> None:
    pattern = RelationshipPattern.parse(text)
    assert (pattern.source_type, pattern.relation, pattern.target_type) == expected
    assert RelationshipPattern.parse(str(pattern)) == pattern


@pytest.mark.parametrize(
    "text",
    ["PERSON WORKS_FOR ORG", "PERSON-WORKS_FOR->ORG", "-R-> B", "A -R->", "A -> B"],
)
def test_malformed_pattern_strings_are_refused(text: str) -> None:
    with pytest.raises(ValidationError, match="relationship_patterns"):
        _config(
            entity_types=["A", "B", "PERSON", "ORG"],
            relationship_types=["R", "WORKS_FOR"],
            relationship_patterns=[text],
        )


@pytest.mark.parametrize(
    ("extraction", "message"),
    [
        ({"entity_types": ["PERSON", "person: again"]}, "name the same type"),
        ({"entity_types": ["WORKS FOR", "works_for"]}, "name the same type"),
        ({"relationship_types": ["R", " r "]}, "name the same type"),
        ({"relationship_types": [": no label"]}, "has no label"),
        (
            {
                "relationship_types": ["R"],
                "relationship_patterns": ["PERSON -R-> GENE"],
            },
            "undeclared entity type 'GENE'",
        ),
        (
            {"relationship_patterns": ["PERSON -EMPLOYS-> ORGANIZATION"]},
            "undeclared relationship type 'EMPLOYS'",
        ),
        (
            {
                "entity_types": [],
                "relationship_types": ["R"],
                "relationship_patterns": ["A -R-> B"],
            },
            "undeclared entity type 'A'",
        ),
        (
            {"schema_source": {"kind": "file"}, "relationship_types": ["R"]},
            "leave them unset",
        ),
        (
            {"schema_source": {"kind": "file"}, "relationship_patterns": ["A -R-> B"]},
            "leave them unset",
        ),
        ({"schema_source": {"path": "schema.yaml"}}, "kind is 'inline'"),
        ({"schema_source": {"kind": "File"}}, "should match pattern"),
        ({"schema_mode": "annotate"}, "schema_mode"),
        ({"entity_types": [{"label": "A", "descripton": "typo"}]}, "descripton"),
        ({"relationship_types": [{"label": "R", "uri": "urn:r"}]}, "uri"),
        ({"entity_types": [{"label": "A", "iri": "urn:a b"}]}, "iri"),
        ({"entity_types": [{"label": "  "}]}, "label"),
        (
            {"relationship_patterns": [{"source_type": "A", "relation": "R"}]},
            "target_type",
        ),
    ],
)
def test_invalid_schemas_are_refused(extraction: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _config(**extraction)


def test_a_mapping_error_names_only_its_own_fields() -> None:
    with pytest.raises(ValidationError) as caught:
        _config(entity_types=[{"label": "A", "bogus": 1}])
    assert caught.value.error_count() == 1
    assert "valid string" not in str(caught.value)


def test_legacy_blank_entity_labels_are_still_skipped() -> None:
    schema = _config(entity_types=[": no label", "PERSON"]).processing.graph_extraction
    assert [spec.label for spec in schema.extraction_schema().entity_types] == [
        "PERSON"
    ]


def test_assignment_is_validated() -> None:
    extraction = _config(**_SCHEMA).processing.graph_extraction
    with pytest.raises(ValidationError, match="undeclared"):
        extraction.relationship_types = ["WORKS_FOR"]
    with pytest.raises(ValidationError, match="kind is 'inline'"):
        extraction.schema_source.path = "x.yaml"


def test_environment_variables_set_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UKR__PROCESSING__GRAPH_EXTRACTION__SCHEMA_MODE", "strict")
    monkeypatch.setenv(
        "UKR__PROCESSING__GRAPH_EXTRACTION__RELATIONSHIP_TYPES",
        '["WORKS_FOR: Employment", {"label": "PART_OF", "iri": "urn:p"}]',
    )
    config = ConfigLoader().load_config()
    extraction = config.processing.graph_extraction
    assert extraction.schema_mode == "strict"
    assert extraction.extraction_schema().relationship_types[1].iri == "urn:p"


def test_an_invalid_environment_schema_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "UKR__PROCESSING__GRAPH_EXTRACTION__RELATIONSHIP_PATTERNS"
    monkeypatch.setenv(variable, '["PERSON -WORKS_FOR-> ORGANIZATION"]')
    with pytest.raises(ConfigurationError, match=variable):
        ConfigLoader().load_config()


def test_unknown_item_keys_in_a_file_fail_the_load(tmp_path: Any) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "processing:\n  graph_extraction:\n    relationship_types:\n"
        "      - {label: R, descriptoin: typo}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="descriptoin"):
        ConfigLoader(path).load_config()


# --- the domain model ----------------------------------------------------------


def test_schema_digest_follows_content_not_mode() -> None:
    def schema(**extraction: Any) -> ExtractionSchema:
        return _config(**extraction).processing.graph_extraction.extraction_schema()

    base = schema(**_SCHEMA)
    assert base.digest == schema(**_SCHEMA, schema_mode="strict").digest
    no_iri = dict(_SCHEMA, entity_types=["PERSON: People", "ORGANIZATION: Companies"])
    assert base.digest != schema(**no_iri).digest
    assert len(base.digest) == 64


def test_schema_model_refuses_inconsistency() -> None:
    with pytest.raises(ValidationError, match="undeclared"):
        ExtractionSchema(
            relationship_patterns=(
                RelationshipPattern(source_type="A", relation="R", target_type="B"),
            )
        )
    spec = EntityTypeSpec(label="A")
    with pytest.raises(ValidationError, match="frozen"):
        spec.label = "B"  # type: ignore[misc]


# --- prompt --------------------------------------------------------------------


def test_spec_entity_types_render_in_the_prompt() -> None:
    config = _config(
        entity_types=[
            "GENE: Genes",
            {"label": "DISEASE", "description": "Disorders", "iri": "urn:d"},
            {"label": "DRUG"},
        ]
    )
    block = GraphExtractionPrompt.config_variables(config)["entity_types"]
    assert block == "- **GENE**: Genes\n- **DISEASE**: Disorders\n- **DRUG**"


# --- cache keys ----------------------------------------------------------------


def _keys(config: Config) -> dict[str, str]:
    return {
        stage.value: stage_input_fingerprint(config, stage, "corpus")
        for stage in PipelineStageType
    }


def test_new_fields_at_their_default_key_nothing() -> None:
    config = Config()
    recorded = extraction_settings(config).values
    for path in cache_keys._KEYED_ONCE_SET:
        assert path not in recorded
        assert path not in answer_settings(config)
    # Set to their defaults explicitly: still nothing.
    explicit = _config(
        relationship_types=[],
        relationship_patterns=[],
        schema_mode="align",
        schema_source={"kind": "inline"},
    )
    assert _keys(explicit) == _keys(config)
    assert extraction_fingerprint(explicit) == extraction_fingerprint(config)
    assert unit_record_fingerprint(explicit, gleaned=True) == (
        unit_record_fingerprint(config, gleaned=True)
    )


@pytest.mark.parametrize(
    ("extraction", "changed"),
    [
        (
            {"relationship_types": ["WORKS_FOR"]},
            ["processing.graph_extraction.relationship_types"],
        ),
        ({"schema_mode": "strict"}, ["processing.graph_extraction.schema_mode"]),
        ({"schema_mode": "off"}, ["processing.graph_extraction.schema_mode"]),
        (
            _SCHEMA,
            [
                "processing.graph_extraction.entity_types",
                "processing.graph_extraction.relationship_patterns",
                "processing.graph_extraction.relationship_types",
            ],
        ),
        (
            {"entity_types": ["PERSON: People", {"label": "ORG", "iri": "urn:o"}]},
            ["processing.graph_extraction.entity_types"],
        ),
    ],
)
def test_a_schema_change_rekeys_extraction_onward(
    extraction: dict[str, Any], changed: list[str]
) -> None:
    before, after = Config(), _config(**extraction)
    keys_before, keys_after = _keys(before), _keys(after)
    for stage in PipelineStageType:
        upstream = list(PipelineStageType).index(stage) < list(PipelineStageType).index(
            _P.GRAPH_EXTRACTION
        )
        assert (keys_before[stage.value] == keys_after[stage.value]) is upstream
    assert extraction_fingerprint(before) != extraction_fingerprint(after)
    assert unit_record_fingerprint(before, gleaned=False) != (
        unit_record_fingerprint(after, gleaned=False)
    )
    assert (
        changed_settings(extraction_settings(before), extraction_settings(after))
        == changed
    )
    # Back to the defaults: changed again, against the record that had it.
    assert changed_settings(extraction_settings(after), extraction_settings(before))


def test_the_source_location_keys_nothing() -> None:
    # Before start-up loads it, a source config keys its kind; where the
    # source reads does not count (the loaded content does).
    here = _config(schema_source={"kind": "file", "path": "a/schema.yaml"})
    there = _config(schema_source={"kind": "file", "path": "b/schema.yaml"})
    assert _keys(here) == _keys(there)
    assert extraction_fingerprint(here) == extraction_fingerprint(there)
    assert answer_settings(here) == answer_settings(there)
    assert _keys(here) != _keys(Config())
