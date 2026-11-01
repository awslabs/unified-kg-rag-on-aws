# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extraction schema draft of ``run-prompt-tuning --schema-out``.

The draft lists the corpus profile's entity types and the relationship
types the grounded examples state most often, with a pattern only where
the examples support it; written as a ``schema_source`` file, it loads,
validates and resolves to the same schema, and a fixed model gives the same
file byte for byte.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.schema_sources import get_schema_source
from unified_kg_rag.application.bootstrap import resolve_extraction_schema
from unified_kg_rag.application.cli import run_prompt_tuning
from unified_kg_rag.application.prompts import tuner as tuner_module
from unified_kg_rag.application.prompts.tuner import PromptTuner, SchemaDraft
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    Relationship,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.domain.models.schema import ExtractionSchema, RelationshipPattern

pytestmark = pytest.mark.unit


def _entity(name: str, type_: str) -> Entity:
    return Entity(id=name.lower(), name=name, type=type_)


def _rel(source: str, target: str, type_: str) -> Relationship:
    return Relationship(
        id=f"{source}-{type_}-{target}",
        source_id=source.lower(),
        target_id=target.lower(),
        source_name=source,
        target_name=target,
        type=type_,
    )


def _record(
    unit: str, entities: list[Entity], relationships: list[Relationship]
) -> TextUnitExtraction:
    return TextUnitExtraction(
        text_unit_id=unit, entities=entities, relationships=relationships
    )


ORGS = [_entity("Vendor", "ORGANIZATION"), _entity("Buyer", "organization")]


# --------------------------------------------------------------------------- #
# SchemaDraft.from_examples
# --------------------------------------------------------------------------- #


def test_relationship_types_are_counted_across_spellings_and_ranked() -> None:
    records = [
        _record(
            "t1",
            ORGS,
            [
                _rel("Vendor", "Buyer", "supplies"),
                _rel("Buyer", "Vendor", "Supplies"),
                _rel("Vendor", "Buyer", "works for"),
                _rel("Vendor", "Buyer", ""),
            ],
        ),
        _record("t2", ORGS, [_rel("Vendor", "Buyer", "SUPPLIES")]),
        _record("t3", ORGS, [_rel("Buyer", "Vendor", "WORKS_FOR")]),
        _record("t4", ORGS, [_rel("Buyer", "Vendor", "audits")]),
    ]
    draft = SchemaDraft.from_examples(["ORGANIZATION", "PRODUCT"], records)
    # Most frequent first, ties by label; spellings of one type are one,
    # written uppercase with '_' separators; an empty type is not counted.
    assert draft.relationship_types == ["SUPPLIES", "WORKS_FOR", "AUDITS"]
    assert draft.relationship_type_counts == {
        "SUPPLIES": 3,
        "WORKS_FOR": 2,
        "AUDITS": 1,
    }


def test_only_the_most_frequent_relationship_types_are_kept(monkeypatch) -> None:
    monkeypatch.setattr(tuner_module, "MAX_DRAFT_RELATIONSHIP_TYPES", 2)
    rels = [_rel("Vendor", "Buyer", t) for t in ("A", "B", "B", "C", "C", "C")]
    draft = SchemaDraft.from_examples([], [_record("t1", ORGS, rels)])
    assert draft.relationship_types == ["C", "B"]


def test_entity_types_come_from_the_profile_once_each() -> None:
    draft = SchemaDraft.from_examples(
        ["ORGANIZATION", "Organization", "LEGAL ENTITY", "DRUG: medicines", ""], []
    )
    assert draft.entity_types == ["ORGANIZATION", "LEGAL ENTITY", "DRUG"]
    assert draft.relationship_types == []
    assert draft.relationship_patterns == []


def test_a_pattern_needs_support_for_every_endpoint_pair() -> None:
    product = _entity("Gadget", "product")
    depot = _entity("Depot", "warehouse")
    records = [
        # SUPPLIES: organization -> organization twice: a pattern.
        _record("t1", ORGS, [_rel("Vendor", "Buyer", "SUPPLIES")]),
        _record("t2", ORGS, [_rel("Buyer", "Vendor", "supplies")]),
        # STOCKS: organization -> product twice, product -> organization
        # once: no pattern (it would flag that statement off pattern).
        _record("t3", [*ORGS, product], [_rel("Vendor", "Gadget", "STOCKS")]),
        _record("t4", [*ORGS, product], [_rel("Buyer", "Gadget", "STOCKS")]),
        _record("t5", [*ORGS, product], [_rel("Gadget", "Vendor", "STOCKS")]),
        # SHIPS: an endpoint of a type the profile lacks, or unlisted: no
        # pattern either.
        _record("t6", [depot, *ORGS], [_rel("Depot", "Vendor", "SHIPS")]),
        _record("t7", [depot, *ORGS], [_rel("Depot", "Buyer", "SHIPS")]),
        _record("t8", ORGS, [_rel("Ghost", "Buyer", "SHIPS")]),
    ]
    draft = SchemaDraft.from_examples(["ORGANIZATION", "PRODUCT"], records)
    assert [str(p) for p in draft.relationship_patterns] == [
        "ORGANIZATION -SUPPLIES-> ORGANIZATION"
    ]


def test_a_draft_is_a_valid_schema() -> None:
    draft = SchemaDraft.from_examples(
        ["ORGANIZATION"],
        [
            _record("t1", ORGS, [_rel("Vendor", "Buyer", "SUPPLIES")]),
            _record("t2", ORGS, [_rel("Buyer", "Vendor", "SUPPLIES")]),
        ],
    )
    schema = draft.extraction_schema()
    assert schema.mode == "align"
    assert [s.label for s in schema.relationship_types] == ["SUPPLIES"]
    assert len(schema.relationship_patterns) == 1


# --------------------------------------------------------------------------- #
# The schema file
# --------------------------------------------------------------------------- #


def _load(path: Path) -> ExtractionSchema:
    """The schema a config naming ``path`` as its schema file resolves to."""
    config = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_mode": "align",
                    "schema_source": {"kind": "file", "path": str(path)},
                }
            }
        }
    )
    resolved = resolve_extraction_schema(config)
    return resolved.processing.graph_extraction.extraction_schema()


def test_the_file_round_trips_through_the_schema_source(tmp_path: Path) -> None:
    records = [
        _record(
            "t1",
            ORGS,
            [
                _rel("Vendor", "Buyer", "SUPPLIES"),
                # Labels YAML or the string forms would misread.
                _rel("Vendor", "Buyer", "yes"),
                _rel("Vendor", "Buyer", "OWNS:PART"),
                _rel("Vendor", "Buyer", "A->B"),
            ],
        ),
        _record(
            "t2",
            ORGS,
            [
                _rel("Buyer", "Vendor", "SUPPLIES"),
                _rel("Buyer", "Vendor", "A->B"),
            ],
        ),
    ]
    draft = SchemaDraft.from_examples(["ORGANIZATION", "NULL"], records)
    path = tmp_path / "schema.yaml"
    path.write_text(draft.to_yaml("schema.yaml"), encoding="utf-8")

    document = get_schema_source("file")(
        SchemaSourceConfig(kind="file", path=str(path))
    )
    assert document.entity_types == ["ORGANIZATION", "NULL"]
    schema = _load(path)
    assert schema == draft.extraction_schema()
    assert [s.label for s in schema.relationship_types] == [
        "A->B",
        "SUPPLIES",
        "OWNS:PART",
        "YES",
    ]
    assert schema.relationship_patterns == (
        RelationshipPattern(
            source_type="ORGANIZATION", relation="A->B", target_type="ORGANIZATION"
        ),
        RelationshipPattern(
            source_type="ORGANIZATION",
            relation="SUPPLIES",
            target_type="ORGANIZATION",
        ),
    )


def test_the_file_explains_how_to_load_it_and_never_says_strict(
    tmp_path: Path,
) -> None:
    draft = SchemaDraft.from_examples(
        ["ORGANIZATION"], [_record("t1", ORGS, [_rel("Vendor", "Buyer", "SUPPLIES")])]
    )
    text = draft.to_yaml("drafts/schema.yaml")
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert '#       schema_source: {kind: file, path: "drafts/schema.yaml"}' in header
    assert "#       schema_mode: align" in header
    assert "#   SUPPLIES: 1" in header
    assert "schema_mode" not in yaml.safe_load(text)
    assert not re.search(r"^\s*schema_mode:\s*strict", text, re.MULTILINE)


def test_without_profile_entity_types_the_file_keeps_the_configs(
    tmp_path: Path,
) -> None:
    draft = SchemaDraft.from_examples(
        [], [_record("t1", ORGS, [_rel("Vendor", "Buyer", "SUPPLIES")])]
    )
    assert "entity_types" not in draft.document()
    path = tmp_path / "schema.yml"
    path.write_text(draft.to_yaml(), encoding="utf-8")
    schema = _load(path)
    # The default entity types stay.
    assert schema.entity_type("PERSON") is not None
    assert [s.label for s in schema.relationship_types] == ["SUPPLIES"]


# --------------------------------------------------------------------------- #
# Prompt tuning with a fixed model
# --------------------------------------------------------------------------- #

CORPUS = [
    "Vendor supplies Buyer. Depot stocks Gadget.",
    "Carrier supplies Vendor. Depot stocks Gadget.",
    "Vendor partners with Carrier.",
]
TYPES = {
    "Vendor": "ORGANIZATION",
    "Buyer": "organization",
    "Carrier": "Organization",
    "Depot": "warehouse",
    "Gadget": "product",
}
# Each document spells the relation its own way.
SPELLINGS = {"supplies": ["SUPPLIES", "supplies"], "stocks": ["Stocks", "STOCKS"]}
_SENTENCE = re.compile(r"(\w+) (supplies|stocks|partners with) (\w+)\.")


def _respond(system: str, human: str) -> str:
    if "expert corpus analyst" in system:
        return json.dumps(
            {
                "domain": "supply chain",
                "language": "English",
                "persona": "You are a supply chain analyst.",
                "entity_types": ["ORGANIZATION", "PRODUCT", "Organization"],
            }
        )
    assert "knowledge graph extraction expert" in system, system[:80]
    text = re.search(r"<input_text>\n(.*?)\n</input_text>", human, re.DOTALL)
    assert text is not None
    source = text.group(1)
    document = next(i for i, doc in enumerate(CORPUS) if doc in source)
    entities, relationships = [], []
    for sentence in _SENTENCE.finditer(source):
        head, verb, tail = sentence.groups()
        for name in (head, tail):
            entities.append(
                f"<entity><name>{name}</name><type>{TYPES[name]}</type>"
                f"<description>{name} in the supply chain</description>"
                f"<confidence>9</confidence><source_text>{sentence.group(0)}"
                "</source_text></entity>"
            )
        spelling = SPELLINGS.get(verb, [verb, verb])[document % 2]
        relationships.append(
            f"<relationship><source>{head}</source><target>{tail}</target>"
            f"<type>{spelling}</type><description>{head} {verb} {tail}"
            "</description><strength>7</strength>"
            f"<source_text>{sentence.group(0)}</source_text></relationship>"
        )
    return (
        f"<entities>{''.join(entities)}</entities>"
        f"<relationships>{''.join(relationships)}</relationships>"
    )


def _tuner(config: Config) -> PromptTuner:
    return PromptTuner(
        config,
        providers=Providers(
            config,
            boto_session=MagicMock(),
            llm_factory=ScriptedLLMFactory(_respond),
        ),
    )


async def test_tuning_drafts_the_schema_of_its_examples() -> None:
    result = await _tuner(Config()).tune(CORPUS, schema_draft=True)
    draft: SchemaDraft = result["schema_draft"]
    assert draft.entity_types == ["ORGANIZATION", "PRODUCT"]
    assert draft.relationship_type_counts == {
        "STOCKS": 2,
        "SUPPLIES": 2,
        "PARTNERS_WITH": 1,
    }
    # SUPPLIES joins two organizations twice; STOCKS starts at a warehouse
    # (not a profile type) and PARTNERS_WITH is stated once.
    assert [str(p) for p in draft.relationship_patterns] == [
        "ORGANIZATION -SUPPLIES-> ORGANIZATION"
    ]
    # The tuned prompts are what tuning without a draft gives.
    plain = await _tuner(Config()).tune(CORPUS)
    assert "schema_draft" not in plain
    assert plain == {k: v for k, v in result.items() if k != "schema_draft"}


def _main(tmp_path: Path, mocker, name: str, *extra: str) -> tuple[int, Path]:
    """Run the CLI over ``CORPUS`` in ``tmp_path`` with ``--schema-out
    tmp_path/name``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "corpus"
    source.mkdir(exist_ok=True)
    for index, text in enumerate(CORPUS):
        (source / f"{index}.txt").write_text(text, encoding="utf-8")
    mocker.patch.object(run_prompt_tuning, "load_cli_config", return_value=Config())
    mocker.patch.object(run_prompt_tuning, "PromptTuner", side_effect=_tuner)
    out = tmp_path / name
    mocker.patch(
        "sys.argv",
        [
            "run-prompt-tuning",
            "--source-directory",
            str(source),
            "--output-path",
            str(tmp_path / "prompts.yaml"),
            "--schema-out",
            str(out),
            *extra,
        ],
    )
    return run_prompt_tuning.main(), out


def test_schema_out_writes_the_same_loadable_draft_each_run(
    tmp_path: Path, mocker, capsys: Any
) -> None:
    code, first = _main(
        tmp_path / "a", mocker, "schema.yaml", "--output-format", "json"
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_path"] == str(first)
    code, second = _main(tmp_path / "b", mocker, "schema.yaml")
    assert code == 0
    assert first.read_bytes() == second.read_bytes()
    # The tuned prompts file holds no draft.
    prompts = yaml.safe_load((tmp_path / "a" / "prompts.yaml").read_text())
    assert set(prompts) == {"profile", "custom_prompts"}

    schema = _load(first)
    assert schema.mode == "align"
    assert [s.label for s in schema.entity_types] == ["ORGANIZATION", "PRODUCT"]
    assert [s.label for s in schema.relationship_types] == [
        "STOCKS",
        "SUPPLIES",
        "PARTNERS_WITH",
    ]


def test_schema_out_must_be_yaml(tmp_path: Path, mocker) -> None:
    with pytest.raises(SystemExit) as exc:
        _main(tmp_path, mocker, "schema.json")
    assert exc.value.code == 2
