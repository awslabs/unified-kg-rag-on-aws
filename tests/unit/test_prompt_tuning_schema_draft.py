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

from tests.fixtures.extras import requires_ontology
from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.schema_sources import (
    SchemaSourceContext,
    get_schema_source,
)
from unified_kg_rag.adapters.schema_sources import registry as schema_registry
from unified_kg_rag.application.bootstrap import resolve_extraction_schema
from unified_kg_rag.application.cli import run_prompt_tuning
from unified_kg_rag.application.prompts import tuner as tuner_module
from unified_kg_rag.application.prompts.tuner import PromptTuner, SchemaDraft
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    Relationship,
    SchemaDocument,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.domain.models.schema import (
    EntityTypeSpec,
    ExtractionSchema,
    RelationshipPattern,
    RelationshipTypeSpec,
)
from unified_kg_rag.shared.exceptions import SchemaSourceError

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


def _labels(specs: list[Any]) -> list[str]:
    return [spec.label for spec in specs]


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
    assert _labels(draft.relationship_types) == ["SUPPLIES", "WORKS_FOR", "AUDITS"]
    assert draft.relationship_type_counts == {
        "SUPPLIES": 3,
        "WORKS_FOR": 2,
        "AUDITS": 1,
    }


def test_only_the_most_frequent_relationship_types_are_kept(monkeypatch) -> None:
    monkeypatch.setattr(tuner_module, "MAX_DRAFT_RELATIONSHIP_TYPES", 2)
    rels = [_rel("Vendor", "Buyer", t) for t in ("A", "B", "B", "C", "C", "C")]
    draft = SchemaDraft.from_examples([], [_record("t1", ORGS, rels)])
    assert _labels(draft.relationship_types) == ["C", "B"]


def test_entity_types_come_from_the_profile_once_each() -> None:
    draft = SchemaDraft.from_examples(
        ["ORGANIZATION", "Organization", "LEGAL ENTITY", "DRUG: medicines", ""], []
    )
    assert _labels(draft.entity_types) == ["ORGANIZATION", "LEGAL ENTITY", "DRUG"]
    assert _labels(draft.relationship_types) == []
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
        SchemaSourceContext(SchemaSourceConfig(kind="file", path=str(path)), Config())
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
    assert _labels(draft.entity_types) == ["ORGANIZATION", "PRODUCT"]
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


_ONTOLOGY = """\
@prefix ex: <https://example.org/trade#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:Organization a owl:Class .
ex:Product a owl:Class .
ex:supplies a owl:ObjectProperty ; rdfs:comment "Delivers goods to"@en ;
    rdfs:domain ex:Organization ; rdfs:range ex:Organization .
"""


@requires_ontology
async def test_a_draft_over_an_ontology_says_what_it_lacks(tmp_path: Path) -> None:
    ontology = tmp_path / "trade.ttl"
    ontology.write_text(_ONTOLOGY, encoding="utf-8")
    config = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_source": {"kind": "owl", "path": str(ontology)}
                }
            }
        }
    )
    result = await _tuner(config).tune(CORPUS, schema_draft=True)
    draft: SchemaDraft = result["schema_draft"]
    assert draft.schema_source == ("owl", str(ontology))
    # The ontology's types, IRIs and descriptions, then what it lacks.
    assert [(s.label, s.iri) for s in draft.relationship_types] == [
        ("SUPPLIES", "https://example.org/trade#supplies"),
        ("STOCKS", None),
        ("PARTNERS_WITH", None),
    ]
    assert draft.added_relationship_types == ["STOCKS", "PARTNERS_WITH"]
    text = draft.to_yaml("draft.yaml")
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert header[:4] == [
        "# Extraction schema draft written by run-prompt-tuning: copied from",
        f"# owl '{ontology}' plus 2 relationship types the source lacks (STOCKS, "
        "PARTNERS_WITH).",
        "# Add them to the source (keep kind: owl), or review this file",
        "# and load it with kind: file to freeze the mapping:",
    ]
    assert '#       schema_source: {kind: file, path: "draft.yaml"}' in header
    # Loaded with kind: file, the draft is the ontology's schema plus them.
    path = tmp_path / "draft.yaml"
    path.write_text(text, encoding="utf-8")
    frozen = _load(path)
    assert frozen.entity_type("ORGANIZATION").iri == (  # type: ignore[union-attr]
        "https://example.org/trade#Organization"
    )
    assert frozen.relationship_type("SUPPLIES").description == "Delivers goods to"  # type: ignore[union-attr]


_UNPATTERNED = _ONTOLOGY + "ex:stocks a owl:ObjectProperty .\n"


@requires_ontology
@pytest.mark.parametrize("patterns", ["closure", "none"])
def test_a_frozen_ontology_draft_is_the_ontologys_schema(
    tmp_path: Path, patterns: str
) -> None:
    ontology = tmp_path / "trade.ttl"
    ontology.write_text(_UNPATTERNED, encoding="utf-8")
    config = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_source": {
                        "kind": "owl",
                        "path": str(ontology),
                        "options": {"patterns": patterns},
                    }
                }
            }
        }
    )
    schema = resolve_extraction_schema(
        config
    ).processing.graph_extraction.extraction_schema()
    # The examples state both declared types, twice each, in directions the
    # ontology does not give.
    entities = [_entity("Vendor", "ORGANIZATION"), _entity("Gadget", "PRODUCT")]
    records = [
        _record(
            unit,
            entities,
            [_rel("Vendor", "Gadget", "SUPPLIES"), _rel("Vendor", "Gadget", "STOCKS")],
        )
        for unit in ("t1", "t2")
    ]
    draft = SchemaDraft.from_examples(
        list(schema.entity_types),
        records,
        relationship_types=schema.relationship_types,
        relationship_patterns=schema.relationship_patterns,
        schema_source=("owl", str(ontology)),
    )
    # Every type the ontology declares keeps its patterns (none under
    # patterns: none) and gets no other.
    assert draft.added_relationship_types == []
    assert draft.relationship_patterns == list(schema.relationship_patterns)
    text = draft.to_yaml("draft.yaml")
    assert "plus 0 relationship types the source lacks." in text
    path = tmp_path / "draft.yaml"
    path.write_text(text, encoding="utf-8")
    assert _load(path).digest == schema.digest


def test_a_draft_over_another_source_counts_the_patterns_it_adds() -> None:
    declared = ExtractionSchema.from_items(
        ["ORGANIZATION"], ["SUPPLIES"], ["ORGANIZATION -SUPPLIES-> ORGANIZATION"]
    )
    records = [
        _record(
            unit,
            ORGS,
            [_rel("Vendor", "Buyer", "AUDITS"), _rel("Buyer", "Vendor", "SUPPLIES")],
        )
        for unit in ("t1", "t2")
    ]
    draft = SchemaDraft.from_examples(
        list(declared.entity_types),
        records,
        relationship_types=declared.relationship_types,
        relationship_patterns=declared.relationship_patterns,
        schema_source=("synthetic_service", "urn:schemas:trade"),
    )
    assert [str(p) for p in draft.relationship_patterns] == [
        "ORGANIZATION -SUPPLIES-> ORGANIZATION",
        "ORGANIZATION -AUDITS-> ORGANIZATION",
    ]
    assert (
        "# synthetic_service 'urn:schemas:trade' plus 1 relationship type the "
        "source lacks (AUDITS) and 1 pattern for it." in draft.to_yaml()
    )


def test_the_header_counts_several_added_types_and_patterns() -> None:
    records = [
        _record(
            unit,
            ORGS,
            [_rel("Vendor", "Buyer", "AUDITS"), _rel("Vendor", "Buyer", "PAYS")],
        )
        for unit in ("t1", "t2")
    ]
    draft = SchemaDraft.from_examples(
        ["ORGANIZATION"],
        records,
        schema_source=("synthetic_service", "urn:schemas:trade"),
    )
    assert (
        "# synthetic_service 'urn:schemas:trade' plus 2 relationship types the "
        "source lacks (AUDITS, PAYS) and 2 patterns for them." in draft.to_yaml()
    )


def test_a_draft_over_a_file_or_no_source_keeps_its_header() -> None:
    draft = SchemaDraft.from_examples(
        ["ORGANIZATION"], [_record("t1", ORGS, [_rel("Vendor", "Buyer", "SUPPLIES")])]
    )
    assert draft.schema_source is None
    assert draft.to_yaml().startswith(
        "# Extraction schema draft written by run-prompt-tuning. Review it\n"
    )
    one = SchemaDraft.from_examples(
        ["ORGANIZATION"],
        [_record("t1", ORGS, [_rel("Vendor", "Buyer", "SUPPLIES")])],
        schema_source=("synthetic_service", "urn:schemas:trade"),
    )
    assert "plus 1 relationship type the source lacks (SUPPLIES)." in one.to_yaml()
    none = SchemaDraft.from_examples(
        ["ORGANIZATION"], [], schema_source=("synthetic_service", "urn:schemas:trade")
    )
    assert "plus 0 relationship types the source lacks." in none.to_yaml()


def test_the_tuner_names_the_source_without_a_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A custom kind's path may be a URL carrying a token: the draft header
    # shows it without the userinfo, the query and the fragment.
    monkeypatch.setitem(
        schema_registry._REGISTRY,
        "synthetic_service",
        lambda context: SchemaDocument(relationship_types=["SUPPLIES"]),
    )
    config = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_source": {
                        "kind": "synthetic_service",
                        "path": "https://reader:SECRET@schemas.example/trade"
                        "?token=SECRET#SECRET",
                    }
                }
            }
        }
    )
    tuner = PromptTuner(
        config,
        providers=Providers(config, boto_session=MagicMock(), llm_factory=MagicMock()),
    )
    assert tuner.schema_source == (
        "synthetic_service",
        "https://schemas.example/trade",
    )
    header = tuner.schema_draft(["ORGANIZATION"], []).to_yaml()
    assert "SECRET" not in header
    assert "# synthetic_service 'https://schemas.example/trade' plus 0" in header


def test_the_tuner_gives_a_file_source_the_plain_header(tmp_path: Path) -> None:
    schema = tmp_path / "schema.yaml"
    schema.write_text("relationship_types: [SUPPLIES]\n", encoding="utf-8")
    config = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_source": {"kind": "file", "path": str(schema)}
                }
            }
        }
    )
    tuner = PromptTuner(
        config,
        providers=Providers(config, boto_session=MagicMock(), llm_factory=MagicMock()),
    )
    assert tuner.schema_source is None
    assert (
        tuner.schema_draft(["ORGANIZATION"], [])
        .to_yaml()
        .startswith(
            "# Extraction schema draft written by run-prompt-tuning. Review it\n"
        )
    )


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
    # The same file but the path it names, which is --schema-out as given.
    assert f"path: {json.dumps(str(first))}}}" in first.read_text(encoding="utf-8")
    assert first.read_text(encoding="utf-8").replace(str(first), "<path>") == (
        second.read_text(encoding="utf-8").replace(str(second), "<path>")
    )
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


# --------------------------------------------------------------------------- #
# The entity types the examples are extracted with
# --------------------------------------------------------------------------- #

TRADE = [
    "Vendor supplies Buyer.",
    "Vendor supplies Buyer again.",
    "Supplier supplies Customer.",
]


def _listing_respond(profile_types: list[str]) -> Any:
    """A model that profiles the corpus as ``profile_types`` and, extracting,
    types each name with the first listed type it contains (``Vendor`` is a
    VENDOR when the prompt lists VENDOR), else ORGANIZATION; relationship
    types are written as the prompt lists them when it declares them."""

    def respond(system: str, human: str) -> str:
        if "expert corpus analyst" in system:
            return json.dumps({"domain": "trade", "entity_types": profile_types})
        listed = re.findall(r"- \*\*([^*]+)\*\*", system + human)
        text = re.search(r"<input_text>\n(.*?)\n</input_text>", human, re.DOTALL)
        assert text is not None
        head, tail = re.findall(r"\b([A-Z]\w+)\b", text.group(1))[:2]

        def type_of(name: str) -> str:
            return next((t for t in listed if t.casefold() in name.casefold()), "")

        sentence = text.group(1).strip()
        entities = "".join(
            f"<entity><name>{name}</name><type>{type_of(name) or 'ORGANIZATION'}"
            f"</type><description>{name}</description><confidence>9</confidence>"
            f"<source_text>{sentence}</source_text></entity>"
            for name in (head, tail)
        )
        relation = "SUPPLIES" if "SUPPLIES" in listed else "supplies"
        return (
            f"<entities>{entities}</entities><relationships><relationship>"
            f"<source>{head}</source><target>{tail}</target><type>{relation}"
            "</type><description>supplies</description><strength>7</strength>"
            f"<source_text>{sentence}</source_text></relationship></relationships>"
        )

    return respond


def _listing_tuner(config: Config, profile_types: list[str]) -> PromptTuner:
    return PromptTuner(
        config,
        providers=Providers(
            config,
            boto_session=MagicMock(),
            llm_factory=ScriptedLLMFactory(_listing_respond(profile_types)),
        ),
    )


async def test_examples_use_the_profile_types_so_the_draft_has_patterns() -> None:
    tuner = _listing_tuner(Config(), ["VENDOR", "BUYER", "Vendor"])
    result = await tuner.tune(TRADE, schema_draft=True)
    draft: SchemaDraft = result["schema_draft"]
    assert _labels(draft.entity_types) == ["VENDOR", "BUYER"]
    # Extracted with the default types, the vendor and the buyer would both
    # be organizations, which the profile does not list: no pattern.
    assert [str(p) for p in draft.relationship_patterns] == ["VENDOR -SUPPLIES-> BUYER"]
    assert "<type>VENDOR</type>" in result["custom_prompts"]["graph_extraction_system"]
    # The caller's config is not changed.
    assert tuner.config.processing.graph_extraction.entity_types == (
        Config().processing.graph_extraction.entity_types
    )


async def test_a_configured_schema_is_honoured(tmp_path: Path) -> None:
    path = tmp_path / "schema.yaml"
    path.write_text(
        "entity_types: [SUPPLIER, CUSTOMER]\n"
        "relationship_types: ['SUPPLIES: Goods delivered']\n",
        encoding="utf-8",
    )
    config = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_source": {"kind": "file", "path": str(path)}
                }
            }
        }
    )
    tuner = _listing_tuner(config, ["VENDOR", "BUYER"])
    # Loaded at construction, as ingestion loads it.
    assert tuner.config.processing.graph_extraction.schema_source.kind == "inline"
    assert tuner.schema_configured
    corpus = ["Supplier supplies Customer.", "Supplier supplies Customer again."]
    result = await tuner.tune([*corpus, TRADE[0]], schema_draft=True)
    draft: SchemaDraft = result["schema_draft"]
    assert _labels(draft.entity_types) == ["SUPPLIER", "CUSTOMER"]
    assert [str(p) for p in draft.relationship_patterns] == [
        "SUPPLIER -SUPPLIES-> CUSTOMER"
    ]
    system = result["custom_prompts"]["graph_extraction_system"]
    assert "common in this corpus: SUPPLIER, CUSTOMER." in system
    assert "VENDOR" not in system
    # The profile still reports what the model saw.
    assert result["profile"]["entity_types"] == ["VENDOR", "BUYER"]


def _extraction(**extraction: Any) -> Config:
    return Config.model_validate({"processing": {"graph_extraction": extraction}})


def test_patterns_alone_make_the_schema_configured() -> None:
    config = _extraction(
        relationship_types=["SUPPLIES"],
        relationship_patterns=["ORGANIZATION -SUPPLIES-> ORGANIZATION"],
    )
    assert _listing_tuner(config, ["VENDOR"]).schema_configured
    assert not _listing_tuner(
        _extraction(relationship_types=["SUPPLIES"]), ["VENDOR"]
    ).schema_configured


def test_profile_types_are_one_per_schema_key_first_spelling_kept() -> None:
    tuner = _listing_tuner(Config(), [])
    profile = tuner_module.CorpusProfile(
        entity_types=["Vendor", "VENDOR", "buyer", "Big Vendor", "big_vendor"]
    )
    assert tuner.example_entity_types(profile) == ["Vendor", "buyer", "Big Vendor"]


def test_a_configured_schema_reaches_the_examples_unchanged() -> None:
    # Descriptions and IRIs kept, not rebuilt from the labels.
    config = _extraction(
        entity_types=[
            {"label": "SUPPLIER", "description": "Sells goods", "iri": "urn:x:S"},
            "CUSTOMER",
        ]
    )
    tuner = _listing_tuner(config, ["VENDOR"])
    profile = tuner_module.CorpusProfile(entity_types=["VENDOR"])
    example = tuner._example_config(tuner.example_entity_types(profile))
    schema = example.processing.graph_extraction.extraction_schema()
    assert [(s.label, s.description, s.iri) for s in schema.entity_types] == [
        ("SUPPLIER", "Sells goods", "urn:x:S"),
        ("CUSTOMER", "", None),
    ]


async def test_generate_examples_uses_the_profile_types(monkeypatch) -> None:
    tuner = _listing_tuner(Config(), ["VENDOR"])
    seen: list[Any] = []

    def extract(texts: list[str], entity_types: list[str] | None = None) -> Any:
        seen.append(entity_types)
        return [], []

    monkeypatch.setattr(tuner, "extract_examples", extract)
    profile = tuner_module.CorpusProfile(entity_types=["VENDOR", "BUYER"])
    await tuner.generate_examples(profile, ["text"])
    assert seen == [["VENDOR", "BUYER"]]


def test_a_broken_schema_source_fails_the_tuner(tmp_path: Path) -> None:
    config = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_source": {
                        "kind": "file",
                        "path": str(tmp_path / "missing.yaml"),
                    }
                }
            }
        }
    )
    factory = MagicMock()
    with pytest.raises(SchemaSourceError, match="not found"):
        PromptTuner(config, providers=Providers(config, llm_factory=factory))
    factory.get_model.assert_not_called()


def test_a_broken_schema_source_exits_2(tmp_path: Path, mocker) -> None:
    config = Config.model_validate(
        {
            "processing": {
                "graph_extraction": {
                    "schema_source": {
                        "kind": "file",
                        "path": str(tmp_path / "missing.yaml"),
                    }
                }
            }
        }
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text(TRADE[0], encoding="utf-8")
    mocker.patch.object(run_prompt_tuning, "load_cli_config", return_value=config)
    # The real tuner on a fake model: nothing would reach AWS either way.
    mocker.patch.object(
        run_prompt_tuning,
        "PromptTuner",
        side_effect=lambda config: _listing_tuner(config, ["VENDOR"]),
    )
    mocker.patch(
        "sys.argv",
        [
            "run-prompt-tuning",
            "--source-directory",
            str(source),
            "--output-path",
            str(tmp_path / "prompts.yaml"),
        ],
    )
    assert run_prompt_tuning.main() == 2
    assert not (tmp_path / "prompts.yaml").exists()


# --------------------------------------------------------------------------- #
# A draft over a curated schema
# --------------------------------------------------------------------------- #

CURATED = """\
entity_types:
  - {label: VENDOR, description: "Sellers", iri: "https://example.org/onto#Vendor"}
  - "BUYER: Purchasers"
  - "PERSON: People"
relationship_types:
  - {label: SUPPLIES, description: "Sells to", iri: "https://example.org/onto#supplies"}
  - "WORKS_FOR: Employment"
relationship_patterns:
  - "VENDOR -SUPPLIES-> BUYER"
"""
# The buyer supplies the vendor twice: without the curated pattern the draft
# would allow SUPPLIES in that direction.
REVERSED = ["Buyer supplies Vendor.", "Buyer supplies Vendor again."]


def test_a_draft_keeps_every_declared_type_and_pattern() -> None:
    records = [
        _record(
            "t1",
            [_entity("Vendor", "VENDOR"), _entity("Buyer", "BUYER")],
            [_rel("Buyer", "Vendor", "supplies"), _rel("Buyer", "Vendor", "audits")],
        ),
        _record(
            "t2",
            [_entity("Vendor", "VENDOR"), _entity("Buyer", "BUYER")],
            [_rel("Buyer", "Vendor", "SUPPLIES"), _rel("Buyer", "Vendor", "AUDITS")],
        ),
    ]
    declared = ExtractionSchema.from_items(
        [
            EntityTypeSpec(label="VENDOR", iri="urn:x:Vendor"),
            "BUYER: Purchasers",
        ],
        [
            RelationshipTypeSpec(label="SUPPLIES", iri="urn:x:supplies"),
            "UNSEEN: Never stated",
        ],
        ["VENDOR -SUPPLIES-> BUYER"],
    )
    draft = SchemaDraft.from_examples(
        list(declared.entity_types),
        records,
        relationship_types=declared.relationship_types,
        relationship_patterns=declared.relationship_patterns,
    )
    # The declared types first, as declared; then the examples' own.
    assert draft.entity_types == list(declared.entity_types)
    assert draft.relationship_types[:2] == list(declared.relationship_types)
    assert _labels(draft.relationship_types) == ["SUPPLIES", "UNSEEN", "AUDITS"]
    assert draft.relationship_type_counts == {"SUPPLIES": 2, "UNSEEN": 0, "AUDITS": 2}
    # SUPPLIES keeps its pattern only; AUDITS, declaring none, gets one.
    assert [str(p) for p in draft.relationship_patterns] == [
        "VENDOR -SUPPLIES-> BUYER",
        "BUYER -AUDITS-> VENDOR",
    ]


def test_the_file_writes_descriptions_and_iris_back(tmp_path: Path) -> None:
    curated = tmp_path / "curated.yaml"
    curated.write_text(CURATED, encoding="utf-8")
    declared = _load(curated)
    draft = SchemaDraft.from_examples(
        list(declared.entity_types),
        [],
        relationship_types=declared.relationship_types,
        relationship_patterns=declared.relationship_patterns,
    )
    path = tmp_path / "draft.yaml"
    path.write_text(draft.to_yaml(), encoding="utf-8")
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["entity_types"][1:] == [
        "BUYER: Purchasers",
        "PERSON: People",
    ]
    assert _load(path) == declared


_LOADS_CURATED = (
    "processing:\n  graph_extraction:\n"
    "    schema_source: {kind: file, path: curated.yaml}\n"
)


def _curated_main(
    tmp_path: Path,
    mocker,
    monkeypatch,
    schema_out: str,
    *extra: str,
    config: str = _LOADS_CURATED,
    config_path: str = "conf/config.yaml",
) -> int:
    """Run the CLI from ``tmp_path`` with ``conf/config.yaml`` (``config``,
    by default one that loads ``conf/curated.yaml``) over ``REVERSED``."""
    monkeypatch.chdir(tmp_path)
    conf = tmp_path / "conf"
    conf.mkdir(exist_ok=True)
    (conf / "curated.yaml").write_text(CURATED, encoding="utf-8")
    (conf / "config.yaml").write_text(config, encoding="utf-8")
    source = tmp_path / "corpus"
    source.mkdir(exist_ok=True)
    for index, text in enumerate(REVERSED):
        (source / f"{index}.txt").write_text(text, encoding="utf-8")
    tuner = mocker.patch.object(
        run_prompt_tuning,
        "PromptTuner",
        side_effect=lambda config: _listing_tuner(config, ["VENDOR", "BUYER"]),
    )
    mocker.patch(
        "sys.argv",
        [
            "run-prompt-tuning",
            "--config-path",
            config_path,
            "--source-directory",
            "corpus",
            "--output-path",
            "prompts.yaml",
            "--schema-out",
            schema_out,
            *extra,
        ],
    )
    try:
        return run_prompt_tuning.main()
    except SystemExit as exc:
        # Refused before the tuner (and so any model call) is built.
        tuner.assert_not_called()
        assert isinstance(exc.code, int)
        return exc.code


@pytest.mark.parametrize("force", [False, True])
def test_schema_out_never_replaces_the_loaded_schema(
    tmp_path: Path, mocker, monkeypatch, capsys: Any, force: bool
) -> None:
    extra = ["--force"] if force else []
    code = _curated_main(tmp_path, mocker, monkeypatch, "conf/curated.yaml", *extra)
    assert code == 2
    assert "is the schema file the config loads" in capsys.readouterr().err
    assert (tmp_path / "conf" / "curated.yaml").read_text(encoding="utf-8") == CURATED
    # --output-path neither.
    mocker.stopall()
    code = _curated_main(
        tmp_path,
        mocker,
        monkeypatch,
        "draft.yaml",
        "--output-path",
        "conf/curated.yaml",
    )
    assert code == 2
    assert (tmp_path / "conf" / "curated.yaml").read_text(encoding="utf-8") == CURATED


def test_schema_out_replaces_an_existing_file_only_with_force(
    tmp_path: Path, mocker, monkeypatch, capsys: Any
) -> None:
    (tmp_path / "conf").mkdir()
    existing = tmp_path / "conf" / "draft.yaml"
    existing.write_text("relationship_types: [KEPT]\n", encoding="utf-8")
    assert _curated_main(tmp_path, mocker, monkeypatch, "conf/draft.yaml") == 2
    assert "exists; pass --force" in capsys.readouterr().err
    assert existing.read_text(encoding="utf-8") == "relationship_types: [KEPT]\n"

    mocker.stopall()
    code = _curated_main(tmp_path, mocker, monkeypatch, "conf/draft.yaml", "--force")
    assert code == 0
    text = existing.read_text(encoding="utf-8")
    # The comment names the path relative to the config file's directory,
    # as the config loader resolves it.
    assert '#       schema_source: {kind: file, path: "draft.yaml"}' in text
    curated = _load(tmp_path / "conf" / "curated.yaml")
    draft = _load(existing)
    # Every curated type keeps its description and IRI, nothing is dropped,
    # and the reversed statements add no SUPPLIES pattern.
    assert draft == curated
    assert "#   SUPPLIES: 2" in text


def test_schema_out_and_output_path_must_differ(
    tmp_path: Path, mocker, monkeypatch
) -> None:
    code = _curated_main(
        tmp_path, mocker, monkeypatch, "same.yaml", "--output-path", "same.yaml"
    )
    assert code == 2


def test_schema_out_absolute_and_output_path_relative_name_one_file(
    tmp_path: Path, mocker, monkeypatch, capsys: Any
) -> None:
    # Neither exists yet: compared resolved.
    code = _curated_main(tmp_path, mocker, monkeypatch, str(tmp_path / "prompts.yaml"))
    assert code == 2
    assert "name one file" in capsys.readouterr().err


_INLINE = (
    "processing:\n  graph_extraction:\n    schema_mode: align\n"
    "    entity_types: [VENDOR, BUYER]\n    relationship_types: [SUPPLIES]\n"
)


@pytest.mark.parametrize("flag", ["--output-path", "--schema-out"])
@pytest.mark.parametrize(
    "name", ["conf/config.yaml", "conf/../conf/config.yaml", "link.yaml", "abs"]
)
def test_no_output_replaces_the_config_file(
    tmp_path: Path, mocker, monkeypatch, capsys: Any, flag: str, name: str
) -> None:
    # The inline schema lives in the config file itself; --force does not
    # let either output replace it, under any spelling or a symlink.
    tmp_path.joinpath("conf").mkdir()
    tmp_path.joinpath("link.yaml").symlink_to("conf/config.yaml")
    if name == "abs":
        name = str(tmp_path / "conf" / "config.yaml")
    extra = [flag, name] if flag == "--output-path" else []
    schema_out = name if flag == "--schema-out" else "draft.yaml"
    code = _curated_main(
        tmp_path, mocker, monkeypatch, schema_out, *extra, "--force", config=_INLINE
    )
    assert code == 2
    assert "is the --config-path file" in capsys.readouterr().err
    assert (tmp_path / "conf" / "config.yaml").read_text(encoding="utf-8") == _INLINE


def test_output_path_replaces_an_existing_file_only_with_force(
    tmp_path: Path, mocker, monkeypatch, capsys: Any
) -> None:
    existing = tmp_path / "prompts.yaml"
    existing.write_text("custom_prompts: {kept: true}\n", encoding="utf-8")
    assert _curated_main(tmp_path, mocker, monkeypatch, "draft.yaml") == 2
    assert "--output-path 'prompts.yaml' exists; pass --force" in (
        capsys.readouterr().err
    )
    assert existing.read_text(encoding="utf-8") == "custom_prompts: {kept: true}\n"
    assert not (tmp_path / "draft.yaml").exists()

    mocker.stopall()
    assert _curated_main(tmp_path, mocker, monkeypatch, "draft.yaml", "--force") == 0
    assert set(yaml.safe_load(existing.read_text(encoding="utf-8"))) == {
        "profile",
        "custom_prompts",
    }


def test_the_comment_path_is_relative_to_a_symlinked_configs_own_directory(
    tmp_path: Path, mocker, monkeypatch
) -> None:
    # sub/config.yaml links to conf/config.yaml: the loader resolves a
    # relative schema_source path against sub/, where the link is.
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "config.yaml").symlink_to("../conf/config.yaml")
    code = _curated_main(
        tmp_path,
        mocker,
        monkeypatch,
        "sub/draft.yaml",
        config=_INLINE,
        config_path="sub/config.yaml",
    )
    assert code == 0
    text = (tmp_path / "sub" / "draft.yaml").read_text(encoding="utf-8")
    assert '#       schema_source: {kind: file, path: "draft.yaml"}' in text
