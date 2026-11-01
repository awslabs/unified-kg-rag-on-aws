# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt domain-customization ergonomics (A + B refactor).

A: domain entity types are a {entity_types} template slot fed from config, so a
   domain swap needs no prompt copy-paste.
B: per-prompt overrides resolve by a `prompt_key` convention (no per-class
   _get_custom_prompts boilerplate), and partial overrides don't crash on the
   strict missing-variable validation that guards the shipped defaults.
"""

from __future__ import annotations

import hashlib

import pytest

from unified_kg_rag.domain.ingestion.cache_keys import unit_record_fingerprint
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import CustomPromptConfig
from unified_kg_rag.domain.prompts import (
    AnswerGenerationPrompt,
    CommunityReportPrompt,
    EntityExtractionPrompt,
    GraphExtractionPrompt,
)
from unified_kg_rag.domain.prompts.base import BasePrompt, fill_slots
from unified_kg_rag.domain.prompts.graph_extraction import (
    DEFAULT_RELATIONSHIP_TYPES,
    format_entity_types,
)

pytestmark = pytest.mark.unit


# --- B: prompt_key generalization -------------------------------------------


def test_prompt_keys_map_to_real_config_fields() -> None:
    # Every prompt that declares a prompt_key (i.e. supports customization) must
    # have matching <key>_system / <key>_human fields on CustomPromptConfig, so
    # the by-convention base lookup actually resolves. Prompts WITHOUT a key are
    # intentionally non-customizable (e.g. internal translation/chunking) — fine.
    import unified_kg_rag.domain.prompts as prompts

    cfg = CustomPromptConfig()
    customizable = [
        getattr(prompts, n)
        for n in prompts.__all__
        if isinstance(getattr(prompts, n), type)
        and issubclass(getattr(prompts, n), BasePrompt)
        and getattr(prompts, n) is not BasePrompt
        and getattr(prompts, n).prompt_key
    ]
    assert customizable, "expected some customizable prompts"
    for cls in customizable:
        assert hasattr(cfg, f"{cls.prompt_key}_system"), cls.__name__
        assert hasattr(cfg, f"{cls.prompt_key}_human"), cls.__name__


def test_system_override_by_key() -> None:
    cp = CustomPromptConfig(graph_extraction_system="MY SYSTEM PROMPT")
    assert (
        GraphExtractionPrompt.resolve(cp).system_prompt_template == "MY SYSTEM PROMPT"
    )


def test_partial_human_only_override_does_not_crash() -> None:
    # The crux of the old "all-or-nothing" trap: overriding only the human
    # template (which need not contain every built-in input variable) must not
    # fail the strict missing-variable validation. Only the data variables are
    # required (custom_prompts load-time check); {target_language} is not.
    human = "list the entities in {query}"
    cp = CustomPromptConfig(entity_extraction_human=human)
    resolved = EntityExtractionPrompt.resolve(cp)
    assert resolved.human_prompt_template == human
    # System stays the shipped default.
    assert (
        resolved.system_prompt_template == EntityExtractionPrompt.system_prompt_template
    )


def test_no_override_returns_defaults() -> None:
    r = CommunityReportPrompt.resolve()
    assert r.system_prompt_template == CommunityReportPrompt.system_prompt_template


def test_shipped_defaults_still_validated() -> None:
    # A default template missing a declared variable must still raise (the
    # validation guard is preserved for shipped prompts).
    with pytest.raises(ValueError, match="not found"):

        class Bad(BasePrompt):
            input_variables = ["missing_var"]
            system_prompt_template = "no vars here"
            human_prompt_template = "none here either"

        Bad.resolve()


# --- A: entity_types template slot ------------------------------------------


def test_extraction_prompt_has_entity_types_slot() -> None:
    r = GraphExtractionPrompt.resolve()
    assert "{entity_types}" in r.system_prompt_template
    assert "entity_types" in r.input_variables
    # The old hardcoded PERSON/ORGANIZATION list is gone from the template body.
    assert "- **PERSON**" not in r.system_prompt_template


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (
            ["PERSON: a person", "GENE: a gene"],
            "- **PERSON**: a person\n- **GENE**: a gene",
        ),
        (["GENE"], "- **GENE**"),
        (["  DRUG  : meds "], "- **DRUG**: meds"),
        ([": no label", "PERSON"], "- **PERSON**"),
    ],
)
def test_format_entity_types(types: list[str], expected: str) -> None:
    assert format_entity_types(types) == expected


def test_format_entity_types_empty_lets_model_choose() -> None:
    out = format_entity_types([])
    assert "any entity types" in out.lower()


# The default extraction prompt rendered (system, a blank line, human) with
# the values below, by the code before {relationship_types} became a slot
# (f739d979). The slot's default must leave it byte for byte the same.
_DEFAULT_EXTRACTION_PROMPT_SHA256 = (
    "202a48fa01eed41a35479957fe6e20b255a493f225b3036457d0cd66f2c3d121"
)


def _rendered_extraction_prompt(config: Config) -> str:
    resolved = GraphExtractionPrompt.resolve(config.custom_prompts)
    values = {
        "input_text": "Alice works for Acme.",
        "max_entities_per_chunk": "50",
        "max_relationships_per_chunk": "50",
        "target_language": "en",
        **GraphExtractionPrompt.config_variables(config),
    }
    return (
        resolved.system_prompt_template.format(**values)
        + "\n\n"
        + resolved.human_prompt_template.format(**values)
    )


def test_default_extraction_prompt_renders_byte_identical() -> None:
    text = _rendered_extraction_prompt(Config())
    assert hashlib.sha256(text.encode()).hexdigest() == (
        _DEFAULT_EXTRACTION_PROMPT_SHA256
    )
    assert text.count(DEFAULT_RELATIONSHIP_TYPES) == 1


def test_relationship_types_is_a_declared_slot() -> None:
    resolved = GraphExtractionPrompt.resolve()
    assert "{relationship_types}" in resolved.system_prompt_template
    assert "relationship_types" in resolved.input_variables
    assert "HIERARCHICAL" not in resolved.system_prompt_template
    # Tuned prompts copy the rules verbatim, the slot with them.
    assert "{relationship_types}" in GraphExtractionPrompt.output_rules
    GraphExtractionPrompt.validate_templates()


def test_custom_prompt_may_use_relationship_types() -> None:
    cfg = CustomPromptConfig(
        graph_extraction_human="Types:\n{relationship_types}\n{input_text}"
    )
    assert cfg.graph_extraction_human is not None


def test_record_fingerprint_hashes_the_slot_filled_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The template with {relationship_types} filled in is the template before
    # the slot: the records extracted then keep their keys.
    resolved = GraphExtractionPrompt.resolve()
    filled = fill_slots(
        resolved.system_prompt_template,
        GraphExtractionPrompt.inlined_slots(Config()),
    )
    assert "{relationship_types}" not in filled
    assert DEFAULT_RELATIONSHIP_TYPES in filled
    assert "{entity_types}" in filled
    # The fingerprint is the one of the template with the text written in.
    expected = unit_record_fingerprint(Config(), gleaned=True)
    monkeypatch.setattr(GraphExtractionPrompt, "system_prompt_template", filled)
    monkeypatch.setattr(
        GraphExtractionPrompt,
        "input_variables",
        [v for v in resolved.input_variables if v != "relationship_types"],
    )
    monkeypatch.setattr(
        GraphExtractionPrompt, "inlined_slots", classmethod(lambda cls, c: {})
    )
    assert unit_record_fingerprint(Config(), gleaned=True) == expected
    monkeypatch.undo()
    # A custom prompt using the slot hashes its own text: two overrides that
    # differ only in it differ.
    with_slot = Config(
        custom_prompts={"graph_extraction_human": "{relationship_types}\n{input_text}"}
    )
    without = Config(custom_prompts={"graph_extraction_human": "{input_text}"})
    assert unit_record_fingerprint(with_slot, gleaned=False) != unit_record_fingerprint(
        without, gleaned=False
    )


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("a {x} b", "a X b"),
        ("{{x}} {x}", "{{x}} X"),
        ("{y} {x} {y!r} {y:>3}", "{y} X {y!r} {y:>3}"),
        ("{x:>3}", "{x:>3}"),
        ("unmatched {", "unmatched {"),
    ],
)
def test_fill_slots(template: str, expected: str) -> None:
    assert fill_slots(template, {"x": "X"}) == expected


# --- answer prompt contract ---------------------------------------------------


def test_answer_prompt_asks_to_chain_facts_before_declaring_insufficiency() -> None:
    system = AnswerGenerationPrompt.system_prompt_template
    assert "Chain facts across sources" in system
    assert "name variants of the same entity" in system
    assert "Only after trying to chain" in system
    assert system.index("Chain facts") < system.index("does not cover")


def test_answer_prompt_keeps_grounding_rules_and_direct_answer_first() -> None:
    system = AnswerGenerationPrompt.system_prompt_template
    assert "Use only the provided context" in system
    assert "general knowledge" in system
    assert "same language as the user's query" in system
    assert "First line: a one-sentence direct answer" in system
    # The verbose-answer pressure the multi-hop rewrite removed.
    assert "actionable insights" not in system
    assert "comprehensive coverage" not in system.lower()


def test_previous_answer_wording_restorable_through_custom_prompts() -> None:
    human = "{query}\n{context}"
    cp = CustomPromptConfig(
        answer_generation_system="LEGACY SYSTEM", answer_generation_human=human
    )
    resolved = AnswerGenerationPrompt.resolve(cp)
    assert resolved.system_prompt_template == "LEGACY SYSTEM"
    assert resolved.human_prompt_template == human


def _shipped_prompt_classes() -> list[type[BasePrompt]]:
    import unified_kg_rag.domain.prompts as prompts

    return [
        cls
        for cls in (getattr(prompts, n) for n in prompts.__all__)
        if isinstance(cls, type)
        and issubclass(cls, BasePrompt)
        and cls is not BasePrompt
    ]


@pytest.mark.parametrize(
    "prompt_class", _shipped_prompt_classes(), ids=lambda c: c.__name__
)
def test_every_shipped_prompt_declares_variables_its_templates_use(
    prompt_class: type[BasePrompt],
) -> None:
    prompt_class.validate_templates()
