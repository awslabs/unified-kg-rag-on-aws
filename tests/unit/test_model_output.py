# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model output as other model families write it: one reading for all.

A reasoning model left its ``<think>`` block in every answer, and the eight
JSON call sites read model JSON four ways: a Llama trailing comma or a
Python dict failed mix, hybrid and iterative, and DRIFT answered from
nothing without a degraded mark. ``ReasoningStripper`` (in every
``setup_chain``) and ``parse_model_object`` (behind ``LLMJsonOutputParser``)
are the one normaliser and the one reader.
"""

from __future__ import annotations

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import BaseModel

from unified_kg_rag.adapters.llm.chain_factory import (
    ReasoningStripper,
    create_json_output_parser,
)
from unified_kg_rag.adapters.llm.langchain import LLMJsonOutputParser
from unified_kg_rag.shared import ModelOutputError
from unified_kg_rag.shared.utils import parse_model_object, strip_leading_reasoning

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('<think>\nWants {"key": 1}. Draft: <entities>\n</think>\n\nVendor', "Vendor"),
        ("  <THINKING>x</thinking>[NO_ANSWER] none", "[NO_ANSWER] none"),
        ("<reasoning>a\nb</reasoning>{}", "{}"),
        # Never closed: the output ran out while reasoning, no answer.
        ("<think>still reasoning", ""),
        # A tag later in the answer is the answer's.
        ("Vendor <think>x</think>", "Vendor <think>x</think>"),
        ("<entities></entities>", "<entities></entities>"),
    ],
)
def test_strip_leading_reasoning(text: str, expected: str) -> None:
    assert strip_leading_reasoning(text) == expected


def _streamed(parts: list[str]) -> tuple[str, int]:
    chunks = [AIMessageChunk(content=part) for part in parts]
    out = list(ReasoningStripper("p").transform(iter(chunks)))
    return "".join(str(c.content) for c in out), len(out)


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (
            ["<th", "ink>\nabc ", "{x}</th", "ink>\n\n", "[NO_ANSWER]", " rest"],
            "[NO_ANSWER] rest",
        ),
        (["  Vendor", "<think>", " ships"], "  Vendor<think> ships"),
        (["<", "b>bold</b>"], "<b>bold</b>"),
        (["<think>never closed"], ""),
        (["<thinking>x</thinking>", "\n", "Answer"], "Answer"),
    ],
)
def test_a_stream_drops_its_leading_reasoning(parts: list[str], expected: str) -> None:
    text, count = _streamed(parts)
    assert text == expected
    # No chunk is dropped: the last carries the stop reason and usage.
    assert count == len(parts)


async def test_the_stripper_reads_a_message_and_a_string() -> None:
    stripper = ReasoningStripper("p")
    message = AIMessage(content="<think>a</think>\nB", response_metadata={"x": 1})
    stripped = await stripper.ainvoke(message)
    assert stripped.content == "B" and stripped.response_metadata == {"x": 1}
    assert stripper.invoke("<think>a</think>B") == "B"
    # Content blocks (Bedrock's typed reasoning) pass as they are.
    blocks = AIMessage(content=[{"type": "text", "text": "B"}])
    assert stripper.invoke(blocks) is blocks


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1, "b": [1, 2]}',
        # Python's repr (single quotes, True/None).
        "{'a': 1, 'b': [1, 2], 'c': None}",
        # Trailing commas (Llama, Gemma).
        '{\n "a": 1,\n "b": [1, 2,],\n}',
        # Comments, a URL's // kept.
        '{\n // the answer\n "a": 1, /* x */ "b": [1, 2], "u": "http://x//y" # end\n}',
        # Typographic quotes as delimiters.
        "{“a”: 1, “b”: [1, 2]}",
        # Prose and a fence around it.
        'Sure! Here you go:\n```json\n{"a": 1, "b": [1, 2]}\n```\nAnything else?',
        # An example object before a Python-dict answer.
        "Format: {\"key\": \"value\"}. Answer: {'a': 1, 'b': [1, 2]}",
        # A raw line break inside a string, an invalid escape.
        '{"a": 1, "b": [1, 2], "s": "x\\_y\nz"}',
    ],
)
def test_parse_model_object_reads_every_family(raw: str) -> None:
    parsed = parse_model_object(raw, answer_keys=("a",))
    assert parsed["a"] == 1 and parsed["b"] == [1, 2]


def test_typographic_quotes_inside_a_string_stay() -> None:
    parsed = parse_model_object('{"a": "He said “hi”", "b": 1,}')
    assert parsed == {"a": "He said “hi”", "b": 1}


def test_a_bare_list_is_the_one_answer_key() -> None:
    assert parse_model_object('[{"d": "x"}]', answer_keys=("points",)) == {
        "points": [{"d": "x"}]
    }


def test_a_cut_off_last_object_and_no_object_raise() -> None:
    with pytest.raises(ModelOutputError, match="unclosed"):
        parse_model_object('{"a": 1} Correction: {"a": 2, "b', answer_keys=("a",))
    with pytest.raises(ModelOutputError, match="no JSON object"):
        parse_model_object("I cannot answer that.")
    # A ValueError, as json.JSONDecodeError is.
    assert issubclass(ModelOutputError, ValueError)


class _Route(BaseModel):
    strategy: str


def test_a_schema_takes_the_last_object_it_accepts() -> None:
    raw = '{"strategy": "local"} then {"other": 1}'
    assert parse_model_object(raw, _Route) == _Route(strategy="local")
    with pytest.raises(ModelOutputError, match="not a _Route"):
        parse_model_object('{"other": 1}', _Route)


def test_the_parser_raises_for_the_fixer_and_tells_it_the_keys() -> None:
    parser = LLMJsonOutputParser(answer_keys=("strategy",), instructions="Or [CANNOT].")
    assert parser.parse("{'strategy': 'local'}") == {"strategy": "local"}
    with pytest.raises(OutputParserException):
        parser.parse("local, I think")
    with pytest.raises(OutputParserException, match="no output"):
        parser.parse("  ")
    instructions = parser.get_format_instructions()
    assert '"strategy"' in instructions and instructions.endswith("Or [CANNOT].")


class _FixingFactory:
    """A factory whose fixing model answers ``fixed`` (and records it)."""

    def __init__(self, fixed: str) -> None:
        self.fixed = fixed
        self.calls = 0
        self.built = 0

    def get_model(self, **kwargs: object) -> object:
        from langchain_core.runnables import RunnableLambda

        self.built += 1

        def answer(_: object) -> AIMessage:
            self.calls += 1
            return AIMessage(content=f"<think>fix it</think>{self.fixed}")

        return RunnableLambda(answer)


def test_the_json_parser_asks_the_fixer_once() -> None:
    factory = _FixingFactory('{"strategy": "global"}')
    parser = create_json_output_parser(
        factory, "fixer", answer_keys=("strategy",)  # type: ignore[arg-type]
    )
    assert parser.parse('{"strategy": "local"}') == {"strategy": "local"}
    # The fixing model is built on the first answer that does not parse.
    assert factory.calls == factory.built == 0
    # The fixer's answer is stripped of its reasoning too.
    assert parser.parse("global, I think") == {"strategy": "global"}
    assert factory.calls == 1
    broken = _FixingFactory("still not JSON")
    with pytest.raises(OutputParserException):
        create_json_output_parser(
            broken, "fixer", answer_keys=("strategy",)  # type: ignore[arg-type]
        ).parse("nope")
    assert broken.calls == 1
    # A blank answer has nothing to repair.
    with pytest.raises(OutputParserException, match="no output"):
        create_json_output_parser(broken, "fixer").parse("")  # type: ignore[arg-type]
    assert broken.calls == 1


async def test_a_cut_off_answer_is_not_repaired() -> None:
    factory = _FixingFactory('{"sufficient": true}')
    parser = create_json_output_parser(
        factory, "fixer", answer_keys=("sufficient",)  # type: ignore[arg-type]
    )
    with pytest.raises(OutputParserException):
        await parser.aparse('{"sufficient":')
    assert factory.calls == 0
    assert await parser.aparse("sufficient: yes") == {"sufficient": True}
    assert factory.calls == 1
