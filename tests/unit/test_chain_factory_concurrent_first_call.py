# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A chain's first calls from several threads at once (AWS-free).

A batch's worker threads share one chain and all make their first call
together. LangChain prompts and chat models cache their serialized form in
the instance at that first call; computing it iterates the instance's
``__dict__``, so another thread's cache write could fail the call with
"dictionary changed size during iteration". ``setup_chain`` computes it
before the chain is shared, so a call writes nothing to a shared step.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.llm.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.domain.models import ModelPurpose
from unified_kg_rag.domain.prompts import ClaimExtractionPrompt

pytestmark = pytest.mark.unit

_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
_INPUT = {"input_text": "Vendor pays Buyer.", "entity_specs": ""}


def _chain(reply: str, *, fixing: bool = False) -> Any:
    factory = ScriptedLLMFactory(lambda system, human: reply)
    parser = create_robust_xml_output_parser(
        factory=factory,
        enable_output_fixing=fixing,
        output_fixing_model_id=_MODEL,
        output_tags=["claims"],
        model_purpose=ModelPurpose.INGESTION,
    )
    return setup_chain(
        factory=factory,
        model_id=_MODEL,
        prompt_class=ClaimExtractionPrompt,
        parser=parser,
        model_purpose=ModelPurpose.INGESTION,
    )


def _shared_steps(chain: Any) -> list[Any]:
    steps = list(chain.bound.steps)
    fixer_chain = getattr(steps[-1], "retry_chain", None)
    return steps + (list(fixer_chain.steps) if fixer_chain is not None else [])


def test_a_call_writes_nothing_to_a_shared_step() -> None:
    # A reply the parser cannot read, so the output fixer's chain runs too.
    chain = _chain("no markup at all", fixing=True)
    steps = _shared_steps(chain)
    before = [set(vars(step)) for step in steps]

    with pytest.raises(OutputParserException):
        chain.invoke(_INPUT)

    assert [set(vars(step)) for step in steps] == before


def test_concurrent_first_calls_all_succeed() -> None:
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    errors: list[BaseException] = []
    try:
        for _ in range(200):
            chain = _chain("<claims></claims>")
            barrier = threading.Barrier(8)

            def call(chain: Any = chain, barrier: threading.Barrier = barrier) -> None:
                barrier.wait()
                try:
                    chain.invoke(_INPUT)
                except RuntimeError as e:
                    errors.append(e)

            threads = [threading.Thread(target=call) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    finally:
        sys.setswitchinterval(previous)

    assert not errors, errors[0]
