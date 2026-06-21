# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""The Bedrock model factories conform to ModelFactoryPort (AWS-free)."""

from __future__ import annotations

import pytest

from aws_graphrag.adapters.aws.bedrock import (
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    BedrockRerankModelFactory,
)
from aws_graphrag.ports import ModelFactoryPort

pytestmark = pytest.mark.unit

_FACTORIES = [
    BedrockLanguageModelFactory,
    BedrockEmbeddingModelFactory,
    BedrockRerankModelFactory,
]


@pytest.mark.parametrize("factory_cls", _FACTORIES)
def test_factory_has_port_methods(factory_cls) -> None:
    # Structural conformance: the port's methods exist and are callable.
    for method in ("get_model", "get_model_info"):
        assert callable(
            getattr(factory_cls, method, None)
        ), f"{factory_cls.__name__} missing {method}"


def test_runtime_checkable_protocol_recognizes_a_conforming_stub() -> None:
    class _Stub:
        def get_model(self, model_id, **kwargs):  # noqa: ANN001, ANN002, ANN003
            return object()

        def get_model_info(self, model_id):  # noqa: ANN001
            return None

    assert isinstance(_Stub(), ModelFactoryPort)

    class _NotAFactory:
        pass

    assert not isinstance(_NotAFactory(), ModelFactoryPort)
