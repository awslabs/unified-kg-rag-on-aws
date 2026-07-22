# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Bedrock model factories conform to ModelFactoryPort (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.aws.bedrock import (
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    BedrockRerankModelFactory,
)
from unified_kg_rag.ports import ModelFactoryPort

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


@pytest.mark.parametrize("factory_cls", _FACTORIES)
def test_boto_config_sets_bounded_connect_timeout(factory_cls) -> None:
    # A bounded connect timeout turns an unreachable-endpoint hang (e.g. a
    # private VPC missing the bedrock-agent-runtime interface endpoint for the
    # Rerank API) into a prompt, catchable error instead of an indefinite block.
    # _boto_config is pure (no network / no __init__), so build via __new__.
    factory = factory_cls.__new__(factory_cls)
    cfg = factory._boto_config(read_timeout=300)
    assert cfg.connect_timeout == factory_cls.BOTO_CONNECT_TIMEOUT
    assert cfg.connect_timeout is not None and cfg.connect_timeout <= 30
    # Also holds when no read_timeout is passed.
    assert factory._boto_config().connect_timeout == factory_cls.BOTO_CONNECT_TIMEOUT


def test_rerank_factory_targets_agent_runtime_endpoint() -> None:
    # The Rerank API lives on bedrock-agent-runtime (NOT bedrock-runtime); this
    # is why the CDK must provision the BEDROCK_AGENT_RUNTIME VPC endpoint. Pin
    # it so a refactor cannot silently point rerank at the wrong service.
    factory = BedrockRerankModelFactory.__new__(BedrockRerankModelFactory)
    assert factory._get_boto_service_name() == "bedrock-agent-runtime"


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
