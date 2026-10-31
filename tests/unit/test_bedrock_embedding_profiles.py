# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference-profile resolution for Bedrock embedding models (AWS-free).

Cohere Embed v4 has no on-demand throughput in most regions (us-west-2 among
them): it is invocable there only through the ``us.``/``global.`` profiles.
The embedding factory resolves profiles the way the chat factory does, with
``bedrock:ListInferenceProfiles`` answered by a botocore ``Stubber``.
"""

from __future__ import annotations

from typing import Any

import boto3
import pytest
from botocore.stub import Stubber

from unified_kg_rag.adapters.aws import bedrock as bedrock_mod
from unified_kg_rag.adapters.aws.bedrock import (
    BedrockCrossRegionModelHelper,
    BedrockEmbeddingModelFactory,
)
from unified_kg_rag.adapters.aws.bedrock_models import profile_target_model_id
from unified_kg_rag.domain.models import Config, EmbeddingModelId
from unified_kg_rag.shared import EmbeddingModelError

pytestmark = pytest.mark.unit

EMBED_V4 = EmbeddingModelId.EMBED_V4.value


class _StubbedBedrockSession:
    """A session whose ``bedrock`` client lists ``profiles`` through a Stubber.

    ``bedrock-runtime`` gets an opaque object: no embedding is invoked here.
    """

    profile_name = "default"

    def __init__(self, region: str, profiles: list[str] | None) -> None:
        self._session = boto3.Session(
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name=region,
        )
        self.bedrock = self._session.client("bedrock", region_name=region)
        self.stubber = Stubber(self.bedrock)
        if profiles is None:
            self.stubber.add_client_error(
                "list_inference_profiles", service_error_code="AccessDeniedException"
            )
        else:
            self.stubber.add_response(
                "list_inference_profiles",
                {
                    "inferenceProfileSummaries": [
                        {
                            "inferenceProfileName": p,
                            "inferenceProfileArn": (
                                f"arn:aws:bedrock:{region}:123456789012:"
                                f"inference-profile/{p}"
                            ),
                            "models": [
                                {
                                    "modelArn": (
                                        f"arn:aws:bedrock:{region}::"
                                        f"foundation-model/{p.split('.', 1)[1]}"
                                    )
                                }
                            ],
                            "inferenceProfileId": p,
                            "status": "ACTIVE",
                            "type": "SYSTEM_DEFINED",
                        }
                        for p in profiles
                    ]
                },
                {"maxResults": 1000, "typeEquals": "SYSTEM_DEFINED"},
            )
        self.stubber.activate()

    def client(self, service_name: str, **kwargs: Any) -> Any:
        return self.bedrock if service_name == "bedrock" else object()

    def get_credentials(self) -> Any:
        return None


@pytest.fixture(autouse=True)
def _clear_profile_cache() -> Any:
    BedrockCrossRegionModelHelper._profiles_by_region.clear()
    yield
    BedrockCrossRegionModelHelper._profiles_by_region.clear()


def _factory(
    region: str, profiles: list[str] | None, *, enable_global: bool = True
) -> tuple[BedrockEmbeddingModelFactory, _StubbedBedrockSession]:
    config = Config()
    config.aws.bedrock.enable_global_profile = enable_global
    session = _StubbedBedrockSession(region, profiles)
    factory = BedrockEmbeddingModelFactory(
        config,
        boto_session=session,  # type: ignore[arg-type]
        region_name=region,
    )
    return factory, session


@pytest.mark.parametrize(
    ("enable_global", "expected"),
    [(True, f"global.{EMBED_V4}"), (False, f"us.{EMBED_V4}")],
)
def test_embed_v4_resolves_profile_in_us_west_2(
    enable_global: bool, expected: str
) -> None:
    factory, session = _factory(
        "us-west-2",
        [f"us.{EMBED_V4}", f"global.{EMBED_V4}", f"eu.{EMBED_V4}"],
        enable_global=enable_global,
    )
    model = factory.get_model(EmbeddingModelId.EMBED_V4)
    assert model.model_id == expected
    # The request body stays Cohere's despite the profile prefix.
    assert model._inferred_provider == "cohere"
    session.stubber.assert_no_pending_responses()


def test_embed_v4_without_profile_in_us_west_2_fails_fast() -> None:
    # No profile visible (here: ListInferenceProfiles denied) and no on-demand
    # throughput in the region: invoking the bare id would fail on the first
    # embed call with an opaque ValidationException.
    factory, _ = _factory("us-west-2", None)
    with pytest.raises(EmbeddingModelError, match="no on-demand throughput"):
        factory.get_model(EmbeddingModelId.EMBED_V4)


def test_embed_v4_seoul_needs_the_global_profile() -> None:
    # ap-northeast-2 has only the global profile for Embed v4.
    factory, _ = _factory("ap-northeast-2", [f"global.{EMBED_V4}"], enable_global=False)
    with pytest.raises(EmbeddingModelError, match="enable_global_profile"):
        factory.get_model(EmbeddingModelId.EMBED_V4)


def test_embed_v4_on_demand_region_keeps_bare_id_without_profile() -> None:
    factory, _ = _factory("us-east-1", [])
    assert factory.get_model(EmbeddingModelId.EMBED_V4).model_id == EMBED_V4


@pytest.mark.parametrize("region", ["us-east-1", "eu-west-1", "ap-northeast-1"])
def test_embed_v4_on_demand_region_ignores_listed_profiles(region: str, mocker) -> None:
    # Embed v4 has on-demand throughput here: a profile in the list must not
    # move its calls to the profile's quota, routing and IAM resources.
    factory, session = _factory(
        region, [f"us.{EMBED_V4}", f"eu.{EMBED_V4}", f"global.{EMBED_V4}"]
    )
    lister = mocker.spy(session.bedrock, "list_inference_profiles")
    assert factory.get_model(EmbeddingModelId.EMBED_V4).model_id == EMBED_V4
    lister.assert_not_called()


@pytest.mark.parametrize(
    ("region", "profiles", "expected"),
    [
        ("us-west-2", [f"us.{EMBED_V4}"], f"us.{EMBED_V4}"),
        ("ap-northeast-2", [f"global.{EMBED_V4}"], f"global.{EMBED_V4}"),
    ],
)
def test_embed_v4_profile_only_region_uses_the_profile(
    region: str, profiles: list[str], expected: str
) -> None:
    factory, session = _factory(region, profiles)
    assert factory.get_model(EmbeddingModelId.EMBED_V4).model_id == expected
    session.stubber.assert_no_pending_responses()


def test_titan_v2_stays_on_demand() -> None:
    # Titan V2 has no system-defined profile: the bare id is invoked as before.
    factory, _ = _factory("us-west-2", [f"us.{EMBED_V4}"])
    model = factory.get_model(EmbeddingModelId.TITAN_EMBED_V2, dimensions=512)
    assert model.model_id == EmbeddingModelId.TITAN_EMBED_V2.value
    assert model.model_kwargs == {"dimensions": 512}


@pytest.mark.parametrize(
    "given",
    [
        f"us.{EMBED_V4}",
        f"arn:aws:bedrock:us-west-2:123456789012:inference-profile/global.{EMBED_V4}",
    ],
)
def test_profile_id_or_arn_passes_through(given: str, mocker) -> None:
    factory, session = _factory("us-west-2", [])
    lister = mocker.spy(session.bedrock, "list_inference_profiles")
    model = factory.get_model(given)
    assert model.model_id == given
    assert model._inferred_provider == "cohere"
    assert factory.get_model_info(given) == factory.get_model_info(EMBED_V4)
    lister.assert_not_called()


def test_application_profile_arn_is_not_a_catalog_model() -> None:
    factory, _ = _factory("us-west-2", [])
    arn = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc"
    assert factory.get_model_info(arn) is None
    with pytest.raises(EmbeddingModelError, match="Unsupported embedding model"):
        factory.get_model(arn)


def test_profile_target_model_id() -> None:
    assert profile_target_model_id(f"apac.{EMBED_V4}") == EMBED_V4
    assert (
        profile_target_model_id(
            f"arn:aws:bedrock:us-west-2::inference-profile/us.{EMBED_V4}"
        )
        == EMBED_V4
    )
    assert profile_target_model_id(EMBED_V4) == EMBED_V4


def test_chat_resolver_passes_arns_through(mocker) -> None:
    lister = mocker.patch.object(
        BedrockCrossRegionModelHelper, "_get_available_profiles"
    )
    arn = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/x"
    out = bedrock_mod.BedrockCrossRegionModelHelper.get_cross_region_model_id(
        object(), arn, "us-west-2"  # type: ignore[arg-type]
    )
    assert out == arn
    lister.assert_not_called()
