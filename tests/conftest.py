# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures.

Tests run AWS-free by default: the ``aws`` marker gates the few that need real
services, and port-based fakes (``tests/fixtures/fakes``) stand in for Neptune /
OpenSearch / DynamoDB. ``moto`` provides mocked AWS APIs where an adapter must
be exercised against a boto3 surface.
"""

from __future__ import annotations

import os

import pytest
from hypothesis import settings

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.adapters.aws.bedrock import BedrockCrossRegionModelHelper
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Config,
    Entity,
    Relationship,
    TextUnit,
)

# Hypothesis profiles, selected with HYPOTHESIS_PROFILE (default: "default").
# "ci" removes the per-example deadline, whose wall-clock limit flakes on
# shared runners, and derandomizes generation so a failure reproduces on
# re-run; print_blob makes any failure replayable locally via @reproduce_failure.
settings.register_profile("ci", deadline=None, derandomize=True, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

_PROFILE_PREFIXES = ("global", "us", "eu", "apac")

# Ambient AWS settings that would make boto3 resolve the developer's real
# credentials/profile (or fail with ProfileNotFound when it does not exist).
_AWS_ENV_TO_CLEAR = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_REGION",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
)
# ConfigLoader._apply_environment_overrides (unified_kg_rag/shared/config.py)
# maps these onto the loaded Config (AWS_PROFILE/AWS_REGION are cleared above),
# so a value exported in the developer's shell would silently change what the
# tests exercise.
_CONFIG_ENV_OVERRIDES = (
    "LOG_LEVEL",
    "LOG_FORMAT",
    "LOG_TO_FILE",
    "LOG_FILE_PATH",
    "NEPTUNE_ENDPOINT",
    "OPENSEARCH_ENDPOINT",
    "OPENSEARCH_USERNAME",
    "OPENSEARCH_PASSWORD",
    "BEDROCK_REGION",
    "BEDROCK_GUARDRAIL_IDENTIFIER",
    "S3_BUCKET_NAME",
    "GRAPHRAG_DOC_STATUS_TABLE",
    "GRAPHRAG_DOC_STATUS_CREATE_TABLE",
)
_FAKE_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",  # nosec B105 - placeholder, not a secret
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_CONFIG_FILE": os.devnull,
    "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
    "AWS_EC2_METADATA_DISABLED": "true",
}


@pytest.fixture(autouse=True)
def _isolated_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every test to fake, local-only AWS credentials and config.

    Without this the suite inherits the developer's shell: a non-existent
    ``AWS_PROFILE`` makes boto3 raise ``ProfileNotFound`` in hundreds of tests,
    and a valid one (or an instance role via IMDS) lets an accidentally
    unmocked client reach a real account. Config files point at ``/dev/null``
    and IMDS is disabled so no credential source other than the fake keys is
    consulted. Tests that need a specific value set it with ``monkeypatch``.
    """
    for name in _AWS_ENV_TO_CLEAR + _CONFIG_ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    for name, value in _FAKE_AWS_ENV.items():
        monkeypatch.setenv(name, value)


class RealAWSCallBlocked(RuntimeError):
    """Raised when a non-``aws`` test lets a botocore request reach the wire."""


@pytest.fixture(autouse=True)
def _block_real_aws_http(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail an unmocked botocore HTTP send immediately instead of dialling AWS.

    botocore treats connection errors as retryable, so an unmocked client used
    to cost ~20 s of backoff per test offline and silently succeeded (or was
    denied) against a real account online. A plain ``RuntimeError`` subclass is
    not retried, so the code under test sees one fast, deterministic failure.
    ``moto`` intercepts requests before this transport layer and is unaffected;
    tests marked ``aws`` keep the real transport.
    """
    if request.node.get_closest_marker("aws") is not None:
        return
    from botocore.httpsession import URLLib3Session

    def _blocked_send(self: URLLib3Session, aws_request: object) -> object:
        url = getattr(aws_request, "url", "<unknown>")
        raise RealAWSCallBlocked(f"real AWS HTTP call blocked in tests: {url}")

    monkeypatch.setattr(URLLib3Session, "send", _blocked_send)


_OFFLINE_REGIONS = ("us-east-1", "us-west-2", "eu-west-1", "ap-northeast-2")


@pytest.fixture(autouse=True)
def _offline_inference_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve Bedrock inference profiles from a static set, never from AWS.

    Profile resolution otherwise calls ``bedrock:ListInferenceProfiles``. That
    call succeeds on a developer machine with credentials and fails in CI, so
    profile-only models (e.g. Claude Haiku 4.5) behaved differently between the
    two. Tests that exercise the lookup itself clear this cache and stub the
    client explicitly.
    """
    from unified_kg_rag.domain.models import LanguageModelId

    profiles = {
        f"{prefix}.{model.value}"
        for model in LanguageModelId
        for prefix in _PROFILE_PREFIXES
    }
    monkeypatch.setattr(
        BedrockCrossRegionModelHelper,
        "_profiles_by_region",
        {region: set(profiles) for region in _OFFLINE_REGIONS},
    )


@pytest.fixture
def config() -> Config:
    """A default ``Config`` (all nested defaults, no external services)."""
    return Config()


@pytest.fixture
def fake_doc_status() -> FakeDocStatusStore:
    """An empty in-memory DocStatusPort implementation."""
    return FakeDocStatusStore()


@pytest.fixture
def sample_entities() -> list[Entity]:
    return [
        Entity(id="e1", name="Alice", type="PERSON", description="A researcher"),
        Entity(id="e2", name="Acme Corp", type="ORG", description="A company"),
        Entity(id="e3", name="Seattle", type="GPE", description="A city"),
    ]


@pytest.fixture
def sample_relationships() -> list[Relationship]:
    return [
        Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            source_name="Alice",
            target_name="Acme Corp",
            description="Alice works at Acme Corp",
            weight=1.0,
        ),
        Relationship(
            id="r2",
            source_id="e2",
            target_id="e3",
            source_name="Acme Corp",
            target_name="Seattle",
            description="Acme Corp is based in Seattle",
            weight=0.8,
        ),
    ]


@pytest.fixture
def sample_text_units() -> list[TextUnit]:
    return [
        TextUnit(id="t1", text="Alice works at Acme Corp.", entity_ids=["e1", "e2"]),
        TextUnit(
            id="t2", text="Acme Corp is based in Seattle.", entity_ids=["e2", "e3"]
        ),
    ]


@pytest.fixture
def sample_communities() -> list[Community]:
    return [
        Community(
            id="c1",
            name="Acme cluster",
            level="0",
            parent="",
            children=[],
            entity_ids=["e1", "e2", "e3"],
            text_unit_ids=["t1", "t2"],
        ),
    ]


@pytest.fixture
def sample_community_reports() -> list[CommunityReport]:
    return [
        CommunityReport(
            id="cr1",
            community_id="c1",
            name="Acme cluster report",
            summary="Alice, Acme Corp, and Seattle form a cluster.",
            full_content="Detailed report about the Acme cluster.",
        ),
    ]
