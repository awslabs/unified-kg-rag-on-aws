# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context values are validated with errors that name the key (no synth)."""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
import pytest

from iac.config import DeploymentConfig


def _config(context: dict[str, Any]) -> DeploymentConfig:
    return DeploymentConfig.from_context(cdk.App(context=context))


def test_bedrock_model_arns_accepts_a_comma_separated_cli_string() -> None:
    # `-c bedrock_model_arns=a,b` arrives as one string.
    arns = (
        "arn:aws:bedrock:us-west-2::foundation-model/model-a, "
        "arn:aws:bedrock:us-west-2::foundation-model/model-b"
    )
    config = _config({"bedrock_model_arns": arns})
    assert config.bedrock_model_arns == [
        "arn:aws:bedrock:us-west-2::foundation-model/model-a",
        "arn:aws:bedrock:us-west-2::foundation-model/model-b",
    ]
    listed = _config({"bedrock_model_arns": ["arn:aws:bedrock:::x"]})
    assert listed.bedrock_model_arns == ["arn:aws:bedrock:::x"]
    assert _config({}).bedrock_model_arns is None


def test_bedrock_model_arns_rejects_a_value_that_is_not_an_arn() -> None:
    with pytest.raises(ValueError, match="bedrock_model_arns: not an ARN: model-a"):
        _config({"bedrock_model_arns": "model-a"})


@pytest.mark.parametrize("key", ["use_cmk", "deletion_protection", "removal_destroy"])
def test_a_misspelled_boolean_is_rejected_not_read_as_false(key: str) -> None:
    with pytest.raises(ValueError, match=f"{key} must be a boolean"):
        _config({key: "ture"})


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("Yes", True), ("1", True), ("off", False), (False, False)],
)
def test_booleans_accept_the_usual_spellings(value: Any, expected: bool) -> None:
    assert _config({"use_cmk": value}).use_cmk is expected


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("sfn_timeout_hours", "0", "sfn_timeout_hours must be at least 1"),
        ("backup_retention_days", "0", "backup_retention_days must be at least 1"),
        ("backup_retention_days", "36", "backup_retention_days must be at most 35"),
        ("opensearch_count", "0", "opensearch_count must be at least 1"),
        ("neptune_instances", "0", "neptune_instances must be at least 1"),
        ("max_azs", "1", "max_azs must be at least 2"),
        ("opensearch_volume_gib", "abc", "opensearch_volume_gib must be an integer"),
    ],
)
def test_out_of_range_numbers_are_rejected(key: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config({key: value})


@pytest.mark.parametrize(
    ("cpu", "memory", "message"),
    [
        ("1000", "2048", "fargate_cpu must be one of"),
        ("2048", "2048", "fargate_memory 2048 is not valid with fargate_cpu 2048"),
        ("512", "8192", "fargate_memory 8192 is not valid with fargate_cpu 512"),
    ],
)
def test_invalid_fargate_sizes_are_rejected(
    cpu: str, memory: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config({"fargate_cpu": cpu, "fargate_memory": memory})


def test_valid_fargate_sizes_pass() -> None:
    config = _config({"fargate_cpu": "4096", "fargate_memory": "16384"})
    assert (config.fargate_cpu, config.fargate_memory) == (4096, 16384)
    assert (_config({}).fargate_cpu, _config({}).fargate_memory) == (2048, 8192)


@pytest.mark.parametrize("value", ["", "nori,cjk", "../x", "-nori"])
def test_korean_analyzer_must_be_an_analyzer_name(value: str) -> None:
    with pytest.raises(ValueError, match="opensearch_korean_analyzer"):
        _config({"opensearch_korean_analyzer": value})


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1-16", (1.0, 16.0)), ("2.5-32", (2.5, 32.0)), ([1, 128], (1.0, 128.0))],
)
def test_neptune_serverless_ncu_parses_a_range(value: Any, expected: Any) -> None:
    assert _config({"neptune_serverless_ncu": value}).neptune_serverless_ncu == expected
    assert _config({}).neptune_serverless_ncu is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("16", "MIN-MAX"),
        ("a-b", "MIN-MAX"),
        ("0.5-4", "1-128 NCU in steps of 0.5"),
        ("1-200", "1-128 NCU in steps of 0.5"),
        ("1.3-4", "1-128 NCU in steps of 0.5"),
        ("8-2", "MIN \\(8\\) is above MAX \\(2\\)"),
    ],
)
def test_neptune_serverless_ncu_rejects_bad_ranges(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=f"neptune_serverless_ncu.*{message}"):
        _config({"neptune_serverless_ncu": value})


def test_neptune_serverless_ncu_excludes_an_instance_class() -> None:
    with pytest.raises(ValueError, match="exclusive"):
        _config({"neptune_serverless_ncu": "1-8", "neptune_instance": "db.r6g.xlarge"})


def test_a_misspelled_context_key_fails_naming_the_closest_setting() -> None:
    with pytest.raises(
        ValueError, match=r"neptune_instnace \(did you mean neptune_instance\?\)"
    ):
        _config({"neptune_instnace": "db.r6g.xlarge"})
    with pytest.raises(ValueError, match="guardrail-identifier"):
        _config({"guardrail-identifier": "gr-1"})


def test_cdk_keys_and_notes_are_not_settings() -> None:
    _config(
        {
            "@aws-cdk/core:checkSecretUsage": True,
            "aws:cdk:enable-path-metadata": True,
            "availability-zones:account=111111111111:region=us-west-2": ["a"],
            "acknowledged-issue-numbers": [1],
            "_doc": "a note",
            "env_name": "dev",
        }
    )
