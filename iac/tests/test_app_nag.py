# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-app cdk-nag synth with a concrete account (offline, no lookups).

CI's `cdk synth` runs without an account, so account ids render as the
<AWS::AccountId> token. A real deploy sets CDK_DEFAULT_ACCOUNT and the same ARNs
contain the literal 12-digit id; suppressions must match both shapes. The AZ
lookup is pre-seeded through CDK_CONTEXT_JSON so no AWS call is needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_IAC = Path(__file__).resolve().parent.parent
_ACCOUNT = "111111111111"
_REGION = "us-west-2"
_PROD_HARDENED = {
    "use_cmk": "true",
    "vpc_flow_logs": "true",
    "neptune_instances": "2",
    "opensearch_count": "2",
    "deletion_protection": "true",
    "removal_destroy": "false",
}


def _synth_errors(context: dict[str, Any], outdir: Path) -> list[str]:
    ctx = {
        "enable_cdk_nag": "true",
        f"availability-zones:account={_ACCOUNT}:region={_REGION}": [
            f"{_REGION}a",
            f"{_REGION}b",
        ],
        **context,
    }
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("AWS_", "CDK_")) and k != "PYTHONPATH"
    }
    env.update(
        CDK_DEFAULT_ACCOUNT=_ACCOUNT,
        CDK_DEFAULT_REGION=_REGION,
        CDK_OUTDIR=str(outdir),
        CDK_CONTEXT_JSON=json.dumps(ctx),
        AWS_EC2_METADATA_DISABLED="true",
    )
    subprocess.run(
        [sys.executable, str(_IAC / "app.py")],
        cwd=_IAC,
        env=env,
        check=True,
        capture_output=True,
    )
    manifest = json.loads((outdir / "manifest.json").read_text())
    assert not manifest.get("missing"), manifest["missing"]
    errors = []
    for artifact in manifest["artifacts"].values():
        # Recent CDK versions move stack metadata out of manifest.json into a
        # per-stack file; read both so the check works either way.
        metadata = dict(artifact.get("metadata", {}))
        if "additionalMetadataFile" in artifact:
            extra = outdir / artifact["additionalMetadataFile"]
            metadata.update(json.loads(extra.read_text()))
        for path, entries in metadata.items():
            errors += [
                f"{path}: {e['data']}" for e in entries if e["type"] == "aws:cdk:error"
            ]
    return errors


_TRACING = {"enable_cloudwatch_tracing": "true"}
_NORI = {"opensearch_nori_package": "true"}
# Scoped model ARNs as iac/README.md documents them: a `*` region, since
# inference profiles route across regions.
_SCOPED_MODELS = {
    "bedrock_model_arns": (
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0,"
        "arn:aws:bedrock:*::foundation-model/amazon.rerank-v1:0"
    )
}


@pytest.mark.parametrize(
    "context",
    [
        {},
        _PROD_HARDENED,
        _TRACING,
        {**_PROD_HARDENED, **_TRACING},
        {**_TRACING, "network_mode": "public"},
        {**_PROD_HARDENED, "ingestion_fargate_spot": "true"},
        {**_TRACING, "fargate_cpu_architecture": "arm64"},
        _NORI,
        {**_PROD_HARDENED, **_NORI},
        _SCOPED_MODELS,
        {**_PROD_HARDENED, "neptune_audit_log": "true"},
    ],
    ids=[
        "dev",
        "prod-hardened",
        "tracing",
        "prod-hardened-tracing",
        "public-tracing",
        "prod-hardened-spot",
        "tracing-arm64",
        "nori",
        "prod-hardened-nori",
        "scoped-models",
        "prod-hardened-neptune-audit",
    ],
)
def test_concrete_account_synth_has_no_nag_errors(
    context: dict[str, Any], tmp_path: Path
) -> None:
    assert _synth_errors(context, tmp_path) == []


def _cdk_json_context() -> dict[str, Any]:
    """The context of cdk.json, which the cdk CLI passes to the app."""
    context: dict[str, Any] = json.loads((_IAC / "cdk.json").read_text())["context"]
    return context


def _warnings(outdir: Path) -> list[str]:
    return [
        str(e["data"])
        for metadata in outdir.glob("*.metadata.json")
        for entries in json.loads(metadata.read_text()).values()
        for e in entries
        if e["type"] == "aws:cdk:warning"
    ]


def _tag_keys(outdir: Path, stack: str) -> set[str]:
    template = json.loads((outdir / f"{stack}.template.json").read_text())
    return {
        tag["Key"]
        for resource in template["Resources"].values()
        for tag in resource.get("Properties", {}).get("Tags", [])
        if isinstance(tag, dict) and "Key" in tag
    }


def test_cdk_json_pins_the_flags_synth_warned_about(tmp_path: Path) -> None:
    # Set to the behaviour the stacks already had (strong cross-stack
    # references; the custom resources' bundled SDK), so synth no longer
    # warns that they defaulted.
    assert _synth_errors(_cdk_json_context(), tmp_path) == []
    warnings = "\n".join(_warnings(tmp_path))
    assert "cross-stack-reference strength" not in warnings
    assert "installLatestAwsSdk" not in warnings
    # Without use_cmk there is no security stack (it would hold nothing),
    # and without an owner no resource carries an owner tag.
    assert not (tmp_path / "GraphRagSecurity.template.json").exists()
    keys = _tag_keys(tmp_path, "GraphRagStorage")
    assert "project" in keys and "owner" not in keys


def test_the_security_stack_and_owner_tag_when_asked(tmp_path: Path) -> None:
    context = {**_cdk_json_context(), "use_cmk": "true", "owner": "data-team"}
    assert _synth_errors(context, tmp_path) == []
    assert (tmp_path / "GraphRagSecurity.template.json").exists()
    assert "owner" in _tag_keys(tmp_path, "GraphRagStorage")


def test_the_app_refuses_an_env_name_too_long_for_its_names(tmp_path: Path) -> None:
    # check_physical_names runs at synth, before CloudFormation would refuse
    # the bucket or guardrail name mid-deploy.
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _synth_errors({"env_name": "x" * 30}, tmp_path)
    assert b"is too long for the cache bucket name" in raised.value.stderr
