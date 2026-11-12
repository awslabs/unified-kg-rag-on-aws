# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""bedrock_region through the full app (offline synth, no lookups).

The private-mode guard lives in app.py, so these run app.py as `cdk synth`
does, with the AZ lookup pre-seeded through CDK_CONTEXT_JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_IAC = Path(__file__).resolve().parent.parent
_ACCOUNT = "111111111111"
_REGION = "us-west-2"


def _synth(context: dict[str, str], outdir: Path) -> subprocess.CompletedProcess:
    ctx = {
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
    return subprocess.run(
        [sys.executable, str(_IAC / "app.py")],
        cwd=_IAC,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("context", "bedrock_region"),
    [
        ({}, _REGION),
        ({"bedrock_region": _REGION}, _REGION),
        ({"network_mode": "public", "bedrock_region": "us-east-1"}, "us-east-1"),
        ({"bedrock_region": "us-east-1"}, None),
    ],
    ids=["unset", "private-same-region", "public-cross-region", "private-cross"],
)
def test_bedrock_region_sets_task_env_and_guardrail_region(
    context: dict[str, str], bedrock_region: str | None, tmp_path: Path
) -> None:
    result = _synth(context, tmp_path)
    if bedrock_region is None:
        # Private mode has Bedrock endpoints in the deploy region only.
        assert result.returncode != 0
        assert "network_mode=private but bedrock_region" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    guardrail = manifest["artifacts"]["GraphRagGuardrail"]
    assert guardrail["environment"] == f"aws://{_ACCOUNT}/{bedrock_region}"
    # The ingestion task definition is the orchestration stack's.
    orchestration = json.loads(
        (tmp_path / "GraphRagOrchestration.template.json").read_text()
    )
    (task_def,) = [
        r
        for logical_id, r in orchestration["Resources"].items()
        if r["Type"] == "AWS::ECS::TaskDefinition"
        and logical_id.startswith("IngestionTask")
    ]
    (app,) = [
        c for c in task_def["Properties"]["ContainerDefinitions"] if c["Name"] == "app"
    ]
    env = {e["Name"]: e["Value"] for e in app["Environment"]}
    assert env["UKR__AWS__BEDROCK__REGION_NAME"] == bedrock_region
    assert env["AWS_REGION"] == _REGION


def test_a_private_synth_rejects_a_secret_of_another_region(tmp_path: Path) -> None:
    secret = "arn:aws:secretsmanager:us-east-1:111111111111:secret:obs/hdr-AbC123"
    result = _synth({"observability_secret_arn": secret}, tmp_path)
    assert result.returncode != 0
    assert "observability_secret_arn is in 'us-east-1'" in result.stderr


def _app_containers(outdir: Path) -> list[dict]:
    """The app container of both task definitions (query: compute stack,
    ingestion: orchestration stack)."""
    containers = []
    for stack in ("GraphRagCompute", "GraphRagOrchestration"):
        template = json.loads((outdir / f"{stack}.template.json").read_text())
        containers += [
            c
            for r in template["Resources"].values()
            if r["Type"] == "AWS::ECS::TaskDefinition"
            for c in r["Properties"]["ContainerDefinitions"]
            if c["Name"] == "app"
        ]
    assert len(containers) == 2
    return containers


def _guardrail_grants(outdir: Path) -> list[str]:
    template = (outdir / "GraphRagCompute.template.json").read_text()
    return [
        json.dumps(s["Resource"])
        for r in json.loads(template)["Resources"].values()
        if r["Type"] == "AWS::IAM::Policy"
        for s in r["Properties"]["PolicyDocument"]["Statement"]
        if s["Action"] == "bedrock:ApplyGuardrail"
    ]


def test_the_default_deploy_attaches_the_guardrail_it_creates(tmp_path: Path) -> None:
    # The README quick start (`cdk deploy --all`, no guardrail context) used
    # to create the guardrail and screen nothing with it.
    result = _synth({}, tmp_path)
    assert result.returncode == 0, result.stderr
    for app in _app_containers(tmp_path):
        env = {e["Name"]: e["Value"] for e in app["Environment"]}
        assert env["UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER"] == (
            "{{resolve:ssm:/graphrag/guardrail/identifier}}"
        )
        assert env["UKR__AWS__BEDROCK__GUARDRAIL__VERSION"] == (
            "{{resolve:ssm:/graphrag/guardrail/version}}"
        )
        assert app["DockerLabels"]["guardrail-policy"]
    # Both task roles may apply that one guardrail.
    grants = _guardrail_grants(tmp_path)
    assert len(grants) == 2
    for grant in grants:
        assert "guardrail/{{resolve:ssm:/graphrag/guardrail/identifier}}" in grant
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    # Deployed after the parameters it reads, and through no export: a policy
    # change replaces the version, and an export in use cannot change.
    assert (
        "GraphRagGuardrail" in manifest["artifacts"]["GraphRagCompute"]["dependencies"]
    )
    guardrail = (tmp_path / "GraphRagGuardrail.template.json").read_text()
    assert "Export" not in guardrail
    assert "is not attached to compute" not in result.stderr


def test_a_cross_region_guardrail_still_needs_the_context(tmp_path: Path) -> None:
    result = _synth({"network_mode": "public", "bedrock_region": "us-east-1"}, tmp_path)
    assert result.returncode == 0, result.stderr
    for app in _app_containers(tmp_path):
        names = {e["Name"] for e in app["Environment"]}
        assert "UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER" not in names
    assert _guardrail_grants(tmp_path) == []
    result = _synth(
        {
            "network_mode": "public",
            "bedrock_region": "us-east-1",
            "guardrail_identifier": "gr-1",
            "guardrail_version": "3",
        },
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    for app in _app_containers(tmp_path):
        env = {e["Name"]: e["Value"] for e in app["Environment"]}
        assert env["UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER"] == "gr-1"
        assert env["UKR__AWS__BEDROCK__GUARDRAIL__VERSION"] == "3"
        assert "DockerLabels" not in app
