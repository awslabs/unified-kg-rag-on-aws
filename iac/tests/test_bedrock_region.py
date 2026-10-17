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
    compute = json.loads((tmp_path / "GraphRagCompute.template.json").read_text())
    (task_def,) = [
        r
        for logical_id, r in compute["Resources"].items()
        if r["Type"] == "AWS::ECS::TaskDefinition"
        and logical_id.startswith("IngestionTask")
    ]
    (app,) = [
        c for c in task_def["Properties"]["ContainerDefinitions"] if c["Name"] == "app"
    ]
    env = {e["Name"]: e["Value"] for e in app["Environment"]}
    assert env["UKR__AWS__BEDROCK__REGION_NAME"] == bedrock_region
    assert env["AWS_REGION"] == _REGION
