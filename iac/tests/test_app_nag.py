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


@pytest.mark.parametrize("context", [{}, _PROD_HARDENED], ids=["dev", "prod-hardened"])
def test_concrete_account_synth_has_no_nag_errors(
    context: dict[str, Any], tmp_path: Path
) -> None:
    assert _synth_errors(context, tmp_path) == []
