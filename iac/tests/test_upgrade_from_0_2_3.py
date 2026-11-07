# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upgrading a 0.2.3 stack set: the doc-status table is replaced, and the
0.2.3 compute and observability stacks import its ARN and name (offline
synth, no lookups).

CloudFormation changes no export another stack imports, so a plain
``cdk deploy --all`` from 0.2.3 failed at the storage stack ("Cannot update
export ...DocStatusTable...Arn... as it is in use by ...Compute"). With
``legacy_doc_status_table`` the storage stack keeps both exports at their
0.2.3 values and the stacks that imported them reference the new table by
name, so the deploy that replaces the table changes no imported export.
"""

from __future__ import annotations

import json
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.compute_stack import ComputeStack
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.observability_stack import ObservabilityStack
from iac.stacks.orchestration_stack import OrchestrationStack
from iac.stacks.storage_stack import LEGACY_DOC_STATUS_EXPORTS, StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")
_ARN_ID, _NAME_ID = LEGACY_DOC_STATUS_EXPORTS
_TABLE = "DocStatusTable4F466A9B"


def _templates(context: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    app = cdk.App(context=context or {})
    config = DeploymentConfig.from_context(app)
    networking = NetworkingStack(app, "Net", config=config, env=_ENV)
    storage = StorageStack(app, "Store", config=config, networking=networking, env=_ENV)
    compute = ComputeStack(
        app, "Compute", config=config, networking=networking, storage=storage, env=_ENV
    )
    orchestration = OrchestrationStack(
        app,
        "Orch",
        config=config,
        networking=networking,
        compute=compute,
        cache_bucket_name=storage.cache_bucket.bucket_name,
        env=_ENV,
    )
    observability = ObservabilityStack(
        app,
        "Obs",
        config=config,
        orchestration=orchestration,
        storage=storage,
        env=_ENV,
    )
    stacks = (networking, storage, compute, orchestration, observability)
    return {s.stack_name: Template.from_stack(s).to_json() for s in stacks}


def _importers(templates: dict[str, dict[str, Any]], export: str) -> list[str]:
    return sorted(
        name
        for name, template in templates.items()
        if json.dumps({"Fn::ImportValue": export}) in json.dumps(template)
    )


def test_the_legacy_ids_are_the_exports_cdk_makes_for_the_table() -> None:
    # The ids 0.2.3's synth gave the exports (the same construct path): a
    # default synth still makes them, and the same stacks import them.
    templates = _templates()
    outputs = templates["Store"]["Outputs"]
    assert outputs[_ARN_ID] == {
        "Value": {"Fn::GetAtt": [_TABLE, "Arn"]},
        "Export": {"Name": f"Store:{_ARN_ID}"},
    }
    assert outputs[_NAME_ID] == {
        "Value": {"Ref": _TABLE},
        "Export": {"Name": f"Store:{_NAME_ID}"},
    }
    assert _importers(templates, f"Store:{_ARN_ID}") == ["Compute"]
    assert _importers(templates, f"Store:{_NAME_ID}") == ["Obs"]


def test_the_upgrade_deploy_keeps_the_0_2_3_exports() -> None:
    templates = _templates({"legacy_doc_status_table": "graphrag-doc-status"})
    outputs = templates["Store"]["Outputs"]
    # The 0.2.3 values (the old table's ARN and name), under the 0.2.3 names.
    assert outputs[_ARN_ID] == {
        "Value": {
            "Fn::Join": [
                "",
                [
                    "arn:",
                    {"Ref": "AWS::Partition"},
                    ":dynamodb:us-west-2:111111111111:table/graphrag-doc-status",
                ],
            ]
        },
        "Export": {"Name": f"Store:{_ARN_ID}"},
    }
    assert outputs[_NAME_ID] == {
        "Value": "graphrag-doc-status",
        "Export": {"Name": f"Store:{_NAME_ID}"},
    }
    # Nothing imports them: the compute and observability stacks name the
    # new table instead.
    for export in (_ARN_ID, _NAME_ID):
        assert _importers(templates, f"Store:{export}") == []
    compute = json.dumps(templates["Compute"])
    assert "table/graphrag-doc-status-v2" in compute
    assert 'table/graphrag-doc-status"' not in compute
    alarms = [
        r["Properties"]
        for r in templates["Obs"]["Resources"].values()
        if r["Type"] == "AWS::CloudWatch::Alarm"
        and r["Properties"].get("Namespace") == "AWS/DynamoDB"
    ]
    assert alarms and all(
        {"Name": "TableName", "Value": "graphrag-doc-status-v2"}
        in a.get("Dimensions", [])
        for a in alarms
    ), alarms
    # The new table itself is unchanged by the setting.
    table = templates["Store"]["Resources"][_TABLE]
    assert table["Properties"]["TableName"] == "graphrag-doc-status-v2"
    assert table == _templates()["Store"]["Resources"][_TABLE]


def test_the_grants_on_the_new_table_are_the_same_by_name() -> None:
    def statements(templates: dict[str, dict[str, Any]]) -> list[Any]:
        found = []
        for resource in templates["Compute"]["Resources"].values():
            if resource["Type"] != "AWS::IAM::Policy":
                continue
            for st in resource["Properties"]["PolicyDocument"]["Statement"]:
                if any("dynamodb:" in a for a in json.dumps(st["Action"]).split('"')):
                    found.append(sorted(st["Action"]))
        return found

    assert statements(
        _templates({"legacy_doc_status_table": "graphrag-doc-status"})
    ) == statements(_templates())


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("graphrag-doc-status-v2", "names the current doc_status_table"),
        ("bad name!", "must be a DynamoDB table name"),
    ],
)
def test_a_legacy_table_name_is_checked(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DeploymentConfig.from_context(
            cdk.App(context={"legacy_doc_status_table": value})
        )
