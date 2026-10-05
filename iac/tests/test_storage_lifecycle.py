# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""StorageStack lifecycle assertions (offline synth, no lookups).

Non-dev data stores must survive a `cdk destroy` / stack deletion (RETAIN) and a
direct delete API call (deletion protection). Dev keeps the cheap tear-down.
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from iac.config import DeploymentConfig
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.storage_stack import StorageStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")
_DATA_STORES = (
    "AWS::S3::Bucket",
    "AWS::DynamoDB::Table",
    "AWS::Neptune::DBCluster",
    "AWS::OpenSearchService::Domain",
)


def _storage_template(context: dict[str, Any]) -> Template:
    app = cdk.App(context=context)
    config = DeploymentConfig.from_context(app)
    networking = NetworkingStack(app, "Net", config=config, env=_ENV)
    storage = StorageStack(app, "Store", config=config, networking=networking, env=_ENV)
    return Template.from_stack(storage)


@pytest.fixture(scope="module")
def prod() -> Template:
    return _storage_template({"env_name": "prod"})


@pytest.mark.parametrize("resource_type", _DATA_STORES)
def test_non_dev_data_stores_are_retained(prod: Template, resource_type: str) -> None:
    resources = prod.find_resources(resource_type)
    assert resources, f"no {resource_type} synthesized"
    for logical_id, resource in resources.items():
        assert resource.get("DeletionPolicy") == "Retain", logical_id
        assert resource.get("UpdateReplacePolicy") == "Retain", logical_id


def test_non_dev_enables_deletion_protection(prod: Template) -> None:
    (table,) = prod.find_resources("AWS::DynamoDB::Table").values()
    assert table["Properties"]["DeletionProtectionEnabled"] is True
    (cluster,) = prod.find_resources("AWS::Neptune::DBCluster").values()
    assert cluster["Properties"]["DeletionProtection"] is True


def test_non_dev_buckets_do_not_auto_delete_objects(prod: Template) -> None:
    assert not prod.find_resources("Custom::S3AutoDeleteObjects")


def test_dev_defaults_tear_down() -> None:
    dev = _storage_template({})
    (cluster,) = dev.find_resources("AWS::Neptune::DBCluster").values()
    assert cluster["DeletionPolicy"] == "Delete"
    assert cluster["Properties"].get("DeletionProtection") in (False, None)
