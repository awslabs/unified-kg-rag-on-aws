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
from aws_cdk.assertions import Match, Template

from iac.config import DeploymentConfig
from iac.stacks.networking_stack import NetworkingStack
from iac.stacks.storage_stack import CACHE_EXPIRY_PREFIXES, StorageStack

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


def test_multi_node_opensearch_uses_smaller_dedicated_masters(prod: Template) -> None:
    (domain,) = prod.find_resources("AWS::OpenSearchService::Domain").values()
    cluster = domain["Properties"]["ClusterConfig"]
    assert cluster["DedicatedMasterEnabled"] is True
    assert cluster["DedicatedMasterType"] == "m6g.large.search"
    assert cluster["InstanceType"] == "r6g.large.search"


def test_opensearch_endpoint_output_is_a_bare_host() -> None:
    """The output feeds OPENSEARCH_ENDPOINT, which the adapter uses as a host."""
    output = _storage_template({}).find_outputs("OpenSearchEndpoint")
    (value,) = (o["Value"] for o in output.values())
    # A bare GetAtt of the domain endpoint, not a Fn::Join with "https://".
    assert "Fn::GetAtt" in value, value
    assert value["Fn::GetAtt"][1] == "DomainEndpoint"


def test_cache_expiry_never_applies_bucket_wide() -> None:
    """A corpus uploaded to the cache bucket must not expire: incremental
    indexing would treat it as deleted and remove its artifacts."""
    dev = _storage_template({})
    buckets = dev.find_resources(
        "AWS::S3::Bucket", {"Properties": {"BucketName": Match.any_value()}}
    )
    (cache,) = buckets.values()
    rules = cache["Properties"]["LifecycleConfiguration"]["Rules"]
    expiring = [r for r in rules if "ExpirationInDays" in r]
    assert sorted(r.get("Prefix") for r in expiring) == sorted(CACHE_EXPIRY_PREFIXES)
