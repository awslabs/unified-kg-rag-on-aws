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


def test_doc_status_table_is_keyed_by_namespace_then_doc_id(prod: Template) -> None:
    # The registry reads one index namespace per run (a Query): the
    # namespace is the partition key (DynamoDBDocStatusStore). The table of
    # the earlier doc_id-only key schema had another name, so the schema
    # change replaces it instead of failing on the custom name.
    (table,) = prod.find_resources("AWS::DynamoDB::Table").values()
    assert table["Properties"]["KeySchema"] == [
        {"AttributeName": "namespace", "KeyType": "HASH"},
        {"AttributeName": "doc_id", "KeyType": "RANGE"},
    ]
    assert table["Properties"]["TableName"].endswith("graphrag-doc-status-v2")
    # An abandoned namespace write lease expires by itself.
    assert table["Properties"]["TimeToLiveSpecification"] == {
        "AttributeName": "lease_expires_at",
        "Enabled": True,
    }


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


@pytest.mark.parametrize(
    ("context", "azs", "zones"),
    [
        ({"env_name": "prod"}, 3, 2),
        ({"opensearch_count": 3, "max_azs": 3}, 3, 3),
        ({"opensearch_count": 6, "max_azs": 3}, 3, 3),
        ({"opensearch_count": 4, "max_azs": 2}, 3, 2),
    ],
    ids=["two-nodes", "three-nodes-three-azs", "six-nodes-three-azs", "two-azs"],
)
def test_opensearch_spans_as_many_zones_as_nodes_and_azs_allow(
    context: dict[str, Any], azs: int, zones: int
) -> None:
    # The AZ lookup the created VPC reads, so the test needs no account.
    lookup = f"availability-zones:account={_ENV.account}:region={_ENV.region}"
    context = {**context, lookup: [f"us-west-2{c}" for c in "abcd"[:azs]]}
    (domain,) = (
        _storage_template(context)
        .find_resources("AWS::OpenSearchService::Domain")
        .values()
    )
    properties = domain["Properties"]
    cluster = properties["ClusterConfig"]
    assert cluster["ZoneAwarenessEnabled"] is True
    assert cluster["ZoneAwarenessConfig"]["AvailabilityZoneCount"] == zones
    assert len(properties["VPCOptions"]["SubnetIds"]) == zones


@pytest.mark.parametrize("count", [3, 5])
def test_odd_node_count_in_two_zones_fails_at_synth(count: int) -> None:
    # OpenSearch rejects it at deploy time; a 3-AZ VPC takes the same count.
    lookup = f"availability-zones:account={_ENV.account}:region={_ENV.region}"
    context = {"opensearch_count": count, lookup: ["us-west-2a", "us-west-2b"]}
    with pytest.raises(ValueError, match="needs an even number of data nodes"):
        _storage_template(context)
    _storage_template(
        {**context, "max_azs": 3, lookup: [f"us-west-2{c}" for c in "abc"]}
    )


def test_neptune_serverless_runs_every_instance_serverless() -> None:
    template = _storage_template(
        {"neptune_serverless_ncu": "1-16", "neptune_instances": "2"}
    )
    template.has_resource_properties(
        "AWS::Neptune::DBCluster",
        {"ServerlessScalingConfiguration": {"MinCapacity": 1, "MaxCapacity": 16}},
    )
    instances = template.find_resources("AWS::Neptune::DBInstance").values()
    assert [i["Properties"]["DBInstanceClass"] for i in instances] == [
        "db.serverless",
        "db.serverless",
    ]


def test_neptune_is_provisioned_by_default() -> None:
    template = _storage_template({})
    (cluster,) = template.find_resources("AWS::Neptune::DBCluster").values()
    assert "ServerlessScalingConfiguration" not in cluster["Properties"]
    (instance,) = template.find_resources("AWS::Neptune::DBInstance").values()
    assert instance["Properties"]["DBInstanceClass"] == "db.r6g.large"
