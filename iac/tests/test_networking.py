# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NetworkingStack VPC-endpoint assertions (offline synth, no lookups)."""

from __future__ import annotations

import json
from typing import Any

import aws_cdk as cdk
from aws_cdk.assertions import Annotations, Match, Template

from iac.config import DeploymentConfig
from iac.stacks.networking_stack import INTERFACE_ENDPOINTS, NetworkingStack

_ENV = cdk.Environment(account="111111111111", region="us-west-2")


def _network(context: dict[str, Any]) -> NetworkingStack:
    app = cdk.App(context=context)
    config = DeploymentConfig.from_context(app)
    return NetworkingStack(app, "Net", config=config, env=_ENV)


def _endpoint_counts(stack: NetworkingStack) -> dict[str, int]:
    endpoints = Template.from_stack(stack).find_resources("AWS::EC2::VPCEndpoint")
    counts = {"Gateway": 0, "Interface": 0}
    for resource in endpoints.values():
        counts[resource["Properties"].get("VpcEndpointType", "Gateway")] += 1
    return counts


def test_private_mode_creates_only_needed_interface_endpoints() -> None:
    counts = _endpoint_counts(_network({}))
    assert counts == {"Gateway": 2, "Interface": len(INTERFACE_ENDPOINTS)}
    short_names = {s.short_name for s in INTERFACE_ENDPOINTS.values()}
    # No data-plane caller for these; each costs per AZ-hour.
    assert not short_names & {"ssm", "secretsmanager", "states", "monitoring"}
    assert {"bedrock-runtime", "bedrock-agent-runtime", "ecr.dkr", "logs"} <= (
        short_names
    )


def test_public_mode_creates_only_free_gateway_endpoints() -> None:
    assert _endpoint_counts(_network({"network_mode": "public"})) == {
        "Gateway": 2,
        "Interface": 0,
    }


def test_flow_log_retention_is_configurable() -> None:
    stack = _network({"vpc_flow_logs": "true", "flow_log_retention_days": "90"})
    template = Template.from_stack(stack)
    template.resource_count_is("AWS::EC2::FlowLog", 1)
    template.has_resource_properties("AWS::Logs::LogGroup", {"RetentionInDays": 90})


def _reuse_warnings(stack: NetworkingStack) -> list[str]:
    found = Annotations.from_stack(stack).find_warning("*", Match.any_value())
    return [str(w.entry.data) for w in found if "Reusing VPC" in str(w.entry.data)]


def test_reused_vpc_warns_with_required_endpoints() -> None:
    stack = _network({"vpc_id": "vpc-0example"})
    assert _endpoint_counts(stack) == {"Gateway": 0, "Interface": 0}
    (warning,) = _reuse_warnings(stack)
    assert "PRIVATE_ISOLATED" in warning
    for service in INTERFACE_ENDPOINTS.values():
        assert service.short_name in warning


def test_reused_vpc_public_mode_warns_about_egress_subnets() -> None:
    stack = _network({"vpc_id": "vpc-0example", "network_mode": "public"})
    (warning,) = _reuse_warnings(stack)
    assert "PRIVATE_WITH_EGRESS" in warning


def test_created_vpc_does_not_warn() -> None:
    assert _reuse_warnings(_network({})) == []


def _ingress_rules(template: Template) -> list[dict[str, Any]]:
    rules = [
        r["Properties"]
        for r in template.find_resources("AWS::EC2::SecurityGroupIngress").values()
    ]
    for sg in template.find_resources("AWS::EC2::SecurityGroup").values():
        rules += sg["Properties"].get("SecurityGroupIngress", [])
    return rules


def test_interface_endpoints_use_their_own_security_group() -> None:
    stack = _network({})
    template = Template.from_stack(stack)
    service_sg = stack.resolve(stack.service_sg.security_group_id)
    assert stack.endpoint_sg is not None
    endpoint_sg = stack.resolve(stack.endpoint_sg.security_group_id)
    for resource in template.find_resources(
        "AWS::EC2::VPCEndpoint", {"Properties": {"VpcEndpointType": "Interface"}}
    ).values():
        assert resource["Properties"]["SecurityGroupIds"] == [endpoint_sg]
    rules = _ingress_rules(template)
    # No CIDR ingress anywhere: OpenSearch (ServiceSg) and the endpoints are
    # reachable from the data-plane security group only.
    assert not [r for r in rules if "CidrIp" in r]
    (endpoint_rule,) = [r for r in rules if r.get("GroupId") == endpoint_sg]
    assert endpoint_rule["SourceSecurityGroupId"] == service_sg
    assert endpoint_rule["FromPort"] == endpoint_rule["ToPort"] == 443


def test_public_mode_creates_no_endpoint_security_group() -> None:
    stack = _network({"network_mode": "public"})
    assert stack.endpoint_sg is None
    Template.from_stack(stack).resource_count_is("AWS::EC2::SecurityGroup", 1)


def _gateway_policies(stack: NetworkingStack) -> dict[str, str | None]:
    """Gateway endpoint policy per service, as resolved JSON text."""
    endpoints = Template.from_stack(stack).find_resources(
        "AWS::EC2::VPCEndpoint", {"Properties": {"VpcEndpointType": "Gateway"}}
    )
    policies: dict[str, str | None] = {}
    for resource in endpoints.values():
        props = resource["Properties"]
        key = "S3" if ".s3" in json.dumps(props["ServiceName"]) else "DynamoDb"
        policy = props.get("PolicyDocument")
        policies[key] = None if policy is None else json.dumps(policy)
    return policies


def test_private_mode_scopes_gateway_endpoints_to_deployment_resources() -> None:
    policies = _gateway_policies(_network({}))
    s3, ddb = policies["S3"], policies["DynamoDb"]
    assert s3 is not None and ddb is not None
    s3_doc = json.loads(s3)
    assert {s["Sid"] for s in s3_doc["Statement"]} == {
        "DeploymentBuckets",
        "EcrImageLayers",
    }
    assert ":s3:::graphrag-cache-111111111111-us-west-2/*" in s3
    # Without the ECR layer bucket, Fargate cannot pull the image.
    assert ":s3:::prod-us-west-2-starport-layer-bucket/*" in s3
    assert ":dynamodb:us-west-2:111111111111:table/graphrag-doc-status" in ddb
    # Only the deployment's own resources: no wildcard bucket or table.
    for doc in (s3_doc, json.loads(ddb)):
        for statement in doc["Statement"]:
            assert statement["Resource"] != "*"


def test_gateway_endpoint_policy_follows_reused_cache_and_corpus_buckets() -> None:
    s3 = _gateway_policies(
        _network({"cache_bucket_name": "my-cache", "corpus_bucket_name": "my-corpus"})
    )["S3"]
    assert s3 is not None
    for arn in (":s3:::my-cache", ":s3:::my-cache/*", ":s3:::my-corpus/*"):
        assert arn in s3
    assert "graphrag-cache-" not in s3


def test_public_mode_keeps_default_gateway_policies() -> None:
    assert _gateway_policies(_network({"network_mode": "public"})) == {
        "S3": None,
        "DynamoDb": None,
    }
