# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NetworkingStack VPC-endpoint assertions (offline synth, no lookups)."""

from __future__ import annotations

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
