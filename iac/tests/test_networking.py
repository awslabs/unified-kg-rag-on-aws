# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NetworkingStack VPC-endpoint assertions (offline synth, no lookups)."""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
from aws_cdk.assertions import Template

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
