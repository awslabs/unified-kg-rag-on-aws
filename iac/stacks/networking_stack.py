# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Networking: VPC (reuse or create), subnets, security groups, VPC endpoints.

Two modes (config.network_mode):
  - "private" (default): isolated subnets with NO NAT gateway. The data plane
    reaches AWS services only through VPC endpoints (Bedrock, S3, DynamoDB,
    ECR, CloudWatch Logs, STS). No internet egress.
  - "public": private subnets WITH NAT egress (simpler; allows arbitrary
    outbound, e.g. pulling public packages at runtime).

An existing VPC is imported when config.vpc_id is set; otherwise a VPC is
created. For a created VPC the free S3/DynamoDB gateway endpoints are added in
both modes; the interface endpoints (billed per AZ-hour) only in private mode,
where they are the data plane's only route to those services. A reused VPC gets
no endpoints: synth warns with the list it must already provide.
"""

from __future__ import annotations

from aws_cdk import Annotations, Stack
from aws_cdk import aws_ec2 as ec2
from constructs import Construct

from iac.config import DeploymentConfig

GATEWAY_ENDPOINTS = {
    "S3": ec2.GatewayVpcEndpointAwsService.S3,
    "DynamoDb": ec2.GatewayVpcEndpointAwsService.DYNAMODB,
}

# Interface endpoints the private (no-NAT) data plane needs: only services the
# app, its entrypoint (`aws s3 sync`, via the S3 gateway) or the Fargate agent
# (image pull, awslogs driver) actually call. Nothing calls SSM, Secrets
# Manager (the task defines no secrets), Step Functions (RUN_JOB tracks the task
# service-side) or CloudWatch monitoring (metrics are EMF records in the logs),
# so those endpoints are not created.
INTERFACE_ENDPOINTS = {
    "Bedrock": ec2.InterfaceVpcEndpointAwsService.BEDROCK,
    "BedrockRuntime": ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
    # The Bedrock Rerank API lives on bedrock-agent-runtime (it was added to the
    # Agents/Knowledge-Bases runtime surface), NOT bedrock-runtime. Reranking is
    # enabled by default, so without this endpoint every query hangs at the TCP
    # level in private (no-NAT) mode — there is no route to the public endpoint.
    # Required even when no Agents feature is used.
    "BedrockAgentRuntime": ec2.InterfaceVpcEndpointAwsService.BEDROCK_AGENT_RUNTIME,
    "EcrApi": ec2.InterfaceVpcEndpointAwsService.ECR,
    "EcrDocker": ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
    "CwLogs": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
    # boto3 calls STS only for assumed-role credentials (adapters/aws/bedrock.py);
    # kept so such a configuration does not hang in private mode.
    "Sts": ec2.InterfaceVpcEndpointAwsService.STS,
}


class NetworkingStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, config: DeploymentConfig, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config

        self.vpc = self._resolve_vpc()
        self.service_sg = self._build_service_security_group()
        if config.create_vpc:
            self._add_vpc_endpoints()
            if config.vpc_flow_logs:
                self._enable_flow_logs()
        else:
            self._warn_reused_vpc_requirements()

    # ------------------------------------------------------------------ VPC
    def _resolve_vpc(self) -> ec2.IVpc:
        if self.config.vpc_id:
            # Reuse an existing VPC (looked up at synth time).
            return ec2.Vpc.from_lookup(self, "Vpc", vpc_id=self.config.vpc_id)

        subnet_type = (
            ec2.SubnetType.PRIVATE_ISOLATED
            if self.config.is_private
            else ec2.SubnetType.PRIVATE_WITH_EGRESS
        )
        return ec2.Vpc(
            self,
            "Vpc",
            max_azs=self.config.max_azs,
            # No NAT gateways in private mode (cost + no-egress guarantee).
            nat_gateways=0 if self.config.is_private else self.config.max_azs,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="app", subnet_type=subnet_type, cidr_mask=22
                ),
            ],
        )

    @property
    def app_subnets(self) -> ec2.SubnetSelection:
        """The subnets the data plane (Neptune/OpenSearch/Fargate) runs in."""
        if self.config.is_private:
            return ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)
        return ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)

    # ------------------------------------------------------ security groups
    def _build_service_security_group(self) -> ec2.SecurityGroup:
        sg = ec2.SecurityGroup(
            self,
            "ServiceSg",
            vpc=self.vpc,
            description="unified-kg-rag-on-aws data plane: Fargate to Neptune/OpenSearch",
            allow_all_outbound=True,
        )
        # Same-SG ingress on Neptune (8182) and OpenSearch (443) so the Fargate
        # tasks (in this SG) can reach the stores (also in this SG).
        sg.add_ingress_rule(sg, ec2.Port.tcp(8182), "Neptune Gremlin (intra-SG)")
        sg.add_ingress_rule(sg, ec2.Port.tcp(443), "OpenSearch HTTPS (intra-SG)")
        return sg

    # --------------------------------------------------------- flow logs
    def _enable_flow_logs(self) -> None:
        self.vpc.add_flow_log(
            "FlowLogs",
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(),
            traffic_type=ec2.FlowLogTrafficType.ALL,
        )

    # ------------------------------------------------------- VPC endpoints
    def _add_vpc_endpoints(self) -> None:
        vpc = self.vpc
        # Gateway endpoints are free; they also keep S3/DynamoDB off the NAT.
        for name, gateway in GATEWAY_ENDPOINTS.items():
            vpc.add_gateway_endpoint(f"{name}Endpoint", service=gateway)
        if not self.config.is_private:
            # Public mode reaches every other service through the NAT gateways.
            return
        for name, service in INTERFACE_ENDPOINTS.items():
            vpc.add_interface_endpoint(
                f"{name}Endpoint",
                service=service,
                subnets=self.app_subnets,
                security_groups=[self.service_sg],
                private_dns_enabled=True,
            )

    def _warn_reused_vpc_requirements(self) -> None:
        # A reused VPC is imported as-is: nothing here adds endpoints or subnets,
        # and a missing endpoint only shows up at runtime as a hung call.
        gateways = "s3, dynamodb"
        if self.config.is_private:
            interfaces = ", ".join(s.short_name for s in INTERFACE_ENDPOINTS.values())
            message = (
                f"Reusing VPC '{self.config.vpc_id}' in private mode: this stack "
                "creates NO VPC endpoints. The VPC needs PRIVATE_ISOLATED subnets "
                "(no NAT or internet gateway route) for the data plane, gateway "
                f"endpoints ({gateways}) on their route tables, and interface "
                "endpoints with private DNS enabled, reachable from those subnets "
                f"on port 443: {interfaces}."
            )
        else:
            message = (
                f"Reusing VPC '{self.config.vpc_id}' in public mode: this stack "
                "creates NO VPC endpoints. The VPC needs PRIVATE_WITH_EGRESS "
                "subnets (a NAT route) for the data plane; gateway endpoints "
                f"({gateways}) are optional and keep that traffic off the NAT."
            )
        Annotations.of(self).add_warning(message)
