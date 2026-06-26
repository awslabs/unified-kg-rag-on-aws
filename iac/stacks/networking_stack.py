# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Networking: VPC (reuse or create), subnets, security groups, VPC endpoints.

Two modes (config.network_mode):
  - "private" (default): isolated subnets with NO NAT gateway. The data plane
    reaches AWS services only through VPC endpoints (Bedrock, S3, DynamoDB,
    ECR, CloudWatch Logs, Step Functions). No internet egress.
  - "public": private subnets WITH NAT egress (simpler; allows arbitrary
    outbound, e.g. pulling public packages at runtime).

An existing VPC is imported when config.vpc_id is set; otherwise a VPC is
created. Endpoints are created in both modes (free for the data plane and
required in private mode).
"""

from __future__ import annotations

from aws_cdk import Stack
from aws_cdk import aws_ec2 as ec2
from constructs import Construct

from iac.config import DeploymentConfig


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
        subnets = self.app_subnets

        # Gateway endpoints (free): S3 + DynamoDB.
        vpc.add_gateway_endpoint(
            "S3Endpoint", service=ec2.GatewayVpcEndpointAwsService.S3
        )
        vpc.add_gateway_endpoint(
            "DynamoDbEndpoint", service=ec2.GatewayVpcEndpointAwsService.DYNAMODB
        )

        # Interface endpoints required for a private (no-NAT) data plane.
        interface_services = {
            "Bedrock": ec2.InterfaceVpcEndpointAwsService.BEDROCK,
            "BedrockRuntime": ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
            "EcrApi": ec2.InterfaceVpcEndpointAwsService.ECR,
            "EcrDocker": ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
            "CwLogs": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
            "CwMonitoring": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_MONITORING,
            "StepFunctions": ec2.InterfaceVpcEndpointAwsService.STEP_FUNCTIONS,
            "Sts": ec2.InterfaceVpcEndpointAwsService.STS,
            "Ssm": ec2.InterfaceVpcEndpointAwsService.SSM,
            "Secrets": ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
        }
        for name, service in interface_services.items():
            vpc.add_interface_endpoint(
                f"{name}Endpoint",
                service=service,
                subnets=subnets,
                security_groups=[self.service_sg],
                private_dns_enabled=True,
            )
