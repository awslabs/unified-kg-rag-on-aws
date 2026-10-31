# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom resource: the AWS-provided Nori plugin package on the domain.

CloudFormation has no resource for OpenSearch package associations, so
``NoriPackage`` (opensearch_nori_package) associates the ``analysis-nori``
package of the domain's engine version through a provider-framework custom
resource whose handler (``iac/custom_resources/nori_package.py``) polls
until the association is active, with roles that reach only this domain and
this stack.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CustomResource, Duration, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_opensearchservice as opensearch
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import custom_resources as cr
from constructs import Construct

_HANDLERS = Path(__file__).resolve().parent.parent / "custom_resources"
NORI_PACKAGE_NAME = "analysis-nori"


class _PollingProvider(Construct):
    """A provider whose ``on_event`` and ``is_complete`` handlers are the
    functions of one handler module, logging to one log group."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        module: str,
        query_interval: Duration,
        total_timeout: Duration,
        kms_key: kms.IKey | None,
        removal_policy: RemovalPolicy,
    ) -> None:
        super().__init__(scope, construct_id)
        self.log_group = logs.LogGroup(
            self,
            "Logs",
            encryption_key=kms_key,
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=removal_policy,
        )
        code = lambda_.Code.from_asset(str(_HANDLERS), exclude=["__pycache__"])

        self.on_event = lambda_.Function(
            self,
            "OnEvent",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler=f"{module}.on_event",
            code=code,
            timeout=Duration.minutes(1),
            log_group=self.log_group,
        )
        self.is_complete = lambda_.Function(
            self,
            "IsComplete",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler=f"{module}.is_complete",
            code=code,
            timeout=Duration.minutes(1),
            log_group=self.log_group,
        )
        self.provider = cr.Provider(
            self,
            "Provider",
            on_event_handler=self.on_event,
            is_complete_handler=self.is_complete,
            query_interval=query_interval,
            total_timeout=total_timeout,
            log_group=self.log_group,
            disable_waiter_state_machine_logging=False,
            waiter_state_machine_log_options=cr.LogOptions(
                destination=self.log_group, level=sfn.LogLevel.ALL
            ),
        )

    def allow_stack_status_read(self) -> None:
        """A Delete reads the stack status (reverts only in an update's cleanup)."""
        self.is_complete.add_to_role_policy(
            iam.PolicyStatement(
                actions=["cloudformation:DescribeStacks"],
                resources=[Stack.of(self).stack_id],
            )
        )


class NoriPackage(Construct):
    """The ``analysis-nori`` package of ``engine_version``, associated with
    ``domain``. Depends on the whole domain construct, so an engine version
    upgrade (EnableVersionUpgrade) finishes before the package of the new
    version is resolved and associated."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        domain: opensearch.Domain,
        engine_version: str,
        kms_key: kms.IKey | None,
        removal_policy: RemovalPolicy,
    ) -> None:
        super().__init__(scope, construct_id)
        self.handlers = _PollingProvider(
            self,
            "Handlers",
            module="nori_package",
            # A blue/green deployment of the domain: minutes to hours. The
            # provider framework allows at most two hours.
            query_interval=Duration.minutes(1),
            total_timeout=Duration.hours(2),
            kms_key=kms_key,
            removal_policy=removal_policy,
        )
        # DescribePackages has no resource-level scoping.
        self.handlers.on_event.add_to_role_policy(
            iam.PolicyStatement(actions=["es:DescribePackages"], resources=["*"])
        )
        self.handlers.is_complete.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "es:AssociatePackage",
                    "es:DescribeDomain",
                    "es:DissociatePackage",
                    "es:ListPackagesForDomain",
                ],
                resources=[domain.domain_arn],
            )
        )
        self.handlers.allow_stack_status_read()
        self.resource = CustomResource(
            self,
            "Resource",
            service_token=self.handlers.provider.service_token,
            resource_type="Custom::OpenSearchNoriPackage",
            properties={
                "DomainName": domain.domain_name,
                "EngineVersion": engine_version,
                "PackageName": NORI_PACKAGE_NAME,
            },
        )
        self.resource.node.add_dependency(domain)
