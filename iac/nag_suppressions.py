# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Justified cdk-nag (AwsSolutions) suppressions.

Applied only when -c enable_cdk_nag=true. Each suppression documents WHY the
finding is accepted; everything else is fixed in the stacks. Kept centralized so
the rationale is reviewable in one place.
"""

from __future__ import annotations

from typing import Any

from cdk_nag import NagSuppressions
from constructs import IConstruct

from iac.config import DeploymentConfig

# Fixed construct id of the singleton Lambda behind CDK's AwsCustomResource.
_AWS_CUSTOM_RESOURCE_ID = "AWS679f53fac002430cb0da5b7982bd2287"


def apply(stacks: dict[str, Any], config: DeploymentConfig) -> None:
    networking = stacks["networking"]
    storage = stacks["storage"]
    compute = stacks["compute"]
    orchestration = stacks["orchestration"]

    # --- Networking ---
    if not config.vpc_flow_logs:
        NagSuppressions.add_stack_suppressions(
            networking,
            [
                {
                    "id": "AwsSolutions-VPC7",
                    "reason": "Flow logs are config-gated (vpc_flow_logs); enable "
                    "for prod. The private data plane has no internet egress.",
                }
            ],
        )
    # Scope EC23 to the ServiceSg resource specifically (not the whole stack) so
    # a genuine 0.0.0.0/0 ingress added to any OTHER security group in this stack
    # is still flagged. The suppression is a false positive only for this SG's
    # intra-SG self-reference, which cdk-nag can't resolve.
    NagSuppressions.add_resource_suppressions(
        networking.service_sg,
        [
            {
                "id": "AwsSolutions-EC23",
                "reason": "False positive: the SG ingress is intra-SG (same "
                "security group), not 0.0.0.0/0; the rule cannot resolve the "
                "self-reference intrinsic and errors out.",
            }
        ],
    )

    # --- Storage / OpenSearch ---
    storage_suppressions: list[dict[str, Any]] = [
        {
            "id": "AwsSolutions-OS3",
            "reason": "IP allowlisting is not applicable to a VPC-bound "
            "OpenSearch domain — network access is already restricted to the "
            "VPC + service security group, and the access policy requires "
            "IAM-signed (es:ESHttp*) requests.",
        },
        # CDK custom-resource Lambda roles are framework generated; their
        # managed policy is not under our control.
        {
            "id": "AwsSolutions-IAM4",
            "appliesTo": [
                "Policy::arn:<AWS::Partition>:iam::aws:policy/"
                "service-role/AWSLambdaBasicExecutionRole"
            ],
            "reason": "CDK-generated custom-resource Lambda execution role.",
        },
    ]
    if config.opensearch_count <= 1:
        # A multi-node domain gets dedicated masters + zone awareness, so these
        # are only accepted for the single-node (cost-controlled) shape.
        storage_suppressions += [
            {
                "id": "AwsSolutions-OS4",
                "reason": "Single-node domain (opensearch_count=1) omits "
                "dedicated master nodes to control cost; opensearch_count > 1 "
                "adds three dedicated masters.",
            },
            {
                "id": "AwsSolutions-OS7",
                "reason": "Single-node domain (opensearch_count=1) cannot be "
                "zone-aware; opensearch_count > 1 enables zone awareness.",
            },
        ]
    NagSuppressions.add_stack_suppressions(storage, storage_suppressions)

    # IAM5 is suppressed per role and per finding (appliesTo), never stack-wide,
    # so a new wildcard anywhere else still fails the synth.
    _suppress_iam5(
        storage.opensearch_domain,
        ["Resource::*"],
        "CDK-generated log-group resource policy custom resource: "
        "logs:PutResourcePolicy/DeleteResourcePolicy have no resource-level "
        "scoping.",
    )
    # Singleton Lambda behind the OpenSearch access-policy custom resource. With
    # use_cmk=true CDK grants it kms:Describe*/List* on the CMK.
    access_policy_provider = storage.node.try_find_child(_AWS_CUSTOM_RESOURCE_ID)
    if access_policy_provider is not None:
        _suppress_iam5(
            access_policy_provider,
            ["Action::kms:Describe*", "Action::kms:List*"],
            "CDK-generated custom-resource role: KMS read actions on the single "
            "CMK ARN, granted by the framework.",
        )

    # --- Compute ---
    NagSuppressions.add_stack_suppressions(
        compute,
        [
            {
                "id": "AwsSolutions-ECS2",
                "reason": "Container env vars carry only non-secret service "
                "endpoints/region/bucket names; no credentials or secrets are "
                "passed via the environment.",
            },
        ],
    )
    _suppress_iam5(
        compute.task_role,
        [
            # bedrock:Rerank and List/GetInferenceProfile: no resource scoping.
            "Resource::*",
            # Cross-region inference fans one call out to models in several
            # regions, so model/profile/guardrail ARNs need region + id wildcards.
            "Resource::arn:aws:bedrock:*::foundation-model/*",
            "Resource::arn:aws:bedrock:*::inference-profile/*",
            "Resource::arn:aws:bedrock:*:<AWS::AccountId>:inference-profile/*",
            "Resource::arn:aws:bedrock:*:<AWS::AccountId>:guardrail/*",
            # Data-plane paths under this one Neptune cluster / OpenSearch domain.
            {
                "regex": r"/^Resource::arn:aws:neptune-db:.+:<Neptune\w+\.ClusterResourceId>\/\*$/"
            },
            "Action::es:ESHttp*",
            {"regex": r"/^Resource::<OpenSearch\w+\.Arn>\/\*$/"},
            # CDK grant_read_write on the cache bucket (objects of this bucket only).
            "Action::s3:Abort*",
            "Action::s3:DeleteObject*",
            "Action::s3:GetBucket*",
            "Action::s3:GetObject*",
            "Action::s3:List*",
            {
                "regex": r"/^Resource::(<CacheBucket\w+\.Arn>|arn:(aws|<AWS::Partition>):s3:::[^/]+)\/\*$/"
            },
            # CDK KMS grants on the single CMK (use_cmk=true).
            "Action::kms:GenerateDataKey*",
            "Action::kms:ReEncrypt*",
        ],
        "Task role wildcards, each scoped to this deployment's resources: "
        "Bedrock model/profile/guardrail ARNs (cross-region inference spans "
        "regions; Rerank and inference-profile reads have no resource scoping), "
        "the Neptune cluster and OpenSearch domain data paths, and CDK "
        "grant-generated S3/KMS actions on the cache bucket and CMK.",
    )
    execution_role = compute.task_definition.execution_role
    assert execution_role is not None
    _suppress_iam5(
        execution_role,
        ["Resource::*"],
        "ecr:GetAuthorizationToken has no resource-level scoping (generated by "
        "the ECS task definition).",
    )

    # --- Orchestration ---
    _suppress_iam5(
        orchestration.state_machine.role,
        [
            "Resource::*",
            {"regex": r"/^Resource::arn:.*IngestionTask\w+.*:\*$/"},
        ],
        "Generated by the EcsRunTask (RUN_JOB) integration, X-Ray tracing and "
        "log delivery: ecs:DescribeTasks/StopTask, logs:*LogDelivery and xray:* "
        "have no resource-level scoping, and RunTask is scoped to every revision "
        "of this one task definition.",
    )


def _suppress_iam5(
    construct: IConstruct, applies_to: list[str | dict[str, str]], reason: str
) -> None:
    NagSuppressions.add_resource_suppressions(
        construct,
        [{"id": "AwsSolutions-IAM5", "appliesTo": applies_to, "reason": reason}],
        apply_to_children=True,
    )
