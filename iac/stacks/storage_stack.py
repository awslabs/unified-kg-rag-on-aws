# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Storage: Neptune cluster, OpenSearch domain, DynamoDB doc-status table, S3 cache.

- Neptune: IAM-auth Gremlin cluster in the app subnets (maps to NeptuneConfig).
- OpenSearch: VPC domain with node-to-node + at-rest encryption + HTTPS
  (maps to OpenSearchConfig; fine-grained access via IAM).
- DynamoDB: the incremental doc-status registry (maps to DynamoDBConfig).
- S3: pipeline cache bucket — reused if config.cache_bucket_name is set, else
  created KMS-encrypted (maps to the pipeline S3 cache sync).
"""

from __future__ import annotations

from aws_cdk import RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_neptune_alpha as neptune
from aws_cdk import aws_opensearchservice as opensearch
from aws_cdk import aws_s3 as s3
from constructs import Construct

from iac.config import DeploymentConfig
from iac.stacks.networking_stack import NetworkingStack


class StorageStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: DeploymentConfig,
        networking: NetworkingStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        self.vpc = networking.vpc
        self.service_sg = networking.service_sg
        self.app_subnets = networking.app_subnets

        self.removal_policy = (
            RemovalPolicy.DESTROY if config.removal_destroy else RemovalPolicy.RETAIN
        )

        self.cache_bucket = self._resolve_cache_bucket()
        self.doc_status_table = self._build_doc_status_table()
        self.neptune_cluster = self._build_neptune()
        self.opensearch_domain = self._build_opensearch()

    # ------------------------------------------------------------- S3 cache
    def _resolve_cache_bucket(self) -> s3.IBucket:
        if self.config.cache_bucket_name:
            return s3.Bucket.from_bucket_name(
                self, "CacheBucket", self.config.cache_bucket_name
            )
        return s3.Bucket(
            self,
            "CacheBucket",
            bucket_name=f"{self.config.prefix}-cache-{self.account}-{self.region}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=False,
            removal_policy=self.removal_policy,
            auto_delete_objects=self.config.removal_destroy,
        )

    # --------------------------------------------------- DynamoDB registry
    def _build_doc_status_table(self) -> dynamodb.Table:
        return dynamodb.Table(
            self,
            "DocStatusTable",
            table_name=self.config.doc_status_table,
            partition_key=dynamodb.Attribute(
                name="doc_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=(
                dynamodb.PointInTimeRecoverySpecification(
                    point_in_time_recovery_enabled=True
                )
            ),
            removal_policy=self.removal_policy,
        )

    # ------------------------------------------------------------- Neptune
    def _build_neptune(self) -> neptune.DatabaseCluster:
        subnet_group = neptune.SubnetGroup(
            self,
            "NeptuneSubnets",
            vpc=self.vpc,
            vpc_subnets=self.app_subnets,
            removal_policy=self.removal_policy,
        )
        return neptune.DatabaseCluster(
            self,
            "Neptune",
            vpc=self.vpc,
            vpc_subnets=self.app_subnets,
            subnet_group=subnet_group,
            instance_type=neptune.InstanceType.of(self.config.neptune_instance),
            instances=1,
            security_groups=[self.service_sg],
            iam_authentication=True,  # matches NeptuneConfig.use_iam = True
            storage_encrypted=True,
            removal_policy=self.removal_policy,
        )

    # ---------------------------------------------------------- OpenSearch
    def _build_opensearch(self) -> opensearch.Domain:
        return opensearch.Domain(
            self,
            "OpenSearch",
            version=opensearch.EngineVersion.OPENSEARCH_2_13,
            vpc=self.vpc,
            vpc_subnets=[self.app_subnets],
            security_groups=[self.service_sg],
            capacity=opensearch.CapacityConfig(
                data_node_instance_type=self.config.opensearch_instance,
                data_nodes=self.config.opensearch_count,
            ),
            zone_awareness=opensearch.ZoneAwarenessConfig(
                enabled=self.config.opensearch_count > 1,
                availability_zone_count=min(self.config.opensearch_count, 2),
            ),
            ebs=opensearch.EbsOptions(
                volume_size=50, volume_type=ec2.EbsDeviceVolumeType.GP3
            ),
            node_to_node_encryption=True,
            encryption_at_rest=opensearch.EncryptionAtRestOptions(enabled=True),
            enforce_https=True,
            removal_policy=self.removal_policy,
        )
