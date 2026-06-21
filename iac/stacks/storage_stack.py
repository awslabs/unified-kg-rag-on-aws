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

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
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
        kms_key: kms.IKey | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        self.vpc = networking.vpc
        self.service_sg = networking.service_sg
        self.app_subnets = networking.app_subnets
        self.kms_key = kms_key  # shared CMK when config.use_cmk, else None

        self.removal_policy = (
            RemovalPolicy.DESTROY if config.removal_destroy else RemovalPolicy.RETAIN
        )

        self.cache_bucket = self._resolve_cache_bucket()
        self.doc_status_table = self._build_doc_status_table()
        self.neptune_cluster = self._build_neptune()
        self.opensearch_domain = self._build_opensearch()
        self._export_outputs()

    # ------------------------------------------------------------ outputs
    def _export_outputs(self) -> None:
        CfnOutput(
            self,
            "NeptuneEndpoint",
            value=self.neptune_cluster.cluster_endpoint.hostname,
            description="Set as NEPTUNE_ENDPOINT for the app",
        )
        CfnOutput(
            self,
            "OpenSearchEndpoint",
            value=f"https://{self.opensearch_domain.domain_endpoint}",
            description="Set as OPENSEARCH_ENDPOINT for the app",
        )
        CfnOutput(self, "CacheBucketName", value=self.cache_bucket.bucket_name)
        CfnOutput(self, "DocStatusTableName", value=self.doc_status_table.table_name)

    # ------------------------------------------------------------- S3 cache
    def _resolve_cache_bucket(self) -> s3.IBucket:
        if self.config.cache_bucket_name:
            return s3.Bucket.from_bucket_name(
                self, "CacheBucket", self.config.cache_bucket_name
            )
        access_logs = s3.Bucket(
            self,
            "CacheAccessLogs",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=self.removal_policy,
            auto_delete_objects=self.config.removal_destroy,
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(90))],
        )
        return s3.Bucket(
            self,
            "CacheBucket",
            bucket_name=f"{self.config.prefix}-cache-{self.account}-{self.region}",
            encryption=(
                s3.BucketEncryption.KMS
                if self.kms_key
                else s3.BucketEncryption.S3_MANAGED
            ),
            encryption_key=self.kms_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            server_access_logs_bucket=access_logs,
            server_access_logs_prefix="cache-access/",
            versioned=False,
            # Cost/sustainability: expire stale pipeline cache + clean up
            # incomplete multipart uploads.
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-cache",
                    expiration=Duration.days(30),
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                )
            ],
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
            encryption=(
                dynamodb.TableEncryption.CUSTOMER_MANAGED
                if self.kms_key
                else dynamodb.TableEncryption.AWS_MANAGED
            ),
            encryption_key=self.kms_key,
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
            # >=2 instances => a reader in another AZ for HA failover.
            instances=max(1, self.config.neptune_instances),
            security_groups=[self.service_sg],
            iam_authentication=True,  # matches NeptuneConfig.use_iam = True
            storage_encrypted=True,
            kms_key=self.kms_key,
            backup_retention=Duration.days(self.config.backup_retention_days),
            deletion_protection=self.config.deletion_protection,
            auto_minor_version_upgrade=True,
            removal_policy=self.removal_policy,
        )

    # ---------------------------------------------------------- OpenSearch
    def _build_opensearch(self) -> opensearch.Domain:
        multi_node = self.config.opensearch_count > 1
        # Zone awareness requires availabilityZoneCount of 2 or 3 and is only
        # valid for multi-node domains; omit it entirely for a single node.
        zone_awareness = (
            opensearch.ZoneAwarenessConfig(
                enabled=True,
                availability_zone_count=min(self.config.opensearch_count, 2),
            )
            if multi_node
            else None
        )
        # OpenSearch needs exactly as many subnets as AZs it spans: 1 for a
        # single node, 2 for a zone-aware multi-node domain.
        selected = self.vpc.select_subnets(
            subnet_type=self.app_subnets.subnet_type
        ).subnets
        os_subnets = ec2.SubnetSelection(subnets=selected[: (2 if multi_node else 1)])
        return opensearch.Domain(
            self,
            "OpenSearch",
            version=opensearch.EngineVersion.OPENSEARCH_2_13,
            vpc=self.vpc,
            vpc_subnets=[os_subnets],
            security_groups=[self.service_sg],
            capacity=opensearch.CapacityConfig(
                data_node_instance_type=self.config.opensearch_instance,
                data_nodes=self.config.opensearch_count,
                # Dedicated master nodes stabilize the cluster under load; enable
                # for HA (multi-node) deployments only.
                master_nodes=3 if multi_node else 0,
                master_node_instance_type=(
                    self.config.opensearch_instance if multi_node else None
                ),
            ),
            zone_awareness=zone_awareness,
            logging=opensearch.LoggingOptions(
                slow_search_log_enabled=True,
                slow_index_log_enabled=True,
                app_log_enabled=True,
            ),
            ebs=opensearch.EbsOptions(
                volume_size=50, volume_type=ec2.EbsDeviceVolumeType.GP3
            ),
            node_to_node_encryption=True,
            encryption_at_rest=opensearch.EncryptionAtRestOptions(
                enabled=True, kms_key=self.kms_key
            ),
            enforce_https=True,
            tls_security_policy=opensearch.TLSSecurityPolicy.TLS_1_2,
            # Network access is already restricted to the VPC + service SG;
            # require IAM-signed requests so callers also need es:ESHttp* (the
            # Fargate task role has it). VPC domains can't be public anyway.
            access_policies=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[iam.AccountRootPrincipal()],
                    actions=["es:ESHttp*"],
                    resources=["*"],
                )
            ],
            removal_policy=self.removal_policy,
        )
