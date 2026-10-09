# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage: Neptune cluster, OpenSearch domain, DynamoDB doc-status table, S3 cache.

- Neptune: IAM-auth Gremlin cluster in the app subnets (maps to NeptuneConfig).
- OpenSearch: VPC domain with node-to-node + at-rest encryption + HTTPS
  (maps to OpenSearchConfig; fine-grained access via IAM).
- DynamoDB: the incremental doc-status registry (maps to DynamoDBConfig).
- S3: pipeline cache bucket — reused if config.cache_bucket_name is set, else
  created with SSE-S3, or the shared CMK when config.use_cmk (maps to the
  pipeline S3 cache sync).
"""

from __future__ import annotations

from aws_cdk import Annotations, CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_neptune_alpha as neptune
from aws_cdk import aws_opensearchservice as opensearch
from aws_cdk import aws_s3 as s3
from constructs import Construct

from iac.config import DeploymentConfig
from iac.stacks.networking_stack import NetworkingStack

# Cache-bucket prefixes the app writes and that are safe to expire: the stage
# checkpoints of `run-ingestion --s3-sync` (PipelineConfig.s3_prefix /
# --s3-prefix, default "pipeline-runs") and the optional persisted embedding
# cache (embedding_cache_s3_key, default "embedding-cache/cache.json"). Anything
# else in the bucket, such as an uploaded corpus, is never expired.
CACHE_EXPIRY_PREFIXES = ("pipeline-runs/", "embedding-cache/")


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
        # Both stores are VPC-only: these hosts resolve and accept connections
        # only from inside the VPC (service security group), e.g. the Fargate
        # task, which already receives them as env vars.
        CfnOutput(
            self,
            "NeptuneEndpoint",
            value=self.neptune_cluster.cluster_endpoint.hostname,
            description="Set as NEPTUNE_ENDPOINT for the app (VPC-only host)",
        )
        CfnOutput(
            self,
            "OpenSearchEndpoint",
            # Bare host, like the task env var: the adapter builds the URL from
            # aws.opensearch.port/use_ssl, so a scheme here breaks the client.
            value=self.opensearch_domain.domain_endpoint,
            description="Set as OPENSEARCH_ENDPOINT for the app (bare VPC-only "
            "host, no https://)",
        )
        CfnOutput(self, "CacheBucketName", value=self.cache_bucket.bucket_name)
        CfnOutput(self, "DocStatusTableName", value=self.doc_status_table.table_name)

    # ------------------------------------------------------------- S3 cache
    def _resolve_cache_bucket(self) -> s3.IBucket:
        if self.config.cache_bucket_name:
            # A name-only import: CDK cannot attach encryption, a TLS-only bucket
            # policy, or block-public-access to it, and none of the guarantees the
            # created-bucket branch below enforces are applied. Warn at synth so
            # the operator knows the reused bucket must provide these itself.
            Annotations.of(self).add_warning(
                "Reusing an external cache bucket "
                f"('{self.config.cache_bucket_name}'): CDK cannot enforce "
                "encryption, enforce-SSL, or block-public-access on it. Ensure "
                "the bucket has SSE and a TLS-only bucket policy before use."
            )
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
            bucket_name=self.config.cache_bucket(self.account, self.region),
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
            # incomplete multipart uploads. Expiry is scoped to the prefixes the
            # app writes (CACHE_EXPIRY_PREFIXES) and never applies bucket-wide:
            # the corpus may live in this bucket too, and with incremental
            # indexing an expired source file looks deleted, so its graph and
            # vector artifacts would be removed on the next run.
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="abort-incomplete-uploads",
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                ),
                *(
                    s3.LifecycleRule(
                        id=f"expire-{prefix.rstrip('/')}",
                        prefix=prefix,
                        expiration=Duration.days(30),
                    )
                    for prefix in CACHE_EXPIRY_PREFIXES
                ),
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
            # Parity with Neptune: protect the critical incremental-indexing
            # lineage state from a direct DeleteTable API/console call in
            # non-dev. (removal_policy=RETAIN only stops CloudFormation deletes.)
            deletion_protection=self.config.deletion_protection,
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
        domain = opensearch.Domain(
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
                # Masters only manage cluster state, so they default to a
                # smaller type than the data nodes (opensearch_master_instance).
                master_node_instance_type=(
                    self.config.opensearch_master_instance if multi_node else None
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
            removal_policy=self.removal_policy,
        )
        # The domain creates these log groups itself with CDK's default
        # RETAIN; give them the stack's policy so a removal_destroy teardown
        # deletes them like the other log groups.
        for log_group in (
            domain.app_log_group,
            domain.slow_index_log_group,
            domain.slow_search_log_group,
        ):
            assert log_group is not None  # enabled in `logging` above
            log_group.apply_removal_policy(self.removal_policy)
            if self.kms_key is not None:
                # The domain construct offers no key option for the log groups
                # it creates, so set it on the CloudFormation resource.
                cfn_log_group = log_group.node.default_child
                assert isinstance(cfn_log_group, logs.CfnLogGroup)
                cfn_log_group.kms_key_id = self.kms_key.key_arn
        # Resource-scoped access policy (added post-construction so it can
        # reference the domain's own ARN). Network access is already restricted
        # to the VPC + service SG; this requires IAM-signed requests AND scopes
        # the resource to THIS domain's indices (not "*"). Principal stays
        # AccountRootPrincipal — scoping it to the Fargate task role would create
        # a cross-stack circular dependency (the role lives in the compute stack
        # which depends on this one); the IAM caller still needs es:ESHttp* on
        # the role, which is granted there. To keep this from being an
        # account-wide grant at the IAM layer (any account principal holding
        # es:ESHttp* would otherwise be authorized), the statement is additionally
        # conditioned on the request originating from THIS VPC — so only
        # in-VPC, IAM-signed, resource-scoped callers are allowed.
        domain.add_access_policies(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                principals=[iam.AccountRootPrincipal()],
                actions=["es:ESHttp*"],
                resources=[f"{domain.domain_arn}/*"],
                conditions={"StringEquals": {"aws:SourceVpc": self.vpc.vpc_id}},
            )
        )
        return domain
