# unified-kg-rag-on-aws — Infrastructure (AWS CDK, Python)

Modular CDK app that provisions the AWS-native stack the library targets:
Bedrock + Neptune + OpenSearch + DynamoDB + S3, an ECS Fargate data plane, and a
Step Functions ingestion pipeline, with CloudWatch observability and an optional
Bedrock Guardrail.

## Stacks

Stack ids are PascalCase with a `GraphRag` prefix and no env segment;
environments are separated by account/region and tracked via the `env` tag.

| Stack | Resources |
|---|---|
| `GraphRagNetwork` | VPC (reuse or create), subnets, security group, VPC endpoints (Bedrock/S3/DDB/ECR/CW/SFN/…) |
| `GraphRagStorage` | Neptune cluster (IAM auth), OpenSearch domain (VPC, encrypted), DynamoDB doc-status table, S3 cache bucket |
| `GraphRagCompute` | ECR repo, ECS cluster, Fargate task definition + least-privilege task role |
| `GraphRagOrchestration` | Step Functions state machine — 4 resumable phases on Fargate + retries + SNS alarms |
| `GraphRagObservability` | CloudWatch dashboard + alarms: pipeline-failure, silent indexing-failure (EMF), and store health (OpenSearch cluster-red / free-storage / JVM pressure, DynamoDB write throttling) → SNS. Synth warns if `alarm_email` is unset (alarms would have no subscriber) |
| `GraphRagSecurity` | Shared customer-managed KMS key (optional, `use_cmk`) |
| `GraphRagGuardrail` | Bedrock Guardrail, **pinned to `bedrock_region`** (reuse or create a baseline PII/prompt-attack guardrail) |

### Resource naming

Physical resources share a single lowercase `graphrag-<purpose>` scheme (no env
segment — environments are separated by account/region), matching the account's
convention (`example-alb-logs-…`, `example-state-…`):

| Resource | Name |
|---|---|
| S3 cache bucket | `graphrag-cache-<account>-<region>` |
| DynamoDB doc-status | `graphrag-doc-status` |
| ECR repository | `graphrag-app` |
| ECS cluster | `graphrag-cluster` |
| Step Functions | `graphrag-ingestion` |
| SNS alarm topic | `graphrag-pipeline-alarms` |
| CloudWatch dashboard | `graphrag-dashboard` |
| KMS alias | `alias/graphrag-data` |
| Bedrock guardrail | `graphrag-guardrail-<region>` |
| Log groups | `/graphrag/tasks`, `/graphrag/pipeline` |

Neptune/OpenSearch/ECS task-def names are CloudFormation-generated (stack-id
derived) to avoid replace-on-rename conflicts.

> **Guardrail is region-pinned.** A guardrail must live in the Bedrock *runtime*
> region (`bedrock_region`), which can differ from the deploy region that hosts
> Neptune/OpenSearch. It is therefore its own stack. Because it lives in another
> region, the two-step flow avoids cross-region CloudFormation references
> (which churn every deploy-region stack's exports):
>
> ```bash
> # 1) create the guardrail in bedrock_region, note its id from the output
> cdk deploy GraphRagGuardrail -c bedrock_region=us-west-2 …
> # 2) pass that id so compute injects BEDROCK_GUARDRAIL_IDENTIFIER
> cdk deploy --all -c guardrail_identifier=<id> -c bedrock_region=us-west-2 …
> ```
>
> Always pass `-c key=value` flags as **individual arguments** — collapsing them
> into one shell variable corrupts context parsing (vpc_id is silently dropped →
> `Vpc.from_lookup` falls back to a dummy VPC).

The orchestration runs the ingestion CLI as four phases sharing one
`--pipeline-id` (the app's S3 stage checkpoints hand off between phases):

```
Prep (parse/load/chunk/translate) → GraphBuild (extract/glean/resolve/claims)
  → Analysis (graph_analysis/community_detection) → Index (indexing)
```

## Configuration (cdk.json context / `-c key=value`)

| Key | Default | Meaning |
|---|---|---|
| `env_name` | `dev` | stack/resource name prefix. `dev` keeps bare `GraphRag*`/`graphrag-*` names; a non-dev env (e.g. `prod`) scopes them (`GraphRagProd*`, `prod-graphrag-*`) so environments don't collide in one account/region |
| `network_mode` | `private` | `private` = isolated subnets + VPC endpoints, **no NAT** (no internet egress); `public` = private subnets with NAT egress |
| `vpc_id` | _(none)_ | **reuse** an existing VPC instead of creating one |
| `max_azs` | `2` | AZs for a newly-created VPC |
| `cache_bucket_name` | _(none)_ | **reuse** an existing S3 cache bucket instead of creating one |
| `neptune_instance` | `db.r6g.large` | Neptune instance class (Graviton) |
| `neptune_instances` | `1` (dev) / `2` (non-dev) | Neptune instances; `>=2` ⇒ Multi-AZ HA (reader in another AZ). dev defaults to 1 (no failover) for cost |
| `opensearch_instance` | `r6g.large.search` | OpenSearch data node type (Graviton) |
| `opensearch_count` | `1` (dev) / `2` (non-dev) | OpenSearch data node count (`>1` ⇒ 3 dedicated masters + zone awareness = 5 nodes; dev runs a single node for cost) |
| `backup_retention_days` | `7` | Neptune automated backup retention |
| `fargate_cpu` | `2048` | Fargate task vCPU units (in-task ProcessPool extractors scale with vCPU) |
| `fargate_memory` | `8192` | Fargate task memory (MiB) |
| `guardrail_identifier` | _(none)_ | **reuse** an existing Bedrock guardrail (else a baseline PII/prompt-attack guardrail is created and its id injected as `BEDROCK_GUARDRAIL_IDENTIFIER`) |
| `use_cmk` | `false` | customer-managed KMS key for at-rest encryption (S3/Neptune/OpenSearch/SNS/DDB) |
| `vpc_flow_logs` | `false` | enable VPC flow logs (created VPC only) |
| `deletion_protection` | `false` | protect Neptune/OpenSearch from deletion |
| `bedrock_model_arns` | _(none)_ | scope Bedrock IAM to specific model ARNs (list) |
| `alarm_email` | _(none)_ | subscribe an email to the pipeline alarm topic |
| `enable_cdk_nag` | `false` | run cdk-nag AwsSolutions (Well-Architected) checks at synth |
| `owner` | `aws-proserve` | `owner` tag applied to every resource |
| `cost_center` | `unified-kg-rag-on-aws` | `cost-center` tag applied to every resource |
| `removal_destroy` | `true` | `DESTROY` (dev) vs `RETAIN` (prod) on stack deletion |

> Every resource is tagged `project=unified-kg-rag-on-aws`, `env=<env_name>`,
> `managed-by=cdk`, `owner`, and `cost-center` for cost allocation and ownership.

### Well-Architected

The stacks apply WAF defaults out of the box — least-privilege IAM (Bedrock
scoped to model/inference-profile ARNs, `neptune-db:connect`, domain-scoped
`es:ESHttp*`), encryption in transit + at rest, Graviton instances, S3 cache
lifecycle + access logs, Step Functions X-Ray tracing + execution logging,
CloudWatch dashboard + failure alarm, and an SSL-only alarm topic. Production
hardening is opt-in via the flags above. Validate with:

```bash
cdk synth -c enable_cdk_nag=true                       # dev
cdk synth -c enable_cdk_nag=true -c use_cmk=true \
  -c vpc_flow_logs=true -c neptune_instances=2 \
  -c deletion_protection=true -c removal_destroy=false  # prod-hardened
```
Both report zero AwsSolutions findings (accepted findings are documented in
`iac/nag_suppressions.py`).

## Usage

```bash
cd iac
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Synthesize (no AWS changes)
cdk synth

# Reuse existing VPC + S3, fully private data plane (recommended for prod):
cdk synth -c vpc_id=vpc-0abc... -c cache_bucket_name=my-cache -c network_mode=private

# Public egress (simpler dev):
cdk synth -c network_mode=public

# Deploy (creates resources — incurs cost; see project "ask first" rule)
cdk bootstrap            # once per account/region
cdk deploy --all
```

> **Cost / approval:** deploying creates Neptune + OpenSearch (hourly billed) and
> NAT gateways in `public` mode. `removal_destroy=true` (dev default) tears
> everything down on `cdk destroy --all`; set `removal_destroy=false` for prod.

## After deploy

1. Build & push the app image to the created ECR repo (tag `latest`); the image
   must contain a `/app/config.yaml` with the deployed endpoints (or rely on the
   injected `NEPTUNE_ENDPOINT` / `OPENSEARCH_ENDPOINT` / `S3_BUCKET_NAME` /
   `BEDROCK_REGION` env vars the app reads).
2. Start an ingestion run:
   ```bash
   aws stepfunctions start-execution \
     --state-machine-arn <…-ingestion arn> \
     --input '{"source_directory":"/data/docs","pipeline_id":"run-001","config_path":"/app/config.yaml"}'
   ```
   (`source_directory` / `pipeline_id` are passed to the tasks as
   `GRAPHRAG_SOURCE_DIRECTORY` / `GRAPHRAG_PIPELINE_ID`.)
