# aws-graphrag — Infrastructure (AWS CDK, Python)

Modular CDK app that provisions the AWS-native stack the library targets:
Bedrock + Neptune + OpenSearch + DynamoDB + S3, an ECS Fargate data plane, and a
Step Functions ingestion pipeline, with CloudWatch observability and an optional
Bedrock Guardrail.

## Stacks

| Stack | Resources |
|---|---|
| `…-networking` | VPC (reuse or create), subnets, security group, VPC endpoints (Bedrock/S3/DDB/ECR/CW/SFN/…) |
| `…-storage` | Neptune cluster (IAM auth), OpenSearch domain (VPC, encrypted), DynamoDB doc-status table, S3 cache bucket |
| `…-compute` | ECR repo, ECS cluster, Fargate task definition + least-privilege task role |
| `…-orchestration` | Step Functions state machine — 4 resumable phases on Fargate + retries + SNS alarms |
| `…-observability` | CloudWatch dashboard + failure alarm over the pipeline & EMF metrics |
| `…-security` | Bedrock Guardrail (reuse or create a baseline PII/prompt-attack guardrail) |

The orchestration runs the ingestion CLI as four phases sharing one
`--pipeline-id` (the app's S3 stage checkpoints hand off between phases):

```
Prep (parse/load/chunk/translate) → GraphBuild (extract/glean/resolve/claims)
  → Analysis (graph_analysis/community_detection) → Index (indexing)
```

## Configuration (cdk.json context / `-c key=value`)

| Key | Default | Meaning |
|---|---|---|
| `env_name` | `dev` | stack/resource name prefix |
| `network_mode` | `private` | `private` = isolated subnets + VPC endpoints, **no NAT** (no internet egress); `public` = private subnets with NAT egress |
| `vpc_id` | _(none)_ | **reuse** an existing VPC instead of creating one |
| `max_azs` | `2` | AZs for a newly-created VPC |
| `cache_bucket_name` | _(none)_ | **reuse** an existing S3 cache bucket instead of creating one |
| `neptune_instance` | `db.r5.large` | Neptune instance class |
| `opensearch_instance` | `r6g.large.search` | OpenSearch data node type |
| `opensearch_count` | `2` | OpenSearch data node count |
| `guardrail_identifier` | _(none)_ | **reuse** an existing Bedrock guardrail |
| `removal_destroy` | `true` | `DESTROY` (dev) vs `RETAIN` (prod) on stack deletion |

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

See `docs/docparser-e2e-plan.md` for an end-to-end test plan on real contracts.
