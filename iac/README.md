# unified-kg-rag-on-aws — Infrastructure (AWS CDK, Python)

Modular CDK app that provisions the AWS-native stack the library targets:
Bedrock + Neptune + OpenSearch + DynamoDB + S3, an ECS Fargate data plane, and a
Step Functions ingestion pipeline, with CloudWatch observability and an optional
Bedrock Guardrail.

## Stacks

Stack ids are PascalCase with a `GraphRag` prefix. The default `dev` env keeps
the bare ids below (`GraphRagNetwork`, …); any other `env_name` adds an env
segment (`-c env_name=prod` → `GraphRagProdNetwork`, …) so several environments
can coexist in one account/region. Every resource also carries an `env` tag.

| Stack | Resources |
|---|---|
| `GraphRagNetwork` | VPC (reuse or create), subnets, security group, VPC endpoints for a created VPC: S3/DynamoDB gateways always; in `private` mode also interface endpoints for Bedrock (`bedrock`, `bedrock-runtime`, `bedrock-agent-runtime`), ECR (`ecr.api`, `ecr.dkr`), CloudWatch Logs and STS |
| `GraphRagStorage` | Neptune cluster (IAM auth), OpenSearch domain (VPC, encrypted), DynamoDB doc-status table, S3 cache bucket |
| `GraphRagCompute` | ECR repo, ECS cluster, Fargate task definition + least-privilege task role |
| `GraphRagOrchestration` | Step Functions state machine — 4 resumable phases on Fargate + retries + SNS alarm topic. The topic is encrypted with its own customer-managed key whose policy lets CloudWatch alarms publish (the AWS-managed `alias/aws/sns` key cannot, so alarm notifications would be dropped) |
| `GraphRagObservability` | CloudWatch dashboard + alarms: pipeline-failure, silent indexing-failure and extraction-failure (EMF), and store health (OpenSearch cluster-red / free-storage / JVM pressure, DynamoDB write throttling) → SNS. Synth warns if `alarm_email` is unset (alarms would have no subscriber) |
| `GraphRagSecurity` | Shared customer-managed KMS key (optional, `use_cmk`) |
| `GraphRagGuardrail` | Bedrock Guardrail, **pinned to `bedrock_region`** (creates and keeps a baseline PII/prompt-attack guardrail; empty with `create_guardrail=false`). It anonymizes email, phone and card numbers but not `NAME`: names are corpus content, and anonymizing them on the query path puts `{NAME}` in answers and removes entity-search seeds. The app applies the guardrail's `DRAFT` version |

### Resource naming

Physical resources share a single lowercase `<prefix>-<purpose>` scheme,
following a common `<app>-<purpose>-…` convention. The prefix is `graphrag` in
the default `dev` env and `<env>-graphrag` otherwise (e.g. `prod-graphrag-doc-status`),
so physical names do not collide across environments. Names below are for `dev`:

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
> Keep passing `-c guardrail_identifier=<id>` on every later deploy, or compute
> stops injecting it. The guardrail stack keeps owning the guardrail in step 2
> and afterwards: `guardrail_identifier` only selects the id compute **uses**,
> and creation is controlled separately by `create_guardrail`. To use an
> externally managed guardrail instead, pass
> `-c create_guardrail=false -c guardrail_identifier=<id>`; the stack then
> creates nothing.
>
> **Behaviour change for bring-your-own users:** setting only
> `-c guardrail_identifier=<id>` used to skip creation; it now *also* creates the
> baseline guardrail (`create_guardrail` defaults to `true`). Synth emits a
> warning when `guardrail_identifier` is set without an explicit
> `create_guardrail`; pass `-c create_guardrail=false` for an external guardrail
> (or `-c create_guardrail=true` to acknowledge and silence it in the two-step flow).
>
> The created guardrail follows `removal_destroy`: `DESTROY` in dev (default), so
> deploy → destroy → deploy cycles work, and `RETAIN` otherwise because its id
> reaches compute outside CloudFormation. After `cdk destroy` of a retained
> guardrail, delete it manually
> (`aws bedrock delete-guardrail --guardrail-identifier <id> --region <bedrock_region>`);
> otherwise the next deploy fails on the duplicate `<prefix>-guardrail-<region>` name.
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
| `vpc_id` | _(none)_ | **reuse** an existing VPC instead of creating one. The stack adds **no** VPC endpoints or subnets to it (synth warns with the list): in `private` mode it needs `PRIVATE_ISOLATED` subnets (no NAT or internet gateway route), S3/DynamoDB gateway endpoints, and private-DNS interface endpoints for `bedrock`, `bedrock-runtime`, `bedrock-agent-runtime`, `ecr.api`, `ecr.dkr`, `logs` and `sts`; in `public` mode it needs `PRIVATE_WITH_EGRESS` subnets (NAT route) |
| `max_azs` | `2` | AZs for a newly-created VPC |
| `cache_bucket_name` | _(none)_ | **reuse** an existing S3 cache bucket instead of creating one |
| `neptune_instance` | `db.r6g.large` | Neptune instance class (Graviton) |
| `neptune_instances` | `1` (dev) / `2` (non-dev) | Neptune instances; `>=2` ⇒ Multi-AZ HA (reader in another AZ). dev defaults to 1 (no failover) for cost |
| `opensearch_instance` | `r6g.large.search` | OpenSearch data node type (Graviton) |
| `opensearch_master_instance` | `m6g.large.search` | dedicated master node type, used only when `opensearch_count > 1`. 8 GiB Graviton, which AWS sizes for up to 10 nodes / 10K shards on OpenSearch 2.13; raise it for larger domains |
| `opensearch_count` | `1` (dev) / `2` (non-dev) | OpenSearch data node count (`>1` ⇒ 3 dedicated masters + zone awareness = 5 nodes; dev runs a single node for cost) |
| `backup_retention_days` | `7` | Neptune automated backup retention |
| `fargate_cpu` | `2048` | Fargate task vCPU units (in-task ProcessPool extractors scale with vCPU) |
| `fargate_memory` | `8192` | Fargate task memory (MiB) |
| `image_tag` | `latest` | container image tag the task pulls; pin a version tag to make ECR tags immutable |
| `create_guardrail` | `true` | `GraphRagGuardrail` creates and keeps a baseline PII/prompt-attack guardrail in `bedrock_region` (retained on stack deletion unless `removal_destroy`). `false` = bring your own guardrail; nothing is created |
| `guardrail_identifier` | _(none)_ | guardrail id the compute task **uses**, injected as `BEDROCK_GUARDRAIL_IDENTIFIER`. The created guardrail's id is **not** injected automatically: pass the `GuardrailIdentifier` output of `GraphRagGuardrail` here (two-step flow above), or an external id with `create_guardrail=false`. Unset = no guardrail on the task |
| `use_cmk` | `false` | customer-managed KMS key for at-rest encryption (S3/Neptune/OpenSearch/DDB). The SNS alarm topic always uses its own customer-managed key (see below) |
| `vpc_flow_logs` | `false` (dev) / `true` (non-dev) | enable VPC flow logs (created VPC only) |
| `deletion_protection` | `false` (dev) / `true` (non-dev) | protect Neptune/OpenSearch from deletion |
| `bedrock_model_arns` | _(none)_ | scope Bedrock IAM to specific model ARNs (list) |
| `alarm_email` | _(none)_ | subscribe an email to the pipeline alarm topic |
| `enable_cdk_nag` | `false` | run cdk-nag AwsSolutions (Well-Architected) checks at synth |
| `owner` | `aws-proserve` | `owner` tag applied to every resource |
| `cost_center` | `unified-kg-rag-on-aws` | `cost-center` tag applied to every resource |
| `removal_destroy` | `true` (dev) / `false` (non-dev) | `DESTROY` vs `RETAIN` on stack deletion for the stateful stores, the KMS keys, the guardrail, and the fixed-name ECR repository (emptied first) and log groups (`/<prefix>/tasks`, `/<prefix>/pipeline`). Retained fixed-name resources make the next deploy fail on the duplicate name, so delete them by hand after a `cdk destroy` with `removal_destroy=false` |

> Every resource is tagged `project=unified-kg-rag-on-aws`, `env=<env_name>`,
> `managed-by=cdk`, `owner`, and `cost-center` for cost allocation and ownership.

### Well-Architected

The stacks apply WAF defaults out of the box — least-privilege IAM
(`bedrock:InvokeModel` scoped to model/inference-profile ARNs, `neptune-db:connect`,
domain-scoped `es:ESHttp*`), encryption in transit + at rest, Graviton instances,
S3 cache
lifecycle (checkpoint prefixes only) + access logs, Step Functions X-Ray tracing + execution logging,
CloudWatch dashboard + failure alarm, and an SSL-only alarm topic. Production
hardening is opt-in via the flags above. Validate with:

```bash
cdk synth -c enable_cdk_nag=true                       # dev
cdk synth -c enable_cdk_nag=true -c use_cmk=true \
  -c vpc_flow_logs=true -c neptune_instances=2 -c opensearch_count=2 \
  -c deletion_protection=true -c removal_destroy=false  # prod-hardened
```
Both report zero AwsSolutions findings, and CI runs both. These CI synths have
no account, so account ids stay `<AWS::AccountId>` tokens; the IaC test
`iac/tests/test_app_nag.py` repeats both shapes with a dummy concrete account,
as a real deploy renders them. Accepted findings are
documented in `iac/nag_suppressions.py`: `AwsSolutions-IAM5` is suppressed per
role for the listed wildcards only (`appliesTo`), so a new wildcard fails the
synth, and the OpenSearch HA findings (OS4/OS7) are accepted only for a
single-node domain.

> **A few Bedrock read actions use `Resource: "*"` by necessity**, not oversight:
> `bedrock:Rerank` (authorizes against a different resource shape than
> `InvokeModel` — scoping it to the model ARNs denies the call) and
> `bedrock:ListInferenceProfiles` / `GetInferenceProfile` (account-level reads
> with no resource scoping). These are read/inference-only actions; the
> mutating/data path (`InvokeModel`) stays ARN-scoped. Each is documented inline
> in `compute_stack.py` and in the cdk-nag suppressions.

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
> everything down on `cdk destroy --all`; non-dev envs default to
> `removal_destroy=false` and `deletion_protection=true`.

## After deploy

1. Build & push the app image (`docker/Dockerfile`, build context = repo root)
   to the created ECR repo (tag `latest`). The image bakes in the tracked,
   endpoint-free `docker/config.yaml` as `/app/config.yaml`; the deployed
   endpoints come from the injected `NEPTUNE_ENDPOINT` / `OPENSEARCH_ENDPOINT` /
   `S3_BUCKET_NAME` / `BEDROCK_REGION` env vars the app reads. The task also injects
   `GRAPHRAG_DOC_STATUS_TABLE` (the table this stack created,
   `graphrag-doc-status` in `dev`) and `GRAPHRAG_DOC_STATUS_CREATE_TABLE=false`,
   which override `aws.dynamodb.table_name` / `create_table_if_missing`, so the
   app and the CloudWatch alarms track the same IaC-managed table. The image
   config sets `aws.dynamodb.enabled: true`, so ingestion runs incrementally.
   S3 cache uploads default to the bucket's own encryption
   (`aws.s3.encryption.encryption_type: BUCKET_DEFAULT`), so `use_cmk=true`
   objects are encrypted with the CMK.
2. Upload the corpus under a prefix of the cache bucket (for example `corpus/`)
   and start an ingestion run:
   ```bash
   aws stepfunctions start-execution \
     --state-machine-arn <…-ingestion arn> \
     --input '{"source_directory":"s3://<cache-bucket>/corpus/","pipeline_id":"run-001"}'
   ```
   The input takes exactly these two keys, passed to every phase as
   `GRAPHRAG_SOURCE_DIRECTORY` / `GRAPHRAG_PIPELINE_ID`. Each phase runs in a
   fresh Fargate task, so `source_directory` must be an `s3://` URI: the
   container entrypoint (`docker/entrypoint.sh`) syncs it to local scratch
   before running the CLI. The config file is fixed at `/app/config.yaml` in
   the image. The task role is granted read/write on the cache bucket only;
   to read a corpus from another bucket, grant the role access to it.

   The bucket's 30-day expiry applies only to the prefixes the app writes,
   `pipeline-runs/` (stage checkpoints) and `embedding-cache/`; any other
   prefix never expires. Do not put the corpus under those two prefixes: with
   incremental indexing an expired source file looks deleted, and the next run
   removes its graph and vector artifacts. A reused bucket
   (`cache_bucket_name`) keeps its own lifecycle rules, so check them the same
   way.
3. To query (`run-rag`) or run other CLIs against the deployed stores, use the
   `GraphRagStorage` outputs: `NeptuneEndpoint` and `OpenSearchEndpoint` are
   bare hostnames (no `https://`), the form `NEPTUNE_ENDPOINT` /
   `OPENSEARCH_ENDPOINT` and `aws.neptune.endpoint` / `aws.opensearch.endpoint`
   expect. Both stores are VPC-only: they accept connections only from inside
   the VPC through the service security group. Run the CLI there, for example as
   a one-off task of the same task definition (`aws ecs run-task` with a
   `run-rag …` command override, in the app subnets and service security
   group), not from a laptop.
