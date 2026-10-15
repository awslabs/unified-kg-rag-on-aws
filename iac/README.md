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
| `GraphRagNetwork` | VPC (reuse or create), subnets, data-plane security group, VPC endpoints for a created VPC: S3/DynamoDB gateways always; in `private` mode also interface endpoints (own security group, 443 from the data plane only) for Bedrock (`bedrock`, `bedrock-runtime`, `bedrock-agent-runtime`), ECR (`ecr.api`, `ecr.dkr`), CloudWatch Logs and STS, each with an endpoint policy that admits only this account's principals (`aws:PrincipalAccount`) |
| `GraphRagStorage` | Neptune cluster (IAM auth), OpenSearch domain (VPC, encrypted), DynamoDB doc-status table, S3 cache bucket |
| `GraphRagCompute` | ECR repo (lifecycle keeps the 10 newest images and never expires the deployed `image_tag`), ECS cluster, Fargate task definition + least-privilege task role (S3 cache read + put, no delete). Injects `OPENSEARCH_NUMBER_OF_REPLICAS` = 1 on a multi-node domain (0 on a single node) |
| `GraphRagOrchestration` | Step Functions state machine — 4 resumable phases on Fargate, retried only when ECS cannot start the task (a phase that exits non-zero is not rerun), one JSON failure notification (execution id, error, cause; the publish is retried on throttling or transient errors) + SNS alarm topic. The topic is encrypted with its own customer-managed key whose policy lets CloudWatch alarms publish (the AWS-managed `alias/aws/sns` key cannot, so alarm notifications would be dropped) |
| `GraphRagObservability` | CloudWatch dashboard + alarms: pipeline failed (a failed phase is also published by the orchestration failure notification, so it notifies twice; the alarm also covers failures the notification cannot report, such as an invalid start input), timed out or aborted, silent indexing-failure and extraction-failure (EMF), and store health (OpenSearch cluster-red / free-storage / JVM pressure, DynamoDB write throttling, Neptune CPU / memory / Gremlin server errors sustained for 3 consecutive minutes, so retried write conflicts during parallel writes do not page) → SNS. Synth warns if `alarm_email` is unset (alarms would have no subscriber) |
| `GraphRagSecurity` | Shared customer-managed KMS key (optional, `use_cmk`) |
| `GraphRagGuardrail` | Bedrock Guardrail, **pinned to `bedrock_region`** (creates and keeps a baseline PII/prompt-attack guardrail; empty with `create_guardrail=false`). It anonymizes email, phone and card numbers but not `NAME`: names are corpus content, and anonymizing them on the query path puts `{NAME}` in answers and removes entity-search seeds. The app applies the guardrail's `DRAFT` version unless `guardrail_version` is set |

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
> (which churn every deploy-region stack's exports). A `bedrock_region` other
> than the deploy region needs `network_mode=public`: the default `private`
> mode reaches Bedrock only through the deploy region's VPC endpoints, so synth
> fails otherwise. With a deploy region of, say, `ap-northeast-2` and Bedrock
> in `us-west-2`:
>
> ```bash
> # 0) once: bootstrap bedrock_region too (cdk bootstrap covers one region)
> cdk bootstrap aws://<account>/us-west-2
> # 1) create the guardrail in bedrock_region, note its id from the output
> cdk deploy GraphRagGuardrail -c network_mode=public -c bedrock_region=us-west-2 …
> # 2) pass that id so compute injects BEDROCK_GUARDRAIL_IDENTIFIER
> cdk deploy --all -c network_mode=public -c bedrock_region=us-west-2 \
>   -c guardrail_identifier=<id> …
> ```
>
> In `private` mode leave `bedrock_region` unset (it defaults to the deploy
> region): the same two steps then create the guardrail in the deploy region,
> and step 0 is not needed.
>
> Keep passing `-c guardrail_identifier=<id>` on every later deploy, or compute
> stops injecting it. The guardrail stack keeps owning the guardrail in step 2
> and afterwards: `guardrail_identifier` only selects the id compute **uses**,
> and creation is controlled separately by `create_guardrail`. To use an
> externally managed guardrail instead, pass
> `-c create_guardrail=false -c guardrail_identifier=<id>`; the stack then
> creates nothing.
>
> Setting only `-c guardrail_identifier=<id>` still creates the baseline
> guardrail, because `create_guardrail` defaults to `true`. Synth warns when
> `guardrail_identifier` is set without an explicit `create_guardrail`; pass
> `-c create_guardrail=false` for an external guardrail (or
> `-c create_guardrail=true` to acknowledge the warning in the two-step flow).
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
| `bedrock_region` | deploy region | Bedrock runtime region: injected into the task as `BEDROCK_REGION`, and the region of the `GraphRagGuardrail` stack. With `network_mode=private`, synth fails unless it equals the deploy region (the Bedrock VPC endpoints serve the deploy region only) |
| `network_mode` | `private` | `private` = isolated subnets + VPC endpoints, **no NAT** (no internet egress); `public` = private subnets with NAT egress |
| `vpc_id` | _(none)_ | **reuse** an existing VPC instead of creating one. The stack adds **no** VPC endpoints or subnets to it (synth warns with the list): in `private` mode it needs `PRIVATE_ISOLATED` subnets (no NAT or internet gateway route), S3/DynamoDB gateway endpoints, and private-DNS interface endpoints for `bedrock`, `bedrock-runtime`, `bedrock-agent-runtime`, `ecr.api`, `ecr.dkr`, `logs` and `sts` (and `xray` with `enable_cloudwatch_tracing`); in `public` mode it needs `PRIVATE_WITH_EGRESS` subnets (NAT route) |
| `max_azs` | `2` | AZs for a newly-created VPC; at least `2` (the Neptune cluster needs subnets in two AZs) |
| `cache_bucket_name` | _(none)_ | **reuse** an existing S3 cache bucket instead of creating one |
| `corpus_bucket_name` | _(none)_ | S3 bucket the corpus is read from when it is not in the cache bucket: the task role gets read-only access and, in `private` mode, the S3 gateway endpoint policy allows it |
| `neptune_instance` | `db.r6g.large` | Neptune instance class (Graviton) |
| `neptune_instances` | `1` (dev) / `2` (non-dev) | Neptune instances; `>=2` ⇒ Multi-AZ HA (reader in another AZ). dev defaults to 1 (no failover) for cost |
| `opensearch_instance` | `r6g.large.search` | OpenSearch data node type (Graviton) |
| `opensearch_master_instance` | `m6g.large.search` | dedicated master node type, used only when `opensearch_count > 1`. 8 GiB Graviton, which AWS sizes for up to 10 nodes / 10K shards on OpenSearch 2.13; raise it for larger domains |
| `opensearch_count` | `1` (dev) / `2` (non-dev) | OpenSearch data node count (`>1` ⇒ 3 dedicated masters + zone awareness, e.g. 2 data nodes = 5 nodes; dev runs a single node for cost). A zone-aware domain spans as many AZs as it has data nodes, up to 3 and up to the AZs of the VPC (`max_azs` for a created VPC): 3 data nodes with `max_azs=3` use 3 AZs. A domain in 2 AZs needs an even count (OpenSearch rejects an odd one), so synth fails on, say, `opensearch_count=3` with `max_azs=2`. **Upgrading:** earlier versions pinned such domains to 2 AZs, so the first deploy of an existing domain with 3 or more data nodes in a VPC with 3 or more AZs moves it to 3 AZs, which OpenSearch applies as a blue/green deployment (new nodes, every shard copied; the domain stays available but slower, which can take hours on a large domain). Deploy it in a quiet window, with no ingestion running |
| `opensearch_volume_gib` | `50` | GP3 volume per OpenSearch data node; the free-storage alarm fires at 40% free |
| `opensearch_korean_analyzer` | `cjk` | OpenSearch analyzer of Korean text, injected as `OPENSEARCH_KOREAN_ANALYZER` (overrides `indexing.opensearch.language_analyzers.ko` of the image config). The domain has no `nori` unless its optional Nori package (`analysis-nori`) is associated, and an index whose mapping names a missing analyzer fails to create; `cjk` (character bigrams) is built into OpenSearch. To use `nori`, associate the package (console **Packages** or `aws opensearch associate-package`, a blue/green deployment), deploy with `nori`, and re-run once with `indexing.reset: true` |
| `doc_status_table` | `graphrag-doc-status` (dev) / `<env>-graphrag-doc-status` (non-dev) | DynamoDB doc-status (incremental-indexing registry) table name, injected into the task as `GRAPHRAG_DOC_STATUS_TABLE`. Changing it on a deployed environment creates a new, empty registry table |
| `backup_retention_days` | `7` | Neptune automated backup retention, 1-35 days (see [Backup and restore](../docs/operations.md#backup-and-restore)) |
| `fargate_cpu` | `2048` | Fargate task vCPU units (in-task ProcessPool extractors scale with vCPU): `256`, `512`, `1024`, `2048`, `4096`, `8192` or `16384` |
| `fargate_memory` | `8192` | Fargate task memory (MiB); must be a size Fargate offers for `fargate_cpu` (with `2048`: 4096-16384 in 1024 steps), else synth fails naming the valid range |
| `image_tag` | `latest` | container image tag the task pulls; pin a version tag to make ECR tags immutable. The ECR lifecycle keeps this tag and the 10 newest images, so pin a tag rather than a digest |
| `sfn_timeout_hours` | `6` | ingestion state machine timeout for a whole run (at least 1); a run that hits it is `TIMED_OUT` (alarmed) |
| `create_guardrail` | `true` | `GraphRagGuardrail` creates and keeps a baseline PII/prompt-attack guardrail in `bedrock_region` (retained on stack deletion unless `removal_destroy`). `false` = bring your own guardrail; nothing is created |
| `guardrail_identifier` | _(none)_ | guardrail id the compute task **uses**, injected as `BEDROCK_GUARDRAIL_IDENTIFIER`. The created guardrail's id is **not** injected automatically: pass the `GuardrailIdentifier` output of `GraphRagGuardrail` here (two-step flow above), or an external id with `create_guardrail=false`. Unset = no guardrail on the task |
| `guardrail_version` | _(none)_ | published guardrail version the task applies, injected as `BEDROCK_GUARDRAIL_VERSION`. Unset = `DRAFT`, which changes whenever the guardrail is edited; publish a version and pin it outside dev |
| `use_cmk` | `false` | customer-managed KMS key for at-rest encryption (S3/Neptune/OpenSearch/DDB/ECR and every log group; the key policy lets CloudWatch Logs use it for this account's log groups, and principals in this account may use it through CloudWatch Logs (`kms:ViaService`), so log writers and readers need no KMS permissions of their own). Choose it before the first deploy: ECR sets a repository's encryption only at creation, and CloudFormation cannot replace the fixed-name repository, so turning `use_cmk` on (or off) for a deployed environment fails until the compute stack is destroyed and redeployed. The SNS alarm topic always uses its own customer-managed key (see below) |
| `vpc_flow_logs` | `false` (dev) / `true` (non-dev) | enable VPC flow logs (created VPC only) |
| `flow_log_retention_days` | `731` | flow-log log group retention; must be a CloudWatch Logs retention value (e.g. `30`, `90`, `365`, `731`) |
| `deletion_protection` | `false` (dev) / `true` (non-dev) | deletion protection on the Neptune cluster and the DynamoDB doc-status table (blocks a direct delete API/console call). OpenSearch domains have no deletion-protection setting; outside dev the domain is only kept by `removal_destroy=false` (CloudFormation `Retain`), which does not stop a direct `DeleteDomain` call |
| `bedrock_model_arns` | _(none)_ | scope Bedrock `InvokeModel` IAM to specific model ARNs: a list in `cdk.json`, or a comma-separated string with `-c bedrock_model_arns=arn:...,arn:...`. The list **replaces** the default `foundation-model/*` grant. If it names only foundation models, the default `inference-profile/*` grants are kept, because the app calls models that are offered only through an inference profile (Claude 4.5 and later) by profile id; a profile call is still authorized against the listed foundation-model ARNs, so no other model becomes callable. Cross-region and global profiles route to several regions, so list foundation models with a `*` region (`arn:aws:bedrock:*::foundation-model/<model-id>`). If the list names any inference-profile ARN, it is used exactly as given and no other profile is granted. `Rerank` and `ListInferenceProfiles` stay on `*` (they have no resource scoping) |
| `alarm_email` | _(none)_ | subscribe an email to the pipeline alarm topic |
| `enable_cdk_nag` | `false` | run cdk-nag AwsSolutions (Well-Architected) checks at synth |
| `enable_cloudwatch_tracing` | `false` | send the app's OpenTelemetry traces to CloudWatch through an ADOT collector sidecar ([CloudWatch tracing](#cloudwatch-tracing-optional)). Adds the sidecar, an ECR pull-through cache rule for its image, X-Ray write actions on the task role and, in `private` mode, an `xray` interface endpoint. Off, the templates are the same as without the option. Needs CloudWatch Transaction Search in the account and region, and an image built with the `otel` extra. With it on, `env_name` may be at most 10 characters (the cache prefix `<env>-graphrag-ecr-public` is limited to 30) |
| `owner` | `aws-proserve` | `owner` tag applied to every resource |
| `cost_center` | `unified-kg-rag-on-aws` | `cost-center` tag applied to every resource |
| `removal_destroy` | `true` (dev) / `false` (non-dev) | `DESTROY` vs `RETAIN` on stack deletion for the stateful stores, the KMS keys, the guardrail, the OpenSearch domain log groups, and the fixed-name ECR repository (emptied first) and log groups (`/<prefix>/tasks`, `/<prefix>/pipeline`). Retained fixed-name resources make the next deploy fail on the duplicate name, so delete them by hand after a `cdk destroy` with `removal_destroy=false` |

Boolean keys take `true`/`false` (also `yes`/`no`, `on`/`off`, `1`/`0`); any
other value fails synth instead of reading as `false`. Counts and sizes must be
positive integers (`opensearch_count`, `neptune_instances`, `sfn_timeout_hours`,
`backup_retention_days`, ...); synth names the key of a value out of range.

> Every resource is tagged `project=unified-kg-rag-on-aws`, `env=<env_name>`,
> `managed-by=cdk`, `owner`, and `cost-center` for cost allocation and ownership.

> **The Neptune engine version is not pinned.** A new cluster starts on the
> engine version Neptune defaults to at creation, so clusters deployed at
> different times run different versions. Adding a pin now would make the next
> deploy of every cluster not already on it change its engine: an upgrade with
> downtime, or a downgrade that Neptune rejects. The client is pinned instead:
> `gremlinpython` stays below 3.8 until Neptune's engine supports TinkerPop 3.8
> (see `pyproject.toml`). To fix a version for a new deployment, set
> `engine_version` in `storage_stack.py` before its first deploy.

### Well-Architected

The stacks apply WAF defaults out of the box — least-privilege IAM
(`bedrock:InvokeModel` and `bedrock:CountTokens` scoped to model/inference-profile ARNs, `neptune-db:connect`,
domain-scoped `es:ESHttp*`), encryption in transit + at rest, Graviton instances,
S3 cache
lifecycle (checkpoint prefixes only) + access logs, Step Functions X-Ray tracing + execution logging,
CloudWatch dashboard + failure notifications and alarms, and an SSL-only alarm topic. Production
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
synth, the OpenSearch HA findings (OS4/OS7) are accepted only for a
single-node domain, and OS9 only for the slow logs, which record nothing
without index slow-log thresholds (the app sets none); the domain publishes
its application logs.

> **A few Bedrock read actions use `Resource: "*"` by necessity**, not oversight:
> `bedrock:Rerank` (authorizes against a different resource shape than
> `InvokeModel` — scoping it to the model ARNs denies the call) and
> `bedrock:ListInferenceProfiles` (an account-level read with no resource
> scoping). These are read/inference-only actions; the
> mutating/data path (`InvokeModel`) stays ARN-scoped. Each is documented inline
> in `compute_stack.py` and in the cdk-nag suppressions.

## Deployment topology

```mermaid
flowchart LR
    operator["Operator"] -->|start-execution| sfn["Step Functions<br/>Prep, GraphBuild, Analysis, Index"]
    operator -->|run-task| task
    sfn -->|RunTask per phase| task
    subgraph vpc["VPC (max_azs AZs)"]
        subgraph app["app subnets: isolated (private) or NAT egress (public)"]
            task["ECS Fargate task<br/>container app"]
            neptune[("Neptune cluster<br/>IAM auth, port 8182")]
            opensearch[("OpenSearch domain<br/>HTTPS 443")]
        end
        iface["Interface endpoints, private mode only<br/>bedrock, bedrock-runtime, bedrock-agent-runtime,<br/>ecr.api, ecr.dkr, logs, sts"]
        gateway["Gateway endpoints<br/>S3, DynamoDB"]
    end
    task --> neptune
    task --> opensearch
    task --> iface
    task --> gateway
    iface --> bedrock["Amazon Bedrock<br/>models, rerank, guardrail"]
    iface --> ecr[("ECR repository")]
    iface --> logs["CloudWatch Logs"]
    gateway --> s3[("S3 cache bucket")]
    gateway --> ddb[("DynamoDB doc-status table")]
    logs --> alarms["Dashboard and alarms"] --> sns["SNS alarm topic"]
```

- One VPC holds the data plane. The Fargate task, the Neptune cluster and the
  OpenSearch domain run in the `app` subnets and share one security group that
  allows Neptune (8182) and OpenSearch (443) traffic between its members, so
  both stores are reachable only from inside the VPC.
- In `private` mode (default) the `app` subnets have no route to the internet,
  and the created VPC has no public subnet or internet gateway (the public
  subnets' address range stays reserved, so the `app` subnets get the same
  CIDRs in both modes). The task reaches AWS services only through the
  interface endpoints and the S3/DynamoDB gateway endpoints shown, so Bedrock
  calls must go to the deploy region (leave `bedrock_region` at the deploy
  region). In `public` mode the
  `app` subnets route through one NAT gateway per AZ, no interface endpoints are
  created, and Bedrock may be in another region.
- Step Functions runs the four ingestion phases as separate Fargate tasks of
  one task definition; they hand off through the S3 stage checkpoints.
  Querying (`run-rag`, `run-eval`) runs as a one-off task of the same task
  definition ([After deploy](#after-deploy)).
- The optional guardrail stack lives in `bedrock_region`, outside the deploy
  region's stacks.

## Cost drivers

No prices are listed here; use the [AWS Pricing Calculator](https://calculator.aws/)
for your region and sizes. What scales each service's cost:

| Service | Cost grows with | Levers in this stack or the app config |
|---|---|---|
| Amazon Bedrock models | Tokens per ingestion (one or more extraction calls per chunk, up to `max_rounds` gleaning calls, one report per community, optional claim extraction) and per query (routing, query processing, map steps, answer) | Fast tier for light roles, `aws.bedrock.ingestion_effort`, `processing.gleaning.max_rounds`, keep claim extraction off, incremental indexing |
| Bedrock embeddings and rerank | Embedded text units, entities, relationships and reports; one rerank call per query over up to `search.reranking.top_k` candidates | `indexing.opensearch.persist_embedding_cache`, `search.reranking.top_k` |
| Amazon Neptune | Instance-hours × `neptune_instances`, storage and I/O, backup retention | `neptune_instance`, `neptune_instances` (1 in dev), `backup_retention_days`, tear down dev |
| Amazon OpenSearch Service | Data node-hours × `opensearch_count`, plus three dedicated masters when `opensearch_count > 1`, and 50 GiB gp3 storage per data node | `opensearch_instance`, `opensearch_master_instance`, `opensearch_count` (1 in dev) |
| AWS Fargate | vCPU and memory per task for the run time of each phase and query task | `fargate_cpu`, `fargate_memory` |
| VPC interface endpoints (`private`) | Seven endpoints (eight with `enable_cloudwatch_tracing`) × `max_azs`, billed per AZ-hour, plus data processed | `max_azs`, or reuse a VPC's endpoints (`vpc_id`) |
| NAT gateways (`public`) | One per AZ, billed per hour, plus data processed | `max_azs`, or use `private` mode |
| Amazon S3 | Stage checkpoints, embedding cache, corpus, access logs | 30-day expiry on `pipeline-runs/` and `embedding-cache/` |
| Amazon DynamoDB | On-demand reads and writes: one full-table diff scan per ingestion plus one record per processed document; point-in-time recovery | Fewer, larger runs |
| Amazon CloudWatch | Log ingestion and retention (task, pipeline and, with `vpc_flow_logs`, flow logs), alarms, dashboard | `vpc_flow_logs`, `flow_log_retention_days` |
| AWS KMS | Customer-managed keys per month and requests: the alarm-topic key always, the data key with `use_cmk` | `use_cmk` |
| AWS Step Functions | State transitions per ingestion run (a few per phase) | — |
| CloudWatch traces (`enable_cloudwatch_tracing`) | Spans ingested by Transaction Search (billed as CloudWatch Logs ingestion into `aws/spans`) and the share of spans indexed beyond the free 1% | `observability.sample_rate` in the image config, the Transaction Search indexing percentage |

Neptune, OpenSearch, interface endpoints and NAT gateways bill while the stack
exists, whether or not anything runs. `cdk destroy --all` removes them in the
default `dev` environment.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.12+ (uv downloads a matching
  Python when none is installed). `iac/` is a uv project of its own, separate
  from the root package: `iac/uv.lock` pins the CDK libraries.
- Node.js on a supported LTS line (22 or 24; CI uses 24) and the AWS CDK CLI
  v2, 2.1143.0 or later (`npm install -g aws-cdk@2.1143.0` is the version CI
  pins). An older CLI can reject the cloud assembly that the `aws-cdk-lib`
  release in `uv.lock` produces; check with `cdk --version`.
- For `cdk bootstrap`/`cdk deploy`: AWS credentials for the target account
  and region. To push the app image after deploy: Docker.

## Usage

```bash
cd iac
uv sync --locked          # installs exactly what iac/uv.lock pins
. .venv/bin/activate      # or prefix each command below with `uv run`

# Synthesize (no AWS changes)
cdk synth

# Reuse existing VPC + S3, fully private data plane (recommended for prod):
cdk synth -c vpc_id=vpc-0abc... -c cache_bucket_name=my-cache -c network_mode=private

# Public egress (simpler dev):
cdk synth -c network_mode=public

# Deploy (creates billable resources)
cdk bootstrap            # once per account/region; with a bedrock_region other
                         # than the deploy region, also
                         # cdk bootstrap aws://<account>/<bedrock_region>
                         # (the guardrail stack deploys there)
cdk deploy --all
```

> **Cost:** deploying creates Neptune + OpenSearch (hourly billed) and
> NAT gateways in `public` mode (see [Cost drivers](#cost-drivers)). `removal_destroy=true` (dev default) tears
> everything down on `cdk destroy --all`; non-dev envs default to
> `removal_destroy=false` and `deletion_protection=true`.

## CloudWatch tracing (optional)

`-c enable_cloudwatch_tracing=true` adds an AWS Distro for OpenTelemetry
(ADOT) collector sidecar to the task. The app exports its spans (ingestion
stages, queries, retrieval, rerank, and model calls with token usage) over
OTLP/HTTP to the collector on `127.0.0.1:4318`; the collector signs them with
the task role (SigV4) and sends them to the CloudWatch OTLP traces endpoint
`https://xray.<region>.amazonaws.com/v1/traces`. Prompt and completion text
is not recorded (`observability.capture_content` stays `false`).

1. Turn on CloudWatch Transaction Search once per account and region; the
   traces endpoint rejects spans until it is on. This is an account-level
   setting that the stacks do not change: see
   [Observability](../docs/operations.md#cloudwatch-through-the-adot-collector-sidecar)
   for the commands and what it switches.
2. Build the image with the `otel` extra (step 2 of [After deploy](#after-deploy)):
   `--build-arg UV_EXTRAS="--extra otel"` (with Markdown parsing,
   `--build-arg UV_EXTRAS="--extra otel --extra unstructured"`). An image
   without it fails at start-up with
   `observability.enabled is true but OpenTelemetry is not installed`.
3. Deploy with the option on:
   `cdk deploy --all -c enable_cloudwatch_tracing=true`.
4. Before the first run, pull the collector image once from a workstation
   (with Docker and permission to pull from the account's ECR). The image
   comes through an ECR pull-through cache, and the first pull through it may
   need an internet route when ECR is reached through an interface endpoint,
   as the `private` data plane does. ECS fails a task when it cannot pull any
   of its images, so until the cache holds the image every phase would fail.
   The pull fills the cache; later pulls come from it. Repeat after an
   upgrade that changes the pinned digest. Outside `dev`, use the
   environment's stack name (for example `GraphRagProdCompute`).

   ```bash
   REGION=<deploy region>
   COLLECTOR_IMAGE=$(aws cloudformation describe-stacks --region "$REGION" \
     --stack-name GraphRagCompute \
     --query "Stacks[0].Outputs[?OutputKey=='CollectorImageUri'].OutputValue" \
     --output text)
   aws ecr get-login-password --region "$REGION" \
     | docker login --username AWS --password-stdin "${COLLECTOR_IMAGE%%/*}"
   docker pull --platform linux/amd64 "$COLLECTOR_IMAGE"
   ```

What the option changes:

- **Task definition**: an `otel-collector` container (ADOT collector
  `v0.50.0`, pinned by digest in `compute_stack.py`) that is not essential, so
  a collector that crashes or cannot export loses traces but never stops a
  run. An image ECS cannot pull does fail the task (step 4). The `app` container
  starts after it and gets `GRAPHRAG_OBSERVABILITY_ENABLED=true`,
  `GRAPHRAG_OBSERVABILITY_BACKEND=otlp`,
  `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318` and
  `OTEL_RESOURCE_ATTRIBUTES=deployment.environment=<env_name>`. The collector
  logs to the task log group under the `otel-collector/` stream prefix.
- **Image source**: the collector image is pulled through an ECR pull-through
  cache rule for ECR Public (prefix `<prefix>-ecr-public`), because the
  `private` data plane cannot reach `public.ecr.aws`. ECR creates the cached
  repository on the first pull (step 4); `cdk destroy` removes the rule but
  not that repository. The `GraphRagCompute` output `CollectorImageUri` is
  the cached image.
- **IAM**: the task role gets `xray:PutTraceSegments`, the one action the
  OTLP traces endpoint authorizes (write-only; X-Ray has no resource-level
  scoping), and the execution role may create and import into the one cached
  repository.
- **Network**: in `private` mode, an `xray` interface endpoint (same
  security group and this-account endpoint policy as the others). A reused
  private VPC must provide it.

Traces appear in the CloudWatch console under **Application Signals,
Transaction Search** (service `unified-kg-rag`), and spans are stored in the
`aws/spans` log group.

## After deploy

The commands below assume the default `dev` environment (stack and resource
names as in [Resource naming](#resource-naming)), a shell in the repository
root, and the AWS CLI v2, Docker and `jq`. For another `env_name`, use its
stack names (for example `GraphRagProdStorage`) and resource prefix
(`prod-graphrag`).

### 1. Read the stack outputs

```bash
REGION=<deploy region>
output() {
  aws cloudformation describe-stacks --region "$REGION" --stack-name "$1" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text
}
STATE_MACHINE_ARN=$(output GraphRagOrchestration StateMachineArn)
CACHE_BUCKET=$(output GraphRagStorage CacheBucketName)
NEPTUNE_ENDPOINT=$(output GraphRagStorage NeptuneEndpoint)
OPENSEARCH_ENDPOINT=$(output GraphRagStorage OpenSearchEndpoint)
```

| Stack | Output | Value |
|---|---|---|
| `GraphRagStorage` | `NeptuneEndpoint`, `OpenSearchEndpoint` | Bare hostnames (no `https://`), the form `aws.neptune.endpoint` / `aws.opensearch.endpoint` and `NEPTUNE_ENDPOINT` / `OPENSEARCH_ENDPOINT` expect |
| `GraphRagStorage` | `CacheBucketName`, `DocStatusTableName` | S3 cache bucket and doc-status table |
| `GraphRagOrchestration` | `StateMachineArn` | Ingestion state machine |
| `GraphRagSecurity` | `KmsKeyArn` | Shared CMK, only with `use_cmk=true` |
| `GraphRagGuardrail` | `GuardrailIdentifier` | Created guardrail, in `bedrock_region` |
| `GraphRagCompute` | `CollectorImageUri` | Collector image to pull once after deploy, only with `enable_cloudwatch_tracing` ([CloudWatch tracing](#cloudwatch-tracing-optional)) |

The task already receives the endpoints, bucket, table and regions as
environment variables, so these outputs are needed only for tools that run
elsewhere in the VPC.

### 2. Build and push the app image

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/amd64 -f docker/Dockerfile -t "$REGISTRY/graphrag-app:latest" .
docker push "$REGISTRY/graphrag-app:latest"
```

The build context is the repository root, and the task runs on the default
Fargate architecture (x86_64). Outside dev, push a unique version tag instead
and deploy with `-c image_tag=<tag>`: any tag other than `latest` makes the
repository's tags immutable, so a pushed image cannot be swapped under the
running task definition. To parse `.md`/`.html` files, build with
`--build-arg UV_EXTRAS="--extra unstructured"`; with `enable_cloudwatch_tracing`,
add the `otel` extra (`--build-arg UV_EXTRAS="--extra otel"`).

The image bakes in the tracked, endpoint-free `docker/config.yaml` as
`/app/config.yaml`; the deployed values come from the environment variables the
task injects: `NEPTUNE_ENDPOINT`, `OPENSEARCH_ENDPOINT`, `S3_BUCKET_NAME`,
`AWS_REGION`, `BEDROCK_REGION`, `LOG_FORMAT`, `OPENSEARCH_NUMBER_OF_REPLICAS`,
`OPENSEARCH_KOREAN_ANALYZER` (`opensearch_korean_analyzer`),
`GRAPHRAG_DOC_STATUS_TABLE`
(the table this stack created) and `GRAPHRAG_DOC_STATUS_CREATE_TABLE=false`,
plus `BEDROCK_GUARDRAIL_IDENTIFIER` when `guardrail_identifier` is set (and
`BEDROCK_GUARDRAIL_VERSION` when `guardrail_version` is). The
image config sets `aws.dynamodb.enabled: true`, so ingestion runs
incrementally. S3 cache uploads default to the bucket's own encryption
(`aws.s3.encryption.encryption_type: BUCKET_DEFAULT`), so with `use_cmk=true`
objects are encrypted with the CMK.

### 3. Upload the corpus and start an ingestion run

```bash
aws s3 sync ./source "s3://$CACHE_BUCKET/corpus/" --region "$REGION"

# Run one ingestion at a time: check that none is in progress first.
aws stepfunctions list-executions --region "$REGION" \
  --state-machine-arn "$STATE_MACHINE_ARN" --status-filter RUNNING

EXECUTION_ARN=$(aws stepfunctions start-execution --region "$REGION" \
  --state-machine-arn "$STATE_MACHINE_ARN" \
  --input "{\"source_directory\":\"s3://$CACHE_BUCKET/corpus/\",\"pipeline_id\":\"run-001\"}" \
  --query executionArn --output text)

aws stepfunctions describe-execution --region "$REGION" \
  --execution-arn "$EXECUTION_ARN" --query status --output text
aws logs tail /graphrag/tasks --region "$REGION" --follow
```

- The input takes exactly two keys, passed to every phase as
  `GRAPHRAG_SOURCE_DIRECTORY` / `GRAPHRAG_PIPELINE_ID`. Each phase runs in a
  fresh Fargate task, so `source_directory` must be an `s3://` URI: the
  container entrypoint (`docker/entrypoint.sh`) syncs it to local scratch
  before running the CLI. `pipeline_id` accepts lowercase letters, digits,
  hyphens and underscores; reuse an id to resume a failed run from its S3
  checkpoints: start a new execution with the same input, and each phase
  skips the stages of its window that already completed, so only the failed
  stage onward runs again.
- The task role can read and write the cache bucket only. To read a corpus
  from another bucket, deploy with `-c corpus_bucket_name=<bucket>`. In
  `private` mode the S3 and DynamoDB gateway endpoints only allow this
  deployment's cache and corpus buckets, its doc-status table and the ECR
  image-layer bucket, so a bucket granted to the role by hand is still
  unreachable from the tasks.
- The bucket's 30-day expiry applies only to the prefixes the app writes,
  `pipeline-runs/` (stage checkpoints) and `embedding-cache/`; any other
  prefix never expires. Do not put the corpus under those two prefixes: with
  incremental indexing an expired source file looks deleted, and the next run
  removes its graph and vector artifacts. A reused bucket
  (`cache_bucket_name`) keeps its own lifecycle rules, so check them the same
  way.
- **Run one ingestion at a time.** The state machine does not stop a second
  execution from starting while one is running, and two runs against the same
  stores race on the doc-status registry and on the graph and vector writes
  (both may index, merge or delete the same documents). See
  [Concurrent runs](../docs/operations.md#concurrent-runs).

### 4. Query from inside the VPC

Both stores accept connections only from inside the VPC, so run `run-rag` as a
one-off task of the same task definition, in the subnets and
security group the state machine uses:

```bash
DEFINITION=$(aws stepfunctions describe-state-machine --region "$REGION" \
  --state-machine-arn "$STATE_MACHINE_ARN" --query definition --output text)
PHASE=$(echo "$DEFINITION" | jq '.States.PrepPhase.Parameters')
CLUSTER=$(echo "$PHASE" | jq -r '.Cluster')
TASK_DEFINITION=$(echo "$PHASE" | jq -r '.TaskDefinition')
SUBNETS=$(echo "$PHASE" | jq -r '.NetworkConfiguration.AwsvpcConfiguration.Subnets | join(",")')
SECURITY_GROUPS=$(echo "$PHASE" | jq -r '.NetworkConfiguration.AwsvpcConfiguration.SecurityGroups | join(",")')

TASK_ARN=$(aws ecs run-task --region "$REGION" --cluster "$CLUSTER" \
  --launch-type FARGATE --task-definition "$TASK_DEFINITION" \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SECURITY_GROUPS],assignPublicIp=DISABLED}" \
  --overrides '{"containerOverrides":[{"name":"app","command":["run-rag","--config-path","/app/config.yaml","--query","What are the main themes?","--search-strategy","global"]}]}' \
  --query 'tasks[0].taskArn' --output text)

aws ecs wait tasks-stopped --region "$REGION" --cluster "$CLUSTER" --tasks "$TASK_ARN"
aws logs tail /graphrag/tasks --region "$REGION" --since 30m
```

The command override must name the CLI explicitly (`run-rag ...`). The answer
is written to the task's log stream (`app/app/<task id>` in
`/graphrag/tasks`).

### 5. Evaluate from inside the VPC

`run-eval` runs the same way. The image holds no dataset and the task's disk is
discarded on exit, so pass `s3://` paths for both: the dataset is downloaded
before the run and the result files are uploaded when it ends. Keep them in
the cache bucket, which the task role can already read and write (and which
the `private`-mode S3 endpoint allows), under a prefix other than the two
expiring ones, for example `eval/`. Reuse the variables from step 4:

```bash
aws s3 cp ./eval_data.json "s3://$CACHE_BUCKET/eval/eval_data.json" --region "$REGION"

OVERRIDES=$(jq -nc \
  --arg data "s3://$CACHE_BUCKET/eval/eval_data.json" \
  --arg out "s3://$CACHE_BUCKET/eval/results/run-001/" \
  '{containerOverrides:[{name:"app",command:["run-eval","--config-path","/app/config.yaml","--eval-data-path",$data,"--outputs-directory",$out,"--search-strategy","auto"]}]}')

TASK_ARN=$(aws ecs run-task --region "$REGION" --cluster "$CLUSTER" \
  --launch-type FARGATE --task-definition "$TASK_DEFINITION" \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SECURITY_GROUPS],assignPublicIp=DISABLED}" \
  --overrides "$OVERRIDES" \
  --query 'tasks[0].taskArn' --output text)

aws ecs wait tasks-stopped --region "$REGION" --cluster "$CLUSTER" --tasks "$TASK_ARN"
aws ecs describe-tasks --region "$REGION" --cluster "$CLUSTER" --tasks "$TASK_ARN" \
  --query "tasks[0].containers[?name=='app'].exitCode | [0]" --output text
aws s3 ls "s3://$CACHE_BUCKET/eval/results/run-001/" --region "$REGION"
```

The container's exit code is `run-eval`'s: `1` when the failure budget
(`--max-failure-rate`) is exceeded, the dataset cannot be downloaded, or any
result file fails to upload; the summary and errors are in the task's log
stream. Uploads use the same `aws.s3.encryption` setting as the cache, so
with `use_cmk=true` the results are encrypted with the CMK. To read a dataset
from another bucket, deploy with `-c corpus_bucket_name=<bucket>`, which grants
the task role read access and, in `private` mode, allows the bucket through the
S3 endpoint; results still go to the cache bucket.

Operational procedures (re-ingesting, failed documents, alarms, caches) are in
the [Operator Runbook](../docs/operations.md).
