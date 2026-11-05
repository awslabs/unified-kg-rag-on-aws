# Getting started on AWS

> 🇰🇷 한국어판: [aws.ko.md](./aws.ko.md)

This page takes you from a fresh checkout to a first ingestion and query
against an Amazon Neptune cluster and an Amazon OpenSearch domain you already
have. The framework connects to services that **already exist** and never
creates them. To provision the whole stack instead, use the bundled CDK app in
[`iac/`](../../iac/README.md) ([Quick start](../../iac/README.md#quick-start));
its stores are VPC-only, so the CLIs run there as tasks that receive the
endpoints as environment variables
([After deploy](../../iac/README.md#after-deploy)). To try the framework
without an AWS account, see [Getting started locally](./local.md).

## Prerequisites

You need Python, uv, AWS credentials with Bedrock access, and the services
below.

### Runtime

- **Python 3.11 – 3.12**
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager; `pip` works too)

### AWS services

[Local mode](../guides/local-mode.md) replaces each of them with a local
component.

| Service | Required? | Used for |
|---|---|---|
| **Amazon Bedrock** | Yes | Every LLM call, embeddings and reranking (enable model access for the model ids you configure) |
| **Amazon Neptune** | Yes | The knowledge graph and multi-hop traversal at query time |
| **Amazon OpenSearch** | Yes | Vector and BM25 lexical indices (text units, entities, community reports, relationships, claims) |
| **Amazon S3** | Only for `--s3-sync` or `s3://` paths | Stage-cache sync, `s3://` datasets and the CDK corpus |
| **Amazon DynamoDB** | Only for incremental indexing | The document-status registry that diffs the corpus by content hash |

### Network and versions

- **Reach both stores from the CLI host.** A Neptune cluster takes
  connections only from inside its VPC, so run the CLIs on a host in that
  VPC (or one connected to it through a VPN, peering or a tunnel), with
  security groups that let it reach Neptune's port and the OpenSearch
  domain. A host outside the VPC cannot connect.
- **OpenSearch.** Tested on OpenSearch 2.13, the version the CDK domain
  runs. The opt-in on-disk vector keys need 2.17 or later, and faiss with
  the default `cosinesimil` space 2.19
  ([Operator Runbook: On-disk vector storage](../operations.md#on-disk-vector-storage)).
- **Neptune.** The CDK app does not pin an engine version (a new cluster
  starts on Neptune's default); the client is `gremlinpython` below 3.8, so
  an engine needs TinkerPop below 3.8
  ([`iac/README.md`](../../iac/README.md#configuration-cdkjson-context---c-keyvalue)).

## Required models

Enable access to these Bedrock models (the code defaults; a model you
configure replaces the one in its row) before the first run:

| Role | Default model id | Config key |
|---|---|---|
| Default tier (extraction, gleaning, community reports, answers) | `anthropic.claude-sonnet-5-5` | `models.default` |
| `run-eval` LLM judges (only when an LLM judge runs) | `anthropic.claude-opus-5-5` | `models.roles.evaluation` |
| Fast tier (chunking, translation, summarization, routing, map steps) | `anthropic.claude-haiku-5-5` | `models.fast` |
| Embeddings (indexing, queries and evaluation) | `amazon.titan-embed-text-v2:0` | `models.embedding.model_id` |
| Reranking | `cohere.rerank-v3-5:0` | `models.rerank` |

- **Region.** Every Bedrock call goes to `aws.bedrock.region_name`
  (default: `aws.region_name`, `us-west-2`, which offers all five). Where
  rerank models are missing (for example `ap-northeast-2`), point it at a
  region that has them or set `providers.rerank.kind: none`; a private
  VPC whose only Bedrock route is its VPC endpoint must keep its own region.
- **Inference profiles.** Claude Sonnet 5.5 and Haiku 5.5 are invocable only
  through a cross-region inference profile: keep
  `aws.bedrock.enable_global_profile: true` and grant
  `bedrock:ListInferenceProfiles` (if no profile resolves, model creation
  fails with a message naming this remedy).
- **Permissions.** See [IAM permissions](#iam-permissions); reranking needs a
  statement of its own.

The [Model Catalog](../reference/models.md) lists every curated model and its
limits; [Model selection notes](../reference/models.md#model-selection-notes)
explain the two model tiers and how to change a role's model.

## Install

```bash
git clone https://github.com/awslabs/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws
uv sync --extra eval --extra viz  # or: pip install -e '.[eval,viz]'
```

`uv sync` alone installs ingestion and queries. Two extras add the rest;
without one, the command that needs it stops with exit `2`, naming it:

- **`eval`**: the LangChain and RAGAS judges of `run-eval` (ragas, datasets,
  pandas, langchain-openai), both enabled by default ([Evaluators](../guides/evaluate.md#evaluators)).
  The LLM-free evaluators need no extra.
- **`viz`**: pyvis for `run-visualization`'s interactive graph ([Visualization](../guides/visualize.md)); the static charts need no extra.

**Markdown and HTML** need the `unstructured` extra
(`uv sync --extra unstructured`; it requires `unstructured>=0.24.1`, which
fixes URL-partitioning SSRF and unescaped markup). Without it, only `.pdf`,
`.txt`, `.csv` and `.json` are parsed and ingestion skips `.md` / `.html` files with one warning
per extension.

The extra tokenizes with a spaCy model (`en_core_web_sm`) that it downloads
from GitHub at the first Markdown or HTML file. Without network access,
install it beforehand, once, where the network is reachable:
`python -c "from unstructured.nlp.tokenize import sent_tokenize; sent_tokenize('Ready.')"`.
Otherwise the first such file waits up to 120 s and fails, and every later
one of the run fails at once with the same error.

The deployed image (`docker/Dockerfile`) leaves the extra out (it adds about
200 MB), so a Step Functions ingest skips `.md` / `.html` files with a
warning, and a corpus of only such files fails with "No supported source
files found in '<dir>'": convert the files, or build with
`--build-arg UV_EXTRAS="--extra unstructured"`, which also installs the spaCy
model in the image (a task in the private VPC cannot download it). The image's
`docker/config.yaml` holds no endpoints; the CDK task definitions inject them.

**OWL/RDFS ontologies** as the extraction schema need the `ontology` extra
(`uv sync --extra ontology`; [Ontology](../guides/ontology.md#install)).

## Authentication

1. **AWS credentials** come from the standard credential chain:
   `aws.profile_name` names a profile, `null` uses the default chain (env
   vars, instance role, ...). Neptune signs with SigV4 when
   `aws.neptune.use_iam: true`.
2. **OpenSearch** uses IAM (`aws.opensearch.use_iam: true`) or, with
   `use_iam: false` (the default), a username and password in a `.env` file
   (copy `.env-template`), which the `run-*` CLIs load:

   ```bash
   UKR__AWS__OPENSEARCH__USERNAME=your_opensearch_username
   UKR__AWS__OPENSEARCH__PASSWORD=your_opensearch_password
   ```

## IAM permissions

The principal running the CLIs needs Bedrock (`InvokeModel`,
`InvokeModelWithResponseStream` and `CountTokens` for your model ids and
embeddings; without `CountTokens` token counts are estimated), Neptune
(connect / SigV4 with `use_iam: true`), OpenSearch (read and write the
configured indices), S3 (the configured bucket) and DynamoDB (with incremental
indexing). Three Bedrock actions need statements of their own:

- **`bedrock:Rerank` on `Resource: "*"`.** The Rerank API authorizes against
  another resource shape than `InvokeModel`, so a statement scoped to model or
  inference-profile ARNs denies it (the CDK task role grants it separately).
  Without it every query logs `Reranking failed: ...` at ERROR and returns
  the fused results unreranked (`metadata.rerank.status: failed`); if you
  cannot grant it, set `providers.rerank.kind: none`.
- **`bedrock:ListInferenceProfiles` on `Resource: "*"`**, an account-level
  read the model resolver makes.
- **`bedrock:ApplyGuardrail`** on the guardrail's ARN
  (`arn:aws:bedrock:<bedrock region>:<account>:guardrail/<id>`), when a
  [guardrail](../guides/guardrail.md) is set.

## First run

Copy the template, then set the two store endpoints in `config.yaml`
(`aws.neptune.endpoint`, `aws.opensearch.endpoint`) and `aws.region_name`
when the stores are not in `us-west-2`. For a domain that authorizes IAM
(SigV4) requests, also set `aws.opensearch.use_iam: true`; the default
`false` uses the username and password from `.env`
([Authentication](#authentication)):

```bash
cp config-template.yaml config.yaml
```

`uv run` runs the CLIs in the project environment; with `pip`, activate the
virtual environment instead and drop the `uv run` prefix. Index a corpus,
then query it with either methodology:

```bash
uv run run-ingestion --source-directory ./source --config-path config.yaml
uv run run-rag --query "What are the main themes?" --search-strategy global --config-path config.yaml
uv run run-rag --query "How are Alice and Acme related?" --search-strategy mix --config-path config.yaml
```

## Next steps

- [Configuration](../guides/configure.md): how the configuration loads, the
  settings most runs change, environment variables and what the logs contain.
- [Ingestion](../guides/ingest.md) and [Querying](../guides/query.md):
  stages, resume and rebuild, filters; which strategy to use is in
  [Retrieval](../concepts/retrieval.md).
- [Incremental indexing](../guides/incremental.md): re-index
  only new and changed documents.
- [`iac/README.md`](../../iac/README.md#quick-start): deploy the whole stack
  with AWS CDK instead.
