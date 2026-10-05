# Unified Knowledge Graph RAG on AWS

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)

🇰🇷 **[한국어 README](./README.ko.md)** · 🤝 **[Contributing](./CONTRIBUTING.md)**

<p align="center">
  <img src="./assets/profile.png" alt="Unified Knowledge Graph RAG on AWS" width="320">
</p>

An AWS-native knowledge graph RAG (Retrieval-Augmented Generation) framework that turns large, multilingual document corpora into a knowledge graph on Amazon Neptune and Amazon OpenSearch Service and answers questions over it with Amazon Bedrock, using multi-hop graph traversal to reason across document boundaries. It reimplements Microsoft GraphRAG and LightRAG on one shared stack, so you choose the retrieval methodology per query rather than per deployment. Running both methodologies on one stack, triple-hybrid search, incremental indexing, and multilingual processing are deliberate enhancements over the source papers.

Read the [AWS Open Source Blog introduction](https://aws.amazon.com/ko/blogs/opensource/unified-knowledge-graph-rag-on-aws-graphrag-and-lightrag-on-one-stack/) for an overview of the architecture, both retrieval methodologies, and benchmark results.

## Why this framework

- **Two methodologies on one AWS stack.** GraphRAG (community summaries) and LightRAG (dual-level keywords) share ingestion, indexing, caching, and search infrastructure; only the retrieval algorithm differs, selected per query. Every answer returns the sources that were actually placed in the model's context.
- **Triple-hybrid search.** BM25 lexical search, vector semantic search, and Neptune graph traversal are fused with Reciprocal Rank Fusion (RRF) and reranked with a Bedrock rerank model.
- **Incremental indexing.** With `aws.dynamodb` enabled, a content-hash registry re-indexes only new or changed documents and merges them into the live graph. Deleting a document removes only the artifacts no other document shares.
- **Multilingual.** Optional translation at indexing and query time, per-language OpenSearch analyzers (for example `nori` for Korean), and multilingual keyword extraction, for both methodologies.
- **Prompt tuning.** `run-prompt-tuning` profiles a sample of your corpus (domain, language, persona, entity types) and writes domain-adapted `custom_prompts`.
- **Graph-aware evaluation and standalone visualization.** `run-eval` adds deterministic entity and relationship coverage to LangChain and RAGAS metrics, and `run-visualization` renders an exported graph without re-ingesting.
- **Pluggable hexagonal design.** Storage and model backends sit behind ports; search strategies, evaluators, and renderers register through registries, so you extend the framework without editing dispatch code.

## Architecture

![Data Ingestion Pipeline](./assets/ingestion_pipeline.png)

Ingestion is a resumable 12-stage pipeline: parse, load, chunk, optionally translate, extract entities and relationships with an LLM, optionally glean, resolve duplicates, optionally extract claims, compute graph metrics, detect Leiden communities with LLM-written reports, and index into OpenSearch and Neptune. Stage checkpoints are cached locally and can be synced to S3.

![Retrieval Pipeline](./assets/retrieval_pipeline.png)

Every strategy runs through the same hybrid scorer and token budget. Pick one with `--search-strategy` (CLI) or `RAGInput.search_strategy` (Python):

| Strategy | Methodology | Use when |
|---|---|---|
| `auto` (default) | GraphRAG | You are not sure; an LLM router picks `simple`, `local`, `global`, or `drift` per query |
| `simple` | GraphRAG | Fast factual lookups; vector + keyword search without graph traversal |
| `local` | GraphRAG | Questions about specific entities and their relationships |
| `global` | GraphRAG | Broad or thematic questions, answered by map-reduce over community reports |
| `drift` | GraphRAG | Complex questions that need iterative exploration |
| `mix` | LightRAG | General LightRAG use; entity and relationship keyword retrieval plus chunk retrieval |
| `hybrid` | LightRAG | Keyword-driven graph questions, without the extra chunk retrieval |
| `naive` | LightRAG | A fast vector-only baseline and comparison runs |

See the [User Guide §4](./docs/user-guide.md#4-querying-run-rag) for per-strategy behavior and the [Design Doc](./docs/design.md) for the layer map, algorithms, and data model.

## Quickstart

### Prerequisites

- Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/) (`pip` also works).
- AWS credentials with access to Amazon Bedrock (model access enabled for the models you configure), an Amazon Neptune cluster, an Amazon OpenSearch Service domain, an S3 bucket, and, for incremental indexing only, DynamoDB. The framework connects to existing services; to create them, see [Deploy on AWS](#deploy-on-aws-optional).

### Install and configure

```bash
git clone https://github.com/awslabs/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws
uv sync                                 # or: pip install -e .
cp config-template.yaml config.yaml     # then set your Bedrock region and service endpoints
```

If OpenSearch uses username/password instead of IAM (`aws.opensearch.use_iam: false`), copy `.env-template` to `.env` and fill in the credentials. Parsing Markdown and HTML needs the optional `unstructured` extra (Python 3.11+: `uv sync --extra unstructured`); PDF, TXT, CSV, and JSON work out of the box.

### Ingest, query, evaluate

```bash
# Index a corpus (incremental when aws.dynamodb is enabled)
run-ingestion --source-directory ./source --config-path config.yaml

# Query with either methodology, or chat with conversation memory
run-rag --query "What are the main themes?" --search-strategy global --config-path config.yaml
run-rag --query "How are Alice and Acme related?" --search-strategy mix --config-path config.yaml
run-rag --interactive --use-memory --conversation-id my-session --config-path config.yaml

# Evaluate (LangChain + RAGAS + graph-aware coverage)
run-eval --eval-data-path eval_data.json --config-path config.yaml

# Optional: render an exported graph, or tune prompts to your domain
run-visualization --data-path visualization_data.json --output-dir ./viz --config-path config.yaml
run-prompt-tuning --source-directory ./source --output tuned_prompts.yaml --config-path config.yaml
```

The graph-aware evaluator reports entity and relationship coverage (recall) against `expected_entities` / `expected_relationships`, without an LLM. It matches on word boundaries for space-delimited scripts and falls back to substring matching for CJK text. The [User Guide](./docs/user-guide.md) covers every configuration section, CLI flag, and the evaluation data format.

## Deploy on AWS (optional)

The [`iac/`](./iac/README.md) AWS CDK app provisions the whole stack: a VPC with endpoints, Neptune, OpenSearch, the DynamoDB document-status table, an S3 cache bucket, an ECS Fargate data plane, a Step Functions ingestion pipeline, CloudWatch dashboards and alarms, and an optional Bedrock Guardrail.

```bash
cd iac && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cdk synth          # preview only: no AWS changes, no cost
cdk bootstrap      # once per account and region
cdk deploy --all   # creates billable resources
```

Copy the Neptune, OpenSearch, and S3 values from the stack outputs into `config.yaml`.

> **Cost and teardown.** Neptune and OpenSearch bill hourly, and `public` network mode adds NAT gateways. The default `dev` environment tears down with `cdk destroy --all`; non-dev environments default to retaining stateful stores with deletion protection on. Review the hardening options (`use_cmk`, `deletion_protection`, Multi-AZ sizing, `vpc_flow_logs`) in [`iac/README.md`](./iac/README.md) before production use.

## Documentation

| Document | Covers |
|---|---|
| [User Guide](./docs/user-guide.md) ([한국어](./docs/user-guide.ko.md)) | Configuration, CLI flags, incremental indexing, evaluation, operations |
| [Design Doc](./docs/design.md) ([한국어](./docs/design.ko.md)) | Hexagonal architecture, algorithms, data model, extension guide, further reading |
| [`iac/README.md`](./iac/README.md) | CDK stacks, `-c key=value` options, Guardrail deployment, Step Functions ingestion runs |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Development setup, tests, quality gate, extension recipes |
| [CHANGELOG.md](./CHANGELOG.md) | Release notes |
| [SECURITY.md](./SECURITY.md) | Reporting security issues |

## Security & Disclaimer

This project is a **reference framework provided for educational and illustrative purposes**. It is offered "AS IS" without warranty of any kind (see [LICENSE](LICENSE)). **It should not be deployed to a production environment without your own additional security testing, threat modeling, and hardening.**

- Run it in your own AWS account against your own resources; you are responsible for IAM policies, network configuration, data classification, and end-user authentication in your deployment.
- The optional CDK stack (`iac/`) provides secure defaults (private-VPC isolation, encryption at rest with AWS-managed keys or an optional customer-managed KMS key via `use_cmk`, enforced TLS, least-privilege IAM, an optional Bedrock Guardrail for PII and prompt-attack filtering), but you own the deployment and should review it for your environment.
- Enable the Bedrock Guardrail (`aws.bedrock.guardrail`) and apply rate limiting and monitoring appropriate to your use case before production use.
- To report a security issue, see [SECURITY.md](SECURITY.md); please do not open a public issue.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and extension recipes; most extensions are a registry registration and need no dispatch-code edits.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details. Maintained by AWS under the awslabs organization.

## Acknowledgments

Thanks to **Jihyeon Kang** for significant contributions to the library's feature development and validation, and to **Yusuke Tanimiya** for a thorough review and real-corpus testing — both of which strengthened the framework ahead of release.

## References

- Microsoft GraphRAG: [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) · [library](https://github.com/microsoft/graphrag)
- LightRAG: [Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779) · [library](https://github.com/HKUDS/LightRAG)

Further reading on GraphRAG (DRIFT search, dynamic community selection, auto-tuning, LazyGraphRAG) is listed in the [Design Doc](./docs/design.md#16-further-reading).
