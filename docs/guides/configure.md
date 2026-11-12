# Configuration

> 🇰🇷 한국어판: [configure.ko.md](./configure.ko.md)

Copy the template and point every CLI at it with `--config-path config.yaml`:

```bash
cp config-template.yaml config.yaml
```

A config file holds only the keys you change; the others keep their
defaults. Every key, its type, default and description is in the generated
[Configuration Reference](../reference/configuration.md); this page covers what needs
more than a table row. Model choice is in the Model Catalog's
[Model selection notes](../reference/models.md#model-selection-notes), query
and answer screening in [Guardrail](./guardrail.md), and tracing in
[Observability](./observability.md).

## How configuration is loaded

Each layer overrides the one before:

1. **Built-in defaults** (`unified_kg_rag/domain/models/config/`). Without
   `--config-path`, a CLI runs on these plus the environment.
2. **The `backend` preset**: `backend: local` (or `--backend local`) fills in
   the local providers and stores for every key the file and the environment
   leave unset ([Local mode](./local-mode.md)).
3. **Your YAML file**. A dict-valued key (such as `logging.library_levels`)
   replaces the whole default dict, it does not merge.
4. **Environment variables** ([below](#environment-variable-overrides)).
5. **Provider flags** (`--backend`, `--llm-provider`, ...), applied last. The
   config is rebuilt from what the file and the environment set, so switching
   `--backend` drops only what the old preset filled in, never a value you set.

The file is validated when it loads. A wrong type or value
(`Configuration validation error: ...`), invalid YAML, a missing
`--config-path` file and an unknown key (in the file or as a `UKR__` variable)
are usage errors (exit 2). Unknown keys are refused because a misspelled key
silently keeps its default (`Config.model_validate` in Python refuses them
too); `--allow-unknown-keys` (`allow_unknown_keys=True` for `get_config`) logs
a WARNING per key and drops it instead.
A moved key fails naming its replacement
([Moved keys](../reference/configuration.md#moved-keys)).

Retries are configured in one section, `retry`: `max_attempts` for model
calls (ingestion items, Bedrock embeddings and query-time calls) with its
backoff, and an attempt count for the callers whose defaults differ (model
servers, the graph-store indexer, OpenSearch queries, RAGAS judges). Every
attempt count includes the first attempt; `1` disables the retry.

## Settings most runs change

| Key | Default | Change it when |
|---|---|---|
| `aws.region_name` | `"us-west-2"` | Stores in another region (Bedrock follows unless `aws.bedrock.region_name` is set) |
| `aws.neptune.endpoint`, `aws.opensearch.endpoint` | `null` | Always on AWS (the CDK stack injects them) |
| `providers.doc_status.enabled` | `false` | [Incremental indexing](../guides/incremental.md) |
| `models.default`, `models.fast` | Claude Sonnet 5.5, Claude Haiku 5.5 | Other models ([notes](../reference/models.md#model-selection-notes)) |
| `processing.max_concurrency` | `40` | Lower on `ThrottlingException` retries, raise if quota allows |
| `processing.graph_extraction.entity_types` | 7 generic types | [Domain adaptation](./domain-adaptation.md#entity-types) |
| `processing.translation.target_language` | `"en"` | Another corpus or answer language ([Multilingual](./multilingual.md)) |
| `processing.claim_extraction.enabled` | `false` | Claims in `local` search (one more LLM call per text unit) |
| `processing.document_parsing.index_value` | `null` (`default`) | Separate corpora or tenants ([Index suffix](../concepts/scopes-and-namespaces.md#index-suffix)) |
| `indexing.methodologies` | `[graphrag, lightrag]` | Build only what one methodology reads: `[lightrag]` makes no community report call ([Methodology profiles](./ingest.md#methodology-profiles)) |
| `graph.community_detection.enabled` | `true` | `false` builds no communities, as `indexing.methodologies: [lightrag]` does (GraphRAG's strategies need them) |
| `providers.rerank.kind` | `bedrock` | `none`: no rerank model in the region, or no reranking wanted |
| `search.token_manager.max_context_tokens` | `30000` | More (or cheaper, less) context ([Context budget](../concepts/retrieval.md#context-budget)) |
| `indexing.reset` | `false` | One run that rebuilds the stores from the whole corpus (quarantined documents skipped) |
| `aws.bedrock.guardrail.identifier` | `null` | Screen queries and answers ([Guardrail](./guardrail.md)) |
| `observability.enabled` | `false` | OpenTelemetry traces ([Observability](./observability.md)) |

## Environment variable overrides

Any config key can be set from the environment as `UKR__` followed by its
path, with double underscores between the segments, in any case:
`UKR__AWS__NEPTUNE__ENDPOINT` sets `aws.neptune.endpoint`, and a segment past
a free-form mapping names one of its keys
(`UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`). A variable wins over the
config file and the defaults. Values are converted and validated as the same
value in the file would be (`true`/`1`/`yes`/`on` or `false`/`0`/`no`/`off`,
in any case, for booleans; JSON for a list or mapping, e.g. `UKR__SEARCH__AUTO_ROUTABLE_STRATEGIES='["local", "mix"]'`).
An invalid value, or a `UKR__` variable that names no key, is a usage error
(exit 2) naming the variable; `--allow-unknown-keys` turns the unknown name
into a WARNING.

- `AWS_PROFILE` and `AWS_REGION` fill `aws.profile_name` and
  `aws.region_name` only when the file leaves them unset, so a shell that
  exports `AWS_REGION` (SSO helpers and CloudShell do) does not move a file's
  region; `AWS_REGION` then also moves Bedrock calls while
  `aws.bedrock.region_name` is `null`. `AWS_DEFAULT_REGION` is not read.
- `UKR_PLUGINS` names the plugin modules a CLI loads instead of the
  installed `unified_kg_rag.plugins` entry points
  ([Loading plugins](./extend.md#loading-plugins)).
- `UKR_SOURCE_DIRECTORY`, `UKR_PIPELINE_ID` (one underscore: not config
  keys) and `UKR_METRICS_ENVIRONMENT` are read by `run-ingestion`
  ([CLI Reference](../reference/cli.md#run-ingestion)). The variables the CDK tasks get
  are listed in [`iac/README.md`](../../iac/README.md#2-build-and-push-the-app-image).
- The CLIs load `./.env` from the current working directory (python-dotenv),
  never one from a parent directory (a `~/.env` setting `AWS_PROFILE` would
  otherwise switch accounts unnoticed); a variable already set wins. Importing
  the package loads no `.env` and never changes `os.environ`: a Python
  caller loads one itself (`dotenv.load_dotenv()` before `get_config()`).

> **LangSmith tracing uploads content.** With `LANGSMITH_TRACING=true` (or
> `LANGCHAIN_TRACING_V2=true`), LangChain sends every traced run to
> LangSmith, prompts, retrieved text and model outputs included. The CLIs log
> a WARNING at startup (from `setup_logging`) but leave it on; unset the
> variable before running on a confidential corpus.

## What the logs contain

At `INFO` and above, user and corpus text is logged only as lengths, counts,
ids and a stable short hash (`query: len=42 sha=1a2b3c4d`). Query text,
rewritten queries, entity names and raw model output are logged only at
`DEBUG`, and exception messages never carry model output. Treat `DEBUG` logs
(`logging.level: DEBUG`, `UKR__LOGGING__LEVEL=DEBUG`, any CLI's `--log-level
DEBUG`; `run-rag --verbose` prints more
of each result but leaves the log level), and libraries set to `DEBUG` in
`logging.library_levels` (`botocore` logs request bodies), as containing user
and corpus data.

Every CLI takes `--log-level` (over `logging.level`) and `--version`. With
`--output-format text` the console logs are plain lines unless
`logging.log_format` is set (the CDK tasks set `structured`, JSON records for
CloudWatch); with `--output-format json` they are JSON records on stderr.

With `logging.log_to_file: true` the CLIs also write `logging.log_file_path`
(`logs/log.txt`, dated `log_YYYYMMDD.txt`, relative to the working directory,
which must be writable); it is off by default. As a library the
package configures no handler: the host's logging applies, with the
structured fields (`pipeline_id`, `trace_id`, ...) as record attributes.

## S3 cache encryption

`aws.s3.encryption.encryption_type: BUCKET_DEFAULT` sends no encryption
header, so the bucket default applies (the CMK with the CDK stack's
`use_cmk=true`; forcing `AES256` bypasses it). On a reused SSE-KMS bucket the
writer needs `kms:GenerateDataKey` and `kms:Decrypt` on its key.
