# Getting started locally

> 🇰🇷 한국어판: [local.ko.md](./local.ko.md)

Local mode runs ingestion, querying and evaluation on one host without an AWS
account, for development and evaluation (AWS stays the default). By default
the graph store, the vector/lexical store and the registry are SQLite files, so
the only service to run is [Ollama](https://ollama.com) at
`http://localhost:11434`. This page indexes a two-file sample corpus, queries
it with a GraphRAG and a LightRAG strategy and scores the answers without an LLM judge.

Local mode is for development and evaluation only: the embedded stores serve
one host and about 100,000 documents per index, the containers have no
authentication or TLS, and small local models extract weaker graphs. The
[Local mode](../guides/local-mode.md) guide covers the providers, the
differences from AWS mode and the stores in containers.

## Install

You need Python 3.11–3.12, [uv](https://docs.astral.sh/uv/) and Ollama.
Install the `local` extra and pull the two default models:

```bash
uv sync --no-dev --extra local
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
```

`--no-dev` leaves out the contributors' test and lint tools, which `uv sync`
and `uv run` install without it. `uv sync` is exact: a later one removes the
extras it is not given, so add an extra to this command, or pass `--inexact`
([Install](./aws.md#install)).

- **Reranking.** The local preset reranks with a cross-encoder, which needs
  the `cross-encoder` extra: `uv sync --no-dev --extra local --extra cross-encoder`.
  Without it queries keep the fused order (logged once). On Linux that extra
  installs the locked CUDA build of PyTorch (several GB); for the CPU build
  run `uv sync --no-dev --extra local` then
  `uv pip install --torch-backend cpu -e '.[local,cross-encoder]'` (a later
  plain `uv sync` removes it).

## Ingest a sample corpus

Create two short text files (the names, companies and figures are made up):

```bash
mkdir -p sample-corpus
cat > sample-corpus/acme.txt <<'EOF'
Acme Robotics is a manufacturer of warehouse robots based in Springfield.
Alice Chen is the chief engineer at Acme Robotics and leads its motion-control team.
In 2024 Acme Robotics signed a supply agreement with Globex Sensors.
EOF
cat > sample-corpus/globex.txt <<'EOF'
Globex Sensors makes lidar sensors for mobile robots.
Bob Martin runs the sensor division of Globex Sensors.
Under its agreement with Acme Robotics, Globex Sensors supplies the lidar units
for the Acme warehouse robots.
EOF
```

Index it:

```bash
uv run --no-dev run-ingestion --backend local --source-directory ./sample-corpus
```

- **Files.** Without a config file the stores live in `.unified-kg-rag/` of
  the working directory (`graph.sqlite3`, `vector.sqlite3` and the
  registry's `doc-status.sqlite3`), so run every command from the
  same directory, or use a config file with `backend: local` (relative paths
  resolve against its directory). To start over, delete the
  `.unified-kg-rag/` directory or run once with `indexing.reset: true`.
- **Incremental indexing** is on in the preset (the SQLite doc-status
  registry; the run logs its absolute path): a run again over the same
  folder processes only new, changed and failed documents.
  `providers.doc_status.enabled: false` turns it off. Stores built by a run
  with the registry off hold documents the registry does not list, so the
  first run with it on exits 2 ("was built by a run with the doc-status
  registry off"): run it once with `indexing.reset: true` (for example
  `UKR__INDEXING__RESET=true`), which rebuilds the stores and fills the
  registry; later runs are incremental.

## Query with GraphRAG and LightRAG

`local` is the GraphRAG strategy for questions about specific entities and
their relationships; `mix` is the general LightRAG strategy:

```bash
uv run --no-dev run-rag --backend local --search-strategy local --verbose --query "Who is the chief engineer at Acme Robotics?"
uv run --no-dev run-rag --backend local --search-strategy mix --verbose --query "How are Acme Robotics and Globex Sensors related?"
```

`--verbose` prints, after the answer, the query's extracted entities,
every source it used (each numbered with its type, `entity`,
`relationship` or `text`, its name or file, its score and a snippet) and
the retrieval metrics. The commands log every step at `INFO`; prefix
them with `UKR__LOGGING__LEVEL=WARNING` to keep only warnings and the
result.

With the `cross-encoder` extra, the first query downloads the default rerank
model, the cross-encoder `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, from
Hugging Face, so it needs internet access once and takes longer; later
queries use the cached copy. Offline, download it beforehand or set
`providers.rerank.kind: none`
([Local mode](../guides/local-mode.md)).

Without `--search-strategy`, an LLM router picks a strategy per query
(`auto`). [Search strategies](../concepts/retrieval.md#search-strategies--when-to-use-which)
explains when to use which.

## Evaluate without an LLM judge

Write a small dataset. `answer` feeds `answer_match`, `reference_sources`
feeds `retrieval` and `expected_entities` feeds `graph_aware`
([Eval data format](../guides/evaluate.md#eval-data-format)):

```bash
cat > sample-eval.json <<'EOF'
[
  {
    "id": "q1",
    "question": "Who is the chief engineer at Acme Robotics?",
    "answer": "Alice Chen",
    "reference_sources": ["acme.txt"],
    "expected_entities": ["Alice Chen", "Acme Robotics"]
  },
  {
    "id": "q2",
    "question": "Which company supplies the lidar units for the Acme warehouse robots?",
    "answer": "Globex Sensors",
    "reference_sources": ["globex.txt"],
    "expected_entities": ["Globex Sensors", "Acme Robotics"]
  }
]
EOF
```

The three evaluators on by default need no LLM judge. Compare the two
strategies on the same questions:

```bash
uv run --no-dev run-eval --backend local --data-path sample-eval.json --search-strategy local,mix
```

Each strategy's results, reports and summary, and
`evaluation_comparison_<timestamp>.json` (each metric of `mix` paired against
`local`), are written to `outputs/evaluation/run_<timestamp>_<id>/`.
[Evaluators](../guides/evaluate.md#evaluators) and
[Comparing strategies](../guides/evaluate.md#comparing-strategies) explain the
metrics; two questions show the mechanics only, so measure on a few hundred
before you trust a difference.

## Next steps

- [Local mode](../guides/local-mode.md): other model servers, the stores in
  containers, Bedrock models with local stores.
- [Configuration](../guides/configure.md): how the configuration loads and
  the settings most runs change.
- [Getting started on AWS](./aws.md): the same flow on Amazon Neptune,
  Amazon OpenSearch and Amazon Bedrock.
