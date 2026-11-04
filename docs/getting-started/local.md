# Getting started locally

> 🇰🇷 한국어판: [local.ko.md](./local.ko.md)

Local mode runs ingestion, querying and evaluation on one host without an AWS
account, for development and evaluation (AWS stays the default). By default
the graph store, the vector/lexical store and the registry are SQLite files, so
the only service to run is [Ollama](https://ollama.com) at
`http://localhost:11434`. This page indexes a two-file sample corpus, queries
it with both methodologies and scores the answers without an LLM judge.

Local mode is for development and evaluation only: the embedded stores serve
one host and about 100,000 documents per index, the containers have no
authentication or TLS, and small local models extract weaker graphs. The
[Local mode](../guides/local-mode.md) guide covers the providers, the
differences from AWS mode and the stores in containers.

## Install

You need Python 3.11–3.12, [uv](https://docs.astral.sh/uv/) and Ollama.
Install the `local` extra and pull the two default models:

```bash
uv sync --extra local
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
```

- **CPU-only PyTorch.** On Linux `uv sync --extra local` installs the locked
  CUDA build (several GB). For the CPU build run `uv sync` then
  `uv pip install --torch-backend cpu -e '.[local]'` (a later plain `uv sync`
  removes it).

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
uv run run-ingestion --backend local --source-directory ./sample-corpus
```

- **Files.** Without a config file the three stores live in
  `.unified-kg-rag/` of the working directory, so run every command from the
  same directory, or use a config file with `backend: local` (relative paths
  resolve against it). To start over, delete the three files together or run
  once with `indexing.reset: true`.
- **Incremental indexing** is off in the preset; turn it on with
  `providers.doc_status.enabled: true` (the run logs the registry's absolute
  path).

## Query with both methodologies

`local` is the GraphRAG strategy for questions about specific entities and
their relationships; `mix` is the general LightRAG strategy:

```bash
uv run run-rag --backend local --search-strategy local --query "Who is the chief engineer at Acme Robotics?"
uv run run-rag --backend local --search-strategy mix --query "How are Acme Robotics and Globex Sensors related?"
```

Without `--search-strategy`, an LLM router picks a strategy per query
(`auto`). [Search strategies](../user-guide.md#search-strategies--when-to-use-which)
explains when to use which.

## Evaluate without an LLM judge

Write a small dataset. `answer` feeds `answer_match`, `reference_sources`
feeds `retrieval` and `expected_entities` feeds `graph_aware`
([Eval data format](../user-guide.md#eval-data-format)):

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

All five evaluators are on by default, and two of them call an LLM judge. Keep
only the three that need no LLM (here through an environment variable, see
[Environment variable overrides](../guides/configure.md#environment-variable-overrides))
and compare the two strategies on the same questions:

```bash
UKR__EVALUATION__ENABLED_EVALUATORS='["answer_match", "retrieval", "graph_aware"]' \
  uv run run-eval --backend local --data-path sample-eval.json --search-strategy local,mix
```

Each strategy's results, reports and summary, and
`evaluation_comparison_<timestamp>.json` (each metric of `mix` paired against
`local`), are written to `outputs/evaluation/`.
[Evaluators](../user-guide.md#evaluators) and
[Comparing strategies](../user-guide.md#comparing-strategies) explain the
metrics; two questions show the mechanics only, so measure on a few hundred
before you trust a difference.

## Next steps

- [Local mode](../guides/local-mode.md): other model servers, the stores in
  containers, Bedrock models with local stores.
- [Configuration](../guides/configure.md): how the configuration loads and
  the settings most runs change.
- [Getting started on AWS](./aws.md): the same flow on Amazon Neptune,
  Amazon OpenSearch and Amazon Bedrock.
