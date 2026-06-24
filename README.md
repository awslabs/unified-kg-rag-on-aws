# AWS Native Graph RAG

📖 **[한국어 README](./README.ko.md)** · 🤝 **[Contributing](./CONTRIBUTING.md)**

![Knowledge Graph](./assets/interactive_graph.jpg)

A production-ready, AWS-native knowledge graph RAG (Retrieval-Augmented Generation) framework that transforms large-scale multilingual documents into dynamic knowledge graphs, enabling intelligent question-answering with complex multi-hop reasoning capabilities.

Built from the ground up based on Microsoft's "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" paper and subsequent research, this framework is specifically designed to leverage AWS native services for enterprise-scale deployment.

> **What's new**
> - **Two retrieval methodologies, one infrastructure.** Choose per query via `search_strategy`: GraphRAG community-summary (`auto`/`drift`/`global`/`local`/`simple`) or **LightRAG dual-level keyword** (`mix`/`hybrid`/`naive`). Both share the same ingestion, indexing, caching, multilingual translation, and hybrid (lexical + semantic + graph) scoring — only the retrieval algorithm differs.
> - **Incremental indexing.** Enable the DynamoDB document-status registry (`aws.dynamodb`) to diff a corpus by content hash and re-index only new/changed documents, merging into the live graph (idempotent upserts; deletions remove a document's exclusive artifacts).
> - **Prompt tuning.** `run-prompt-tuning` profiles a corpus (domain/language/persona/entity-types) and emits domain-adapted `custom_prompts`.
> - **Standalone visualization & graph-aware evaluation.** `run-visualization` renders from exported graph data without re-ingesting; the `graph_aware` evaluator scores entity/relationship coverage.
> - **Hexagonal architecture.** Ports & adapters (`domain/ports/adapters/application` layers) + registries make storage backends, retrieval strategies, evaluators, and renderers pluggable. See `docs/tech-doc.md`.

## 📋 Table of Contents

- [✨ Features & Advantages](#-features--advantages)
- [🏛️ Architecture Overview](#️-architecture-overview)
- [🚀 Installation](#-installation)
- [📖 Usage](#-usage)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📚 References](#-references)

## ✨ Features & Advantages

### 🏗️ **AWS-Native Design**
- **AWS Service Integration**: Seamless integration with Bedrock, Neptune, OpenSearch, S3, and other AWS services
- **Scalable**: Parallel processing architecture supporting large-scale document processing
- **Cost-Optimized**: Intelligent caching and S3 synchronization for cost optimization with elastic task retry
- **Enterprise Security**: S3 encryption and private VPC-based enterprise-grade security implementation

### 🚀 **Triple Hybrid Search Architecture**
- **Semantic Search**: High-quality vector search based on Amazon Bedrock embedding models
- **Lexical Search**: Precise keyword matching using BM25 algorithm
- **Relationship-based Search**: Connectivity analysis through knowledge graph traversal
- **Result Optimization**: Enhanced search accuracy with RRF algorithm and Amazon Bedrock reranking models

### 🧠 **Advanced Knowledge Graph Processing**
- **Precise Entity Resolution**: Automatic detection and integration of duplicate entities
- **Topic Clustering**: Efficient community detection based on Leiden algorithm
- **Complex Reasoning**: Multi-hop reasoning capabilities across document boundaries
- **Source Transparency**: Verifiable information sources provided for all responses

### 🔍 **Two Selectable Methodologies, One Infrastructure**
Pick per query via `search_strategy`; both share the same ingestion, indexing,
caching, multilingual, and hybrid-scoring stack — only the retrieval algorithm differs.
- **GraphRAG (community-summary)**: `simple` (direct), `local` (entity-focused),
  `global` (community-based), `drift` (progressive exploration), `auto` (LLM router)
- **LightRAG (dual-level keyword)**: `mix`, `hybrid`, `naive` — high/low keyword
  extraction over an entity index + a relationship vector index + graph expansion

### ♻️ **Incremental Indexing**
- **Content-hash delta detection**: a DynamoDB document-status registry re-indexes
  only new/changed documents and merges into the live graph (idempotent upserts)
- **Deletion lineage**: removing a document deletes only its *exclusive* artifacts

### 🎯 **Comprehensive Evaluation Framework**
- **LangChain-based Evaluation**: RAG performance measurement through built-in evaluators
- **RAGAS Metrics**: Answer faithfulness, relevancy, and context accuracy
- **Graph-aware Evaluation**: entity/relationship coverage (recall of expected
  graph artifacts surfaced in the answer) against ground-truth expectations
  (deterministic, LLM-free, word-boundary matching)

### 🔧 **User Support**
- **Domain-specific Prompts**: customizable per-prompt overrides via config
- **Automatic Prompt Tuning**: profile a corpus (domain/language/persona/entity-types)
  and emit domain-adapted prompts (`run-prompt-tuning`)
- **Flexible Configuration**: detailed option adjustment through YAML configuration files
- **Comprehensive Monitoring**: structured logging and performance metrics

### 🌍 **Multilingual Support**
- **Automatic Language Processing**: translation during indexing/search, language-aware
  analyzers, and multilingual keyword extraction — applied to both methodologies

### 📊 **Visualization & Analytics Tools**
- **Interactive Graph**: Node2Vec + UMAP graph visualization
- **Network Analysis**: centrality metrics and graph statistics
- **Standalone CLI**: render from exported graph data without re-ingesting (`run-visualization`)

### 🧱 **Hexagonal Architecture**
- **Ports & adapters**: pluggable storage/retrieval backends and a registry for
  strategies, evaluators, and renderers — extend without editing dispatch code (see `CLAUDE.md`)

## 🏛️ Architecture Overview

The framework implements a sophisticated indexing and retrieval pipeline:

### Data Ingestion Pipeline
![Data Ingestion Pipeline](./assets/ingestion_pipeline.png)

#### Core Stages:
- **Document Loading/Parsing**: Multi-format support (PDF, TXT, MD, CSV, JSON)
- **Text Chunking**: Simple/intelligent strategies with context preservation
- **Graph Extraction**: Entity/relationship extraction via LLM
- **Graph Resolution**: Fuzzy matching and deduplication of entities/relationships
- **Graph Analysis**: Centrality metrics (degree, betweenness, PageRank) and graph statistics
- **Community Detection**: Leiden algorithm for topic clustering
- **Indexing**: Storage backend integration (OpenSearch + Neptune)

#### Optional Stages:
- **Translation**: Multi-language support with automatic language detection
- **Gleaning**: Iterative graph refinement for improved accuracy
- **Claim Extraction/Resolution**: Factual assertions extraction and validation
  (opt-in: `processing.claim_extraction.enabled`, off by default — claims are
  indexed but not yet consumed by retrieval)

#### Key Features:
- **Incremental Indexing**: content-hash delta detection + merge (DynamoDB registry)
- **Resumable Pipeline**: Stage checkpointing for interrupted runs
- **Comprehensive Caching**: S3 sync with local cache management
- **Parallel Processing**: Batch optimization and concurrent execution
- **Configurable Strategies**: Flexible processing approaches per stage
- **Error Handling**: Optional continuation on stage failures

### Retrieval Pipeline
![Retrieval Pipeline](./assets/retrieval_pipeline.png)

#### Multi-Strategy Architecture

The framework offers two retrieval methodologies — GraphRAG community-summary
and LightRAG dual-level keyword — sharing one ingestion/indexing/caching/
hybrid-search infrastructure and selectable per query via
`RAGInput.search_strategy`. The GraphRAG `auto` strategy automatically selects
the optimal approach based on query analysis.

##### GraphRAG strategies

**Simple Strategy**: Direct OpenSearch retrieval for basic queries
- Vector and keyword search without graph traversal
- Fastest response time for straightforward questions
- Ideal for factual lookups and simple information retrieval

**Local Strategy**: Entity-focused search using graph traversal + text retrieval
- Identifies key entities in the query
- Performs graph traversal to find related entities and relationships
- Combines graph context with vector/keyword search results
- Optimal for detailed analysis of specific entities or concepts

**Global Strategy**: Community-based analysis for broad questions
- Leverages community detection results for comprehensive coverage
- Uses map-reduce approach for large-scale information synthesis
- Dynamic community selection based on query relevance
- Best for high-level insights and thematic analysis

**Drift Strategy**: Iterative query evolution with convergence detection
- Starts with initial search and iteratively refines based on results
- Expands context through multiple search rounds
- Convergence detection prevents infinite loops
- Excellent for complex, multi-faceted questions requiring exploration

##### LightRAG strategies (dual-level keyword)

**Mix / Hybrid Strategy**: Extracts high-level and low-level keywords
(`KeywordsExtractionPrompt`), then queries the relationship vector index
(high-level) + entity index (low-level) with Neptune neighbourhood expansion.
`mix` additionally blends naive vector chunk retrieval. Both run through the
same `HybridScorer` (lexical + semantic + graph, RRF + Bedrock rerank).

**Naive Strategy**: Pure vector chunk retrieval — the LightRAG baseline, useful
as a fast lexical/semantic fallback and for comparison evaluation.

#### Component Architecture

**Dual Retriever System**:
- **Neptune Graph DB**: Relationship traversal and entity-centric search
- **OpenSearch**: Vector similarity and keyword matching with BM25

**Query Processing Pipeline**:
- Language detection and translation (if needed)
- Entity extraction using LLM
- Strategy selection based on query characteristics
- Multi-retriever coordination and result fusion

**Fusion and Ranking Mechanisms**:
- **RRF (Reciprocal Rank Fusion)**: Combines scores from different retrievers
- **Diversity Filtering**: Reduces redundancy in search results
- **LLM Reranking**: Context-aware result prioritization using Bedrock models
- **Hybrid Scoring**: Weighted combination of lexical and semantic similarity

**Context Optimization**:
- **Token Management**: Dynamic context sizing within model limits
- **Priority Scoring**: Relevance-based content selection
- **Memory Integration**: Conversational context tracking for multi-turn queries
- **Entity Tracking**: Maintains entity focus across conversation turns

## 🚀 Installation

### Prerequisites
- **Python 3.10+** with pip package manager
- **AWS CLI** configured with appropriate permissions
- **AWS Services** deployed and accessible:
  - Amazon Bedrock (with model access enabled)
  - Amazon Neptune cluster
  - Amazon OpenSearch domain
  - Amazon S3 bucket

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd aws-graphrag

# Install the framework
pip install -e .

# Copy and configure settings
cp config-template.yaml config.yaml
# Edit config.yaml with your AWS service endpoints

# Copy and configure environment variables (if using username/password authentication)
cp .env-template .env
# Edit .env file with your OpenSearch credentials if not using IAM authentication
```

### Environment Configuration

If your OpenSearch cluster uses username/password authentication instead of IAM, create a `.env` file:

```bash
cp .env-template .env
```

Then edit the `.env` file with your OpenSearch credentials:

```bash
# OpenSearch Authentication (only required if use_iam is false in config.yaml)
OPENSEARCH_USERNAME=your_opensearch_username
OPENSEARCH_PASSWORD=your_opensearch_password
```

**Note**: The `.env` file is only needed when `use_iam: false` is set in your `config.yaml` OpenSearch configuration. If you're using IAM authentication (`use_iam: true`), you can skip this step.

## 📖 Usage

### Configuration
Create a `config.yaml` file based on the provided `config-template.yaml`:

```bash
cp config-template.yaml config.yaml
```

Edit the configuration file with your AWS service endpoints and settings:

```yaml
# AWS Configuration
aws:
  region_name: "us-east-1"  # AWS region for services
  profile_name: null        # AWS profile name (optional)

  bedrock:
    region_name: "us-west-2"  # Bedrock service region
    # Optional Amazon Bedrock Guardrails — applied to every LLM call (content/
    # PII/grounding policies). Disabled unless an identifier is set.
    guardrail:
      identifier: null        # Guardrail ID or ARN; enables guardrails when set
      version: "DRAFT"        # "DRAFT" or a published version number
      trace: false            # Emit guardrail trace for auditing

  neptune:
    endpoint: "your-neptune-cluster.cluster-xyz.us-east-1.neptune.amazonaws.com"  # Required: Neptune cluster endpoint
    port: 8182
    use_iam: true
    pool_size: 4              # Gremlin connection pool; raise with indexing.neptune.index_concurrency

  opensearch:
    endpoint: "https://your-opensearch-domain.us-east-1.es.amazonaws.com"  # Required: OpenSearch domain endpoint
    port: 443
    use_ssl: true
    verify_certs: true
    use_iam: false  # Set to false if using username/password authentication

  s3:
    bucket_name: "your-s3-bucket-name"  # Required: S3 bucket for caching

# Processing Configuration
processing:
  max_concurrency: 5        # Maximum parallel operations
  batch_size: 10           # Batch processing size

  # Text Chunking
  chunking:
    chunker_type: "intelligent"  # "intelligent" or "simple"
    chunking_model_id: "anthropic.claude-3-5-haiku-20241022-v1:0"
    min_chunk_size: 5000
    max_chunk_size: 50000
    chunk_overlap: 500

  # Graph Extraction
  graph_extraction:
    extraction_model_id: "anthropic.claude-sonnet-4-20250514-v1:0"
    max_entities_per_chunk: 100
    max_relationships_per_chunk: 100

  # Translation
  translation:
    translation_model_id: "anthropic.claude-3-5-haiku-20241022-v1:0"
    target_language: "en"

# Indexing Configuration
indexing:
  opensearch:
    embedding_model_id: "amazon.titan-embed-text-v2:0"  # Bedrock embedding model
    text_units_index_alias: "graphrag-text-units"
    entities_index_alias: "graphrag-entities"
    community_reports_index_alias: "graphrag-community-reports"

# Search Configuration
search:
  answer_generation_model_id: "anthropic.claude-sonnet-4-20250514-v1:0"  # Main LLM for answer generation
  entity_extraction_model_id: "anthropic.claude-3-5-haiku-20241022-v1:0"

  # Hybrid search weights
  hybrid:
    lexical_weight: 0.5
    vector_weight: 0.5

# Custom Prompts Configuration (Optional)
custom_prompts:
  # Graph Extraction Prompts (Variables: input_text, max_entities_per_chunk, max_relationships_per_chunk)
  graph_extraction_system: |
    You are an expert knowledge graph extractor specialized in [DOMAIN].
    Extract entities and relationships from the provided text, focusing on [DOMAIN-SPECIFIC CONCEPTS].

  graph_extraction_human: |
    Extract entities and relationships from this [DOMAIN] text:
    {input_text}

    Extraction Limits:
    - Maximum Entities: {max_entities_per_chunk}
    - Maximum Relationships: {max_relationships_per_chunk}

  # Claim Extraction Prompts (Variables: input_text, entity_specs)
  claim_extraction_system: |
    You are a [DOMAIN] claim extraction specialist. Extract factual assertions
    focusing on [DOMAIN-SPECIFIC CLAIMS].

  claim_extraction_human: |
    Extract claims from this [DOMAIN] text:
    {input_text}

    Entity specifications: {entity_specs}

  # Graph Refinement/Gleaning Prompts (Variables: input_text, entity_specs, relationships_specs)
  graph_refinement_system: |
    You are a [DOMAIN] knowledge graph refinement expert. Improve and enhance
    the extracted entities and relationships for [DOMAIN-SPECIFIC ACCURACY].

  graph_refinement_human: |
    Refine the knowledge graph from this [DOMAIN] text:
    {input_text}

    Current entities: {entity_specs}
    Current relationships: {relationships_specs}

  # Community Report Prompts (Variables: community_summary, community_entities, community_relationships)
  community_report_system: |
    You are a [DOMAIN] analyst creating comprehensive community reports.
    Focus on [DOMAIN-SPECIFIC ANALYSIS CRITERIA].

  community_report_human: |
    Create a comprehensive report for this [DOMAIN] community:

    Summary: {community_summary}
    Key Entities: {community_entities}
    Relationships: {community_relationships}

  # Entity Extraction Prompts (Variables: query, target_language)
  entity_extraction_system: |
    You are a [DOMAIN] expert. Extract relevant entities from user queries.
    Pay special attention to [DOMAIN-SPECIFIC TERMINOLOGY].

  entity_extraction_human: |
    Extract key entities from this [DOMAIN] query: "{query}"
    Target language: {target_language}

  # Keyword Expansion Prompts (Variables: query, entities, topics, max_keywords)
  keyword_expansion_system: |
    You are a [DOMAIN] search specialist. Expand queries with relevant keywords
    focusing on [DOMAIN-SPECIFIC TERMINOLOGY].

  keyword_expansion_human: |
    Expand keywords for this [DOMAIN] query: "{query}"
    Entities: {entities}
    Topics: {topics}
    Maximum keywords: {max_keywords}

  # Query Refinement Prompts (Variables: original_query, results_summary, iteration)
  query_refinement_system: |
    You are a [DOMAIN] search specialist. Refine queries for better [DOMAIN] results
    based on previous search iterations.

  query_refinement_human: |
    Refine this [DOMAIN] query based on results:
    Original query: "{original_query}"
    Results summary: {results_summary}
    Iteration: {iteration}
```

**Domain Customization Examples:**

**Medical Domain:**
```yaml
custom_prompts:
  graph_extraction_system: |
    You are a medical knowledge extractor. Extract medical entities (diseases, symptoms, treatments, medications)
    and their relationships. Focus on clinical accuracy and medical terminology.

  graph_extraction_human: |
    Extract medical entities and relationships from this clinical text:
    {input_text}

    Extraction Limits:
    - Maximum Entities: {max_entities_per_chunk}
    - Maximum Relationships: {max_relationships_per_chunk}

  entity_extraction_system: |
    You are a medical expert. Extract medical entities from queries including diseases, symptoms,
    treatments, medications, and anatomical terms.

  entity_extraction_human: |
    Extract medical entities from this query: "{query}"
    Target language: {target_language}
```

**Legal Domain:**
```yaml
custom_prompts:
  graph_extraction_system: |
    You are a legal document analyzer. Extract legal entities (cases, statutes, regulations, parties)
    and their relationships. Focus on legal precedents and regulatory connections.

  community_report_system: |
    You are a legal analyst. Create reports focusing on case law, regulatory frameworks,
    and legal precedents within each topic cluster.

  community_report_human: |
    Create a comprehensive legal analysis report for this community:

    Summary: {community_summary}
    Key Legal Entities: {community_entities}
    Legal Relationships: {community_relationships}
```

**Financial Domain:**
```yaml
custom_prompts:
  graph_extraction_system: |
    You are a financial analyst. Extract financial entities (companies, markets, instruments, metrics)
    and their relationships. Focus on financial performance and market connections.

  keyword_expansion_system: |
    You are a financial search specialist. Expand queries with financial terminology,
    market indicators, and economic concepts.

  keyword_expansion_human: |
    Expand financial keywords for this query: "{query}"
    Financial entities: {entities}
    Market topics: {topics}
    Maximum keywords: {max_keywords}
```

### CLI Usage

#### 1. Index Documents
```bash
# Index local documents
run-ingestion --source-directory ./documents --config-path config.yaml

# With S3 sync for caching
run-ingestion --source-directory ./documents --config-path config.yaml --s3-sync --s3-bucket-name your-bucket

# Force rebuild (ignore cache)
run-ingestion --source-directory ./documents --config-path config.yaml --force-rebuild

# Resume from specific stage
run-ingestion --source-directory ./documents --config-path config.yaml --pipeline-id <id> --resume-from-stage graph_extraction

# Emit pipeline metrics as CloudWatch EMF (auto-extracted by CloudWatch Logs;
# default is `none`, a no-op sink with zero AWS dependency)
run-ingestion --source-directory ./documents --config-path config.yaml --metrics-sink cloudwatch
```

#### 2. Query the Knowledge Graph
```bash
# Single query
run-rag --query "What are the main themes in the documents?" --config-path config.yaml

# Interactive mode
run-rag --interactive --config-path config.yaml

# Specify search strategy and type
run-rag --query "Your question" --search-strategy local --search-type hybrid --config-path config.yaml

# With conversation memory
run-rag --interactive --use-memory --conversation-id my-session --config-path config.yaml
```

#### 3. Evaluate Performance
```bash
# Run evaluation on test dataset
run-eval --eval-data-path my_eval_data.json --config-path config.yaml

# Save results to specific directory
run-eval --eval-data-path my_eval_data.json --outputs-directory ./results --config-path config.yaml

# Evaluate with specific search strategy
run-eval --eval-data-path my_eval_data.json --search-strategy global --search-type vector --config-path config.yaml

# Use a LightRAG methodology mode
run-rag --query "Your question" --search-strategy mix --config-path config.yaml
```

#### 4. Visualize the Graph (standalone)
```bash
# Render visualizations from previously exported graph data (no re-ingestion)
run-visualization --data-path visualization_data.json --output-dir ./viz --config-path config.yaml

# Render only specific renderers
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

#### 5. Tune Prompts for Your Domain
```bash
# Profile a corpus and emit domain-adapted custom_prompts YAML to merge into config
run-prompt-tuning --source-dir ./source --output tuned_prompts.yaml --config-path config.yaml
```

**Evaluation Dataset Format**

Create an evaluation dataset file (e.g., `my_eval_data.json`) with the following structure:

```json
[
  {
    "id": "q1",
    "question": "What are the main themes discussed in the documents?",
    "answer": "The main themes include artificial intelligence, machine learning, and data processing.",
    "category": "general",
    "difficulty": "easy",
    "reference_sources": ["doc1.pdf", "doc2.txt"],
    "expected_entities": ["AI", "machine learning", "data processing"],
    "expected_relationships": ["AI enables machine learning"],
    "metadata": {
      "search_strategy": "global",
      "custom_field": "value"
    }
  },
  {
    "id": "q2",
    "question": "How do entities X and Y relate to each other?",
    "answer": "Entity X influences Entity Y through relationship Z.",
    "category": "relationships",
    "difficulty": "medium"
  }
]
```

**Required**: `question` field only. All other fields (`id`, `answer`, `category`, `difficulty`, `reference_sources`, `expected_entities`, `expected_relationships`, `metadata`) are optional.

### Python API Usage

```python
import nest_asyncio

from aws_graphrag.shared import get_config
from aws_graphrag.domain.models import PipelineConfig, SearchStrategy, SearchType
from aws_graphrag.retrieval import RAGInput, create_rag_chain
from aws_graphrag.ingestion import DataIngestionPipeline

nest_asyncio.apply()

# Initialize configuration
config = get_config("config.yaml")  # Specify your config path

# Index documents
pipeline = DataIngestionPipeline(config=config, pipeline_config=PipelineConfig())
await pipeline.run(source_directory="./documents")

# Create RAG chain
rag_chain = await create_rag_chain(config)

# Query with different strategies
rag_input = RAGInput(
    query="What are the key relationships between entities X and Y?",
    search_strategy=SearchStrategy.AUTO,
    search_type=SearchType.HYBRID,
    top_k=10,
    use_memory=True,
    conversation_id="session-123"
)

result = await rag_chain.ainvoke(rag_input)
print(f"Answer: {result.answer}")
```

### Advanced Usage

#### Custom Search Strategies
```python
# Use specific search strategy with custom parameters
rag_input = RAGInput(
    query="Detailed analysis of specific entity",
    search_strategy=SearchStrategy.LOCAL,
    search_type=SearchType.HYBRID,
    top_k=20,
    retrieval_multiplier=2,
    filters={"category": "research"}
)

result = await rag_chain.ainvoke(rag_input)
```

#### Interactive Mode with Memory
```python
# Enable conversation memory
rag_input = RAGInput(
    query="What are the main themes?",
    use_memory=True,
    conversation_id="my-session",
    search_strategy=SearchStrategy.GLOBAL
)

# Follow-up query with context
follow_up = RAGInput(
    query="Can you elaborate on the first theme?",
    use_memory=True,
    conversation_id="my-session"  # Same conversation ID
)
```

#### Pipeline Resume and Caching
```python
# Resume pipeline from specific stage
pipeline = DataIngestionPipeline(
    config=config,
    pipeline_config=PipelineConfig(),
)

await pipeline.run(
    source_directory="./documents",
    pipeline_id="existing-pipeline-id",
    resume_from_stage="graph_extraction"
)

# Force rebuild ignoring cache
await pipeline.run(
    source_directory="./documents"
)
```

#### Evaluation and Benchmarking
```python
from aws_graphrag.evaluation import EvaluationManager

# Run comprehensive evaluation
eval_manager = EvaluationManager(config, rag_chain)
queries, ground_truths = eval_manager.load_data(
    eval_data_path="my_eval_data.json",
)

results = await eval_manager.evaluate_dataset(
  queries, ground_truths
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT-0 License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
- [GraphRAG: Unlocking LLM Discovery on Narrative Private Data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [GraphRAG: New Tool for Complex Data Discovery Now on GitHub](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/)
- [GraphRAG Auto-Tuning Provides Rapid Adaptation to New Domains](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/)
- [Introducing DRIFT Search: Combining Global and Local Search Methods to Improve Quality and Efficiency](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
- [GraphRAG: Improving Global Search via Dynamic Community Selection](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/)
- [LazyGraphRAG: Setting a New Standard for Quality and Cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [Introducing GraphRAG 1.0](https://www.microsoft.com/en-us/research/blog/moving-to-graphrag-1-0-streamlining-ergonomics-for-developers-and-users/)
- [Microsoft GraphRAG Library](https://github.com/microsoft/graphrag)

## 🏢 About

This project is developed by Amazon Web Services as an enterprise-grade solution for knowledge graph-based RAG systems, released under the MIT-0 License.
