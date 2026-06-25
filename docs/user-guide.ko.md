# AWS GraphRAG — 사용자 가이드

> 🇬🇧 English version: [docs/user-guide.md](./user-guide.md)

이 문서는 **aws-graphrag**의 실용적인 사용 방법 가이드입니다. aws-graphrag는
대규모 다국어 문서 코퍼스로부터 지식 그래프를 구축하고 그 위에서 질문에 답하는
AWS 네이티브 Knowledge Graph RAG 프레임워크입니다. 두 가지 검색 방법론 —
**Microsoft GraphRAG**(커뮤니티 요약)와 **LightRAG**(이중 레벨 키워드) — 을
하나의 스택 위에 재구현하여 질의마다 선택할 수 있습니다.

- *무엇을 / 왜*와 1분 빠른 시작은 [README.md](../README.md)를 참고하세요.
- *내부 구조 / 아키텍처*(헥사고날 레이어, 포트 & 어댑터, 의존성 규칙)는
  기술 문서([docs/design.md](./design.md) 한국어 / [docs/design.en.md](./design.en.md)
  영어)를 참고하세요.

아래 내용은 모두 코드베이스의 실제 CLI 플래그와 설정 키에 기반합니다. 다섯 개의
콘솔 진입점(`pyproject` 스크립트로 정의)은 다음과 같습니다.

| 스크립트 | 모듈 | 용도 |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | 지식 그래프 구축 / 업데이트 |
| `run-rag` | `application.cli.run_rag_chain` | 그래프 질의 |
| `run-eval` | `application.cli.run_evaluation` | 검색 + 생성 평가 |
| `run-visualization` | `application.cli.run_visualization` | 내보낸 그래프 렌더링 (인제스천 없음) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | 도메인 적응 프롬프트 생성 |

---

## 1. 사전 요구사항 & 설치

### 런타임

- **Python 3.10 – 3.12**
- **[uv](https://docs.astral.sh/uv/)** (권장 패키지 매니저, `pip`도 사용 가능)

### AWS 서비스

| 서비스 | 필수 여부 | 용도 |
|---|---|---|
| **Amazon Bedrock** | 예 | 모든 LLM 호출(청킹, 추출, gleaning, 커뮤니티 리포트, 답변 생성), 임베딩, 리랭킹. 설정한 모델 ID에 대해 모델 액세스를 활성화하세요. |
| **Amazon Neptune** | 예 | 지식 그래프(엔티티, 관계, 커뮤니티) 및 질의 시 멀티홉 순회. |
| **Amazon OpenSearch** | 예 | 벡터 + BM25 렉시컬 인덱스(텍스트 유닛, 엔티티, 커뮤니티 리포트, 관계, claim). |
| **Amazon S3** | 예 | 파이프라인 캐시 동기화, 선택적 임베딩 캐시 영속화, 문서 저장. |
| **Amazon DynamoDB** | 증분 인덱싱 시에만 | 콘텐츠 해시로 코퍼스를 diff하는 문서-상태 레지스트리. |

### 설치

```bash
git clone <repository-url>
cd aws-graphrag

# uv (recommended)
uv sync --extra dev

# or pip
pip install -e .
```

선택적 추가 패키지: **Markdown(.md)** 및 **HTML(.html)** 파싱에는
`unstructured` 패키지가 필요합니다. 이 패키지가 없으면 `.pdf`, `.txt`, `.csv`,
`.json`만 파싱됩니다(파서는 `.md`/`.html`에 대해 누락된 패키지 이름을 명시하는
명확한 에러를 발생시킵니다). 해당 포맷이 필요하면 패키징 도구로 설치하세요.

### 인증

서로 독립적인 두 가지 인증 사항이 있습니다.

1. **AWS 자격증명** — 표준 자격증명 체인을 통해 공급됩니다. 명명된 프로파일을
   사용하려면 `config.yaml`에서 `aws.profile_name`을 설정하고, 기본 체인(환경
   변수, 인스턴스 역할 등)을 사용하려면 `null`로 두세요. Neptune은
   `aws.neptune.use_iam: true`일 때 SigV4를 사용합니다.

2. **OpenSearch 인증** — IAM(`aws.opensearch.use_iam: true`) 또는
   username/password 중 하나입니다. username/password를 사용하려면
   `use_iam: false`로 설정하고 `.env` 파일을 생성하세요(`.env-template` 복사).

   ```bash
   # .env — only needed when aws.opensearch.use_iam is false
   OPENSEARCH_USERNAME=your_opensearch_username
   OPENSEARCH_PASSWORD=your_opensearch_password
   ```

   `.env` 파일은 CLI(`run-ingestion`, `run-rag`)가 자동으로 로드합니다.
   `use_iam: true`일 때는 `.env`가 필요하지 않습니다.

---

## 2. 설정

템플릿으로부터 설정 파일을 만들고 모든 CLI에 `--config-path config.yaml`로
지정하세요.

```bash
cp config-template.yaml config.yaml
```

설정은 중첩된 Pydantic 모델입니다(`aws_graphrag/domain/models/config.py`).
`config-template.yaml`에 전체 스키마와 인라인 주석이 담겨 있으며, 가장 유용한
섹션과 실제로 튜닝하게 될 항목들은 아래와 같습니다.

### 2.1 `aws` — 서비스 엔드포인트 & 자격증명

```yaml
aws:
  region_name: "ap-northeast-2"
  profile_name: null              # named AWS profile, or null for default chain

  bedrock:
    region_name: "ap-northeast-2" # Bedrock can live in a different region
    assumed_role_arn: null
    enable_global_profile: true   # use cross-region inference profiles
    guardrail:                    # optional Bedrock Guardrails on every LLM call
      identifier: null            # set a guardrail ID/ARN to enable
      version: "DRAFT"
      trace: false

  neptune:
    endpoint:                     # REQUIRED — Neptune cluster endpoint
    port: 8182
    use_iam: true
    pool_size: 4                  # raise alongside indexing.neptune.index_concurrency

  opensearch:
    endpoint:                     # REQUIRED — OpenSearch domain endpoint
    port: 443
    use_ssl: true
    verify_certs: true
    use_iam: false                # false => username/password from .env

  s3:
    bucket_name:                  # REQUIRED for cache sync / embedding-cache persistence
    encryption:
      encryption_type: "AES256"   # NONE | AES256 | aws:kms
      kms_key_id: null

  dynamodb:                       # incremental indexing registry
    enabled: false                # set true to enable delta indexing
    table_name: "aws-graphrag-doc-status"
    create_table_if_missing: true
    billing_mode: "PAY_PER_REQUEST"
```

> **Guardrail 배치 주의:** 멀티 리전으로 배포할 때 Bedrock Guardrail은
> `region_name`이 아니라 `bedrock.region_name`(LLM 호출이 전달되는 리전)에
> 존재해야 합니다.

### 2.2 `fixing` — 잘못된 형식의 모델 출력 자동 복구

```yaml
fixing:
  enabled: true
  fixing_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
```

구조화된 스테이지에서 LLM이 잘못된 형식의 JSON을 반환하면, 실패시키는 대신
모델에게 복구를 다시 요청합니다. 켜 둔 채로 두세요.

### 2.3 `processing` — 동시성, 청킹, 번역, 추출

LLM 스테이지는 Bedrock I/O 바운드이므로 동시성을 CPU 수보다 훨씬 높게 잡을 수
있습니다.

```yaml
processing:
  max_concurrency: 20      # concurrent LLM calls within a batch
  chunk_concurrency: 4     # mini-batch chunks running at once
  batch_size: 10
  max_retries: 3
  ignore_errors: false
  deduplicate: false
  resolution_method: "minhash"      # minhash | sequence_matcher
  similarity_threshold: 0.6         # entity-resolution fuzzy-match threshold

  document_parsing:
    source_directory:               # overridden by --source-directory CLI flag
    target_directory: null
    index_value: null
```

**청킹** — `intelligent`는 LLM으로 시맨틱 경계를 고르고, `simple`은 크기로
분할합니다. 가장 많이 튜닝하는 항목: `min_chunk_size` / `max_chunk_size`.

```yaml
  chunking:
    chunker_type: "intelligent"     # intelligent | simple
    chunking_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
    content_type: "markdown"
    min_chunk_size: 5000
    max_chunk_size: 50000
    chunk_overlap: 500
    pre_chunk_size: 50000
    pre_chunk_overlap: 500
    fallback_chunk_size: 50000
    max_marker_miss_rate: 0.1
```

**번역** — 파이프라인 스테이지로 실행되지만, `source_language ==
target_language`이고 `additional_target_languages`가 비어 있으면 **no-op**(LLM
비용 0)입니다. 다국어 워크플로는 §3을 참고하세요.

```yaml
  translation:
    enabled: true
    translation_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
    source_language: "en"           # predominant source language (no-op skip only)
    target_language: "en"
    additional_target_languages: null
```

**그래프 추출** — 인제스천의 핵심입니다. `entity_types`는 도메인 적응에서 가장
영향력이 큰 단일 항목입니다(§9 참고). 각 항목은 `"LABEL: short description"`
형식이며, 빈 리스트로 두면 모델이 자유롭게 선택합니다.

```yaml
  graph_extraction:
    extraction_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_entities_per_chunk: 100
    max_relationships_per_chunk: 100
    entity_confidence_threshold: 0.0
    enable_confidence_extraction: true
    entity_types:
      - "PERSON: Names, individuals, roles, titles"
      - "ORGANIZATION: Companies, institutions, departments, groups"
      - "LOCATION: Places, addresses, geographic areas, facilities"
      - "CONCEPT: Ideas, theories, methodologies, frameworks, principles"
      - "OBJECT: Documents, tools, products, systems, technologies"
      - "EVENT: Meetings, projects, activities, processes, incidents"
      - "TEMPORAL: Dates, time periods, schedules, deadlines"
    description_summarization:      # collapse over-long merged descriptions
      enabled: true
      summary_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
      force_summary_threshold_tokens: 600
      max_summary_tokens: 256
```

**Gleaning** — 첫 번째 패스에서 놓친 엔티티/관계를 잡아내는 반복 추출
패스입니다(품질 대 비용 트레이드오프).

```yaml
  gleaning:
    enabled: true
    graph_refinement_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_rounds: 3
    convergence_threshold: 0.8
    quality_threshold: 0.9
    min_improvement_threshold: 0.03
    # ...count-based quality/convergence scaling constants
```

**Claim 추출** — 기본값은 OFF이며, 텍스트 유닛마다 추가 LLM 호출이 듭니다. ON일
때 `local` 검색은 매칭되는 claim(MS GraphRAG covariate)을 주입하고, `simple`
검색은 스윕에 claim 인덱스를 포함합니다.

```yaml
  claim_extraction:
    enabled: false
    extraction_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_entities_per_prompt: 200
```

### 2.4 `graph` — 분석, 커뮤니티 탐지, 시각화

```yaml
graph:
  analysis:
    centrality:
      calculate_degree: true
      calculate_betweenness: true
      calculate_pagerank: true
      calculate_closeness: false
      calculate_eigenvector: false
      pagerank_alpha: 0.85
    statistics:
      calculate_density: true
      calculate_clustering: true
      calculate_components: true

  community_detection:              # Leiden clustering
    resolution: 1.0
    random_state: 42
    max_levels: 5
    min_community_size: 3
    auto_resolution: true
    report_generation:              # LLM-generated community summaries (used by global search)
      enabled: true
      report_generation_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
      max_entities_per_report: 50
      max_report_context_tokens: 4000

  visualization:
    enabled: true
    outputs_directory: "outputs/visualization"
    embedding_method: "node2vec"
    layout_method: "umap"           # umap | tsne | pca
```

### 2.5 `indexing` — OpenSearch & Neptune 쓰기 측

```yaml
indexing:
  reset: false
  additional_suffix: null           # appended to default index/label suffix
  cross_run_merge: false            # on delta runs, union with existing graph state

  opensearch:
    embedding_model_id: "amazon.titan-embed-text-v2:0"
    embedding_dimension: null
    persist_embedding_cache: false  # cache embeddings to S3 across runs/phases
    text_units_index_prefix: "graphrag-text-units"
    entities_index_prefix: "graphrag-entities"
    community_reports_index_prefix: "graphrag-community-reports"
    relationships_index_prefix: "graphrag-relationships"   # enables LightRAG high-level retrieval
    claims_index_prefix: "graphrag-claims"
    default_analyzer: "standard"
    language_analyzers:             # per-language text analyzer (extend freely)
      en: "english"
      ko: "nori"
    vector_search:
      ef_construction: 128
      m: 24
      ef_search: 100
      space_type: "cosinesimil"
      engine: "faiss"               # faiss is the modern kNN engine (nmslib deprecated)

  neptune:
    batch_size: 100
    index_concurrency: 1            # >1 fans write batches over a thread pool
    max_hops: 3                     # neighbor-expansion depth at retrieval time
    max_results_per_hop: 50
    min_entity_importance: 0.5
```

> `*_index_prefix` 키가 실제 설정 이름입니다. (예전 README는 `*_index_alias`를
> 언급했으나 — 그런 키는 존재하지 않습니다.)

### 2.6 `search` — 검색, 융합, 리랭킹, 전략별 항목

```yaml
search:
  translation_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
  entity_extraction_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
  strategy_selection_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"   # the `auto` router
  context_building_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
  answer_generation_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"    # the answer LLM

  hybrid:
    lexical_weight: 0.5
    vector_weight: 0.5

  fusion:
    method: "rrf"                   # rrf | weighted
    rrf_k: 60
    diversity_lambda: 0.5           # MMR: 1.0 = pure relevance, 0.0 = max diversity
    fusion_weights: { ... }         # only used when method: weighted

  reranking:
    enabled: true
    rerank_model_id: "cohere.rerank-v3-5:0"
    top_k: 100

  lightrag_search:
    raw_query_fallback_max_len: 50  # short queries fall back to raw query as a keyword

  global_search:
    max_communities: 10
    use_dynamic_selection: true
    enable_map_reduce: true
    map_model_id: "anthropic.claude-haiku-4-5-20251001-v1:0"
    max_map_reduce_tokens: 8000

  local_search:
    entity_frequency_threshold: 20  # drop overly-generic graph-expanded entities

  drift_search:
    enable_query_refinement: true
    enable_keyword_extraction: true
    max_iterations: 3
    initial_top_k: 5

  token_manager:
    max_context_tokens: 200000
```

### 2.7 `memory`, `cache`, `logging`

```yaml
memory:
  max_conversations: 100
  max_messages_per_conversation: 20
  max_conversation_age_hours: 168

cache:
  ttl_seconds: 86400               # null = never expire
  chunking:
    enabled: true
    max_file_size_mb: 50

logging:
  level: "INFO"
  log_format: "structured"
  log_to_file: true
  log_file_path: "logs/log.txt"
```

### 2.8 `evaluation`

```yaml
evaluation:
  outputs_directory: "outputs/evaluation"
  evaluation_model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0"
  enabled_evaluators:
    - langchain
    - ragas
    # - graph_aware                # opt-in; needs expected_entities/relationships
  langchain_metrics: [correctness, partial_correctness]
  ragas_metrics: [answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness]
  save_detailed_results: true
```

### 2.9 `custom_prompts`

모든 프롬프트에는 `*_system` / `*_human` 오버라이드가 있습니다(기본값 `null` =
`aws_graphrag/domain/prompts/`의 내장 프롬프트 사용). §9를 참고하세요. 필요한
것만 오버라이드하고 나머지는 `null`로 두세요.

---

## 3. 인제스천 (`run-ingestion`)

인제스천은 문서 디렉터리를 OpenSearch + Neptune에 인덱싱된 지식 그래프로
변환합니다.

### CLI 플래그 (검증됨)

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--source-directory` | `$GRAPHRAG_SOURCE_DIRECTORY` | 소스 문서 디렉터리 (실행에 필수) |
| `--target-directory` | source dir | 파싱된 문서가 기록될 위치 |
| `--cache-directory` | `cache` | 파이프라인 캐시 + 중간 결과 |
| `--force-rebuild` | off | 기존 캐시를 모두 무시하고 처음부터 재구축 |
| `--s3-sync` | off | 캐시를 S3에 동기화 (`--s3-bucket-name` 필요) |
| `--s3-bucket-name` | — | 캐시 동기화용 S3 버킷 |
| `--s3-prefix` | `pipeline-runs` | 캐시 파일의 S3 키 프리픽스 |
| `--pipeline-id` | `$GRAPHRAG_PIPELINE_ID` | 재개/검사할 기존 실행 |
| `--resume-from-stage` | — | 재개할 스테이지 (`--pipeline-id` 필요) |
| `--verify-metadata` | off | 파이프라인 메타데이터 무결성 검증 (`--pipeline-id` 필요) |
| `--repair-metadata` | off | 메타데이터 복구 시도 (`--pipeline-id` 필요) |
| `--continue-on-error` | off | 스테이지 에러 시에도 계속 진행 |
| `--enabled-stages` | all | 실행할 스테이지 목록(쉼표 구분) |
| `--metrics-sink` | `none` | `none` 또는 `cloudwatch` (EMF를 stdout으로) |
| `--config-path` | — | `config.yaml` 경로 |

### 12개 파이프라인 스테이지

실행 순서(`DataIngestionPipeline.STAGE_CLASSES`). `--enabled-stages` /
`--resume-from-stage`에는 스테이지 **이름**(대소문자 무관)을 사용하세요.

1. **`document_parsing`** — 포맷별 텍스트 추출(`.pdf`, `.txt`, `.csv`,
   `.json`; `unstructured` 추가 패키지로 `.md`/`.html`).
2. **`document_loading`** — 파싱된 문서를 파이프라인 코퍼스로 로드.
3. **`text_chunking`** — 문서를 텍스트 유닛으로 분할(`processing.chunking`).
4. **`translation`** — 선택적; `target_language`로 번역(source == target이고
   추가 타겟이 없으면 no-op).
5. **`graph_extraction`** — LLM이 청크마다 엔티티 + 관계를 추출.
6. **`gleaning`** — 선택적 반복 정제 패스(`processing.gleaning`).
7. **`graph_resolution`** — 중복 엔티티/관계를 퍼지 매칭하여 병합.
8. **`claim_extraction`** — 선택적; 사실 claim 추출(기본 OFF).
9. **`claim_resolution`** — 선택적; 추출된 claim 중복 제거.
10. **`graph_analysis`** — 중심성 지표 + 그래프 통계.
11. **`community_detection`** — Leiden 클러스터링 + LLM 커뮤니티 리포트.
12. **`indexing`** — 모든 것을 OpenSearch + Neptune에 기록(활성화 시 DynamoDB
    레지스트리에도).

### 예시

```bash
# Full build
run-ingestion --source-directory ./documents --config-path config.yaml

# With S3 cache sync
run-ingestion --source-directory ./documents --config-path config.yaml \
  --s3-sync --s3-bucket-name your-bucket

# Force a clean rebuild (ignore cache)
run-ingestion --source-directory ./documents --config-path config.yaml --force-rebuild

# Resume an interrupted run from a stage
run-ingestion --source-directory ./documents --config-path config.yaml \
  --pipeline-id <id> --resume-from-stage graph_extraction

# Run only specific stages
run-ingestion --source-directory ./documents --config-path config.yaml \
  --enabled-stages DOCUMENT_PARSING,TEXT_CHUNKING,GRAPH_EXTRACTION

# Emit metrics as CloudWatch EMF (auto-extracted by CloudWatch Logs)
run-ingestion --source-directory ./documents --config-path config.yaml --metrics-sink cloudwatch
```

**재개 vs. 강제 재구축:** `--force-rebuild` 없이 실행하면 완료된 스테이지는
캐싱되어 재실행 시 건너뜁니다. 특정 이전 실행을 재개하려면 `--pipeline-id`를
전달하세요. `--resume-from-stage`와 함께 쓰면 선택한 스테이지부터 재실행하고,
그렇지 않으면 파이프라인이 처음 실패/미완료된 스테이지를 자동 감지합니다.
`--force-rebuild`는 모든 캐시를 버리고 처음부터 다시 시작합니다.

**S3 동기화**는 스테이지 캐시를 `s3://<bucket>/<prefix>/...`에 유지하므로, 새
프로세스(예: 새 Fargate 태스크)가 완료된 스테이지를 재계산하지 않고 재개할 수
있습니다. 임베딩에 한해서는 `indexing.opensearch.persist_embedding_cache: true`
로 설정하면 실행 간에 변경되지 않은 텍스트를 다시 임베딩하지 않습니다.

### 다국어 인제스천

코퍼스의 주된 언어와 인덱싱하려는 타겟을 설정하세요.

```yaml
processing:
  translation:
    enabled: true
    source_language: "ko"
    target_language: "en"
    additional_target_languages: ["ja"]   # index additional languages too
```

`source_language == target_language`이고 `additional_target_languages`가
비어 있거나 null이면, 번역 스테이지는 `is_noop`으로 건너뜁니다 — 영어 전용
코퍼스는 `enabled: true`라도 번역 LLM 비용을 **전혀** 내지 않습니다. 언어 인식
OpenSearch analyzer는 `indexing.opensearch.language_analyzers`(예: `ko: nori`)
아래에서 설정합니다. 목록에 없는 언어는 `default_analyzer`로 폴백됩니다.

---

## 4. 질의 (`run-rag`)

### CLI 플래그 (검증됨)

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--query`, `-q` | — | 단일 질의 (`--interactive`와 상호 필수) |
| `--interactive`, `-i` | off | 인터랙티브 채팅 (메모리 자동 활성화) |
| `--mode` | `rag` | `rag`(전체 생성) 또는 `search`(검색만) |
| `--conversation-id` | — | 기존 대화 이어가기 |
| `--use-memory` | off | 대화 메모리 활성화 (인터랙티브에서는 자동) |
| `--suffix` | — | 멀티테넌트 또는 버전별 인덱스용 인덱스/라벨 접미사 |
| `--enable-thinking` | off | 모델의 단계별 추론 활성화 |
| `--search-strategy` | `auto` | `auto` `drift` `global` `local` `simple` `mix` `hybrid` `naive` |
| `--search-type` | `hybrid` | `hybrid` `lexical` `vector` |
| `--top-k` | `10` | 최대 검색 결과 수 |
| `--retrieval-multiplier` | `1` | 검색 깊이 증가 |
| `--disable-query-processing` | off | 번역 + 엔티티 추출 건너뛰기 |
| `--filters` | — | `key:value` 속성 필터 (공백 구분) |
| `--output-format` | `text` | `text` 또는 `json` |
| `--verbose`, `-v` | off | 질의 처리 정보, 출처, 메트릭 표시 |
| `--config-path` | — | `config.yaml` 경로 |

### 검색 전략 — 어느 것을 언제 쓸까

**방법론 선택.** GraphRAG 전략은 코퍼스에 대한 *요약 및 주제별 종합*에
탁월합니다(커뮤니티 리포트가 전역적 커버리지를 제공). LightRAG 전략은 더 빠르고
*이중 레벨 키워드* 검색에 의존합니다 — 키워드 기반 조회와 저비용 베이스라인으로
좋습니다. 두 방법론 모두 동일한 하이브리드 스코어러(BM25 렉시컬 + 벡터 시맨틱 +
그래프 순회 + RRF + Bedrock 리랭킹)를 거치며, 검색 알고리즘만 다릅니다.

**GraphRAG (커뮤니티 요약):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `simple` | 빠른 사실 조회; 단순한 질문 | OpenSearch 벡터 + 키워드 직접 검색, 그래프 순회 없음. 가장 빠름. claim 추출이 켜져 있으면 claim 인덱스 포함. |
| `local` | 특정 엔티티/개념에 대한 상세 질문 | 질의 엔티티 추출 → 이웃/관계를 위한 Neptune 그래프 순회 → 벡터/키워드 결과와 결합. 활성화 시 claim(covariate) 주입. |
| `global` | 광범위하고 주제적인, "주요 주제가 무엇인가" 류의 질문 | 커뮤니티 리포트 + 동적으로 선택된 커뮤니티에 대한 map-reduce 사용. 고수준 종합에 최적. |
| `drift` | 탐색이 필요한 복잡하고 다면적인 질문 | 라운드 간 수렴 감지를 동반한 반복적 질의 정제/확장. |
| `auto` | 모를 때 / 일반 용도 (기본값) | LLM 라우터(`search.strategy_selection_model_id`)가 질의로부터 최적 전략 선택. |

**LightRAG (이중 레벨 키워드):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `mix` | 일반 LightRAG 용도; 그래프 + 청크 균형 | 저수준 키워드 → 엔티티 인덱스, 고수준 키워드 → 관계 인덱스, Neptune 확장, **추가로** naive 벡터 청크 검색을 섞음. |
| `hybrid` | 키워드 기반 그래프 질문 | `mix`와 동일하나 추가 naive 청크 혼합 없음. |
| `naive` | 빠른 베이스라인 / 비교 평가 | 순수 벡터 청크 검색, 그래프 없음. LightRAG 베이스라인. |

> `mix`/`hybrid`의 경우 관계 벡터 인덱스가 구축되었는지 확인하세요
> (`indexing.opensearch.relationships_index_prefix`, 인제스천 중 자동 구축) —
> 이것이 고수준 키워드 검색을 구동합니다. 키워드가 나오지 않는 짧은 질의는 원본
> 질의를 저수준 키워드로 사용하는 방식으로 폴백합니다
> (`search.lightrag_search.raw_query_fallback_max_len`로 제어).

### 예시

```bash
# Single query (auto strategy, hybrid search)
run-rag --query "What are the main themes in the documents?" --config-path config.yaml

# Pick a strategy + search type
run-rag --query "How does entity X relate to Y?" \
  --search-strategy local --search-type hybrid --config-path config.yaml

# LightRAG mode
run-rag --query "Your question" --search-strategy mix --config-path config.yaml

# Retrieval only (no answer generation), JSON output
run-rag --query "..." --mode search --output-format json --config-path config.yaml

# Verbose: show extracted entities, top sources, and metrics
run-rag --query "..." --verbose --config-path config.yaml

# Attribute filters
run-rag --query "..." --filters category:research entity_type:person --config-path config.yaml
```

### 인터랙티브 모드 & 대화 메모리

```bash
run-rag --interactive --config-path config.yaml
# or continue a named session:
run-rag --interactive --conversation-id my-session --config-path config.yaml
```

인터랙티브 모드는 메모리를 자동 활성화합니다. 세션 내 명령어:

- `help` — 명령어 목록
- `new` — 새 대화 시작 (새 ID)
- `set-filter key:value` — 필터 추가/업데이트
- `clear-filters` — 모든 필터 제거
- `show-config` — 활성 설정 표시
- `quit` / `exit` — 종료

CLI에서 단발 멀티턴을 하려면 동일한 `--conversation-id`를 `--use-memory`와 함께
재사용하세요. 메모리 제한은 `memory` 설정 섹션 아래에 있습니다.

---

## 5. 증분 인덱싱

증분(델타) 인덱싱은 지난 실행 이후 **신규이거나 변경된** 문서만 다시 인덱싱하고
이를 라이브 그래프에 병합합니다 — 전체를 재구축하는 대신.

### 활성화

```yaml
aws:
  dynamodb:
    enabled: true
    table_name: "aws-graphrag-doc-status"
    create_table_if_missing: true
```

이것을 켜면, 각 `run-ingestion`은 코퍼스를 DynamoDB 문서-상태 레지스트리와
**콘텐츠 해시**로 diff합니다.

### 워크플로

- **문서 추가:** 새 파일을 소스 디렉터리에 넣고 `run-ingestion`을 재실행합니다.
  새 파일만 파싱/추출/인덱싱되며, 그 엔티티와 관계는 기존 그래프에
  병합됩니다(멱등 `upsert_*`).
- **문서 수정:** 파일을 편집하고 재실행합니다. 콘텐츠 해시가 바뀌므로 문서는
  변경된 것으로 취급됩니다: 기존 아티팩트가 제거되고 새 버전이 재인덱싱됩니다.
- **문서 삭제:** 소스 디렉터리에서 제거하고 재실행합니다. 그 문서에서만 보이는
  **독점적** 아티팩트(엔티티/관계, 레지스트리의 문서별 계보로 추적)는
  삭제됩니다. 살아남은 문서와 공유되는 아티팩트는 유지됩니다.

### 실행 간 병합 (cross-run merge)

기본적으로 델타 실행은 영향받은 그래프 필드를 덮어씁니다. 대신
`indexing.cross_run_merge: true`로 설정하면 upsert 전에 델타를 기존 그래프
상태(description / `text_unit_ids` / frequency / weight)와 *합집합*합니다 —
엔티티의 description이 여러 문서에 걸쳐 누적되어야 할 때 유용합니다. read-back을
지원하는 그래프 어댑터가 필요합니다. 기본값은 OFF.

---

## 6. 평가 (`run-eval`)

### CLI 플래그 (검증됨)

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--eval-data-path` | **필수** | 질문 + 정답이 담긴 JSON 파일 |
| `--outputs-directory` | `evaluation.outputs_directory` | 결과 저장 위치 |
| `--suffix` | — | 인덱스/라벨 접미사 |
| `--enable-thinking` | off | 모델 추론 |
| `--search-strategy` | `auto` | 각 질문에 답하는 데 사용할 전략 |
| `--search-type` | `hybrid` | 검색 방법 |
| `--top-k` | `10` | 최대 결과 수 |
| `--retrieval-multiplier` | `1` | 검색 깊이 |
| `--verbose`, `-v` | off | 디버그 로깅 |
| `--config-path` | — | `config.yaml` 경로 |

### 평가자

`evaluation.enabled_evaluators`로 선택합니다.

- **`langchain`** — LangChain 기반 텍스트 유사도(`langchain_metrics`:
  `correctness`, `partial_correctness`). `answer` 정답이 필요합니다.
- **`ragas`** — RAGAS 지표(`answer_correctness`, `answer_relevancy`,
  `context_precision`, `context_recall`, `faithfulness`).
- **`graph_aware`** — 결정적이고 **LLM 불필요**한 엔티티/관계 **커버리지 =
  recall**: 기대되는 그래프 아티팩트 중 몇 개가 생성된 답변에 나타나는지(대소문자
  무관 부분 문자열 매칭). 데이터셋에 `expected_entities` /
  `expected_relationships`가 필요합니다. **precision과 F1은 의도적으로
  미산출**됩니다 — 자유 텍스트 답변에서 모든 엔티티를 열거하는 것은 신뢰성 있게
  불가능하므로, precision/F1을 보고하는 것은 recall 신호에 다른 이름표만
  붙이는 셈이기 때문입니다. (`enabled_evaluators`에서 `graph_aware`의 주석을
  해제하여 opt-in.)

### 평가 데이터 포맷

객체의 JSON 배열입니다. `question`만 필수이고 나머지는 모두 선택입니다.
`expected_entities` / `expected_relationships`는 `graph_aware` 평가자에*만*
필요합니다.

```json
[
  {
    "id": "q1",
    "question": "What are the main themes discussed in the documents?",
    "answer": "The main themes include AI, machine learning, and data processing.",
    "category": "general",
    "difficulty": "easy",
    "reference_sources": ["doc1.pdf", "doc2.txt"],
    "expected_entities": ["AI", "machine learning", "data processing"],
    "expected_relationships": ["AI enables machine learning"],
    "metadata": { "search_strategy": "global" }
  },
  {
    "id": "q2",
    "question": "How do entities X and Y relate to each other?"
  }
]
```

항목별 `metadata`(예: `search_strategy`)는 해당 질문에 대해 CLI 기본값을
오버라이드합니다. `id`는 `query_id`로 줄 수도 있습니다.

### 예시

```bash
run-eval --eval-data-path my_eval_data.json --config-path config.yaml

run-eval --eval-data-path my_eval_data.json --outputs-directory ./results --config-path config.yaml

run-eval --eval-data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml
```

결과(질의별 상세 + 지표별 평균/중앙값/표준편차/최소/최대 요약)는 출력
디렉터리에 기록됩니다.

---

## 7. 시각화 (`run-visualization`)

이것은 이미 내보내진 시각화 데이터 JSON(인제스천 중
`GraphVisualizationManager.export_visualization_data`가 생성)으로부터 그리는
**독립형** 렌더러입니다. 인제스천을 다시 실행하거나 AWS를 건드리지 **않습니다**.

### CLI 플래그 (검증됨)

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--data-path` | **필수** | 내보낸 시각화 데이터 JSON |
| `--output-dir` | `visualization_outputs` | 렌더링 파일을 기록할 위치 |
| `--renderers` | 등록된 전체 | 실행할 렌더러: `interactive`, `static` |
| `--config-path` | — | `config.yaml` 경로 |

등록된 두 렌더러는 **`interactive`**(pyvis)와 **`static`**(Bokeh)입니다. 이들의
설정은 `graph.visualization` 아래에 있습니다(`interactive.*`, `static.*`,
그리고 `embedding_method`/`layout_method`).

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-dir ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

---

## 8. 프롬프트 튜닝 (`run-prompt-tuning`)

디렉터리에서 문서를 샘플링하고, Bedrock으로 코퍼스를 프로파일링(도메인 / 언어 /
페르소나 / 엔티티 타입)한 뒤, 검토 후 `config.yaml`에 병합할 도메인 적응
`custom_prompts` YAML 조각을 작성합니다.

### CLI 플래그 (검증됨)

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--source-dir` | **필수** | 텍스트 문서 디렉터리 (`.txt`, `.md`, `.markdown`) |
| `--output` | `tuned_prompts.yaml` | 출력 YAML 경로 |
| `--max-docs` | `20` | 샘플링할 최대 문서 수 |
| `--config-path` | — | `config.yaml` 경로 |

```bash
run-prompt-tuning --source-dir ./source --output tuned_prompts.yaml --config-path config.yaml
```

출력 YAML에는 `custom_prompts` 블록(과 감지된 도메인이 담긴 `profile`)이
포함됩니다. **검토한 후** 원하는 프롬프트를 `config.yaml`의 `custom_prompts:`
아래로 복사하세요. 프로파일링에는 일반 텍스트 포맷(`.txt`/`.md`/`.markdown`)만
읽는다는 점에 유의하세요.

---

## 9. 도메인 적응

상호 보완적인 두 가지 레버로 범용 파이프라인을 도메인 특화(의료, 법률, 금융
등)로 바꿉니다.

### A. `entity_types` (가장 저렴하고 영향력 큰 레버)

추출 프롬프트에 주입되는 엔티티 카테고리를 오버라이드합니다 — 프롬프트 재작성
불필요.

```yaml
processing:
  graph_extraction:
    entity_types:
      - "GENE: Genes, gene products, loci"
      - "DISEASE: Disorders, syndromes, conditions"
      - "DRUG: Medications, compounds, dosages"
      - "TRIAL: Clinical trials, studies, cohorts"
```

### B. `custom_prompts` 오버라이드

어떤 프롬프트든 `*_system` / `*_human` 텍스트를 오버라이드합니다(기본값 `null`
= 내장 프롬프트 사용). `{braces}` 안의 변수는 프레임워크가 채웁니다 — 그대로
두세요. 흔한 오버라이드:

```yaml
custom_prompts:
  graph_extraction_system: |
    You are a medical knowledge extractor. Extract diseases, symptoms, treatments,
    and medications and their relationships. Prioritize clinical accuracy.
  graph_extraction_human: |
    Extract medical entities and relationships from this clinical text:
    {input_text}
    Extraction Limits:
    - Maximum Entities: {max_entities_per_chunk}
    - Maximum Relationships: {max_relationships_per_chunk}

  community_report_system: |
    You are a legal analyst. Report on case law, regulatory frameworks, and
    legal precedents within each topic cluster.

  entity_extraction_system: |
    You are a financial expert. Extract companies, instruments, markets, and metrics
    from user queries.
```

사용 가능한 오버라이드 키(각각 `_system` + `_human`): `graph_extraction`,
`description_summarization`, `claim_extraction`, `graph_refinement`,
`community_report`, `answer_generation`, `context_building`,
`entity_extraction`, `keyword_expansion`, `query_refinement`,
`strategy_selection`, `keywords_extraction`(LightRAG 이중 레벨),
`global_map`(글로벌 검색 map-reduce), 그리고 프롬프트 튜닝
프롬프트(`corpus_profile`, `extraction_examples`).

**권장 흐름:** `run-prompt-tuning`을 실행해 시작점 생성 → 검토 → 유용한
프롬프트 병합 + `entity_types`를 직접 튜닝 → 재인제스천.

---

## 10. 운영 & 트러블슈팅

### IAM 권한

CLI를 실행하는 주체에게 다음 접근 권한을 부여하세요: Bedrock(모델 ID에 대한
InvokeModel / InvokeModelWithResponseStream 및 임베딩), Neptune(`use_iam: true`
시 connect / SigV4), OpenSearch(설정된 인덱스 읽기/쓰기), S3(설정된 버킷),
DynamoDB(증분 인덱싱이 켜진 경우).

> **Bedrock 리랭킹은 별도 statement가 필요합니다.** Rerank API
> (`bedrock:Rerank`, 그리고 rerank 모델에 대한 `bedrock:InvokeModel`)는
> 챗/임베딩 모델 호출과는 별개의 액션입니다. 자체 statement에서 `Resource: "*"`
> (또는 적절한 rerank 모델/추론 프로파일 ARN)를 부여하세요 — 모델 범위의
> `InvokeModel` statement만으로는 리랭킹이 인가되지 않으며, 리랭킹은 기본
> 활성화되어 있습니다(`search.reranking.enabled`). 부여할 수 없다면
> `search.reranking.enabled: false`로 설정하세요.

### 흔한 에러

- **`--source-directory is required`** — 전달하세요(또는
  `$GRAPHRAG_SOURCE_DIRECTORY` 설정). 메타데이터 전용 작업(`--verify-metadata`
  / `--repair-metadata`)은 대신 `--pipeline-id`가 필요합니다.
- **`--s3-bucket-name must be specified for S3 sync`** — `--s3-sync`는
  `--s3-bucket-name`을 요구합니다.
- **`--pipeline-id is required for --resume-from-stage`** — 재개에는 이전
  실행의 파이프라인 ID가 필요합니다.
- **`Invalid stage names provided`** — §3의 정확한 스테이지 이름을 사용하세요
  (CLI가 유효한 집합을 출력합니다).
- **`No module named 'unstructured'`** — `.md`/`.html`을 파싱하려면
  `unstructured` 추가 패키지를 설치하거나, 해당 문서를 지원 포맷으로
  변환하세요.
- **`use_iam: false`에서 OpenSearch 인증 실패** — `.env`에
  `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD`가 있는지 확인하세요.
- **LightRAG `mix`/`hybrid`가 아무것도 반환하지 않음** — 인제스천 중 관계
  인덱스가 구축되었고 키워드 추출이 키워드를 생성했는지 확인하세요(매우 짧은
  질의는 `raw_query_fallback_max_len` 하에서만 원본 질의로 폴백).
- **실행 중간에 파이프라인 실패** — `--pipeline-id <id>`로 재실행하여
  실패/미완료 스테이지부터 재개하세요. 손상 여부 확인에는 `--verify-metadata`,
  복구 시도에는 `--repair-metadata`, 깨끗하게 다시 시작하려면
  `--force-rebuild`를 사용하세요.

### 대규모 / 다국어 / 이종(heterogeneous) 코퍼스

- **대규모 코퍼스:** `processing.max_concurrency` /
  `processing.chunk_concurrency`를 높이세요(LLM 스테이지는 I/O 바운드). 그래프
  쓰기는 `indexing.neptune.index_concurrency`*와* `aws.neptune.pool_size`를
  함께 높이세요. 재실행과 다단계 작업이 재계산하지 않도록
  `indexing.opensearch.persist_embedding_cache` + `--s3-sync`를 활성화하세요.
- **다국어:** `processing.translation.source_language` / `target_language`
  (+ `additional_target_languages`)를 설정하고
  `indexing.opensearch.language_analyzers` 아래에 언어 analyzer를 추가하세요.
  번역 스테이지는 단일 언어 코퍼스에서 no-op됩니다.
- **이종 도메인:** `entity_types`를 도메인들의 합집합에 맞게 튜닝하세요(또는
  멀티테넌트 분리를 위해 `--suffix` / `indexing.additional_suffix`로 도메인별
  별도 인덱스를 운영).
- **증분:** DynamoDB를 활성화하여 대규모 코퍼스가 후속 실행에서 변경된 델타에
  대해서만 비용을 내도록 하세요.

### 비용 참고

LLM 호출이 비용을 좌우합니다. 가장 큰 요인: `graph_extraction`(청크당 1회
이상), `gleaning`(`max_rounds`만큼의 추가 패스), `community_detection` 리포트
생성, `claim_extraction`(텍스트 유닛당 1회 호출 — 기본 OFF), 질의당 답변 생성.
레버: 기계적인 스테이지에 더 저렴한 모델 사용(청킹 / 번역 / map-reduce /
description 요약은 이미 Haiku급 모델이 기본값), `gleaning.max_rounds` 제한,
필요하지 않으면 `claim_extraction` OFF 유지, 임베딩/스테이지 캐싱 활성화, 전체
재인제스천을 피하기 위한 증분 인덱싱 사용.

---

*함께 보기: 개요와 빠른 시작은 [README.md](../README.md), 아키텍처와 내부
구조는 [docs/design.md](./design.md)를 참고하세요.*
