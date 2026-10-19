# Unified Knowledge Graph RAG on AWS — 사용자 가이드

> 🇬🇧 English version: [docs/user-guide.md](./user-guide.md)

용어는 [용어집](./glossary.ko.md)을 따릅니다.

이 문서는 **unified-kg-rag-on-aws**를 실제로 사용하는 방법을 설명하는 가이드입니다.
unified-kg-rag-on-aws는 대규모 다국어 문서 코퍼스로 지식 그래프를 만들고 그 그래프를
근거로 질문에 답하는 AWS 네이티브 지식 그래프 RAG 프레임워크입니다. 두 가지 검색
방법론인 **Microsoft GraphRAG**(커뮤니티 요약)와 **LightRAG**(이중 레벨 키워드)를
하나의 스택 위에 재구현했으며, 질의마다 방법론을 고를 수 있습니다.

- 프로젝트 소개와 1분 빠른 시작은 [README.ko.md](../README.ko.md)를 참고하세요.
- 내부 구조와 아키텍처(헥사고날 레이어, 포트와 어댑터, 의존성 규칙)는
  [설계 문서](./design.ko.md)를 참고하세요.

콘솔 진입점 여섯 개(`pyproject` 스크립트로 정의)는 다음과 같습니다.

| 스크립트 | 모듈 | 용도 |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | 지식 그래프 구축과 갱신 |
| `run-rag` | `application.cli.run_rag_chain` | 그래프 질의 |
| `run-eval` | `application.cli.run_evaluation` | 검색과 생성 평가 |
| `run-visualization` | `application.cli.run_visualization` | 내보낸 그래프 렌더링(수집 없음) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | 도메인에 맞춘 프롬프트 생성 |
| `prepare-benchmark` | `application.cli.prepare_benchmark` | 공개 벤치마크(MuSiQue)를 코퍼스와 `run-eval` 데이터셋으로 변환 |

각 CLI의 `--help` 끝에는 실행 예시와 이 문서의 해당 절 링크가 있으며, 모든 플래그는 [CLI 레퍼런스](./cli.md)(영문)에 있습니다.

## 목차

1. [사전 요구사항과 설치](#1-사전-요구사항과-설치)([필수 모델](#필수-모델) 포함)
2. [설정](#2-설정)(모든 키: [설정 레퍼런스](./configuration.md))
3. [수집 (`run-ingestion`)](#3-수집-run-ingestion)
4. [질의 (`run-rag`)](#4-질의-run-rag)
5. [증분 인덱싱](#5-증분-인덱싱)
6. [평가 (`run-eval`)](#6-평가-run-eval)
7. [시각화 (`run-visualization`)](#7-시각화-run-visualization)
8. [프롬프트 튜닝 (`run-prompt-tuning`)](#8-프롬프트-튜닝-run-prompt-tuning)
9. [도메인 적응](#9-도메인-적응)
10. [운영과 문제 해결](#10-운영과-문제-해결)
11. [Python 라이브러리로 사용하기](#11-python-라이브러리로-사용하기)
12. [제한 사항](#12-제한-사항)

관련 문서: [CLI 레퍼런스](./cli.md)(영문) · [설정 레퍼런스](./configuration.md)(영문) · [모델 카탈로그](./models.ko.md) · [운영 런북](./operations.ko.md) ·
[설계 문서](./design.ko.md) · [`iac/README.md`](../iac/README.md)

---

## 1. 사전 요구사항과 설치

### 런타임

- **Python 3.10 – 3.12**
- **[uv](https://docs.astral.sh/uv/)**(권장 패키지 관리자. `pip`도 사용 가능)

### AWS 서비스

AWS 모드에만 해당합니다. [로컬 모드](#로컬-모드-aws-없이)에서는 각 서비스를 로컬
구성 요소가 대신합니다.

| 서비스 | 필수 여부 | 용도 |
|---|---|---|
| **Amazon Bedrock** | 필수 | 모든 LLM 호출, 임베딩, 재순위화(설정한 모델 ID의 모델 액세스를 활성화해야 함) |
| **Amazon Neptune** | 필수 | 지식 그래프와 질의 시점의 다중 홉 순회 |
| **Amazon OpenSearch** | 필수 | 벡터 인덱스와 BM25 어휘 인덱스(텍스트 단위, 엔티티, 커뮤니티 리포트, 관계, 주장) |
| **Amazon S3** | `--s3-sync`, `indexing.opensearch.persist_embedding_cache`, `s3://` 경로를 쓸 때만 | 단계 캐시 동기화, 임베딩 캐시, `s3://` 데이터셋, CDK 코퍼스 |
| **Amazon DynamoDB** | 증분 인덱싱을 쓸 때만 | 콘텐츠 해시로 코퍼스의 변경분을 계산하는 문서 상태 레지스트리 |

### 필수 모델

처음 실행하기 전에 다음 Bedrock 모델의 액세스를 활성화하세요(코드 기본값이며, 다른
모델을 설정하면 그 행의 모델이 바뀝니다).

| 역할 | 기본 모델 ID | 설정 키 |
|---|---|---|
| 기본 계층(추출, gleaning, 커뮤니티 리포트, 답변, 판정 모델) | `anthropic.claude-sonnet-5-5` | `models.default` |
| 빠른 계층(청킹, 번역, 요약, 라우팅, map 단계) | `anthropic.claude-haiku-5-5` | `models.fast` |
| 임베딩(인덱싱, 질의, 평가, 시각화 레이아웃) | `amazon.titan-embed-text-v2:0` | `models.embedding.model_id` |
| 재순위화 | `cohere.rerank-v3-5:0` | `models.rerank` |

- **리전.** 모든 Bedrock 호출은 `aws.bedrock.region_name`(기본값은
  `aws.region_name`인 `us-west-2`이며 네 모델이 모두 있음)으로 갑니다. 재순위화 모델이
  없는 리전(예: `ap-northeast-2`)에서는 이 값을 모델이 있는 리전으로 지정하거나
  `search.reranking.enabled: false`로 설정하세요. Bedrock으로 가는 경로가 VPC
  엔드포인트뿐인 프라이빗 VPC는 자기 리전을 유지해야 합니다.
- **추론 프로파일.** Claude Sonnet 5.5와 Haiku 5.5는 교차 리전 추론 프로파일로만
  호출할 수 있습니다. `aws.bedrock.enable_global_profile: true`를 유지하고
  `bedrock:ListInferenceProfiles` 권한을 부여하세요(프로파일을 찾지 못하면 모델 생성이
  실패하며 오류 메시지에 이 해결 방법이 나옴).
- **권한.** [IAM 권한](#iam-권한)을 참고하세요. 재순위화에는 별도 정책 문이 필요합니다.

[모델 카탈로그](./models.ko.md)에 선별된 모델 전체와 각 모델의 한도가 있습니다.

이 프레임워크는 **이미 있는** 서비스에 연결할 뿐 서비스를 만들지 않습니다.
[로컬 모드](#로컬-모드-aws-없이)로 실행하거나, 기존 서비스에 연결하거나(엔드포인트를
`config.yaml`에 적음, [§2](#2-설정)), 스택 전체를 만드는 번들 CDK 앱
[`iac/`](../iac/README.md)를 쓰세요([Quick start](../iac/README.md#quick-start)). CDK
스택의 저장소는 VPC 안에서만 접근할 수 있으므로 CLI는 엔드포인트를 환경 변수로 받는
태스크로 VPC 안에서 실행합니다([After deploy](../iac/README.md#after-deploy)).

### 설치

```bash
git clone https://github.com/awslabs/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws
uv sync              # or: pip install -e .
```

**Markdown과 HTML**을 파싱하려면 `unstructured` extra가 필요합니다
(`uv sync --extra unstructured`, Python 3.11 또는 3.12. URL 분할의 SSRF와 마크업
미이스케이프 문제를 고친 `unstructured>=0.24.1`을 요구함). extra가 없거나 Python
3.10이면 `.pdf`, `.txt`, `.csv`, `.json`만 파싱하며, 수집은 `.md` / `.html` 파일을
건너뛰고 확장자마다 경고를 한 번 남깁니다.

배포용 이미지(`docker/Dockerfile`)에는 이 extra가 빠져 있으므로(약 200MB 증가) Step
Functions로 `.md` / `.html` 파일을 수집하면 "No supported files found"로 실패합니다.
파일을 변환하거나 `--build-arg UV_EXTRAS="--extra unstructured"`로 빌드하세요.
이미지의 `docker/config.yaml`에는 엔드포인트가 없으며 CDK compute 스택이 주입합니다.

### 인증

1. **AWS 자격 증명**은 표준 자격 증명 체인에서 가져옵니다. `aws.profile_name`은
   프로파일 이름이며, `null`이면 기본 체인(환경 변수, 인스턴스 역할 등)을 씁니다.
   `aws.neptune.use_iam: true`이면 Neptune 요청에 SigV4로 서명합니다.
2. **OpenSearch**는 IAM(`aws.opensearch.use_iam: true`)을 쓰거나, `use_iam: false`이면
   `.env` 파일(`.env-template` 복사)의 사용자 이름과 비밀번호를 씁니다. `run-*` CLI가
   이 파일을 읽습니다.

   ```bash
   UKR__AWS__OPENSEARCH__USERNAME=your_opensearch_username
   UKR__AWS__OPENSEARCH__PASSWORD=your_opensearch_password
   ```

### IAM 권한

CLI를 실행하는 주체에는 Bedrock(설정한 모델 ID와 임베딩에 대한 `InvokeModel`,
`InvokeModelWithResponseStream`, `CountTokens`. `CountTokens`가 없으면 토큰 수를
추정함), Neptune(`use_iam: true`일 때 connect / SigV4), OpenSearch(설정한 인덱스
읽기·쓰기), S3(설정한 버킷), DynamoDB(증분 인덱싱을 켠 경우) 권한이 필요합니다. Bedrock
작업 세 가지는 별도 정책 문이 필요합니다.

- **`Resource: "*"`에 `bedrock:Rerank`.** Rerank API는 `InvokeModel`과 다른 리소스
  형태로 인가하므로 모델 ARN이나 추론 프로파일 ARN으로 범위를 좁힌 정책 문으로는
  거부됩니다(CDK 태스크 역할은 별도로 허용함). 이 권한이 없으면 질의마다
  `Reranking failed: ...` ERROR가 남고 재순위화하지 않은 융합 결과를
  반환합니다(`metadata.rerank.status: failed`). 권한을 줄 수 없다면
  `search.reranking.enabled: false`로 설정하세요.
- **`Resource: "*"`에 `bedrock:ListInferenceProfiles`.** 모델 해석기가 호출하는 계정
  수준 읽기 권한입니다.
- **`bedrock:ApplyGuardrail`.** 가드레일을 쓰면 해당 가드레일
  ARN(`arn:aws:bedrock:<Bedrock 리전>:<계정>:guardrail/<id>`)에 필요합니다.

### 로컬 모드 (AWS 없이)

로컬 모드는 AWS 계정 없이 호스트 한 대에서 수집, 질의, 평가를 실행하는 개발·평가용
모드입니다(기본값은 여전히 AWS). 기본 설정에서는 그래프 저장소, 벡터·어휘 저장소,
레지스트리가 모두 SQLite 파일이므로 실행할 서비스는
`http://localhost:11434`의 [Ollama](https://ollama.com) 하나뿐입니다.

```bash
uv sync --extra local
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
uv run run-ingestion --backend local --source-directory ./docs-in
uv run run-rag --backend local --query "..."
```

- **파일.** 설정 파일이 없으면 저장소 세 개가 작업 디렉터리의 `.unified-kg-rag/`에
  생기므로, 모든 명령을 같은 디렉터리에서 실행하거나 `backend: local`을 적은 설정
  파일을 쓰세요(상대 경로는 그 파일 기준으로 해석함). 처음부터 다시 하려면 세 파일을
  함께 지우거나 `indexing.reset: true`로 한 번 실행합니다.
- **증분 인덱싱**은 프리셋에서 꺼져 있으며 `providers.doc_status.enabled: true`로
  켭니다(실행 로그에 레지스트리의 절대 경로가 남음).
- **CPU 전용 PyTorch.** Linux에서 `uv sync --extra local`은 잠금 파일이 고정한 CUDA
  빌드(수 GB)를 설치합니다. CPU 빌드를 쓰려면 `uv sync` 뒤에
  `uv pip install --torch-backend cpu -e '.[local]'`을 실행하세요(나중에 그냥
  `uv sync`를 실행하면 지워짐).

`backend: local`(설정 파일에 적거나 `--backend local`)은 설정에서 비워 둔 모든 키에
아래 공급자를 채우며, 파일에 적은 키(`providers.<역할>.kind`, `aws.neptune` /
`aws.opensearch` 연결 키, 임베딩·재순위화 모델 ID)가 항상 우선합니다.
`config-template.yaml`은 이 키들을 주석으로 남겨 두었으므로 사본에 `backend: local`만
적으면 전환됩니다. 역할은 섞어 쓸 수 있습니다. 예를 들어 저장소만 로컬로 두고 모델은
Bedrock을 쓸 수 있습니다.

| 역할 | `providers.<역할>.kind` | 로컬 기본값 |
|---|---|---|
| 언어 모델 | `llm`: `bedrock` \| `openai_compatible` \| `ollama` | `ollama`, `http://localhost:11434`의 `qwen2.5:7b-instruct` |
| 임베딩 | `embedding`: `bedrock` \| `openai_compatible` \| `ollama` | `ollama`, `bge-m3`(1024차원) |
| 재순위화 | `rerank`: `bedrock` \| `cross_encoder` \| `none` | `cross_encoder`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| 그래프 저장소 | `graph`: `neptune` \| `gremlin` \| `sqlite` | `providers.graph.sqlite_path`(기본값 `.unified-kg-rag/graph.sqlite3`)의 `sqlite` |
| 벡터·어휘 저장소 | `vector`: `opensearch` \| `sqlite` | `providers.vector.sqlite_path`(기본값 `.unified-kg-rag/vector.sqlite3`)의 `sqlite` |
| 문서 상태 레지스트리 | `doc_status`: `dynamodb` \| `sqlite` | `providers.doc_status.sqlite_path`(기본값 `.unified-kg-rag/doc-status.sqlite3`)의 `sqlite` |

`gremlin`(TinkerPop Gremlin Server)과 `opensearch`(로컬 OpenSearch)는 프리셋이
`localhost`로 지정하는 `aws.neptune`과 `aws.opensearch` 키로
연결합니다([컨테이너](#컨테이너로-띄운-로컬-저장소)).

- **모델.** Bedrock이 아닌 LLM을 쓰면 `models.default`가 필수이고(프리셋이 지정함)
  `models.fast`는 그 값을 따릅니다. `providers.llm.context_window_size`와
  `max_output_tokens`로 모델 한도를 알려 주며(컨텍스트 창은 Ollama에 `num_ctx`로
  전달), 프리셋은 `search.token_manager.max_context_tokens`(22000)를 기본
  32,768토큰 창에 맞춥니다. 이 값은 그 창 그대로인 Bedrock 외 LLM에만 적용되며, Bedrock
  LLM이나 `context_window_size`를 지정한 경우에는 `30000`을 유지합니다. OpenAI 호환
  서버(vLLM, llama.cpp server, LM Studio)에는
  `base_url`(`/v1` 루트)과, 필요하면 `api_key`를 지정합니다.
- **임베딩.** Bedrock이 아닌 공급자에서는 `models.embedding.model_id`와 `dimension`이
  필수입니다. 차원은 벡터 매핑에 고정되므로 둘 중 하나를 바꾸면 전체를 다시 인덱싱해야
  하며, `bge-m3`가 아닌 모델은 그 모델의 차원도 지정해야 합니다.
- **재순위화.** 크로스 인코더는 첫 질의 때 Hugging Face에서 모델을 내려받습니다.
  `[local]` extra가 없거나 `kind: none`이면 융합 순서를 그대로 쓰며, 로그에 한 번 남고
  `metadata.rerank.status`가 `unavailable`(`kind: none`이면 `disabled`)로 보고됩니다.
- **내장 저장소.** `sqlite` 저장소는 각각 처음 쓸 때 만들어지는 파일 하나이며(상대
  경로 `sqlite_path`는 다른 경로와 같은 방식으로 해석), `indexing.neptune`과
  `indexing.opensearch` 설정이 그대로 적용됩니다.
- **토큰 계산**은 문자 체계를 고려한 추정치를 씁니다(BPE 인코딩은
  `providers.llm.tokenizer: tiktoken`).
- **사용자 정의 kind**는 `adapters.provider_registry.register_provider`로 빌더를
  등록합니다([설계 문서 §15](./design.ko.md#15-확장-가이드)).

#### AWS 모드와 다른 점

- **품질.** 작은 로컬 모델은 추출 품질이 눈에 띄게 낮습니다(실패한 청크가 늘고 엔티티
  해석과 리포트가 약해짐). 하드웨어가 감당하는 가장 큰 모델을 쓰세요. `langchain` /
  `ragas` 심사도 점수의 신뢰도가 낮습니다.
- **모델 동작.** 프롬프트 캐싱과 추론 노력(effort)이 없고, 출력 한도에서 잘린 응답을
  감지하지 못하며(`max_output_tokens`는 모델이 실제로 생성할 수 있는 값으로 둠), 토큰
  수와 그에 따른 컨텍스트 예산은 추정치입니다. 서버가 호출을 거부하거나(HTTP 4xx)
  계속 연결되지 않으면 "찾을 수 없음"으로 답하지 않고 클라이언트의 오류로 질의를
  실패시킵니다.
- **가드레일.** `aws.bedrock.guardrail.identifier`를 지정하면 질의 체인은 여전히
  Bedrock `ApplyGuardrail`로 질의와 답변을 검사합니다(AWS 자격 증명 필요).

**내장 벡터 저장소**는 OpenSearch와 다음이 다릅니다.

- `nori`, `kuromoji`, `smartcn`으로 매핑한 텍스트는 형태소 분석 대신 서로 겹치는 CJK
  문자 바이그램으로 나눕니다. 긴 단어 안의 이름도 찾지만(홍길동으로부터의 홍길동) 순위는
  다릅니다.
- `english`는 `[local]` extra의 `snowballstemmer`가 있을 때만 어간을 추출하며(없으면
  경고 한 번) 이모지는 색인하지 않습니다. 퍼지 매칭이 없으므로 오타와, `standard`
  분석기에서는 굴절형("vendors")도 벡터 검색으로만 찾습니다.
- kNN은 정확 검색(필터를 통과한 문서를 모두 채점)이므로 비용이 인덱스 크기에 비례합니다.
- 검색하는 인덱스를 메모리에 올리며(벡터 필드마다 약 문서 수 x 차원 x 4바이트), 문서 약
  100,000개를 기준으로 설계했습니다(더 크면 경고를 남기므로 OpenSearch를 쓰세요). 쓰기
  뒤 첫 질의는 인덱스를 다시 읽고, 첫 어휘 검색 질의는 검색하는 필드를 다시 분석하며, 120단어
  문서 50,000개 기준 약 10초가 걸립니다. `english`에서 서로 다른 단어가 약 65,000개를 넘으면
  약 세 배 느립니다.

**SQLite 파일**은 한 번에 한 프로세스만 씁니다. 저장소 파일마다 수집은 하나씩
실행하고(질의는 함께 읽을 수 있음), `search.store_timeout_seconds`보다 오래 기다린
쓰기는 "stayed locked"로 실패합니다. 파일은 NFS, SMB, Amazon EFS가 아닌 로컬 디스크에
두세요([자주 보는 오류](./operations.ko.md#로컬-모드)).

#### 공급자 플래그

`run-ingestion`, `run-rag`, `run-eval`, `run-prompt-tuning`, `run-visualization`은
공급자 플래그(`--backend`, `--llm-provider`, `--llm-model`, `--embedding-provider`,
`--embedding-model`, `--embedding-dimension`, `--rerank-provider`,
`--model-base-url`)를 받습니다. 플래그마다 읽어 들인 설정의 해당 키를 덮어쓰며(파일에
적은 키 포함), 생략하면 설정을 그대로 씁니다(`run-visualization`은 모델이나 저장소를
호출하지 않으므로 받기만 함). 플래그별로 설정하는 키는
[CLI 레퍼런스](./cli.md#provider-flags)(영문)에 있습니다.

#### 컨테이너로 띄운 로컬 저장소

[`docker/compose.local.yaml`](../docker/compose.local.yaml)은 저장소를 서버로
실행합니다. Gremlin Server(메모리 기반 TinkerGraph), `analysis-nori`를 설치한 단일 노드
OpenSearch 2.13(보안 플러그인 끔), 그리고 `models` 프로필에서 Ollama를 띄웁니다. 이
OpenSearch는 Amazon OpenSearch Service와 같은 엔진이므로 AWS에 배포하기 전에 순위를
확인할 때 쓰세요(AWS의 `nori`는 [다국어 수집](#다국어-수집) 참고).
[`docker/config.local.yaml`](../docker/config.local.yaml)은
`providers.graph.kind: gremlin`과 `providers.vector.kind: opensearch`를 지정하고 증분
인덱싱을 켭니다. Docker Compose 2.23.1 이상이 필요합니다.

```bash
uv sync --extra local
docker compose -f docker/compose.local.yaml --profile models up -d --wait
docker compose -f docker/compose.local.yaml exec ollama ollama pull qwen2.5:7b-instruct
docker compose -f docker/compose.local.yaml exec ollama ollama pull bge-m3
uv run run-ingestion --config-path docker/config.local.yaml --source-directory ./docs-in
uv run run-rag --config-path docker/config.local.yaml --query "..."
docker compose -f docker/compose.local.yaml down -v
```

- **`down -v`**는 저장소 데이터를 지우지만 SQLite 레지스트리(저장소 루트의
  `.unified-kg-rag/doc-status.sqlite3`)는 지우지 않습니다. 새 저장소에 수집하기 전에 이
  파일을 지우거나 `indexing.reset: true`로 설정하세요. 그러지 않으면 증분 실행이
  아무것도 쓰기 전에 멈춥니다.
- 모델은 자동으로 내려받지 않으며, 프리셋은 SigV4나 인증 없이 `ws://`와 `http://`로
  연결합니다.
- 모델 없이 저장소만 확인하는 스모크 테스트:
  `LOCAL_STORES=1 uv run pytest tests/integration/test_local_stores.py`.

Neptune과 다른 점은 다음과 같습니다.

- TinkerGraph는 메모리에만 그래프를 두므로, Gremlin Server를 다시 시작한 뒤의 증분
  실행은 아무것도 쓰기 전에 멈추고 `indexing.reset: true`(또는 레지스트리 삭제)를
  요청합니다.
- 정점을 만들 때 쓴 목록 속성은 중복 값을 유지합니다.
- 쓰기는 `id` 속성으로 요소를 찾으며 compose 파일의 서버는 이 속성에 인덱스를 만듭니다
  (인덱스가 없으면 그래프가 커질수록 느려짐). 개발용 규모의 코퍼스에 쓰세요.
- 저장소에는 인증도 TLS도 없으므로 compose 파일은 `127.0.0.1`에만 바인딩하고 Linux
  capability를 모두 버립니다.

#### 로컬 저장소와 Bedrock 모델

`backend: local`을 적은 설정에서 `providers.llm.kind`, `providers.embedding.kind`,
`providers.rerank.kind`를 `bedrock`으로 바꾸고(이때 Bedrock 모델 ID는 카탈로그로 검증함)
Bedrock 액세스 권한이 있는 AWS 자격 증명을 쓰세요. 컨테이너를 쓰려면 저장소만
띄우고(`docker compose -f docker/compose.local.yaml up -d --wait`)
`docker/config.local.yaml` 사본을 고칩니다.

---

## 2. 설정

템플릿으로 설정 파일을 만들고 모든 CLI에 `--config-path config.yaml`로 지정하세요.

```bash
cp config-template.yaml config.yaml
```

설정 파일에는 바꿀 키만 적고, 나머지 키는 기본값을 유지합니다. 모든 키의 타입, 기본값,
설명은 코드에서 생성한 [설정 레퍼런스](./configuration.md)(영문)에 있고, 이 절에서는 표
한 줄로 설명하기 어려운 내용을 다룹니다.

### 설정을 읽는 순서

뒤 단계가 앞 단계를 덮어씁니다.

1. **내장 기본값**(`unified_kg_rag/domain/models/config.py`). `--config-path` 없이
   실행하면 CLI는 이 기본값에 환경 변수만 적용합니다.
2. **`backend` 프리셋**: `backend: local`(또는 `--backend local`)은 파일과 환경 변수가
   지정하지 않은 모든 키에 로컬 공급자와 저장소를 채웁니다([로컬 모드](#로컬-모드-aws-없이)).
3. **YAML 파일**. 값이 딕셔너리인 키(예: `logging.library_levels`)는 기본 딕셔너리와
   병합되지 않고 통째로 대체됩니다.
4. **환경 변수**([아래](#환경-변수로-덮어쓰기)).
5. **공급자 플래그**(`--backend`, `--llm-provider` 등)는 마지막에 적용됩니다. 설정을
   파일과 환경 변수가 지정한 값으로 다시 만들므로, `--backend`를 바꾸면 이전 프리셋이
   채운 값만 사라지고 사용자가 지정한 값은 남습니다.

파일은 읽을 때 검증합니다. 타입이나 값이 틀린 경우(`Configuration validation error: ...`),
올바른 YAML이 아닌 파일, 존재하지 않는 `--config-path` 파일, 알 수 없는 키(파일이나
`UKR__` 변수)는 사용법 오류(종료 코드 2)입니다. 철자가 틀린 키는 아무 표시 없이 설정을
기본값으로 남기므로 알 수 없는 키를 거부하며, `--allow-unknown-keys`(`get_config`에서는
`allow_unknown_keys=True`)를 주면 키마다 WARNING만 남깁니다. 위치가 바뀐 키는 대신할 키를
알려 주며 실패합니다([Moved keys](./configuration.md#moved-keys)).

`max_attempts` 키는 모두 첫 시도를 포함한 총 시도 횟수이며, `1`이면 재시도하지
않습니다.

### 대부분의 실행에서 바꾸는 설정

| 키 | 기본값 | 바꿀 때 |
|---|---|---|
| `aws.region_name` | `"us-west-2"` | 저장소가 다른 리전에 있을 때(`aws.bedrock.region_name`이 없으면 Bedrock도 따름) |
| `aws.neptune.endpoint`, `aws.opensearch.endpoint` | `null` | AWS에서는 항상(CDK 스택이 주입함) |
| `providers.doc_status.enabled` | `false` | 증분 인덱싱([§5](#5-증분-인덱싱)) |
| `models.default`, `models.fast` | Claude Sonnet 5.5, Claude Haiku 5.5 | 다른 모델([참고 사항](#모델-선택-참고-사항)) |
| `processing.max_concurrency` | `40` | `ThrottlingException` 재시도가 보이면 낮추고, 할당량이 허용하면 높임 |
| `processing.graph_extraction.entity_types` | 일반 유형 7개 | 도메인 적응([§9](#9-도메인-적응)) |
| `processing.translation.target_language` | `"en"` | 코퍼스나 답변 언어가 다를 때([다국어 수집](#다국어-수집)) |
| `processing.claim_extraction.enabled` | `false` | `local` 검색에 주장을 넣을 때(텍스트 단위마다 LLM 호출 1회 추가) |
| `processing.document_parsing.index_value` | `null`(`default`) | 코퍼스나 테넌트 분리([인덱스 접미사](#인덱스-접미사)) |
| `graph.community_detection.enabled` | `true` | LightRAG만 쓰고 수집을 가볍게 할 때(`global`, `drift`에는 필요) |
| `search.reranking.enabled` | `true` | 리전에 재순위화 모델이 없거나 재순위화를 쓰지 않을 때 |
| `search.token_manager.max_context_tokens` | `30000` | 컨텍스트를 늘리거나 비용을 줄이려고 줄일 때([컨텍스트 예산](#컨텍스트-예산)) |
| `indexing.reset` | `false` | 코퍼스 전체로 저장소를 다시 만드는 한 번의 실행(격리된 문서는 건너뜀) |
| `aws.bedrock.guardrail.identifier` | `null` | 질의와 답변 검사([가드레일](#가드레일)) |
| `observability.enabled` | `false` | OpenTelemetry 트레이스([관측성](#관측성)) |

### 모델 선택 참고 사항

`models` 섹션이 역할마다 쓸 모델을 정하며, 모델 ID는 그 계열의 공급자
(`providers.llm.kind`, `providers.embedding.kind`, `providers.rerank.kind`)를 기준으로
검사합니다. 모든 LLM 역할은 두 계층 중 하나에서 실행되므로 한 줄로 파이프라인 전체를
바꿀 수 있고, `models.roles` 항목(모델 ID, 또는 다른 계층으로 옮기는 `default` /
`fast`)이 계층 설정보다 우선합니다.

| 계층 | 역할 |
| --- | --- |
| `default` | `fixing`, `graph_extraction`, `gleaning`, `claim_extraction`, `community_report`, `entity_extraction`, `answer_generation`, `evaluation` |
| `fast` | `chunking`, `translation`, `description_summary`, `query_translation`, `strategy_selection`, `query_rewrite`, `community_relevance`, `global_map`, `global_reduce`, `drift_primer`, `drift_action` |

```yaml
models:
  default: "openai.gpt-6-sol"          # every default-tier role
  roles:
    answer_generation: "anthropic.claude-opus-5-5"   # one role pinned
    query_rewrite: default                          # moved to the other tier
```

- 임베딩 모델(`models.embedding.model_id`)은 인덱스, 질의, RAGAS, 시각화 배치에 함께
  쓰이며, 이 모델이나 `dimension`을 바꾸면 다시 인덱싱해야 합니다. Bedrock에서는 임베딩과
  재순위화 모델 ID를 정해진 목록에서만 고르고([모델 카탈로그](./models.ko.md#모델-id와-기능-정보-레코드)),
  다른 공급자는 어떤 모델 이름이든 받습니다.
- 선택하지 않은 공급자의 설정(Bedrock 모델을 쓰는데 `providers.llm.temperature`를
  지정한 경우 등)은 무시하고 WARNING을 남깁니다.
- **출력 상한.** Bedrock은 요청을 시작할 때 입력 + `max_tokens`를 분당 토큰 할당량에서
  미리 확보하므로, 모델 최대값을 요청하면 동시 수집 호출이 스로틀링됩니다.
  `aws.bedrock.default_max_output_tokens`가 16384인 이유입니다. 출력이 긴 프롬프트는 더
  높은 하한을 계산하고 그 값이 우선하며(기본으로 그래프 추출과 gleaning 23192, 문서 번역
  18992, 주장 29792), 청크당 한도를 올리면 함께 올라갑니다.
- 선별된 모델, 요청 형식, 프롬프트 캐싱, `aws.bedrock.model_overrides`는
  [모델 카탈로그](./models.ko.md)에 있습니다.

### 컨텍스트 예산

`search.token_manager.max_context_tokens`(`30000`, upstream LightRAG의 예산)는 답변
프롬프트가 받는 검색 컨텍스트입니다. `models.roles.answer_generation`가 출력 예약분과
함께 받을 수 있는 크기로 줄이며 줄일 때 경고를 남기고, `null`이면 그 모델의 컨텍스트
창에서 도출합니다. 유형별 비율과 `local` 슬롯 할당은
[레퍼런스](./configuration.md#searchtoken_manager)(영문)에 있습니다.

### 가드레일

`aws.bedrock.guardrail.identifier`를 지정하면 질의마다 두 텍스트를 `ApplyGuardrail`
API로 검사합니다. 가드레일은 `aws.bedrock.region_name`에 있어야 합니다.

- **검사 대상.** 메모리, 재작성, 라우팅, 검색보다 먼저 한 번 검사하는 사용자 원본
  질의(`check_input`)와 최종 답변(`check_output`. 스트리밍 답변은 한 덩어리로 전달됨)이며,
  이 둘만 과금합니다.
- **결과.**
  - 차단된 질의에는 가드레일 메시지로 답하고, 검색도 메모리 저장도 하지 않으며
    `RAGOutput.metadata.blocked`를 설정합니다. 마스킹된 질의는 마스킹된 텍스트로만
    계속 처리합니다.
  - 차단되거나 마스킹된 답변은 가드레일의 텍스트로 바꿉니다(차단이면 `sources`도 비움).
    마스킹은 답변에만 적용되고 출처(검색된 그대로의 코퍼스 텍스트)에는 적용되지 않으므로,
    호출자가 보면 안 되는 텍스트는 코퍼스에서 빼세요.
  - 개입 내역은 `RAGOutput.metadata.guardrail`, WARNING 로그, 트레이스에 남습니다.
    가드레일 API 오류는 질의를 실패시키거나, `processing.ignore_errors`면 `metadata.error`가
    붙은 저하된 답변을 돌려주며, 검사하지 않은 채로 답하지 않습니다.
- **모델 호출에는 붙이지 않음.** 질의 안의 모델 호출은 가드레일 없이 실행합니다. 모델
  호출에 붙인 가드레일은 프롬프트 전체를 평가하므로 `PROMPT_ATTACK` 필터가 검색된 텍스트를
  사용자 입력으로 판정해 잘못 차단하고, 비용도 컨텍스트에 비례해 늘어납니다
  ([프롬프트 공격](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-prompt-attack.html)).
  컨텍스트 근거(contextual grounding) 검사는 하지 않습니다.
- **수집과 평가 심사 모델은 가드레일 없이 실행**(`run-eval`의 답변은 검사하는 질의 경로를
  거침). `NAME`을 익명화하는 PII 가드레일은 엔티티 이름을
  `{NAME}`으로 바꿔 서로 다른 사람을 합치고, `PROMPT_ATTACK` 필터는 지시문처럼 보이는
  코퍼스 텍스트를 차단할 수 있습니다. 질의 경로에서도 `NAME` 익명화는 질문이 검색에 쓸
  이름을 지우므로, `iac/`가 만드는 기본 가드레일은 이메일, 전화번호, 카드 번호만
  익명화하고 `NAME`은 익명화하지 않습니다([iac/README.md](../iac/README.md#guardrail)(영문)).
- **사용자 코드**에서 질의가 아닌 작업에 `setup_chain`이나 `get_model`을 쓰면
  `model_purpose=ModelPurpose.INGESTION`(또는 `EVALUATION`. `from unified_kg_rag import
  ModelPurpose`)을 넘기세요. 표시하지 않은 호출은 `QUERY`로 간주하므로 질의 시점의 일시적
  오류 재시도가 적용되고, 응답 캐시와 수집 effort는 적용되지 않습니다.

### S3 캐시 암호화

`aws.s3.encryption.encryption_type: BUCKET_DEFAULT`는 암호화 헤더를 보내지 않으므로
버킷 기본 암호화가 적용됩니다(CDK 스택의 `use_cmk=true`면 CMK이며, `AES256`을 강제하면
이 키를 우회함). SSE-KMS 버킷을 재사용하면 쓰기 주체에 그 키의 `kms:GenerateDataKey`와
`kms:Decrypt` 권한이 필요합니다.

### 로그에 남는 내용

`INFO` 이상에서 사용자 텍스트와 코퍼스 텍스트는 길이, 개수, ID, 항상 같은 짧은
해시(`query: len=42 sha=1a2b3c4d`)로만 기록합니다. 질의 원문, 다시 쓴 질의, 엔티티
이름, 모델 원본 출력은 `DEBUG`에서만 기록하며, 예외 메시지에는 모델 출력을 넣지
않습니다. `DEBUG` 로그(`logging.level: DEBUG`, `UKR__LOGGING__LEVEL=DEBUG`,
`--verbose`)와 `logging.library_levels`에서 `DEBUG`로 둔 라이브러리(`botocore`는 요청
본문을 기록함)에는 사용자 데이터와 코퍼스 데이터가 들어 있다고 보세요.

`logging.log_to_file`이 `false`가 아니면 CLI는 `logging.log_file_path`(`logs/log.txt`,
날짜별 `log_YYYYMMDD.txt`)에도 씁니다. 라이브러리로 쓰면 패키지는 핸들러를 설정하지
않으며, 호스트의 로깅 설정이 적용되고 구조화된 필드(`pipeline_id`, `trace_id` 등)는
로그 레코드의 속성으로 들어갑니다.

### 사용자 정의 프롬프트

지능형 청커(`TextChunkingPrompt`), 문서 번역(`TextTranslationPrompt`), 질의 번역
(`TranslationPrompt`), 전역 검색의 커뮤니티 평가(`CommunityRelevancePrompt`)와 축약 단계
(`MapReduceSummaryPrompt`)의 프롬프트를 뺀 모든 프롬프트에는 `custom_prompts.*_system` /
`*_human` 재정의 키가 있습니다(기본값 `null` = `unified_kg_rag/domain/prompts/`의 내장
프롬프트). 이 다섯 개는 재정의할 수 없습니다. 필요한 것만 재정의하고
나머지는 `null`로 두세요([§9](#9-도메인-적응)). 재정의는 설정을 읽을 때 검사합니다. 알
수 없는 `{variables}`와 빠진 데이터 변수(`{input_text}` 등)는 오류이며, 중괄호를
글자 그대로 쓰려면 `{{` / `}}`로 씁니다.

### 환경 변수로 덮어쓰기

모든 설정 키는 `UKR__` 뒤에 키 경로를 붙인 환경 변수로 지정할 수 있습니다. 경로의
마디 사이에는 밑줄 두 개를 쓰고 대소문자는 가리지 않습니다.
`UKR__AWS__NEPTUNE__ENDPOINT`는 `aws.neptune.endpoint`를 지정하며, 자유 형식 매핑
다음 마디는 그 매핑의 키를 가리킵니다(`UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`).
환경 변수는 설정 파일과 기본값보다 우선합니다. 값은 파일에 같은 값을 쓴 경우처럼
변환하고 검증합니다(불리언은 `true`/`1`/`yes`/`on`, 목록이나 매핑은 JSON, 예:
`UKR__SEARCH__AUTO_ROUTABLE_STRATEGIES='["local", "mix"]'`). 값이 잘못되었거나 어떤
키도 가리키지 않는 `UKR__` 변수는 그 변수를 밝힌 사용법 오류(종료 코드 2)이며,
`--allow-unknown-keys`를 주면 알 수 없는 변수는 WARNING만 남깁니다.

- `AWS_PROFILE`과 `AWS_REGION`은 파일이 `aws.profile_name`과 `aws.region_name`을
  지정하지 않았을 때만 그 값을 채우므로, 셸이 `AWS_REGION`을 내보내도(SSO 도우미와
  CloudShell이 그렇게 함) 파일에 적은 리전은 바뀌지 않습니다. 이때
  `aws.bedrock.region_name`이 `null`이면 `AWS_REGION`이 Bedrock 호출 리전도 정합니다.
  `AWS_DEFAULT_REGION`은 읽지 않습니다.
- `UKR_SOURCE_DIRECTORY`, `UKR_PIPELINE_ID`(설정 키가 아니므로 밑줄 하나)와
  `UKR_METRICS_ENVIRONMENT`는 `run-ingestion`이 읽습니다([CLI 레퍼런스](./cli.md#run-ingestion)).
  CDK 태스크가 받는 변수는
  [`iac/README.md`](../iac/README.md#2-build-and-push-the-app-image)에 있습니다.
- CLI는 python-dotenv로 현재 작업 디렉터리의 `./.env`만 읽고 상위 디렉터리의 파일은
  찾지 않습니다(그러지 않으면 `AWS_PROFILE`을 설정한 `~/.env`가 모르는 사이에 계정을
  바꿀 수 있음). 이미 설정된 변수가 우선합니다. 패키지를 가져오는 것만으로는 `.env`를
  읽지 않고 `os.environ`을 바꾸지 않으므로, Python 코드에서는 직접 읽습니다
  (`get_config()` 전에 `dotenv.load_dotenv()`).

> **LangSmith 추적은 내용을 업로드합니다.** `LANGSMITH_TRACING=true`(또는
> `LANGCHAIN_TRACING_V2=true`)이면 LangChain은 프롬프트, 검색된 본문, 모델 출력을 포함해
> 추적한 실행을 모두 LangSmith로 보냅니다. CLI는 시작할 때(`setup_logging`에서)
> WARNING을 남기지만 끄지는 않으므로, 기밀 코퍼스로 실행하기 전에 이 변수를 해제하세요.

### 관측성

추적은 기본적으로 꺼져 있습니다. `observability.enabled: true`로 켜면 수집 실행과
질의가 각각 하나의 OpenTelemetry 트레이스가 되며, OTLP를 받는 백엔드(Jaeger, Grafana
Tempo, OpenTelemetry Collector, Langfuse, Phoenix)에서 볼 수 있습니다. `otel`
extra(`uv sync --extra otel`)가 필요합니다. extra 없이 추적을 켜면 시작할 때 extra
이름을 알려 주는 `ConfigurationError`로 실패하고, 추적이 꺼져 있으면 OpenTelemetry 관련
모듈을 전혀 불러오지 않습니다.

- `observability.backend`: `none`(호스트 애플리케이션이나 `opentelemetry-instrument`가
  설정하는 프로세스 전역 트레이서 공급자), `otlp`(패키지가 설치하는 OTLP 익스포터),
  `langfuse` / `phoenix`(그 익스포터를 해당 벤더로 보냄). 나머지 키(`capture_content`,
  `redact_patterns`, `sample_rate`, `service_name`, `export_timeout_seconds`)는
  [레퍼런스](./configuration.md#observability)에 있습니다.
- 익스포터는 설정 파일이 아니라 OpenTelemetry 표준 환경 변수로 설정하므로 수집기 키가
  설정 파일에 들어가지 않습니다. `OTEL_EXPORTER_OTLP_ENDPOINT`(기본값
  `http://localhost:4318`), `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TIMEOUT`,
  `OTEL_EXPORTER_OTLP_PROTOCOL`(`http/protobuf`, 4317 포트는 `grpc`)을 씁니다.
- 질의의 모든 스팬에 `session.id`(`RAGInput.conversation_id`)와, 지정한 경우
  `user.id`(`RAGInput.user_id`)가 붙으며 Langfuse와 Phoenix는 이 이름으로 트레이스를
  묶습니다. 사용자 ID는 이메일 주소가 아닌 불투명한 ID를 넘기세요. 스팬 안에서 남긴 로그
  레코드에는 그 스팬의 `trace_id`와 `span_id`가 들어갑니다.
- `capture_content: true`가 아니면 스팬에 질의 본문, 문단, 모델 출력이 들어가지 않습니다.

Jaeger, Langfuse, Phoenix 절차와 [트레이스 구조](./operations.ko.md#트레이스-구조)(모든
스팬과 속성, 벤더 속성 이름, 알려진 제한), 전송 제한은 [운영 런북](./operations.ko.md#관측성)에
있습니다. CDK 스택의 CloudWatch 추적은
[iac/README.md](../iac/README.md#cloudwatch-tracing-optional)(영문)에 있습니다.

---

## 3. 수집 (`run-ingestion`)

수집은 문서 디렉터리를 그래프 저장소와 벡터 저장소의 지식 그래프로 바꿉니다.
`run-ingestion --source-directory <dir>`이 12개 단계([설계 §4](./design.ko.md#4-수집-파이프라인)에
그림이 있음)를 실행하며 단계마다 체크포인트를 남기고, 같은 `--pipeline-id`로 다시
실행하면 마지막으로 유효한 단계 다음부터 재개합니다. 문서 상태 레지스트리를 켜면
증분으로 동작합니다([§5](#5-증분-인덱싱)).

모든 플래그와 기본값은 [CLI 레퍼런스](./cli.md#run-ingestion)(영문)에 있습니다.

### 파이프라인 단계 12개

실행 순서대로이며, `--enabled-stages` / `--resume-from-stage`에는 단계 **이름**(대소문자
무관)을 씁니다.

1. **`document_parsing`**: 형식별 텍스트 추출(`.pdf`, `.txt`, `.csv`, `.json`.
   `unstructured` extra가 있으면 `.md`/`.html`도).
2. **`document_loading`**: 이번 실행의 코퍼스 구성(MinHash 중복 제거, 증분 필터).
   `document_parsing`을 끄면 미리 파싱된 `Document` `.json` 파일을 읽음.
3. **`text_chunking`**: 텍스트 단위로 분할(`processing.chunking`).
4. **`translation`**(선택): `target_language`로 번역. 원문 언어와 대상 언어가 같고
   추가 대상 언어가 없으면 아무 일도 하지 않음.
5. **`graph_extraction`**: 청크마다 엔티티와 관계 추출(LLM).
6. **`gleaning`**(선택): 정제 패스(`processing.gleaning`).
7. **`graph_resolution`**: 중복 엔티티와 관계 병합(이름 변형,
   `processing.fuzzy_entity_resolution`이면 비슷한 이름도).
8. **`claim_extraction`**(선택, 기본은 꺼짐): 사실 주장 추출.
9. **`claim_resolution`**(선택): 주장 중복 제거.
10. **`graph_analysis`**: 그래프를 만들고 `graph.analysis.statistics`를 기록.
11. **`community_detection`**: Leiden 군집화와 LLM 커뮤니티 리포트.
12. **`indexing`**: OpenSearch와 Neptune에 씀(켜져 있으면 레지스트리에도).

### 예시

```bash
# Full build
run-ingestion --source-directory ./documents --config-path config.yaml

# Keep the stage cache in S3 so a fresh task can resume
run-ingestion --source-directory ./documents --config-path config.yaml \
  --s3-sync --s3-bucket-name your-bucket

# Resume an interrupted run from a stage
run-ingestion --source-directory ./documents --config-path config.yaml \
  --pipeline-id <id> --resume-from-stage graph_extraction

# Run only specific stages
run-ingestion --source-directory ./documents --config-path config.yaml \
  --enabled-stages DOCUMENT_PARSING,TEXT_CHUNKING,GRAPH_EXTRACTION
```

**재개와 강제 재구축.** `--force-rebuild` 없이 다시 실행하면 완료된 단계는 캐시를
쓰고 건너뜁니다.

- `--pipeline-id <id>`는 그 실행을 처음으로 실패했거나 끝나지 않은 단계부터
  재개합니다(`--enabled-stages`가 있으면 그 범위 안에서). 그 뒤에 켠 단계(gleaning,
  주장)는 끝나지 않은 단계로 봅니다. `--pipeline-id`가 없는 실행은 새 ID를 받습니다.
- 코퍼스나 설정이 바뀌면 CLI는 출력이 더는 캐시에 없는 완료 단계를 나열하고, 다시
  계산하기 전에 묻습니다(터미널이 없으면 `--yes`).
- 증분 인덱싱에서 델타를 계산한 뒤 추출 설정이 바뀐 재개는 `document_loading`부터 다시
  시작하므로 델타가 새 설정을 따릅니다.
- `--force-rebuild`는 모든 단계를 다시 계산하고 캐시를 새로 씁니다.

**S3 동기화**(`--s3-sync`)는 단계 캐시를 `s3://<bucket>/<prefix>/...`에 두고 단계가
끝날 때마다 업로드하므로, 새 태스크가 거기서 재개합니다. SIGTERM(ECS `StopTask`, Step
Functions 시간 초과)을 받으면 실행 중인 단계의 체크포인트를 업로드하고 143으로
종료합니다. 단계 사이의 업로드 실패는 다음 단계에서 다시 시도하고, 처음 다운로드나
마지막 업로드가 실패하면 `CacheSyncError`로 실행이 실패합니다. 만료와 다른 캐시는
[캐시와 접두사](./operations.ko.md#캐시와-접두사)에 있습니다. 실행 사이에 바뀌지 않은
텍스트를 다시 임베딩하지 않으려면 `indexing.opensearch.persist_embedding_cache: true`를
설정하세요.

**LLM 응답 캐시**(`cache.llm_responses`, 기본값 켜짐). 수집 단계의 모델 응답은 모두 실행의
캐시 디렉터리 `llm-cache/`에 두고 함께 동기화하므로, 같은 pipeline id로 다시 실행하면
중지되거나 비정상 종료된 단계가 하지 않은 호출만 다시 합니다. 키는 렌더링한 프롬프트와
모델 ID, 최대 토큰, temperature, 추론 설정입니다.

- 이전 실행의 응답만, 실행마다 한 번까지만 씁니다. 출력 토큰 한도에서 잘린 응답은
  보관하지 않습니다. temperature가 0보다 크면 다시 실행할 때 이전 샘플을 재사용합니다.
- `--force-rebuild`는 캐시에서 답하지 않으며, 질의와 평가 호출은 캐시하지 않습니다.
- 캐시는 모델 호출마다 수 KB씩 커지고 실행을 시작할 때 메모리로 읽습니다.
  `cache.llm_responses.flush_every`개 응답 단위의 세그먼트로 쓰므로, 예고 없이 종료된
  태스크가 잃는 것은 세그먼트 하나 이하입니다. 중지하면 실행 중인 모델 호출의 재시도도
  끝납니다.

### 인덱스 접미사

코퍼스의 인덱스와 그래프 레이블 이름은 모두 `<prefix>-<suffix>`이며
`indexing.additional_suffix`를 지정하면 `-<additional_suffix>`가 붙습니다. 이
`<suffix>`가 **인덱스 접미사**입니다. 수집은
`processing.document_parsing.index_value`에서 가져오고(플래그 없음), 질의는
`--suffix`(Python에서는 `RAGInput.suffix`)로 고릅니다. 둘 다 기본값은 `default`입니다.
테넌트나 코퍼스 버전마다 인덱스 접미사와 설정 파일을 하나씩 두고,
`indexing.additional_suffix`는 양쪽에서 같게 하세요. 증분 인덱싱과 삭제는 인덱스 접미사
단위입니다(§5).

접미사는 이름을 나눌 뿐 접근을 막지 않습니다. 질의할 수 있는 호출자는 어떤 접미사든
지정할 수 있으므로, 멀티 테넌트 서비스는 인증된 호출자를 기준으로 서버에서 접미사를
정해야 합니다. [운영 런북: 질의 권한과 테넌트 격리](./operations.ko.md#질의-권한과-테넌트-격리)를
참고하세요.

### 다국어 수집

코퍼스의 주 언어와 인덱싱할 대상 언어를 지정하세요.

```yaml
processing:
  translation:
    enabled: true
    source_language: "ko"
    target_language: "en"
    additional_target_languages: ["ja"]   # index additional languages too
```

`source_language == target_language`이고 추가 대상 언어가 없으면 `enabled: true`여도
번역 단계를 비용 없이 건너뜁니다.

- **분석기**는 `indexing.opensearch.language_analyzers`(예: `ko: nori`)에서 정하며,
  나머지 언어는 `default_analyzer`를 씁니다. 청크 `text`는 `source_language`로
  분석하므로 번역하지 않더라도 코퍼스 언어로 지정하세요. `source_language: en`이면 한국어
  코퍼스가 조사가 붙은 단어("홍길동으로부터")를 토큰 하나로 둡니다. 바꾸면 매핑이
  바뀌므로 텍스트 단위를 다시 인덱싱하세요.
- **유니코드.** 텍스트와 질의는 NFC로 정규화하므로 분해형 한글(NFD)도 조합형과 맞습니다.
- **분석기는 클러스터에 있어야** 하며, 없으면 인덱싱 단계가
  `the cluster has no analyzer '<name>'`로 실패합니다. `nori`는 `analysis-nori`
  플러그인에서 오는데, [로컬 OpenSearch 컨테이너](#컨테이너로-띄운-로컬-저장소)에는 있고
  Amazon OpenSearch Service 도메인은 Nori 패키지를 연결해야 갖습니다(블루/그린 배포). 그래서
  [CDK 배포](../iac/README.md)는 `ko`를 기본 제공 `cjk` 분석기(문자 바이그램)에
  매핑합니다. 도메인을 Nori로 바꾸는 방법은
  [설정](../iac/README.md#configuration-cdkjson-context---c-keyvalue)(영문)의
  `opensearch_korean_analyzer` 행에 있습니다.
- **실패.** 번역에 실패한 텍스트 단위는 원문 언어로 추출합니다. 대상 언어 하나의 호출
  전체가 실패하면 `processing.ignore_errors: true`가 아닌 한 단계가 실패합니다. 증분
  실행은 `failed_units`의 문서를 실패로 기록하므로 다음 실행이 다시 시도합니다.

---

## 4. 질의 (`run-rag`)

`run-rag --query "..."`는 질문 하나에 답하고, `--interactive`는 대화를 이어 갑니다.
전략은 `--search-strategy`로 고르며(기본값 `auto`, LLM 라우터), 모든 답변은 답변 모델이
실제로 본 출처를 함께 돌려줍니다. 전략별 검색 방식은
[설계 §6](./design.ko.md#6-검색-두-방법론)의 그림에 있습니다.

모든 플래그와 기본값은 [CLI 레퍼런스](./cli.md#run-rag)(영문)에 있습니다.

### 검색 전략 선택

GraphRAG 전략은 코퍼스 전체를 *요약하고 주제별로 종합*하는 데 강하고, LightRAG 전략은
더 빠르며 *이중 레벨 키워드* 검색을 중심으로 동작하므로 키워드 중심 조회와 저비용
기준선에 적합합니다. 두 방법론 모두 같은 하이브리드 스코어러(BM25, 벡터, 그래프, RRF,
재순위화)를 씁니다.

**GraphRAG(커뮤니티 요약):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `simple` | 빠른 사실 조회 | 그래프 순회 없이 OpenSearch 벡터·키워드로 검색하며, 주장이 켜져 있으면 주장 인덱스도 포함합니다. 가장 빠릅니다. |
| `local` | 특정 엔티티에 관한 질문 | 질의 엔티티 → 그래프 순회로 이웃과 관계를 찾아 벡터·키워드 결과와 결합합니다. 켜져 있으면 주장도 넣습니다. |
| `global` | "주요 주제는 무엇인가" 같은 넓은 주제 질문 | 계층 한 단면(`search.global_search.community_depth`, 기본값 2. 리포트가 `max_reports_per_level`보다 많은 단면은 들어맞는 가장 깊은 얕은 단면으로 내려감)의 커뮤니티 리포트를 fast 모델로 점수 매긴 핵심 내용으로 map하고(리포트 `map_context_tokens`마다 호출 1회), 순위를 매긴 핵심 내용으로 답합니다. 동적 선택은 대신 루트부터 내려갑니다. |
| `drift` | 큰 그림과 구체적 사실이 함께 필요한 여러 부분으로 된 질문 | 가상 답변(HyDE)과 비슷한 리포트로 중간 답변과 후속 질의를 만들고, 후속 질의마다 local search를 실행해 새 후속 질의를 얻는 과정을 `n_depth` 라운드 반복한 뒤 답변 모델이 종합합니다. LLM 호출이 가장 많습니다(약 `primer_folds + k_followups * n_depth`회). |
| `auto` | 일반 용도(기본값) | LLM 라우터(`models.roles.strategy_selection`)가 `search.auto_routable_strategies`(기본 local, mix, global. 질의마다 LLM 호출이 최대 27번인 drift는 추가했을 때만) 중에서 고릅니다. |

**LightRAG(이중 레벨 키워드):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `mix` | 일반적인 LightRAG 사용 | 저수준 키워드 → 엔티티 인덱스, 고수준 키워드 → 관계 인덱스, 1홉 연결 관계·끝점 엔티티 확장(Neptune 다중 홉은 `search.lightrag_search.enable_graph_expansion`으로 선택)에 **더해** naive 청크 검색 결과를 섞습니다. |
| `hybrid` | 키워드 중심의 그래프 질문 | naive 청크 결과를 섞지 않는 `mix`입니다. |
| `naive` | 빠른 기준선, 비교 평가 | 청크 검색만 합니다(`search.lightrag_search.chunk_stream_top_k`, 기본 20개). 기본값 `--search-type hybrid`는 BM25를 더하며, upstream LightRAG의 순수 벡터 기준선은 `--search-type vector`로 재현합니다. |

`mix`와 `hybrid`는 수집이 만드는 관계 인덱스를 읽습니다. 키워드가 나오지 않는 짧은
질의는 원래 질의를 저수준 키워드로 씁니다(`search.lightrag_search.raw_query_fallback_max_len`
이하).

### 예시

```bash
# Single query (auto strategy, hybrid search)
run-rag --query "What are the main themes in the documents?" --config-path config.yaml

# Pick a strategy and search type
run-rag --query "How does entity X relate to Y?" \
  --search-strategy local --search-type hybrid --config-path config.yaml

# Retrieval only, JSON output
run-rag --query "..." --mode search --output-format json --config-path config.yaml

# Verbose (entities, top sources, metrics) with attribute filters
run-rag --query "..." --verbose --filters attr_category:research type:person \
  --config-path config.yaml
```

### 재순위화 상태

모든 질의는 재순위화 결과를 `SearchResult`와 RAG 출력의 `metadata.rerank`(따라서
`--output-format json`에도)에 남깁니다. `status`, `reason`, `stages`로 구성되며,
`stages`에는 재순위화 단계마다 `status`, `scope`(`all`, 또는 재순위화한 결과 유형. 예:
`local`, `drift`, `mix`, `hybrid`의 `text`), `reason`이 있습니다.

| 상태 | 의미 |
|---|---|
| `applied` | 재순위화 모델이 후보 순서를 바꿨습니다. |
| `disabled` | `search.reranking.enabled: false` 또는 `providers.rerank.kind: none`. |
| `unavailable` | 재순위화 모델을 만들 수 없습니다(초기화 실패는 ERROR로 기록, 또는 크로스 인코더의 `[local]` extra 미설치). |
| `skipped` | 재순위화할 대상이 없습니다: 후보 없음, 재순위화 대상 유형의 결과 없음, 질의 텍스트 없음. |
| `failed` | 호출이 예외를 냈거나(`bedrock:Rerank` 거부, 모델이 없는 리전, 연결할 수 없는 엔드포인트) 쓸 수 있는 결과를 돌려주지 않아 융합 순서를 유지했습니다. |

- `failed`와 `unavailable`의 `reason`은 메시지 없이 AWS 오류 코드나 예외 유형만 밝힙니다.
- 기본 제공 전략은 질의마다 한 번 재순위화합니다. 같은 기록에 여러 번 융합하는 호출자는
  융합마다 단계가 생기고, 최상위 `status`는 `failed` > `unavailable` > `applied` >
  `skipped` > `disabled` 순으로 가장 나쁜 단계의 값입니다. 모델이 순서를 바꾼 결과에는
  `metadata.reranked`가 표시됩니다.
- Bedrock 권한이나 리전 문제는 `unavailable`이 아니라 첫 질의의 `failed`로 나타납니다.
  `search.reranking.required: true`이면 이런 질의를 융합 순서로 답하지 않고
  `RerankModelError`로 실패시킵니다.

### 필터

`--filters key:value ...`(또는 `RAGInput.filters`)는 검색이 순위를 매길 대상을
좁힙니다. 필터는 OpenSearch에서는 `term`/`terms`/`range` 절로, Neptune에서는 `has` 단계로
변환됩니다. 저장소마다 해당 인덱서가 쓰는 필드를 받습니다.

| 저장소 | 필터 가능한 필드 |
|---|---|
| 텍스트 단위 | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.filters.<key>` |
| 엔티티 | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `attr_<key>`, `attributes.<path>` |
| 관계 | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids`, `attr_<key>`, `attributes.filters.<key>` |
| 주장 | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `description`, `source_text`, `text_unit_ids`, `attr_<key>`, `attributes.filters.<key>` |
| 커뮤니티 리포트 | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `level`, `depth`, `max_cut_depth`, `parent_community_id`, `child_community_ids`, `attr_<key>`, `attributes.filters.<key>` |
| Neptune 엔티티 정점 | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `attr_<key>`(있는 경우) |
| Neptune 커뮤니티 정점 | `id`, `name`, `level`, `parent`, `size`, `period`, `children` |

- OpenSearch에서 `attr_<key>`는 문서 `filters` 메타데이터의 `<key>` 항목입니다(여러
  문서에서 나온 항목은 그 값을 모두 가짐). Neptune 엔티티 정점에서는 엔티티 자체의
  속성입니다(`attr_role`).
- 엔티티와 주장의 `type`, 주장의 `status`는 소문자로 저장되고 `term`은 정확히 일치해야
  하므로 `type:PERSON`이 아니라 `type:person`으로 필터링하세요. keyword나 숫자 필드를
  쓰는 편이 좋습니다. `description`처럼 분석되는 필드에 `term`을 걸면 소문자 토큰 하나와만
  일치합니다. 범위 필터는 Python API에서 `{"gte": ..., "lte": ...}`로 넘깁니다.
- 필터는 해당 필드를 선언한 저장소에만 적용됩니다(`type:person`은 텍스트 단위를
  거르지 않음). 선택한 전략이 읽는 저장소 어디에도 선언되지 않은 키(예: `category`)는
  필터 가능한 키 목록과 함께 `InvalidFilterError`를 일으킵니다
  (`unified_kg_rag/adapters/storage/filter_schema.py`).
- OpenSearch는 `attr_<key>`와 `attributes.<path>`를 엄격하게(fail-closed) 적용합니다.
  속성이 없는 항목은 제외됩니다. 관계, 주장, 커뮤니티 리포트는 자기 텍스트 단위가 속한
  문서들의 필터 값을 가지므로 그중 하나가 일치하면 일치합니다. 벡터나 하이브리드 검색은 그
  속성을 가진 문서가 하나도 없는 인덱스를 건너뜁니다(OpenSearch 2.13은 인덱스가 매핑하지
  않은 필드에 대한 kNN 필터를 거부함). 관계와 주장이 문서 필터를 갖기 전에 만든 인덱스는
  다시 빌드할 때까지(`indexing.reset: true`) 속성 필터를 건 질의에서 관계와 주장을 하나도
  돌려주지 않습니다.
- 문서를 제거하면 남은 엔티티, 관계, 주장에서 그 문서의 필터 값을 지웁니다. 남은 문서
  어디에도 없는 키는 null로 저장되고 null은 어떤 필터와도 일치하지 않습니다. 이 문제가
  고쳐지기(#370) 전에 문서 제거를 겪은 인덱스는 그 뒤로 다시 쓰이지 않은 항목에 제거된
  문서의 값을 그대로 가지고 있어 필터가 여전히 그 문서와 일치합니다. 다시 빌드하면
  (`indexing.reset: true`) 지워집니다. 그 수정 뒤 #376 전에 증분 실행이 쓴 인덱스에서
  문서 제거가 커뮤니티를 정리했다면, 남은 텍스트 단위가 문서 필터를 잃어 어떤 필터와도
  일치하지 않으므로 이 인덱스도 다시 빌드하세요.
- `attr_<key>`와 `attributes.filters.<key>`는 문서 속성입니다. 그래프 저장소에는 이것이
  없으므로 그래프 확장은 시드를 제외한 확장 엔티티를, 그 엔티티가 나온 모든 문서의 필터를
  가진 엔티티 인덱스 문서가 이를 통과한 경우에만 남깁니다. 키가 둘 이상이면 그 엔티티의
  텍스트 단위(한 문서의 것) 중 하나도 모든 키를 통과해야 합니다. 또한 필터를 통과한 관계만
  따라 시드에서 `max_hops` 안에 닿아야 하며(제외된 문서만 진술한 관계는 따라가지 않음),
  `Path:`도 그 관계를 따릅니다. 텍스트 단위에 걸리는 필터가
  있으면 local과 mix는 상위 순위 청크만이 아니라 엔티티와 연결된 청크 중 필터를 통과한 것으로
  청크 할당량을 채웁니다. 그 밖의 `attributes.<path>` 키는 항목 자체의 속성이며 엔티티 인덱스에만
  적용됩니다. Neptune에서 `attr_<key>`는 엔티티 정점 속성(`attr_role`)이기도 해서 "있는 경우"
  기준으로 적용되지만(그 속성이 없는 정점은 통과) 위의 엔티티 인덱스 확인도 적용되므로, 문서에
  `role` 필터가 없으면 `attr_role:buyer`는 아무것도 고르지 않습니다. 그 밖의 키는 두 저장소
  모두 엄격하게 적용합니다. 이 확인은 확장한 엔티티에만 적용되고 시드와 그 사이의 정점에는
  적용되지 않으므로, 확장 엔티티의 `Path:` 줄이 필터를 통과하지 못하는 중간 엔티티의 이름을
  담을 수 있습니다.

**필터는 접근 제어 경계가 아닙니다.** 키를 선언하지 않은 저장소는 내용을 거르지 않고
돌려주며, 여러 문서에서 나온 엔티티, 관계, 커뮤니티 리포트는 그중 하나만 일치해도
통과하므로 제외된 문서의 내용을 담을 수 있습니다. 여러 문서에서 나온 항목은 여러 키로 된
필터의 키마다 따로 일치하기도 합니다. 시드, 관계, 주장, 커뮤니티 리포트는 키마다 모든 문서의
값을 가지므로, 문서가 `{category: x, grp: a}`와 `{category: y, grp: b}`인 항목은 두 조건을
모두 갖춘 문서가 없어도 `attr_category:x attr_grp:b`에 일치합니다.
테넌트나 사용자는 서버에서 정한 [인덱스 접미사](#인덱스-접미사)로 나누세요.

### 대화형 모드와 대화 메모리

```bash
run-rag --interactive --conversation-id my-session --config-path config.yaml
```

대화형 모드에서는 메모리가 켜집니다. 명령: `help`, `new`(새 대화 ID),
`set-filter key:value`, `clear-filters`, `show-config`, `quit` / `exit`.

대화 메모리는 프로세스 안에만 있습니다(`MemoryManager`, 한도는 `memory` 섹션).

- 세션 하나의 턴 사이에서, 또는 `use_memory=True`와 같은 `suffix`, `user_id`로
  `conversation_id`를 다시 쓰는 `GraphRAGChain` 하나의 호출 사이에서 이어집니다(다른
  테넌트나 사용자의 턴은 섞이지 않음). 체인끼리는 같은 `MemoryManager(config)`를
  `memory_manager=`로 넘겨야만 공유합니다.
- `run-rag --query`를 따로 실행하면 매번 이력 없이 시작하므로 `--use-memory`와
  `--conversation-id`가 효과가 없습니다(`run-rag`가 WARNING을 남김). CLI에서 여러 턴을
  이어 가려면 `--interactive`를 쓰세요.
- 검색 전에 후속 질문을 대화 이력으로 독립된 질문으로 다시 씁니다(이력이 있는 턴마다
  `models.roles.query_rewrite` 호출 한 번, 프롬프트 키 `query_rewrite`). 그래서 "그 회사의
  공급업체는?"도 앞선 턴의 엔티티 이름이 들어간 질문이 됩니다. 재작성은 최근
  `memory.max_messages_per_conversation`개 메시지를 읽고 앞선 답변은 각각 1,000자까지만
  쓰며, `enable_query_processing`을 꺼도 실행됩니다(이 플래그는 번역과 엔티티·키워드
  추출을 끔). 메모리는 이 외에 LLM을 추가로 호출하지 않습니다.

---

## 5. 증분 인덱싱

증분(delta) 인덱싱은 전체를 다시 구축하지 않고, 마지막 실행 이후 **새로 생겼거나
바뀐** 문서만 다시 인덱싱해 운영 중인 그래프에 병합합니다.

### 켜기

```yaml
providers:
  doc_status:
    enabled: true          # kind: dynamodb(기본값) 또는 sqlite
aws:
  dynamodb:
    table_name: "unified-kg-rag-on-aws-doc-status-v2"
    create_table_if_missing: true
```

이 설정을 켜면 `run-ingestion`은 실행할 때마다 DynamoDB 문서 상태 레지스트리와
**콘텐츠 해시**로 코퍼스를 비교합니다. 로컬 모드에서는 레지스트리가 SQLite
파일입니다(`providers.doc_status.kind: sqlite`. [로컬 모드](#로컬-모드-aws-없이) 참고).

### 작업 흐름

- **문서 추가:** 새 파일을 소스 디렉터리에 넣고 `run-ingestion`을 다시 실행합니다.
  새 파일만 파싱·추출·인덱싱하고, 그 엔티티와 관계는 기존 그래프에 병합합니다.
- **문서 수정:** 파일을 고치고 다시 실행합니다. 이전 버전의 산출물을 제거하고 새
  버전을 인덱싱합니다. 모델에는 바뀐 청크만 보냅니다. 추출 저장소
  (`indexing.opensearch.extraction_records_index_prefix`)에 본문과 추출 설정이 같은
  레코드가 있는 청크는 어느 문서의 것이든 저장된 레코드를 씁니다(`--force-rebuild`는
  모두 다시 추출).
- **문서 삭제:** 파일을 지우고 다시 실행합니다. 그 문서의 청크에서만 서술된 내용
  (엔티티, 관계, 주장)은 삭제하고, 공유하는 산출물은 그 문서의 청크만 빼고 유지합니다.
  커뮤니티와 리포트는 인용하는 청크가 하나도 남지 않으면 지웁니다.
- **추출 설정 변경**(튜닝한 `custom_prompts`, `entity_types`, 추출·gleaning·청킹 모델,
  청크 크기, 번역, 해석): 다음 실행은 다른 설정으로 추출된 변경 없는 문서를 다시
  추출하고, 그 수와 바뀐 설정을 WARNING으로 남깁니다. 설정은 필드별로 비교하므로 새
  릴리스가 추가한 설정은 기본값인 동안 변경으로 보지 않습니다. 레코드를 정하는
  설정(모델, 추출·gleaning·주장 설정, 대상 언어, 프롬프트 템플릿)이 그대로면 다시
  추출하는 청크도 저장된 레코드를 씁니다(예: 해석이나 청킹만 바꾼 경우). 내장 프롬프트
  템플릿(코드)을 고친 것은 감지하지 못하므로 고친 뒤에는 초기화하세요.
- **임베딩 모델이나 차원 변경**(`models.embedding.model_id` / `dimension`):
  OpenSearch 인덱스마다 만들 때 쓴 모델과 차원을 기록하며, 인덱싱 단계는 그 값이 이번
  실행과 다르면 아무것도 쓰기 전에 멈춥니다(기록이 없는 인덱스도 거부).
  `indexing.reset: true`로 한 번 다시 실행하세요. 다른 모델이나 차원으로 `run-rag`·`run-eval`을
  실행하면 빈 검색 결과로 답하지 않고 `ConfigurationError`로 실패합니다.
- **코퍼스 언어나 번역 언어 변경**(`processing.translation`의 언어 설정,
  `indexing.opensearch.language_analyzers`): 운영 중인 텍스트 필드는 만들 때의
  분석기를 그대로 쓰므로 인덱싱 단계가 아무것도 쓰기 전에 멈춥니다.
  `indexing.reset: true`로 한 번 다시 실행하세요. 언어를 뺄 때는 멈추지 않습니다.
- **레지스트리를 끄고 접미사 인덱싱**(`backend: local` 프리셋처럼): 그 접미사의 다음
  증분 실행은 아무것도 쓰기 전에 멈춥니다. 레지스트리를 켜고 `indexing.reset: true`로
  한 번 다시 실행하거나, 그 접미사는 레지스트리를 끈 채로 쓰세요.
- **재구축**(`indexing.reset: true`): 접미사를 서비스에서 내리지 않고 코퍼스 전체를
  다시 인덱싱합니다. 격리된 문서는 건너뛰고 레코드를 그대로 둡니다. 실패한 reset은 아무것도
  지우지 않으므로 다시 실행해도 안전합니다
  (동작 방식: [운영 런북: 다시 수집하기](./operations.ko.md#다시-수집하기)).

### 삭제 범위

실행의 **범위**는 인덱스 접미사(`document_parsing.index_value`와
`indexing.additional_suffix`)와 코퍼스 소스(`document_parsing.source_scope`. 기본값은
해석된 소스 디렉터리이며, 컨테이너 엔트리포인트는 동기화 원본인 S3 URI로 설정)입니다.
문서의 레지스트리 키는 이 범위와 소스 디렉터리 기준 상대 경로이며, 실행은 자기 범위의
문서만 비교하고 삭제 대상으로 봅니다. 따라서 다른 테넌트(다른 `index_value`)의 실행은 이
테넌트의 문서를 삭제하지 않고, 하위 폴더를 소스 디렉터리로 따로 실행해도 코퍼스의
나머지는 삭제되지 않으며(그 파일들은 별도 문서로 등록되므로 같은 파일을 두 루트에서 한
접미사로 인덱싱하지 마세요), 소스가 다른 두 코퍼스는 한 접미사에서 상대 경로가 같아도
서로 다른 문서입니다. 파싱이나 로딩에 실패한 파일은 `failed`로 보고되며, 다음 실행에서
다시 읽을 때까지 인덱싱된 내용을 유지합니다.

범위가 기본값인 **코퍼스를 옮기면** 범위가 바뀌는데, 어떤 실행도 다른 범위의 문서를
삭제하지 않으므로(이 호스트에 없는 디렉터리가 다른 호스트의 코퍼스일 수 있음) 이전
내용이 인덱스에 남습니다. 다음 중 하나를 따르세요.

- 옮기기 전에 `document_parsing.source_scope`를 고정된 이름으로 지정합니다.
- 새 위치에서 `--retire-source-scope <이전 소스 디렉터리>`(또는
  `indexing.retire_source_scopes`)로 한 번 실행합니다. 이 실행은 접미사에서 이전 범위의
  레코드를 모두 삭제된 문서로 처리하고, 새 위치의 문서를 새 문서로 인덱싱합니다. 범위는
  레지스트리에 저장된 형태로 지정합니다. 절대 디렉터리(끝과 중복된 `/`는 무시)나 이전
  `source_scope` 값이며 `s3://` URI도 됩니다. URI가 아닌 값은
  `--source-directory`처럼 해석한 경로로도 찾습니다. 경로 표시가 없는 값(`old`)은 이름이거나
  상대 경로이므로 레코드가 있는 쪽만 지우고, 둘 다 있으면 쓰기 전에 실행을 멈춥니다(`./old`나
  절대 경로로 지정). 실행 자신의 범위를 지정하면 레지스트리를
  읽기 전에 실패하며, 해석 중 하나가 실행 자신의 범위인 경로 표시 없는 값도 마찬가지입니다.
  자신의 범위가 이름이면 디렉터리를 `./<값>`으로 지정합니다. 자신의 디렉터리이면 이름이 다른
  코퍼스의 `source_scope`일 수 있으므로, 그 이름은 그 값이 실행 자신의 디렉터리가 아닌 작업
  디렉터리에서 같은 설정(같은 저장소와 레지스트리. 예: 같은 `--config-path`)으로 지웁니다.
  레코드가 없는 범위는 WARNING으로 알립니다.

범위가 기본값인 로컬 실행에 새 문서가 있으면, 같은 접미사에 레코드가 있는 다른 로컬
소스 디렉터리를 WARNING으로 알립니다. 아무것도 삭제하지 않으며, 그 디렉터리들이 별도
코퍼스라면 코퍼스마다 `source_scope`를 지정하면 경고가 나오지 않습니다.

레지스트리는 인덱스 네임스페이스(문서 키의 `<namespace>#` 접두사)별로 나뉘므로 실행은
자기 네임스페이스의 레코드만 읽습니다. `doc_id`만으로 키를 잡던 이전 키 스키마의 테이블은
처음 쓸 때 거부됩니다. 새 테이블로 바꾸고 `indexing.reset: true`로 한 번 실행하세요
([운영 런북: 증분 실행](./operations.ko.md#증분-실행)).

### 실패한 문서

텍스트 단위 하나라도 번역, 추출, gleaning, 주장 추출에 실패했거나, 그 문서를 포함한 기본
수준 커뮤니티의 리포트가 생성되지 않았거나, `--continue-on-error` 실행에서 산출물 단계가
비정상 종료했거나, 산출물 하나를 쓰는 데 실패했거나, 텍스트 단위의 추출 레코드를 저장하지
못한 문서는 `FAILED`로 기록됩니다. 다음 실행은 파일이 바뀌지 않았더라도 이 문서를 다시
처리하며, 같은 내용으로 `indexing.max_document_failures`(기본값 `3`)번 연속 실패하면 더
재시도하지 않습니다. 그 뒤에는 파일을 수정하거나 한도를 올리면 다시 시도합니다.
[운영 런북: 실패한 문서](./operations.ko.md#실패한-문서)를 참고하세요.

### 격리된 문서

처리하다가 태스크를 죽이는 문서(메모리 부족, 프로세스를 죽이는 파서)는 이후 실행도 같은
식으로 멈추게 합니다. 그 문서만 처리하다 죽은 실행이 `indexing.max_document_attempts`(기본값
`3`)번 이어지면, 다음 실행은 시도들이 쓴 내용을 지우고 문서를 `QUARANTINED`로 기록하며
파일 이름이 담긴 WARNING을 남기고 나머지를 인덱싱합니다. 파일이 바뀌거나 한도를 올릴
때까지 이 문서는 건너뜁니다.

- 죽은 실행만 셉니다. 완료하거나 실패하거나 SIGTERM으로 중지된 실행은 처리 중 표시를
  지우고, 전체 재구축은 시도를 세지 않습니다.
- 여러 문서를 처리하다 죽었다면 다음 실행들이 원인 문서만 남을 때까지 의심 문서를 절반씩
  줄이고, 그동안 다른 새 문서와 변경된 문서는 미룹니다(`total_documents_deferred`).

[운영 런북: 격리된 문서](./operations.ko.md#격리된-문서)를 참고하세요.

### 동시 실행

레지스트리를 쓰면 실행은 첫 단계부터 끝날 때까지 인덱스 네임스페이스의 쓰기 임대를
유지하며, 같은 네임스페이스의 두 번째 실행은 저장소를 읽거나 무엇이든 쓰기 전에
`NamespaceLeaseError`로 멈춥니다. [운영 런북: 동시 실행](./operations.ko.md#동시-실행)을
참고하세요.

### 문서 크기 한도

DynamoDB 레지스트리는 문서의 텍스트 단위(청크) ID를 항목 하나에 저장하며, 항목 한도는
400KB(ID 약 10,000개)입니다. 수정한 문서의 진행 중 레코드는 이전 ID와 새 ID를 함께
담으므로 수정하는 문서는 그 절반쯤입니다. 한도를 넘는 문서는 아무것도 쓰기 전에 파일
이름이 담긴 오류로 인덱싱 단계를 실패시키므로 더 작은 파일로 나누세요. 로컬 모드의
SQLite 레지스트리에는 이 한도가 없습니다.

### 실행 간 병합

증분 실행은 건드리는 엔티티, 관계, 주장을 저장소에서 다시 읽어 변경분을 그 안에
병합하므로, 변경되지 않은 문서와 공유하는 엔티티는 그 문서들의 설명과 청크
계보(`mix`가 따라감)를 유지합니다. 요약 예산을 넘은 병합 설명은 다시 요약합니다. 문서가
바뀌거나 사라지면 그 문서가 공유하던 항목은 그 문서의 청크를 잃고 남은 청크에서 다시
도출되며, 엔티티는 전체 빌드와 같이 현재 우세한 이름의 ID를 받습니다(저장된 커뮤니티와
리포트는 재구축 전까지 이전 ID를 유지). 병합은 항상 켜져 있으며, 제거된
`indexing.cross_run_merge` 키는 WARNING을 남기고 무시합니다. 규칙은
[설계 §5.4](./design.ko.md#54-식별-도출한-내용-병합-규칙)에 있습니다.

---

## 6. 평가 (`run-eval`)

`run-eval --data-path <dataset.json>`은 데이터셋의 모든 질문에 답한 뒤 답변과 출처를
채점합니다. 평가기 셋은 LLM 없이 비용 없이 동작하고, 둘은 LLM 심사 모델을 씁니다.
전략을 여럿 주거나 `--repeats`를 주면 짝지은 비교가 됩니다.

모든 플래그와 기본값은 [CLI 레퍼런스](./cli.md#run-eval)(영문)에 있습니다.

### 평가기

`evaluation.enabled_evaluators`로 고르며, 다섯 개 모두 기본으로 켜져 있습니다. 결정적
평가기는 데이터셋 필드가 없는 질의를 건너뜁니다. 심사 모델 없이 실행하려면
`enabled_evaluators: [answer_match, retrieval, graph_aware]`로 설정하세요.

| 평가기 | LLM | 데이터셋에 필요한 필드 | 주요 지표 |
|---|---|---|---|
| `answer_match` | 없음 | `answer`(+ `answer_aliases`) | `answer_contains`(대표 지표), `exact_match`, `token_f1` |
| `retrieval` | 없음 | `reference_sources` | `hit_at_k`, `recall_at_k`, `mrr`(k = `evaluation.retrieval_k`, 기본값 5) |
| `graph_aware` | 없음 | `expected_entities` / `expected_relationships` | 답변과 검색된 출처에서의 엔티티·관계 커버리지(재현율) |
| `langchain` | 심사 | `answer` | `correctness`, `partial_correctness` |
| `ragas` | 심사 | `answer` | `answer_correctness`, `answer_relevancy`, `context_precision`, `context_recall`, `faithfulness` |

켜 둔 평가기를 만들 수 없거나(예: 심사 모델의 Bedrock 접근 권한 없음) 설정을 거부하면
실행이 멈춥니다. `processing.ignore_errors: true`이면 그 평가기를 빼고
`run_manifest.dropped_evaluators`에 기록합니다. 채점은 모든 답변을 생성한 뒤에
시작하므로 평가기 오류가 나도 실행 결과는 버리지 않습니다. 해당 질의의 지표를 실패로
기록하고(쓸 만한 점수가 없는 심사 응답도 0이 아니라 실패), 종료 코드는
`--max-failure-rate`가 정합니다.

#### `ragas` 컨텍스트

- `context_precision`은 컨텍스트마다 심사 호출을 한 번 하므로 순위 상위 출처를 최대
  `evaluation.ragas_max_contexts`개(기본값 20, `null`이면 제한 없음)까지만
  채점합니다(`context_precision@N`). 이 제한이 없으면 출처를 100개 넘게 보고하는 전략은
  `ragas_timeout`에 걸립니다.
- `faithfulness`와 `context_recall`은 `max_context_tokens` 안의 모든 출처를 봅니다.
  기본값(`evaluation.max_context_tokens: null`)에서는 답변 모델의
  예산(`search.token_manager.max_context_tokens`)이므로 답변을 그 답변이 받은
  컨텍스트로 채점합니다. 더 작은 값을 지정하면 경고를 남깁니다.
- 보고서마다 심사 모델이 본 내용을 기록하며(`judge_contexts`, `judge_context_tokens`
  등), 이 설정들은 답변 모델이 본 내용을 바꾸지 않습니다.
- ragas 사용 통계 전송은 꺼져 있습니다(기본값 `RAGAS_DO_NOT_TRACK=true`). 다시 켜려면
  환경 변수 `RAGAS_DO_NOT_TRACK=false`를 지정합니다.

#### `graph_aware`

기대한 그래프 산출물 가운데 생성된 답변에 나타난 비율(커버리지 = 재현율)을
`answer_contains`의 구문 매처로 계산합니다. `{"source": "A", "target": "B"}`나
`"A -> B"`로 준 관계는 답변이 두 끝점을 모두 언급하면 맞은 것으로 보고, 그 밖의 문자열은
구문으로 나타나야 합니다. 자유 형식 답변의 엔티티를 빠짐없이 나열하기 어려우므로 정밀도와
F1은 내지 않습니다.

답변 텍스트의 커버리지는 답변이 길수록 올라가므로, 전략은
`retrieved_entity_coverage` / `retrieved_relationship_coverage`(같은 기대값을 답변
모델이 받은 출처의 엔티티와 관계 끝점에 맞춰 본 값)로 비교하고, 답변 커버리지는
`answer_length`와 함께 읽으세요. 검색 커버리지는 컨텍스트에 그 유형의 출처가 없으면
0이 아니라 건너뛰므로(`global`은 설계상 그러함), 비교에서는 그 지표의 `pairs` /
`unpaired`를, 단일 실행에서는 `metric_outcomes`의 건너뛴 수를 확인하세요.

#### `retrieval`

답변 모델이 본 출처에 정답 문서가 들어 있었는지 봅니다.

- **귀속.** 텍스트 단위 출처는 파일 이름을 직접 가집니다. 엔티티, 관계, 커뮤니티 리포트
  출처는 인용하는 텍스트 단위의 파일에 귀속하며, 그 텍스트 단위를 가장 많이 가진 파일이
  먼저입니다.
- **k는 출처가 아니라 파일 수입니다.** 출처를 순위대로 서로 다른 파일 목록 하나로
  펼치므로, 파일 여러 개에 귀속되는 커뮤니티 리포트가 hit@k를 부풀리지 못합니다.
- **출처 유형별.** `by_source_type`(`grouped_statistics.source_type`에 집계)은 같은
  지표를 섹션 유형별로 계산합니다. 답변 모델이 그대로 읽은 텍스트 단위만 채점하는 `text`
  행이 전략 간 비교에 가장 공정합니다. `attributable_fraction`(귀속할 수 있는 출처 /
  보고된 출처)은 지표가 컨텍스트의 어느 정도를 다루는지 보여 줍니다.
- **매칭.** 파일 이름에 NFKC 정규화와 대소문자 무시를 적용하고 참조의 디렉터리는
  무시합니다. `docs/Terms.pdf`, `terms.pdf`, `terms`는 모두 출처 `Terms.pdf`와 맞고
  `terms.docx`는 맞지 않으며, 버전 접미사는 확장자가 아닙니다(`notes.v1`은
  `notes.v2`와 맞지 않음). `AC/DC` 같은 제목은 통째로 비교합니다.
- **건너뛰기.** 질의에 `reference_sources`가 없거나 귀속할 수 있는 출처가 없으면
  건너뜁니다. 출처를 하나도 가져오지 못한 질의는 0점입니다.

#### `answer_match`

`answer`와 선택 항목 `answer_aliases`(문자열이나 목록, 최상위나 `metadata` 아래에 두며
`metadata`가 우선)에 맞춰 채점하고 그중 최댓값을 씁니다.

- **`answer_contains`**가 대표 지표입니다. 정답이나 별칭이 생성된 답변에 단어 단위
  구문으로 나타나면 1.0입니다. 긴 RAG 답변이 짧은 정답 구간과 똑같은 경우는 드물어서
  exact match보다 정답 여부를 훨씬 잘 따라갑니다.
- **한국어와 CJK.** 단어 뒤의 한국어 조사는 허용합니다(`AWS`는 `AWS는`과 맞음). 숫자만
  있는 정답에는 한국어 단위가 붙어도 되지만(`2024`는 `2024년에`와 맞음) 수의 크기 단위는
  안 됩니다(`3`은 `3억`과 맞지 않음). CJK 문자 옆의 띄어쓰기는 무시하고(`3억 원` =
  `3억원`), 한 단어로 된 CJK 정답은 부분 문자열로 맞추되 더 긴 숫자나 라틴 단어 안에서는
  맞추지 않습니다(`2년`은 `12년`과 맞지 않음).
- **예/아니요 정답**(`yes`/`no`/`nope`, 한국어 `예`/`네`/`아니요`/`아니오`/`아닙니다`,
  중국어 `是`/`对`/`否`/`不是`/`没有`, 일본어 `はい`/`いいえ`)은 HotpotQA처럼 질문에
  답하는 첫 예/아니요 단어의 극성으로 채점합니다(`there is no clause`, `네 가지`,
  `(예: …)`, `No. 5`는 답이 아니므로 `네, 맞습니다`는 예로 답한 것이지만
  `네 맞습니다`는 아님).
- **`exact_match`와 `token_f1`**은 SQuAD v1.1 정규화(NFKC, 소문자화, 문장 부호와 영어
  관사 제거, 공백 압축)를 따르되 의미를 바꾸는 문장 부호(`-5`, `1.5`, `C++`)는 남기므로,
  점수는 리더보드 수치에 가깝지만 그대로 비교할 수는 없습니다. 공식 스크립트를 쓰려면
  `evaluation.answer_normalization: official_squad`로 설정하세요. token F1은 공백으로
  나누므로 중국어와 일본어에서는 exact match와 같아지고, 한국어 조사가 붙으면 두 지표
  모두 낮게 나옵니다.

### 평가 데이터 형식

객체의 JSON 배열입니다. `question`만 필수이며, `expected_entities` /
`expected_relationships`는 `graph_aware`에만 필요합니다.

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
    "expected_relationships": [
      { "source": "AI", "target": "machine learning" },
      "machine learning -> data processing"
    ],
    "metadata": { "search_strategy": "global", "answer_aliases": ["AI and ML"] }
  },
  {
    "id": "q2",
    "question": "How do entities X and Y relate to each other?"
  }
]
```

- 항목별 `metadata`(예: `search_strategy`)는 그 질문에 한해 CLI 기본값보다 우선합니다.
  `id`는 `query_id`로 써도 됩니다.
- 코퍼스로 답할 수 없는 질문에는 `"metadata": {"answerable": false}`를 지정하세요. 어떤
  평가기로도 채점하지 않고 `abstention_statistics`(아래)에만 셉니다.
- 파일은 질의를 실행하기 전에 검증합니다. 읽을 수 없는 파일, 비어 있거나 배열이 아닌
  데이터셋, `question` 누락, 중복 ID, 타입이 틀린 필드(목록이 아닌 `expected_entities`,
  문자열이나 문자열 목록이 아닌 `answer_aliases`), RAG 체인이 거부하는 `metadata`(알 수
  없는 `search_strategy`)가 있으면 `Error: invalid evaluation dataset:`과 항목 번호,
  ID를 출력하고 종료 코드 2로 멈춥니다.

### 예시

```bash
run-eval --data-path my_eval_data.json --config-path config.yaml

run-eval --data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml

run-eval --data-path s3://my-bucket/eval/my_eval_data.json \
  --output-directory s3://my-bucket/eval/results/ --config-path config.yaml
```

두 경로 모두 `s3://` URI를 받으므로 `run-eval`도 CDK 스택에서 일회성 태스크로
실행할 수 있습니다([After deploy](../iac/README.md#5-evaluate-from-inside-the-vpc)).
결과는 실행이 끝날 때 `aws.s3.encryption` 설정으로 올리며, 업로드가 실패하면 0이 아닌
코드로 끝나고 로컬 파일을 남깁니다(경로를 출력). S3에는 설정의 `aws.profile_name` /
`aws.region_name`으로 접속하며, 셸의 `AWS_PROFILE` / `AWS_REGION`은 설정이 비워 둔 경우에만
그 값을 채웁니다. 실행 주체에는 데이터셋에 대한
`s3:GetObject`와 접두사에 대한 `s3:PutObject` 권한이 필요합니다.

결과는 `evaluation_{results,reports,summary}_<timestamp>.json`으로 씁니다(모든 질의가
한 전략을 요청했거나 썼다면 `..._<strategy>_<timestamp>.json`, `auto` 실행은
`..._auto_...`). `<timestamp>`는 `YYYYMMDD_HHMMSS_<16진수 8자리>`이므로(같은 초에 시작한
실행끼리 겹치지 않음) 최신 실행은 이름을 만들어 내지 말고 이름을 정렬해 찾으세요. 요약에는
지표별 통계(`metric_statistics`)와 채점/실패/건너뜀 개수(`metric_outcomes`)가 있고, 다음
항목도 들어 있습니다.

- `grouped_statistics`: 같은 통계를 `search_strategy`(실제로 쓴 전략. `auto`에서는
  질의마다 다름), `category`, `difficulty`, `source_type`(출처 섹션 유형별 검색 지표)으로
  나눈 값입니다.
- `abstention_statistics`: 체인이 답변 대신 고정된 컨텍스트 없음 응답("I could not find
  relevant information…")을 반환한 빈도입니다. `abstained`, `answered`,
  `abstention_rate`, `per_strategy`가 있고 `answerable: false` 항목에는 `unanswerable`
  블록이 있습니다. 체인은 검색 결과가 없을 때만 이 응답을 반환하고 모델이 자기 말로 거절한
  것은 감지하지 않으므로, 데이터가 색인된 상태에서 `unanswerable` 비율은 올바른 거절의
  하한입니다. 각 결과에는 `abstained`가 기록됩니다.
- `rerank_status_counts`: [재순위화 상태](#재순위화-상태)별 질의 수입니다. 각 결과에는
  `rerank_status`가 기록됩니다(답변 생성이 먼저 실패했으면 `null`).
- `guardrail_blocked_counts`: `input` / `output`별 차단된 질의 수입니다(차단된 항목은
  차단 메시지로 채점. 차단된 답변은 출처를 버리므로 검색 지표와 RAGAS 컨텍스트 지표
  (`context_precision`, `context_recall`, `faithfulness`)는 `guardrail_blocked`로 건너뜀.
  입력에서 차단된 질의는 아무것도 검색하지 않았으므로 검색 실패로 보아 컨텍스트 정밀도·재현율이
  0이고, `faithfulness`는 컨텍스트 없는 다른 답변처럼 건너뜀: `no_contexts`). 각 결과에는
  `guardrail_blocked`가 기록됩니다.
- `run_manifest`: 무엇을 측정했는지 기록하므로 두 실행을 비교할 수 있습니다.
  - CLI 인자(URL은 자격 정보를 뺌), 버전, `git_sha` / `git_dirty`, 데이터셋(경로, 해시,
    질의 수), UTC 타임스탬프, `dropped_evaluators`
  - 모델 ID와 `judge_is_answer_model`(심사 모델이 답변 모델이면, 즉 기본값이면 심사
    점수가 부풀려지므로 시스템을 비교할 때는 `models.roles.evaluation`을 다른 모델로 두세요)
  - `retrieval_config_sha256`: 답변이나 점수를 바꿀 수 있는 모든 설정의 해시(모델 ID와
    수집 설정 포함, 리전·엔드포인트·자격 증명·경로·시간 제한 같은 운영 설정은 제외)이므로
    같은 평가는 어느 환경에서든 같은 해시가 됩니다.
  - `index`: 질의한 접미사마다 벡터 인덱스를 마지막으로 쓴 수집 실행(`pipeline_id`,
    `mode`, `built_at`, `documents`, 추출 설정, 증분 실행이면 그 실행이 올라선
    `full_build`)입니다. 최선 노력 방식이며 인덱스 메타데이터가 없으면 `null`입니다.

  `EvaluationManager.evaluate_dataset`이 만들므로 라이브러리 호출자도 받습니다
  (`dataset_path=` / `cli_args=`를 넘김).

질의별 리포트에는 평가기마다 `metrics`가 있고(서로 무관한 지표의 평균은 의미가 없으므로
종합 점수는 없음), 보고된 출처마다 순위 순으로 `retrieved_source_ids`(귀속된 파일 이름,
귀속할 수 없으면 `[]`), `retrieved_source_types`, `retrieved_entities`,
`retrieved_relationships`가 기록됩니다.

### 전략 비교

`run-eval --search-strategy local,mix,global --repeats 3`은 같은 질의를 전략마다
`--repeats`번 답하고 채점하며(지정한 전략이 항목별 `search_strategy`보다 우선), 실행마다
결과 파일을 저장하고 `evaluation_comparison_<timestamp>.json`을 추가로 씁니다.

- `strategies.<s>.metrics.<evaluator>/<metric>`: 질의별 반복 평균을 질의 전체로 평균한
  값과 `repeat_stdev`(실행 간 잡음)입니다. `routed_strategies`는 `auto`가 어떻게
  분기했는지 보여 줍니다.
- `deltas_vs_baseline.<s>.<metric>`: 두 전략이 모두 채점한 질의에서 첫 전략과 짝지어
  비교한 값입니다. `mean_delta`, 95% 짝지은 부트스트랩 구간(`ci_low`/`ci_high`. 2,000번
  재표집, 시드 0), `wins`/`ties`/`losses`, 질의별 차이, `unpaired`가 있습니다. 답변
  생성에 실패한 질의는 다른 실행이 채점한 모든 지표에서 0점이므로(`answer_length` 제외)
  실패로 이득을 볼 수 없습니다(`failed_answers` / `failed_queries`. 차단 수는
  `guardrail_blocked_counts`. 전략마다 실패하거나 차단된 수가 다르면 `warnings`에 이름이
  나옴). 구간이 0을 포함하지 않으면 그 차이가 뒷받침됩니다.
  이진 지표의 10%p 차이는 질의 50개에서 15%, 200개에서 59%의 실행에서만 찾아지므로 수백
  개 질의로 비교하세요.
- `pairwise_vs_baseline.<s>`(`--pairwise-judge` 사용 시): GraphRAG / LightRAG 논문의
  맞비교 LLM 판정으로, `models.roles.evaluation`이 첫 반복의 답변을 포괄성
  (comprehensiveness), 다양성(diversity), 이해 도움(empowerment), 직접성(directness)으로
  판정합니다(프롬프트: `custom_prompts.pairwise_comparison_*`). 쌍마다 순서를 바꿔 두 번
  판정하고 두 순서가 일치한 기준만 승리로 세며, `candidate_win_rate`는 무승부를 0.5로
  셉니다. 한쪽 답변이 실패한 쌍은 판정하지 않고 다른 쪽이 이깁니다(`candidate_errors`,
  `baseline_errors`. 둘 다 실패: `both_errors`). `failed`는 판정 모델이 결정하지 못한
  쌍(호출 실패, 또는 `1`, `2`, `tie`가 아닌 판정)의 수이며 `--max-failure-rate`가 적용됩니다.
- 첫 실행의 `run_manifest`.

콘솔에도 같은 요약이 출력되며, 실패 허용 기준은 모든 실행에 적용됩니다.

### 벤치마크 (MuSiQue)

`prepare-benchmark musique`는 [MuSiQue](https://github.com/StonyBrookNLP/musique)(답변
가능, 다중 홉, CC BY 4.0. Trivedi 외, TACL 2022)를 코퍼스와 `run-eval` 데이터셋으로
바꿉니다. 데이터는 이 저장소에 포함하지 않습니다. 기본적으로 공개 미러의 고정 리비전에서
답변 가능 dev 분할(질문 2,417개)을 내려받아 SHA-256을 확인하며, `--data-path`로 로컬
`musique_ans_v1.0_*.jsonl`을 대신 읽을 수 있습니다. `corpus/`(서로 다른 단락마다 텍스트
파일 하나. 질문마다 근거 단락 2-4개와 방해 단락 최대 18개)와 `eval_data.json`(`answer`,
`answer_aliases`, 근거 단락인 `reference_sources`, 홉 수인 `category`)을 씁니다.
`corpus/`가 비어 있지 않으면 아무것도 내려받기 전에 사용 오류(종료 코드 2)로 끝납니다.

```bash
# 1. 시드 0으로 질문 100개(단락 약 1,900개) 표집
prepare-benchmark musique --output-directory bench/musique --limit 100

# 2. 단락을 별도 접미사로 수집
run-ingestion --source-directory bench/musique/corpus --config-path musique.yaml

# 3. 두 방법론을 짝지어 비교하고 논문과 같은 방식으로 채점
run-eval --data-path bench/musique/eval_data.json --suffix musique \
  --search-strategy local,mix,global,naive --repeats 2 --config-path musique.yaml
```

`musique.yaml`에는 평소 설정에 다음을 더합니다.

```yaml
processing:
  document_parsing:
    index_value: musique          # 벤치마크를 기존 인덱스와 분리
evaluation:
  answer_normalization: official_squad   # MuSiQue가 보고하는 방식의 EM / F1
  enabled_evaluators: [answer_match, retrieval]   # 판정 모델 없이. 필요하면 langchain / ragas 추가
```

`answer_contains` / `token_f1`과 `retrieval`(`category`별)을 읽으세요. 수집 비용은 단락
수(질문당 약 19개)에 비례합니다. 결과를 발표할 때는 논문을 인용하세요.

---

## 7. 시각화 (`run-visualization`)

이미 내보낸 시각화 데이터 JSON으로 그림을 그리는 **독립** 렌더러이며, 수집을 다시
실행하거나 AWS에 접근하지 **않습니다**. 모든 플래그와 기본값은
[CLI 레퍼런스](./cli.md#run-visualization)(영문)에 있습니다.

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

렌더러는 **`interactive`**(pyvis)와 **`static`**(Bokeh) 두 개이며 `graph.visualization`
아래(`interactive.*`, `static.*`, `embedding_method`, `layout_method`)에서 설정합니다.

- **입력.** `run-ingestion`의 커뮤니티 탐지 단계는 그래프가 코퍼스 전체를 담는
  실행(전체 빌드, 초기화, 모든 문서가 새 문서이거나 변경된 증분 실행)마다
  `visualization_data.json`을 씁니다. 변경되지 않은 문서를 건너뛴 증분 실행은 아무것도
  쓰지 않고(INFO 로그만 남김) 이전 파일을 그대로 둡니다. 전체 그래프를 보려면 전체
  빌드(`indexing.reset: true`)를 한 번 실행하세요.
- **위치.** `graph.visualization.outputs_directory`이며 기본값은
  `<캐시 디렉터리>/<pipeline_id>/visualization/`입니다. S3 캐시 동기화는 `.json`만
  올리므로 HTML은 로컬에서 다시 렌더링하세요.
- **내용.** 노드(레벨 0 `community_id`로 색을 정함), 간선, 커뮤니티 계층이며 벡터는
  뺍니다. 기본값에서는 파일을 쓸 때 모델을 호출하지 않고 `run-visualization`이 렌더링할
  때 시드를 고정한 spring 레이아웃을 계산합니다. `graph.visualization.enabled: true`이면
  이 단계가 렌더링도 하고 임베딩 기반 레이아웃을 저장합니다(수집마다 노드당 임베딩 호출
  한 번).
- **크기.** 대화형 그래프는 차수 기준 상위 `interactive.max_nodes`개 노드(기본 2000)만
  남기고 그 노드만 배치합니다.
- **레이아웃.** `embedding_method: "none"`은 임베딩 클라이언트 없이 spring 레이아웃을
  쓰고, `"description"`은 노드마다 `name: description`을 임베딩해(`models.embedding`)
  `layout_method`로 차원을 줄입니다. 임베딩이 실패하면 시각화 단계가 실패하며(수집은
  계속함), `processing.ignore_errors: true`이면 ERROR를 남기고 spring 레이아웃으로
  대체합니다. 차원 축소가 실패하면 항상 대체합니다. 대체하면 `"layout_degraded": true`를
  기록합니다.
- 간선 두께와 불투명도는 그래프 자체의 가중치 범위에 맞춰 로그 스케일로 조정합니다.
- 렌더러가 실제로 쓴 파일만 보고하며(같은 이름의 기존 파일은 먼저 지움), 아무것도
  렌더링하지 못하면(예: 빈 그래프) 상태 코드 `1`로 종료합니다.

---

## 8. 프롬프트 튜닝 (`run-prompt-tuning`)

디렉터리에서 문서를 표본으로 뽑아 `models.roles.graph_extraction`으로 코퍼스
특성(도메인, 언어, 페르소나, 엔티티 유형)을 분석하고, 검토해서 `config.yaml`에 합칠
도메인 맞춤 `custom_prompts` YAML 조각을 씁니다. 모든 플래그와 기본값은
[CLI 레퍼런스](./cli.md#run-prompt-tuning)(영문)에 있습니다.

```bash
run-prompt-tuning --source-directory ./source --output-path tuned_prompts.yaml --config-path config.yaml
```

출력에는 `custom_prompts` 블록과, 감지한 도메인과 엔티티 유형이 담긴 `profile`이
있습니다. **검토한 뒤** 원하는 프롬프트를 `custom_prompts:` 아래에 복사하세요.

- 튜닝한 `graph_extraction_system`과 `community_report_system`은 도메인 머리말 뒤에
  내장 규칙과 출력 형식을 그대로 붙인 것이며, 생성한 퓨샷 예시는 마지막에 옵니다.
- 엔티티 범주는 여전히 `processing.graph_extraction.entity_types`(`{entity_types}`)에서
  오므로 프로필의 `entity_types`를 그곳에 복사하세요.
- 텍스트가 아닌 형식은 수집과 같은 로더(`ParserFactory.register_loader`로 추가한 형식
  포함)를 거치며, 파싱하지 못한 파일은 경고와 함께 건너뜁니다.
- 모델이 JSON 프로필을 돌려주지 않으면 0이 아닌 코드로 끝나고 아무것도 쓰지 않습니다.

---

## 9. 도메인 적응

파이프라인을 의료, 법률, 금융 같은 도메인에 맞추는 방법은 두 가지입니다.

### A. `entity_types`(비용이 가장 적고 효과가 가장 큼)

프롬프트를 다시 쓰지 않고 추출 프롬프트가 찾을 범주를 정합니다.

```yaml
processing:
  graph_extraction:
    entity_types:
      - "GENE: Genes, gene products, loci"
      - "DISEASE: Disorders, syndromes, conditions"
      - "DRUG: Medications, compounds, dosages"
      - "TRIAL: Clinical trials, studies, cohorts"
```

### B. `custom_prompts` 재정의

프롬프트의 `*_system` / `*_human` 텍스트를 재정의합니다(`null`이면 내장 프롬프트를
씀. 재정의 키가 없는 프롬프트 다섯 개는 [사용자 정의 프롬프트](#사용자-정의-프롬프트)
참고). `{braces}` 안의 변수는 프레임워크가 채우므로 그대로 두세요.

```yaml
custom_prompts:
  # human만 오버라이드하면 내장 system 프롬프트와 출력 형식이 그대로 유지됩니다.
  graph_extraction_human: |
    Extract medical entities and relationships from this clinical text:
    {input_text}
    Extraction Limits:
    - Maximum Entities: {max_entities_per_chunk}
    - Maximum Relationships: {max_relationships_per_chunk}

  entity_extraction_system: |
    You are a financial expert. Extract companies, instruments, markets, and metrics
    from user queries.
```

재정의 키(각각 `_system` + `_human`): `graph_extraction`,
`description_summarization`, `claim_extraction`, `graph_refinement`,
`community_report`, `answer_generation`, `query_rewrite`, `entity_extraction`,
`drift_hyde`, `drift_primer`, `drift_action`, `strategy_selection`,
`keywords_extraction`, `global_map`, `pairwise_comparison`(`--pairwise-judge` 판정
모델. JSON 판정 형식은 유지), 그리고 프롬프트 튜닝의 `corpus_profile`.

재정의는 설정을 읽을 때 검사하며, 잘못되면 해당 키를 알리고 읽기에 실패합니다.

- 프롬프트 호출에 쓰이지 않는 `{name}`은 오류입니다(호출할 때마다 `KeyError`가 남).
  중괄호를 글자 그대로 쓰려면 `{{` / `}}`로 씁니다: `{{"name": "Vendor"}}`.
  `run-prompt-tuning` 출력은 이미 이스케이프되어 있습니다.
- 각 프롬프트의 데이터 변수는 system이나 human 템플릿에 있어야 합니다
  (`graph_extraction`과 `claim_extraction`은 `{input_text}`, `answer_generation`은
  `{query}`와 `{context}` 등). `{max_entities_per_chunk}` 같은 한도와
  `{entity_types}`는 빼도 됩니다.

`*_system` 오버라이드는 출력 지시를 포함한 내장 시스템 프롬프트 전체를 대체합니다.
`graph_extraction`과 `community_report`에서는 파서가 읽는 XML 형식(추출에서는 환각 방지
검사가 확인하는 원문 그대로의 `<source_text>` 규칙 포함)이 여기에 있으므로 남겨야
합니다. `run-prompt-tuning` 출력에서 시작하거나
`unified_kg_rag/domain/prompts/graph_extraction.py`의 형식을 복사하세요. 두 프롬프트의
오버라이드에 파서가 읽는 태그(`<entities>`, `<relationships>`, `<source_text>`,
`<community_name>`, `<summary>`, `<rating>`, `<rating_explanation>`, `<findings>`)가
빠지면 빠진 태그를 알리는 경고를 남깁니다.

**권장 흐름:** `run-prompt-tuning` → 검토 → 쓸 만한 프롬프트를 합치고 `entity_types`를
조정 → 다시 수집.

---

## 10. 운영과 문제 해결

[IAM 권한](#iam-권한)은 §1에, 오류 메시지와 해결 방법은
[운영 런북: 자주 보는 오류](./operations.ko.md#자주-보는-오류)에 있습니다.

### CLI 종료 코드와 JSON 출력

모든 CLI(`run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
`run-prompt-tuning`, `prepare-benchmark`)는 다음 코드로 종료합니다.

| 코드 | 의미 |
|---|---|
| `0` | 성공 |
| `1` | 실행 실패: 단계나 질의 실패(`run-ingestion`은 실행 중에 발견한 `ConfigurationError`도 포함. 예: 레지스트리를 끈 채 만든 인덱스에 대한 증분 실행), `--max-failure-rate` 초과, 업로드 실패, 렌더링한 파일 없음 |
| `2` | 사용법·설정 오류: 잘못된 플래그, 없거나 잘못된 입력(`run-eval` 데이터셋, `run-visualization` 데이터 파일, `prepare-benchmark` 데이터 파일), 잘못된 설정 파일이나 덮어쓰기, 설정하지 않은 저장소 엔드포인트(모두 작업 시작 전), 또는 `run-ingestion` 외의 CLI가 실행 중에 발견한 `ConfigurationError` |
| `130` | Ctrl-C(SIGINT)로 중단 |
| `143` | SIGTERM으로 중지(ECS `StopTask`, Step Functions 시간 초과). `run-ingestion`은 끝난 단계를 먼저 체크포인트로 저장하고, `run-eval`은 끝난 실행의 결과를 `s3://` 출력에 업로드함(실패하면 로컬에 남기고 경로를 로그로 남김) |

`130`이나 `143`으로 중지된 실행은 프로세스를 바로 끝냅니다. 진행 중인 모델 호출은 기다리지 않고 버리며, 그 전에 로그, 추적 데이터, 표준 출력을 비우고 프로세스 풀의 작업자도 함께 끝냅니다.

플래그는 전체 이름으로 써야 하며, `--output`처럼 앞부분만 쓰면 거부합니다.

`--output-format json`은 JSON 문서 하나를 표준 출력에 쓰고 배너·진행 상황·로그는 표준
오류로 보내므로 출력을 `jq`로 바로 넘길 수 있습니다(표준 출력에 함께 쓰는
`--metrics-sink cloudwatch`와는 같이 쓸 수 없음). 담기는 내용은 답변이나 검색
결과(`run-rag`), 파이프라인 ID와 상태, 단계별 결과(`run-ingestion`), 실행별 요약과 비교,
실패 예산 초과 내역(`run-eval`), 쓴 파일(`run-visualization`), 출력 경로와 프로필
(`run-prompt-tuning`), 쓴 항목 수(`prepare-benchmark`)입니다. 문서를 출력하기 전에 신호로 중지되면
대신 `{"status": "interrupted", "exit_code": 143}`(Ctrl-C는 `130`)을, 출력하기 전에
실패하면 `{"status": "failed", "exit_code": 1}`(또는 `2`)을 출력하므로 표준 출력은 항상
JSON입니다. `run-ingestion`은 재개할 때 쓸 `pipeline_id`를 덧붙입니다.

### 대규모, 다국어, 이질적인 코퍼스

- **대규모:** Bedrock 할당량이 허용하는 만큼 `processing.max_concurrency`를 올리고,
  `indexing.neptune.index_concurrency`는 효과를 측정하기 전까지 `1`로 두세요(동시 쓰기
  배치는 공유 정점에서 충돌해 재시도함). `indexing.opensearch.persist_embedding_cache`와
  `--s3-sync`를 켜면 다시 실행할 때 임베딩을 다시 계산하지 않고, 문서 상태
  레지스트리(`providers.doc_status.enabled`)를 켜면 다음 실행부터 바뀐 문서에만 비용이
  듭니다.
- **다국어:** [다국어 수집](#다국어-수집)을 참고하세요.
- **이질적인 도메인:** `entity_types`를 여러 도메인의 합집합으로 맞추거나, 도메인마다
  [인덱스 접미사](#인덱스-접미사)를 따로 두세요.

### 비용 참고

비용은 대부분 LLM 호출에서 나옵니다. 추출(청크당 1회 이상), gleaning(청크당 최대
`max_rounds`회), 커뮤니티 리포트, 주장(텍스트 단위당 1회, 기본은 꺼짐), 질의마다의 답변
생성입니다. 기계적인 역할에 Haiku급 모델 쓰기(청킹, 번역, map-reduce, 요약의 기본값),
gleaning 횟수 줄이기, 주장 끄기, 캐시와 증분 인덱싱으로 줄일 수 있습니다.

---

## 11. Python 라이브러리로 사용하기

CLI는 공개 클래스 두 개를 감싼 래퍼입니다. 이 클래스와 그 밖에 쓰는 이름은 모두 패키지
최상위에서 import하세요. `unified_kg_rag.__all__`의 이름이 공개 API이고, 그 밖의 모듈
경로는 내부 구현이라 바뀔 수 있습니다. `get_config`는 CLI와 같은 설정 파일·환경 변수
단계를 적용합니다(§2).

```python
from unified_kg_rag import (
    DataIngestionPipeline,
    GraphRAGChain,
    PipelineConfig,
    PipelineStageStatus,
    RAGInput,
    get_config,
)

config = get_config("config.yaml")

# Ingest. PipelineConfig carries the run-ingestion run options (pipeline_id,
# cache directory, S3 sync, enabled stages, force_rebuild). run() ingests the
# directory it is given: parsing, loading and the registry scope all use it.
pipeline = DataIngestionPipeline(config, PipelineConfig(pipeline_id="run-001"))
try:
    context = pipeline.run("./source")
    if context.status == PipelineStageStatus.FAILED:
        raise SystemExit("ingestion failed")
finally:
    pipeline.close()  # releases the indexers' Neptune/OpenSearch connections

# Query. One chain serves many queries, including concurrent ones.
chain = GraphRAGChain(config)
try:
    output = chain.invoke(RAGInput(query="How are Alice and Acme related?", search_strategy="mix"))
    print(output.answer)
    for source in output.sources:  # only what the answer model saw
        print(source)
finally:
    chain.close()
```

- 단계가 실패하면 예외를 던지지 않고 반환한 `PipelineContext`의 `status`를 `FAILED`로
  설정하므로 위 예시처럼 확인하세요.
- `RAGInput`은 `run-rag` 옵션을 필드로 받습니다(`search_strategy`, `search_type`,
  `top_k`, `suffix`([인덱스 접미사](#인덱스-접미사)), `filters`, `conversation_id`,
  `use_memory`, `target_language`). 일반 딕셔너리도 됩니다.
- `chain.search(...)`(`await chain.asearch(...)`)는 답변 없이 검색만 해서
  `SearchOutput`(`search_results`, `processed_query`, `metadata`)을 반환합니다.
- 비동기 코드에서는 `await chain.ainvoke(...)`와 `await chain.aclose()`를 쓰고, 시작할
  때 이벤트 루프의 실행기 크기를 한 번 지정하세요:
  `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`.
- 만든 객체는 항상 `close()` / `aclose()`로 닫으세요. 두 클래스 모두 처음 쓸 때 연결을
  열고, 닫지 않으면 가비지 컬렉션 때까지 유지합니다.
- 패키지를 import해도 로깅 핸들러는 설정하지 않습니다. CLI와 같은 로깅은
  `unified_kg_rag.setup_logging(config)`로 켭니다. 호출하지 않으면 구조화 필드
  (`structlog.contextvars`로 바인딩한 값 포함)가 레코드 속성으로 호스트 핸들러에
  전달되며, `LogRecord` 속성과 이름이 같은 필드(`name`, `msg` 등)에는 끝에 `_`가
  붙습니다.

확장 코드도 패키지 최상위에서 import합니다. 포트(`LLMFactoryPort`, `GraphIndexer`,
`DocStatusPort` 등), `Providers`, `register_provider`, `BaseSearchStrategy`와
`register_strategy`, `ParserFactory`, `register_renderer`가 여기에 있습니다. 두 생성자
모두 백엔드를 `providers=Providers(config, ...)` 번들이나 개별 인자(`doc_status=`,
`vector_indexer=`, `model_factory=` 등) 중 하나로 주입받으며, 둘을 함께 쓸 수는
없습니다. [설계 문서 §15](./design.ko.md#15-확장-가이드)를 참고하세요.

---

## 12. 제한 사항

- **추출 단계에서 환각이 생길 수 있습니다.** 원문이 뒷받침하지 않는 엔티티나 관계가
  만들어져 검색까지 이어질 수 있습니다.
  `processing.graph_extraction.entity_grounding.enabled`(기본은 꺼짐)는 인용한
  `source_text`가 청크에 없는 항목을 버리거나 가중치를 낮춥니다.
- **실행 결과를 재현할 수 없습니다.** LLM 출력(추출, 리포트, `auto` 라우터, 답변)은
  달라지므로 설정은 여러 번 실행해 비교하세요. 모델 응답이 같으면 나머지 처리는 어느
  프로세스에서든(`PYTHONHASHSEED`와 관계없이) 결정적입니다.
- **재순위화는 리전에 따라 다릅니다**([필수 모델](#필수-모델)). 없으면 융합 순서를 쓰며,
  `metadata.rerank`에 나타납니다([재순위화 상태](#재순위화-상태)).
- **기본 제공 백엔드.** 모델: Bedrock(기본값), Ollama, OpenAI 호환 서버. 재순위화:
  Bedrock이나 sentence-transformers 크로스 인코더([로컬 모드](#로컬-모드-aws-없이)).
  저장소: Neptune(또는 Gremlin Server)과 OpenSearch, 또는 내장 SQLite 저장소(호스트 한
  대, 인덱스당 문서 약 100,000개). 그 밖의 것은 `register_provider` 빌더나 사용자 정의
  어댑터가 필요합니다([설계 문서 §15](./design.ko.md#15-확장-가이드)).
- **서빙 계층이 없습니다.** HTTP API, 최종 사용자 인증, 요청 속도 제한은 없습니다.
- **필터는 접근 제어가 아닙니다**(§4). 테넌트는 인덱스 접미사로 분리하세요.
- **가벼운 수집은 쓸 수 있는 전략을 제한합니다.** 커뮤니티 탐지를 끄면
  (`graph.community_detection.enabled: false`) `global`과 `drift`가 쓸 리포트가 없고,
  관계 벡터 인덱스를 끄면(`indexing.opensearch.build_relationship_vector_index: false`)
  `mix`와 `hybrid`가 고수준 키워드를, `local`이 관계 섹션을 쓰지 못합니다.
- **증분 인덱싱은 근사치입니다**(§5). 증분 실행은 다시 군집화하지 않고 커뮤니티를
  덧붙이며, 엔티티가 현재 이름의 id를 받은 뒤에도 저장된 커뮤니티와 리포트에는 이전
  id가 남습니다. 둘 다 전체 재구축(`indexing.reset: true`)으로 갱신됩니다. DynamoDB에서는
  문서 하나의 청크 ID가 400KB 항목 하나에 들어가야 합니다(*문서 크기 한도*).
- **동시 실행에는 레지스트리가 필요합니다.** 레지스트리의 쓰기 임대가 한
  네임스페이스의 두 번째 실행을 멈춥니다. 레지스트리가 없거나 `IncrementalIndexer`
  API를 직접 쓰면 잠그는 장치가 없어, 같은 접미사의 수집 두 개가 서로의 인덱스를 지울
  수 있습니다([동시 실행](./operations.ko.md#동시-실행)).
- **태스크를 죽이는 문서를 가려내려면 여러 번 실행해야 합니다**(*격리된 문서*). 함께
  죽은 문서 `n`개 가운데 원인 문서 하나에 대략 `log2(n) + 1`번이 걸리고, 그동안 미룬
  문서는 기다립니다.
- **Markdown과 HTML**에는 `unstructured` extra(Python 3.11 이상)가 필요하며, 컨테이너
  이미지에는 기본으로 빠져 있습니다.
- **임베딩 모델(또는 차원)을 바꾸면** `indexing.reset: true`가 필요합니다. 증분 실행은
  다른 모델의 인덱스에 쓰지 않습니다([작업 흐름](#작업-흐름)).

---

*함께 보기: 개요와 빠른 시작은 [README.ko.md](../README.ko.md), 아키텍처와 내부 구조는
[설계 문서](./design.ko.md)를 참고하세요.*
