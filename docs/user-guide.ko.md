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

각 CLI의 `--help` 끝에는 실행 예시와 이 문서의 해당 절 링크가 있습니다.

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

관련 문서: [설정 레퍼런스](./configuration.md)(영문) · [모델 카탈로그](./models.ko.md) · [운영 런북](./operations.ko.md) ·
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
| **Amazon Bedrock** | 필수 | 모든 LLM 호출(청킹, 추출, gleaning, 커뮤니티 리포트, 답변 생성), 임베딩, 재순위화. 설정한 모델 ID의 모델 액세스를 활성화해야 합니다. |
| **Amazon Neptune** | 필수 | 지식 그래프(엔티티, 관계, 커뮤니티)와 질의 시점의 다중 홉 순회. |
| **Amazon OpenSearch** | 필수 | 벡터 인덱스와 BM25 어휘 인덱스(텍스트 단위, 엔티티, 커뮤니티 리포트, 관계, 주장). |
| **Amazon S3** | `--s3-sync`, `indexing.opensearch.persist_embedding_cache`, `s3://` 경로를 쓸 때만 | 단계 캐시 동기화, 임베딩 캐시 저장, `s3://` 평가 데이터와 결과. CDK 파이프라인은 코퍼스도 S3에 둡니다. |
| **Amazon DynamoDB** | 증분 인덱싱을 쓸 때만 | 콘텐츠 해시로 코퍼스의 변경분을 계산하는 문서 상태 레지스트리. |

### 필수 모델

처음 실행하기 전에 다음 Bedrock 모델의 액세스를 활성화하세요. 표의 모델은 코드
기본값이며, 다른 모델을 설정하면 그 행의 모델이 설정한 모델로 바뀝니다.

| 역할 | 기본 모델 ID | 설정 키 |
|---|---|---|
| 기본 계층(추출, gleaning, 커뮤니티 리포트, 답변, 판정 모델) | `anthropic.claude-sonnet-5-5` | `models.default` |
| 빠른 계층(청킹, 번역, 요약, 라우팅, map 단계) | `anthropic.claude-haiku-5-5` | `models.fast` |
| 임베딩(인덱싱, 질의, 평가, 시각화 레이아웃) | `amazon.titan-embed-text-v2:0` | `models.embedding.model_id` |
| 재순위화 | `cohere.rerank-v3-5:0` | `models.rerank` |

- **리전.** 재순위화 호출을 포함한 모든 Bedrock 호출은 `aws.bedrock.region_name`으로
  가며, 이 값의 기본값은 `aws.region_name`(`us-west-2`)입니다. 이 리전에는 기본 모델
  네 개가 모두 있습니다. 재순위화 모델은 모든 리전에 있지 않습니다(예:
  `ap-northeast-2`에는 없음). 그런 리전에서는 `aws.bedrock.region_name`을 재순위화
  모델이 있는 리전으로 지정하거나 `search.reranking.enabled: false`로 설정하세요.
  Bedrock으로 가는 경로가 VPC 엔드포인트뿐인 프라이빗 VPC에서는 Bedrock 리전을 VPC와
  같은 리전으로 유지해야 합니다.
- **추론 프로파일.** Claude Sonnet 5.5와 Haiku 5.5는 교차 리전 추론 프로파일로만
  호출할 수 있습니다. `aws.bedrock.enable_global_profile: true`를 유지하고
  `bedrock:ListInferenceProfiles` 권한을 부여하세요. 프로파일을 찾지 못하면 모델
  생성이 실패하며, 오류 메시지에 이 해결 방법이 나옵니다.
- **권한.** [§10 IAM 권한](#iam-권한)을 참고하세요. 재순위화에는 별도 정책 문이
  필요합니다.

[모델 카탈로그](./models.ko.md)에 선별된 모델 전체와 각 모델의 한도가 있습니다.

이 프레임워크는 **이미 있는** 서비스에 연결할 뿐 서비스를 만들지 않습니다. 실행
방법은 세 가지입니다.

- **로컬 모드(개발용).** 모든 구성 요소(Ollama 모델과 내장 SQLite 저장소, 또는
  컨테이너로 띄운 Gremlin Server와 OpenSearch)를 로컬에서 실행하거나, 저장소만 로컬로
  두고 모델은 Bedrock을 씁니다. 아래 [로컬 모드](#로컬-모드-aws-없이)를 참고하세요.

- **기존 서비스 사용.** Neptune과 OpenSearch(쓴다면 S3와 DynamoDB도)가 이미 실행
  중이라면 엔드포인트를 `config.yaml`(아래 §2)에 적고 다음 단계로 넘어가면
  됩니다. 설정한 모델 ID의 Bedrock 모델 액세스가 활성화되어 있는지 확인하세요.
- **번들 CDK 앱으로 전체 프로비저닝.** 저장소의 [`iac/`](../iac/README.md)에는
  스택 전체를 명령 하나로 만드는 선택형 Well-Architected AWS CDK 앱이 있습니다.
  네트워킹(VPC와 엔드포인트), Neptune 클러스터, OpenSearch 도메인, DynamoDB 문서
  상태 테이블, S3 캐시 버킷, ECS Fargate 데이터 플레인, Step Functions 수집
  파이프라인, CloudWatch 관측성을 만들고, 선택적으로 리전을 고정한 Bedrock 가드레일도
  만듭니다. 배포 명령([Usage](../iac/README.md#usage)), `-c key=value` 옵션 전체,
  프로덕션 강화 체크리스트는 [`iac/README.md`](../iac/README.md)에 있습니다.
  Neptune과 OpenSearch는 VPC 안에서만 접근할 수 있으므로 CLI는 배포된 태스크 정의의
  태스크로 VPC 안에서 실행하며, 이 태스크는 엔드포인트를 환경 변수로 받습니다
  ([After deploy](../iac/README.md#after-deploy)).

### 설치

```bash
git clone https://github.com/awslabs/unified-kg-rag-on-aws.git
cd unified-kg-rag-on-aws

# uv (recommended)
uv sync

# or pip
pip install -e .
```

선택 extra: **Markdown(.md)**과 **HTML(.html)**을 파싱하려면 `unstructured`
패키지가 필요합니다. 이 패키지가 없으면 `.pdf`, `.txt`, `.csv`, `.json`만
파싱합니다. 수집은 `.md`/`.html` 파일을 건너뛰고 확장자마다 설치 명령이 담긴 경고를
한 번 남깁니다(`Skipping 2 '.md' file(s) ... install the
optional 'unstructured' extra`). Python 3.11 또는 3.12에서는
`uv sync --extra unstructured` 또는 `pip install -e '.[unstructured]'`로 패치된
파서를 설치하세요. 이 extra는 `unstructured>=0.24.1`을 요구하며, 이 버전은 URL
분할의 SSRF 취약점과 생성 HTML의 마크업 미이스케이프 문제를 고쳤고 NLTK에 의존하지 않습니다. Python 3.10에서는 이 extra를
지정해도 파서가 설치되지 않으므로 PDF/TXT/CSV/JSON을 쓰거나, Markdown/HTML이
필요하면 Python을 올리세요.

배포용 컨테이너 이미지(`docker/Dockerfile`)에는 이 extra가 기본으로 빠져 있습니다.
약 200MB(spaCy 등)가 늘어나기 때문입니다. 따라서 Step Functions로 `.md`/`.html`
파일을 수집하면 "No supported files found"로 실패합니다. 해당 파일을 지원 형식으로
바꾸거나, extra를 넣어 이미지를 빌드하세요:
`docker build --build-arg UV_EXTRAS="--extra unstructured" -f docker/Dockerfile .`.
이미지에 들어가는 `docker/config.yaml`에는 엔드포인트가 없으며, CDK compute 스택이
엔드포인트를 환경 변수로 주입합니다.

### 인증

인증은 서로 독립된 두 가지입니다.

1. **AWS 자격 증명** — 표준 자격 증명 체인으로 공급합니다. 이름 있는 프로파일을
   쓰려면 `config.yaml`에 `aws.profile_name`을 지정하고, 기본 체인(환경 변수,
   인스턴스 역할 등)을 쓰려면 `null`로 둡니다. `aws.neptune.use_iam: true`이면
   Neptune 요청에 SigV4를 씁니다.

2. **OpenSearch 인증** — IAM(`aws.opensearch.use_iam: true`) 또는
   username/password 중 하나입니다. username/password를 쓰려면 `use_iam: false`로
   설정하고 `.env` 파일을 만드세요(`.env-template` 복사).

   ```bash
   # .env — only needed when aws.opensearch.use_iam is false
   UKR__AWS__OPENSEARCH__USERNAME=your_opensearch_username
   UKR__AWS__OPENSEARCH__PASSWORD=your_opensearch_password
   ```

   `run-*` CLI는 `.env` 파일을 자동으로 읽습니다.
   `use_iam: true`이면 `.env`가 필요 없습니다.

### 로컬 모드 (AWS 없이)

로컬 모드는 AWS 계정 없이 노트북이나 호스트 한 대에서 수집, 질의, 평가를 실행합니다.
역할마다 Bedrock, Neptune, OpenSearch, DynamoDB 대신 로컬 구성 요소를 씁니다. 기본값과
주 대상은 여전히 AWS이며, 로컬 모드는 개발과 평가용입니다. 기본 설정에서는 그래프 저장소,
벡터·어휘 저장소, 증분 인덱싱 레지스트리가 모두 SQLite 파일이므로 실행할 서비스는
[Ollama](https://ollama.com)(직접 설치, `http://localhost:11434`에서 서비스) 하나뿐입니다.

```bash
uv sync --extra local
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
uv run run-ingestion --backend local --source-directory ./docs-in
uv run run-rag --backend local --query "..."
```

설정 파일이 없으면 저장소는 작업 디렉터리의 `.unified-kg-rag/`에 만들어집니다
(`graph.sqlite3`, `vector.sqlite3`, `doc-status.sqlite3`). 따라서 모든 명령을 같은
디렉터리에서 실행하거나, `backend: local`을 적은 설정 파일에 설정을 두세요(이때 상대
경로는 그 파일 기준으로 해석합니다). 프리셋에서 증분 인덱싱은 꺼져 있으며,
`providers.doc_status.enabled: true`로 켭니다. 처음부터 다시 하려면 세 파일을 함께
지우거나(레지스트리는 저장소에 든 내용을 기록합니다) `indexing.reset: true`로 한 번
실행하세요. Gremlin Server와 OpenSearch를 컨테이너로 실행하려면
[컨테이너로 띄운 로컬 저장소](#컨테이너로-띄운-로컬-저장소)를 참고하세요.

Linux에서 `uv sync --extra local`은 GPU 유무와 관계없이 잠금 파일이 고정한 CUDA용
PyTorch 빌드(수 GB)를 크로스 인코더용으로 설치합니다. CPU 빌드를 쓰려면 uv의 pip
인터페이스로 extra를 설치하세요.

```bash
uv sync
uv pip install --torch-backend cpu -e '.[local]'
```

`uv run`은 이 패키지를 그대로 두지만, 나중에 그냥 `uv sync`를 실행하면 지워지고
`uv sync --extra local`을 실행하면 잠금 파일의 CUDA 빌드로 돌아갑니다.

`backend: local`(설정 파일에 적거나 `--backend local`로 지정. [공급자 플래그](#공급자-플래그)
참고)은 설정에서 비워 둔 모든 키에 아래 공급자를 채웁니다. 파일에 적은 키는 항상
프리셋보다 우선합니다. `providers.<역할>.kind`, `aws.neptune`이나 `aws.opensearch`의
연결 키(`port`, `use_iam`, `use_ssl` 등), 임베딩·재순위화 모델 ID를 적은 설정은
`backend: local`에서도 그 값을 유지합니다. `config-template.yaml`은 이 키들을 주석으로
남겨 두었으므로, 사본에 `backend: local`만 적으면 전환됩니다. 역할별 선택은 `providers:`
아래에 있고, 역할을 섞어 쓸 수도 있습니다. 예를 들어 저장소만 로컬로 두고 모델은
Bedrock을 쓸 수 있습니다.

| 역할 | `providers.<역할>.kind` | 로컬 기본값 |
|---|---|---|
| 언어 모델 | `llm`: `bedrock` \| `openai_compatible` \| `ollama` | `ollama`, `http://localhost:11434`의 `qwen2.5:7b-instruct` |
| 임베딩 | `embedding`: `bedrock` \| `openai_compatible` \| `ollama` | `ollama`, `bge-m3`(1024차원) |
| 재순위화 | `rerank`: `bedrock` \| `cross_encoder` \| `none` | `cross_encoder`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| 그래프 저장소 | `graph`: `neptune` \| `gremlin` \| `sqlite` | `providers.graph.sqlite_path`(기본값 `.unified-kg-rag/graph.sqlite3`)의 `sqlite` |
| 벡터·어휘 저장소 | `vector`: `opensearch` \| `sqlite` | `providers.vector.sqlite_path`(기본값 `.unified-kg-rag/vector.sqlite3`)의 `sqlite` |
| 문서 상태 레지스트리 | `doc_status`: `dynamodb` \| `sqlite` | `providers.doc_status.sqlite_path`(기본값 `.unified-kg-rag/doc-status.sqlite3`)의 `sqlite` |

`gremlin`은 TinkerPop Gremlin Server, `opensearch`는 로컬 OpenSearch이며, 프리셋이
`localhost`로 지정하는 `aws.neptune`과 `aws.opensearch` 키로 연결합니다
([컨테이너로 띄운 로컬 저장소](#컨테이너로-띄운-로컬-저장소) 참고).

- **모델.** Bedrock이 아닌 LLM을 쓰면 `models.default`가 필수이고(프리셋이 지정합니다)
  `models.fast`는 그 값을 따릅니다. `models.roles` 항목을 지정하면 그 역할에서는 그
  값이 우선합니다. `providers.llm.context_window_size`와
  `max_output_tokens`로 모델 한도를 알려 줍니다(Bedrock 모델은 카탈로그 값을 씁니다).
  컨텍스트 창은 Ollama에 `num_ctx`로 전달됩니다. 프리셋은
  `search.token_manager.max_context_tokens`도 22000으로 설정합니다. 기본 32,768토큰
  창이 8,192토큰 출력 예약과 함께 받을 수 있는 값이며, AWS 기본값 30000이면 질의마다
  잘립니다. 이 값은 기본 창을 그대로 쓰는 Bedrock 외 LLM에만 적용되며,
  `context_window_size`를 지정하거나 `kind: bedrock`을 쓰면 모델 창에 맞춰 잘리는 일반
  기본값을 씁니다. OpenAI 호환 서버(vLLM, llama.cpp
  server, LM Studio)에는 `base_url`(`/v1` 루트)과, 필요하면 `api_key`를 지정합니다.
- **임베딩.** Bedrock이 아닌 공급자에서는 `models.embedding.model_id`와 `dimension`이
  필수입니다. 차원은 벡터 저장소 매핑에 고정되므로 둘 중 하나를 바꾸면 전체를 다시
  인덱싱해야 합니다. 프리셋의 차원(1024)은 프리셋 모델(`bge-m3`)의 차원이므로 다른
  모델을 지정하면 차원도 함께 지정해야 합니다.
- **재순위화.** 크로스 인코더는 첫 질의 때 Hugging Face에서 모델을 내려받습니다.
  `[local]` extra가 없거나 `kind: none`이면 융합 순서를 그대로 씁니다. 이 경우 로그에 한 번
  남고 융합 스팬에 `kg_rag.rerank_enabled=false`로, `metadata.rerank.status`에
  `unavailable`(`kind: none`이면 `disabled`)로 보입니다.
- **내장 저장소.** `sqlite` 저장소는 각각 파일 하나이며, 처음 쓸 때 만들어집니다.
  상대 경로 `sqlite_path`는 설정 파일 디렉터리(설정 파일이 없으면 작업 디렉터리)를
  기준으로 해석합니다. `indexing.neptune`과 `indexing.opensearch` 설정(인덱스 접두사,
  배치 크기, 임베딩 설정)은 Neptune과 OpenSearch에서처럼 이 저장소에도 적용되며, 쓰기는
  다른 프로세스의 잠금을 `search.store_timeout_seconds`까지 기다립니다.
- **증분 인덱싱**은 `providers.doc_status.enabled: true`(또는 기존 스위치인
  `aws.dynamodb.enabled: true`)일 때 SQLite 레지스트리를 씁니다.
  [`docker/config.local.yaml`](../docker/config.local.yaml)은 이를 켭니다.
  상대 경로 `sqlite_path`는 저장소와 같은 방식으로 해석하며, 실행 로그에
  데이터베이스의 절대 경로를 남깁니다.
- **토큰 계산**은 문자 체계를 고려한 추정치(Bedrock이 대체 수단으로 쓰는 것과 같음)를
  씁니다. `providers.llm.tokenizer: tiktoken`이면 BPE 인코딩으로 세며, 처음 쓸 때 인코딩
  파일을 내려받습니다.
- **사용자 정의 kind**는 `adapters.provider_registry.register_provider`로 빌더를 등록합니다.
  [설계 문서 §15](./design.ko.md#15-확장-가이드)를 참고하세요.

AWS 모드와 다른 점:

- 작은 로컬 모델은 추출 품질이 눈에 띄게 낮습니다. 형식이 잘못된 출력이 많아지고(재시도
  후 실패한 청크로 기록), 엔티티 해석과 커뮤니티 리포트도 약해집니다. 하드웨어가 감당하는
  가장 큰 모델을 쓰세요.
- Bedrock 프롬프트 캐싱이 없고, 로컬 공급자는 모델 호출에 가드레일을 붙이지 않습니다.
  `aws.bedrock.guardrail.identifier`를 지정하면 질의 체인은 여전히 Bedrock
  `ApplyGuardrail`로 질의와 답변을 검사합니다(AWS 자격 증명 필요). 추론 노력(effort)
  설정도 적용되지 않습니다.
- 내장 벡터 저장소는 OpenSearch 어댑터가 만드는 질의에 답하지만 다섯 가지가 다릅니다.
  `nori`(한국어), `kuromoji`(일본어), `smartcn`(중국어)으로 매핑한 텍스트는 형태소
  분석 대신 서로 겹치는 CJK 문자 바이그램으로 나눕니다. 그래서 긴 단어 안의 이름도
  찾지만(홍길동으로부터의 홍길동) 순위는 OpenSearch와 다릅니다. `english` 분석기는
  `[local]` extra의 `snowballstemmer`가 있을 때만 어간을 추출합니다(없으면 경고를 한 번
  남깁니다). 이모지는 색인하지 않습니다(OpenSearch 토크나이저는 이모지 하나를 한 단어로
  둡니다). 퍼지 매칭(OpenSearch의 `fuzziness: AUTO`)이 없으므로 철자가 틀린 질의어와,
  어간을 추출하지 않는 `standard` 분석기에서는 굴절형("vendor"에 대한 "vendors")도 벡터
  검색으로만 찾습니다. kNN 검색은 정확 검색입니다. 필터를 통과한 문서를 모두 채점하므로
  정확하지만 비용이 인덱스 크기에 비례해 늘어납니다. 하이브리드 검색은 두 저장소 모두에서
  리트리버가 같은 방식으로 결합합니다(최소 50건씩 가져오는 어휘 검색과 kNN 검색을 따로 실행하고
  각각을 자기 결과 안에서 min-max 정규화). 따라서 차이는 그 두 검색이 다른 부분에서만 생깁니다.
- 내장 벡터 저장소는 검색하는 인덱스를 메모리에 올립니다(벡터 필드마다 약 문서 수 x
  차원 x 4바이트). 인덱스당 문서 약 100,000개를 기준으로 설계했으며, 더 큰 인덱스를
  올리면 경고를 남깁니다. 그 이상은 OpenSearch를 쓰세요. 쓰기가 있으면 다음 질의에서
  인덱스를 다시 읽고, 그 뒤 첫 어휘 검색은 검색하는 텍스트 필드를 다시 분석합니다.
  120단어 문서 50,000개 기준 약 10초입니다. `english` 분석기에서 서로 다른 단어가 약
  65,000개를 넘는 필드는 어간 추출 캐시에 다 들어가지 않아 이 질의가 약 세 배 느려집니다.
- SQLite는 한 번에 한 프로세스만 씁니다. 저장소 파일마다 수집은 한 번에 하나만
  실행하세요(질의는 수집과 함께 읽을 수 있습니다). `search.store_timeout_seconds`보다
  오래 기다린 쓰기는 "stayed locked"로 실패합니다([자주 보는 오류](./operations.ko.md#자주-보는-오류)
  참고). 파일은 로컬 디스크에 두세요. 네트워크 파일 시스템(NFS, SMB, Amazon EFS)은
  지원하지 않습니다.
- [컨테이너](#컨테이너로-띄운-로컬-저장소)를 쓰면 한국어 어휘 분석은 `nori` 플러그인(compose
  파일이 설치) 또는 OpenSearch에 있는 동등한 분석기에 의존합니다.
- 토큰 수는 추정치이므로 컨텍스트 예산도 근사치입니다.
- 로컬 모델의 응답이 출력 한도에서 잘려도 잘림으로 감지하지 못합니다(Bedrock은 중단
  사유를 알려 줍니다). `max_output_tokens`는 모델이 실제로 생성할 수 있는 값으로 두세요.
- `run-eval`의 LLM 심사(`langchain`, `ragas`)는 로컬 모델로 실행되어 점수의 신뢰도가
  낮습니다. 결정적 평가기는 영향을 받지 않습니다.

Bedrock과 마찬가지로 모델 서버가 질의의 호출을 거부하거나(잘못된 키나 모델 이름,
HTTP 4xx) 클라이언트 재시도 후에도 연결할 수 없으면, "찾을 수 없음"으로 답하지 않고
클라이언트의 오류로 질의를 실패시킵니다.
[자주 보는 오류](./operations.ko.md#자주-보는-오류)를 참고하세요.

#### 공급자 플래그

`run-ingestion`, `run-rag`, `run-eval`, `run-prompt-tuning`, `run-visualization`은 다음
플래그를 받습니다(`run-visualization`은 모델이나 저장소를 호출하지 않으므로, 모든 CLI에
같은 명령줄을 쓸 수 있도록 받기만 합니다).
플래그마다 읽어 들인 설정의 해당 키를 덮어쓰며, 파일에 적은 키도 덮어씁니다. 플래그를
생략하면 설정을 그대로 씁니다.

| 플래그 | 설정하는 키 |
|---|---|
| `--backend {aws,local}` | `backend`. 프리셋은 여전히 설정에서 비워 둔 키만 채웁니다. `backend: local` 파일에 `--backend aws`를 주면 로컬 프리셋이 채운 값을 버려 해당 키가 AWS 기본값으로 돌아갑니다. 파일이나 환경 변수로 지정한 값은 프리셋과 같더라도 유지합니다 |
| `--llm-provider KIND` | `providers.llm.kind`: `bedrock`, `openai_compatible`, `ollama` |
| `--llm-model MODEL` | `models.default` |
| `--embedding-provider KIND` | `providers.embedding.kind`: `bedrock`, `openai_compatible`, `ollama` |
| `--embedding-model MODEL` | `models.embedding.model_id` |
| `--embedding-dimension N` | `models.embedding.dimension` |
| `--rerank-provider KIND` | `providers.rerank.kind`: `bedrock`, `cross_encoder`, `none` |
| `--model-base-url URL` | `providers.llm.base_url`과 `providers.embedding.base_url` |

#### 컨테이너로 띄운 로컬 저장소

[`docker/compose.local.yaml`](../docker/compose.local.yaml)은 저장소를 서버로
실행합니다. TinkerPop Gremlin Server(메모리 기반 TinkerGraph), 단일 노드 OpenSearch
2.13, 그리고 `models` 프로필에서 Ollama를 실행합니다. OpenSearch는 보안 플러그인을 끄고
`analysis-nori` 플러그인을 설치합니다(한국어 매핑이 `nori`를 씁니다. Amazon OpenSearch
Service 도메인에는 선택 사항인 패키지를 연결해야만 생깁니다. [다국어 수집](#다국어-수집) 참고). 이 OpenSearch는 Amazon OpenSearch Service와 같은
질의 엔진이므로 AWS에 배포하기 전에 순위(형태소 분석기, 퍼지 매칭)를 확인할 때 쓰세요.
[`docker/config.local.yaml`](../docker/config.local.yaml)은
`providers.graph.kind: gremlin`과 `providers.vector.kind: opensearch`를 지정하고 증분
인덱싱을 켭니다. Docker와 Compose 2.23.1 이상이 필요합니다(compose 파일은 Gremlin
Server 초기화 스크립트를 파일 안에 정의하며, 이전 릴리스는 이를 거부합니다).

```bash
uv sync --extra local
docker compose -f docker/compose.local.yaml --profile models up -d --wait
docker compose -f docker/compose.local.yaml exec ollama ollama pull qwen2.5:7b-instruct
docker compose -f docker/compose.local.yaml exec ollama ollama pull bge-m3
uv run run-ingestion --config-path docker/config.local.yaml --source-directory ./docs-in
uv run run-rag --config-path docker/config.local.yaml --query "..."
docker compose -f docker/compose.local.yaml down -v
```

`down -v`는 저장소 데이터를 지우지만 SQLite 레지스트리(`docker/config.local.yaml`이면
저장소 루트의 `.unified-kg-rag/doc-status.sqlite3`)는 지우지 않습니다. 새 저장소에
다시 수집하기 전에 이 파일을 지우거나 `indexing.reset: true`로 설정하세요. 그러지 않으면
증분 실행이 아무것도 쓰기 전에 멈춥니다(아래 Gremlin Server 설명 참고).

모델은 자동으로 내려받지 않습니다. `backend: local` 프리셋은
`aws.neptune.use_ssl: false`, `use_iam: false`(SigV4 없는 `ws://`)와
`aws.opensearch.allow_anonymous: true`, `use_ssl: false`(인증 없는 `http://`)로
`gremlin`과 `opensearch` kind를 이 컨테이너에 연결합니다.

모델 없이 저장소만 확인하려면 해싱 임베딩 공급자와 가짜 모델 서버를 쓰는 스모크 테스트를
실행하세요: `LOCAL_STORES=1 uv run pytest tests/integration/test_local_stores.py`.

Neptune과 다른 점이 있습니다. TinkerGraph는 그래프를 메모리에만 두므로 Gremlin Server를
다시 시작하면 그래프는 비지만 SQLite 레지스트리의 레코드는 남습니다. 이때 증분 실행은 잃어버린
문서를 변경 없음으로 처리하지 않고, 아무것도 쓰기 전에 멈추며 `indexing.reset: true`(또는
레지스트리 삭제)를 안내합니다. 또 정점을 만들 때 쓴 리스트 속성에 중복 값을 그대로 저장합니다(Neptune은 한 번만 저장). 모든 쓰기는 재시도해도 중복되지 않도록 먼저 `id` 속성으로 정점이나 엣지를 찾으며, compose 파일의 Gremlin Server는 시작할 때 정점과 엣지의 이 속성에 TinkerGraph 인덱스를 만듭니다. 이 인덱스 없이 띄운 Gremlin Server는 찾을 때마다 정점이나 엣지를 모두 훑으므로 그래프가 커질수록 느려집니다(엔티티 10,000개와 커뮤니티에 약 20초 대신 60초). 개발용 규모의 코퍼스에만 쓰세요. 컨테이너는 Linux
capability를 모두 버리지만 저장소에 인증과 TLS가 없으므로 compose 파일은 `127.0.0.1`에만 바인딩합니다.

#### 로컬 저장소와 Bedrock 모델

모델은 Bedrock을 쓰고 저장소만 로컬로 실행하려면 `backend: local`을 적은 설정에서
`providers.llm.kind`, `providers.embedding.kind`, `providers.rerank.kind`를 `bedrock`으로
바꾸세요(이때 Bedrock 모델 ID는 카탈로그로 검증합니다). Bedrock 액세스 권한이 있는 AWS
자격 증명이 필요합니다. 내장 저장소는 그 밖에 필요한 것이 없습니다. 컨테이너를 쓰려면
`models` 프로필 없이 저장소만 띄우고(`docker compose -f docker/compose.local.yaml up -d --wait`),
`docker/config.local.yaml` 사본에서 바꾸세요.

---

## 2. 설정

템플릿으로 설정 파일을 만들고 모든 CLI에 `--config-path config.yaml`로 지정하세요.

```bash
cp config-template.yaml config.yaml
```

설정 파일에는 바꿀 키만 적으면 되고, 나머지 키는 내장 기본값을 유지합니다. 모든 키의
타입, 기본값, 설명은 [설정 레퍼런스](./configuration.md)(영문)에 있습니다. 코드에서
생성하므로 항상 최신입니다. 이 절에서는 설정을 읽는 방식, 대부분의 실행에서 바꾸는
설정, 표 한 줄로 설명하기 어려운 주제(모델, 가드레일, 로그, 환경 변수, 추적)를
다룹니다.

### 설정을 읽는 순서

값은 여러 단계로 정해지며, 뒤 단계가 앞 단계를 덮어씁니다.

1. **내장 기본값**: `unified_kg_rag/domain/models/config.py`의 Pydantic 모델입니다.
   `--config-path` 없이 실행하면 CLI는 이 기본값에 환경 변수만 적용해 동작합니다.
2. **`backend` 프리셋**: `backend: local`(또는 `--backend local`)은 파일과 환경
   변수가 지정하지 않은 모든 키에 로컬 공급자와 저장소를 채웁니다
   ([로컬 모드](#로컬-모드-aws-없이)).
3. **YAML 파일**: 지정한 키는 기본값을 대체하고, 지정하지 않은 키는 기본값을
   유지합니다. 값이 딕셔너리인 키(예: `logging.library_levels`,
   `graph.visualization.interactive`)는 기본 딕셔너리와 병합되지 않고 통째로
   대체됩니다.
4. **환경 변수**([아래](#환경-변수로-덮어쓰기)): 파일 위에 적용됩니다.
5. **공급자 플래그**(`--backend`, `--llm-provider`, `--rerank-provider` 등): 마지막에
   적용됩니다. 설정은 파일과 환경 변수가 지정한 값(파이썬에서는 그 뒤 `Config` 객체에
   바꾼 값도)으로 다시 만들어지므로, `--backend`를 바꾸면 이전 프리셋이 채운 값만
   사라지고 사용자가 지정한 값은 프리셋 값과 같더라도 남습니다.

파일은 읽을 때 검증합니다. 타입이 틀리거나 지원하지 않는 값이면 잘못된 키를 밝힌
`Configuration validation error: ...`와 함께 CLI가 멈춥니다. 이 경우와 올바른 YAML이
아닌 파일, 존재하지 않는 `--config-path` 파일, 알 수 없는 키(파일이나 `UKR__` 변수)는
사용법 오류이며, 모든 CLI가 오류 한 줄을 출력하고 종료 코드 2로 끝납니다. 알 수 없는
키를 거부하는 이유는 철자가 틀린 키가 아무 표시 없이 설정을 기본값으로 남기기
때문입니다(가드레일 ID를 잘못된 키에 적으면 질의를 검사하지 않습니다).
`--allow-unknown-keys`(`get_config` / `ConfigLoader`에서는 `allow_unknown_keys=True`)를
주면 키마다 WARNING만 남기고 계속합니다. 위치가 바뀐 키는 대신할 키를 알려 주며
실패합니다([Moved keys](./configuration.md#moved-keys)).

`max_attempts` 키는 모두 첫 시도를 포함한 총 시도 횟수이며, `1`이면 재시도하지
않습니다.

### 대부분의 실행에서 바꾸는 설정

| 키 | 기본값 | 바꿀 때 |
|---|---|---|
| `aws.region_name` | `"us-west-2"` | 저장소가 다른 리전에 있을 때. `aws.bedrock.region_name`을 지정하지 않으면 Bedrock도 따름 |
| `aws.neptune.endpoint`, `aws.opensearch.endpoint` | `null` | AWS 저장소를 쓸 때 항상(CDK 스택이 주입함) |
| `aws.dynamodb.enabled` | `false` | 증분 인덱싱을 쓸 때([§5](#5-증분-인덱싱)) |
| `models.default`, `models.fast` | Claude Sonnet 5.5, Claude Haiku 5.5 | 다른 모델을 쓸 때([모델 선택 참고 사항](#모델-선택-참고-사항)) |
| `processing.max_concurrency` | `40` | 로그에 `ThrottlingException` 재시도가 보일 때(낮춤), Bedrock 할당량이 허용할 때(높임) |
| `processing.graph_extraction.entity_types` | 일반 유형 7개 | 추출을 도메인에 맞출 때([§9](#9-도메인-적응)) |
| `processing.translation.target_language` | `"en"` | 코퍼스나 답변을 다른 언어로 다룰 때([다국어 수집](#다국어-수집)) |
| `processing.claim_extraction.enabled` | `false` | `local` 검색에 주장을 넣을 때(텍스트 단위마다 LLM 호출 1회 추가) |
| `processing.document_parsing.index_value` | `null`(`default`) | 코퍼스나 테넌트를 여럿 분리할 때([인덱스 접미사](#인덱스-접미사)) |
| `graph.community_detection.enabled` | `true` | LightRAG로만 질의하고 수집을 가볍게 할 때(`global`, `drift`에는 필요) |
| `search.reranking.enabled` | `true` | Bedrock 리전에 재순위화 모델이 없거나 재순위화를 쓰지 않을 때 |
| `search.token_manager.max_context_tokens` | `30000` | 답변에 검색 컨텍스트가 더(또는 비용을 줄이려고 덜) 필요할 때([컨텍스트 예산](#컨텍스트-예산)) |
| `indexing.reset` | `false` | 코퍼스 전체로 저장소를 다시 만들 때(한 번의 실행에만) |
| `aws.bedrock.guardrail.identifier` | `null` | Bedrock 가드레일로 질의와 답변을 검사할 때([가드레일](#가드레일)) |
| `observability.enabled` | `false` | OpenTelemetry 트레이스를 남길 때([관측성](#관측성)) |

### 모델 선택 참고 사항

최상위 `models` 섹션이 역할마다 어떤 모델을 쓸지 정합니다. 각 모델 계열의 공급자는
`providers.llm.kind`, `providers.embedding.kind`, `providers.rerank.kind`이며, 모델
ID는 그 공급자를 기준으로 검사합니다. 모든 LLM 역할은 두 계층 중 하나에서 실행되므로
파이프라인 전체의 모델을 한 줄로 바꿀 수 있고, `models.roles` 항목(모델 ID, 또는 다른
계층으로 옮기는 `default` / `fast`)은 계층 설정보다 우선합니다.

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

임베딩 모델(`models.embedding.model_id`)은 인덱스, 질의, RAGAS, 시각화 배치에 함께
쓰이며, 이 모델이나 `dimension`을 바꾸면 다시 인덱싱해야 합니다. 임베딩 모델 ID와
재순위화 모델 ID는 정해진 목록에서만 고릅니다(`amazon.titan-embed-text-v2:0`,
`amazon.titan-embed-text-v1`, `cohere.embed-v4:0`, `cohere.embed-english-v3`,
`cohere.embed-multilingual-v3`; `cohere.rerank-v3-5:0`, `amazon.rerank-v1:0`). 그 밖의
ID는 검증에서 실패합니다. 선택하지 않은 공급자의 설정(언어 모델이 Bedrock인데
`providers.llm.temperature`를 지정한 경우 등)은 무시되며 로드할 때 WARNING을 남깁니다.

선별된 모델, Claude 세대별·OpenAI GPT 모델별 요청 차이, 프롬프트 캐싱, 패키지가 모르는
모델을 기술하는 방법(`aws.bedrock.model_overrides`)은 [모델 카탈로그](./models.ko.md)에
있습니다.

**출력 상한.** Bedrock은 요청을 시작할 때 입력 + `max_tokens`를 분당 토큰 할당량에서
미리 확보합니다. 그래서 모델 최대값(Claude 5.x는 128K)을 요청하면 실제 사용량이
한도에 닿기 훨씬 전에 동시 수집 호출이 스로틀링됩니다. `default_max_output_tokens`가
16384인 이유입니다. thinking 토큰도 이 값에 포함됩니다. 출력이 긴 프롬프트는 자기
한도에서 더 높은 하한을 계산하고, 그 하한이 우선합니다. 하한은 그 한도에서 나올 수
있는 가장 긴 답변에 추론 여유분 8192 토큰을 더한 값입니다. 그래프 추출과
gleaning(과 그 출력 수정기)은 `max_entities_per_chunk` +
`max_relationships_per_chunk`개 레코드에 레코드당 150 토큰(기본 23192), 문서 번역은
`max_chunk_size` 문자당 1.35 토큰(18992), 주장 추출은 그 두 배(29792)를 잡습니다.
커뮤니티 리포트는 기본 상한 안에 들어갑니다. 이 한도들을 올리면 하한도 함께
올라갑니다.

### 컨텍스트 예산

`search.token_manager.max_context_tokens`(`30000`)는 답변 프롬프트가 받는 검색
컨텍스트입니다. upstream LightRAG의 전체 컨텍스트 예산과 같습니다(MS GraphRAG는
12000). 1M 토큰 창에서 도출한 예산(약 785K)은 한도에 닿는 일이 없어서 유형별 예산도
아무것도 잘라 내지 못합니다. 이 값은 항상 `models.roles.answer_generation`가 출력
예약분(`aws.bedrock.default_max_output_tokens`)과 함께 받을 수 있는 크기로 줄이며, 줄일
때는 경고를 남깁니다. `null`이면 그 모델의 컨텍스트 창에서 출력 예약분과
`context_window_headroom_ratio`를 뺀 값으로 도출합니다. 1M이 베타인 모델에서는
`aws.bedrock.enable_1m_context`를 켜면 예산이 넓어집니다. 섹션 유형별
비율(`search.token_manager.type_budgets`)과 `local` 검색의 슬롯
할당(`search.local_search.type_quota`)은 레퍼런스에 있습니다.

### 가드레일

`aws.bedrock.guardrail.identifier`를 지정하면 질의마다 두 텍스트를 `ApplyGuardrail`
API로 검사합니다. 가드레일은 `aws.bedrock.region_name`에 있어야 합니다.

- **검사 대상.** 대화 메모리, 후속 질문 재작성, 라우팅, 검색보다 먼저 한 번 검사하는
  사용자 원본 질의(`check_input`)와 최종 답변(`check_output`. 스트리밍 답변은 검사 뒤
  한 덩어리로 전달됨)입니다. 이 두 텍스트만 평가하고 과금합니다.
- **결과.** 차단된 질의에는 가드레일의 차단 메시지로 답하고, 검색도 메모리 저장도 하지
  않으며 `RAGOutput.metadata.blocked`가 `true`입니다. 마스킹된 질의(PII 익명화)는
  마스킹된 텍스트로 계속 처리하므로 검색, 모델, 메모리 어디에도 원문이 가지 않습니다.
  답변이 차단되거나 마스킹되면 가드레일이 돌려준 텍스트로 바꿉니다. 모든 개입은
  `RAGOutput.metadata.guardrail`(source, action, policies)에 남고, 누적 횟수와 함께
  WARNING 로그와 `kg_rag.guardrail` 스팬으로도 기록됩니다. 가드레일 API가 실패하면
  질의가 실패합니다(`processing.ignore_errors`면 표준 오류 답변). 검사하지 않은 채로
  답하지는 않습니다.
- **모델 호출에 붙이지 않는 이유.** 질의 안의 모델 호출(재작성, 라우팅, 키워드 추출,
  global/DRIFT map 단계, 답변 생성)에는 가드레일을 붙이지 않습니다. 모델 호출에 붙인
  가드레일은 프롬프트 전체를 평가합니다. Converse API에서는 검색된 코퍼스 텍스트와
  개발자 지시문까지 사용자 입력처럼 `PROMPT_ATTACK` 필터가 판정해 잘못 차단하고, 비용도
  컨텍스트 크기와 호출 수에 비례해 늘어납니다. InvokeModel API에서는 사용자 입력을 입력
  태그로 감싸지 않으면 `PROMPT_ATTACK` 필터가 동작하지 않습니다
  ([Bedrock Guardrails: 프롬프트 공격](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-prompt-attack.html)).
  컨텍스트 근거(contextual grounding) 검사는 하지 않습니다. 근거 자료와 질의를
  한정자(qualifier)가 붙은 콘텐츠로 보내야 하는데 답변 검사는 이를 보내지 않습니다.
- **수집과 평가.** 수집 모델, 프롬프트 튜너, 평가용 판정 모델은 기본적으로
  가드레일 없이 실행합니다(`apply_to: "query"`). `NAME`을 익명화하는 PII 가드레일은
  추출된 엔티티 이름을 `{NAME}` 같은 자리 표시자로 바꿔 서로 다른 사람을 한 노드로
  합치고, `PROMPT_ATTACK` 필터는 지시문처럼 보이는 코퍼스 텍스트를 차단할 수 있기
  때문입니다. `apply_to: "all"`은 이 호출에도 가드레일을 붙이므로 추출에 써도 안전한
  정책일 때만 쓰세요. 질의 경로에서도 `NAME` 익명화는 문제가 됩니다. 인물에 관한 질문은
  검색에 쓸 이름을 잃고, 답변에는 `{NAME}`이 나옵니다. 그래서 `iac/`가 만드는 기본
  가드레일은 이메일, 전화번호, 카드 번호만 익명화하고 `NAME`은 익명화하지 않습니다.
- **사용자 코드.** `setup_chain`으로 체인을 만들거나 질의가 아닌 작업에서 `get_model`을
  호출하는 코드는 `model_purpose=ModelPurpose.INGESTION`(또는 `EVALUATION`)을 넘겨야
  합니다. 표시하지 않은 호출은 `QUERY`로 간주하므로 질의 시점의 일시적 오류 재시도가
  적용되고, 가드레일은 붙지 않습니다.

### S3 캐시 암호화

`aws.s3.encryption.encryption_type: BUCKET_DEFAULT`는 암호화 헤더를 보내지 않으므로
버킷의 기본 암호화가 적용됩니다(CDK 스택에서 `use_cmk=true`이면 고객 관리형 KMS 키이며,
`AES256`을 강제하면 이 키를 우회합니다). 기본 암호화가 SSE-KMS인 버킷을 재사용하면 쓰기
주체에 해당 키의 `kms:GenerateDataKey`와 `kms:Decrypt` 권한이 필요합니다.

### 로그에 남는 내용

`INFO` 이상에서 패키지는 사용자 텍스트와 코퍼스 텍스트 대신
길이, 개수, ID, 짧은 해시만 기록합니다(예: `query: len=42 sha=1a2b3c4d`). 해시는
항상 같은 값이 나오므로 질의 내용을 드러내지 않고도 같은 질의를 여러 레코드에서
따라갈 수 있습니다. 질의 원문, 다시 쓴 질의(DRIFT와 번역), 엔티티 이름, 모델 원본
출력은 `DEBUG`에서만 기록하며, 예외 메시지에는 모델 출력을 넣지 않습니다. `DEBUG`
로그(`logging.level: DEBUG`, `UKR__LOGGING__LEVEL=DEBUG`, `--verbose`)에는 사용자 데이터와
코퍼스 데이터가 들어 있다고 보세요. 로그를 공유 저장소로 보내는 환경에서는 켜지
마세요. `logging.library_levels`에서 `DEBUG`로 둔 라이브러리(예: `botocore`)는 요청
본문도 기록할 수 있습니다.

`logging.log_to_file`이 `false`가 아니면 CLI는 로그를 `logging.log_file_path`
(`logs/log.txt`, 날짜별로 `log_YYYYMMDD.txt`)에도 씁니다. 라이브러리로 가져오면
패키지는 핸들러나 파일을 설정하지 않습니다. 호스트 애플리케이션의 로깅 설정이 적용되고,
그 포매터는 일반 메시지를 보여 주며, 구조화된 필드(`logger`, `level`, `pipeline_id`,
`trace_id` 등)는 로그 레코드의 속성으로 들어갑니다.

### 사용자 정의 프롬프트

모든 프롬프트에는 `custom_prompts.*_system` / `*_human` 재정의 키가 있습니다(기본값
`null` = `unified_kg_rag/domain/prompts/`의 내장 프롬프트). 필요한 것만 재정의하고
나머지는 `null`로 두세요([§9](#9-도메인-적응)). 재정의는 설정을 읽을 때 검사합니다. 알
수 없는 `{variables}`와 빠진 데이터 변수(`{input_text}` 등)는 오류이며, 중괄호를
글자 그대로 쓰려면 `{{` / `}}`로 씁니다.

### 환경 변수로 덮어쓰기

모든 설정 키는 `UKR__` 뒤에 키 경로를 붙인 환경 변수로 지정할 수 있습니다. 경로의
마디 사이에는 밑줄 두 개를 쓰고 대소문자는 가리지 않습니다.
`UKR__AWS__NEPTUNE__ENDPOINT`는 `aws.neptune.endpoint`를,
`UKR__SEARCH__RERANKING__TOP_K`는 `search.reranking.top_k`를 지정하며, 자유 형식
매핑 다음 마디는 그 매핑의 키를 가리킵니다
(`UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`). 환경 변수는 설정 파일과 내장
기본값보다 우선합니다. 값은 필드 타입으로 변환하고(불리언은 `true`/`1`/`yes`/`on`을
참으로 봄. 목록이나 매핑 설정은 JSON, 예:
`UKR__SEARCH__AUTO_ROUTABLE_STRATEGIES='["local", "mix"]'`) 파일에 같은 값을 쓴
경우처럼 필드 제약 조건으로 검증합니다(`UKR__LOGGING__LEVEL`은 대소문자를 가리지
않음). 값이 잘못되었거나 어떤 설정 키도 가리키지 않는 `UKR__` 변수는 사용법 오류이며,
CLI가 해당 변수를 밝힌 한 줄을 출력한 뒤 종료 코드 2로 끝납니다
(`--allow-unknown-keys`를 주면 알 수 없는 변수는 WARNING만 남깁니다).

AWS SDK의 `AWS_PROFILE`과 `AWS_REGION`은 설정 파일이 `aws.profile_name`과
`aws.region_name`을 지정하지 않았을 때만 그 값을 채웁니다. SDK와 마찬가지로 명시한
설정이 환경 변수보다 우선하므로, 셸이 `AWS_REGION`을 내보내도(SSO 도우미와
CloudShell이 그렇게 합니다) 파일에 적은 리전은 바뀌지 않습니다. `AWS_DEFAULT_REGION`은
읽지 않으며, `UKR__AWS__REGION_NAME`은 다른 `UKR__` 변수처럼 파일보다 우선합니다.

배포에서 주로 지정하는 변수는 다음과 같습니다.

| 환경 변수 | 지정하는 키 | 참고 |
|---|---|---|
| `AWS_PROFILE` / `AWS_REGION` | `aws.profile_name` / `aws.region_name` | 파일이 지정하지 않았을 때만. `aws.bedrock.region_name`이 `null`이면 `AWS_REGION`이 Bedrock 호출 리전도 정합니다. |
| `UKR__AWS__BEDROCK__REGION_NAME` | `aws.bedrock.region_name` | |
| `UKR__AWS__BEDROCK__GUARDRAIL__IDENTIFIER` / `__VERSION` | `aws.bedrock.guardrail.identifier` / `version` | |
| `UKR__AWS__NEPTUNE__ENDPOINT` | `aws.neptune.endpoint` | |
| `UKR__AWS__OPENSEARCH__ENDPOINT` | `aws.opensearch.endpoint` | |
| `UKR__AWS__OPENSEARCH__USERNAME` / `__PASSWORD` | `aws.opensearch.username` / `password` | `aws.opensearch.use_iam: false`일 때의 기본 인증이며, 비밀번호는 로그에 가려서 표시합니다. |
| `UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS` | `indexing.opensearch.index_settings.number_of_replicas` | 이후에 만드는 인덱스에 적용됩니다. |
| `UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO` | `indexing.opensearch.language_analyzers.ko` | 이후에 만드는 인덱스에 적용됩니다. |
| `UKR__AWS__S3__BUCKET_NAME` | `aws.s3.bucket_name` | |
| `UKR__AWS__DYNAMODB__TABLE_NAME` / `__CREATE_TABLE_IF_MISSING` | `aws.dynamodb.table_name` / `create_table_if_missing` | |
| `UKR__PROCESSING__DOCUMENT_PARSING__SOURCE_SCOPE` | `processing.document_parsing.source_scope` | 컨테이너 엔트리포인트가 동기화한 S3 URI로 지정합니다. |
| `UKR__OBSERVABILITY__ENABLED` / `__BACKEND` | `observability.enabled` / `backend` | |
| `UKR__LOGGING__LEVEL` / `__LOG_FORMAT` | `logging.level` / `log_format` | |

`run-ingestion`은 `UKR_SOURCE_DIRECTORY`와 `UKR_PIPELINE_ID`(설정 키가 아니므로
밑줄 하나)도 `--source-directory`와 `--pipeline-id`의 기본값으로 읽습니다. 플래그를
직접 지정하면 플래그가 우선합니다. `UKR_METRICS_ENVIRONMENT`는
`--metrics-sink cloudwatch`의 `Environment` 차원입니다. CDK compute 스택이 주입하는
변수는 [`iac/README.md`](../iac/README.md#2-build-and-push-the-app-image)에 있습니다.
그 환경에서도 증분 인덱싱을 쓰려면 설정 파일에 `aws.dynamodb.enabled: true`가 있어야
합니다(이미지의 설정 파일에는 들어 있습니다).

CLI는 python-dotenv로 `.env` 파일도 읽습니다. 현재 작업 디렉터리의 `./.env`만 읽고
상위 디렉터리의 파일은 찾지 않습니다(그러지 않으면 `AWS_PROFILE`을 설정한 `~/.env`가
모르는 사이에 계정을 바꿀 수 있음). 환경에 이미 설정된 변수가 `.env`보다 우선합니다.
패키지를 가져오는 것만으로는 `.env`를 읽지 않으며 `os.environ`을 바꾸지 않습니다.
Python 코드에서 `.env`가 필요하면 직접 읽으세요(예: `get_config()` 전에
`dotenv.load_dotenv()`).

> **LangSmith 추적은 내용을 업로드합니다.** `LANGSMITH_TRACING=true`(또는 구형
> 변수 `LANGCHAIN_TRACING_V2=true`)가 설정되어 있으면 LangChain은 추적한 실행을 모두
> LangSmith로 보냅니다. 여기에는 프롬프트, 검색된 문서 본문, 모델 출력이 들어
> 있습니다. CLI는 추적이 켜져 있으면 시작할 때 WARNING을 남기지만, 의도한 설정일 수
> 있으므로 끄지는 않습니다. 기밀 코퍼스로 실행하기 전에는 이 변수를 해제하세요. 이
> 경고는 `setup_logging`이 남기므로, 이 함수를 호출하지 않는 라이브러리 코드에서는
> 나오지 않습니다.

### 관측성

추적은 기본적으로 꺼져 있습니다. `observability.enabled: true`로 켜면 수집 실행과
질의가 각각 하나의 OpenTelemetry 트레이스가 되며, OTLP를 받는 백엔드(Jaeger, Grafana
Tempo, OpenTelemetry Collector 등)에서 볼 수 있습니다. `otel` extra가 필요합니다.

```bash
uv sync --extra otel          # 또는: pip install -e '.[otel]'
```

extra 없이 추적을 켜면 시작할 때 extra 이름을 알려 주는 `ConfigurationError`로
실패합니다. 추적이 꺼져 있으면 OpenTelemetry 관련 모듈을 전혀 불러오지 않습니다.
`observability.backend`는 스팬을 보낼 곳을 정합니다(`none`: 호스트 애플리케이션이나
`opentelemetry-instrument`가 설정하는 프로세스 전역 트레이서 공급자, `otlp`: 패키지가
설치하는 OTLP 익스포터, `langfuse` / `phoenix`: 그 익스포터를 해당 벤더로 보냄, 아래
참고). `capture_content`, `redact_patterns`, `sample_rate`, `service_name`,
`export_timeout_seconds`는 [레퍼런스](./configuration.md#observability)에 있습니다.

익스포터는 설정 파일이 아니라 OpenTelemetry 표준 환경 변수로 설정하므로 수집기 API
키가 설정 파일에 들어가지 않습니다. `OTEL_EXPORTER_OTLP_ENDPOINT`(기본값
`http://localhost:4318`), `OTEL_EXPORTER_OTLP_HEADERS`,
`OTEL_EXPORTER_OTLP_TIMEOUT`, `OTEL_EXPORTER_OTLP_PROTOCOL`(기본값
`http/protobuf`, 4317 포트는 `grpc`)을 씁니다. 자격 증명이 담긴 헤더는 실행 시점에
Secrets Manager나 태스크 시크릿에서 주입하세요.

**Langfuse와 Phoenix.** 둘 다 OTLP를 받으므로 이 백엔드에는 `otel` extra만 있으면
되고 벤더 SDK는 필요 없습니다. 설정은 역시 환경 변수에서만 읽으며, 표준
`OTEL_EXPORTER_OTLP_*` 변수가 벤더 변수보다 우선합니다.

| 백엔드 | 엔드포인트 | 자격 증명 |
|---|---|---|
| `langfuse` | `LANGFUSE_HOST`(예: `http://localhost:3000`, 익스포터는 `<host>/api/public/otel/v1/traces`로 보냄). 기본 호스트가 없으므로 필수입니다. 대신 `OTEL_EXPORTER_OTLP_ENDPOINT`를 쓰면 `<host>/api/public/otel`을 지정합니다. HTTP만 지원하며 gRPC면 시작할 때 실패합니다. | `LANGFUSE_PUBLIC_KEY`와 `LANGFUSE_SECRET_KEY`(필수). Basic `Authorization` 헤더와 `x-langfuse-ingestion-version: 4`로 보냅니다. |
| `phoenix` | `PHOENIX_COLLECTOR_ENDPOINT`(기본값 `http://localhost:6006`, `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`이면 `:4317`). | 서버가 요구할 때 `PHOENIX_API_KEY`를 Bearer `authorization` 헤더로 보냅니다. |

스팬은 `gen_ai.*`, `kg_rag.*` 속성을 그대로 두고 내보낼 때 벤더 속성이 더해집니다. Langfuse에는 `langfuse.observation.type`(모델 호출은
`generation`, `embedding`, 검색과 재순위화는 `retriever`, 나머지는 `span`)과
`capture_content`일 때 `langfuse.observation.input`/`output`이, Phoenix
(OpenInference)에는 `openinference.span.kind`, `llm.model_name`,
`llm.provider`, `llm.token_count.*`와 `capture_content`일 때 `input.value`/`output.value`가
붙습니다.

**세션과 사용자.** 질의의 모든 스팬에 `session.id`(`RAGInput.conversation_id`)와,
지정한 경우 `user.id`(`RAGInput.user_id`)가 붙습니다. Langfuse와 Phoenix 모두 이
이름으로 트레이스를 묶습니다. `user_id`는 스팬에만 기록되고 받은 그대로 내보내므로
이메일 주소가 아닌 불투명한 ID를 넘기세요.

**트레이스 구조**

- 수집: `kg_rag.ingestion.run`(실행이 실패하면 상태 `ERROR`) →
  `kg_rag.ingestion.stage`(단계마다 하나, 입력·출력 수와 상태, 버린 단위 수 같은
  단계의 수치 지표를 `kg_rag.stage.<metric>`으로, 단계가 예외로 끝나면 `error.type`과
  예외) → `kg_rag.batch`(항목 수와 실패 수) → 모델
  호출마다 `chat <model>` 스팬, 임베딩 호출마다 `embeddings <model>` 스팬
- 질의: `kg_rag.query`(`capture_content`일 때 마스킹한 질의를 `input.value`로,
  `stream` / `astream`이 아니면 마스킹한 답변을 `output.value`로 기록하며 Langfuse와
  Phoenix는 이 값을 트레이스 목록에 표시) → `kg_rag.strategy`(질의의 [재순위화 상태](#재순위화-상태)인
  `kg_rag.rerank_status`) → `kg_rag.retrieve`(결과 유형별 적중 수와
  최대 50개의 소스 ID) → `kg_rag.fusion`(입력·출력 수, 해당 단계의
  `kg_rag.rerank_status`) → `kg_rag.rerank`
  (`kg_rag.rerank.status` = `ok`, `empty`, `error`와 문서 수) →
  `kg_rag.context_build`(사용한 토큰과 토큰 예산) → 답변 생성의 `chat <model>` 스팬.
  가드레일이 있으면 검사한 질의와 답변마다 `kg_rag.query` 아래에 `kg_rag.guardrail`
  스팬이 붙고(`aws.bedrock.guardrail.id`, `.source` = `input` 또는 `output`,
  `.action` = `none`, `masked`, `blocked`, `.intervened`, `.policies`), 질의 스팬에는
  `kg_rag.blocked`가 기록됩니다.
- 평가(`run-eval`): 답변 생성 전체가 `kg_rag.batch` 루트 하나(항목 수와 실패 수만
  기록) → 질문마다 `kg_rag.query` 하나이며, 질문별 입력과 출력은 이 자식 스팬에 있음.
  이어서 평가기마다 `kg_rag.evaluate` 루트 하나(`kg_rag.evaluator`, `kg_rag.items`) →
  심사 모델의 `chat <model>` 스팬(`langchain`, `ragas`)
- 모델 스팬은 OpenTelemetry GenAI 시맨틱 규약을 따릅니다(임베딩, Bedrock 재순위화
  스팬과 같이 스팬 종류는 `CLIENT`). `gen_ai.request.model`,
  `gen_ai.request.max_tokens` / `temperature`,
  `gen_ai.usage.input_tokens` / `output_tokens`, Bedrock이
  프롬프트 캐시 사용량을 알려 주면 `gen_ai.usage.cache_read.input_tokens` /
  `cache_creation.input_tokens`, 종료 사유(Bedrock, OpenAI 호환 서버, Ollama),
  호출이 실패하면 `error.type`, `aws.bedrock.guardrail.intervened`를 기록합니다. 비용은 기록하지 않습니다.

스팬이 활성 상태일 때 남긴 로그 레코드에는 그 스팬의 `trace_id`와 `span_id`가
들어가므로, 트레이스에서 로그를 찾거나 로그에서 트레이스를 찾을 수 있습니다.

**개인정보와 전송 제한.** `capture_content: true`가 아니면 스팬에 질의 본문, 문단,
모델 출력이 들어가지 않습니다. 무엇을 내보내는지, 자격 증명을 어떻게 다루는지,
수집기에 연결할 수 없을 때 어떤 비용이 드는지는 운영 런북의
[전송 제한, 프라이빗 VPC, 개인정보](./operations.ko.md#전송-제한-프라이빗-vpc-개인정보)에 있습니다.

알려진 제한: `stream` / `astream`의 스트리밍 답변은 별도 트레이스로 기록됩니다(검색과
컨텍스트 구성은 `kg_rag.query` 트레이스에 있음). 비동기 배치 경로는 청크가 아니라
배치마다 스팬을 하나 기록합니다. 스트리밍 답변의 모델 스팬은 답변 단계 실행
메타데이터의 `session_id` / `user_id`에서 `session.id` / `user.id`를 가져오므로
같은 세션으로 묶입니다. `ainvoke` 도중 취소된 모델 호출(예: `asyncio.wait_for` 시간 초과)은
LangChain이 콜백을 호출하지 않으므로 모델 스팬이 남지 않습니다. 호출자가 스트림을 중간에
그만 읽으면 그 스팬은 오류 없이 정상 종료됩니다. `otlp`, `langfuse`, `phoenix`에서는
프로세스가 처음 설치한 백엔드가 OpenTelemetry 전역 트레이서 공급자(tracer provider)도
되는데, 전역 공급자는 한 번만 설정할 수 있습니다. 나중에 다른 설정으로 다시 구성하면 패키지의
스팬은 새 내보내기 도구로 가지만, 다른 라이브러리가 전역 OpenTelemetry API로 남기는
스팬은 버려집니다. CloudWatch로 보내려면 태스크 옆에 ADOT나 OpenTelemetry Collector를 두고
`backend: otlp`를 쓰세요. CDK 스택에서는 `-c enable_cloudwatch_tracing=true`가 그
수집기를 추가하고 백엔드를 설정합니다(절차, Transaction Search, Bedrock 호출 로깅
보조 수단은 [운영 런북](operations.ko.md#adot-수집기-사이드카를-거친-cloudwatch)에
있습니다). 로컬 Langfuse와 Phoenix 절차는 [운영 런북](operations.ko.md#관측성)에
있습니다.

---

## 3. 수집 (`run-ingestion`)

수집은 문서 디렉터리를 OpenSearch와 Neptune에 인덱싱된 지식 그래프로 바꿉니다.

### CLI 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--source-directory` | `$UKR_SOURCE_DIRECTORY` | 소스 문서 디렉터리입니다. 실행에 필요하며, 플래그를 생략하면 환경 변수 `UKR_SOURCE_DIRECTORY`를 씁니다. |
| `--output-directory` | 없음(내보내지 않음) | 파싱한 문서를 확인용 JSON으로 내보낼 위치(소스 디렉터리는 지정 불가) |
| `--cache-directory` | `cache` | 파이프라인 캐시와 중간 결과 |
| `--force-rebuild` | off | 캐시된 단계 출력을 무시하고 모든 단계를 다시 계산. 새 출력으로 캐시를 다시 씀 |
| `--s3-sync` | off | 캐시를 S3에 동기화(`--s3-bucket-name` 필요) |
| `--s3-bucket-name` | — | 캐시 동기화용 S3 버킷 |
| `--s3-prefix` | `pipeline-runs` | 캐시 파일의 S3 키 접두사 |
| `--pipeline-id` | `$UKR_PIPELINE_ID` | 재개하거나 검사할 기존 실행입니다. 플래그를 생략하면 환경 변수 `UKR_PIPELINE_ID`를 씁니다. 이 ID가 실행의 캐시 디렉터리와 S3 접두사 이름이 되므로 영문 소문자, 숫자, 하이픈, 밑줄만 쓸 수 있습니다. |
| `--resume-from-stage` | — | 재개할 단계 이름(대소문자 무관, `--pipeline-id` 필요). 없는 이름이면 사용법 오류(종료 코드 2) |
| `--verify-metadata` | off | 파이프라인 메타데이터 무결성 확인(`--pipeline-id` 필요). 손상되었거나 실행을 찾지 못하면 0이 아닌 코드로 종료 |
| `--yes`, `-y` | off | 확인 질문(무결성 검사 실패 후 계속 진행)에 모두 예로 답함. stdin이 터미널이 아니면 질문할 수 없으므로 `--yes` 없이는 실행이 실패 |
| `--continue-on-error` | off | 단계에서 오류가 나도 계속 진행. 산출물을 만드는 단계(번역부터 커뮤니티 탐지까지)가 비정상 종료하면 그 실행의 모든 문서가 FAILED로 기록되어 다음 증분 실행이 다시 처리 |
| `--enabled-stages` | 전체 | 실행할 단계 목록(쉼표로 구분) |
| `--metrics-sink` | `none` | `none` 또는 `cloudwatch`(CloudWatch EMF — Embedded Metric Format — 지표를 stdout에 출력. `UKR_METRICS_ENVIRONMENT`가 있으면 `Environment` 차원을 붙이고(CDK 태스크는 `env_name`을 설정), 없으면 차원 없음. `pipeline_id`는 차원이 아니라 로그 속성으로 기록하므로 실행마다 지표 시계열이 늘지 않음). 각 단계의 지표는 그 단계가 끝날 때, 그 단계를 실행한 실행만 내보내므로 재개한 실행이 복원한 단계의 수치를 다시 보내지 않음 |
| `--retire-source-scope SCOPE` | — | 증분 실행에서 이전 소스 범위 `SCOPE`(이번 실행의 인덱스 접미사)의 레지스트리 레코드를 모두 삭제된 문서로 처리해 그 문서에만 속한 내용을 지웁니다. 여러 번 지정할 수 있고 `indexing.retire_source_scopes`에 더해집니다. `SCOPE`가 실행 자신의 범위이면 실패합니다(§5 삭제 범위 참고). |
| `--output-format` | `text` | `text`, 또는 표준 출력에 JSON 결과 하나를 쓰는 `json`([종료 코드와 JSON 출력](#cli-종료-코드와-json-출력) 참고) |
| `--config-path` | — | `config.yaml` 경로. 파일이 없으면 사용법 오류(종료 코드 2) |
| `--allow-unknown-keys` | off | 알 수 없는 설정 키(와 `UKR__` 변수)에 실패하지 않고 WARNING만 남긴 채 계속합니다([설정을 읽는 순서](#설정을-읽는-순서) 참고) |
| `--backend`, `--llm-provider` 등 | — | 공급자 선택. [공급자 플래그](#공급자-플래그) 참고 |

### 파이프라인 단계 12개

실행 순서(`DataIngestionPipeline.STAGE_CLASSES`)입니다. `--enabled-stages` /
`--resume-from-stage`에는 단계 **이름**(대소문자 무관)을 씁니다.

1. **`document_parsing`** — 형식별로 텍스트를 추출합니다(`.pdf`, `.txt`, `.csv`,
   `.json`. `unstructured` extra가 있으면 `.md`/`.html`도).
2. **`document_loading`** — 파싱한 문서로 이번 실행의 코퍼스를 구성합니다(MinHash
   중복 제거, 증분 필터). `document_parsing`을 끄면 소스 디렉터리에서 미리 파싱된
   `Document` `.json` 파일을 읽습니다.
3. **`text_chunking`** — 문서를 텍스트 단위로 나눕니다(`processing.chunking`).
4. **`translation`** — 선택 단계입니다. `target_language`로 번역합니다(원문 언어와
   대상 언어가 같고 추가 대상 언어가 없으면 아무 일도 하지 않음).
5. **`graph_extraction`** — LLM이 청크마다 엔티티와 관계를 추출합니다.
6. **`gleaning`** — 선택 단계입니다. 반복 정제 패스를 실행합니다(`processing.gleaning`).
7. **`graph_resolution`** — 중복 엔티티와 관계를 병합합니다(이름 변형, `processing.fuzzy_entity_resolution`이면 비슷한 이름도).
8. **`claim_extraction`** — 선택 단계입니다. 사실 주장을 추출합니다(기본은 꺼짐).
9. **`claim_resolution`** — 선택 단계입니다. 추출한 주장의 중복을 제거합니다.
10. **`graph_analysis`** — 그래프를 만들고 그 통계(`graph.analysis.statistics`)를 기록합니다.
11. **`community_detection`** — Leiden 군집화와 LLM 커뮤니티 리포트 생성을 실행합니다.
12. **`indexing`** — 모든 결과를 OpenSearch와 Neptune에 씁니다(켜져 있으면 DynamoDB
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

**재개와 강제 재구축:** `--force-rebuild` 없이 다시 실행하면 완료된 단계는 캐시를
쓰고 건너뜁니다. 특정 실행을 재개하려면 `--pipeline-id`를 넘기세요.
`--resume-from-stage`를 함께 쓰면 지정한 단계부터 다시 실행하고, 쓰지 않으면
처음으로 실패했거나 완료되지 않은 단계부터 재개합니다(`--enabled-stages`를 쓰면 그
목록 안에서 처음인 단계). 지난 실행 이후 켠 단계(예: gleaning, claim extraction)도
완료되지 않은 단계로 봅니다. 증분 인덱싱에서 자동 재개의 추출 설정이 델타를 계산할 때와
다르면(단계를 켜거나 모델·프롬프트를 바꾼 경우) `document_loading`부터 다시 시작하므로,
델타와 커밋된 문서에 기록하는 지문이 새 설정을 따릅니다(`document_parsing`은 캐시를
그대로 씁니다). 재개하기 전에 CLI는 완료된 단계의 출력이 현재 입력 기준으로
캐시에 남아 있는지 확인합니다. 남아 있지 않으면(코퍼스나 설정이 바뀐 경우) 해당 단계를
보여 주고 계속할지 묻고, 재개하면서 그 단계를 다시 계산합니다. 터미널이 없으면 `--yes`를
넘겨야 계속합니다. `--force-rebuild`는 캐시된 출력을 무시하고 모든 단계를 다시 계산한 뒤
새 출력을 캐시에 쓰므로, 이후 실행은 그 출력부터 재개합니다. `--pipeline-id` 없이 실행하면
새 ID를 생성합니다.

**S3 동기화**는 단계 캐시를 `s3://<bucket>/<prefix>/...`에 보관하므로, 새
프로세스(예: 새 Fargate 태스크)가 끝난 단계를 다시 계산하지 않고 재개할 수 있습니다.
실행은 단계가 끝날 때마다 바뀐 파일을 업로드하므로, 태스크가 중지되거나 죽어도 끝난 단계는
모두 남습니다. SIGTERM(ECS StopTask, Step Functions 시간 초과)을 받으면 `run-ingestion`은
실행 중인 단계를 멈추고 체크포인트를 업로드한 뒤 143으로 종료하며, 같은 pipeline id로 다시
실행하면 그 단계부터 시작합니다. 단계 사이의 업로드가 실패하면 로그를 남기고 다음 업로드에서
다시 시도합니다. 시작 시 다운로드나 마지막 업로드가 실패하거나 일부만 되면 `CacheSyncError`로
실행이 실패하고 `run-ingestion`은 0이 아닌 코드로 종료합니다. 따라서 체크포인트를 잃은
실행이 실패로 표시됩니다. 새 pipeline id에서 원격
캐시가 비어 있는 것은 오류가 아닙니다. S3에 동기화한 체크포인트는 `cache.ttl_seconds`로
만료되지 않고 버킷의 수명 주기 규칙이 만료시킵니다(CDK 배포에서는 30일). 임베딩은 따로
`indexing.opensearch.persist_embedding_cache: true`를 설정하면 바뀌지 않은 텍스트를
실행마다 다시 임베딩하지 않습니다.

**LLM 응답 캐시**(`cache.llm_responses`, 기본값 켜짐): 수집 단계의 모델 호출(추출, gleaning,
주장, 번역, 설명 요약, 커뮤니티 리포트, LLM 청킹, 출력 수정) 응답을 실행의 캐시 디렉터리
`llm-cache/`에 두고 단계 캐시와 함께 동기화합니다. 같은 pipeline id로 다시 실행하면 이미 보낸
프롬프트는 여기서 답하므로, 중지되거나 시간 초과되거나 비정상 종료된 단계는 하지 않은 호출만
다시 합니다. 키는 렌더링한 프롬프트와 모델 및 생성 설정(모델 ID, 최대 토큰, temperature,
추론, 가드레일)이므로 프롬프트, 모델, 설정이 바뀌면 모델을 다시 호출합니다.
알아 둘 점은 다음과 같습니다.

- 캐시한 응답은 실행마다 한 번까지만, 그리고 이전 실행의 응답만 씁니다. 실행 안에서 다시
  시도하면(응답을 파싱하지 못한 경우) 모델을 다시 호출하고 새 응답이 이전 응답을 대체합니다.
  출력 토큰 한도에서 잘린 응답은 보관하지 않습니다.
- temperature가 0보다 크면 다시 실행할 때 새로 샘플링하지 않고 이전 결과를 재사용합니다.
  `--force-rebuild`는 캐시에서 답하지 않습니다(새 응답은 저장합니다).
- 새 pipeline id는 빈 캐시로 시작합니다. 질의와 평가 호출은 캐시하지 않습니다.
- 비용: 모델 호출마다 응답 하나(수 KB)만큼 커집니다. `cache.llm_responses.flush_every`개
  응답 단위의 세그먼트를 채워질 때와 단계 체크포인트마다 쓰고(`--s3-sync`면 업로드),
  예고 없이 종료된 태스크도 잃는 것은 세그먼트 하나 이하입니다. 캐시는 실행을 시작할 때
  메모리로 읽습니다.
- 중지하면 실행 중인 모델 호출의 재시도도 멈추므로, 진행 중인 호출이 끝나면 태스크가
  종료됩니다.

### 인덱스 접미사

코퍼스가 쓰는 OpenSearch 인덱스와 Neptune 레이블의 이름은 모두 `<prefix>-<suffix>`이고,
`indexing.additional_suffix`를 지정하면 `<prefix>-<suffix>-<additional_suffix>`입니다.
이 문서에서는 `<suffix>`를 **인덱스 접미사**라고 부릅니다. 같은 값이 수집 쪽과 질의
쪽에서 다른 이름으로 불립니다.

- 수집은 `processing.document_parsing.index_value`의 값으로 씁니다(`run-ingestion`에는
  이 값을 지정하는 플래그가 없음).
- 질의는 `run-rag`/`run-eval`의 `--suffix`, Python에서는 `RAGInput.suffix`로 고릅니다.

둘 다 기본값은 `default`입니다. 테넌트나 코퍼스 버전마다 인덱스 접미사를 하나씩 쓰고
설정 파일도 하나씩 두세요. `indexing.additional_suffix`는 양쪽에서 같아야 합니다.
증분 인덱싱과 삭제는 인덱스 접미사 단위로 이루어집니다(§5).

접미사는 이름을 나눌 뿐 접근을 막지 않습니다. 질의할 수 있는 호출자는 어떤 접미사든
지정할 수 있습니다. 멀티 테넌트 서비스는 접미사를 요청에서 받지 말고 인증된 호출자를
기준으로 서버에서 정해야 합니다. 인프라로 강제하는 방법은 [운영 런북: 질의 권한과
테넌트 격리](./operations.ko.md#질의-권한과-테넌트-격리)를 참고하세요.

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

`source_language == target_language`이고 `additional_target_languages`가 비어
있거나 null이면 번역 단계는 `is_noop`으로 건너뜁니다. 영어만 있는 코퍼스는
`enabled: true`여도 번역 LLM 비용이 **전혀** 들지 않습니다. 언어별 OpenSearch
분석기는 `indexing.opensearch.language_analyzers`(예: `ko: nori`)에서 지정하며,
목록에 없는 언어는 `default_analyzer`를 씁니다. 원문 청크의 `text`는
`source_language`의 분석기로, 번역문은 `target_language`의 분석기로 분석합니다.
따라서 번역을 하지 않더라도 `source_language`는 코퍼스 언어로 지정하세요. 한국어
코퍼스를 `source_language: en`으로 두면 `english` 분석기가 적용되어, 조사가 붙은
어절("홍길동으로부터")을 토큰 하나로 남깁니다. `source_language`를 바꾸면 매핑이
달라지므로 텍스트 단위를 다시 인덱싱해야 합니다. 파싱한 텍스트와 질의는 NFC로
정규화하므로, 자모가 분해된 한글(NFD. macOS 파일 시스템과 많은 PDF 텍스트 레이어가
쓰는 형태)도 조합형과 일치합니다.

분석기는 클러스터에 있어야 합니다. 매핑이 클러스터에 없는 분석기를 가리키면 인덱스를
만들지 못하고 인덱싱 단계가 `the cluster has no analyzer '<name>'` 오류로 실패합니다.
`nori`는 `analysis-nori` 플러그인이 제공하며, [로컬 OpenSearch
컨테이너](#컨테이너로-띄운-로컬-저장소)에는 설치되어 있지만 Amazon OpenSearch Service
도메인에는 선택 사항인 Nori 패키지를 연결해야만 생깁니다. 그래서 [CDK
배포](../iac/README.md)는 `UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`로 `ko`를 OpenSearch에 내장된
`cjk` 분석기(겹치는 문자 바이그램. 임베디드 엔진이 `nori`를 처리하는 방식과 비슷함)로
매핑하며, 이 값이 이미지의 설정 파일보다 우선합니다. 도메인에서 한국어 형태소 분석을
쓰려면 Nori 패키지를 연결한 뒤(콘솔 **Packages** 또는 `aws opensearch
associate-package`. 블루/그린 배포가 일어남) `-c opensearch_korean_analyzer=nori`로
배포하고, `indexing.reset: true`로 한 번 다시 실행해 한국어 인덱스를 다시 만드세요.

번역에 실패한 텍스트 단위는 원문을 유지하고 원문 언어로 추출합니다. 대상 언어 하나의
호출 전체가 실패하면 `processing.ignore_errors`가 `true`가 아닌 한 단계가 실패합니다.
어느 경우든 단계는 실패한 단위(`failed_units`)를 보고하고, 증분 실행은 해당 문서를
실패로 기록해 다음 실행에서 다시 번역합니다(§5).

---

## 4. 질의 (`run-rag`)

### CLI 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--query`, `-q` | — | 단일 질의. 이 플래그와 `--interactive` 중 정확히 하나가 필요합니다. |
| `--interactive`, `-i` | off | 대화형 채팅(메모리 자동 활성화, 항상 `rag` 모드) |
| `--mode` | `rag` | `rag`(답변 생성까지) 또는 `search`(검색만. 단일 `--query`에서만) |
| `--conversation-id` | — | 대화 이름 지정. 메모리는 프로세스가 실행되는 동안만 유지([대화 메모리](#대화형-모드와-대화-메모리) 참고) |
| `--use-memory` | off | 프로세스 안에 두는 대화 메모리 활성화(대화형 모드에서는 자동) |
| `--suffix` | —(`default`) | 질의할 [인덱스 접미사](#인덱스-접미사). 코퍼스의 `processing.document_parsing.index_value`와 같아야 합니다. |
| `--enable-thinking` | off | 모델의 단계별 추론 활성화. 항상 추론하는 모델(Claude 4.7+ / 5.x, OpenAI GPT)에는 효과가 없으며, 이런 모델의 추론 강도는 `aws.bedrock.default_effort` / `fast_effort`로 정합니다 |
| `--search-strategy` | `auto` | `auto`, `drift`, `global`, `local`, `simple`, `mix`, `hybrid`, `naive` |
| `--search-type` | `hybrid` | `hybrid`, `lexical`, `vector`. 어휘 검색은 질문의 앞 150단어(한글·한자·가나는 한 글자를 한 단어로 셈)만 검색하고, 오타 허용은 10단어 이하 질문에만 적용합니다. 벡터 검색은 질문 전체를 임베딩합니다 |
| `--top-k` | `10` | 최대 검색 결과 수 |
| `--retrieval-multiplier` | `1` | 검색 깊이 배수 |
| `--disable-query-processing` | off | 번역과 엔티티 추출 건너뛰기 |
| `--filters` | — | `key:value` 속성 필터(공백으로 구분) |
| `--output-format` | `text` | `text`, 또는 표준 출력에 JSON 결과 하나를 쓰는 `json`([종료 코드와 JSON 출력](#cli-종료-코드와-json-출력) 참고) |
| `--verbose`, `-v` | off | 질의 처리 정보, 출처, 지표 표시 |
| `--config-path` | — | `config.yaml` 경로 |
| `--allow-unknown-keys` | off | 알 수 없는 설정 키(와 `UKR__` 변수)에 실패하지 않고 WARNING만 남긴 채 계속합니다([설정을 읽는 순서](#설정을-읽는-순서) 참고) |
| `--backend`, `--llm-provider` 등 | — | 공급자 선택. [공급자 플래그](#공급자-플래그) 참고 |

### 검색 전략 선택

**방법론 선택.** GraphRAG 전략은 코퍼스 전체를 *요약하고 주제별로 종합*하는 데
강합니다(`global`은 계층 한 단면의 커뮤니티 리포트 전체에 map을 적용함). LightRAG 전략은 더 빠르고 *이중 레벨
키워드* 검색을 중심으로 동작하므로 키워드 중심 조회와 저비용 기준선에 적합합니다. 두
방법론 모두 같은 하이브리드 스코어러(BM25 어휘 검색 + 벡터 의미 검색 + 그래프 순회 +
RRF + Bedrock 재순위화)를 거치며, 검색 알고리즘만 다릅니다.

**GraphRAG(커뮤니티 요약):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `simple` | 빠른 사실 조회, 단순한 질문 | OpenSearch 벡터·키워드 직접 검색이며 그래프 순회가 없습니다. 가장 빠릅니다. 주장 추출이 켜져 있으면 주장 인덱스도 포함합니다. |
| `local` | 특정 엔티티나 개념에 관한 자세한 질문 | 질의에서 엔티티를 추출 → Neptune 그래프 순회로 이웃과 관계를 찾음 → 벡터·키워드 결과와 결합합니다. 켜져 있으면 주장(covariate)을 넣습니다. |
| `global` | "주요 주제는 무엇인가" 같은 넓은 주제 질문 | MS GraphRAG global search입니다. 계층의 한 단면(`search.global_search.community_depth`, 기본값은 루트부터 2)에 속한 커뮤니티 리포트 전체를 토큰 크기 배치로 나눠 fast 모델로 점수 매긴 핵심 내용으로 만들고, 순위를 매긴 핵심 내용으로 답합니다. 선택 사항인 동적 선택은 대신 루트부터 계층을 따라 내려갑니다. 비용은 단면 크기에 비례합니다(리포트 `map_context_tokens`마다 map 호출 1회). 상위 수준 종합에 가장 적합합니다. |
| `drift` | 큰 그림과 거기서 몇 단계 떨어진 구체적 사실이 함께 필요한 여러 부분으로 된 질문 | MS GraphRAG DRIFT search입니다. 가상 답변(HyDE)과 가장 비슷한 커뮤니티 리포트로 중간 답변과 후속 질의를 만들고, 후속 질의마다 local search를 실행해 fast 모델로 답하며 새 후속 질의를 얻는 과정을 `n_depth` 라운드 반복합니다. 답변 모델이 순위를 매긴 중간 답변과 그 근거를 종합합니다. 질의당 LLM 호출이 전략 중 가장 많습니다(약 `primer_folds + k_followups * n_depth`회). |
| `auto` | 어떤 전략이 맞는지 모를 때, 일반 용도(기본값) | LLM 라우터(`models.roles.strategy_selection`)가 질의를 보고 `search.auto_routable_strategies`(기본 local, mix, global, drift) 중 가장 알맞은 전략을 고릅니다. |

**LightRAG(이중 레벨 키워드):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `mix` | 일반적인 LightRAG 사용, 그래프와 청크의 균형 | 저수준 키워드 → 엔티티 인덱스, 고수준 키워드 → 관계 인덱스, 1홉 연결 관계·끝점 엔티티 확장(Neptune 다중 홉 확장은 `search.lightrag_search.enable_graph_expansion`으로 선택)에 **더해** naive 벡터 청크 검색 결과를 섞습니다. |
| `hybrid` | 키워드 중심의 그래프 질문 | `mix`와 같지만 naive 청크 결과를 섞지 않습니다. |
| `naive` | 빠른 기준선, 비교 평가 | 그래프 없이 청크 검색만 하며 `search.lightrag_search.chunk_stream_top_k`개(기본 20개) 청크를 반환합니다. 다른 전략처럼 `--search-type`을 따르므로 기본값 `hybrid`는 벡터 검색에 BM25를 더합니다. upstream LightRAG의 순수 벡터 기준선은 `--search-type vector`로 재현합니다. |

> `mix`/`hybrid`를 쓰려면 관계 벡터 인덱스
> (`indexing.opensearch.relationships_index_prefix`, 수집 중 자동 생성)가 만들어져
> 있어야 합니다. 고수준 키워드 검색이 이 인덱스를 씁니다. 키워드가 나오지 않는 짧은
> 질의는 원래 질의를 저수준 키워드로 씁니다
> (`search.lightrag_search.raw_query_fallback_max_len`로 제한).

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
run-rag --query "..." --filters attr_category:research type:person --config-path config.yaml
```

### 재순위화 상태

모든 질의는 재순위화 결과를 `SearchResult`와 RAG 출력의 `metadata.rerank`에
남깁니다(따라서 `--output-format json`에도 나옵니다). `status`, `reason`, `stages`로
구성되며, `stages`에는 재순위화 단계마다 `status`, `scope`(`all`, 또는 재순위화한
결과 유형. 예: `local`, `drift`, `mix`, `hybrid`의 `text`), `reason`이 있습니다.
`--verbose`는 실행 지표에 `Rerank:` 한 줄로 출력합니다.

| 상태 | 의미 |
|---|---|
| `applied` | 재순위화 모델이 후보 순서를 바꿨습니다. |
| `disabled` | 재순위화가 꺼져 있습니다: `search.reranking.enabled: false` 또는 `providers.rerank.kind: none`. |
| `unavailable` | 재순위화가 켜져 있지만 재순위화 모델을 만들 수 없습니다(초기화 실패, 또는 크로스 인코더의 `[local]` extra 미설치). 초기화 실패는 ERROR로 기록됩니다. |
| `skipped` | 재순위화할 대상이 없습니다: 후보 없음(전략이 일찍 반환한 경우 포함), 재순위화 대상 유형의 결과 없음, 질의 텍스트 없음. |
| `failed` | 재순위화 호출이 예외를 냈거나(`bedrock:Rerank` 거부, 모델이 없는 리전, 연결할 수 없는 엔드포인트) 쓸 수 있는 결과를 돌려주지 않았습니다. 융합 순서를 유지했습니다. |

`failed`와 `unavailable`의 `reason`은 메시지 없이 실패 원인만 밝힙니다. Bedrock
오류는 AWS 오류 코드(`AccessDeniedException`, `ValidationException`), 그 밖에는
예외 유형입니다.

기본 제공 전략은 질의마다 한 번만 재순위화하므로 `stages`에는 항목이 하나입니다.
같은 기록에 여러 번 융합하는 호출자는 융합마다 단계가 하나씩 생기고, 최상위
`status`는 가장 나쁜 단계의 상태입니다. 순서는 `failed` > `unavailable` >
`applied` > `skipped` > `disabled`입니다. 각 결과의
`metadata.reranked` 플래그는 그대로 모델이 순서를 바꾼 결과를 표시합니다. Bedrock
재순위화 모델을 만들 때는 AWS를 호출하지 않으므로, 권한이나 리전 문제는
`unavailable`이 아니라 첫 질의의 `failed`로 나타납니다. 이런 질의를 융합 순서로
답하는 대신 `RerankModelError`로 실패시키려면 `search.reranking.required: true`를
설정하세요. 이 상태는 `kg_rag.fusion`과 `kg_rag.strategy` 스팬의
`kg_rag.rerank_status` 속성에도 기록됩니다.

필터는 OpenSearch에서는 `term`/`terms`/`range` 절로, Neptune에서는 `has` 단계로
변환됩니다. 저장소마다 해당 인덱서가 쓰는 필드를 받습니다.

| 저장소 | 필터 가능한 필드 |
|---|---|
| 텍스트 단위 | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.<path>` |
| 엔티티 | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `attr_<key>`, `attributes.<path>` |
| 관계 | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids` |
| 주장 | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `description`, `source_text`, `text_unit_ids` |
| 커뮤니티 리포트 | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `level`, `depth`, `max_cut_depth`, `parent_community_id`, `child_community_ids`, `attr_<key>`, `attributes.<path>` |
| Neptune 엔티티 정점 | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `attr_<key>`(있는 경우) |
| Neptune 커뮤니티 정점 | `id`, `name`, `level`, `parent`, `size`, `period`, `children` |

OpenSearch에서 `attr_<key>`는 문서 속성입니다. 문서의 `filters` 메타데이터 항목
`<key>`가 `attr_<key>`로 인덱싱됩니다. Neptune 엔티티 정점에서 `attr_<key>`는
엔티티 자체에서 추출한 속성입니다(예: `attr_role`). 여러 문서에서 추출한 엔티티나 관계는
`filters` 키마다 그 문서들의 값을 모두(목록으로) 가지므로 그중 어느 값으로 필터링해도
일치합니다. 엔티티와 주장의 `type`, 주장의 `status`는 소문자로 저장되고 `term`은
정확히 일치해야 하므로 소문자 값으로 필터링하세요(`type:PERSON`이 아니라
`type:person`). 정확히 필터링하려면 정확 일치
필드(keyword 또는 숫자)를 쓰세요. `description`처럼 분석되는 텍스트 필드에 `term`
필터를 걸면 소문자 토큰 하나와만 일치합니다. 범위 필터는 Python API에서
`{"gte": ..., "lte": ...}` 형태로 넘깁니다.

필터는 해당 필드를 선언한 저장소에만 적용됩니다. 예를 들어 `type:person`은 엔티티,
주장, Neptune 엔티티를 좁히고 텍스트 단위는 필터링하지 않습니다. 두 저장소는
`attr_<key>`를 다르게 다룹니다.

- OpenSearch는 텍스트 단위, 엔티티, 커뮤니티 리포트에 `attr_<key>`와
  `attributes.<path>`를 엄격하게 적용합니다. 속성이 없는 문서는 제외되므로, 대개 문서
  속성이 없는 커뮤니티 리포트는 속성 필터를 건 질의에서 빠집니다. 의도한
  동작(fail-closed)입니다. 속성 필터는 보증할 수 없는 내용을 반환하지 않습니다. 벡터
  검색과 하이브리드 검색은 그 속성과 일치하는 문서가 하나도 없는 인덱스를 건너뜁니다
  (OpenSearch 2.13은 인덱스에 매핑되지 않은 필드로 kNN을 필터링하면 거부합니다).
- Neptune은 엔티티 정점에 `attr_<key>`를 "있는 경우" 기준으로 적용합니다. 속성이
  일치하거나 그 속성이 없는 정점은 통과합니다. 따라서 `attr_category` 같은 문서 속성
  필터는 그래프 확장에 영향을 주지 않고, `attr_role:buyer` 같은 엔티티 속성 필터는
  역할이 다른 엔티티를 제외합니다. 그 밖의 키는 두 저장소 모두 엄격하게 적용합니다.

**필터는 접근 제어 경계가 아닙니다.** 필터는 검색이 순위를 매길 대상을 좁힐 뿐
데이터를 숨기지 않습니다. 키를 선언하지 않은 저장소는 내용을 필터링 없이 반환합니다.
관계와 주장에는 `attr_<key>`나 `document_ids`가 없고, Neptune 커뮤니티 정점은 자기
필드만 선언하며, Neptune은 `attr_<key>` 속성이 없는 엔티티 정점을 남깁니다. 그래프
확장과 커뮤니티 리포트로 필터가 제외할 문서의 내용이 들어올 수도 있습니다. 권한이 다른
테넌트나 사용자를 필터로 분리하지 마세요. 대신 각각에 별도 [인덱스 접미사](#인덱스-접미사)를
주세요. 서로 다른 `--suffix`(와 `document_parsing.index_value`, 쓴다면
`indexing.additional_suffix`)를 쓰면 OpenSearch 인덱스와 Neptune 레이블이 분리되며,
질의 접미사는 인덱스 대상을 넓힐 수 없도록 검증합니다. 호출자가 어떤 접미사에 질의할 수
있는지는 호출하는 애플리케이션에서 정하세요.

선택한 전략이 읽는 저장소 중 어디에도 선언되지 않은 필터 키(예: `category`,
`entity_type`)는 `InvalidFilterError`를 일으키며, 오류 메시지에 필터 가능한 키 목록이
나옵니다. 스키마는 `unified_kg_rag/adapters/storage/filter_schema.py`에 정의되어
있습니다.

### 대화형 모드와 대화 메모리

```bash
run-rag --interactive --config-path config.yaml
# or name the session (its id appears in logs and JSON output):
run-rag --interactive --conversation-id my-session --config-path config.yaml
```

대화형 모드에서는 메모리가 자동으로 켜집니다. 세션 안에서 쓸 수 있는 명령은 다음과
같습니다.

- `help` — 명령 목록 표시
- `new` — 새 대화 시작(새 ID)
- `set-filter key:value` — 필터 추가 또는 변경
- `clear-filters` — 필터 모두 제거
- `show-config` — 현재 설정 표시
- `quit` / `exit` — 종료

대화 메모리는 프로세스 안에만 있습니다(`MemoryManager`가 메모리에 들고 있으며 디스크나
저장소에 쓰지 않음). 한 대화형 세션의 턴 사이, 또는 한 `GraphRAGChain`을 같은
`conversation_id`, `use_memory=True`, 같은 `suffix`와 `user_id`로 여러 번 호출할 때
이어집니다. 대화는 인덱스 접미사와 사용자에 속하므로 다른 테넌트나 사용자의 턴은 섞이지
않습니다. 메모리는 체인마다 따로 있으며, 여러 체인이 공유하게 하려면 `MemoryManager(config)`
하나를 `memory_manager=`로 넘기세요.
`run-rag --query`를 따로 실행하면 공유하지 않습니다. 실행마다 기록 없이 시작하므로 단일
질의에서 `--use-memory`와 `--conversation-id`는 효과가 없습니다(`--query`와 함께 쓰면
`run-rag`가 WARNING 로그를 남김). CLI에서 여러 턴을 이어 가려면 `--interactive`를
쓰세요. 메모리 한도는 `memory` 설정 섹션에 있습니다.

검색 전에는 후속 질문을 대화 이력으로 독립된 질문으로 다시 씁니다(이력이 있는 턴마다
`models.roles.query_rewrite` 호출 한 번, 프롬프트 재정의 키 `query_rewrite`). 그래서
"그 회사의 공급업체는?" 같은 후속 질문도 앞선 턴의 엔티티 이름이 들어간 질문으로
라우팅, 검색, 답변됩니다. 재작성은 최근 `memory.max_messages_per_conversation`개
메시지를 읽고, 앞선 답변은 각각 앞 1,000자까지만 씁니다. `use_memory`로 이력을 찾으면
`enable_query_processing`을 꺼도 재작성은 실행됩니다. 이 플래그는 번역과 엔티티·키워드
추출을 끄는 설정이고, 재작성 없이는 후속 질문으로 쓸 만한 결과를 찾지 못하기
때문입니다. 메모리는 이 외에 턴마다 LLM을 추가로 호출하지 않습니다.

---

## 5. 증분 인덱싱

증분(delta) 인덱싱은 전체를 다시 구축하지 않고, 마지막 실행 이후 **새로 생겼거나
바뀐** 문서만 다시 인덱싱해 운영 중인 그래프에 병합합니다.

### 켜기

```yaml
aws:
  dynamodb:
    enabled: true
    table_name: "unified-kg-rag-on-aws-doc-status-v2"
    create_table_if_missing: true
```

이 설정을 켜면 `run-ingestion`은 실행할 때마다 DynamoDB 문서 상태 레지스트리와
**콘텐츠 해시**로 코퍼스를 비교합니다. 로컬 모드에서는 레지스트리가 SQLite 파일입니다
(`providers.doc_status.kind: sqlite`, `providers.doc_status.enabled: true`로 켬.
[로컬 모드](#로컬-모드-aws-없이) 참고).

### 작업 흐름

- **문서 추가:** 새 파일을 소스 디렉터리에 넣고 `run-ingestion`을 다시 실행합니다.
  새 파일만 파싱·추출·인덱싱하고, 그 엔티티와 관계는 기존 그래프에 병합합니다(멱등
  `upsert_*`).
- **문서 수정:** 파일을 고치고 다시 실행합니다. 콘텐츠 해시가 바뀌므로 변경된
  문서로 처리해, 이전 버전의 산출물을 제거하고 새 버전을 다시 인덱싱합니다.
  모델에는 바뀐 청크만 보냅니다. 추출 저장소
  (`indexing.opensearch.extraction_records_index_prefix`)에 본문과 추출 설정이 같은
  레코드가 있는 청크는 이 문서든 다른 문서든 저장된 레코드를 씁니다(`--force-rebuild`는
  모두 다시 추출하고 저장소를 새로 씁니다).
- **문서 삭제:** 소스 디렉터리에서 파일을 지우고 다시 실행합니다. 그 문서에만 있는
  **전용** 산출물(그 문서의 청크에서만 서술된 엔티티, 관계, 주장. 레지스트리에는
  문서별 청크가 있고, 무엇이 그 청크를 인용하는지는 저장소에 묻습니다)을 삭제하고,
  남은 문서와 공유하는 산출물은 그 문서의 청크만 빼고 유지합니다. 커뮤니티와
  리포트는 인용하는 청크가 하나도 남지 않으면 지웁니다.
- **추출 설정 변경**(튜닝한 `custom_prompts`, `entity_types`, 추출·gleaning·청킹 모델,
  청크 크기, 번역, 해소 설정): 각 레코드는 그 문서를 추출할 때 쓴 설정의 지문을
  저장합니다. 다음 실행은 지문이 다른 미변경 문서를 모두 다시 추출하고 그 수와
  바뀐 설정을 WARNING 로그로 남깁니다. 레지스트리는 코퍼스마다 설정 자체도 보관해
  필드별로 비교합니다. 새 릴리스가 추가한 설정은 기본값이면 변경으로 보지 않으므로
  업그레이드만으로 코퍼스를 다시 추출하지 않습니다. 내장 프롬프트 템플릿(코드)을 고친 것은
  감지하지 못하므로 고친 뒤에는 초기화하세요. 다시 추출하는 문서라도 청크의
  레코드를 정하는 설정(모델, 추출·gleaning·주장 설정, 대상 언어, 내장 템플릿을 포함한
  프롬프트 템플릿)이 그대로이면(예: 해소나 청킹 설정만 바꾼 경우) 청크는 저장된
  레코드를 씁니다.
- **임베딩 모델이나 차원 변경**(`models.embedding.model_id` /
  `dimension`): 증분 실행은 새 문서와 변경된 문서만 임베딩하므로 한 인덱스에 두
  모델의 벡터가 섞입니다. OpenSearch 인덱스마다 만들 때 쓴 모델과 차원을 기록하고,
  인덱싱 단계는 그 값이 이번 실행과 다르면 아무것도 쓰기 전에 멈춥니다.
  `indexing.reset: true`로 한 번 다시 실행해 코퍼스 전체를 다시 임베딩하세요. 임베딩
  설정을 기록하지 않은 인덱스(이전 릴리스가 만든 인덱스)도 같은 방식으로 거부하므로
  한 번 초기화하세요.
  질의도 같은 기준으로 확인합니다. 인덱스를 만들 때와 다른 임베딩 모델이나 차원으로
  `run-rag`·`run-eval`을 실행하면 빈 검색 결과로 답하지 않고 `ConfigurationError`로
  실패합니다.
- **코퍼스 언어나 번역 언어 변경**(`processing.translation`의 `source_language`,
  `target_language`, `additional_target_languages`, 또는
  `indexing.opensearch.language_analyzers`): 운영 중인 OpenSearch 인덱스의 텍스트
  필드는 만들 때의 분석기를 그대로 쓰고, 새로 번역하는 언어의 필드는 keyword로
  매핑됩니다. 인덱싱 단계는 인덱스의 텍스트 필드가 이번 실행의 언어와 맞지 않으면
  아무것도 쓰기 전에 멈춥니다. `indexing.reset: true`로 한 번 다시 실행하세요. 언어를
  뺄 때는 멈추지 않습니다.
- **레지스트리를 끄고 접미사 인덱싱**(`backend: local` 프리셋과 플래그만 쓴 실행은
  레지스트리가 꺼져 있습니다): 이런 실행은 문서를 기록하지 않고 인덱스를 교체하므로
  레지스트리가 인덱스의 문서를 더는 설명하지 못합니다. 인덱스마다 레지스트리가
  추적하는지 기록하고, 그 접미사의 다음 증분 실행은 아무것도 쓰기 전에 멈춥니다.
  레지스트리를 켜고 `indexing.reset: true`로 한 번 다시 실행하거나, 그 접미사는
  레지스트리를 끈 채로 쓰세요.
- **`indexing.reset: true`로 재구축**: 실행은 접미사를 서비스에서 내리지 않고 코퍼스
  전체를 인덱싱합니다. 먼저 지우는 것은 없습니다. 벡터 인덱스는 새 인덱스에 쓰고 하나씩
  다 만들어지면 별칭을 옮기며, 그래프는 제자리에서 다시 쓰되 재구축이 쓰는 모든 정점과
  엣지에 태그를 붙이고, 그 뒤 재구축이 쓰지 않은 것을 지웁니다. 그동안 질의는 이전
  내용을 계속 읽습니다(지우기 전까지 그래프에는 이전 항목과 다시 구축한 항목이 섞임).
  레지스트리는 재구축이 코퍼스를 기록할 때까지 레코드를 유지하고, 그 뒤 네임스페이스의
  나머지 레코드를 지웁니다. 실패한 reset은 아무것도 지우지 않으므로 다시 실행해도
  안전합니다.

### 삭제 범위

실행의 **범위**는 인덱스 접미사(`document_parsing.index_value`와
`indexing.additional_suffix`)와 코퍼스 소스(`document_parsing.source_scope`.
기본값은 해석된 소스 디렉터리이며, 컨테이너 엔트리포인트는 동기화 원본인 S3 URI로
설정)입니다. 문서의 레지스트리 키는 이 범위와 소스 디렉터리 기준 상대 경로이며,
실행은 자기 범위에 기록된 문서만 비교하고 삭제 대상으로 봅니다. 따라서 다음과 같습니다.

- 다른 테넌트(다른 `index_value`)의 실행은 두 코퍼스를 같은 로컬 디렉터리에 두더라도
  이 테넌트의 문서를 삭제하지 않습니다.
- 하위 폴더를 소스 디렉터리로 지정해 따로 실행해도 코퍼스의 나머지는 삭제되지
  않습니다(그 파일들은 별도 문서로 등록되므로, 같은 파일을 두 루트에서 한 접미사로
  인덱싱하지 마세요).
- 소스가 다른 두 코퍼스를 한 접미사로 인덱싱하면, 둘 다 같은 상대 경로의 파일
  (각자의 `contract.txt`)을 가져도 서로 다른 문서입니다. 각 실행은 자기 문서만
  갱신하거나 삭제합니다.
- 파싱이나 로딩에 실패한 파일은 `failed`로 보고되며, 다음 실행에서 다시 읽을 때까지
  인덱싱된 내용을 유지합니다.

범위가 기본값(소스 디렉터리)인 코퍼스를 옮기면 범위가 바뀌고, 이전 범위의 레코드는
레지스트리에 남습니다. 어떤 실행도 다른 범위의 문서를 삭제하지 않으며, 실행은 옮긴
코퍼스와 다른 호스트가 같은 레지스트리에 인덱싱하는 두 번째 코퍼스를 구분할 수 없습니다
(이 호스트에 없는 디렉터리가 그 호스트에는 있을 수 있습니다). 남은 레코드는 옮기는 중에
지우거나 이름을 바꾼 파일의 내용까지 인덱스에 남겨 둡니다. 코퍼스를 옮길 때는 다음 중
하나를 따르세요.

- 옮기기 전에 `document_parsing.source_scope`를 고정된 이름으로 지정해 범위가 바뀌지
  않게 합니다.
- 새 위치에서 `--retire-source-scope <이전 소스 디렉터리>`(또는
  `indexing.retire_source_scopes`)로 한 번 실행합니다. 이 실행은 인덱스 접미사에서
  이전 범위의 레코드를 모두 삭제된 문서로 처리합니다. 그 문서에만 속한 텍스트 단위,
  엔티티, 관계를 지우고, 남는 문서와 공유하는 내용은 유지하며, 제거가 성공하면 레코드를
  삭제합니다. 새 위치의 문서는 같은 실행에서 새 문서로 인덱싱합니다. 범위는
  레지스트리에 저장된 형태로 지정합니다. 절대 디렉터리(끝의 `/`는 무시)나 이전
  `source_scope` 값이며, `s3://` URI도 됩니다. 실행 자신의 범위를 지정하면 레지스트리를
  읽기 전에 실패하고, 레코드가 없는 범위는 WARNING으로 알립니다.

범위가 기본값인 로컬 실행에 새 문서가 있으면, 같은 인덱스 접미사에 레코드가 있는 다른
로컬 소스 디렉터리를 WARNING 한 줄로 알리고 위 두 방법을 안내합니다(diff가 이미
읽은 레지스트리 레코드를 씁니다). 아무것도 삭제하지 않습니다. 그 디렉터리들이 별도
코퍼스라면 코퍼스마다 `source_scope`를 지정하면 경고가 나오지 않습니다.

레지스트리는 인덱스 네임스페이스(문서 키의 `<namespace>#` 접두사, DynamoDB 파티션
키)별로 나뉘므로 실행은 자기 네임스페이스의 레코드만 읽습니다. `doc_id`만으로 키를
잡던 이전 키 스키마의 테이블은 처음 쓸 때 거부됩니다. 새 테이블로 바꾸고
`indexing.reset: true`로 한 번 실행하세요.
[운영 런북: 증분 실행](./operations.ko.md#증분-실행)을 참고하세요.

### 실패한 문서

다음 네 경우에 문서가 `FAILED`로 기록됩니다.

- 텍스트 단위 하나라도 번역, 그래프 추출, gleaning, 주장 추출에 실패함
- 텍스트 단위를 포함한 기본 수준 커뮤니티의 리포트가 생성되지 않음
- `--continue-on-error` 실행에서 산출물을 만드는 단계(번역부터 커뮤니티 탐지까지)가
  비정상 종료함. 이때는 그 실행의 모든 문서가 `FAILED`로 기록됨
- 산출물(텍스트 단위, 엔티티, 관계, 주장, 커뮤니티, 리포트) 하나를 저장소에 쓰는 데 실패함

상위 수준 커뮤니티의 리포트 실패는 WARNING으로
남기고 단계의 `reports_failed` 지표에 세지만 문서를 `FAILED`로 기록하지는 않습니다. 상위
커뮤니티는 많은 문서에 걸쳐 있어, 리포트 하나 때문에 이를 모두 재시도하면 코퍼스 대부분을
다시 추출하게 됩니다. 리포트 실패와 쓰기 실패는 각각 커뮤니티 수와 산출물 유형별 쓰기 수의
`indexing.max_failure_rate`까지 허용하며, 이를 넘으면 해당 단계가 실패하고 아무것도
기록하지 않습니다. 다음 실행은 파일이 바뀌지 않았더라도 이 문서를 변경된 문서로 처리합니다.
실패한 실행이 쓴 내용을 제거하고 문서를 다시 처리합니다. 내용이 같은 채로
`indexing.max_document_failures`(기본 `3`)회 연속 실패한 문서는 더 재시도하지
않습니다. `FAILED` 상태와 인덱싱된 내용을 그대로 유지하고, 파일 이름이 담긴 WARNING과
함께 변경 없음으로 건너뜁니다. 파일을 수정하거나(내용이 바뀌면 횟수를 다시 셈) 한도를
올리면 다시 시도합니다.

### 격리된 문서

인덱싱하다 실행을 죽이는 문서(메모리 부족, 태스크 시간 초과)는 이후 모든 실행을 같은
방식으로 멈추게 합니다. 문서의 내용을 인덱싱하기 시작한 실행은 진행 중 레코드에 시도를
기록하며, 끝내지 못한 실행이 연속으로 `indexing.max_document_attempts`번(기본값 `3`)에
이르면 다음 실행이 그 문서를 격리합니다. 시도들이 쓴 것을 지우고, 문서를
`QUARANTINED`로 기록하고, 파일 이름을 담은 WARNING을 남긴 뒤 나머지를 인덱싱합니다
(`total_documents_quarantined` 지표가 그 수를 셉니다). 격리된 문서는 파일이나 추출
설정이 바뀌거나 한도를 올릴 때까지 건너뜁니다. 원인 문서를 따로 가려내지는 않습니다.
계속 죽는 델타의 모든 문서가 시도로 세어져 함께 격리됩니다.
[운영 런북: 격리된 문서](./operations.ko.md#격리된-문서)를 참고하세요.

### 동시 실행

레지스트리를 쓰면 실행은 첫 단계부터 끝날 때까지 인덱스 네임스페이스의 쓰기 임대를
유지하며, `indexing.lease_seconds`(기본값 300초)의 3분의 1마다 갱신합니다. 같은
네임스페이스의 두 번째 실행은 무엇이든 읽거나 쓰기 전에 보유자를 알려 주는
`NamespaceLeaseError`로 멈춥니다. 죽은 실행은 갱신을 멈추므로 그 임대는 늦어도
`lease_seconds` 뒤에 만료됩니다.
[운영 런북: 동시 실행](./operations.ko.md#동시-실행)을 참고하세요.

### 문서 크기 한도

DynamoDB 레지스트리는 문서마다 텍스트 단위(청크) ID를 항목 하나에 저장하며, 항목
크기 한도는 400KB(ID 약 10,000개)입니다. 수정한 문서의 진행 중 레코드는 이전 청크
ID와 새 청크 ID를 함께 담으므로, 수정하는 문서의 한도는 그 절반쯤입니다. 한도를 넘는
문서는 아무것도 쓰기 전에 파일 이름이 담긴 오류와 함께 인덱싱 단계를 실패시킵니다.
이런 문서는 더 작은 파일로 나누세요. 로컬 모드의 SQLite 레지스트리에는 이 한도가 없습니다.

### 실행 간 병합

기본값(`indexing.cross_run_merge: true`)에서 증분 실행은 upsert 전에 변경분을 기존
그래프 상태(설명, `text_unit_ids`, 빈도, 가중치)와 *합칩니다*. 그래서 바뀌지 않은
문서와 공유하는 엔티티는 그 문서들의 설명과 청크 계보(`mix`가 따라가는 정보)를
유지합니다. 증분 실행은 먼저 영향을 받는 엔티티와 관계를 그래프에서 읽어 오며, 병합한
설명이 요약 예산을 넘으면 다시 요약합니다. `false`로 두면 해당 필드를 변경분의 값으로
덮어쓰며, 그러면 엔티티는 바뀌지 않은 문서의 청크와의 계보를 잃습니다. 이 기능에는
읽어 오기를 지원하는 그래프 어댑터가 필요합니다(지원하지 않으면 덮어쓰기로
동작합니다). 주장(claim)도 같은 방식으로 합칩니다. 주장 ID는 문서나 표기가 아니라 주어와
목적어가 해석된 엔티티, 유형, 날짜로 정해집니다. 그래서 여러 문서가 진술한 주장은(한
문서에서는 "Acme Inc", 다른 문서에서는 "Acme"라고 써도) 하나의 주장으로서 각 문서의
설명, 원문, 텍스트 단위를 모두 유지하며 이 정보는
OpenSearch 주장 인덱스에서 읽어 옵니다(주장이 텍스트 단위를 저장하기 전에 색인된 주장은
이번 실행의 값에서 시작합니다). 문서가 바뀌거나 삭제되면, 그 문서가 다른 문서와 공유하던 엔티티, 관계, 주장은
`text_unit_ids`에서 그 문서의 청크를 잃고(빈도와 가중치도 따라 바뀜), 이름, 유형, 설명,
rank, 주장 문장은 남은 청크에서 다시 파생하므로 그 문서가 보탠 내용도 함께 사라집니다.
엔티티 id는 현재 이름을 따릅니다. 다른 표기가 이기면(이긴 표기를 가진 문서가 사라졌거나
새 문서가 득표를 뒤집은 경우) 엔티티는 전체 구축과 마찬가지로 그 이름의 id를 받고, 엣지와
주장도 따라갑니다. 저장된 커뮤니티와 리포트에는 전체 재구축 전까지 이전 id가 남습니다.
변경분의 이름이 병합될 수 있는 저장된 엔티티는 압축 이름 키로 찾으며,
`processing.fuzzy_entity_resolution: true`이면 저장된 이름을 모두 읽습니다. 제거된
청크만 가리키던 추출 저장소 레코드는 실행이 성공할 때마다 지웁니다. 저장된 항목을 읽어 오지 못하면 아무것도
덮어쓰지 않습니다. 병합 중 읽기가 실패하면 쓰기 전에 인덱싱 단계가 실패하고(문서는
기록되지 않은 채 다음에 재시도됨), 문서의 청크를 제거하던 중 읽기가 실패하면 제거
실패로 처리합니다(레지스트리 행은 재시도를 위해 남음).

---

## 6. 평가 (`run-eval`)

### CLI 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--data-path` | **필수** | 질문과 정답이 담긴 JSON 파일. 로컬 경로 또는 `s3://bucket/key` URI |
| `--output-directory` | `evaluation.outputs_directory` | 결과 저장 위치. 로컬 디렉터리 또는 `s3://bucket/prefix/` URI |
| `--suffix` | —(`default`) | 평가 대상 [인덱스 접미사](#인덱스-접미사) |
| `--enable-thinking` | off | `run-rag`와 같은 모델 추론(항상 추론하는 모델에는 효과 없음) |
| `--search-strategy` | `auto` | 각 질문에 답할 때 쓰는 전략입니다. 쉼표로 구분한 목록(`local,mix,global`)을 주면 첫 전략을 기준으로 [짝지은 비교](#전략-비교)를 실행합니다 |
| `--repeats` | `1` | 전략마다 실행할 횟수입니다. 비교는 질의마다 반복 실행의 평균을 내고 실행 간 편차를 보고합니다 |
| `--pairwise-judge` | 꺼짐 | 전략이 둘 이상이면 각 전략의 답변을 첫 전략의 답변과 LLM으로 맞비교합니다([전략 비교](#전략-비교) 참고) |
| `--search-type` | `hybrid` | 검색 방식 |
| `--top-k` | `10` | 최대 결과 수 |
| `--retrieval-multiplier` | `1` | 검색 깊이 |
| `--max-failure-rate` | `1.0` | 답변 생성에 실패한 질의의 비율, 또는 어떤 지표에서 시도한 값 중 실패한 값의 비율(`metric_outcomes`. 건너뛴 값은 세지 않음)이 이 값(0.0-1.0)을 넘으면 0이 아닌 코드로 종료합니다. 모든 질의가 실패했거나 어떤 지표의 모든 시도가 실패한 실행은 항상 0이 아닌 코드로 종료합니다 |
| `--verbose`, `-v` | off | 디버그 로깅 |
| `--output-format` | `text` | `text`, 또는 표준 출력에 JSON 결과 하나를 쓰는 `json`([종료 코드와 JSON 출력](#cli-종료-코드와-json-출력) 참고) |
| `--config-path` | — | `config.yaml` 경로 |
| `--allow-unknown-keys` | off | 알 수 없는 설정 키(와 `UKR__` 변수)에 실패하지 않고 WARNING만 남긴 채 계속합니다([설정을 읽는 순서](#설정을-읽는-순서) 참고) |
| `--backend`, `--llm-provider` 등 | — | 공급자 선택. [공급자 플래그](#공급자-플래그) 참고 |

### 평가기

`evaluation.enabled_evaluators`로 고릅니다. 켠 평가기를 만들 수 없거나(예: 판정
모델에 쓸 Bedrock 액세스가 없음) 평가기가 설정을 거부하면 실행이 멈춥니다.
`processing.ignore_errors: true`이면 그 평가기를 빼고 `run_manifest.dropped_evaluators`에
기록합니다. 채점은 모든 답변을 생성한 뒤에 시작하므로, 채점 중 평가기 오류(예: 판정
모델 호출 실패)가 나도 실행 결과를 버리지 않습니다. 그 질의의 지표를 실패로 기록하고
결과를 저장하며, 종료 코드는 `--max-failure-rate`로 정합니다. 판정 모델 응답에 쓸 수
있는 점수가 없을 때도 0이 아니라 실패로 기록합니다. 기본으로 다섯 개가 모두 켜져 있습니다. LLM을 쓰지 않는 결정적
평가기(`answer_match`, `retrieval`, `graph_aware`)는 비용이 없으며, 필요한 데이터셋
필드가 없는 질의는 건너뜁니다. 판정 모델 없이 실행하려면
`enabled_evaluators: [answer_match, retrieval, graph_aware]`로 설정하세요.

- **`langchain`** — LangChain 기반 텍스트 유사도(`langchain_metrics`:
  `correctness`, `partial_correctness`)입니다. `answer` 정답이 필요합니다.
- **`ragas`** — RAGAS 지표(`answer_correctness`, `answer_relevancy`,
  `context_precision`, `context_recall`, `faithfulness`)입니다. `context_precision`은
  컨텍스트마다 판정 모델을 한 번 호출하므로, 질의당 상위
  `evaluation.ragas_max_contexts`개(기본 20, `null`이면 제한 없음) 출처만 채점합니다.
  이 제한은 `max_context_tokens` 예산보다 먼저 적용되므로 지표는
  `context_precision@N`입니다. 제한이 없으면 출처를 100개 넘게 보고하는 전략(예:
  LightRAG `mix`)이 `ragas_timeout`에 걸립니다. `faithfulness`와 `context_recall`은
  `max_context_tokens` 안의 모든 출처를 보므로, 순위가 낮은 출처가 뒷받침하는 진술도
  근거 없음으로 처리되지 않습니다. 기본값(`evaluation.max_context_tokens: null`)에서는
  판정 모델이 답변 모델과 같은 예산(`search.token_manager.max_context_tokens`, 30000)을
  받으므로, 답변이 실제로 받은 컨텍스트를 기준으로 채점합니다. 이보다 작은 값을 직접
  지정하면 경고를 남깁니다. 각 리포트에는 판정 모델이 본 내용이
  기록됩니다(`judge_contexts` / `judge_context_tokens`,
  `context_precision_contexts` / `context_precision_context_tokens`). 이 설정은 답변
  모델이 본 내용에는 영향을 주지 않습니다.
- **`graph_aware`** — 결정적이며 **LLM을 쓰지 않는** 엔티티·관계 **커버리지 =
  재현율**입니다. 기대한 그래프 산출물 중 생성된 답변에 나오는 비율이며,
  `answer_contains`와 같은 정규화와 구문 매처를 씁니다(단어 단위 일치, 한국어 조사
  허용, 단어 하나인 CJK 텍스트는 부분 문자열 일치). `{"source": "A", "target": "B"}`나
  `"A -> B"`로 쓴 관계는 답변에 두 끝점이 모두 나오면 일치로 봅니다. 그 밖의 문자열은
  구문으로 나와야 합니다. 데이터셋에 `expected_entities` / `expected_relationships`가
  필요합니다. **정밀도와 F1은 의도적으로 출력하지 않습니다.** 자유 형식 답변에 나온
  엔티티를 빠짐없이 열거할 신뢰할 만한 방법이 없으므로, 정밀도와 F1을 보고해도 재현율
  신호에 이름만 바꿔 붙이는 셈이기 때문입니다. 답변 텍스트 기준 커버리지는 답변이
  길수록 올라가므로 `retrieved_entity_coverage` / `retrieved_relationship_coverage`도
  보고합니다. 같은 기대값을 답변 모델이 받은 엔티티 출처의 이름과 관계 출처의 두
  끝점에 맞춰 봅니다(쌍은 방향과 관계없이 관계 하나와 일치해야 하고, 그 밖의 문자열은
  끝점이나 설명에 나와야 함. `global`처럼 컨텍스트에 해당 유형의 출처가 없으면 건너뜀).
  `answer_length`(문자 수)도 함께 기록합니다. 전략 비교에는 검색 기준 커버리지를 쓰고,
  답변 텍스트 기준 커버리지는 길이와 함께 읽으세요.
- **`retrieval`** — 결정적이며 LLM을 쓰지 않습니다. 답변 모델이 본 출처에 정답 문서가
  들어 있었는지 봅니다. `reference_sources`를 기준으로 `hit_at_k`, `recall_at_k`(k =
  `evaluation.retrieval_k`, 기본 5), `mrr`을 계산합니다. 텍스트 단위 출처는 파일을
  직접 가리킵니다. 엔티티, 관계, 커뮤니티 리포트 출처는 계보(`text_unit_ids`)에 있는
  텍스트 단위의 파일로 귀속하며, 체인의 문서 저장소에서 접미사마다 한 번에 일괄
  조회합니다. 그 텍스트 단위를 가장 많이 담은 파일이 앞에 옵니다. 출처는 순위 순으로
  서로 다른 파일 목록 하나로 펼치고, **k는 출처가 아니라 파일을 셉니다.** 커뮤니티
  리포트는 커뮤니티의 모든 파일에 귀속되므로, 출처 하나로 세면 파일 50개 코퍼스에서
  무작위 리포트 5개가 hit@5 0.98을 받았습니다(무작위 텍스트 단위 5개는 0.19). 각
  리포트에는 `by_source_type`(섹션 유형별 출처만으로 계산한 같은 지표)도 있고, 요약의
  `grouped_statistics.source_type`에 모입니다. 그중 `text` 행은 답변 모델이 원문
  그대로 읽은 텍스트 단위만 채점하므로 전략 간 비교에 가장 공정합니다. 귀속할 수 없는
  출처는 파일을 내지 않으며, `attributable_fraction`(귀속 가능한 출처 / 보고된 출처. 전략별 값은
  `grouped_statistics`에도 있음)은 순위 지표가 컨텍스트의 얼마만큼을 다루는지
  보여 줍니다. 일치 비교는 NFKC 정규화 후 대소문자를 구분하지 않으며(그래서 자모가
  분해된 NFD 한글 파일 이름과 전각 문자도 일치), 이름으로 비교합니다. 파일
  확장자(`.` + 영문자·숫자 1-5자, 영문자 하나 이상 포함. `.v2` 같은 버전은 제외)가 없는
  참조는 출처의 확장자를 뺀 이름과도 일치하고, 확장자가 있는 참조는 확장자까지 같아야
  합니다. `docs/Terms.pdf`, `terms.pdf`, `terms`는 모두 출처 `Terms.pdf`와 일치하지만
  `terms.docx`는 일치하지 않고, `notes.v1`은 `notes.v2`와 일치하지 않습니다. 출처에는
  파일 이름만 있으므로 참조의 디렉터리는 무시합니다. `/`는 경로처럼 보이는 값(파일
  확장자, URI 스킴, 또는 앞에 `/`, `./`, `~/`가 있는 값)에서만 디렉터리 구분자로 보므로,
  `St. Louis Cardinals`나 `AC/DC` 같은 제목은 통째로 비교합니다. 질의에
  `reference_sources`가 없거나, 출처는 있지만 귀속할 수 있는 것이 없으면 순위 지표를
  건너뜁니다. 출처를 하나도 검색하지 못한 질의는 0점(놓침)입니다.
- **`answer_match`** — 결정적이며 LLM을 쓰지 않는 답변 점수입니다. `answer`와 선택
  항목인 `answer_aliases`(문자열 또는 목록. 최상위나 `metadata` 아래에 둘 수 있고, 둘 다
  있으면 `metadata` 쪽을 씀)를 기준으로 계산해 그중 최댓값을 씁니다.
  **`answer_contains`**(정답이나 별칭이 생성된 답변에 단어 단위 구문으로 나오면 1.0)가
  대표 결정적 지표입니다. 긴 RAG 답변이 짧은 정답 구간과 똑같은 경우는 드물기 때문에,
  이 지표가 exact match보다 정답 여부를 훨씬 잘 반영합니다. 한국어 조사는 단어 뒤에서
  허용합니다(정답 `서울 특별시`는 `서울 특별시는`과, `AWS`는 `AWS는`과 일치). 숫자만으로
  된 정답 뒤에는 단위와 조사도 허용하지만(`2024`는 `2024년에`와, `2`는 `2개월`과 일치),
  값을 바꾸는 크기 단위는 허용하지 않습니다(`3`은 `3억`과, `5`는 `5천만`과 일치하지
  않음). CJK 문자 옆의
  띄어쓰기는 무시합니다(`3억 원` = `3억원`, `가나다 상사` = `가나다상사`). 단어 하나인
  CJK 정답은 부분 문자열로 비교하지만, 더 긴 숫자나 영문 단어 안에서는 일치로 보지
  않습니다(`2년`은 `12년`과 일치하지 않음). 정답이 예/아니요 단어(`yes`/`no`/`nope`, 한국어
  `예`/`네`/`아니요`/`아니오`/`아닙니다`, 중국어 `是`/`对`/`否`/`不是`/`没有`, 일본어
  `はい`/`いいえ`)이면
  긴 답변에는 거의 항상 "no"(또는 `예외`의 `예`)가 들어 있으므로, HotpotQA처럼 질문에
  답하는 첫 예/아니요 단어의 극성으로 채점합니다. 단어 전체가 일치해야 하며, Markdown
  제목 줄, `/` 옆의 단어(`Yes/No: No`, `(네/아니요): 아니요`의 선택지), 뒤에 문장
  부호(`No:`, `**No**:`, `No (`, `No -`)나 줄 끝이 오지 않는 `no`와 `네`(`there is
  no clause`, `네 가지`. 뒤에 오는 강조 표시나 여는 따옴표는 다음 단어의 시작으로 보므로
  `no **cap**`, `no "cap"`도 해당), 이 경우와 더불어 뒤에 `:`나 `)`가 오는 `예`(`(예: …)`의
  "예를 들어"), 뒤에 숫자가 오는 `no`(`No. 5`)는 건너뜁니다. 나머지 예/아니요 단어는 위치와
  관계없이 답으로 읽습니다. 그래서 `네, 맞습니다`는 yes로 읽지만
  `네 맞습니다`는 답으로 보지 않습니다. SQuAD
  방식의 `exact_match`와 `token_f1`도 출력합니다. 텍스트는 NFKC로 정규화한 뒤 SQuAD
  v1.1 스크립트처럼 소문자로 바꾸고, 문장 부호를 지우고, 영어 관사를 빼고, 공백을 하나로
  합칩니다. 다만 유니코드 문장 부호도 지우고, 의미를 바꾸는 문장 부호는 남깁니다. 숫자
  앞의 음수 부호(`-5`는 `5`와 다름), 숫자 사이의 소수점과 쉼표(`1.5`는 `15`와 다름. 천
  단위 구분 기호는 지우므로 `1,000` = `1000`), 숫자 뒤의 퍼센트 기호(`3.5%`는 `3.5`나
  `35%`와 다름)와 단어 끝의 `+`/`#`(`C++`, `C#`)입니다. 두 단어나 숫자 사이의 하이픈은
  공백으로 보므로 `1990-1995`는 숫자 두 개이고 `5-year` = `5 year`입니다. 관사만으로 된 정답(`A`)은 그대로
  둡니다. 따라서 점수는 공식 스크립트로 계산한 리더보드 수치와 비슷하지만 직접 비교할
  수는 없습니다. Token F1은 공백으로 나누므로 중국어·일본어 텍스트에서는 exact match와
  같아지며, 한국어 조사(`서울은`) 때문에 두 지표 모두 실제보다 낮게 나옵니다.

### 평가 데이터 형식

객체의 JSON 배열입니다. `question`만 필수이고 나머지는 선택입니다.
`expected_entities` / `expected_relationships`는 `graph_aware` 평가기에*만*
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

항목별 `metadata`(예: `search_strategy`)는 그 질문에 한해 CLI 기본값보다 우선합니다.
코퍼스로 답할 수 없는 질문에는 `"metadata": {"answerable": false}`를 지정하세요. 이런
질문은 어떤 평가기로도 채점하지 않고, 체인이 답변을 거절했는지만 봅니다(아래
`abstention_statistics` 참고). `id`는 `query_id`로 써도 됩니다. 파일은 질의를
실행하기 전에 검증합니다. 데이터셋이 비어 있거나, 파일이 배열이 아니거나, `question`이
없거나, ID가 중복되거나, 필드 타입이 틀리거나, RAG 체인이 거부하는 `metadata` 값(예:
알 수 없는 `search_strategy`)이 있으면 항목 번호와 ID를 알리고 실행을 멈춥니다.

### 예시

```bash
run-eval --data-path my_eval_data.json --config-path config.yaml

run-eval --data-path my_eval_data.json --output-directory ./results --config-path config.yaml

run-eval --data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml

run-eval --data-path s3://my-bucket/eval/my_eval_data.json \
  --output-directory s3://my-bucket/eval/results/ --config-path config.yaml
```

두 경로 모두 `s3://` URI를 받으므로 `run-eval`도 CDK 스택에서 일회성 태스크로
실행할 수 있습니다([After deploy](../iac/README.md#5-evaluate-from-inside-the-vpc)).
`s3://` 데이터셋은 실행 전에 임시 파일로 내려받습니다. 출력 디렉터리가 `s3://`이면
결과 파일을 임시 디렉터리에 쓴 뒤 실행이 끝날 때 해당 접두사 아래로 올리며, 캐시
업로드와 같은 `aws.s3.encryption` SSE 설정을 적용합니다. 업로드가 하나라도 실패하면
`run-eval`은 0이 아닌 코드로 끝나고 로컬 파일을 남깁니다(경로를 출력함). S3 호출은
`aws.profile_name`과 `aws.region_name`을 사용합니다. 실행 주체에는 데이터셋에 대한
`s3:GetObject`와 출력 접두사에 대한 `s3:PutObject` 권한이 필요합니다.

읽을 수 없거나(파일 없음, 잘못된 JSON) 검증에 실패한 데이터셋(빈 배열, `question`
누락, 중복 id, 문자열 목록이 아닌 `expected_entities`나 문자열 또는 문자열 목록이
아닌 `answer_aliases`처럼 타입이 틀린 필드, RAG 체인이 거부하는 메타데이터)이면
`run-eval`은 질의를 하나도 실행하지 않고 `Error: invalid evaluation dataset:`과
항목 번호, 질의 id를 출력한 뒤 종료 코드 2로 끝납니다.

결과는 출력 디렉터리에 `evaluation_{results,reports,summary}_<timestamp>.json`으로
씁니다. 답변한 질의가 모두 같은 전략을 썼다면 파일 이름은
`..._<strategy>_<timestamp>.json`이 됩니다. `<timestamp>`는
`YYYYMMDD_HHMMSS_<16진수 8자리>`입니다. 임의 부분은 같은 초에 시작한 실행(여러
전략 병렬 실행, CI 매트릭스)이 서로의 파일을 덮어쓰지 않게 하므로, 스크립트는 이름을
만들어 내지 말고 이름이나 파일 시각으로 정렬해 최신 실행을 찾습니다. 요약에는 지표마다
평균/중앙값/표준편차/최솟값/최댓값/개수(`metric_statistics`)와
채점/실패/건너뜀 개수(`metric_outcomes`)가 있고, 다음 항목도 들어 있습니다.

- `grouped_statistics` — 같은 통계를 `search_strategy`(실제로 쓴 전략. `auto`에서는
  질의마다 다름), `category`, `difficulty`별로 나눈 값입니다. `source_type`에는 출처
  섹션 유형(`text`, `entity`, `relationship`, `community` 등)별 검색 지표가 있습니다.
- `abstention_statistics` — 체인이 답변 대신 고정된 컨텍스트 없음 응답("I could not
  find relevant information…")을 반환한 빈도입니다. `abstained`, `answered`,
  `abstention_rate`, 같은 값의 `per_strategy`가 있고, `answerable: false` 항목에는
  같은 형태의 `unanswerable` 블록이 있습니다. 고정 응답만 거절로 세며 체인은 검색
  결과가 없을 때만 이 응답을 반환합니다. 모델이 자기 말로 답변을 거절하면 감지하지
  않으므로, 데이터가 색인된 상태에서 `unanswerable`의 비율은 정확도가 아니라 올바른
  거절의 하한입니다. 답할 수 있는 항목에서 거절은 다른 답변과 똑같이 채점합니다(보통
  놓침). 각 결과에는 `abstained`가 기록됩니다.
- `rerank_status_counts` — 재순위화 상태별 질의 수입니다(예:
  `{"applied": 48, "failed": 2}`). 각 결과에는 `rerank_status`(`applied`,
  `disabled`, `unavailable`, `skipped`, `failed`, 답변 생성이 먼저 실패했으면 `null`)가
  기록됩니다([재순위화 상태](#재순위화-상태) 참고).
- `run_manifest` — CLI 인자(`--model-base-url` 같은 `*_url` 인자는 사용자 정보·쿼리·
  프래그먼트를 뺀 값), 모델 ID(답변 생성, 평가 판정 모델·임베딩, 그리고
  `judge_is_answer_model`. LLM 판정 모델은 자기 출력을 선호하므로 판정 모델이 답변
  모델과 같으면(기본값. 둘 다 `default` 티어) 판정 점수가 부풀려집니다. 시스템을
  비교할 때는 `models.roles.evaluation`를 다른 모델로 지정하세요), 패키지 버전,
  git 커밋(`git_sha`. 추적 파일이 커밋과 다르면 `git_dirty`도 기록. 패키지를 해당
  커밋을 추적하는 체크아웃에서 실행하지 않으면 둘 다 `null`), `retrieval_config_sha256`,
  `index`, `library_versions`(ragas, langchain*), 데이터셋(경로와 파일 sha256,
  질의 수, 파싱한 내용의 해시), UTC 타임스탬프가 있어 두 실행을 비교할 수 있습니다.
  `retrieval_config_sha256`은 답변과 점수에 영향을 주는 설정(`backend`, `search`,
  `custom_prompts`, `evaluation`, `providers`, `processing.translation`,
  `indexing.additional_suffix`, `indexing.opensearch`, `aws.bedrock`의 모델과 effort)만
  해시합니다. 리전, 엔드포인트, 자격 증명, 파일 경로, 가드레일 ID, 출력 디렉터리,
  재시도 설정은 빼므로 같은 평가는 어느 환경에서든 같은 해시가 됩니다. `index`에는
  질의한 접미사마다 그 벡터 인덱스를 마지막으로 쓴 수집 실행이 기록됩니다. 수집은
  인덱스 매핑의 `_meta`에 `pipeline_id`, `mode`(`full` 또는 `incremental`),
  `built_at`, `documents`(실행이 색인한 문서 수. 증분 실행은 새 문서와 바뀐 문서만 셈),
  `extraction_fingerprint`, `extraction_settings`(필드별 추출 설정), `package_version`을 기록합니다(최선 노력 방식. 매핑 수정
  권한이 없는 역할이면 경고만 남김). 기록하기 전에 만든 인덱스나 인덱스 메타데이터가
  없는 백엔드는 `null`입니다.
  `dropped_evaluators`도 기록합니다. `EvaluationManager.evaluate_dataset`이 이
  매니페스트를 만들므로 라이브러리 호출자도 얻을 수 있습니다(기록하려면
  `dataset_path=` / `cli_args=`를 넘기세요).

질의별 리포트에는 평가기마다 `metrics`가 있습니다. 서로 무관한 지표의 평균은
의미가 없으므로 종합 점수는 두지 않습니다. 각
결과에는 `retrieved_source_ids`도 기록됩니다. 보고된 출처마다 순위 순으로, 그 출처가
귀속된 파일 이름 목록입니다(귀속할 수 없으면 `[]`). 출처마다 섹션 유형인
`retrieved_source_types`와 모델이 받은 그래프 요소인 `retrieved_entities`,
`retrieved_relationships`도 기록됩니다.

### 전략 비교

`run-eval --search-strategy local,mix,global --repeats 3`은 같은 질의를 전략마다
`--repeats`번 답하고 채점합니다(지정한 전략이 항목별 `search_strategy`보다 우선).
실행마다 평소처럼 결과 파일을 저장하고, `evaluation_comparison_<timestamp>.json`을
추가로 씁니다.

- `strategies.<s>.metrics.<evaluator>/<metric>` — 질의마다 반복 실행의 평균을 내고 이를
  질의 전체로 평균한 값, 질의 수, `repeat_stdev`(실행별 평균의 편차. 답변 모델의 실행 간
  잡음)입니다. `routed_strategies`는 각 답변이 실제로 쓴 전략을 세므로 `auto`가 어떻게
  분기했는지 보여 줍니다.
- `deltas_vs_baseline.<s>.<metric>` — 두 전략이 모두 채점한 질의에서 첫 전략과 짝지어
  비교한 값입니다. `mean_delta`, 95% 짝지은 부트스트랩 신뢰 구간(`ci_low`/`ci_high`.
  질의를 2,000번 재표집, 시드 0), `wins`/`ties`/`losses`, 질의별 차이가 있습니다. 구간이
  0을 포함하지 않으면 질의 표본이 그 차이를 뒷받침합니다. 이진 지표로 시뮬레이션해 보면
  10%p 차이를 질의 50개에서는 15%, 200개에서는 59%의 실행에서 찾아냈습니다(구간이 참값을
  포함한 비율은 95-98%). 그러니 수백 개 질의로 비교하세요.
- `pairwise_vs_baseline.<s>`(`--pairwise-judge` 사용 시) — GraphRAG / LightRAG 논문의
  맞비교 LLM 판정입니다. `models.roles.evaluation`로 포괄성(comprehensiveness),
  다양성(diversity), 이해 도움(empowerment), 직접성(directness)을 판정합니다(프롬프트:
  `custom_prompts.pairwise_comparison_*`). 판정 모델은 먼저 읽은 답변을 선호하므로 쌍마다
  순서를 바꿔 두 번 판정하고, 두 순서가 같은 답변을 고른 기준만 승리로 셉니다.
  `candidate_win_rate`는 무승부를 0.5로 셉니다. 첫 반복 실행의 답변을 판정하며, 질의와
  전략마다 순서별로 한 번씩 호출합니다.
- 첫 실행의 `run_manifest`.

콘솔에는 평균, 신뢰 구간을 붙인 차이, `auto`의 분기 결과, 맞비교 승률이 출력됩니다.
실패 허용 기준은 모든 실행에 적용됩니다.

### 벤치마크 (MuSiQue)

`prepare-benchmark musique`는 [MuSiQue](https://github.com/StonyBrookNLP/musique)(답변
가능, 다중 홉, CC BY 4.0. Trivedi 외, TACL 2022)를 코퍼스와 `run-eval` 데이터셋으로
바꿉니다. 데이터는 이 저장소에 포함하지 않습니다. 기본적으로 공개 미러의 고정 리비전에서
답변 가능 dev 분할(질문 2,417개)을 내려받아 SHA-256을 확인하며, `--data-path`로 공식 배포본의
`musique_ans_v1.0_*.jsonl`을 대신 읽을 수 있습니다. `corpus/`(서로 다른 단락마다 텍스트
파일 하나: 제목, 빈 줄, 본문. 질문마다 근거 단락 2-4개와 방해 단락 최대 18개)와
`eval_data.json`(`answer`, `answer_aliases`, 근거 단락의 파일 이름인
`reference_sources`, 홉 수인 `category`)을 씁니다.

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

답변 점수는 `answer_contains`와 `token_f1`을 보세요(긴 RAG 답변은 짧은 정답 구간과
정확히 일치하는 경우가 드뭅니다). `retrieval`은 근거 단락이 컨텍스트에 들어갔는지를
채점하며, `grouped_statistics`에서 `category`별로 볼 수 있습니다. 수집 비용은 단락 수에
비례하며 `--limit` 질문 하나마다 단락이 약 19개 늘어납니다. 결과를 공개할 때는 논문을
인용하세요.

---

## 7. 시각화 (`run-visualization`)

이미 내보낸 시각화 데이터 JSON으로 그림을 그리는 **독립** 렌더러입니다. 수집을 다시
실행하거나 AWS에 접근하지 **않습니다**.

`run-ingestion`의 커뮤니티 탐지 단계는 그래프가 코퍼스 전체를 담는 실행(전체 빌드,
초기화, 모든 문서가 새 문서이거나 변경된 증분 실행)마다 `visualization_data.json`을
`graph.visualization.outputs_directory`에 씁니다. 변경되지 않은 문서를 건너뛴 증분
실행은 메모리에 델타만 들고 있으므로 내보내지도 렌더링하지도 않고 INFO 로그만 남기며,
고정한 `outputs_directory`에 있던 이전 파일은 그대로 둡니다. 증분 실행 뒤 전체 그래프를
보려면 전체 빌드(`indexing.reset: true`)를 한 번 실행하고 그 실행이 쓴 파일을
`run-visualization`에 넘기세요. 기본값(`graph.visualization.export_data:
true`)에서는 그래프와 커뮤니티 계층만 담습니다. 그래프를 한 번 훑어 쓰므로 임베딩이나
모델 호출이 없고, `run-visualization`이 렌더링할 때 그래프 토폴로지로 시드를 고정한
spring 레이아웃과 설정한 중심성 지표를 계산합니다. `graph.visualization.enabled`가
`true`이면(기본값은 `false`. 수집마다 노드당 임베딩 호출과 레이아웃 차원 축소가
추가되기 때문) 이 단계가 시각화도 렌더링하고, 파일에 임베딩 기반 레이아웃과 중심성이
들어갑니다. `outputs_directory`를 지정하지 않으면(기본값) `<캐시 디렉터리>/<pipeline_id>/visualization/`(`--cache-directory`, `PipelineConfig.local_directory`)에
쓰므로, S3 캐시 동기화를 켜면 `visualization_data.json`이 캐시와 함께 업로드됩니다(동기화는
`.json` 파일만 복사하므로 HTML은 로컬에서 `run-visualization`으로 다시 렌더링하세요).
이 파일이 `--data-path` 입력입니다. 대화형 그래프는 큰 그래프도 브라우저에서 렌더링할 수
있도록 차수 기준 상위 `interactive.max_nodes`개 노드(기본 2000)만 남기며, 레이아웃(노드 임베딩이나
spring 레이아웃)도 이 노드만으로 계산합니다. 파일에는
그래프 노드와 간선(노드마다 렌더러가 색을 정하는 레벨 0 `community_id` 포함), 커뮤니티 계층과, 수집 중에 렌더링했다면 계산된 `layout`과 중심성이
들어 있으며, 파일 크기를
줄이려고 벡터 속성(`embedding`, `*_embedding`)은 뺍니다.

레이아웃과 오류 처리 방식은 다음과 같습니다.

- `embedding_method: "none"`은 spring 레이아웃을 쓰며 임베딩 클라이언트를
  만들지 않습니다.
- `embedding_method: "description"`은 노드마다 `name: description`을 임베딩
  공급자(`providers.embedding`, 기본값 Bedrock)로 임베딩하고 `layout_method`로 차원을 줄입니다. 임베딩이 실패하면
  `processing.ignore_errors`가 `false`일 때 시각화 단계가 실패합니다(수집 자체는
  계속하며 실패를 로그에 남김). `ignore_errors: true`이면 ERROR를 남기고 토폴로지만
  쓰는 spring 레이아웃으로 대체하며, `visualization_data.json`에
  `"layout_degraded": true`를 기록합니다. 차원 축소가 실패해도 `ignore_errors`와
  관계없이 같은 방식으로 대체합니다(시드를 고정한 spring 레이아웃,
  `layout_degraded: true`).
- 노드 임베딩은 인덱스와 같은 임베딩 모델과 차원(`models.embedding`)을 씁니다.
- 대화형 그래프의 간선 두께와 불투명도는 그래프 자체의 가중치 범위에 맞춰
  조정합니다(로그 스케일 후 min-max 정규화). 그래서 1-10 강도 점수와 병합된 개수를
  모두 구분할 수 있습니다.
- 렌더러가 실제로 쓴 파일만 보고합니다(같은 이름의 기존 파일은 먼저 지웁니다).
  아무것도 렌더링하지 못하면(예: 빈 그래프) `run-visualization`은 오류를 로그에
  남기고 상태 코드 `1`로 종료합니다.

### CLI 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--data-path` | **필수** | 내보낸 시각화 데이터 JSON |
| `--output-directory` | `visualization_outputs` | 렌더링한 파일을 쓸 위치 |
| `--renderers` | 등록된 전체 | 실행할 렌더러: `interactive`, `static` |
| `--output-format` | `text` | `text`, 또는 표준 출력에 JSON 결과 하나를 쓰는 `json`([종료 코드와 JSON 출력](#cli-종료-코드와-json-출력) 참고) |
| `--config-path` | — | `config.yaml` 경로 |
| `--allow-unknown-keys` | off | 알 수 없는 설정 키(와 `UKR__` 변수)에 실패하지 않고 WARNING만 남긴 채 계속합니다([설정을 읽는 순서](#설정을-읽는-순서) 참고) |
| `--backend`, `--llm-provider` 등 | — | 다른 CLI처럼 받기만 합니다(모델이나 저장소를 호출하지 않음). [공급자 플래그](#공급자-플래그) 참고 |

등록된 렌더러는 **`interactive`**(pyvis)와 **`static`**(Bokeh) 두 개입니다. 설정은
`graph.visualization` 아래(`interactive.*`, `static.*`, `embedding_method`/`layout_method`)에
있습니다.

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

---

## 8. 프롬프트 튜닝 (`run-prompt-tuning`)

디렉터리에서 문서를 표본으로 뽑고 설정된 언어 모델
(`models.roles.graph_extraction`)로 코퍼스 특성(도메인, 언어,
페르소나, 엔티티 유형)을 분석한 뒤, 검토해서 `config.yaml`에 합칠 수 있는 도메인 맞춤
`custom_prompts` YAML 조각을 씁니다.

### CLI 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--source-directory` | **필수** | 문서 디렉터리(`.txt`/`.md`/`.markdown`과 `run-ingestion`이 파싱하는 모든 형식. 예: `.pdf`) |
| `--output-path` | `tuned_prompts.yaml` | 출력 YAML 경로 |
| `--max-docs` | `20` | 표본으로 뽑을 최대 문서 수 |
| `--output-format` | `text` | `text`, 또는 표준 출력에 JSON 결과 하나를 쓰는 `json`([종료 코드와 JSON 출력](#cli-종료-코드와-json-출력) 참고) |
| `--config-path` | — | `config.yaml` 경로 |
| `--allow-unknown-keys` | off | 알 수 없는 설정 키(와 `UKR__` 변수)에 실패하지 않고 WARNING만 남긴 채 계속합니다([설정을 읽는 순서](#설정을-읽는-순서) 참고) |
| `--backend`, `--llm-provider` 등 | — | 공급자 선택. [공급자 플래그](#공급자-플래그) 참고 |

```bash
run-prompt-tuning --source-directory ./source --output-path tuned_prompts.yaml --config-path config.yaml
```

출력 YAML에는 `custom_prompts` 블록(과 감지한 도메인과 엔티티 유형이 담긴
`profile`)이 있습니다. 튜닝된 `graph_extraction_system`과
`community_report_system`은 도메인에 맞춘 서문 뒤에 내장 규칙과 출력 형식을 그대로
붙인 것이고, 생성된 few-shot 예시는 맨 뒤에 옵니다. 엔티티 유형은 여전히
`processing.graph_extraction.entity_types`(`{entity_types}` 변수)에서 가져오므로,
모델이 프로필의 `entity_types`를 따르게 하려면 그 값을 여기에 복사하세요.
**내용을 검토한 뒤** 원하는 프롬프트를 `config.yaml`의 `custom_prompts:` 아래에
복사하세요. 일반 텍스트 파일은 그대로 읽고, 그 밖의 형식(PDF, CSV, JSON,
`ParserFactory.register_loader`로 등록한 형식)은 수집과 같은 로더를 거칩니다. 파싱에
실패한 파일은 경고를 남기고 건너뜁니다. 분석 모델이 JSON 프로필을 반환하지 않으면
명령은 0이 아닌 코드로 종료하고 아무것도 쓰지 않습니다(기본 프로필을 쓰면 튜닝된
것처럼 보이기 때문입니다).

---

## 9. 도메인 적응

범용 파이프라인을 의료, 법률, 금융 같은 특정 도메인에 맞추는 방법은 서로 보완하는
두 가지입니다.

### A. `entity_types`(비용이 가장 적고 효과가 가장 큼)

추출 프롬프트에 넣는 엔티티 범주를 바꿉니다. 프롬프트를 다시 쓸 필요가 없습니다.

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

어떤 프롬프트든 `*_system` / `*_human` 텍스트를 재정의할 수 있습니다(기본값 `null` =
내장 프롬프트 사용). `{braces}` 안의 변수는 프레임워크가 채우므로 그대로 두세요. 자주
쓰는 재정의는 다음과 같습니다.

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

재정의할 수 있는 키(각각 `_system` + `_human`): `graph_extraction`,
`description_summarization`, `claim_extraction`, `graph_refinement`,
`community_report`, `answer_generation`, `query_rewrite`,
`entity_extraction`, `drift_hyde`, `drift_primer`, `drift_action`(DRIFT 질의 확장,
primer, 후속 질의 답변),
`strategy_selection`, `keywords_extraction`(LightRAG 이중 레벨),
`global_map`(global 검색 map-reduce), `pairwise_comparison`(`run-eval --pairwise-judge`
판정 모델. JSON 판정 형식은 유지), 그리고 프롬프트 튜닝의 `corpus_profile`
프롬프트입니다.

재정의는 설정을 읽을 때 검사하며, 잘못된 부분이 있으면 해당 키를 알리고 읽기에
실패합니다.

- 프롬프트 호출에 쓰이지 않는 `{name}`은 오류입니다(그대로 두면 호출할 때마다
  `KeyError`가 납니다). JSON 예시처럼 중괄호를 글자 그대로 쓰려면 `{{`와 `}}`로
  쓰세요: `{{"name": "Vendor"}}`.
- 각 프롬프트의 데이터 변수는 system이나 human 템플릿에 들어 있어야 합니다.
  `graph_extraction`과 `claim_extraction`은 `{input_text}`, `answer_generation`은
  `{query}`와 `{context}` 같은 식입니다. 그 밖의 변수(`{max_entities_per_chunk}` 같은
  한도, `{entity_types}`)는 빼도 됩니다.
- `run-prompt-tuning` 출력은 이미 이스케이프되어 있습니다.

`*_system` 오버라이드는 출력 지시를 포함한 내장 시스템 프롬프트 전체를
대체합니다. `graph_extraction`과 `community_report`의 시스템 프롬프트에는 파서가
읽는 XML 형식(추출에서는 환각 방지 검사가 확인하는 원문 그대로의 `<source_text>`
규칙 포함)이 들어 있으므로 대체 프롬프트에도 이를 남겨야 합니다. 내장 규칙과 형식
앞에 도메인 서문을 붙여 주는 `run-prompt-tuning` 출력에서 시작하거나
`unified_kg_rag/domain/prompts/graph_extraction.py`의 형식 부분을 복사하세요. 두
프롬프트의 오버라이드에 파서가 읽는 태그(`<entities>`, `<relationships>`,
`<source_text>`, `<community_name>`, `<summary>`, `<rating>`,
`<rating_explanation>`, `<findings>`)가 빠져 있으면 설정을 읽을 때 빠진 태그를
알리는 경고를 남깁니다.

**권장 흐름:** `run-prompt-tuning`으로 시작점을 만듦 → 검토 → 쓸 만한 프롬프트를
합치고 `entity_types`를 직접 조정 → 다시 수집.

---

## 10. 운영과 문제 해결

### IAM 권한

CLI를 실행하는 주체에 다음 권한을 주세요. Bedrock(설정한 모델 ID에 대한 InvokeModel /
InvokeModelWithResponseStream / CountTokens와 임베딩. CountTokens가 없으면 토큰 수를
추정값으로 계산함), Neptune(`use_iam: true`일 때 connect /
SigV4), OpenSearch(설정한 인덱스 읽기·쓰기), S3(설정한 버킷), DynamoDB(증분 인덱싱을
켠 경우).

> **Bedrock 재순위화에는 별도 정책 문이 필요합니다.** 재순위화는 Rerank API를
> 호출하며, 이 API는 `bedrock:Rerank`를 `InvokeModel`과 다른 리소스 형태로
> 인가합니다. 파운데이션 모델 ARN이나 추론 프로파일 ARN으로 범위를 좁힌 정책 문으로는
> 거부됩니다. CDK 태스크 역할(`iac/stacks/compute_stack.py`)처럼 별도 정책 문에서
> `Resource: "*"`에 `bedrock:Rerank`를 허용하세요. 이 권한이 없으면 질의마다
> `Reranking failed: ...` ERROR가 남고, `metadata.rerank.status: failed`를 보고하며,
> 재순위화하지 않은 융합 결과를 반환합니다(`search.reranking.required: true`이면
> 질의가 실패합니다).
> 재순위화는 기본으로 켜져 있습니다(`search.reranking.enabled`). 이 권한을 줄 수
> 없다면 `false`로 설정하세요.
>
> 모델 해석기는 `bedrock:ListInferenceProfiles`도 호출합니다. 이 권한은
> `Resource: "*"`를 쓰는 계정 수준 읽기 권한입니다. 가드레일을 쓰려면 해당 가드레일
> ARN(`arn:aws:bedrock:<Bedrock 리전>:<계정>:guardrail/<id>`)에 대한
> `bedrock:ApplyGuardrail`이 필요합니다.

### 자주 보는 오류

CLI가 보고하는 메시지와 그 의미, 해결 방법은 AWS 모드와 로컬 모드 모두
[운영 런북: 자주 보는 오류](./operations.ko.md#자주-보는-오류)에 정리되어 있습니다.

### CLI 종료 코드와 JSON 출력

모든 CLI(`run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
`run-prompt-tuning`, `prepare-benchmark`)는 다음 코드로 종료합니다.

| 코드 | 의미 |
|---|---|
| `0` | 성공 |
| `1` | 실행 실패: 단계나 질의 실패, `--max-failure-rate` 초과, 업로드 실패, 렌더링한 파일 없음 |
| `2` | 작업을 시작하기 전의 사용법·설정 오류: 알 수 없거나 잘못된 플래그, 없는 입력 파일이나 디렉터리, 잘못된 설정 파일이나 환경 변수 덮어쓰기, 설정하지 않은 저장소 엔드포인트 |
| `130` | Ctrl-C(SIGINT)로 중단 |
| `143` | SIGTERM으로 중지(ECS `StopTask`, Step Functions 시간 초과). `run-ingestion`은 끝난 단계를 먼저 체크포인트로 저장합니다 |

파일을 가리키는 플래그 이름은 모든 CLI에서 같습니다. `--source-directory`(코퍼스
디렉터리), `--data-path`(입력 데이터 파일), `--output-directory` / `--output-path`(명령이
쓰는 위치), `--cache-directory`, `--config-path`입니다. 플래그는 전체 이름으로 써야 하며,
`--output`처럼 앞부분만 쓰면 더 긴 플래그로 해석하지 않고 거부합니다.

`--output-format json`은 결과를 JSON 문서 하나로 표준 출력에 쓰고, 배너·진행 상황·로그는
표준 오류로 보내므로 출력을 `jq`로 바로 넘길 수 있습니다. 담기는 내용은 답변이나 검색
결과(`run-rag`), 파이프라인 id·상태·단계별 결과(`run-ingestion`. 표준 출력에 함께 쓰는
`--metrics-sink cloudwatch`와는 같이 쓸 수 없음), 실행별 요약·비교·실패 허용치
초과(`run-eval`), 쓴 파일(`run-visualization`), 출력 경로와 감지한 프로필
(`run-prompt-tuning`), 쓴 항목 수(`prepare-benchmark`)입니다.

### 대규모, 다국어, 이질적인 코퍼스

- **대규모 코퍼스:** `processing.max_concurrency`를 올리세요(LLM 단계는 I/O가
  병목). 그래프 쓰기는 `indexing.neptune.index_concurrency`를
  올리면 Gremlin 연결 풀도 함께 늘어납니다. `indexing.opensearch.persist_embedding_cache`와
  `--s3-sync`를 켜면 다시 실행할 때 다시 계산하지 않습니다.
- **다국어:** `processing.translation.source_language` / `target_language`(+
  `additional_target_languages`)를 지정하고 `indexing.opensearch.language_analyzers`에
  언어 분석기를 추가하세요. 단일 언어 코퍼스에서는 번역 단계가 아무 일도 하지 않습니다.
- **이질적인 도메인:** `entity_types`를 여러 도메인의 합집합으로 맞추거나, 도메인이나
  테넌트마다 인덱스를 따로 두세요. 각각을 자기 `processing.document_parsing.index_value`로
  수집하고 같은 값을 `--suffix`로 넘겨 질의합니다(`indexing.additional_suffix`는 두
  번째 구간을 더하며 양쪽에서 같아야 함).
- **증분:** DynamoDB를 켜면 대규모 코퍼스도 다음 실행부터는 바뀐 부분에만 비용이
  듭니다.

### 비용 참고

비용은 대부분 LLM 호출에서 나옵니다. 주요 비용 요인은 `graph_extraction`(청크당 1회
이상), `gleaning`(청크당 최대 `max_rounds`회), `community_detection`의 리포트 생성,
`claim_extraction`(텍스트 단위당 1회, 기본은 꺼짐), 질의마다의 답변 생성입니다. 줄이는
방법은 다음과 같습니다. 기계적인 단계에는 저렴한 모델을 쓰고(청킹, 번역, map-reduce,
설명 요약은 기본으로 Haiku급 모델 사용), `gleaning.max_rounds`를 제한하고, 필요하지
않으면 `claim_extraction`을 끈 채로 두고, 임베딩·단계 캐시를 켜고, 증분 인덱싱으로
전체 재수집을 피하세요.


---

## 11. Python 라이브러리로 사용하기

CLI는 공개 클래스 두 개를 감싼 얇은 래퍼입니다. 이 클래스와 그 밖에 쓰는 이름은 모두
패키지 최상위에서 import하세요. `unified_kg_rag`가 내보내는 이름(`unified_kg_rag.__all__`)이
공개 API이고, 그 밖의 모듈 경로는 내부 구현이라 릴리스마다 바뀔 수 있습니다. `Config`는
`get_config`로 만드세요. `get_config`는 CLI와 같은 설정 파일·환경 변수 단계를 적용합니다(§2).

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

# Ingest. PipelineConfig carries the run options the run-ingestion flags set
# (pipeline_id, cache directory, S3 sync, enabled stages, force_rebuild).
# run() ingests the directory it is given (the constructor's source_directory
# defaults to processing.document_parsing.source_directory and is replaced by
# it): parsing, loading and the incremental registry scope all use it.
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

- `pipeline.run`은 `PipelineContext`를 반환합니다. 단계가 실패하면 예외를 던지지 않고
  `status`를 `FAILED`로 설정하므로, 위 예시처럼 확인하세요(CLI는 이 경우 0이 아닌
  코드로 종료합니다).
- `RAGInput`은 `run-rag` 옵션을 필드로 받습니다: `search_strategy`, `search_type`,
  `top_k`, `suffix`([인덱스 접미사](#인덱스-접미사)), `filters`, `conversation_id`와
  `use_memory`, `target_language`. 이 키를 담은 일반 딕셔너리도 됩니다.
- `chain.search(...)`(`await chain.asearch(...)`)는 답변을 생성하지 않고 같은 검색을
  실행해 `SearchOutput`을 반환합니다. 순위가 매겨진 `search_results`, `processed_query`,
  `metadata`를 담습니다.
- 비동기 코드에서는 `await chain.ainvoke(...)`와 `await chain.aclose()`를 쓰세요.
  비동기 호스트는 시작할 때 이벤트 루프의 실행기 크기도 한 번 지정해야 합니다:
  `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`
  (`unified_kg_rag`).
- 만든 객체는 항상 닫으세요. 두 클래스 모두 처음 사용할 때 연결을 열며,
  `close()`/`aclose()`를 호출하지 않으면 가비지 컬렉션 때까지 연결이 열려 있습니다.
- 패키지를 import해도 로깅 핸들러는 설정하지 않습니다. CLI와 같은 로깅을 쓰려면
  `unified_kg_rag.setup_logging(config)`를 호출하세요. 호출하지 않으면 패키지의
  레코드는 일반 메시지로 호스트 핸들러에 전달되고, 구조화 필드(`structlog.contextvars`로
  바인딩한 값 포함)는 레코드 속성이 됩니다. `LogRecord` 속성과 이름이 같은
  필드(`name`, `process`, `msg` 등)에는 끝에 `_`가 붙습니다.

확장 코드도 패키지 최상위에서 import합니다. 포트(`LLMFactoryPort`, `GraphIndexer`,
`DocStatusPort` 등), `Providers`, `register_provider`, `BaseSearchStrategy`와
`register_strategy`, `ParserFactory`, `register_renderer`가 여기에 있습니다.
두 생성자 모두 `providers=` 번들로 백엔드를 주입받습니다
(`Providers(config, llm_factory=..., doc_status=..., vector_indexer_factory=...,
graph_indexer_factory=..., retriever_builders=...)`). 번들 없이 쓸 때
`DataIngestionPipeline`은 `doc_status`, `vector_indexer`, `graph_indexer`도 받고,
`GraphRAGChain`은 `model_factory`, `retriever_builders`를 받습니다. 이 중 하나를
`providers=`와 함께 넘기면 `ValueError`가 납니다. `GraphRAGChain(memory_manager=...)`는
어느 쪽과도 함께 쓸 수 있습니다.
[설계 문서 §15](./design.ko.md#15-확장-가이드)를 참고하세요.

---

## 12. 제한 사항

- **추출 단계에서 환각이 생길 수 있습니다.** 그래프 추출은 LLM 단계이므로 원문이
  뒷받침하지 않는 엔티티나 관계를 만들 수 있고, 이런 항목이 검색까지 이어집니다.
  `processing.graph_extraction.entity_grounding.enabled`는 인용한 `source_text`가
  청크에 없는 항목을 버리거나 가중치를 낮추며, 기본으로 꺼져 있습니다.
- **실행 결과를 재현할 수 없습니다.** 추출, gleaning, 커뮤니티 리포트, `auto` 라우터,
  답변은 모두 LLM 출력이므로, 같은 코퍼스를 두 번 수집하면 서로 다른 그래프가 만들어지고
  같은 질의에도 다른 답변이 나올 수 있습니다. Leiden 군집화는 같은 그래프에서는
  결정적입니다. 설정을 비교할 때는 수집과 질의를 두 번 이상 실행해 비교하세요.
- **재순위화는 리전에 따라 다릅니다.** 재순위화 모델은 모든 Bedrock 리전에 있지
  않습니다([필수 모델](#필수-모델) 참고). 재순위화가 없으면 융합 결과의 순위만 씁니다.
  질의마다 `metadata.rerank`에 이 사실이 나타납니다([재순위화 상태](#재순위화-상태)).
- **기본 제공 모델 공급자는 세 가지입니다.** 언어 모델과 임베딩은 Amazon Bedrock(기본값),
  Ollama, OpenAI 호환 서버를, 재순위화는 Bedrock이나 sentence-transformers 크로스 인코더를
  씁니다([로컬 모드](#로컬-모드-aws-없이)). 다른 공급자는 `register_provider`로 빌더를
  등록해야 합니다([설계 문서 §15](./design.ko.md#15-확장-가이드)).
- **기본 제공 저장소 조합은 두 가지입니다.** Neptune(로컬 개발에서는 TinkerPop
  Gremlin Server)과 OpenSearch, 그리고 로컬 모드용 내장 SQLite 그래프·벡터
  저장소(호스트 한 대, 인덱스당 문서 약 100,000개)입니다. 다른 저장소를 쓰려면 사용자
  정의 어댑터가 필요합니다.
- **서빙 계층이 없습니다.** 패키지는 CLI와 Python 라이브러리를 제공하며, HTTP API,
  최종 사용자 인증, 요청 속도 제한은 없습니다.
- **필터는 접근 제어가 아닙니다**(§4). 테넌트는 인덱스 접미사로 분리하세요.
- **가벼운 수집은 쓸 수 있는 전략을 제한합니다.**
  `graph.community_detection.enabled: false`이면 `global`과 `drift`가 쓸 커뮤니티
  리포트가 없고, `indexing.opensearch.build_relationship_vector_index: false`이면
  `mix`와 `hybrid`가 고수준 키워드 검색을, `local`이 관계 섹션을 쓰지 못합니다.
- **증분 인덱싱은 근사치입니다**(§5). 증분 실행은 그래프 전체를 다시 군집화하지 않고
  커뮤니티를 덧붙이며, 엔티티가 현재 이름의 id를 받은 뒤에도 저장된 커뮤니티와
  리포트에는 이전 id가 남습니다. 둘 다 전체 재구축(`indexing.reset: true`)을 해야만
  갱신됩니다. 설명, 이름, id, 유형, rank, 주장은 현재 코퍼스를 따릅니다. DynamoDB
  레지스트리에서는 문서 하나의 청크 ID가 400KB 항목 하나에 들어가야 합니다(§5,
  *문서 크기 한도*).
- **동시 실행에는 레지스트리가 필요합니다.** 레지스트리를 쓰면 한 인덱스
  네임스페이스의 두 번째 실행은 `NamespaceLeaseError`로 멈춥니다(§5, *동시 실행*).
  레지스트리가 없으면 잠그는 장치가 없으니, 같은 인덱스 접미사에 대해 수집 두 개를
  동시에 돌리지 마세요(전체 재색인의 별칭 교체와 이전 인덱스 정리가 다른 실행의
  인덱스를 지울 수 있음). `IncrementalIndexer`를 직접 쓰는 API는 임대를 얻지 않습니다.
  [운영 런북: 동시 실행](./operations.ko.md#동시-실행)을 참고하세요.
- **격리는 원인 문서를 가려내지 않습니다**(§5). 인덱싱이 계속 죽는 델타의 모든 문서가
  실행을 죽이는 문서와 함께 격리됩니다.
- **Markdown과 HTML**에는 선택 extra인 `unstructured`(Python 3.11 이상)가 필요하며,
  컨테이너 이미지에는 기본으로 빠져 있습니다.
- **임베딩 모델(또는 차원)을 바꾸면** 다시 인덱싱해야 합니다(`indexing.reset: true`).
  증분 실행은 다른 모델의 인덱스에 쓰지 않습니다([작업 흐름](#작업-흐름) 참고).

---

*함께 보기: 개요와 빠른 시작은 [README.ko.md](../README.ko.md), 아키텍처와 내부 구조는
[설계 문서](./design.ko.md)를 참고하세요.*
