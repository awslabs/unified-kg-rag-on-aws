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

콘솔 진입점 다섯 개(`pyproject` 스크립트로 정의)는 다음과 같습니다.

| 스크립트 | 모듈 | 용도 |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | 지식 그래프 구축과 갱신 |
| `run-rag` | `application.cli.run_rag_chain` | 그래프 질의 |
| `run-eval` | `application.cli.run_evaluation` | 검색과 생성 평가 |
| `run-visualization` | `application.cli.run_visualization` | 내보낸 그래프 렌더링(수집 없음) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | 도메인에 맞춘 프롬프트 생성 |

각 CLI의 `--help` 끝에는 실행 예시와 이 문서의 해당 절 링크가 있습니다.

## 목차

1. [사전 요구사항과 설치](#1-사전-요구사항과-설치)([필수 모델](#필수-모델) 포함)
2. [설정](#2-설정)
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

관련 문서: [모델 카탈로그](./models.ko.md) · [운영 런북](./operations.ko.md) ·
[설계 문서](./design.ko.md) · [`iac/README.md`](../iac/README.md)

---

## 1. 사전 요구사항과 설치

### 런타임

- **Python 3.10 – 3.12**
- **[uv](https://docs.astral.sh/uv/)**(권장 패키지 관리자. `pip`도 사용 가능)

### AWS 서비스

| 서비스 | 필수 여부 | 용도 |
|---|---|---|
| **Amazon Bedrock** | 필수 | 모든 LLM 호출(청킹, 추출, gleaning, 커뮤니티 리포트, 답변 생성), 임베딩, 재순위화. 설정한 모델 ID의 모델 액세스를 활성화해야 합니다. |
| **Amazon Neptune** | 필수 | 지식 그래프(엔티티, 관계, 커뮤니티)와 질의 시점의 다중 홉 순회. |
| **Amazon OpenSearch** | 필수 | 벡터 인덱스와 BM25 어휘 인덱스(텍스트 단위, 엔티티, 커뮤니티 리포트, 관계, 주장). |
| **Amazon S3** | 필수 | 파이프라인 캐시 동기화, 선택적인 임베딩 캐시 저장, 문서 저장. |
| **Amazon DynamoDB** | 증분 인덱싱을 쓸 때만 | 콘텐츠 해시로 코퍼스의 변경분을 계산하는 문서 상태 레지스트리. |

### 필수 모델

처음 실행하기 전에 다음 Bedrock 모델의 액세스를 활성화하세요. 표의 모델은 코드
기본값이며, 다른 모델을 설정하면 그 행의 모델이 설정한 모델로 바뀝니다.

| 역할 | 기본 모델 ID | 설정 키 |
|---|---|---|
| 기본 계층(추출, gleaning, 커뮤니티 리포트, 답변, 판정 모델) | `anthropic.claude-sonnet-5-5` | `aws.bedrock.default_model_id` |
| 빠른 계층(청킹, 번역, 요약, 라우팅, map 단계) | `anthropic.claude-haiku-5-5` | `aws.bedrock.fast_model_id` |
| 임베딩(인덱싱, 평가, 시각화 레이아웃) | `amazon.titan-embed-text-v2:0` | `indexing.opensearch.embedding_model_id`, `evaluation.embedding_model_id` |
| 재순위화 | `cohere.rerank-v3-5:0` | `search.reranking.rerank_model_id` |

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

- **로컬 저장소(개발용).** Neptune과 OpenSearch 대신 컨테이너를 띄우고 모델은
  Bedrock을 씁니다. 아래 [로컬 저장소 (개발용)](#로컬-저장소-개발용)를 참고하세요.

- **기존 서비스 사용.** Neptune, OpenSearch, S3(필요하면 DynamoDB)가 이미 실행
  중이라면 엔드포인트를 `config.yaml`(아래 §2.1)에 적고 다음 단계로 넘어가면
  됩니다. 설정한 모델 ID의 Bedrock 모델 액세스가 활성화되어 있는지 확인하세요.
- **번들 CDK 앱으로 전체 프로비저닝.** 저장소의 [`iac/`](../iac/README.md)에는
  스택 전체를 명령 하나로 만드는 선택형 Well-Architected AWS CDK 앱이 있습니다.
  네트워킹(VPC와 엔드포인트), Neptune 클러스터, OpenSearch 도메인, DynamoDB 문서
  상태 테이블, S3 캐시 버킷, ECS Fargate 데이터 플레인, Step Functions 수집
  파이프라인, CloudWatch 관측성을 만들고, 선택적으로 리전을 고정한 Bedrock 가드레일도
  만듭니다.

  ```bash
  cd iac
  python -m venv .venv && . .venv/bin/activate
  pip install -r requirements.txt

  cdk synth              # preview — no AWS changes, no cost
  cdk bootstrap          # once per account/region
  cdk deploy --all       # creates billable resources (Neptune + OpenSearch are hourly)
  ```

  배포가 끝나면 CloudFormation 스택 출력값의 Neptune / OpenSearch / S3 엔드포인트를
  `config.yaml`에 복사하세요(엔드포인트는 스킴 없는 호스트 이름입니다). Neptune과
  OpenSearch는 VPC 안에서만 접근할 수 있으므로 CLI도 VPC 안에서 실행해야 합니다.
  예를 들어 배포된 태스크 정의로 태스크를 실행하면 됩니다. 스택이 만든 태스크는
  엔드포인트를 환경 변수로 받습니다. `dev` 프로파일은 `cdk destroy --all`로 정리합니다.
  스택 목록, `-c key=value` 옵션 전체(VPC 재사용, 인스턴스 크기, CMK, 삭제 보호,
  cdk-nag), 프로덕션 강화 체크리스트는 [`iac/README.md`](../iac/README.md)를
  참고하세요.

### 설치

```bash
git clone <repository-url>
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
파서를 설치하세요. 이 extra는 `unstructured>=0.24.0`을 요구하며, 이 버전은 URL
분할의 SSRF 취약점을 고쳤고 NLTK에 의존하지 않습니다. Python 3.10에서는 이 extra를
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
   OPENSEARCH_USERNAME=your_opensearch_username
   OPENSEARCH_PASSWORD=your_opensearch_password
   ```

   CLI(`run-ingestion`, `run-rag`)는 `.env` 파일을 자동으로 읽습니다.
   `use_iam: true`이면 `.env`가 필요 없습니다.

### 로컬 저장소 (개발용)

개발할 때는 Neptune과 OpenSearch를 로컬 컨테이너로 대체할 수 있습니다. 모델은
여전히 Bedrock에서 호출하므로 Bedrock 액세스 권한이 있는 AWS 자격 증명이 필요합니다.
S3는 `--s3-sync`를 쓸 때만, DynamoDB는 증분 인덱싱을 쓸 때만 필요합니다.

```bash
docker compose -f docker/compose.local.yaml up -d --wait
uv run run-ingestion --config-path docker/config.local.yaml --source-directory ./docs-in
uv run run-rag --config-path docker/config.local.yaml --query "..."
docker compose -f docker/compose.local.yaml down -v
```

[`docker/compose.local.yaml`](../docker/compose.local.yaml)은 TinkerPop Gremlin
Server(메모리 기반 TinkerGraph)와 단일 노드 OpenSearch 2.13을 실행합니다.
OpenSearch는 보안 플러그인을 끄고 `analysis-nori` 플러그인을 설치합니다(한국어
매핑이 `nori`를 쓰며, Amazon OpenSearch Service에는 기본으로 들어 있습니다).
[`docker/config.local.yaml`](../docker/config.local.yaml)은
`aws.neptune.use_ssl: false`, `use_iam: false`(SigV4 없는 `ws://`)와
`aws.opensearch.allow_anonymous: true`, `use_ssl: false`(인증 없는 `http://`)로
프레임워크를 이 컨테이너에 연결합니다.

Bedrock 없이 저장소만 확인하려면 해싱 임베딩 공급자를 쓰는 스모크 테스트를
실행하세요: `LOCAL_STORES=1 uv run pytest tests/integration/test_local_stores.py`.

Neptune과 다른 점이 있습니다. TinkerGraph는 그래프를 메모리에만 두며, 정점을 만들 때
쓴 리스트 속성에 중복 값을 그대로 저장합니다(Neptune은 한 번만 저장). compose 파일에는
보안 강화 설정이 없으며 `127.0.0.1`에만 바인딩합니다.

---

## 2. 설정

템플릿으로 설정 파일을 만들고 모든 CLI에 `--config-path config.yaml`로 지정하세요.

```bash
cp config-template.yaml config.yaml
```

**모든 옵션의 기준 문서는 `config-template.yaml`입니다.** 키마다 기본값과 역할
설명 주석이 있습니다. 이 절에서는 설정을 읽는 방식, 자주 조정하는 항목, 설정 파일을
덮어쓰는 환경 변수를 다룹니다. 여기에 없는 항목은 템플릿을 참고하세요.

### 설정을 읽는 순서

값은 세 단계로 정해지며, 뒤 단계가 앞 단계를 덮어씁니다.

1. **내장 기본값**: `unified_kg_rag/domain/models/config.py`의 Pydantic 모델입니다.
   `--config-path` 없이 실행하면 CLI는 이 기본값에 환경 변수만 적용해 동작합니다.
2. **YAML 파일**: 지정한 키는 기본값을 대체하고, 지정하지 않은 키는 기본값을
   유지합니다. 따라서 `config.yaml`에는 바꿀 키만 적어도 됩니다. 값이 딕셔너리인
   키(예: `logging.library_levels`, `graph.visualization.interactive`)는 기본
   딕셔너리와 병합되지 않고 통째로 대체됩니다.
3. **환경 변수**(§2.10): 마지막에 적용되어 앞의 두 단계를 모두 덮어씁니다.

YAML 값은 파일을 읽을 때 검증합니다. 타입이 틀리거나 지원하지 않는 값이면
`Configuration validation error: ...`와 함께 CLI가 멈춥니다. 알 수 없는 키(오타나
새 릴리스에서 제거된 키)는 실행을 멈추지 않습니다. `Unknown config key '<path>' is ignored`
WARNING을 남기고 버리므로, 파일을 고치거나 업그레이드한 뒤에는 로그를 확인하세요.

폐기 예정 키도 받아들입니다. 각 키의 값은 대체 키에 적용되고
`Config key '<old>' is deprecated; applied as '<new>: <value>'` WARNING이 남습니다.
대체 키도 함께 지정했다면 폐기 예정 키는 무시합니다.

| 폐기 예정 키 | 대체 키 |
|---|---|
| `search.llm_retry` | `aws.bedrock.transient_retry` |
| `aws.bedrock.effort` | `aws.bedrock.default_effort`(값 그대로) |
| `processing.max_retries` | `processing.max_attempts`(값 그대로) |
| `indexing.neptune.max_retries` | `indexing.neptune.max_attempts`, 값에 1을 더함(폐기 예정 키는 첫 시도 뒤의 재시도만 셉니다) |
| `evaluation.ragas_max_retries` | `evaluation.ragas_max_attempts`(값 그대로) |

`max_attempts` 키는 모두 첫 시도를 포함한 총 시도 횟수이며, `1`이면 재시도하지
않습니다.

아래 표의 기본값은 내장 기본값이며, `config-template.yaml`도 같은 값을 씁니다.

### 2.1 `aws` — 서비스 엔드포인트와 자격 증명

| 키 | 기본값 | 역할과 변경 시점 |
|---|---|---|
| `aws.region_name` | `"us-west-2"` | Neptune, OpenSearch, S3, DynamoDB의 리전이며, `aws.bedrock.region_name`을 지정하지 않으면 Bedrock 리전이기도 합니다. `AWS_REGION`이 이 값을 덮어씁니다(§2.10 참고). 기본 리전에는 재순위화 모델 두 개를 포함한 기본 모델이 모두 있지만, 일부 리전(예: `ap-northeast-2`)에는 재순위화 모델이 없습니다. |
| `aws.profile_name` | `null` | 이름 있는 AWS 프로파일입니다. `null`이면 기본 자격 증명 체인을 씁니다. |
| `aws.bedrock.region_name` | `null` | Bedrock 모델·임베딩·재순위화 호출을 보내는 리전이며, 가드레일도 이 리전에 있어야 합니다. `null`이면 `aws.region_name`을 씁니다. 모델을 다른 리전에서 활성화한 경우에만 지정하세요. Bedrock으로 가는 경로가 VPC 엔드포인트뿐인 프라이빗 VPC에서는 `null`로 두거나 VPC 리전을 지정하세요. |
| `aws.bedrock.enable_global_profile` | `true` | 교차 리전(global) 추론 프로파일을 찾아 씁니다. Claude 4.7 이후 모델과 GPT 모델은 프로파일로만 호출할 수 있으므로 켜 두세요. |
| `aws.bedrock.default_model_id` | `"anthropic.claude-sonnet-5-5"` | 기본 계층 역할 전체가 쓰는 모델입니다(모델 선택 참고 사항 참고). |
| `aws.bedrock.fast_model_id` | `"anthropic.claude-haiku-5-5"` | 빠른 계층 역할 전체가 쓰는 모델입니다. |
| `aws.bedrock.default_max_output_tokens` | `16384` | 요청마다 보내는 `max_tokens`이며, 모델 최대값을 넘으면 최대값으로 맞춥니다. 답변이 잘리면(`stopReason: max_tokens`) 올리세요. `null`이면 모델 최대값을 보냅니다. |
| `aws.bedrock.default_effort` | `"high"` | `default_model_id` 호출(adaptive thinking을 쓰는 Claude 모델과 GPT 모델)의 추론 강도입니다. `low`, `medium`, `high`, `xhigh`, `max` 중 하나이며, 낮추면 비용과 지연 시간이 줄어듭니다. |
| `aws.bedrock.ingestion_effort` | `null` | 기본 계층 수집 호출(그래프 추출, gleaning, 주장 추출, 커뮤니티 리포트. 출력 수정기는 `fixing.fixing_model_id`가 기본 계층 모델일 때만)의 추론 강도입니다. `null`이면 `default_effort`를 따릅니다. 질의 시점의 추론 강도는 그대로 두고 수집 비용만 줄이려면 낮추세요(예: `"medium"`). 빠른 계층 수집 호출은 `fast_effort`를 그대로 씁니다. |
| `aws.bedrock.fast_effort` | `"low"` | `fast_model_id`가 `default_model_id`와 다를 때 `fast_model_id` 호출의 추론 강도입니다. 기본 제공 Claude Haiku 5.5는 adaptive thinking을 하므로 이 값이 추론 강도를 정합니다. 추론하지 않는 빠른 모델에는 영향이 없습니다. |
| `aws.bedrock.enable_1m_context` | `false` | 1M 컨텍스트 창이 베타인 모델에서 1M 창을 씁니다(추가 요금). Claude 5는 1M 창이 기본입니다. |
| `aws.bedrock.model_overrides` | `{}` | 패키지가 모르는 언어 모델의 기능 정보 레코드입니다(모델 선택 참고 사항 참고). 임베딩 모델과 재순위화 모델은 정해진 목록에서만 고릅니다. |
| `aws.bedrock.guardrail.identifier` | `null` | Bedrock 가드레일 ID 또는 ARN입니다. 지정하면 가드레일이 켜집니다. |
| `aws.bedrock.guardrail.apply_to` | `"query"` | `query`는 사용자 질의 경로에만, `all`은 모든 호출에 가드레일을 적용합니다(아래 가드레일 참고 사항 참고). |
| `aws.bedrock.guardrail.trace` | `false` | 가드레일 trace를 출력합니다. InvokeModel 경로에서 개입을 감지하려면 필요합니다. |
| `aws.bedrock.transient_retry.max_attempts` | `5` | botocore가 재시도하지 않는 일시적 Bedrock 오류(HTTP 424 등)에 대한 호출당 시도 횟수입니다. 임베딩과 질의 시점 호출에 적용합니다. `1`이면 재시도하지 않습니다. |
| `aws.neptune.endpoint` | `null` | **필수.** Neptune 클러스터 엔드포인트입니다. |
| `aws.neptune.use_iam` | `true` | Neptune 요청에 SigV4 서명을 붙입니다. |
| `aws.neptune.use_ssl` | `true` | `wss://`로 연결합니다. Neptune에서는 필수이며, 로컬 Gremlin Server에서만 `use_iam: false`와 함께 `false`로 둡니다(§1 로컬 저장소). |
| `aws.neptune.pool_size` | `4` | Gremlin 연결 풀 크기입니다. `indexing.neptune.index_concurrency`가 더 크면 클라이언트가 그 값으로 늘립니다. |
| `aws.opensearch.endpoint` | `null` | **필수.** OpenSearch 도메인 엔드포인트입니다. |
| `aws.opensearch.use_iam` | `false` | `false`이면 환경 변수 `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD`를 읽습니다(§1 인증). |
| `aws.opensearch.allow_anonymous` | `false` | 인증 없이 연결합니다. 보안 플러그인을 끈 로컬 OpenSearch용입니다(§1 로컬 저장소). `use_iam`이나 username/password와 함께 쓸 수 없습니다. |
| `aws.opensearch.sigv4_service_name` | `"es"` | 관리형 도메인은 `es`, OpenSearch Serverless는 `aoss`입니다. 값이 틀리면 검색 결과가 0건으로 나오는 경우가 많습니다. |
| `aws.s3.bucket_name` | `null` | 캐시 동기화와 임베딩 캐시 저장에 쓰는 버킷입니다. |
| `aws.s3.encryption.encryption_type` | `"BUCKET_DEFAULT"` | `BUCKET_DEFAULT`는 버킷 기본 암호화를 따릅니다. `AES256`이나 `aws:kms`(`kms_key_id`와 함께)는 객체별 헤더를 강제합니다. 단계 캐시 동기화와 저장된 임베딩 캐시에 적용합니다. |
| `aws.dynamodb.enabled` | `false` | 증분 인덱싱용 문서 상태 레지스트리를 켭니다(§5). |
| `aws.dynamodb.table_name` | `"unified-kg-rag-on-aws-doc-status"` | 문서 상태 테이블 이름입니다. |
| `aws.dynamodb.create_table_if_missing` | `true` | 처음 사용할 때 테이블을 만듭니다. 테이블을 IaC로 관리한다면 `false`로 두세요. |

> **가드레일 적용 범위와 위치.** 가드레일은 LLM 호출이 가는 리전인
> `aws.bedrock.region_name`에 있어야 합니다. 기본값 `apply_to: "query"`에서는
> 답변 생성, 질의 정제, 질의 시점 엔티티·키워드 추출, global/DRIFT map-reduce에
> 가드레일을 붙입니다. 수집 모델, 프롬프트 튜너, 평가용 판정 모델에는 붙이지
> 않습니다. `NAME`을 익명화하는 PII 가드레일은 추출된 엔티티 이름을 `{NAME}` 같은
> 자리 표시자로 바꿔 서로 다른 사람을 한 노드로 합치고, `PROMPT_ATTACK` 필터는
> 지시문처럼 보이는 코퍼스 텍스트를 차단할 수 있기 때문입니다. `apply_to: "all"`은
> 추출에 써도 안전한 정책일 때만 쓰세요. 질의 경로에서도 `NAME` 익명화는 문제가
> 됩니다. 답변에 해당 인물 대신 `{NAME}`이 나오고, 엔티티를 시작점으로 삼는 검색이
> 시작점을 잃습니다. 그래서 `iac/`가 만드는 기본 가드레일은 이메일, 전화번호, 카드
> 번호만 익명화하고 `NAME`은 익명화하지 않습니다. 가드레일이 개입할 때마다 누적
> 횟수와 함께 WARNING
> (`Bedrock guardrail '<id>' intervened on a <purpose> model call ...`)을 남깁니다. InvokeModel 경로(`ChatBedrock`, 교차 리전이
> 아닌 모델 ID에 사용)에서는 `trace: true`일 때만 개입을 감지하며, 가드레일 자체는
> 어느 경우든 적용됩니다.
>
> `setup_chain`으로 체인을 만들거나 질의가 아닌 작업에서 `get_model`을 호출하는
> 사용자 코드는 `model_purpose=ModelPurpose.INGESTION`(또는 `EVALUATION`)을 넘겨야
> 합니다. 표시하지 않은 호출은 `QUERY`로 간주하므로 가드레일이 적용되고 질의 시점의
> 일시적 오류 재시도도 적용됩니다.

> **S3 캐시 암호화.** CDK 스택에서 `use_cmk=true`이면 버킷 기본 암호화가 고객 관리형
> KMS 키이므로 `BUCKET_DEFAULT`도 그 키를 씁니다. `AES256`을 강제하면 버킷의 CMK를
> 우회하게 됩니다. 기본 암호화가 SSE-KMS인 버킷을 재사용하면 쓰기 주체에 해당 키의
> `kms:GenerateDataKey`와 `kms:Decrypt` 권한이 필요합니다.

#### 모델 선택 참고 사항

모든 LLM 역할은 두 계층 중 하나에 속합니다. 추론 비중이 큰 역할은
`aws.bedrock.default_model_id`, 가벼운 역할은 `aws.bedrock.fast_model_id`를 쓰므로
파이프라인 전체의 모델을 한 줄로 바꿀 수 있습니다. 역할별 `*_model_id` 키를 지정하면
계층 설정보다 우선합니다.

| 계층 | 역할(`*_model_id` 키) |
| --- | --- |
| `default` | `fixing.fixing_model_id`, `processing.graph_extraction.extraction_model_id`, `processing.gleaning.graph_refinement_model_id`, `processing.claim_extraction.extraction_model_id`, `graph.community_detection.report_generation.report_generation_model_id`, `search.{entity_extraction,context_building,answer_generation}_model_id`, `evaluation.evaluation_model_id` |
| `fast` | `processing.chunking.chunking_model_id`, `processing.translation.translation_model_id`, `processing.graph_extraction.description_summarization.summary_model_id`, `search.{translation,strategy_selection}_model_id`, `search.global_search.{community_relevance,map_reduce,map}_model_id`, `search.drift_search.{query_refinement,keyword_expansion,convergence_assessment,primer}_model_id` |

```yaml
aws:
  bedrock:
    default_model_id: "openai.gpt-6-sol"    # every default-tier role
search:
  answer_generation_model_id: "anthropic.claude-opus-5-5"  # one role pinned
```

선별된 모델 전체 목록, Claude 세대별·OpenAI GPT 모델별 요청 차이, 프롬프트 캐싱,
패키지가 모르는 모델을 기술하는 방법(`aws.bedrock.model_overrides`)은
[모델 카탈로그](./models.ko.md)에 있습니다. 임베딩 모델 ID와 재순위화 모델 ID는
정해진 목록에서만 고릅니다(`amazon.titan-embed-text-v2:0`, `amazon.titan-embed-text-v1`,
`cohere.embed-v4:0`, `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`;
`cohere.rerank-v3-5:0`, `amazon.rerank-v1:0`). 그 밖의 ID는 설정 검증에서 실패합니다.

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

### 2.2 `fixing` — 형식이 잘못된 모델 출력 자동 복구

| 키 | 기본값 | 역할과 변경 시점 |
|---|---|---|
| `fixing.enabled` | `true` | 구조화된 출력을 쓰는 단계에서 모델이 형식이 잘못된 JSON을 반환하면, 실행을 실패시키는 대신 모델에게 복구를 요청합니다. 켜 두세요. |

### 2.3 `processing` — 동시성, 청킹, 번역, 추출

LLM 단계는 Bedrock I/O가 병목이므로 동시성을 CPU 수보다 훨씬 높게 잡을 수 있습니다.

| 키 | 기본값 | 역할과 변경 시점 |
|---|---|---|
| `processing.max_concurrency` | `20` | 배치 하나에서 동시에 보내는 LLM 호출 수입니다. Bedrock이 스로틀링하면 낮추고, 할당량에 여유가 있으면 올리세요. |
| `processing.chunk_concurrency` | `4` | 동시에 처리하는 미니 배치 청크 수입니다. Bedrock 연결 풀 크기는 `max_concurrency` × `chunk_concurrency`입니다. |
| `processing.max_attempts` | `5` | 배치 호출이 실패한 수집 LLM 항목을 하나씩 다시 호출할 때의 시도 횟수입니다. 일시적 Bedrock 오류, 호출 시간 초과, 파싱할 수 없는 출력은 재시도하고 그 밖의 오류는 바로 실패합니다. `1`이면 재시도하지 않습니다. |
| `processing.io_workers` | `64` | CLI와 체인의 동기 메서드에서 질의 경로의 블로킹 I/O(Bedrock 호출, Neptune 순회, 재순위화)를 처리하는 스레드 수입니다. Python 기본값은 `min(32, CPUs + 4)`라서 vCPU 2개인 태스크에서는 6개입니다. 비동기 호스트는 시작할 때 `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`(`unified_kg_rag.shared.utils`)를 호출합니다. |
| `processing.ignore_errors` | `false` | LLM 단계가 실패한 항목을 건너뛰고 실행을 계속합니다. |
| `processing.deduplicate` | `false` | 추출 전에 중복 문서를 제거합니다. |
| `processing.resolution_method` | `"minhash"` | 엔티티 해석 방식입니다. `minhash` 또는 `sequence_matcher`입니다. |
| `processing.similarity_threshold` | `0.6` | 엔티티 해석의 퍼지 매칭 임계값입니다. 서로 다른 엔티티가 합쳐지면 올리세요. |
| `processing.document_parsing.source_directory` | `"source"` | 라이브러리 호출용 기본값입니다. `run-ingestion`에는 `--source-directory`(또는 `GRAPHRAG_SOURCE_DIRECTORY`)가 반드시 필요합니다. |
| `processing.document_parsing.target_directory` | `null` | 파싱한 문서를 확인용 `<stem>.json`으로 내보냅니다(`--target-directory`와 같음). 소스 디렉터리는 지정할 수 없습니다. |
| `processing.document_parsing.index_value` | `null` | 실행이 쓰는 인덱스 접미사입니다. 모든 OpenSearch 인덱스와 Neptune 레이블 이름이 `<prefix>-<index_value>`가 됩니다(`null`이면 `default`). 이 코퍼스에 질의할 때는 같은 값을 `run-rag`/`run-eval --suffix`로 넘깁니다. `run-ingestion`에는 이 값을 지정하는 플래그가 없으므로 테넌트나 버전마다 설정 파일을 따로 두세요. |
| `processing.document_parsing.source_scope` | `null` | 증분 삭제에 쓰는 코퍼스 식별값입니다. 실행은 자기 인덱스 접미사와 소스 범위에 속한 레지스트리 문서만 삭제합니다. `null`이면 해석된 소스 디렉터리 경로를 쓰며, 컨테이너 엔트리포인트는 S3 URI로 설정합니다(`GRAPHRAG_SOURCE_SCOPE`). §5를 참고하세요. |
| `processing.chunking.chunker_type` | `"intelligent"` | `intelligent`는 LLM이 의미 경계를 고르고, `simple`은 크기로 나눕니다. |
| `processing.chunking.min_chunk_size` | `1000` | 최소 청크 크기(문자 수)입니다. 이보다 짧은 조각은 이웃 청크에 합칩니다. |
| `processing.chunking.max_chunk_size` | `8000` | 최대 청크 크기(문자 수)입니다. 임베딩과 재순위화 입력 한도 안에 들어야 합니다. 크기는 토큰이 아니라 문자 수입니다. 영어는 평균 약 4자가 1토큰이지만 한자·한글·가나는 1자가 약 1토큰이므로, CJK 8,000자는 약 8K 토큰입니다. Titan Text Embeddings V2(8,192)에는 들어가지만 Cohere Rerank 3.5의 문서 한도(4,096 토큰)는 넘으므로, 재순위화를 쓰는 CJK 코퍼스는 크기를 절반 정도로 줄이세요. |
| `processing.chunking.chunk_overlap` | `500` | 청크 사이의 겹침(문자 수)입니다. |
| `processing.chunking.fallback_chunk_size` | `4800` | 크기 기반 분할기(`simple`, 또는 intelligent 청킹이 실패할 때)의 목표 크기입니다. 문단, 줄, CJK 문장 종결 부호(`。．｡！？；`) 뒤, 공백 순으로 나눕니다. |
| `processing.translation.enabled` | `true` | 번역 단계를 실행합니다. 원문 언어와 대상 언어가 같고 추가 대상 언어가 없으면 아무 일도 하지 않습니다(LLM 비용 0). |
| `processing.translation.source_language` | `"en"` | 코퍼스의 주 언어입니다. 번역 생략 여부 판단에만 씁니다. |
| `processing.translation.target_language` | `"en"` | 코퍼스를 번역할 언어입니다(§3 다국어 수집 참고). 엔티티와 관계 설명도 이 언어로 작성합니다. |
| `processing.translation.additional_target_languages` | `null` | 추가로 번역할 대상 언어 목록입니다. 언어마다 해당 언어의 분석기를 쓰는 `translated_text_<language>` 필드에 인덱싱하고 어휘 검색에 포함합니다(질의별 `RAGInput.target_language`와 함께 쓰면 유용). 추출, 임베딩, 답변 컨텍스트는 `target_language`만 씁니다. 언어 하나마다 청크당 LLM 호출이 하나씩 늘어납니다. |
| `processing.graph_extraction.entity_types` | 범용 유형 7개 | 추출 프롬프트에 넣는 `"LABEL: description"` 항목입니다. 도메인 적응에 가장 효과가 큰 설정입니다(§9). 빈 목록이면 모델이 유형을 고릅니다. |
| `processing.graph_extraction.max_entities_per_chunk` | `50` | 청크당 엔티티 상한입니다(관계는 `max_relationships_per_chunk`, 역시 `50`). |
| `processing.graph_extraction.entity_confidence_threshold` | `0.0` | 신뢰도가 이 값보다 낮은 엔티티를 버립니다. `0.0`이면 모두 유지합니다. |
| `processing.graph_extraction.description_summarization.enabled` | `true` | 병합된 설명이 `force_summary_threshold_tokens`(`600`)보다 길면 LLM으로 다시 요약합니다. |
| `processing.graph_extraction.entity_grounding.enabled` | `false` | 환각 방지 장치입니다. 원문 그대로의 `source_text` 구간이 청크에 없는 엔티티와 관계를 버리거나, `action: "penalize"`이면 가중치를 낮춥니다. 문장 부호는 무시하며, 한자·한글·가나 구간은 문자 바이그램으로 비교하므로 한국어 조사가 달라져도 원문에 있는 것으로 봅니다. gleaning이 추가한 항목에도 적용합니다. |
| `processing.gleaning.enabled` | `true` | 놓친 엔티티와 관계를 찾는 추가 추출 패스입니다. |
| `processing.gleaning.max_rounds` | `3` | 텍스트 단위당 gleaning 횟수입니다. 다음 회차에는 직전 응답이 새 엔티티나 관계를 추가한 단위만 다시 보내므로, 응답에 새 항목이 없으면 그 단위는 바로 멈춥니다. `1`은 MS GraphRAG 기본값과 같습니다. |
| `processing.claim_extraction.enabled` | `false` | 주장을 추출합니다(텍스트 단위마다 LLM 호출 1회 추가). 켜면 `local` 검색이 일치하는 주장을 컨텍스트에 넣고 `simple` 검색이 주장 인덱스도 검색합니다. |

### 2.4 `graph` — 분석, 커뮤니티 탐지, 시각화

| 키 | 기본값 | 역할과 변경 시점 |
|---|---|---|
| `graph.community_detection.enabled` | `true` | Leiden 군집화와 커뮤니티 리포트 생성입니다. GraphRAG `global`/`drift`에 필요합니다. LightRAG만 쓰는 가벼운 수집이라면 `false`로 두세요. |
| `graph.community_detection.auto_resolution` | `false` | `true`이면 계층마다 `auto_resolution_candidates`를 차례로 시험해 모듈성이 가장 높은 해상도를 고릅니다. 그렇지 않으면 `resolution`(`1.0`)을 씁니다. |
| `graph.community_detection.auto_resolution_max_nodes` | `10000` | 노드 수가 이보다 많으면 해상도 탐색을 건너뛰고 `resolution`을 씁니다. |
| `graph.community_detection.max_levels` | `5` | 커뮤니티 계층의 최대 깊이입니다. |
| `graph.community_detection.min_community_size` | `3` | 이보다 작은 커뮤니티는 이웃 커뮤니티에 합칩니다. |
| `graph.community_detection.report_generation.max_report_context_tokens` | `4000` | 리포트 프롬프트 하나에 넣는 엔티티·관계 컨텍스트의 토큰 예산입니다. |
| `graph.community_detection.report_generation.content_length` | `"medium"` | 리포트 길이입니다. `short`, `medium`, `long` 중 하나입니다. |
| `graph.analysis.centrality.calculate_betweenness` | `true` | 매개 중심성을 계산합니다. 노드가 `betweenness_auto_sample_threshold`(`2000`)보다 많은 그래프에서는 정확히 계산하지 않고 샘플링합니다. |
| `graph.visualization.enabled` | `true` | 수집 중에 시각화 데이터를 내보냅니다. |
| `graph.visualization.outputs_directory` | `null` | 지정하지 않으면 `<cache dir>/<pipeline_id>/visualization`에 기록하므로 데이터가 캐시와 함께 S3로 동기화됩니다. |
| `graph.visualization.layout_method` | `"umap"` | `umap`, `tsne`, `pca` 중 하나입니다. |
| `graph.visualization.interactive.max_nodes` | `2000` | `interactive_graph.html`에 남길 차수 기준 상위 N개 노드입니다. `0`이나 `null`이면 제한하지 않습니다. `interactive`를 지정하면 기본 딕셔너리가 통째로 대체되므로, physics를 끈 상태로 두려면 `physics_enabled: false`도 함께 지정하세요. |

### 2.5 `indexing` — OpenSearch와 Neptune 쓰기

| 키 | 기본값 | 역할과 변경 시점 |
|---|---|---|
| `indexing.reset` | `false` | 인덱싱 전에 기존 인덱싱 데이터를 지웁니다. |
| `indexing.additional_suffix` | `null` | 모든 OpenSearch 인덱스 이름과 Neptune 레이블에서 실행 접미사 뒤에 붙습니다(`<prefix>-<suffix>-<additional_suffix>`). `<suffix>`는 수집 시 `processing.document_parsing.index_value`, 질의 시 `--suffix`이며 지정하지 않으면 둘 다 `default`입니다. 수집과 질의에 같은 값을 지정하세요. 버전별·테넌트별 분리에 씁니다. |
| `indexing.cross_run_merge` | `true` | 증분 실행에서 변경분을 기존 그래프 상태에 덮어쓰지 않고 합칩니다. 그래서 바뀌지 않은 문서와 공유하는 엔티티의 계보가 유지됩니다(§5). `false`이면 덮어씁니다. |
| `indexing.cross_run_fuzzy_merge` | `false` | `cross_run_merge`에 엔티티 이름 퍼지 매칭을 더합니다. |
| `indexing.max_failure_rate` | `0.2` | 인덱스 유형별 쓰기 실패율이 이 값을 넘으면 인덱싱 단계가 실패합니다. `1.0`이면 부분 실패 검사를 끕니다. |
| `indexing.max_document_failures` | `3` | 증분 실행에서 내용이 바뀌지 않은 문서가 연속으로 이 횟수만큼 FAILED로 기록되면 더는 재시도하지 않습니다(§5). |
| `indexing.opensearch.embedding_model_id` | `"amazon.titan-embed-text-v2:0"` | 임베딩 모델이며, 정해진 목록에서 고릅니다(모델 선택 참고 사항 참고). 바꾸면 다시 인덱싱해야 합니다. |
| `indexing.opensearch.build_relationship_vector_index` | `true` | LightRAG `mix`/`hybrid`용 관계 벡터 인덱스입니다. GraphRAG만 쓰는 배포라면 `false`로 두세요. |
| `indexing.opensearch.persist_embedding_cache` | `false` | 임베딩 캐시를 S3에 저장해, 바뀌지 않은 텍스트를 실행마다 다시 임베딩하지 않습니다. `aws.s3.bucket_name`이 필요합니다. |
| `indexing.opensearch.language_analyzers` | `{en: english, ko: nori}` | 언어 코드별 텍스트 분석기입니다. 목록에 없는 언어는 `default_analyzer`(`standard`)를 씁니다. |
| `indexing.opensearch.vector_search.engine` | `"lucene"` | HNSW 엔진입니다. `lucene`은 1024차원까지 `cosinesimil`을 지원합니다. `faiss`는 템플릿을 참고하세요. |
| `indexing.opensearch.index_settings.refresh_interval` | `"1s"` | 대량 적재를 빠르게 하려면 늘리거나 `"-1"`로 두고, 실제 질의 전에 되돌리세요. |
| `indexing.neptune.batch_size` | `100` | Neptune 쓰기 배치당 항목 수입니다. |
| `indexing.neptune.index_concurrency` | `1` | 동시에 보내는 쓰기 배치 수입니다. `aws.neptune.pool_size`가 이보다 작으면 Gremlin 연결 풀이 이 값에 맞춰 늘어납니다. |
| `indexing.neptune.max_attempts` | `4` | 첫 시도를 포함한 Neptune 쓰기당 시도 횟수입니다. 실패하면 `retry_delay_seconds`(`2`)에서 시작하는 지터 포함 지수 백오프로 재시도합니다. 단, 재시도로 해결되지 않는 오류(잘못된 쿼리, 액세스 거부, 잘못된 파라미터)는 첫 시도에서 실패합니다. `1`이면 재시도하지 않습니다. |
| `indexing.neptune.max_hops` | `3` | 검색 시점의 이웃 확장 깊이입니다. |
| `indexing.neptune.property_max_length` | `4000` | Neptune 속성 값당 최대 문자 수입니다. 다시 요약되지 않는 가장 긴 설명보다 크게 두세요(요약은 600토큰, 약 2,400자를 넘으면 실행). 다시 수집해야 반영됩니다. |
| `indexing.neptune.entity_importance_source` | `"rank"` | 그래프 확장 관련도에 쓰는 엔티티 중요도입니다. `rank`(인덱싱된 엔티티 rank), `degree`(질의 시점의 간선 수), `none`(모든 엔티티에 중립값 0.5) 중 하나입니다. |
| `indexing.neptune.traversal_fetch_multiplier` | `3` | 그래프 확장이 결과 폭의 이 배수만큼 가져와 순위를 매긴 뒤 자릅니다. `1`이면 순회 순서대로 자릅니다. |

### 2.6 `search` — 검색, 융합, 재순위화, 전략별 설정

| 키 | 기본값 | 역할과 변경 시점 |
|---|---|---|
| `search.auto_routable_strategies` | `["local", "mix", "global", "drift"]` | `auto` 라우터가 고를 수 있는 전략입니다. 라우터 프롬프트는 이 전략들만 이 순서대로 설명합니다. `auto` 자체는 넣을 수 없습니다. 어떤 전략이든 직접 지정할 수는 있습니다. |
| `search.hybrid.lexical_weight` | `0.5` | OpenSearch 하이브리드 파이프라인의 어휘 검색 가중치입니다(벡터는 `vector_weight`, 역시 `0.5`). |
| `search.fusion.method` | `"rrf"` | `rrf`(reciprocal rank fusion) 또는 `weighted`입니다. |
| `search.fusion.rrf_k` | `60` | RRF 상수 `k`입니다. |
| `search.fusion.fusion_weights` | 버킷마다 `1.0` | 소스 버킷별 가중치입니다. `rrf`와 `weighted` 모두에서 각 버킷의 기여도에 곱합니다. |
| `search.fusion.diversity_lambda` | `0.5` | MMR 균형값입니다. `1.0`은 관련도만, `0.0`은 다양성을 최대로 반영합니다. |
| `search.reranking.enabled` | `true` | 융합 결과를 `rerank_model_id`(`cohere.rerank-v3-5:0`)로 재순위화합니다. |
| `search.reranking.top_k` | `100` | 재순위화 모델에 보내는 후보 수입니다. |
| `search.lightrag_search.kg_stream_top_k` | `40` | LightRAG 엔티티·관계 벡터 질의의 폭입니다(요청의 `top_k`에 대한 하한). |
| `search.lightrag_search.chunk_stream_top_k` | `20` | LightRAG 청크 스트림의 폭입니다. |
| `search.lightrag_search.enable_graph_expansion` | `false` | `mix`/`hybrid`에서 일치한 항목을 Neptune으로 추가 확장합니다. |
| `search.global_search.max_communities` | `10` | `global` 검색이 살펴보는 커뮤니티 리포트 수입니다. |
| `search.global_search.map_batch_size` | `5` | map 단계 LLM 호출 하나에 넣는 리포트 수입니다. 리포트가 길면 낮추세요. |
| `search.global_search.max_map_reduce_tokens` | `8000` | reduce 단계에 넣는, 순위를 매긴 핵심 내용의 토큰 예산입니다. |
| `search.global_search.reduce_with_llm` | `false` | `true`이면 reduce LLM이 묶은 핵심 내용을 먼저 요약하고 답변 모델이 이를 다시 씁니다(LLM 호출 1회 추가). `false`이면 핵심 내용을 답변 모델에 바로 넘깁니다. |
| `search.global_search.reserve_report_slots` | `true` | 융합 슬롯 `max_communities`개를 커뮤니티 리포트용으로 예약하고, 텍스트 단위는 `text_unit_slots`개로 제한합니다. `false`이면 리포트와 청크를 한 번의 `top_k` 기준으로 자릅니다. |
| `search.global_search.text_unit_slots` | `null` | 예약한 리포트 슬롯과 함께 두는 텍스트 단위 슬롯 수입니다. `null`이면 질의의 `top_k`입니다. |
| `search.local_search.entity_frequency_threshold` | `20` | 그래프 확장으로 얻은 엔티티 중 이보다 많은 텍스트 단위에 나오는 것(너무 일반적인 것)을 버립니다. |
| `search.local_search.include_bridge_relationships` | `true` | 확장된 엔티티에 연결된 관계도 가져오며, 검색된 두 엔티티를 잇는 간선(다중 홉 연결)을 먼저 둡니다. 관계 인덱스가 필요합니다. `false`이면 관계 벡터 질의만 씁니다. |
| `search.drift_search.max_iterations` | `3` | DRIFT 반복 횟수 상한입니다. |
| `search.drift_search.enable_primer` | `false` | MS GraphRAG의 primer → follow-up 흐름을 씁니다(처음에 LLM 호출 1회 추가). |
| `search.drift_search.enable_llm_convergence` | `false` | 반복마다 LLM으로 수렴 여부를 확인합니다(반복당 호출 1회 추가). |
| `search.token_manager.max_context_tokens` | `30000` | 답변 프롬프트에 넣는 검색 컨텍스트의 토큰 예산입니다(아래 참고 사항 참고). |
| `search.token_manager.context_window_headroom_ratio` | `0.1` | 예산을 도출할 때(`max_context_tokens: null`) 남겨 두는 컨텍스트 창 비율입니다. |

섹션 유형별 비율(`search.token_manager.type_budgets`)과 `local` 검색의 유형별 슬롯
할당(`search.local_search.type_quota`)은 템플릿을 참고하세요.

> **컨텍스트 예산.** `30000`은 upstream LightRAG의 전체 컨텍스트 예산과 같습니다(MS
> GraphRAG는 12000). 1M 토큰 창에서 도출한 예산(약 785K)은 한도에 닿는 일이 없어서
> 유형별 예산도 아무것도 잘라 내지 못합니다. 이 값은 항상
> `search.answer_generation_model_id`가 출력 예약분
> (`aws.bedrock.default_max_output_tokens`)과 함께 받을 수 있는 크기로 줄이며, 줄일
> 때는 경고를 남깁니다. `null`이면 그 모델의 컨텍스트 창에서 출력 예약분과 여유 비율을
> 뺀 값으로 예산을 도출합니다. 1M이 베타인 모델에서는 `aws.bedrock.enable_1m_context`를
> 켜면 예산이 넓어집니다.

### 2.7 `memory`, `cache`, `logging`

| 키 | 기본값 | 역할과 변경 시점 |
|---|---|---|
| `memory.max_conversations` | `100` | 대화 메모리에 보관하는 대화 수입니다(§4 대화형 모드). |
| `memory.max_messages_per_conversation` | `20` | 대화당 보관하는 메시지 수입니다. |
| `memory.max_conversation_age_hours` | `168` | 대화가 쓰이지 않은 채 머물 수 있는 시간입니다. 이보다 오래된 대화는 버리고 새로 시작합니다. |
| `cache.ttl_seconds` | `86400` | 캐시 항목 TTL입니다. `null`이면 만료하지 않습니다. |
| `logging.level` | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` 중 하나입니다. |
| `logging.log_format` | `"structured"` | `structured` 또는 `plain`입니다. |
| `logging.log_to_file` | `true` | CLI가 로그를 `log_file_path`(`logs/log.txt`, 실제 파일 이름은 `log_YYYYMMDD.txt`)에도 씁니다. 상대 경로는 작업 디렉터리 기준입니다. 패키지를 라이브러리로 import하면 핸들러나 파일을 설정하지 않으며, 호스트 애플리케이션의 로깅 설정을 따릅니다. |
| `logging.library_levels` | `{langchain_aws: WARNING, botocore: WARNING, urllib3: WARNING}` | 로그가 많은 라이브러리의 로거별 수준입니다. 이 키를 지정하면 기본 맵 전체가 대체됩니다. |

**로그에 남는 내용.** `INFO` 이상에서 패키지는 사용자 텍스트와 코퍼스 텍스트 대신
길이, 개수, ID, 짧은 해시만 기록합니다(예: `query: len=42 sha=1a2b3c4d`). 해시는
항상 같은 값이 나오므로 질의 내용을 드러내지 않고도 같은 질의를 여러 레코드에서
따라갈 수 있습니다. 질의 원문, 다시 쓴 질의(DRIFT와 번역), 엔티티 이름, 모델 원본
출력은 `DEBUG`에서만 기록하며, 예외 메시지에는 모델 출력을 넣지 않습니다. `DEBUG`
로그(`logging.level: DEBUG`, `LOG_LEVEL=DEBUG`, `--verbose`)에는 사용자 데이터와
코퍼스 데이터가 들어 있다고 보세요. 로그를 공유 저장소로 보내는 환경에서는 켜지
마세요. `logging.library_levels`에서 `DEBUG`로 둔 라이브러리(예: `botocore`)는 요청
본문도 기록할 수 있습니다.

### 2.8 `evaluation`

| 키 | 기본값 | 역할과 변경 시점 |
|---|---|---|
| `evaluation.enabled_evaluators` | `[langchain, ragas, answer_match, retrieval, graph_aware]` | 결정적 평가기(`answer_match`, `retrieval`, `graph_aware`)는 필요한 정답 필드가 없는 질의를 건너뜁니다(§6). 판정 모델 비용 없이 평가하려면 LLM 판정 평가기를 빼세요. |
| `evaluation.judge_effort` | `"low"` | LLM 판정 모델의 추론 강도입니다. `null`이면 판정 모델이 속한 계층의 추론 강도(기본은 `aws.bedrock.default_effort`)를 따릅니다. |
| `evaluation.ragas_timeout` | `300` | 샘플 하나의 지표 하나를 채점하는 제한 시간(초)입니다. 시간을 넘기면 NaN이 됩니다. |
| `evaluation.ragas_max_contexts` | `20` | RAGAS `context_precision`이 샘플마다 채점하는 상위 컨텍스트 수입니다(그래서 "@20"입니다). faithfulness와 context_recall은 토큰 예산 안의 전체 컨텍스트를 봅니다. `null`이면 제한하지 않습니다. |
| `evaluation.ragas_max_workers` | `8` | 동시에 실행하는 RAGAS 작업 수입니다. Bedrock이 판정 모델 호출을 스로틀링하면 낮추세요. |
| `evaluation.ragas_max_attempts` | `3` | 판정 모델 호출당 총 시도 횟수입니다. |
| `evaluation.max_context_tokens` | `8192` | 판정 모델에 넘기는 컨텍스트의 토큰 상한입니다. |
| `evaluation.retrieval_k` | `5` | `retrieval` 평가기의 hit@k / recall@k 기준값입니다. |
| `evaluation.outputs_directory` | `"outputs/evaluation"` | 결과를 쓰는 위치입니다. |

지표 목록(`langchain_metrics`, `ragas_metrics`)은 템플릿을 참고하세요.

### 2.9 `custom_prompts`

모든 프롬프트에는 `*_system` / `*_human` 재정의 키가 있습니다(기본값 `null` =
`unified_kg_rag/domain/prompts/`의 내장 프롬프트 사용). §9를 참고하세요. 필요한
것만 재정의하고 나머지는 `null`로 두세요. 재정의는 설정을 읽을 때 검사합니다. 알 수
없는 `{variables}`와 빠진 데이터 변수(`{input_text}` 등)는 오류이며, 중괄호를 글자
그대로 쓰려면 `{{` / `}}`로 씁니다(§9.B).

### 2.10 환경 변수로 덮어쓰기

다음 환경 변수를 지정하면 설정 파일(과 내장 기본값)보다 우선합니다. 값은 필드
타입으로 변환하지만(불리언은 `true`/`1`/`yes`/`on`을 참으로 봄) 다시 검증하지는
않습니다.

| 환경 변수 | 덮어쓰는 키 | 참고 |
|---|---|---|
| `AWS_PROFILE` | `aws.profile_name` | |
| `AWS_REGION` | `aws.region_name` | `aws.bedrock.region_name`이 `null`이면 Bedrock 호출 리전도 바뀝니다. `AWS_DEFAULT_REGION`은 읽지 않습니다. |
| `BEDROCK_REGION` | `aws.bedrock.region_name` | |
| `BEDROCK_GUARDRAIL_IDENTIFIER` | `aws.bedrock.guardrail.identifier` | |
| `NEPTUNE_ENDPOINT` | `aws.neptune.endpoint` | |
| `OPENSEARCH_ENDPOINT` | `aws.opensearch.endpoint` | |
| `OPENSEARCH_USERNAME` | `aws.opensearch.username` | `aws.opensearch.use_iam: false`일 때의 기본 인증입니다. |
| `OPENSEARCH_PASSWORD` | `aws.opensearch.password` | 로그에는 가려서 표시합니다. |
| `S3_BUCKET_NAME` | `aws.s3.bucket_name` | |
| `GRAPHRAG_DOC_STATUS_TABLE` | `aws.dynamodb.table_name` | |
| `GRAPHRAG_DOC_STATUS_CREATE_TABLE` | `aws.dynamodb.create_table_if_missing` | |
| `LOG_LEVEL` | `logging.level` | |
| `LOG_FORMAT` | `logging.log_format` | |
| `LOG_TO_FILE` | `logging.log_to_file` | |
| `LOG_FILE_PATH` | `logging.log_file_path` | |

`run-ingestion`은 `GRAPHRAG_SOURCE_DIRECTORY`와 `GRAPHRAG_PIPELINE_ID`도
`--source-directory`와 `--pipeline-id`의 기본값으로 읽습니다. 플래그를 직접
지정하면 플래그가 우선합니다.

> **의도하지 않은 `AWS_REGION`이 설정 파일보다 우선합니다.** SSO 자격 증명 도우미,
> CloudShell, 셸 프로필은 `AWS_REGION`을 내보내는 경우가 많습니다. 이 값은
> `aws.region_name`을 아무 표시 없이 대체하므로, 실행이 Neptune과
> OpenSearch(`aws.bedrock.region_name`을 지정하지 않았다면 Bedrock도)를 엉뚱한
> 리전에서 찾게 됩니다. 실행 전에 `env | grep -E '^(AWS_REGION|BEDROCK_REGION)='`로
> 확인하고, 찾은 값은 해제하거나 바로잡으세요.

> **LangSmith 추적은 내용을 업로드합니다.** `LANGSMITH_TRACING=true`(또는 구형
> 변수 `LANGCHAIN_TRACING_V2=true`)가 설정되어 있으면 LangChain은 추적한 실행을 모두
> LangSmith로 보냅니다. 여기에는 프롬프트, 검색된 문서 본문, 모델 출력이 들어
> 있습니다. CLI는 추적이 켜져 있으면 시작할 때 WARNING을 남기지만, 의도한 설정일 수
> 있으므로 끄지는 않습니다. 기밀 코퍼스로 실행하기 전에는 이 변수를 해제하세요. 이
> 경고는 `setup_logging`이 남기므로, 이 함수를 호출하지 않는 라이브러리 코드에서는
> 나오지 않습니다.

CLI는 python-dotenv로 `.env` 파일도 읽습니다. 환경에 이미 설정된 변수가 `.env`보다
우선합니다. `.env`는 현재 디렉터리가 아니라 패키지 위치에서 위쪽으로 올라가며
찾으므로, 소스 체크아웃에서는 저장소 루트에 두세요.

CDK compute 스택은 `AWS_REGION`, `BEDROCK_REGION`,
`NEPTUNE_ENDPOINT`, `OPENSEARCH_ENDPOINT`, `S3_BUCKET_NAME`,
`GRAPHRAG_DOC_STATUS_TABLE`, `GRAPHRAG_DOC_STATUS_CREATE_TABLE=false`(테이블은
IaC가 관리하며 태스크 역할에는 테이블 생성 권한이 없음), `LOG_FORMAT`을 주입하고,
가드레일을 배포했다면 `BEDROCK_GUARDRAIL_IDENTIFIER`도 주입합니다. 증분 인덱싱을
쓰려면 설정 파일에 `aws.dynamodb.enabled: true`가 있어야 합니다.

---

## 3. 수집 (`run-ingestion`)

수집은 문서 디렉터리를 OpenSearch와 Neptune에 인덱싱된 지식 그래프로 바꿉니다.

### CLI 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--source-directory` | `$GRAPHRAG_SOURCE_DIRECTORY` | 소스 문서 디렉터리입니다. 실행에 필요하며, 플래그를 생략하면 환경 변수 `GRAPHRAG_SOURCE_DIRECTORY`를 씁니다. |
| `--target-directory` | 없음(내보내지 않음) | 파싱한 문서를 확인용 JSON으로 내보낼 위치(소스 디렉터리는 지정 불가) |
| `--cache-directory` | `cache` | 파이프라인 캐시와 중간 결과 |
| `--force-rebuild` | off | 기존 캐시를 모두 무시하고 처음부터 다시 구축 |
| `--s3-sync` | off | 캐시를 S3에 동기화(`--s3-bucket-name` 필요) |
| `--s3-bucket-name` | — | 캐시 동기화용 S3 버킷 |
| `--s3-prefix` | `pipeline-runs` | 캐시 파일의 S3 키 접두사 |
| `--pipeline-id` | `$GRAPHRAG_PIPELINE_ID` | 재개하거나 검사할 기존 실행입니다. 플래그를 생략하면 환경 변수 `GRAPHRAG_PIPELINE_ID`를 씁니다. 이 ID가 실행의 캐시 디렉터리와 S3 접두사 이름이 되므로 영문 소문자, 숫자, 하이픈, 밑줄만 쓸 수 있습니다. |
| `--resume-from-stage` | — | 재개할 단계(`--pipeline-id` 필요) |
| `--verify-metadata` | off | 파이프라인 메타데이터 무결성 확인(`--pipeline-id` 필요). 손상되었으면 0이 아닌 코드로 종료 |
| `--repair-metadata` | off | 메타데이터 복구 시도(`--pipeline-id` 필요). 복구에 실패하면 0이 아닌 코드로 종료 |
| `--continue-on-error` | off | 단계에서 오류가 나도 계속 진행 |
| `--enabled-stages` | 전체 | 실행할 단계 목록(쉼표로 구분) |
| `--metrics-sink` | `none` | `none` 또는 `cloudwatch`(CloudWatch EMF — Embedded Metric Format — 지표를 차원 없는 시계열로 stdout에 출력. `pipeline_id`는 차원이 아니라 로그 속성으로 기록하므로 실행마다 지표 시계열이 늘지 않음) |
| `--config-path` | — | `config.yaml` 경로 |

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
7. **`graph_resolution`** — 중복 엔티티와 관계를 퍼지 매칭으로 찾아 병합합니다.
8. **`claim_extraction`** — 선택 단계입니다. 사실 주장을 추출합니다(기본은 꺼짐).
9. **`claim_resolution`** — 선택 단계입니다. 추출한 주장의 중복을 제거합니다.
10. **`graph_analysis`** — 중심성 지표와 그래프 통계를 계산합니다.
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
파이프라인이 처음으로 실패했거나 완료되지 않은 단계를 자동으로 찾습니다.
`--force-rebuild`는 캐시를 모두 버리고 처음부터 시작합니다.

**S3 동기화**는 단계 캐시를 `s3://<bucket>/<prefix>/...`에 보관하므로, 새
프로세스(예: 새 Fargate 태스크)가 끝난 단계를 다시 계산하지 않고 재개할 수 있습니다.
동기화(시작 시 다운로드나 마지막 업로드)가 실패하거나 일부만 되면 `CacheSyncError`로
실행이 실패하고 `run-ingestion`은 0이 아닌 코드로 종료합니다. 따라서 Step Functions로
나눠 실행할 때는 체크포인트를 잃은 단계가 실패로 표시됩니다. 새 pipeline id에서 원격
캐시가 비어 있는 것은 오류가 아닙니다. 임베딩은 따로
`indexing.opensearch.persist_embedding_cache: true`를 설정하면 바뀌지 않은 텍스트를
실행마다 다시 임베딩하지 않습니다.

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
| `--interactive`, `-i` | off | 대화형 채팅(메모리 자동 활성화) |
| `--mode` | `rag` | `rag`(답변 생성까지) 또는 `search`(검색만) |
| `--conversation-id` | — | 기존 대화 이어 가기 |
| `--use-memory` | off | 대화 메모리 활성화(대화형 모드에서는 자동) |
| `--suffix` | —(`default`) | 질의할 [인덱스 접미사](#인덱스-접미사). 코퍼스의 `processing.document_parsing.index_value`와 같아야 합니다. |
| `--enable-thinking` | off | 모델의 단계별 추론 활성화 |
| `--search-strategy` | `auto` | `auto`, `drift`, `global`, `local`, `simple`, `mix`, `hybrid`, `naive` |
| `--search-type` | `hybrid` | `hybrid`, `lexical`, `vector` |
| `--top-k` | `10` | 최대 검색 결과 수 |
| `--retrieval-multiplier` | `1` | 검색 깊이 배수 |
| `--disable-query-processing` | off | 번역과 엔티티 추출 건너뛰기 |
| `--filters` | — | `key:value` 속성 필터(공백으로 구분) |
| `--output-format` | `text` | `text` 또는 `json` |
| `--verbose`, `-v` | off | 질의 처리 정보, 출처, 지표 표시 |
| `--config-path` | — | `config.yaml` 경로 |

### 검색 전략 선택

**방법론 선택.** GraphRAG 전략은 코퍼스 전체를 *요약하고 주제별로 종합*하는 데
강합니다(커뮤니티 리포트가 코퍼스 전체를 다룸). LightRAG 전략은 더 빠르고 *이중 레벨
키워드* 검색을 중심으로 동작하므로 키워드 중심 조회와 저비용 기준선에 적합합니다. 두
방법론 모두 같은 하이브리드 스코어러(BM25 어휘 검색 + 벡터 의미 검색 + 그래프 순회 +
RRF + Bedrock 재순위화)를 거치며, 검색 알고리즘만 다릅니다.

**GraphRAG(커뮤니티 요약):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `simple` | 빠른 사실 조회, 단순한 질문 | OpenSearch 벡터·키워드 직접 검색이며 그래프 순회가 없습니다. 가장 빠릅니다. 주장 추출이 켜져 있으면 주장 인덱스도 포함합니다. |
| `local` | 특정 엔티티나 개념에 관한 자세한 질문 | 질의에서 엔티티를 추출 → Neptune 그래프 순회로 이웃과 관계를 찾음 → 벡터·키워드 결과와 결합합니다. 켜져 있으면 주장(covariate)을 넣습니다. |
| `global` | "주요 주제는 무엇인가" 같은 넓은 주제 질문 | 커뮤니티 리포트를 쓰며, 동적으로 고른 커뮤니티에 map-reduce를 적용합니다. 상위 수준 종합에 가장 적합합니다. |
| `drift` | 탐색이 필요한 복잡하고 여러 측면을 가진 질문 | 질의를 반복해서 정제·확장하며, 회차마다 수렴 여부를 감지합니다. |
| `auto` | 어떤 전략이 맞는지 모를 때, 일반 용도(기본값) | LLM 라우터(`search.strategy_selection_model_id`)가 질의를 보고 `search.auto_routable_strategies`(기본 local, mix, global, drift) 중 가장 알맞은 전략을 고릅니다. |

**LightRAG(이중 레벨 키워드):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `mix` | 일반적인 LightRAG 사용, 그래프와 청크의 균형 | 저수준 키워드 → 엔티티 인덱스, 고수준 키워드 → 관계 인덱스, 1홉 연결 관계·끝점 엔티티 확장(Neptune 다중 홉 확장은 `search.lightrag_search.enable_graph_expansion`으로 선택)에 **더해** naive 벡터 청크 검색 결과를 섞습니다. |
| `hybrid` | 키워드 중심의 그래프 질문 | `mix`와 같지만 naive 청크 결과를 섞지 않습니다. |
| `naive` | 빠른 기준선, 비교 평가 | 그래프 없이 벡터 청크 검색만 합니다. LightRAG 기준선입니다. |

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
run-rag --query "..." --filters attr_category:research type:PERSON --config-path config.yaml
```

필터는 OpenSearch에서는 `term`/`terms`/`range` 절로, Neptune에서는 `has` 단계로
변환됩니다. 저장소마다 해당 인덱서가 쓰는 필드를 받습니다.

| 저장소 | 필터 가능한 필드 |
|---|---|
| 텍스트 단위 | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.<path>` |
| 엔티티 | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `attr_<key>`, `attributes.<path>` |
| 관계 | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids` |
| 주장 | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `description`, `source_text` |
| 커뮤니티 리포트 | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `attr_<key>`, `attributes.<path>` |
| Neptune 엔티티 정점 | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `attr_<key>`(있는 경우) |
| Neptune 커뮤니티 정점 | `id`, `name`, `level`, `parent`, `size`, `period`, `children` |

OpenSearch에서 `attr_<key>`는 문서 속성입니다. 문서의 `filters` 메타데이터 항목
`<key>`가 `attr_<key>`로 인덱싱됩니다. Neptune 엔티티 정점에서 `attr_<key>`는
엔티티 자체에서 추출한 속성입니다(예: `attr_role`). 정확히 필터링하려면 정확 일치
필드(keyword 또는 숫자)를 쓰세요. `description`처럼 분석되는 텍스트 필드에 `term`
필터를 걸면 소문자 토큰 하나와만 일치합니다. 범위 필터는 Python API에서
`{"gte": ..., "lte": ...}` 형태로 넘깁니다.

필터는 해당 필드를 선언한 저장소에만 적용됩니다. 예를 들어 `type:PERSON`은 엔티티,
주장, Neptune 엔티티를 좁히고 텍스트 단위는 필터링하지 않습니다. 두 저장소는
`attr_<key>`를 다르게 다룹니다.

- OpenSearch는 텍스트 단위, 엔티티, 커뮤니티 리포트에 `attr_<key>`와
  `attributes.<path>`를 엄격하게 적용합니다. 속성이 없는 문서는 제외되므로, 대개 문서
  속성이 없는 커뮤니티 리포트는 속성 필터를 건 질의에서 빠집니다. 의도한
  동작(fail-closed)입니다. 속성 필터는 보증할 수 없는 내용을 반환하지 않습니다.
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
# or continue a named session:
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

CLI에서 한 번씩 실행하며 여러 턴을 이어 가려면 같은 `--conversation-id`와
`--use-memory`를 함께 쓰세요. 메모리 한도는 `memory` 설정 섹션에 있습니다.

메모리는 턴 사이의 엔티티도 추적합니다. 사용자 메시지마다 LLM 호출
(`search.entity_extraction_model_id`)이 메시지에 나온 엔티티를 추출하고, 앞선 턴의
엔티티 중 관련도가 가장 높은 것을 다음 질의의 엔티티 초점에 더합니다. 그래서 "그
회사의 공급업체는?" 같은 후속 질문도 앞선 턴에서 언급한 엔티티를 중심으로 검색합니다.

---

## 5. 증분 인덱싱

증분(delta) 인덱싱은 전체를 다시 구축하지 않고, 마지막 실행 이후 **새로 생겼거나
바뀐** 문서만 다시 인덱싱해 운영 중인 그래프에 병합합니다.

### 켜기

```yaml
aws:
  dynamodb:
    enabled: true
    table_name: "unified-kg-rag-on-aws-doc-status"
    create_table_if_missing: true
```

이 설정을 켜면 `run-ingestion`은 실행할 때마다 DynamoDB 문서 상태 레지스트리와
**콘텐츠 해시**로 코퍼스를 비교합니다.

### 작업 흐름

- **문서 추가:** 새 파일을 소스 디렉터리에 넣고 `run-ingestion`을 다시 실행합니다.
  새 파일만 파싱·추출·인덱싱하고, 그 엔티티와 관계는 기존 그래프에 병합합니다(멱등
  `upsert_*`).
- **문서 수정:** 파일을 고치고 다시 실행합니다. 콘텐츠 해시가 바뀌므로 변경된
  문서로 처리해, 이전 버전의 산출물을 제거하고 새 버전을 다시 인덱싱합니다.
- **문서 삭제:** 소스 디렉터리에서 파일을 지우고 다시 실행합니다. 그 문서에만 있는
  **전용** 산출물(그 문서에서만 나온 엔티티와 관계. 레지스트리의 문서별 계보로
  추적)을 삭제하고, 남은 문서와 공유하는 산출물은 유지합니다.

### 삭제 범위

문서의 레지스트리 키는 인덱스 접미사(`document_parsing.index_value`와
`indexing.additional_suffix`)와 소스 디렉터리 기준 상대 경로입니다. 실행은 자기
**범위**, 곧 같은 인덱스 접미사와 같은 코퍼스 소스(`document_parsing.source_scope`.
기본값은 해석된 소스 디렉터리이며, 컨테이너 엔트리포인트는 동기화 원본인 S3 URI로
설정)에 기록된 문서만 삭제 대상으로 봅니다. 따라서 다음과 같습니다.

- 다른 테넌트(다른 `index_value`)의 실행은 두 코퍼스를 같은 로컬 디렉터리에 두더라도
  이 테넌트의 문서를 삭제하지 않습니다.
- 하위 폴더를 소스 디렉터리로 지정해 따로 실행해도 코퍼스의 나머지는 삭제되지
  않습니다(그 파일들은 별도 문서로 등록되므로, 같은 파일을 두 루트에서 한 접미사로
  인덱싱하지 마세요).
- 파싱이나 로딩에 실패한 파일은 `failed`로 보고되며, 다음 실행에서 다시 읽을 때까지
  인덱싱된 내용을 유지합니다.

로컬 코퍼스를 다른 디렉터리로 옮기면 기본 범위가 바뀝니다. 옮기기 전에 `source_scope`를
고정된 이름으로 지정하거나, 전체 재구축을 하세요.

### 실패한 문서

텍스트 단위 하나라도 번역, 그래프 추출, gleaning, 주장 추출에 실패했거나, 산출물 하나를
저장소에 쓰는 데 실패한 문서는 `FAILED`로 기록됩니다(`indexing.max_failure_rate` 이하일
때이며, 이를 넘으면 인덱싱 단계가 실패하고 아무것도 기록하지 않습니다). 다음 실행은 파일이 바뀌지 않았더라도 이 문서를 변경된 문서로 처리합니다.
실패한 실행이 쓴 내용을 제거하고 문서를 다시 처리합니다. 내용이 같은 채로
`indexing.max_document_failures`(기본 `3`)회 연속 실패한 문서는 더 재시도하지
않습니다. `FAILED` 상태와 인덱싱된 내용을 그대로 유지하고, 파일 이름이 담긴 WARNING과
함께 변경 없음으로 건너뜁니다. 파일을 수정하거나(내용이 바뀌면 횟수를 다시 셈) 한도를
올리면 다시 시도합니다.

### 문서 크기 한도

레지스트리는 문서마다 산출물 ID(텍스트 단위, 엔티티, 관계, 주장, 커뮤니티, 리포트)를
DynamoDB 항목 하나에 저장하며, 항목 크기 한도는 400KB(ID 약 10,000개)입니다. 이보다
많은 산출물을 만드는 문서는 파일 이름이 담긴 오류와 함께 인덱싱 단계를 실패시킵니다.
이런 문서는 더 작은 파일로 나누세요.

### 실행 간 병합

기본값(`indexing.cross_run_merge: true`)에서 증분 실행은 upsert 전에 변경분을 기존
그래프 상태(설명, `text_unit_ids`, 빈도, 가중치)와 *합칩니다*. 그래서 바뀌지 않은
문서와 공유하는 엔티티는 그 문서들의 설명과 청크 계보(`mix`가 따라가는 정보)를
유지합니다. 증분 실행은 먼저 영향을 받는 엔티티와 관계를 그래프에서 읽어 오며, 병합한
설명이 요약 예산을 넘으면 다시 요약합니다. `false`로 두면 해당 필드를 변경분의 값으로
덮어쓰며, 그러면 엔티티는 바뀌지 않은 문서의 청크와의 계보를 잃습니다. 이 기능에는
읽어 오기를 지원하는 그래프 어댑터가 필요합니다(지원하지 않으면 덮어쓰기로
동작합니다). 문서가 바뀌거나 삭제되면, 그 문서가 다른 문서와 공유하던 엔티티와 관계는
`text_unit_ids`에서 그 문서의 청크를 잃습니다(빈도와 가중치도 따라 바뀜). 다만 그
문서가 보탠 설명은 전체 재구축 전까지 남습니다. 저장된 항목을 읽어 오지 못하면 아무것도
덮어쓰지 않습니다. 병합 중 읽기가 실패하면 쓰기 전에 인덱싱 단계가 실패하고(문서는
기록되지 않은 채 다음에 재시도됨), 문서의 청크를 제거하던 중 읽기가 실패하면 제거
실패로 처리합니다(레지스트리 행은 재시도를 위해 남음).

---

## 6. 평가 (`run-eval`)

### CLI 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--eval-data-path` | **필수** | 질문과 정답이 담긴 JSON 파일 |
| `--outputs-directory` | `evaluation.outputs_directory` | 결과 저장 위치 |
| `--suffix` | —(`default`) | 평가 대상 [인덱스 접미사](#인덱스-접미사) |
| `--enable-thinking` | off | 모델 추론 |
| `--search-strategy` | `auto` | 각 질문에 답할 때 쓰는 전략 |
| `--search-type` | `hybrid` | 검색 방식 |
| `--top-k` | `10` | 최대 결과 수 |
| `--retrieval-multiplier` | `1` | 검색 깊이 |
| `--max-failure-rate` | `1.0` | 답변 생성에 실패한 질의의 비율, 또는 어떤 지표에서 시도한 값 중 실패한 값의 비율(`metric_outcomes`. 건너뛴 값은 세지 않음)이 이 값(0.0-1.0)을 넘으면 0이 아닌 코드로 종료합니다. 모든 질의가 실패했거나 어떤 지표의 모든 시도가 실패한 실행은 항상 0이 아닌 코드로 종료합니다 |
| `--verbose`, `-v` | off | 디버그 로깅 |
| `--config-path` | — | `config.yaml` 경로 |

### 평가기

`evaluation.enabled_evaluators`로 고릅니다. 켠 평가기를 만들 수 없거나(예: 판정
모델에 쓸 Bedrock 액세스가 없음) 평가기가 설정을 거부하면 실행이 멈춥니다.
`processing.ignore_errors: true`이면 그 평가기를 빼고 `run_manifest.dropped_evaluators`에
기록합니다. 채점 중에도 같습니다. 평가기 오류(예: 판정 모델 호출 실패)는
`ignore_errors`가 `true`가 아니면 실행을 멈추고, `true`이면 그 지표를 실패로
기록합니다. 판정 모델 응답에 쓸 수 있는 점수가 없으면 0이 아니라 항상 실패로
기록합니다. 기본으로 다섯 개가 모두 켜져 있습니다. LLM을 쓰지 않는 결정적
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
  근거 없음으로 처리되지 않습니다. 각 리포트에는 판정 모델이 본 내용이
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
  신호에 이름만 바꿔 붙이는 셈이기 때문입니다.
- **`retrieval`** — 결정적이며 LLM을 쓰지 않습니다. 답변 모델이 본 출처에 정답 문서가
  들어 있었는지 봅니다. `reference_sources`를 기준으로 `hit_at_k`, `recall_at_k`(k =
  `evaluation.retrieval_k`, 기본 5), `mrr`을 계산합니다. 텍스트 단위 출처는 파일을
  직접 가리킵니다. 엔티티, 관계, 커뮤니티 리포트 출처는 계보(`text_unit_ids`)에 있는
  텍스트 단위의 파일로 귀속하며, 체인의 문서 저장소에서 접미사마다 한 번에 일괄
  조회합니다. 커뮤니티 리포트의 계보는 커뮤니티 전체이므로 `global`/`drift` 점수는
  답변 모델이 실제로 읽은 범위의 상한입니다. 귀속할 수 있는 출처만 순위에 넣으며,
  `attributable_fraction`(귀속 가능한 출처 / 보고된 출처. 전략별 값은
  `grouped_statistics`에도 있음)은 순위 지표가 컨텍스트의 얼마만큼을 다루는지
  보여 줍니다. 일치 비교는 NFKC 정규화 후 대소문자를 구분하지 않으며(그래서 자모가
  분해된 NFD 한글 파일 이름과 전각 문자도 일치), 전체 이름으로 비교하거나 이름이 파일
  확장자(`.` + 영문자·숫자 1-5자, 영문자 하나 이상 포함)로 끝나면 확장자를 뺀 이름으로도
  비교합니다: `docs/Terms.pdf` = `terms.pdf` = `terms`. `/`는 경로처럼 보이는 값(파일
  확장자, URI 스킴, 또는 앞에 `/`, `./`, `~/`가 있는 값)에서만 디렉터리 구분자로 보므로,
  `St. Louis Cardinals`나 `AC/DC` 같은 제목은 통째로 비교합니다. 질의에
  `reference_sources`가 없거나, 출처는 있지만 귀속할 수 있는 것이 없으면 순위 지표를
  건너뜁니다. 출처를 하나도 검색하지 못한 질의는 0점(놓침)입니다.
- **`answer_match`** — 결정적이며 LLM을 쓰지 않는 답변 점수입니다. `answer`와 선택
  항목인 `metadata.answer_aliases`를 기준으로 계산해 그중 최댓값을 씁니다.
  **`answer_contains`**(정답이나 별칭이 생성된 답변에 단어 단위 구문으로 나오면 1.0)가
  대표 결정적 지표입니다. 긴 RAG 답변이 짧은 정답 구간과 똑같은 경우는 드물기 때문에,
  이 지표가 exact match보다 정답 여부를 훨씬 잘 반영합니다. 한국어 조사는 영문 단어와
  숫자를 포함한 모든 단어 뒤에서 허용하며(정답 `서울 특별시`는 `서울 특별시는`과,
  `AWS`는 `AWS는`과, `2024`는 `2024년에`와 일치), CJK 문자 옆의 띄어쓰기는 무시합니다
  (`3억 원` = `3억원`, `가나다 상사` = `가나다상사`). 단어 하나인 CJK 정답은 부분
  문자열로 비교하지만, 더 긴 숫자나 영문 단어 안에서는 일치로 보지 않습니다(`2년`은
  `12년`과 일치하지 않음). 공개 벤치마크와 비교할 수 있도록 SQuAD 방식의
  `exact_match`와 `token_f1`도 출력합니다. 텍스트는 NFKC로 정규화한 뒤 공식 SQuAD v1.1
  스크립트와 같이 정규화합니다. 소문자로 바꾸고, 문장 부호를 지우고(`1,000` = `1000`),
  영어 관사를 빼고, 공백을 하나로 합칩니다. Token F1은 공백으로 나누므로 중국어·일본어
  텍스트에서는 exact match와 같아지며, 한국어 조사(`서울은`) 때문에 두 지표 모두 실제보다
  낮게 나옵니다.

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
run-eval --eval-data-path my_eval_data.json --config-path config.yaml

run-eval --eval-data-path my_eval_data.json --outputs-directory ./results --config-path config.yaml

run-eval --eval-data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml
```

결과는 출력 디렉터리에 `evaluation_{results,reports,summary}_<timestamp>.json`으로
씁니다. 답변한 질의가 모두 같은 전략을 썼다면 파일 이름은
`..._<strategy>_<timestamp>.json`이 됩니다. 요약에는 지표마다
평균/중앙값/표준편차/최솟값/최댓값/개수(`metric_statistics`)와
채점/실패/건너뜀 개수(`metric_outcomes`)가 있고, 다음 항목도 들어 있습니다.

- `grouped_statistics` — 같은 통계를 `search_strategy`(실제로 쓴 전략. `auto`에서는
  질의마다 다름), `category`, `difficulty`별로 나눈 값입니다.
- `abstention_statistics` — 체인이 답변 대신 고정된 컨텍스트 없음 응답("I could not
  find relevant information…")을 반환한 빈도입니다. `abstained`, `answered`,
  `abstention_rate`, 같은 값의 `per_strategy`가 있고, `answerable: false` 항목에는
  `unanswerable` 블록(`total`, `correct_abstentions`, `accuracy`)이 있습니다. 답할 수
  있는 항목에서 거절은 다른 답변과 똑같이 채점합니다(보통 놓침). 각 결과에는
  `abstained`가 기록됩니다.
- `run_manifest` — CLI 인자, 모델 ID(답변 생성, 평가 판정 모델·임베딩), 패키지 버전,
  git 커밋(`git_sha`. 추적 파일이 커밋과 다르면 `git_dirty`도 기록. 패키지를 해당
  커밋을 추적하는 체크아웃에서 실행하지 않으면 둘 다 `null`), 해석된 전체 설정의
  `config_sha256`, `library_versions`(ragas, langchain*), 데이터셋(경로와 파일 sha256,
  질의 수, 파싱한 내용의 해시), UTC 타임스탬프가 있어 두 실행을 비교할 수 있습니다.
  `dropped_evaluators`도 기록합니다. `EvaluationManager.evaluate_dataset`이 이
  매니페스트를 만들므로 라이브러리 호출자도 얻을 수 있습니다(기록하려면
  `dataset_path=` / `cli_args=`를 넘기세요).

질의별 리포트에는 평가기마다 `metrics`가 있습니다. `overall_score`는 JSON 호환성을
위해 남겨 두었지만 항상 `null`입니다(서로 무관한 지표의 평균은 의미가 없음). 각
결과에는 `retrieved_source_ids`도 기록됩니다. 보고된 출처마다 순위 순으로, 그 출처가
귀속된 파일 이름 목록입니다(귀속할 수 없으면 `[]`).

---

## 7. 시각화 (`run-visualization`)

이미 내보낸 시각화 데이터 JSON으로 그림을 그리는 **독립** 렌더러입니다. 수집을 다시
실행하거나 AWS에 접근하지 **않습니다**.

`graph.visualization.enabled`가 `true`이면 `run-ingestion`의 커뮤니티 탐지 단계가
시각화를 렌더링하고 `visualization_data.json`도 `graph.visualization.outputs_directory`에
씁니다. 이 값을 지정하지 않으면(기본값) `<cache.local_directory>/<pipeline_id>/visualization/`에
쓰므로, S3 캐시 동기화를 켜면 `visualization_data.json`이 캐시와 함께 업로드됩니다(동기화는
`.json` 파일만 복사하므로 HTML은 로컬에서 `run-visualization`으로 다시 렌더링하세요).
이 파일이 `--data-path` 입력입니다. 대화형 그래프는 큰 그래프도 브라우저에서 렌더링할 수
있도록 차수 기준 상위 `interactive.max_nodes`개 노드(기본 2000)만 남깁니다. 파일에는
그래프 노드와 간선, 계산된 `layout`, 커뮤니티 계층, 중심성이 들어 있으며, 파일 크기를
줄이려고 벡터 속성(`embedding`, `*_embedding`)은 뺍니다.

레이아웃과 오류 처리 방식은 다음과 같습니다.

- `embedding_method: "none"`은 spring 레이아웃을 쓰며 Bedrock 임베딩 클라이언트를
  만들지 않습니다.
- `embedding_method: "node2vec"`은 노드마다 `name: description`을 Bedrock으로
  임베딩하고 `layout_method`로 차원을 줄입니다. 임베딩이 실패하면
  `processing.ignore_errors`가 `false`일 때 시각화 단계가 실패합니다(수집 자체는
  계속하며 실패를 로그에 남김). `ignore_errors: true`이면 ERROR를 남기고 토폴로지만
  쓰는 spring 레이아웃으로 대체하며, `visualization_data.json`에
  `"layout_degraded": true`를 기록합니다. 차원 축소가 실패해도 `ignore_errors`와
  관계없이 같은 방식으로 대체합니다(시드를 고정한 spring 레이아웃,
  `layout_degraded: true`).
- `embeddings.bedrock_model_id`는 지원하는 임베딩 모델 ID 중 하나여야 하며, 그 밖의
  값은 설정을 읽을 때 거부합니다.
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
| `--output-dir` | `visualization_outputs` | 렌더링한 파일을 쓸 위치 |
| `--renderers` | 등록된 전체 | 실행할 렌더러: `interactive`, `static` |
| `--config-path` | — | `config.yaml` 경로 |

등록된 렌더러는 **`interactive`**(pyvis)와 **`static`**(Bokeh) 두 개입니다. 설정은
`graph.visualization` 아래(`interactive.*`, `static.*`, `embedding_method`/`layout_method`)에
있습니다.

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-dir ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

---

## 8. 프롬프트 튜닝 (`run-prompt-tuning`)

디렉터리에서 문서를 표본으로 뽑고 Bedrock으로 코퍼스 특성(도메인, 언어, 페르소나,
엔티티 유형)을 분석한 뒤, 검토해서 `config.yaml`에 합칠 수 있는 도메인 맞춤
`custom_prompts` YAML 조각을 씁니다.

### CLI 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--source-directory` | **필수** | 문서 디렉터리(`.txt`/`.md`/`.markdown`과 `run-ingestion`이 파싱하는 모든 형식. 예: `.pdf`). 별칭 `--source-dir`도 받습니다 |
| `--output` | `tuned_prompts.yaml` | 출력 YAML 경로 |
| `--max-docs` | `20` | 표본으로 뽑을 최대 문서 수 |
| `--config-path` | — | `config.yaml` 경로 |

```bash
run-prompt-tuning --source-directory ./source --output tuned_prompts.yaml --config-path config.yaml
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
`community_report`, `answer_generation`, `context_building`,
`entity_extraction`, `keyword_expansion`, `query_refinement`,
`drift_primer`(`enable_primer`를 켰을 때의 DRIFT primer),
`strategy_selection`, `keywords_extraction`(LightRAG 이중 레벨),
`global_map`(global 검색 map-reduce), 그리고 프롬프트 튜닝의 `corpus_profile`
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
InvokeModelWithResponseStream과 임베딩), Neptune(`use_iam: true`일 때 connect /
SigV4), OpenSearch(설정한 인덱스 읽기·쓰기), S3(설정한 버킷), DynamoDB(증분 인덱싱을
켠 경우).

> **Bedrock 재순위화에는 별도 정책 문이 필요합니다.** 재순위화는 Rerank API를
> 호출하며, 이 API는 `bedrock:Rerank`를 `InvokeModel`과 다른 리소스 형태로
> 인가합니다. 파운데이션 모델 ARN이나 추론 프로파일 ARN으로 범위를 좁힌 정책 문으로는
> 거부됩니다. CDK 태스크 역할(`iac/stacks/compute_stack.py`)처럼 별도 정책 문에서
> `Resource: "*"`에 `bedrock:Rerank`를 허용하세요. 이 권한이 없으면 질의마다
> `Reranking failed: ...` ERROR가 남고, 재순위화하지 않은 융합 결과를 반환합니다.
> 재순위화는 기본으로 켜져 있습니다(`search.reranking.enabled`). 이 권한을 줄 수
> 없다면 `false`로 설정하세요.
>
> 모델 해석기는 `bedrock:ListInferenceProfiles`도 호출합니다. 이 권한은
> `Resource: "*"`를 쓰는 계정 수준 읽기 권한입니다. 가드레일을 쓰려면
> `bedrock:ApplyGuardrail`이 필요합니다.

### 자주 나는 오류

다시 수집하기, 실패한 문서, 캐시 접두사, 시작 시 엔드포인트 확인은
[운영 런북](./operations.ko.md)에서 더 자세히 다룹니다.


- **`--source-directory is required`** — 이 플래그를 넘기거나
  `$GRAPHRAG_SOURCE_DIRECTORY`를 설정하세요. 메타데이터만 다루는 작업
  (`--verify-metadata` / `--repair-metadata`)에는 대신 `--pipeline-id`가 필요합니다.
- **`--s3-bucket-name must be specified for S3 sync`** — `--s3-sync`에는
  `--s3-bucket-name`이 필요합니다.
- **`--pipeline-id is required for --resume-from-stage`** — 재개하려면 이전 실행의
  파이프라인 ID가 필요합니다.
- **`Invalid stage names provided`** — §3의 단계 이름을 정확히 쓰세요(CLI가 유효한
  이름 목록을 출력합니다).
- **`Skipping N '.md' file(s) ... unsupported file type`**(남은 파일이 없으면
  `No supported source files found in '<dir>'`도) — `.md`/`.html`을 파싱하려면
  `unstructured` extra를 설치하거나(`uv sync --extra unstructured`, Python 3.11 이상)
  해당 문서를 지원 형식으로 바꾸세요.
- **`No indices found for suffix '<suffix>'`** — `run-rag`/`run-eval`이 아무것도
  수집되지 않은 접미사에 질의했습니다. 수집을 아직 실행하지 않았거나, `--suffix`가
  코퍼스를 수집할 때 쓴 `processing.document_parsing.index_value`와 다릅니다(둘 다
  기본값은 `default`).
- **`use_iam: false`에서 OpenSearch 인증 실패** — `.env`에
  `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD`가 있는지 확인하세요.
- **LightRAG `mix`/`hybrid` 결과가 없음** — 수집 중에 관계 인덱스가 만들어졌는지,
  키워드 추출이 키워드를 만들었는지 확인하세요(아주 짧은 질의는
  `raw_query_fallback_max_len` 이하일 때만 원래 질의를 씁니다).
- **파이프라인이 실행 중에 실패함** — `--pipeline-id <id>`로 다시 실행하면 실패했거나
  완료되지 않은 단계부터 재개합니다. 손상 여부는 `--verify-metadata`로 확인하고,
  `--repair-metadata`로 복구를 시도하거나 `--force-rebuild`로 처음부터 시작하세요.

### 대규모, 다국어, 이질적인 코퍼스

- **대규모 코퍼스:** `processing.max_concurrency` / `processing.chunk_concurrency`를
  올리세요(LLM 단계는 I/O가 병목). 그래프 쓰기는 `indexing.neptune.index_concurrency`를
  올리면 Gremlin 연결 풀도 함께 늘어납니다. `indexing.opensearch.persist_embedding_cache`와
  `--s3-sync`를 켜면 재실행과 여러 단계로 나눈 작업에서 다시 계산하지 않습니다.
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

CLI는 공개 클래스 두 개를 감싼 얇은 래퍼입니다. `Config`는 `get_config`로 만드세요.
`get_config`는 CLI와 같은 설정 파일·환경 변수 단계를 적용합니다(§2).

```python
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import PipelineConfig, PipelineStageStatus
from unified_kg_rag.shared import get_config

config = get_config("config.yaml")

# Ingest. PipelineConfig carries the run options the run-ingestion flags set
# (pipeline_id, cache directory, S3 sync, enabled stages, force_rebuild).
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
- `GraphRAGChain(config, mode=ChainMode.SEARCH)`는 답변을 생성하지 않고 검색 결과를
  반환합니다(`ChainMode`는 같은 모듈에 있음).
- 비동기 코드에서는 `await chain.ainvoke(...)`와 `await chain.aclose()`를 쓰세요.
  비동기 호스트는 시작할 때 이벤트 루프의 실행기 크기도 한 번 지정해야 합니다:
  `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`
  (`unified_kg_rag.shared.utils`).
- 만든 객체는 항상 닫으세요. 두 클래스 모두 처음 사용할 때 연결을 열며,
  `close()`/`aclose()`를 호출하지 않으면 가비지 컬렉션 때까지 연결이 열려 있습니다.
- 패키지를 import해도 로깅 핸들러는 설정하지 않습니다. CLI와 같은 로깅을 쓰려면
  `unified_kg_rag.shared.setup_logging(config)`를 호출하세요.

두 생성자 모두 백엔드 주입(`providers`, `retriever_builders`, `doc_status`,
`vector_indexer`, `graph_indexer`)을 받습니다.
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
- **Bedrock만 지원합니다.** 모델 호출은 Amazon Bedrock으로 갑니다. 다른 공급자를 쓰려면
  사용자 정의 모델 팩토리가 필요합니다([설계 문서 §15](./design.ko.md#15-확장-가이드)).
- **기본 제공 저장소 백엔드는 두 가지입니다.** Neptune(로컬 개발에서는 TinkerPop
  Gremlin Server)과 OpenSearch입니다. 다른 저장소를 쓰려면 사용자 정의 어댑터가
  필요합니다.
- **서빙 계층이 없습니다.** 패키지는 CLI와 Python 라이브러리를 제공하며, HTTP API,
  최종 사용자 인증, 요청 속도 제한은 없습니다.
- **필터는 접근 제어가 아닙니다**(§4). 테넌트는 인덱스 접미사로 분리하세요.
- **가벼운 수집은 쓸 수 있는 전략을 제한합니다.**
  `graph.community_detection.enabled: false`이면 `global`과 `drift`가 쓸 커뮤니티
  리포트가 없고, `indexing.opensearch.build_relationship_vector_index: false`이면
  `mix`와 `hybrid`가 고수준 키워드 검색을 쓰지 못합니다.
- **증분 인덱싱은 근사치입니다**(§5). 공유 엔티티나 관계는 바뀌거나 삭제된 문서가
  보탠 설명을 유지하며, 증분 실행은 그래프 전체를 다시 군집화하지 않고 커뮤니티를
  덧붙입니다. 둘 다 전체 재구축(`indexing.reset: true`)을 해야만 갱신됩니다. 실행마다
  문서 상태 테이블 전체를 스캔하며, 문서 하나의 산출물 ID는 400KB DynamoDB 항목 하나에
  들어가야 합니다.
- **저장소 한 세트에서는 수집을 한 번에 하나만 실행하세요.** 동시에 실행하면
  레지스트리와 저장소에서 경합이 생깁니다.
- **Markdown과 HTML**에는 선택 extra인 `unstructured`(Python 3.11 이상)가 필요하며,
  컨테이너 이미지에는 기본으로 빠져 있습니다.
- **임베딩 모델(또는 차원)을 바꾸면** 다시 인덱싱해야 합니다.

---

*함께 보기: 개요와 빠른 시작은 [README.ko.md](../README.ko.md), 아키텍처와 내부 구조는
[설계 문서](./design.ko.md)를 참고하세요.*
