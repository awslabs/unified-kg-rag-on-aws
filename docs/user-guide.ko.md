# Unified Knowledge Graph RAG on AWS — 사용자 가이드

> 🇬🇧 English version: [docs/user-guide.md](./user-guide.md)

이 문서는 **unified-kg-rag-on-aws**의 실용적인 사용 방법 가이드입니다. unified-kg-rag-on-aws는
대규모 다국어 문서 코퍼스로부터 지식 그래프를 구축하고 그 위에서 질문에 답하는
AWS 네이티브 Knowledge Graph RAG 프레임워크입니다. 두 가지 검색 방법론 —
**Microsoft GraphRAG**(커뮤니티 요약)와 **LightRAG**(이중 레벨 키워드) — 을
하나의 스택 위에 재구현하여 질의마다 선택할 수 있습니다.

- *무엇을 / 왜*와 1분 빠른 시작은 [README.md](../README.md)를 참고하세요.
- *내부 구조 / 아키텍처*(헥사고날 레이어, 포트 & 어댑터, 의존성 규칙)는
  기술 문서([docs/design.ko.md](./design.ko.md) 한국어 / [docs/design.md](./design.md)
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

이 프레임워크는 **이미 존재하는** 서비스에 연결할 뿐, 서비스를 직접 생성하지
않습니다. 준비 방법은 두 가지입니다.

- **기존 서비스 사용.** Neptune, OpenSearch, S3(및 선택적으로 DynamoDB)가 이미
  실행 중이라면, 엔드포인트를 `config.yaml`(아래 §2.1)에 기록하고 이 절을 건너뛰면
  됩니다. 설정한 모델 ID에 대해 Bedrock 모델 액세스가 활성화되어 있는지 확인하세요.
- **번들 CDK 앱으로 전체 프로비저닝.** 저장소에는 스택 전체를 명령 한 번으로
  세워 주는 선택적, Well-Architected AWS CDK 앱이 [`iac/`](../iac/README.md)에
  포함되어 있습니다 — 네트워킹(VPC + 엔드포인트), Neptune 클러스터, OpenSearch
  도메인, DynamoDB 문서-상태 테이블, S3 캐시 버킷, ECS Fargate 데이터 플레인,
  Step Functions 인제스천 파이프라인, CloudWatch 관측성, 그리고 선택적 리전 고정
  Bedrock Guardrail까지 생성합니다.

  ```bash
  cd iac
  python -m venv .venv && . .venv/bin/activate
  pip install -r requirements.txt

  cdk synth              # 미리보기 — AWS 변경/비용 없음
  cdk bootstrap          # 계정/리전당 최초 1회
  cdk deploy --all       # 과금 리소스 생성 (Neptune + OpenSearch는 시간당 과금)
  ```

  배포가 끝나면 CloudFormation 출력값의 Neptune / OpenSearch / S3 엔드포인트를
  `config.yaml`에 복사하세요(엔드포인트는 `https://` 없는 호스트 이름입니다).
  Neptune과 OpenSearch는 VPC 안에서만 접근할 수 있으므로, CLI는 배포된 태스크
  정의의 태스크처럼 VPC 안에서 실행하세요. 스택이 만든 태스크는 엔드포인트를 환경
  변수로 이미 받습니다. `cdk destroy --all`로 `dev` 프로파일을 다시 정리할 수
  있습니다. 모든 스택, `-c key=value` 옵션(VPC 재사용, 인스턴스 사이징, CMK, 삭제
  보호, cdk-nag), 프로덕션 하드닝 체크리스트는 [`iac/README.md`](../iac/README.md)를
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

선택적 추가 패키지: **Markdown(.md)** 및 **HTML(.html)** 파싱에는
`unstructured` 패키지가 필요합니다. 이 패키지가 없으면 `.pdf`, `.txt`, `.csv`,
`.json`만 파싱됩니다(파서는 `.md`/`.html`에 대해 누락된 패키지 이름을 명시하는
명확한 에러를 발생시킵니다). Python 3.11 또는 3.12에서
`uv sync --extra unstructured` 또는 `pip install -e '.[unstructured]'`로
설치하세요. 이 추가 패키지는 URL 파싱의 SSRF 취약점을 수정하고 NLTK 의존성을
제거한 `unstructured>=0.24.0`을 사용합니다. Python 3.10에서는 이 추가 패키지를
선택해도 파서가 설치되지 않습니다. PDF/TXT/CSV/JSON을 사용하거나,
Markdown/HTML 지원이 필요하면 Python 버전을 올리세요.

배포용 컨테이너 이미지(`docker/Dockerfile`)는 이 추가 패키지가 약 200MB(spaCy 등)를
더하므로 기본적으로 포함하지 않습니다. 따라서 Step Functions로 `.md`/`.html` 파일을
수집하면 "No supported files found" 오류로 실패합니다. 해당 파일을 지원 포맷으로
변환하거나, 추가 패키지를 넣어 이미지를 빌드하세요:
`docker build --build-arg UV_EXTRAS="--extra unstructured" -f docker/Dockerfile .`.
이미지에는 엔드포인트가 없는 `docker/config.yaml`이 포함되며, 엔드포인트는 CDK
compute 스택이 환경 변수로 주입합니다.

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

### 로컬 저장소 (개발용)

개발할 때는 Neptune과 OpenSearch 대신 로컬 컨테이너를 쓸 수 있습니다. 모델은
여전히 Bedrock을 호출하므로 Bedrock 접근 권한이 있는 AWS 자격증명이 필요합니다.
S3는 `--s3-sync`를 쓸 때만, DynamoDB는 증분 인덱싱을 쓸 때만 필요합니다.

```bash
docker compose -f docker/compose.local.yaml up -d --wait
uv run run-ingestion --config-path docker/config.local.yaml --source-directory ./docs-in
uv run run-rag --config-path docker/config.local.yaml --query "..."
docker compose -f docker/compose.local.yaml down -v
```

[`docker/compose.local.yaml`](../docker/compose.local.yaml)은 TinkerPop Gremlin
Server(메모리 기반 TinkerGraph)와 단일 노드 OpenSearch 2.13을 띄웁니다.
OpenSearch는 보안 플러그인을 끄고 `analysis-nori` 플러그인을 설치합니다. 한국어
매핑이 `nori`를 쓰기 때문이며, Amazon OpenSearch Service에는 기본으로 들어 있습니다.
[`docker/config.local.yaml`](../docker/config.local.yaml)은
`aws.neptune.use_ssl: false`, `use_iam: false`(SigV4 없는 `ws://`)와
`aws.opensearch.allow_anonymous: true`, `use_ssl: false`(인증 없는 `http://`)로
이 컨테이너에 연결합니다.

Bedrock 없이 저장소만 확인하려면 해싱 임베딩을 쓰는 스모크 테스트를 실행합니다:
`LOCAL_STORES=1 uv run pytest tests/integration/test_local_stores.py`.

Neptune과 다른 점: TinkerGraph는 그래프를 메모리에만 두고, 정점을 만들 때 쓴 리스트
속성은 중복 값을 그대로 저장합니다(Neptune은 한 번만 저장). compose 파일은 보안
설정을 하지 않았으며 `127.0.0.1`에만 바인딩합니다.

---

## 2. 설정

템플릿으로부터 설정 파일을 만들고 모든 CLI에 `--config-path config.yaml`로
지정하세요.

```bash
cp config-template.yaml config.yaml
```

**전체 설정 항목의 기준 문서는 `config-template.yaml`입니다.** 모든 키와 기본값,
그리고 각 항목의 역할을 주석으로 담고 있습니다. 이 절에서는 설정을 읽어 들이는
방식, 자주 조정하는 항목, 설정 파일을 덮어쓰는 환경 변수를 다룹니다. 여기에 없는
항목은 템플릿을 참고하세요.

### 설정을 읽는 순서

값은 세 단계로 정해지며, 뒤 단계가 앞 단계를 덮어씁니다.

1. **내장 기본값**: `unified_kg_rag/domain/models/config.py`의 Pydantic 모델에
   정의된 값입니다. `--config-path` 없이 CLI를 실행하면 이 기본값에 환경 변수만
   적용합니다.
2. **YAML 파일**: 지정한 키는 기본값을 대체하고, 지정하지 않은 키는 기본값을
   유지합니다. 따라서 `config.yaml`에는 바꿀 키만 적어도 됩니다. 딕셔너리 값을
   갖는 키(예: `logging.library_levels`, `graph.visualization.interactive`)는
   기본 딕셔너리와 병합되지 않고 통째로 대체됩니다.
3. **환경 변수**(§2.10): 마지막에 적용되어 앞의 두 단계를 모두 덮어씁니다.

YAML 값은 파일을 읽을 때 검증합니다. 타입이 틀리거나 지원하지 않는 값이면
`Configuration validation error: ...`로 CLI가 중단됩니다. 알 수 없는 키(오타나 새
릴리스에서 제거된 키)는 실행을 멈추지 않고 `Unknown config key '<path>' is
ignored` WARNING 로그를 남긴 뒤 버려집니다. 파일을 고치거나 업그레이드한 뒤에는
로그를 확인하세요.

이름이 바뀐 키도 계속 받아들입니다. 이전 키의 값은 새 키에 적용되고
`Config key '<old>' is deprecated; applied as '<new>: <value>'` WARNING 로그가
남습니다. 새 키도 함께 지정했다면 이전 키는 무시합니다.

| 이전 키 | 새 키 |
|---|---|
| `search.llm_retry` | `aws.bedrock.transient_retry` |
| `aws.bedrock.effort` | `aws.bedrock.default_effort`(값 그대로) |
| `processing.max_retries` | `processing.max_attempts`(값 그대로) |
| `indexing.neptune.max_retries` | `indexing.neptune.max_attempts`(값에 1을 더함. 이전 키는 첫 시도를 뺀 재시도 횟수였습니다) |
| `evaluation.ragas_max_retries` | `evaluation.ragas_max_attempts`(값 그대로) |

`max_attempts` 키는 모두 첫 시도를 포함한 총 시도 횟수이며, `1`이면 재시도하지
않습니다.

아래 표의 기본값은 내장 기본값입니다. `config-template.yaml`도 같은 값을 씁니다.

### 2.1 `aws` — 서비스 엔드포인트 & 자격증명

| 키 | 기본값 | 역할 / 바꿀 때 |
|---|---|---|
| `aws.region_name` | `"us-west-2"` | Neptune, OpenSearch, S3, DynamoDB가 있는 리전이며, `aws.bedrock.region_name`을 지정하지 않으면 Bedrock 호출도 이 리전으로 보냅니다. `AWS_REGION`이 이 값을 덮어씁니다(§2.10 참고). 기본값 리전에는 리랭크 모델 두 개를 포함한 기본 모델이 모두 있습니다. `ap-northeast-2` 같은 일부 리전에는 리랭크 모델이 없습니다. |
| `aws.profile_name` | `null` | 사용할 AWS 프로파일 이름입니다. `null`이면 기본 자격 증명 체인을 씁니다. |
| `aws.bedrock.region_name` | `null` | Bedrock 모델·임베딩·리랭크 호출을 보내는 리전이며, Guardrail도 이 리전에 있어야 합니다. `null`이면 `aws.region_name`을 씁니다. 모델 액세스를 다른 리전에서 활성화한 경우에만 지정하세요. Bedrock으로 가는 경로가 VPC 엔드포인트뿐인 프라이빗 VPC에서는 `null`로 두거나 VPC와 같은 리전을 지정하세요. |
| `aws.bedrock.enable_global_profile` | `true` | 크로스 리전(global) 추론 프로파일을 사용합니다. Claude 4.7 이후 모델과 GPT 모델은 프로파일로만 호출할 수 있으므로 켜 두세요. |
| `aws.bedrock.default_model_id` | `"anthropic.claude-sonnet-5-5"` | default 등급 역할 전체가 쓰는 모델입니다(모델 선택 주의사항 참고). |
| `aws.bedrock.fast_model_id` | `"anthropic.claude-haiku-5-5"` | fast 등급 역할 전체가 쓰는 모델입니다. |
| `aws.bedrock.default_max_output_tokens` | `16384` | 요청마다 보내는 `max_tokens`이며 모델 최대값을 넘지 않게 맞춥니다. 답변이 잘리면(`stopReason: max_tokens`) 올리고, `null`이면 모델 최대값을 보냅니다. |
| `aws.bedrock.default_effort` | `"high"` | `default_model_id` 호출의 추론 깊이입니다(adaptive thinking Claude와 GPT 모델). `low`, `medium`, `high`, `xhigh`, `max` 중 하나이며, 낮추면 비용과 지연 시간이 줄어듭니다. |
| `aws.bedrock.fast_effort` | `"low"` | `fast_model_id`가 `default_model_id`와 다를 때 fast 모델 호출의 추론 깊이입니다. 기본 fast 모델인 Claude Haiku 5.5는 adaptive 사고를 하므로 이 값이 추론 깊이를 정합니다. 추론하지 않는 fast 모델에는 효과가 없습니다. |
| `aws.bedrock.enable_1m_context` | `false` | 1M 컨텍스트가 베타인 모델에서 이를 사용합니다(추가 요금). Claude 5는 기본으로 1M입니다. |
| `aws.bedrock.model_overrides` | `{}` | 패키지가 모르는 언어 모델의 기능 정보를 지정합니다(모델 선택 주의사항 참고). 임베딩·리랭킹 모델은 정해진 목록에서만 고릅니다. |
| `aws.bedrock.guardrail.identifier` | `null` | Bedrock Guardrail ID 또는 ARN입니다. 지정하면 Guardrail이 켜집니다. |
| `aws.bedrock.guardrail.apply_to` | `"query"` | `query`는 사용자 질의 경로에만, `all`은 모든 호출에 Guardrail을 적용합니다(아래 Guardrail 참고 사항). |
| `aws.bedrock.guardrail.trace` | `false` | Guardrail trace를 출력합니다. InvokeModel 경로에서 개입을 감지하려면 필요합니다. |
| `aws.bedrock.transient_retry.max_attempts` | `5` | botocore가 재시도하지 않는 일시적 Bedrock 오류(HTTP 424 등)에 대한 호출당 시도 횟수입니다. 임베딩과 질의 시점 호출에 적용하며 `1`이면 재시도하지 않습니다. |
| `aws.neptune.endpoint` | `null` | **필수.** Neptune 클러스터 엔드포인트입니다. |
| `aws.neptune.use_iam` | `true` | Neptune 요청에 SigV4 서명을 붙입니다. |
| `aws.neptune.use_ssl` | `true` | `wss://`로 연결합니다. Neptune에는 필수이며, 로컬 Gremlin Server에서만 `use_iam: false`와 함께 `false`로 둡니다(§1 로컬 저장소). |
| `aws.neptune.pool_size` | `4` | Gremlin 연결 풀 크기입니다. `indexing.neptune.index_concurrency`가 더 크면 클라이언트가 그 값으로 늘립니다. |
| `aws.opensearch.endpoint` | `null` | **필수.** OpenSearch 도메인 엔드포인트입니다. |
| `aws.opensearch.use_iam` | `false` | `false`이면 환경 변수 `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD`를 씁니다(§1 인증). |
| `aws.opensearch.allow_anonymous` | `false` | 인증 없이 연결합니다. 보안 플러그인을 끈 로컬 OpenSearch용입니다(§1 로컬 저장소). `use_iam`이나 username/password와 함께 쓸 수 없습니다. |
| `aws.opensearch.sigv4_service_name` | `"es"` | 관리형 도메인은 `es`, OpenSearch Serverless는 `aoss`입니다. 값이 틀리면 검색 결과가 0건으로 나오는 경우가 많습니다. |
| `aws.s3.bucket_name` | `null` | 캐시 동기화와 임베딩 캐시 저장에 쓰는 버킷입니다. |
| `aws.s3.encryption.encryption_type` | `"BUCKET_DEFAULT"` | `BUCKET_DEFAULT`는 버킷 기본 암호화를 따릅니다. `AES256`이나 `aws:kms`(`kms_key_id` 필요)는 객체별 헤더를 강제합니다. 단계 캐시 동기화와 영구 임베딩 캐시 업로드에 모두 적용됩니다. |
| `aws.dynamodb.enabled` | `false` | 증분 인덱싱용 doc-status 레지스트리를 켭니다(§5). |
| `aws.dynamodb.table_name` | `"unified-kg-rag-on-aws-doc-status"` | doc-status 테이블 이름입니다. |
| `aws.dynamodb.create_table_if_missing` | `true` | 처음 사용할 때 테이블을 만듭니다. IaC로 관리하는 테이블이면 `false`로 두세요. |

> **Guardrail 적용 범위와 위치.** Guardrail은 LLM 호출이 전달되는
> `aws.bedrock.region_name` 리전에 있어야 합니다. 기본값 `apply_to: "query"`는
> 답변 생성, 질의 정제, 질의 시점 엔터티·키워드 추출, global/DRIFT map-reduce에만
> Guardrail을 붙입니다. 인제스천 모델, 프롬프트 튜너, 평가 모델에는 붙이지
> 않습니다. `NAME`을 익명화하는 PII Guardrail은 추출된 엔터티 이름을 `{NAME}` 같은
> 플레이스홀더로 바꿔 서로 다른 인물을 한 노드로 병합하고, `PROMPT_ATTACK` 필터는
> 지시문처럼 보이는 코퍼스 텍스트를 차단할 수 있기 때문입니다. `apply_to: "all"`은
> 추출에 안전한 정책일 때만 사용하세요. 질의 경로에서도 `NAME` 익명화는 답변의
> 인물 이름을 `{NAME}`으로 바꾸고 엔터티 기반 검색의 시드를 없애므로, `iac/`가
> 만드는 기본 Guardrail은 이메일·전화번호·카드 번호만 익명화하고 `NAME`은
> 익명화하지 않습니다. Guardrail이 개입할 때마다 누적 횟수와 함께
> WARNING 로그(`Bedrock guardrail '<id>' intervened on a <purpose> model call ...`)를
> 남깁니다. InvokeModel 경로(`ChatBedrock`, 크로스 리전이 아닌 모델 ID 사용 시)에서는
> `trace: true`일 때만 개입을 감지하며, Guardrail 자체는 어느 경우든 적용됩니다.
>
> 업그레이드 시 참고: 이전 릴리스는 모든 호출에 Guardrail을 적용했습니다. 같은
> 동작을 유지하려면 `apply_to: "all"`로 설정합니다. 질의가 아닌 작업에서
> `setup_chain`으로 체인을 만들거나 `get_model`을 직접 호출하는 사용자 코드는
> `model_purpose=ModelPurpose.INGESTION`(또는 `EVALUATION`)을 넘겨야 합니다.
> 지정하지 않은 호출은 `QUERY`로 간주해 Guardrail이 계속 적용되고, 질의 시점의
> 일시적 오류 재시도도 함께 적용됩니다.

> **S3 캐시 암호화.** CDK 스택에서 `use_cmk=true`이면 버킷 기본 암호화가 고객 관리형
> KMS 키이므로 `BUCKET_DEFAULT`도 그 키를 씁니다. 이 기본값 이전 릴리스는 `AES256`을
> 보내 버킷의 CMK를 조용히 우회했습니다. 기본 암호화가 SSE-KMS인 기존 버킷을
> 재사용하면 업로드 주체에 해당 키의 `kms:GenerateDataKey`와 `kms:Decrypt` 권한이
> 필요합니다.

#### 모델 선택 주의사항

모든 LLM 역할은 두 등급 중 하나에 속합니다. 추론 비중이 큰 역할은
`aws.bedrock.default_model_id`, 경량 역할은 `aws.bedrock.fast_model_id`를 쓰므로
파이프라인 전체의 모델을 한 줄로 바꿀 수 있습니다. 역할별 `*_model_id` 키를
지정하면 등급보다 우선합니다.

| 등급 | 역할(`*_model_id` 키) |
| --- | --- |
| `default` | `fixing.fixing_model_id`, `processing.graph_extraction.extraction_model_id`, `processing.gleaning.graph_refinement_model_id`, `processing.claim_extraction.extraction_model_id`, `graph.community_detection.report_generation.report_generation_model_id`, `search.{entity_extraction,context_building,answer_generation}_model_id`, `evaluation.evaluation_model_id` |
| `fast` | `processing.chunking.chunking_model_id`, `processing.translation.translation_model_id`, `processing.graph_extraction.description_summarization.summary_model_id`, `search.{translation,strategy_selection}_model_id`, `search.global_search.{community_relevance,map_reduce,map}_model_id`, `search.drift_search.{query_refinement,keyword_expansion,convergence_assessment,primer}_model_id` |

```yaml
aws:
  bedrock:
    default_model_id: "openai.gpt-6-sol"    # default 등급 역할 전체
search:
  answer_generation_model_id: "anthropic.claude-opus-5-5"  # 역할 하나만 고정
```

모델 ID 키에는 어떤 Bedrock 모델 ID든, `us.anthropic.claude-sonnet-5-5` 같은
추론 프로파일 ID(그대로 사용)든 지정할 수 있습니다. 아래 모델(과 기존 Claude
3.x/4.x ID)은 검증된 기능 정보가 있습니다. 그 밖의 ID도 동작합니다.
`anthropic.claude-*`는 세대별 요청 형식, `openai.gpt-*`는 GPT 형식, 다른
공급자는 보수적인 Converse 요청(추론·샘플링 파라미터 없음, 32K 컨텍스트, 4K
출력)을 쓰고 WARNING을 한 번 남깁니다. 모델 정보를 지정하거나 고치려면
`aws.bedrock.model_overrides`를 씁니다. 키는 `context_window_size`,
`max_output_tokens` 같은 기능 정보 필드이며, 알 수 없는 키는 즉시 오류가 납니다.

```yaml
aws:
  bedrock:
    model_overrides:
      "amazon.nova-pro-v1:0":
        context_window_size: 300000
        max_output_tokens: 10000
```

임베딩·리랭킹 모델 ID는 다르게 동작합니다. 정해진 목록의 값만 받으며
`model_overrides`도 적용되지 않습니다. `embedding_model_id`에는
`amazon.titan-embed-text-v2:0`, `amazon.titan-embed-text-v1`,
`cohere.embed-v4:0`, `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`
중 하나를, `rerank_model_id`에는 `cohere.rerank-v3-5:0`이나
`amazon.rerank-v1:0`을 지정합니다. 그 밖의 ID는 설정 검증에서 오류가 납니다.
임베딩 차원은 OpenSearch 벡터 매핑에 기록되고 모델 정보와 대조하므로, 모르는
모델에는 안전한 기본값이 없기 때문입니다. `indexing.opensearch.embedding_dimension`은
목록에 있는 모델이 지원하는 차원 중 하나를 고릅니다(Titan Embed V2는 256, 512,
1024이며 지정하지 않으면 가장 큰 값). 모델을 추가하려면 코드를 바꿔야 합니다.
`domain/models/config.py`의 `EmbeddingModelId`/`RerankModelId`에 항목을,
`adapters/aws/bedrock_models.py`에 기능 정보를 추가합니다.

**출력 상한.** Bedrock은 요청 시작 시 입력 + `max_tokens`를 분당 토큰 할당량에서
미리 차감하므로, 모델 최대값(Claude 5.x는 128K)을 요청하면 실제 사용량보다 훨씬
먼저 동시 수집 호출이 스로틀링됩니다. 그래서 `default_max_output_tokens`가
16384입니다. thinking 토큰도 여기에 포함되며, 출력이 긴 프롬프트는 더 높은 하한을
선언해 그 값이 우선합니다(그래프·클레임 추출, gleaning, 커뮤니티 보고서와 그 출력
수정기 32768, 문서 번역 65536).

**프롬프트 캐싱.** 명시적 프롬프트 캐싱을 지원하는 Claude 모델에서는 각 시스템
프롬프트 끝을 캐시 지점으로 표시합니다. Converse API(모든 추론 프로파일)에서는
`cachePoint` 블록, InvokeModel에서는 `cache_control`을 씁니다. 시스템 프롬프트가
모델의 최소 캐시 지점 크기(Claude Sonnet/Opus 5.5와 Opus 5는 512토큰, 대부분은
1024토큰, Claude Haiku 4.5와 Opus 4.5-4.7은 4096토큰)보다 짧으면 표시하지
않습니다. Bedrock이 요청은 받지만 아무것도 캐시하지 않기 때문입니다. 캐시 적중은
응답의 `usage_metadata.input_token_details`에 `cache_read`로 나타나며, 캐시에서
읽은 입력 토큰은 분당 토큰 할당량에 포함되지 않습니다.

| 모델 ID | 공급자 | 컨텍스트 / 최대 출력 | 추론 제어 |
| --- | --- | --- | --- |
| `anthropic.claude-sonnet-5-5`(기본값) | Anthropic | 1M / 128K | adaptive, 항상 켜짐. `effort` low–max |
| `anthropic.claude-opus-5-5` | Anthropic | 1M / 128K | adaptive, 항상 켜짐. `effort` low–max |
| `anthropic.claude-haiku-5-5` | Anthropic | 1M / 128K | adaptive, 기본 켜짐. `effort` low–max |
| `anthropic.claude-sonnet-5`, `anthropic.claude-opus-5` | Anthropic | 1M / 128K | adaptive, 항상 켜짐. `effort` |
| `anthropic.claude-opus-4-8`, `anthropic.claude-opus-4-7` | Anthropic | 1M / 128K | adaptive, 항상 켜짐. `effort` low–max |
| `anthropic.claude-opus-4-6-v1` | Anthropic | 1M / 128K | 선택(`--enable-thinking`), adaptive. `effort` low/medium/high/max |
| `anthropic.claude-sonnet-4-6` | Anthropic | 1M / 64K | 선택, adaptive. `effort` low/medium/high/max |
| `openai.gpt-6.1-sol` | OpenAI | 1M / 131K | `reasoning.effort` low–max, 항상 켜짐 |
| `openai.gpt-6-astra`, `openai.gpt-6-sol`, `openai.gpt-6-luna` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, 항상 켜짐 |
| `openai.gpt-5.6-sol`, `openai.gpt-5.6-terra`, `openai.gpt-5.6-luna` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, 항상 켜짐 |
| `openai.gpt-5.5`, `openai.gpt-5.4` | OpenAI | 1.05M / 128K | `reasoning.effort` low–max, 항상 켜짐 |

모두 추론 프로파일 전용입니다. OpenAI는 독점 GPT 모델만 지원하며 오픈 웨이트
`gpt-oss` 모델은 제외했습니다.

Claude 4.7 이후 모델은 세 가지가 다릅니다.

- **추론 프로파일이 필수입니다.** `ON_DEMAND` 처리량 없이 출시되므로 순수 모델
  ID로는 호출할 수 없고 크로스 리전 프로파일이 반드시 해석돼야 합니다.
  `enable_global_profile: true`를 유지하고 `bedrock:ListInferenceProfiles` 권한을
  부여하세요. 프로파일이 없으면 어댑터가 해결 방법과 함께 즉시 실패합니다.
  특히 `ap-northeast-2`에는 Claude 5용 `global.` 프로파일만 존재하고 `apac.`은
  없으므로, 글로벌 프로파일을 끄면 사용 경로가 없습니다.
- **`effort`가 사고 토큰 예산을 대체합니다.** 이 모델들에서는
  `thinking_budget_tokens`가 무시됩니다(기존 `budget_tokens` 형식은 400으로
  거부됨). 대신 `bedrock.default_effort` / `bedrock.fast_effort`를 설정하세요.
  호출 모델이 `fast_model_id`이고 `default_model_id`와 다르면 `fast_effort`를,
  그 밖에는 `default_effort`를 씁니다. 기본 fast 모델인 Claude Haiku 5.5는
  adaptive 사고를 하므로, `fast_effort`(기본값 `low`)가 fast 등급 호출의 추론
  깊이를 정합니다. Claude Sonnet 5.5는 사고를 끌 수
  없어 `--enable-thinking`이 무의미하며, 깊이는 `effort`로만 조절합니다.
  모델이 받지 않는 수준(예: Opus·Sonnet 4.6의 `xhigh`)은 즉시 실패합니다.
- **샘플링 파라미터가 제거됩니다.** `temperature`/`top_k`는 수용되지 않으므로
  요청에서 자동 생략됩니다. 동작 제어는 프롬프트로 하세요.

OpenAI GPT 모델은 Claude와 다음이 다릅니다.

- 항상 `us.`/`global.` 추론 프로파일에서 Converse API로 호출합니다. `apac.`/`eu.`
  지역 프로파일이 없으므로 미국 외 리전에서는 `enable_global_profile: true`를
  유지하세요. 등급별 effort(`bedrock.default_effort` / `fast_effort`)는
  `reasoning: {effort: ...}`로 전달됩니다(평면 필드 `reasoning_effort`는 거부됨).
  GPT-5.6과 GPT-6.x는 `effort: low`에서도 짧은 프롬프트 응답에 약 10~25초가
  걸렸으므로 타임아웃과 동시성을 이에 맞춰 설정하세요.
- Anthropic 전용 필드(`thinking`, `output_config`, `anthropic_beta`, `\n\nHuman:`
  중지 시퀀스)를 보내지 않으며 샘플링 파라미터도 생략합니다.
- 명시적 프롬프트 캐시 마커를 보내지 않습니다. 이 모델들은 Converse에서 암묵적
  캐싱만 지원합니다. Bedrock CountTokens도 지원하지 않으므로 검색 컨텍스트 예산은
  로컬 토큰 추정치를 사용합니다.

Claude Fable 5 / 5.1은 제공하지 않습니다. 기본값이 아닌 계정 데이터 보존 모드
(Data Retention API로만 설정)가 필요하며, 기본 모드 계정에서는 모든 호출이
`data retention mode 'default' is not available for this model`로 거부됩니다.

### 2.2 `fixing` — 잘못된 형식의 모델 출력 자동 복구

| 키 | 기본값 | 역할 / 바꿀 때 |
|---|---|---|
| `fixing.enabled` | `true` | 구조화된 스테이지에서 모델이 잘못된 형식의 JSON을 반환하면, 실행을 실패시키는 대신 모델에게 복구를 요청합니다. 켜 두세요. |

### 2.3 `processing` — 동시성, 청킹, 번역, 추출

LLM 스테이지는 Bedrock I/O 바운드이므로 동시성을 CPU 수보다 훨씬 높게 잡을 수
있습니다.

| 키 | 기본값 | 역할 / 바꿀 때 |
|---|---|---|
| `processing.max_concurrency` | `20` | 배치 하나에서 동시에 보내는 LLM 호출 수입니다. Bedrock 스로틀링이 나면 낮추고, 할당량에 여유가 있으면 올립니다. |
| `processing.chunk_concurrency` | `4` | 동시에 실행하는 미니 배치 수입니다. Bedrock 연결 풀은 `max_concurrency` × `chunk_concurrency`로 잡힙니다. |
| `processing.max_attempts` | `5` | 배치 호출이 실패한 인제스천 LLM 항목을 하나씩 다시 호출할 때의 시도 횟수입니다. 일시적인 Bedrock 오류, 호출 시간 초과, 파싱할 수 없는 출력만 재시도하고 나머지 오류는 바로 실패합니다. `1`이면 재시도하지 않습니다. |
| `processing.io_workers` | `64` | CLI와 체인 동기 메서드에서 질의 경로의 블로킹 I/O(Bedrock 호출, Neptune 순회, 재순위)를 처리하는 스레드 수입니다. Python 기본값은 `min(32, CPU 수 + 4)`라서 vCPU 2개 작업에서는 6개입니다. 비동기 서버는 시작할 때 `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`(`unified_kg_rag.shared.utils`)를 호출합니다. |
| `processing.ignore_errors` | `false` | LLM 단계가 실패한 항목을 건너뛰고 실행을 계속합니다. |
| `processing.deduplicate` | `false` | 추출 전에 중복 문서를 제거합니다. |
| `processing.resolution_method` | `"minhash"` | 엔터티 해소 방식입니다. `minhash` 또는 `sequence_matcher`입니다. |
| `processing.similarity_threshold` | `0.6` | 엔터티 해소의 유사도 임계값입니다. 서로 다른 엔터티가 합쳐지면 올립니다. |
| `processing.document_parsing.source_directory` | `"source"` | 라이브러리 호출용 기본값입니다. `run-ingestion`은 `--source-directory`(또는 `GRAPHRAG_SOURCE_DIRECTORY`)가 반드시 필요합니다. |
| `processing.document_parsing.target_directory` | `null` | 파싱한 문서를 `<stem>.json`으로 내보내 확인할 디렉터리입니다(`--target-directory`와 같음). 소스 디렉터리로는 지정할 수 없습니다. |
| `processing.document_parsing.source_scope` | `null` | 증분 삭제에 쓰는 코퍼스 식별자입니다. 실행은 자기 인덱스 접미사와 소스 범위에 속한 레지스트리 문서만 삭제합니다. `null`이면 소스 디렉터리의 절대 경로이며, 컨테이너 엔트리포인트는 S3 URI로 설정합니다(`GRAPHRAG_SOURCE_SCOPE`). §5 참고. |
| `processing.chunking.chunker_type` | `"intelligent"` | `intelligent`는 LLM이 의미 경계를 고르고, `simple`은 크기로 나눕니다. |
| `processing.chunking.min_chunk_size` | `1000` | 최소 청크 크기(문자)입니다. 이보다 짧은 조각은 이웃 청크에 합칩니다. |
| `processing.chunking.max_chunk_size` | `8000` | 최대 청크 크기(문자)입니다. 임베딩과 리랭크 입력 한도 안에 들어야 합니다. |
| `processing.chunking.chunk_overlap` | `500` | 청크 간 겹침(문자)입니다. |
| `processing.chunking.fallback_chunk_size` | `4800` | 크기 기반 분할(`simple`, 또는 intelligent 청킹이 실패할 때)의 목표 크기입니다. |
| `processing.translation.enabled` | `true` | 번역 스테이지를 실행합니다. 원문과 대상 언어가 같고 추가 대상 언어가 없으면 아무것도 하지 않습니다(LLM 비용 0). |
| `processing.translation.source_language` | `"en"` | 코퍼스의 주 언어입니다. 번역 생략 여부 판단에만 씁니다. |
| `processing.translation.target_language` | `"en"` | 코퍼스를 번역할 언어입니다(§3 다국어 인제스천 참고). |
| `processing.translation.additional_target_languages` | `null` | 추가로 번역할 대상 언어 목록입니다. |
| `processing.graph_extraction.entity_types` | 범용 유형 7개 | 추출 프롬프트에 넣는 `"LABEL: 설명"` 목록입니다. 도메인 적응에 가장 효과가 큰 항목입니다(§9). 빈 목록이면 모델이 유형을 고릅니다. |
| `processing.graph_extraction.max_entities_per_chunk` | `50` | 청크당 엔터티 상한입니다(관계는 `max_relationships_per_chunk`, 역시 `50`). |
| `processing.graph_extraction.entity_confidence_threshold` | `0.0` | 신뢰도가 이 값보다 낮은 엔터티를 버립니다. `0.0`이면 모두 유지합니다. |
| `processing.graph_extraction.description_summarization.enabled` | `true` | 병합된 설명이 `force_summary_threshold_tokens`(`600`)를 넘으면 LLM으로 다시 요약합니다. |
| `processing.graph_extraction.entity_grounding.enabled` | `false` | 환각 방지 장치입니다. 원문 `source_text` 구간이 청크에 없는 엔터티와 관계를 버리거나, `action: "penalize"`이면 가중치를 낮춥니다. gleaning이 추가한 항목에도 적용됩니다. |
| `processing.gleaning.enabled` | `true` | 첫 추출에서 놓친 엔터티와 관계를 찾는 추가 추출 단계입니다. |
| `processing.gleaning.max_rounds` | `3` | 텍스트 단위당 최대 gleaning 횟수입니다. 다음 회차에는 직전 응답에서 새 엔터티나 관계가 추가된 단위만 다시 보내므로, 응답에 새 항목이 없으면 그 단위는 바로 멈춥니다. `1`이 MS GraphRAG 기본값과 같습니다. |
| `processing.claim_extraction.enabled` | `false` | claim을 추출합니다(텍스트 단위마다 LLM 호출 1회 추가). 켜면 `local` 검색이 관련 claim을 컨텍스트에 넣고 `simple` 검색이 claim 인덱스도 함께 찾습니다. |

### 2.4 `graph` — 분석, 커뮤니티 탐지, 시각화

| 키 | 기본값 | 역할 / 바꿀 때 |
|---|---|---|
| `graph.community_detection.enabled` | `true` | Leiden 클러스터링과 커뮤니티 리포트 생성입니다. GraphRAG `global`/`drift`에 필요합니다. LightRAG 전용으로 가볍게 인제스천하려면 `false`로 둡니다. |
| `graph.community_detection.auto_resolution` | `false` | `true`이면 계층마다 `auto_resolution_candidates`를 차례로 시험해 모듈성이 가장 높은 해상도를 고릅니다. 그렇지 않으면 `resolution`(`1.0`)을 씁니다. |
| `graph.community_detection.auto_resolution_max_nodes` | `10000` | 노드 수가 이보다 많으면 해상도 탐색을 건너뛰고 `resolution`을 씁니다. |
| `graph.community_detection.max_levels` | `5` | 커뮤니티 계층의 최대 깊이입니다. |
| `graph.community_detection.min_community_size` | `3` | 이보다 작은 커뮤니티는 이웃 커뮤니티에 합칩니다. |
| `graph.community_detection.report_generation.max_report_context_tokens` | `4000` | 리포트 프롬프트 하나에 넣는 엔터티·관계 컨텍스트의 토큰 예산입니다. |
| `graph.community_detection.report_generation.content_length` | `"medium"` | 리포트 길이입니다. `short`, `medium`, `long` 중 하나입니다. |
| `graph.analysis.centrality.calculate_betweenness` | `true` | 매개 중심성을 계산합니다. 노드가 `betweenness_auto_sample_threshold`(`2000`)보다 많으면 정확한 계산 대신 샘플링합니다. |
| `graph.visualization.enabled` | `true` | 인제스천 중에 시각화 데이터를 내보냅니다. |
| `graph.visualization.outputs_directory` | `null` | 지정하지 않으면 `<캐시 디렉터리>/<pipeline_id>/visualization`에 기록해 캐시와 함께 S3로 동기화합니다. |
| `graph.visualization.layout_method` | `"umap"` | `umap`, `tsne`, `pca` 중 하나입니다. |
| `graph.visualization.interactive.max_nodes` | `2000` | `interactive_graph.html`에 남길 상위 N개 노드(차수 기준)입니다. `0`이나 `null`이면 제한하지 않습니다. `interactive`를 지정하면 기본 딕셔너리가 대체되므로, physics를 끈 상태로 두려면 `physics_enabled: false`도 함께 적으세요. |

### 2.5 `indexing` — OpenSearch & Neptune 쓰기 측

| 키 | 기본값 | 역할 / 바꿀 때 |
|---|---|---|
| `indexing.reset` | `false` | 인덱싱 전에 기존 데이터를 지웁니다. |
| `indexing.additional_suffix` | `null` | 모든 OpenSearch 인덱스 이름과 Neptune 레이블에서 실행 접미사 뒤에 붙습니다(`<prefix>-<suffix>-<additional_suffix>`, `<suffix>`는 `--suffix` 값 또는 `default`). 버전별·테넌트별로 분리할 때 씁니다. |
| `indexing.cross_run_merge` | `true` | 증분 실행에서 기존 그래프를 덮어쓰지 않고 새 데이터와 합쳐, 변경되지 않은 문서와 공유되는 엔터티의 계보를 유지합니다(§5). `false`면 덮어씁니다. |
| `indexing.cross_run_fuzzy_merge` | `false` | `cross_run_merge`에 엔터티 이름 유사도 매칭을 더합니다. |
| `indexing.max_failure_rate` | `0.2` | 인덱스 유형별 쓰기 실패율이 이 값을 넘으면 인덱싱 스테이지를 실패로 처리합니다. `1.0`이면 부분 실패 검사를 끕니다. |
| `indexing.opensearch.embedding_model_id` | `"amazon.titan-embed-text-v2:0"` | 임베딩 모델이며 정해진 목록에서 고릅니다(모델 선택 주의사항 참고). 바꾸면 다시 인덱싱해야 합니다. |
| `indexing.opensearch.build_relationship_vector_index` | `true` | LightRAG `mix`/`hybrid`가 쓰는 관계 벡터 인덱스를 만듭니다. GraphRAG만 쓰는 배포라면 `false`로 둡니다. |
| `indexing.opensearch.persist_embedding_cache` | `false` | 임베딩 캐시를 S3에 저장해 바뀌지 않은 텍스트를 실행마다 다시 임베딩하지 않습니다. `aws.s3.bucket_name`이 필요합니다. |
| `indexing.opensearch.language_analyzers` | `{en: english, ko: nori}` | 언어 코드별 텍스트 분석기입니다. 목록에 없는 언어는 `default_analyzer`(`standard`)를 씁니다. |
| `indexing.opensearch.vector_search.engine` | `"lucene"` | HNSW 엔진입니다. `lucene`은 1024차원까지 `cosinesimil`을 지원합니다. `faiss`는 템플릿 주석을 참고하세요. |
| `indexing.opensearch.index_settings.refresh_interval` | `"1s"` | 대량 적재 속도를 높이려면 늘리거나 `"-1"`로 두고, 실제 질의 전에 되돌립니다. |
| `indexing.neptune.batch_size` | `100` | Neptune 쓰기 배치당 항목 수입니다. |
| `indexing.neptune.index_concurrency` | `1` | 동시에 보내는 쓰기 배치 수입니다. `aws.neptune.pool_size`가 이보다 작으면 Gremlin 연결 풀을 이 값까지 늘립니다. |
| `indexing.neptune.max_attempts` | `4` | 첫 시도를 포함한 Neptune 쓰기당 시도 횟수입니다. 실패하면 `retry_delay_seconds`(`2`)에서 시작하는 지수 백오프(지터 포함)로 재시도합니다. 다만 잘못된 쿼리, 권한 거부, 잘못된 파라미터처럼 재시도해도 고쳐지지 않는 오류는 첫 시도에서 바로 실패합니다. `1`이면 재시도하지 않습니다. |
| `indexing.neptune.max_hops` | `3` | 검색 시점의 이웃 확장 깊이입니다. |
| `indexing.neptune.property_max_length` | `4000` | Neptune 속성 값의 최대 문자 수입니다. 재요약되지 않는 가장 긴 설명보다 커야 합니다(요약은 600토큰, 영어 약 2,400자를 넘을 때만 실행). 재인제스트해야 반영됩니다. |
| `indexing.neptune.entity_importance_source` | `"rank"` | 그래프 확장 관련도에 쓰는 엔터티 중요도입니다. `rank`(인덱싱된 엔터티 rank), `degree`(질의 시 계산한 엣지 수), `none`(모두 0.5, 이전 동작). |
| `indexing.neptune.traversal_fetch_multiplier` | `3` | 그래프 확장이 결과 폭의 이 배수만큼 가져와 순위를 매긴 뒤 자릅니다. `1`이면 순회 순서대로 자릅니다(이전 동작). |

### 2.6 `search` — 검색, 융합, 리랭킹, 전략별 항목

| 키 | 기본값 | 역할 / 바꿀 때 |
|---|---|---|
| `search.auto_routable_strategies` | `["local", "mix", "global", "drift"]` | `auto` 라우터가 고를 수 있는 전략입니다. 어떤 전략이든 직접 지정할 수는 있습니다. |
| `search.hybrid.lexical_weight` | `0.5` | OpenSearch 하이브리드 파이프라인의 어휘 검색 가중치입니다(벡터는 `vector_weight`, 역시 `0.5`). |
| `search.fusion.method` | `"rrf"` | `rrf`(reciprocal rank fusion) 또는 `weighted`입니다. |
| `search.fusion.rrf_k` | `60` | RRF 상수 `k`입니다. |
| `search.fusion.fusion_weights` | 버킷마다 `1.0` | 검색 소스 버킷별 가중치입니다. `rrf`와 `weighted` 모두에서 버킷의 기여도에 곱해집니다. |
| `search.fusion.diversity_lambda` | `0.5` | MMR 균형값입니다. `1.0`은 관련도만, `0.0`은 다양성을 최대로 반영합니다. |
| `search.reranking.enabled` | `true` | 융합 결과를 `rerank_model_id`(`cohere.rerank-v3-5:0`)로 다시 정렬합니다. |
| `search.reranking.top_k` | `100` | 리랭커에 보내는 후보 수입니다. |
| `search.lightrag_search.kg_stream_top_k` | `40` | LightRAG 엔터티·관계 벡터 질의의 폭입니다(요청의 `top_k`에 대한 하한). |
| `search.lightrag_search.chunk_stream_top_k` | `20` | LightRAG 청크 스트림의 폭입니다. |
| `search.lightrag_search.enable_graph_expansion` | `false` | `mix`/`hybrid`에서 일치한 항목을 Neptune으로 추가 확장합니다. |
| `search.global_search.max_communities` | `10` | `global` 검색이 살펴보는 커뮤니티 리포트 수입니다. |
| `search.global_search.map_batch_size` | `5` | map 단계 LLM 호출 하나에 넣는 리포트 수입니다. 리포트가 길면 낮춥니다. |
| `search.global_search.max_map_reduce_tokens` | `8000` | reduce 단계에 넣는, 순위를 매긴 핵심 내용의 토큰 예산입니다. |
| `search.global_search.reduce_with_llm` | `false` | `true`이면 reduce LLM이 팩된 핵심 내용을 먼저 요약하고 답변 모델이 이를 다시 씁니다(LLM 호출 1회 추가). `false`이면 핵심 내용을 답변 모델에 바로 넘깁니다. |
| `search.global_search.reserve_report_slots` | `true` | 커뮤니티 리포트에 `max_communities`개 융합 슬롯을 예약하고 텍스트 단위는 `text_unit_slots`개로 제한합니다. `false`이면 리포트와 청크를 한 번의 `top_k` 컷으로 자릅니다(이전 동작). |
| `search.global_search.text_unit_slots` | `null` | 예약된 리포트 슬롯과 함께 둘 텍스트 단위 슬롯 수입니다. `null`이면 질의의 `top_k`입니다. |
| `search.local_search.entity_frequency_threshold` | `20` | 그래프 확장으로 얻은 엔터티 중 이보다 많은 텍스트 단위에 나오는 것(너무 일반적인 것)을 버립니다. |
| `search.local_search.include_bridge_relationships` | `true` | 확장된 엔터티에 연결된 관계도 가져오며, 조회된 두 엔터티를 잇는 엣지(다중 홉 연결 고리)를 먼저 둡니다. 관계 인덱스가 필요합니다. `false`이면 관계 벡터 질의만 씁니다. |
| `search.drift_search.max_iterations` | `3` | DRIFT 반복 횟수 상한입니다. |
| `search.drift_search.enable_primer` | `false` | MS GraphRAG의 primer → follow-up 흐름을 씁니다(처음에 LLM 호출 1회 추가). |
| `search.drift_search.enable_llm_convergence` | `false` | 반복마다 LLM으로 수렴 여부를 판단합니다(반복당 호출 1회 추가). |
| `search.token_manager.max_context_tokens` | `30000` | 답변 프롬프트에 넣는 검색 컨텍스트 예산입니다(아래 참고 사항). |
| `search.token_manager.context_window_headroom_ratio` | `0.1` | 예산을 자동 도출할 때(`max_context_tokens: null`) 남겨 두는 컨텍스트 창 비율입니다. |

섹션 유형별 비율(`search.token_manager.type_budgets`)과 `local` 검색의 유형별
슬롯(`search.local_search.type_quota`)은 템플릿을 참고하세요.

> **컨텍스트 예산.** `30000`은 upstream LightRAG의 전체 컨텍스트 예산과 같습니다(MS
> GraphRAG는 12000). 1M 토큰 창에서 도출한 예산(약 785K)은 실제로 제한이 걸리지
> 않아 유형별 예산이 아무것도 잘라 내지 못합니다. 이 값은 항상
> `search.answer_generation_model_id`가 출력 예약분
> (`aws.bedrock.default_max_output_tokens`)과 함께 받을 수 있는 한도로 줄어들며,
> 그때 경고 로그를 남깁니다. `null`이면 해당 모델의 창에서 출력 예약분과 헤드룸
> 비율을 뺀 값으로 예산을 도출하며, 1M이 베타인 모델에서는
> `aws.bedrock.enable_1m_context`를 켜면 예산이 넓어집니다.

### 2.7 `memory`, `cache`, `logging`

| 키 | 기본값 | 역할 / 바꿀 때 |
|---|---|---|
| `memory.max_conversations` | `100` | 대화 메모리에 유지하는 대화 수입니다(§4 인터랙티브 모드). |
| `memory.max_messages_per_conversation` | `20` | 대화당 유지하는 메시지 수입니다. |
| `memory.max_conversation_age_hours` | `168` | 이 시간이 지난 대화는 정리 대상이 됩니다. |
| `cache.ttl_seconds` | `86400` | 캐시 항목 TTL입니다. `null`이면 만료하지 않습니다. |
| `logging.level` | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` 중 하나입니다. |
| `logging.log_format` | `"structured"` | `structured` 또는 `plain`입니다. |
| `logging.log_to_file` | `true` | CLI가 로그를 `log_file_path`(`logs/log.txt`, 실제 파일명은 `log_YYYYMMDD.txt`)에도 기록합니다. 상대 경로는 작업 디렉터리 기준입니다. 패키지를 라이브러리로 import하면 핸들러나 파일을 설정하지 않고 호스트 애플리케이션의 로깅 설정을 따릅니다. |
| `logging.library_levels` | `{langchain_aws: WARNING, botocore: WARNING, urllib3: WARNING}` | 로그가 많은 라이브러리의 로거별 수준입니다. 지정하면 기본 목록 전체가 대체됩니다. |

**로그에 남는 내용.** `INFO` 이상 수준에서는 사용자·코퍼스 텍스트 대신 길이,
개수, ID와 짧은 해시만 기록합니다(예: `query: len=42 sha=1a2b3c4d`). 해시는
항상 같은 값이므로 내용을 드러내지 않고 같은 질의를 여러 로그 레코드에서 추적할
수 있습니다. 질의 원문, 다시 쓴 질의(DRIFT·번역), 엔터티 이름과 모델 원본 출력은
`DEBUG`에서만 기록하며 예외 메시지에는 모델 출력을 넣지 않습니다. `DEBUG`
로그(`logging.level: DEBUG`, `LOG_LEVEL=DEBUG`, `--verbose`)에는 사용자·코퍼스
데이터가 들어 있다고 보고, 로그를 공유 저장소로 보내는 환경에서는 켜지 마세요.
`logging.library_levels`에서 `DEBUG`로 둔 라이브러리(예: `botocore`)는 요청
본문까지 기록할 수 있습니다.

### 2.8 `evaluation`

| 키 | 기본값 | 역할 / 바꿀 때 |
|---|---|---|
| `evaluation.enabled_evaluators` | `[langchain, ragas, answer_match, retrieval, graph_aware]` | 결정적 평가기(`answer_match`, `retrieval`, `graph_aware`)는 필요한 정답 필드가 없는 질의를 건너뜁니다(§6). 평가 모델 비용 없이 평가하려면 LLM 평가기를 빼면 됩니다. |
| `evaluation.judge_effort` | `"low"` | LLM 평가 모델의 추론 깊이입니다. `null`이면 평가 모델이 속한 등급의 effort(기본은 `aws.bedrock.default_effort`)를 따릅니다. |
| `evaluation.ragas_timeout` | `300` | 샘플 하나의 지표 하나를 계산하는 제한 시간(초)입니다. 넘으면 NaN이 됩니다. |
| `evaluation.ragas_max_contexts` | `20` | RAGAS `context_precision`이 채점하는 샘플별 상위 컨텍스트 수입니다("@20"). faithfulness와 context_recall은 토큰 예산 안의 전체 컨텍스트를 봅니다. `null`이면 제한하지 않습니다. |
| `evaluation.ragas_max_workers` | `8` | 동시에 실행하는 RAGAS 작업 수입니다. 평가 모델 호출이 스로틀링되면 낮춥니다. |
| `evaluation.ragas_max_attempts` | `3` | 평가 모델 호출당 총 시도 횟수입니다. |
| `evaluation.max_context_tokens` | `8192` | 평가 모델에 넘기는 컨텍스트의 토큰 상한입니다. |
| `evaluation.retrieval_k` | `5` | `retrieval` 평가자의 hit@k / recall@k 기준값입니다. |
| `evaluation.outputs_directory` | `"outputs/evaluation"` | 결과를 기록하는 디렉터리입니다. |

지표 목록(`langchain_metrics`, `ragas_metrics`)은 템플릿을 참고하세요.

### 2.9 `custom_prompts`

모든 프롬프트에는 `*_system` / `*_human` 오버라이드가 있습니다(기본값 `null` =
`unified_kg_rag/domain/prompts/`의 내장 프롬프트 사용). §9를 참고하세요. 필요한
것만 오버라이드하고 나머지는 `null`로 두세요.

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

> **남아 있는 `AWS_REGION`이 설정 파일보다 우선합니다.** SSO 자격 증명 도우미,
> CloudShell, 셸 프로필은 `AWS_REGION`을 내보내는 경우가 많습니다. 이 값이
> `aws.region_name`을 조용히 대체하면 Neptune과 OpenSearch(그리고
> `aws.bedrock.region_name`을 지정하지 않았다면 Bedrock)를 엉뚱한 리전에서 찾게 됩니다. 실행 전에 `env | grep -E '^(AWS_REGION|BEDROCK_REGION)='`로
> 확인하고, 의도하지 않은 값은 해제하거나 고치세요.

> **LangSmith 추적을 켜면 내용이 외부로 전송됩니다.** `LANGSMITH_TRACING=true`(또는
> 이전 이름인 `LANGCHAIN_TRACING_V2=true`)가 설정되어 있으면 LangChain이 추적하는
> 모든 실행을 LangSmith로 보냅니다. 여기에는 프롬프트, 검색된 문서 본문, 모델 출력이
> 포함됩니다. CLI는 추적이 켜져 있으면 시작할 때 WARNING을 남기지만, 의도한 설정일
> 수 있으므로 끄지는 않습니다. 기밀 코퍼스로 실행하기 전에는 이 변수를 해제하세요.
> 경고는 `setup_logging`에서 남기므로, 이를 호출하지 않는 라이브러리 코드에서는
> 나오지 않습니다.

CLI는 python-dotenv로 `.env` 파일도 읽습니다. 이미 환경에 설정된 변수가 `.env`보다
우선합니다. `.env`는 현재 디렉터리가 아니라 패키지 위치에서 상위 디렉터리로
올라가며 찾으므로, 소스 체크아웃에서는 저장소 루트에 두세요.

CDK compute 스택은 `AWS_REGION`, `BEDROCK_REGION`, `NEPTUNE_ENDPOINT`,
`OPENSEARCH_ENDPOINT`, `S3_BUCKET_NAME`, `GRAPHRAG_DOC_STATUS_TABLE`,
`GRAPHRAG_DOC_STATUS_CREATE_TABLE=false`(테이블은 IaC가 관리하고 태스크 역할에는
테이블 생성 권한이 없음), `LOG_FORMAT`을 주입하며, Guardrail을 배포했다면
`BEDROCK_GUARDRAIL_IDENTIFIER`도 주입합니다. 증분 인덱싱을 쓰려면 여전히 설정
파일에 `aws.dynamodb.enabled: true`가 필요합니다.

---

## 3. 인제스천 (`run-ingestion`)

인제스천은 문서 디렉터리를 OpenSearch + Neptune에 인덱싱된 지식 그래프로
변환합니다.

### CLI 플래그 (검증됨)

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--source-directory` | `$GRAPHRAG_SOURCE_DIRECTORY` | 소스 문서 디렉터리. 실행에 필수이며, 플래그를 생략하면 `GRAPHRAG_SOURCE_DIRECTORY` 환경 변수로 대체됩니다. |
| `--target-directory` | 없음(내보내지 않음) | 파싱된 문서를 확인용 JSON으로 내보낼 위치(소스 디렉터리는 지정 불가) |
| `--cache-directory` | `cache` | 파이프라인 캐시 + 중간 결과 |
| `--force-rebuild` | off | 기존 캐시를 모두 무시하고 처음부터 재구축 |
| `--s3-sync` | off | 캐시를 S3에 동기화 (`--s3-bucket-name` 필요) |
| `--s3-bucket-name` | — | 캐시 동기화용 S3 버킷 |
| `--s3-prefix` | `pipeline-runs` | 캐시 파일의 S3 키 프리픽스 |
| `--pipeline-id` | `$GRAPHRAG_PIPELINE_ID` | 재개/검사할 기존 실행. 플래그를 생략하면 `GRAPHRAG_PIPELINE_ID` 환경 변수로 대체됩니다. ID가 실행의 캐시 디렉터리와 S3 접두사 이름이 되므로 영문 소문자, 숫자, 하이픈, 밑줄만 허용합니다. |
| `--resume-from-stage` | — | 재개할 스테이지 (`--pipeline-id` 필요) |
| `--verify-metadata` | off | 파이프라인 메타데이터 무결성 검증 (`--pipeline-id` 필요). 손상되었으면 0이 아닌 코드로 종료 |
| `--repair-metadata` | off | 메타데이터 복구 시도 (`--pipeline-id` 필요). 복구에 실패하면 0이 아닌 코드로 종료 |
| `--continue-on-error` | off | 스테이지 에러 시에도 계속 진행 |
| `--enabled-stages` | all | 실행할 스테이지 목록(쉼표 구분) |
| `--metrics-sink` | `none` | `none`, 또는 `cloudwatch` (CloudWatch EMF — Embedded Metric Format — 메트릭을 차원 없는 시리즈로 stdout에 출력. `pipeline_id`는 차원이 아닌 로그 속성으로 기록하므로 실행할 때마다 메트릭 시리즈가 늘지 않음) |
| `--config-path` | — | `config.yaml` 경로 |

### 12개 파이프라인 스테이지

실행 순서(`DataIngestionPipeline.STAGE_CLASSES`). `--enabled-stages` /
`--resume-from-stage`에는 스테이지 **이름**(대소문자 무관)을 사용하세요.

1. **`document_parsing`** — 포맷별 텍스트 추출(`.pdf`, `.txt`, `.csv`,
   `.json`; `unstructured` 추가 패키지로 `.md`/`.html`).
2. **`document_loading`** — 파싱된 문서로 실행 대상 코퍼스를 구성(MinHash 중복
   제거, 증분 필터). `document_parsing`을 비활성화하면 소스 디렉터리에서 미리 파싱된
   `Document` `.json` 파일을 읽습니다.
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
있습니다. 시작 시 다운로드나 종료 시 업로드가 실패하거나 일부 파일만 동기화되면
`CacheSyncError`로 실행이 실패하고 `run-ingestion`은 0이 아닌 코드로 종료합니다.
따라서 Step Functions 단계별 실행에서는 체크포인트를 잃은 단계가 실패로
표시됩니다. 새 pipeline id에 원격 캐시가 비어 있는 것은 오류가 아닙니다.
임베딩에 한해서는 `indexing.opensearch.persist_embedding_cache: true`
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
| `--query`, `-q` | — | 단일 질의 — 이 옵션 또는 `--interactive` 중 정확히 하나가 필요합니다. |
| `--interactive`, `-i` | off | 인터랙티브 채팅 (메모리 자동 활성화) |
| `--mode` | `rag` | `rag`(전체 생성) 또는 `search`(검색만) |
| `--conversation-id` | — | 기존 대화 이어가기 |
| `--use-memory` | off | 대화 메모리 활성화 (인터랙티브에서는 자동) |
| `--suffix` | — | 멀티테넌트 또는 버전별 인덱스용 인덱스/라벨 접미사 |
| `--enable-thinking` | off | 모델의 단계별 추론 활성화 |
| `--search-strategy` | `auto` | `auto`, `drift`, `global`, `local`, `simple`, `mix`, `hybrid`, `naive` |
| `--search-type` | `hybrid` | `hybrid`, `lexical`, `vector` |
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
| `auto` | 모를 때 / 일반 용도 (기본값) | LLM 라우터(`search.strategy_selection_model_id`)가 질의로부터 `search.auto_routable_strategies`(기본값 local, mix, global, drift) 중 최적 전략 선택. |

**LightRAG (이중 레벨 키워드):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `mix` | 일반 LightRAG 용도; 그래프 + 청크 균형 | 저수준 키워드 → 엔티티 인덱스, 고수준 키워드 → 관계 인덱스, 1홉 연결 관계·끝점 엔티티 확장(Neptune 다중 홉 확장은 `search.lightrag_search.enable_graph_expansion`으로 선택), **추가로** naive 벡터 청크 검색을 섞음. |
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
run-rag --query "..." --filters attr_category:research type:PERSON --config-path config.yaml
```

필터는 OpenSearch에서 `term`/`terms`/`range` 절로, Neptune에서 `has` 단계로
변환됩니다. 각 저장소는 인덱서가 기록하는 필드를 받습니다.

| 저장소 | 필터 가능 필드 |
|---|---|
| Text unit | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.<path>` |
| Entity | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `attr_<key>`, `attributes.<path>` |
| Relationship | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids` |
| Claim | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `description`, `source_text` |
| Community report | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `attr_<key>`, `attributes.<path>` |
| Neptune entity 정점 | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `attr_<key>`(있는 경우에만) |
| Neptune community 정점 | `id`, `name`, `level`, `parent`, `size`, `period`, `children` |

OpenSearch에서 `attr_<key>`는 문서 속성입니다. 문서 `filters` 메타데이터의
`<key>` 항목이 `attr_<key>`로 색인됩니다. Neptune entity 정점의 `attr_<key>`는
entity 자체에서 추출한 속성(예: `attr_role`)입니다. 정확한 필터링에는 완전 일치 필드(keyword 또는 숫자)를
사용하십시오. `description`처럼 분석되는 텍스트 필드에 대한 `term` 필터는 소문자
단일 토큰과 일치합니다. 범위 필터는 Python API에서 `{"gte": ..., "lte": ...}`로
지정합니다.

각 필터는 해당 필드를 선언한 저장소에만 적용되므로 `type:PERSON`은 entity, claim,
Neptune entity만 좁히고 text unit에는 적용되지 않습니다. 두 저장소는
`attr_<key>`를 다르게 적용합니다.

- OpenSearch는 text unit, entity, community report에 `attr_<key>`와
  `attributes.<path>`를 엄격하게 적용합니다. 해당 속성이 없는 문서는 제외되므로
  보통 문서 속성이 없는 community report는 속성 필터 질의에서 빠집니다. 이는 의도한
  동작(fail-closed)으로, 속성 필터가 확인할 수 없는 내용을 반환하지 않게 합니다.
- Neptune은 entity 정점에 `attr_<key>`를 속성이 있는 경우에만 적용합니다. 속성이
  일치하거나 해당 속성이 없는 정점은 통과합니다. 따라서 `attr_category` 같은 문서
  속성 필터는 그래프 확장을 비우지 않고, `attr_role:buyer` 같은 entity 속성 필터는
  역할이 다른 entity를 제외합니다. 그 밖의 키는 두 저장소 모두 엄격하게 적용합니다.

**필터는 접근 제어 경계가 아닙니다.** 필터는 검색이 순위를 매길 대상을 좁힐 뿐
데이터를 숨기지 않습니다. 키를 선언하지 않은 저장소는 필터 없이 내용을 반환합니다.
relationship과 claim에는 `attr_<key>`와 `document_ids`가 없고, Neptune community
정점은 자체 필드만 선언하며, Neptune은 `attr_<key>` 속성이 없는 entity 정점을
그대로 둡니다. 그래프 확장과 community report도 필터가 제외할 문서의 내용을 가져올
수 있습니다. 권한이 다른 테넌트나 사용자를 필터로 구분하지 마십시오. 대신 각각에
별도 네임스페이스를 주십시오. 서로 다른 `--suffix`(와 `document_parsing.index_value`,
사용한다면 `indexing.additional_suffix`)를 쓰면 OpenSearch 인덱스와 Neptune 레이블이
분리되고, 질의 suffix는 검증되므로 인덱스 대상을 넓힐 수 없습니다. 호출자가 어떤
suffix를 질의할 수 있는지는 호출하는 애플리케이션에서 결정하십시오.

선택한 전략이 읽는 어떤 저장소도 선언하지 않은 필터 키(예: 이전 예시의 `category`,
`entity_type`)는 `InvalidFilterError`를 발생시키며, 오류 메시지에 필터 가능 키
목록이 포함됩니다. 이전 릴리스는 이런 키를 조용히 무시하고 필터링되지 않은 결과를
반환했습니다. 스키마는 `unified_kg_rag/adapters/storage/filter_schema.py`에
정의되어 있습니다.

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

메모리는 대화 턴 사이의 엔티티도 추적합니다. 사용자 메시지마다 LLM
(`search.entity_extraction_model_id`)이 언급된 엔티티를 추출하고, 이전 턴에서 나온
주요 엔티티를 다음 질의의 엔티티 초점에 더합니다. 그래서 "그 회사의 공급업체는?"
같은 후속 질문도 앞선 턴에서 언급한 엔티티를 중심으로 검색합니다.

---

## 5. 증분 인덱싱

증분(델타) 인덱싱은 지난 실행 이후 **신규이거나 변경된** 문서만 다시 인덱싱하고
이를 라이브 그래프에 병합합니다 — 전체를 재구축하는 대신.

### 활성화

```yaml
aws:
  dynamodb:
    enabled: true
    table_name: "unified-kg-rag-on-aws-doc-status"
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

### 삭제 범위 (scope)

레지스트리에서 문서의 키는 인덱스 접미사(`document_parsing.index_value`와
`indexing.additional_suffix`)와 소스 디렉터리 기준 상대 경로입니다. 실행은 자기
**범위**, 즉 같은 인덱스 접미사와 같은 코퍼스 소스(`document_parsing.source_scope`,
기본값은 소스 디렉터리의 절대 경로이며 컨테이너 엔트리포인트는 동기화 원본 S3
URI로 설정)에 기록된 문서만 삭제된 것으로 판단합니다. 따라서

- 다른 테넌트(다른 `index_value`)의 실행은 두 코퍼스를 같은 로컬 디렉터리에
  내려받아 처리하더라도 이 테넌트의 문서를 삭제하지 않습니다.
- 하위 폴더를 소스 디렉터리로 지정해 실행해도 나머지 코퍼스는 삭제되지 않습니다.
  다만 그 파일들은 별도 문서로 등록되므로 같은 파일을 두 루트에서 같은 접미사로
  인덱싱하지 않습니다.
- 파싱이나 로드에 실패한 파일은 `failed`로 보고되고, 다음 실행에서 다시 읽힐
  때까지 인덱싱된 내용이 유지됩니다.

로컬 코퍼스를 다른 디렉터리로 옮기면 기본 범위가 바뀝니다. 먼저 `source_scope`를
고정된 이름으로 설정하거나 재구축합니다.

### 문서 크기 한도

레지스트리는 문서마다 아티팩트 id(텍스트 단위, 엔티티, 관계, 클레임, 커뮤니티,
리포트)를 DynamoDB 항목 하나에 저장하며, 항목 크기는 400 KB(약 10,000개 id)로
제한됩니다. 이보다 많은 아티팩트를 만드는 문서는 파일명을 담은 오류와 함께 인덱싱
단계를 실패시키므로 더 작은 파일로 나눕니다.

### 실행 간 병합 (cross-run merge)

기본값(`indexing.cross_run_merge: true`)에서 델타 실행은 upsert 전에 델타를 기존
그래프 상태(description / `text_unit_ids` / frequency / weight)와 *합집합*합니다.
따라서 변경되지 않은 문서와 공유되는 엔티티도 그 문서들의 description과 청크
계보(`mix`가 따라가는 경로)를 유지합니다. 델타 실행마다 영향받은 엔티티와 관계를
그래프에서 먼저 읽어 오고, 병합된 description이 요약 예산을 넘으면 다시
요약합니다. `false`로 설정하면 영향받은 필드를 델타 값으로 덮어쓰며, 이때
엔티티는 변경되지 않은 문서 청크로의 계보를 잃습니다. read-back을 지원하는
그래프 어댑터가 필요합니다(지원하지 않으면 덮어쓰기로 동작). 문서가 변경되거나
삭제되면 다른 문서와 공유하는 엔티티와 관계의 `text_unit_ids`에서 그 문서의 청크가
빠지고 frequency와 weight도 다시 계산됩니다. 다만 그 문서가 기여한 description은
전체 재구축 전까지 남습니다.

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
| `--max-failure-rate` | `1.0` | 답변 생성에 실패한 질문의 비율, 또는 지표별로 시도한 값 중 실패한 비율(`metric_outcomes`, 건너뛴 값은 제외)이 이 값(0.0-1.0)을 넘으면 0이 아닌 코드로 종료. 모든 질문이 실패했거나 한 지표의 모든 시도가 실패한 실행은 항상 0이 아닌 코드로 종료 |
| `--verbose`, `-v` | off | 디버그 로깅 |
| `--config-path` | — | `config.yaml` 경로 |

### 평가자

`evaluation.enabled_evaluators`로 선택합니다. 활성화한 평가자를 만들 수 없거나
(예: judge용 Bedrock 접근 불가) 설정이 잘못되면 실행을 멈춥니다.
`processing.ignore_errors: true`이면 해당 평가자를 빼고 계속하며
`run_manifest.dropped_evaluators`에 기록합니다. 채점 중에도 같습니다.
평가자 오류(예: judge 호출 실패)가 나면 실행을 멈추고, `ignore_errors: true`이면
해당 지표를 실패로 기록합니다. judge 응답에 쓸 수 있는 점수가 없으면 0점이 아니라
항상 실패로 기록합니다. 기본값은 다섯 개 모두
활성화입니다. 결정적이고 LLM이 필요 없는 평가자(`answer_match`, `retrieval`,
`graph_aware`)는 비용이 없고 필요한 데이터셋 필드가 없는 질의는 건너뜁니다. LLM
judge 없이 실행하려면 `enabled_evaluators: [answer_match, retrieval, graph_aware]`로
설정합니다.

- **`langchain`** — LangChain 기반 텍스트 유사도(`langchain_metrics`:
  `correctness`, `partial_correctness`). `answer` 정답이 필요합니다.
- **`ragas`** — RAGAS 지표(`answer_correctness`, `answer_relevancy`,
  `context_precision`, `context_recall`, `faithfulness`). `context_precision`은
  컨텍스트마다 판정 호출을 한 번씩 하므로 질의마다 순위 상위 소스를 최대
  `evaluation.ragas_max_contexts`개(기본값 20, `null`이면 제한 없음)까지만
  채점하며, 이 제한은 `max_context_tokens` 예산보다 먼저 적용됩니다. 즉 이 지표는
  `context_precision@N`입니다. 제한이 없으면 소스를 100개 이상 보고하는 전략(예:
  LightRAG `mix`)은 `ragas_timeout`에 걸립니다. `faithfulness`와
  `context_recall`은 `max_context_tokens` 안의 모든 소스를 보므로, 순위가 낮은
  소스가 뒷받침하는 주장도 근거 없음으로 처리되지 않습니다. 각 리포트에는 판정
  모델이 본 범위(`judge_contexts` / `judge_context_tokens`,
  `context_precision_contexts` / `context_precision_context_tokens`)가 기록됩니다.
  답변 모델이 본 컨텍스트는 바뀌지 않습니다.
- **`graph_aware`** — 결정적이고 **LLM 불필요**한 엔티티/관계 **커버리지 =
  recall**: 기대되는 그래프 아티팩트 중 몇 개가 생성된 답변에 나타나는지
  (`answer_contains`와 같은 정규화·구문 매칭 사용: 단어 단위 매칭, 한국어 조사
  허용, 한 단어짜리 CJK 텍스트는 부분 문자열 매칭).
  `{"source": "A", "target": "B"}` 또는 `"A -> B"` 형식의 관계는 답변이 양 끝
  엔티티를 모두 언급하면 매칭으로 보고, 그 밖의 문자열은 구문 그대로 나타나야
  합니다. 데이터셋에 `expected_entities` / `expected_relationships`가 필요합니다. **precision과 F1은 의도적으로
  미산출**됩니다 — 자유 텍스트 답변에서 모든 엔티티를 열거하는 것은 신뢰성 있게
  불가능하므로, precision/F1을 보고하는 것은 recall 신호에 다른 이름표만
  붙이는 셈이기 때문입니다.
- **`retrieval`** — 결정적이고 LLM 불필요: 답변 모델이 본 소스에 정답 문서가
  포함됐는지를 `reference_sources`와 비교해 `hit_at_k`, `recall_at_k`(k =
  `evaluation.retrieval_k`, 기본값 5), `mrr`로 산출합니다. 텍스트 유닛 소스는 파일
  이름을 직접 갖고, 엔티티·관계·커뮤니티 리포트 소스는 계보(`text_unit_ids`)에
  있는 텍스트 유닛의 파일로 귀속합니다(체인의 문서 저장소에서 접미사별로 한 번에
  조회). 커뮤니티 리포트의 계보는 커뮤니티 전체이므로 `global`/`drift` 점수는
  답변 모델이 실제로 읽은 범위의 상한입니다. 순위는 귀속 가능한 소스만으로 매기며,
  `attributable_fraction`(귀속 가능한 소스 / 보고된 소스, `grouped_statistics`에서
  전략별로도 제공)으로 순위 지표가 컨텍스트를 얼마나 보는지 알 수 있습니다.
  매칭은 대소문자를 구분하지 않고 전체 이름으로 하며, 이름이 파일 확장자(`.` +
  영문자·숫자 1-5자, 영문자 1개 이상)로 끝나면 stem으로도 비교합니다:
  `docs/Terms.pdf` = `terms.pdf` = `terms`. `/`는 경로처럼 보이는 값(파일 확장자,
  URI 스킴, `/`·`./`·`~/`로 시작)에서만 디렉터리 구분자로 보므로 `St. Louis
  Cardinals`, `AC/DC` 같은 제목은 통째로 비교합니다. `reference_sources`가
  없거나, 소스는 있지만 귀속 가능한 소스가 하나도 없으면 순위 지표를 건너뜁니다.
  소스를 하나도 검색하지 못한 질의는 0점(miss)입니다.
- **`answer_match`** — 결정적이고 LLM 불필요한 답변 지표를 `answer`와 선택 항목
  `metadata.answer_aliases`에 대해 계산하고 최댓값을 씁니다. 대표 결정적 지표는
  **`answer_contains`**(정답이나 별칭이 생성된 답변에 단어 단위 구문으로 나타나면
  1.0)입니다. RAG 답변은 길어서 짧은 정답과 정확히 일치하는 경우가 드물기 때문에
  exact match보다 정답 여부를 훨씬 잘 반영합니다. 한국어 조사는 허용하고(정답
  `서울 특별시`가 `서울 특별시는`과 매칭), 한 단어짜리 CJK 정답은 부분 문자열로
  비교합니다. 공개 벤치마크와 비교할 수 있도록 SQuAD 방식 `exact_match`와
  `token_f1`도 함께 산출합니다. 텍스트는 NFKC로 정규화한 뒤 공식 SQuAD v1.1 스크립트와 같이
  소문자화, 문장 부호 삭제(`1,000` = `1000`), 영어 관사 제거, 공백 정리를
  거칩니다. 토큰 F1은 공백으로 나누므로 중국어/일본어 텍스트에서는 exact match와
  같아지고, 한국어는 조사(`서울은`) 때문에 두 지표 모두 낮게 나옵니다.

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

항목별 `metadata`(예: `search_strategy`)는 해당 질문에 대해 CLI 기본값을
오버라이드합니다. 코퍼스로 답할 수 없는 질문은 `"metadata": {"answerable": false}`로
표시합니다. 이런 질문은 어떤 평가자도 채점하지 않고, 체인이 답변을 거부했는지만
봅니다(아래 `abstention_statistics` 참고). `id`는 `query_id`로 줄 수도 있습니다. 파일은 질문을 실행하기
전에 검증합니다. 데이터셋이 비었거나, 배열이 아니거나, `question`이 없거나, ID가
중복되거나, 필드 타입이 틀리거나, RAG 체인이 거부할 `metadata` 값(예: 알 수 없는
`search_strategy`)이 있으면 항목 인덱스와 ID를 담은 오류로 실행을 멈춥니다.

### 예시

```bash
run-eval --eval-data-path my_eval_data.json --config-path config.yaml

run-eval --eval-data-path my_eval_data.json --outputs-directory ./results --config-path config.yaml

run-eval --eval-data-path my_eval_data.json \
  --search-strategy global --search-type vector --config-path config.yaml
```

결과는 출력 디렉터리에 `evaluation_{results,reports,summary}_<timestamp>.json`으로
기록되며, 답변한 모든 질문이 같은 전략을 썼다면 파일 이름이
`..._<strategy>_<timestamp>.json`이 됩니다. 요약에는 지표별
평균/중앙값/표준편차/최소/최대/개수(`metric_statistics`)와 scored/failed/skipped
개수(`metric_outcomes`)에 더해 다음이 담깁니다.

- `grouped_statistics` — 같은 통계를 `search_strategy`(실제로 사용한 전략이며
  `auto`에서는 질문마다 다를 수 있음), `category`, `difficulty`별로 나눈 값.
- `abstention_statistics` — 체인이 답변 대신 고정된 컨텍스트 없음 응답("I could
  not find relevant information…")을 돌려준 빈도: `abstained`, `answered`,
  `abstention_rate`, 같은 값의 `per_strategy`, 그리고 `answerable: false` 항목에
  대한 `unanswerable` 블록(`total`, `correct_abstentions`, `accuracy`). 답할 수
  있는 질문에서의 답변 거부는 일반 답변처럼 채점되며(대개 오답), 각 결과에는
  `abstained`가 기록됩니다.
- `run_manifest` — CLI 인자, 모델 ID(답변 생성, 평가 judge/임베딩), 패키지 버전,
  git 커밋(`git_sha`, 추적 파일이 커밋과 다르면 `git_dirty`가 true. 패키지를
  추적하는 체크아웃에서 실행하지 않으면 둘 다 `null`), 전체 확정 설정의
  `config_sha256`, `library_versions`(ragas, langchain*), 데이터셋(경로와 파일
  sha256, 질문 수, 파싱된 내용의 해시), UTC 타임스탬프. 두 실행을 비교할 때
  사용합니다. 제외된 평가자(`dropped_evaluators`)도 담깁니다.
  `EvaluationManager.evaluate_dataset`이 만들기 때문에 라이브러리로 호출해도
  포함됩니다(`dataset_path=` / `cli_args=`를 넘기면 함께 기록).

질의별 리포트에는 평가자별 `metrics`가 담깁니다. `overall_score`는 JSON 호환을
위해 남겨 두었지만 항상 `null`입니다(서로 다른 지표의 평균은 의미가 없음). 각
결과에는 `retrieved_source_ids`(보고된 소스별로 귀속된 파일 이름, 순위 순,
귀속할 수 없으면 `[]`)도 기록됩니다.

---

## 7. 시각화 (`run-visualization`)

이것은 이미 내보낸 시각화 데이터 JSON으로부터 그리는 **독립형** 렌더러입니다.
인제스천을 다시 실행하거나 AWS를 건드리지 **않습니다**.

`graph.visualization.enabled`가 `true`이면 `run-ingestion`의 커뮤니티 탐지
단계가 시각화를 렌더링하면서 `graph.visualization.outputs_directory`에
`visualization_data.json`도 기록합니다. 이 값을 설정하지 않으면(기본값)
인제스천은 `<cache.local_directory>/<pipeline_id>/visualization/`에 기록하므로,
S3 캐시 동기화를 켜면 `visualization_data.json`이 캐시와 함께 업로드됩니다(동기화는
`.json` 파일만 복사하므로 HTML은 `run-visualization`으로 로컬에서 다시
렌더링합니다). 이 파일이 `--data-path` 입력입니다. 인터랙티브 그래프는 브라우저에서
열 수 있도록 연결 수 기준 상위 `interactive.max_nodes`개 노드(기본 2000)만
남깁니다. 그래프 노드·엣지, 계산된 `layout`, 커뮤니티 계층, 중심성을 담으며,
파일 크기를 줄이기 위해 벡터 속성(`embedding`, `*_embedding`)은 제외합니다.

레이아웃과 오류 처리:

- `embedding_method: "none"`은 spring 레이아웃을 쓰며 Bedrock 임베딩
  클라이언트를 만들지 않습니다.
- `embedding_method: "node2vec"`은 각 노드의 `name: description`을 Bedrock으로
  임베딩한 뒤 `layout_method`로 차원을 축소합니다. 임베딩이 실패하면
  `processing.ignore_errors`가 `false`일 때 시각화 단계가 실패합니다(인제스천은
  계속 진행하고 실패를 로그에 남깁니다). `ignore_errors: true`이면 ERROR 로그를
  남기고 그래프 구조만 반영하는 spring 레이아웃으로 대체하며,
  `visualization_data.json`에 `"layout_degraded": true`를 기록합니다. 차원
  축소가 실패해도 `ignore_errors`와 관계없이 같은 방식(시드를 고정한 spring
  레이아웃, `layout_degraded: true`)으로 대체합니다.
- `embeddings.bedrock_model_id`는 지원하는 임베딩 모델 ID 중 하나여야 하며, 다른
  값은 설정을 읽을 때 거부합니다.
- 인터랙티브 그래프의 엣지 두께·불투명도는 그래프 자체의 가중치 범위를
  기준으로 조정합니다(로그 스케일 후 min-max 정규화). 따라서 1–10 강도 점수와
  병합 횟수 모두 구분됩니다.
- 렌더러가 실제로 기록한 파일만 보고합니다(같은 이름의 이전 실행 파일은 먼저
  삭제합니다). 아무것도 렌더링하지 못하면(예: 빈 그래프) `run-visualization`은
  오류를 로그에 남기고 종료 코드 `1`을 반환합니다.

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
| `--source-directory` | **필수** | 문서 디렉터리 (`.txt`/`.md`/`.markdown`과 `run-ingestion`이 파싱하는 모든 포맷, 예: `.pdf`); `--source-dir`도 별칭으로 허용 |
| `--output` | `tuned_prompts.yaml` | 출력 YAML 경로 |
| `--max-docs` | `20` | 샘플링할 최대 문서 수 |
| `--config-path` | — | `config.yaml` 경로 |

```bash
run-prompt-tuning --source-directory ./source --output tuned_prompts.yaml --config-path config.yaml
```

출력 YAML에는 `custom_prompts` 블록(과 감지된 도메인이 담긴 `profile`)이
포함됩니다. **검토한 후** 원하는 프롬프트를 `config.yaml`의 `custom_prompts:`
아래로 복사하세요. 일반 텍스트 파일은 그대로 읽고, 그 밖의 포맷(PDF, CSV, JSON,
`ParserFactory.register_loader`로 등록한 포맷)은 인제스션과 같은 로더로 파싱합니다.
파싱에 실패한 파일은 경고를 남기고 건너뜁니다. 프로파일링 모델이 JSON
프로파일을 돌려주지 않으면 아무것도 쓰지 않고 0이 아닌 코드로 종료합니다(기본
프로파일은 튜닝된 것처럼 보이기 때문입니다).

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
`drift_primer`(DRIFT primer, `enable_primer` 설정 시),
`strategy_selection`, `keywords_extraction`(LightRAG 이중 레벨),
`global_map`(글로벌 검색 map-reduce), 그리고 프롬프트 튜닝
프롬프트(`corpus_profile`).

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
  쓰기는 `indexing.neptune.index_concurrency`로 높이며, Gremlin 연결 풀도 이
  값만큼 함께 늘어납니다. 재실행과 다단계 작업이 재계산하지 않도록
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
이상), `gleaning`(청크당 최대 `max_rounds`회 호출), `community_detection` 리포트
생성, `claim_extraction`(텍스트 유닛당 1회 호출 — 기본 OFF), 질의당 답변 생성.
레버: 기계적인 스테이지에 더 저렴한 모델 사용(청킹 / 번역 / map-reduce /
description 요약은 이미 Haiku급 모델이 기본값), `gleaning.max_rounds` 제한,
필요하지 않으면 `claim_extraction` OFF 유지, 임베딩/스테이지 캐싱 활성화, 전체
재인제스천을 피하기 위한 증분 인덱싱 사용.

---

*함께 보기: 개요와 빠른 시작은 [README.md](../README.md), 아키텍처와 내부
구조는 [docs/design.md](./design.md)를 참고하세요.*
