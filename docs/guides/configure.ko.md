# 설정

> 🇬🇧 English version: [configure.md](./configure.md)

템플릿으로 설정 파일을 만들고 모든 CLI에 `--config-path config.yaml`로 지정하세요.

```bash
cp config-template.yaml config.yaml
```

설정 파일에는 바꿀 키만 적고, 나머지 키는 기본값을 유지합니다. 모든 키의 타입, 기본값,
설명은 코드에서 생성한 [설정 레퍼런스](../reference/configuration.md)(영문)에 있고, 이 문서에서는 표
한 줄로 설명하기 어려운 내용을 다룹니다. 모델 선택은 모델 카탈로그의
[모델 선택 참고 사항](../reference/models.md#model-selection-notes)(영문), 질의와 답변 검사는
[가드레일](./guardrail.md)(영문), 추적은 [관측성](./observability.md)(영문)에 있습니다.

## 설정을 읽는 순서

뒤 단계가 앞 단계를 덮어씁니다.

1. **내장 기본값**(`unified_kg_rag/domain/models/config/`). `--config-path` 없이
   실행하면 CLI는 이 기본값에 환경 변수만 적용합니다.
2. **`backend` 프리셋**: `backend: local`(또는 `--backend local`)은 파일과 환경 변수가
   지정하지 않은 모든 키에 로컬 공급자와 저장소를 채웁니다([로컬 모드](./local-mode.md), 영문).
3. **YAML 파일**. 값이 딕셔너리인 키(예: `logging.library_levels`)는 기본 딕셔너리와
   병합되지 않고 통째로 대체됩니다.
4. **환경 변수**([아래](#환경-변수로-덮어쓰기)).
5. **공급자 플래그**(`--backend`, `--llm-provider` 등)는 마지막에 적용됩니다. 설정을
   파일과 환경 변수가 지정한 값으로 다시 만들므로, `--backend`를 바꾸면 이전 프리셋이
   채운 값만 사라지고 사용자가 지정한 값은 남습니다.

파일은 읽을 때 검증합니다. 타입이나 값이 틀린 경우(`Configuration validation error: ...`),
올바른 YAML이 아닌 파일, 존재하지 않는 `--config-path` 파일, 알 수 없는 키(파일이나
`UKR__` 변수)는 사용법 오류(종료 코드 2)입니다. 철자가 틀린 키는 아무 표시 없이 설정을
기본값으로 남기므로 알 수 없는 키를 거부하며(Python의 `Config.model_validate`도 거부합니다),
`--allow-unknown-keys`(`get_config`에서는 `allow_unknown_keys=True`)를 주면 키마다 WARNING을
남기고 그 키를 버립니다. 위치가 바뀐 키는 대신할 키를
알려 주며 실패합니다([Moved keys](../reference/configuration.md#moved-keys), 영문).

재시도는 `retry` 섹션 하나에서 설정합니다. `max_attempts`와 백오프는 모델 호출(수집 항목,
Bedrock 임베딩, 질의 시점 호출)에 적용되고, 기본값이 다른 호출자(모델 서버, 그래프 저장소
인덱서, OpenSearch 질의, RAGAS 평가 모델)는 시도 횟수를 따로 둡니다. 시도 횟수는 모두 첫
시도를 포함하며, `1`이면 재시도하지 않습니다.

## 대부분의 실행에서 바꾸는 설정

| 키 | 기본값 | 바꿀 때 |
|---|---|---|
| `aws.region_name` | `"us-west-2"` | 저장소가 다른 리전에 있을 때(`aws.bedrock.region_name`이 없으면 Bedrock도 따름) |
| `aws.neptune.endpoint`, `aws.opensearch.endpoint` | `null` | AWS에서는 항상(CDK 스택이 주입함) |
| `providers.doc_status.enabled` | `false` | [증분 인덱싱](../guides/incremental.ko.md) |
| `models.default`, `models.fast` | Claude Sonnet 5.5, Claude Haiku 5.5 | 다른 모델([참고 사항](../reference/models.md#model-selection-notes), 영문) |
| `processing.max_concurrency` | `40` | `ThrottlingException` 재시도가 보이면 낮추고, 할당량이 허용하면 높임 |
| `processing.graph_extraction.entity_types` | 일반 유형 7개 | [도메인 적응](./domain-adaptation.ko.md#엔티티-유형) |
| `processing.translation.target_language` | `"en"` | 코퍼스나 답변 언어가 다를 때([다국어 코퍼스](./multilingual.ko.md)) |
| `processing.claim_extraction.enabled` | `false` | `local` 검색에 주장을 넣을 때(텍스트 단위마다 LLM 호출 1회 추가) |
| `processing.document_parsing.index_value` | `null`(`default`) | 코퍼스나 테넌트 분리([인덱스 접미사](../concepts/scopes-and-namespaces.ko.md#인덱스-접미사)) |
| `indexing.methodologies` | `[graphrag, lightrag]` | 한 방법론이 읽는 것만 만들 때. `[lightrag]`는 커뮤니티 리포트를 호출하지 않음([방법론 프로필](./ingest.ko.md#방법론-프로필)) |
| `graph.community_detection.enabled` | `true` | `false`면 `indexing.methodologies: [lightrag]`처럼 커뮤니티를 만들지 않음(GraphRAG 전략에는 필요) |
| `providers.rerank.kind` | `bedrock` | `none`: 리전에 재순위화 모델이 없거나 재순위화를 쓰지 않을 때 |
| `search.token_manager.max_context_tokens` | `30000` | 컨텍스트를 늘리거나 비용을 줄이려고 줄일 때([컨텍스트 예산](../concepts/retrieval.ko.md#컨텍스트-예산)) |
| `indexing.reset` | `false` | 코퍼스 전체로 저장소를 다시 만드는 한 번의 실행(격리된 문서는 건너뜀) |
| `aws.bedrock.guardrail.identifier` | `null` | 질의와 답변 검사([가드레일](./guardrail.md), 영문) |
| `observability.enabled` | `false` | OpenTelemetry 트레이스([관측성](./observability.md), 영문) |

## 환경 변수로 덮어쓰기

모든 설정 키는 `UKR__` 뒤에 키 경로를 붙인 환경 변수로 지정할 수 있습니다. 경로의
마디 사이에는 밑줄 두 개를 쓰고 대소문자는 가리지 않습니다.
`UKR__AWS__NEPTUNE__ENDPOINT`는 `aws.neptune.endpoint`를 지정하며, 자유 형식 매핑
다음 마디는 그 매핑의 키를 가리킵니다(`UKR__INDEXING__OPENSEARCH__LANGUAGE_ANALYZERS__KO`).
환경 변수는 설정 파일과 기본값보다 우선합니다. 값은 파일에 같은 값을 쓴 경우처럼
변환하고 검증합니다(불리언은 대소문자 구분 없이 `true`/`1`/`yes`/`on` 또는
`false`/`0`/`no`/`off`, 목록이나 매핑은 JSON, 예:
`UKR__SEARCH__AUTO_ROUTABLE_STRATEGIES='["local", "mix"]'`). 값이 잘못되었거나 어떤
키도 가리키지 않는 `UKR__` 변수는 그 변수를 밝힌 사용법 오류(종료 코드 2)이며,
`--allow-unknown-keys`를 주면 알 수 없는 변수는 WARNING만 남깁니다.

- `AWS_PROFILE`과 `AWS_REGION`은 파일이 `aws.profile_name`과 `aws.region_name`을
  지정하지 않았을 때만 그 값을 채우므로, 셸이 `AWS_REGION`을 내보내도(SSO 도우미와
  CloudShell이 그렇게 함) 파일에 적은 리전은 바뀌지 않습니다. 이때
  `aws.bedrock.region_name`이 `null`이면 `AWS_REGION`이 Bedrock 호출 리전도 정합니다.
  `AWS_DEFAULT_REGION`은 읽지 않습니다.
- `UKR_PLUGINS`는 CLI가 설치된 `unified_kg_rag.plugins` 진입점(entry point) 대신 불러올
  플러그인 모듈을 지정합니다([Loading plugins](./extend.md#loading-plugins), 영문).
- `UKR_SOURCE_DIRECTORY`, `UKR_PIPELINE_ID`(설정 키가 아니므로 밑줄 하나)와
  `UKR_METRICS_ENVIRONMENT`는 `run-ingestion`이 읽습니다([CLI 레퍼런스](../reference/cli.md#run-ingestion), 영문).
  CDK 태스크가 받는 변수는
  [`iac/README.md`](../../iac/README.md#2-build-and-push-the-app-image)(영문)에 있습니다.
- CLI는 python-dotenv로 현재 작업 디렉터리의 `./.env`만 읽고 상위 디렉터리의 파일은
  찾지 않습니다(그러지 않으면 `AWS_PROFILE`을 설정한 `~/.env`가 모르는 사이에 계정을
  바꿀 수 있음). 이미 설정된 변수가 우선합니다. 패키지를 가져오는 것만으로는 `.env`를
  읽지 않고 `os.environ`을 바꾸지 않으므로, Python 코드에서는 직접 읽습니다
  (`get_config()` 전에 `dotenv.load_dotenv()`).

> **LangSmith 추적은 내용을 업로드합니다.** `LANGSMITH_TRACING=true`(또는
> `LANGCHAIN_TRACING_V2=true`)이면 LangChain은 프롬프트, 검색된 본문, 모델 출력을 포함해
> 추적한 실행을 모두 LangSmith로 보냅니다. CLI는 시작할 때(`setup_logging`에서)
> WARNING을 남기지만 끄지는 않으므로, 기밀 코퍼스로 실행하기 전에 이 변수를 해제하세요.

## 로그에 남는 내용

`INFO` 이상에서 사용자 텍스트와 코퍼스 텍스트는 길이, 개수, ID, 항상 같은 짧은
해시(`query: len=42 sha=1a2b3c4d`)로만 기록합니다. 질의 원문, 다시 쓴 질의, 엔티티
이름, 모델 원본 출력은 `DEBUG`에서만 기록하며, 예외 메시지에는 모델 출력을 넣지
않습니다. 모델 제공자나 라이브러리의 텍스트가 담길 수 있는 오류(단계가 흡수한 모델 호출
오류, 어느 화면에서든 질의의 예기치 않은 실패)는 `INFO` 이상에서 유형, 다이제스트와 발생
위치(유형 연쇄와 `file:line` 프레임, 메시지는 제외)로만 기록하고, 텍스트와 traceback은
`DEBUG`에서만 기록합니다. 저장소나 모델에 닿지 않는 경우, 제한 시간 초과, 거부처럼 자체
메시지가 있는 예상된 결과는 텍스트와 함께 기록합니다. `DEBUG` 로그(`logging.level: DEBUG`, `UKR__LOGGING__LEVEL=DEBUG`, 모든 CLI의
`--log-level DEBUG`. `run-rag --verbose`는 결과를 더
자세히 출력할 뿐 로그 수준은 바꾸지 않음)와 `logging.library_levels`에서 `DEBUG`로 둔
라이브러리(`botocore`는 요청 본문을 기록함)에는 사용자 데이터와 코퍼스 데이터가 들어 있다고
보세요.

모든 CLI는 `--log-level`(`logging.level`보다 우선)과 `--version`을 받습니다.
`--output-format text`에서는 `logging.log_format`을 지정하지 않으면 콘솔 로그가 일반 텍스트
줄로 나옵니다(CDK 작업은 CloudWatch용 JSON 레코드인 `structured`를 지정). `--output-format
json`에서는 stderr에 JSON 레코드로 나옵니다.

`logging.log_to_file: true`로 두면 CLI는 `logging.log_file_path`(`logs/log.txt`,
날짜별 `log_YYYYMMDD.txt`, 작업 디렉터리 기준이라 그 디렉터리에 쓸 수 있어야 함)에도
씁니다. 기본값은 꺼짐입니다. 라이브러리로 쓰면 패키지는 핸들러를 설정하지
않으며, 호스트의 로깅 설정이 적용되고 구조화된 필드(`pipeline_id`, `trace_id` 등)는
로그 레코드의 속성으로 들어갑니다.

## S3 캐시 암호화

`aws.s3.encryption.encryption_type: BUCKET_DEFAULT`는 암호화 헤더를 보내지 않으므로
버킷 기본 암호화가 적용됩니다(CDK 스택의 `use_cmk=true`면 CMK이며, `AES256`을 강제하면
이 키를 우회함). SSE-KMS 버킷을 재사용하면 쓰기 주체에 그 키의 `kms:GenerateDataKey`와
`kms:Decrypt` 권한이 필요합니다.
