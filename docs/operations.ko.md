# 운영 런북

> 🇬🇧 English version: [docs/operations.md](./operations.md)

용어는 [용어집](./glossary.ko.md)을 따릅니다.

이미 배포된 환경에서 운영자가 하는 작업을 정리합니다. 다시 수집하기, 증분
실행, 실패한 문서 처리, 캐시 관리, CLI가 내는 오류 해석이 대상입니다. 설정 키는
[사용자 가이드](./user-guide.ko.md) §2에, 배포 스택과 명령은
[`iac/README.md`](../iac/README.md#after-deploy)에 있습니다.

## 실행 전 확인

1. **리전과 자격 증명.** 환경 변수 `AWS_REGION`은 `aws.region_name`보다
   우선하며, `aws.bedrock.region_name`이 비어 있으면 Bedrock 리전도 바꿉니다.
   `env | grep -E '^(AWS_REGION|BEDROCK_REGION)='`와
   `aws sts get-caller-identity`로 확인합니다.
2. **모델.** 사용자 가이드 §1의 "필수 모델"에 있는 모델이 Bedrock 리전에서
   활성화되어 있어야 합니다.
3. **엔드포인트.** `run-ingestion`, `run-rag`, `run-eval`은 시작할 때 모델을
   호출하기 전에 필요한 저장소 엔드포인트가 설정되어 있는지 검사합니다
   ([시작 시 엔드포인트 검사](#시작-시-엔드포인트-검사) 참고).
4. **같은 저장소에서 다른 수집이 실행 중이지 않아야 합니다.** 동시 실행은
   지원하지 않습니다([동시 실행](#동시-실행) 참고). CDK 스택에서는
   `aws stepfunctions list-executions --state-machine-arn <arn> --status-filter RUNNING`으로
   확인합니다.

## 동시 실행

수집 실행을 동시에 여러 개 돌리는 것은 지원하지 않습니다. 다음 대상에 대해 다른
실행이 진행 중이면 새 실행을 시작하지 마세요.

- **같은 인덱스 접미사**(같은 OpenSearch 별칭과 Neptune 레이블). 전체 재색인은
  타임스탬프가 붙은 새 인덱스에 쓰고, 별칭을 새 인덱스로 옮긴 뒤 그 별칭의 이전
  인덱스를 지웁니다. 그래서 한 실행이 다른 실행이 쓰고 있는 인덱스를 지우거나,
  별칭을 다른 실행의 인덱스에 남길 수 있습니다. 재구축(reset)은 다른 실행이
  쓰는 도중에 그 접미사의 저장소를 비웁니다.
- **같은 문서 상태 테이블 네임스페이스.** 증분 실행은 무엇을 지우고 어떤 레코드를
  남길지 계획할 때 레지스트리를 한 번만 읽습니다. 그 사이 다른 실행이 쓴 레코드는
  정리될 수 있고, 그 산출물은 계획과 어긋나게 지워지거나 남을 수 있습니다.

인덱스 접미사가 다르고 문서 상태 테이블(`aws.dynamodb.table_name`)도 따로 쓰는
실행은 어느 쪽도 공유하지 않으므로 함께 실행할 수 있습니다. 그 밖의 경우에는 한
실행이 끝난 뒤 다음 실행을 시작합니다.

CDK 스택에서 `start-execution` 한 번은 네 단계를 차례로 실행하는 실행 하나를
시작합니다. 스택에는 일정이나 이벤트 트리거가 없으므로 누군가(또는 직접 추가한
자동화가) `start-execution`을 호출할 때만 실행이 시작되지만, 상태 머신은 실행
중에 두 번째 실행이 시작되는 것을 막지 않습니다. 먼저 위 명령으로 `RUNNING`
실행이 있는지 확인하고 실행이 겹치지 않게 합니다. 일정을 추가한다면 이전 실행이
아직 돌고 있을 수 있는 동안(상태 머신 제한 시간 6시간까지)에는 실행되지 않게
설정합니다.

## 다시 수집하기

비슷해 보이지만 건드리는 상태가 다른 작업이 세 가지 있습니다.

| 목적 | 방법 | 단계 캐시 | 저장소와 레지스트리 |
|---|---|---|---|
| 중단된 실행 마무리 | 같은 `--pipeline-id`로 다시 실행합니다. 처음으로 실패했거나 끝나지 않은 단계부터 재개하며, `--resume-from-stage <stage>`로 단계를 지정할 수 있습니다. | 재사용 | 실행되는 단계가 기록 |
| 모든 단계 다시 계산 | `--force-rebuild` | 무시하고 새로 기록 | 평소대로 기록(활성화 시 증분) |
| 저장소를 처음부터 재구축 | 한 번의 실행에만 `indexing.reset: true` | 변경 없음 | 비운 뒤 코퍼스 전체를 인덱싱하고 모든 문서를 다시 기록 |

저장된 데이터에 반영되는 설정을 바꾼 뒤에는 `indexing.reset: true`를 쓰고, 단계
결과도 바뀌어야 하면 `--force-rebuild`를 함께 씁니다.

- 임베딩 모델 또는 `indexing.opensearch.embedding_dimension`
- `processing.translation.source_language` 또는 언어 분석기(텍스트 단위 매핑이
  바뀝니다)
- 증분 실행이 갱신하지 않는 설명과 커뮤니티를 새로 고칠 때
  ([증분 실행](#증분-실행) 참고)

끝나면 `indexing.reset`을 다시 `false`로 돌립니다. 그대로 두면 실행할 때마다
재구축합니다.

`--verify-metadata`와 `--repair-metadata`(둘 다 `--pipeline-id` 필요)는 실행의
단계 메타데이터를 검사하고 복구합니다. 실패하면 0이 아닌 종료 코드를 반환합니다.

## 증분 실행

`aws.dynamodb.enabled: true`(컨테이너 이미지의 기본값)이면 `run-ingestion`은
실행할 때마다 콘텐츠 해시로 코퍼스와 문서 상태 레지스트리를 비교하고, 새 문서와
바뀐 문서만 인덱싱합니다. 비교 범위는 실행의
인덱스 접미사(사용자 가이드 §3)와 소스 범위입니다. 소스 범위는
`processing.document_parsing.source_scope`이며 기본값은 해석된 소스
디렉터리이고, 컨테이너 엔트리포인트는 `s3://` 소스 URI로 설정합니다. 둘 다 문서의
레지스트리 키에 들어가므로, 한 접미사를 쓰는 두 코퍼스는 상대 경로가 같은 파일이
있어도 레코드를 공유하지 않습니다. 소스에서 사라진 파일은 삭제된 것으로 보고, 그
문서에만 속한 산출물을 지웁니다. 따라서 다음을 지킵니다.

- 한 코퍼스에는 같은 소스 디렉터리(또는 고정한 `source_scope`)를 씁니다.
  범위가 기본값인 코퍼스를 옮기면 범위가 바뀌고, 이전 범위의 레코드는 저절로
  넘겨받거나 지우지 않으므로 이전 내용(옮기는 중에 지운 파일 포함)이 인덱스에
  남습니다. 옮기기 전에 `source_scope`를 고정하거나, 새 위치에서
  `--retire-source-scope <이전 소스 디렉터리>`로 한 번 실행하세요. 이 실행이 이전
  범위에만 속한 내용과 레코드를 지웁니다(사용자 가이드 §5 참고). 새 문서가 있는
  실행은 같은 접미사의 다른 로컬 소스 디렉터리를 WARNING으로 알립니다.
- 코퍼스 파일이 의도와 다르게 만료되거나 사라지지 않게 합니다. 다음 실행이 해당
  그래프·벡터 데이터를 지웁니다.
- 테넌트나 코퍼스 버전마다 인덱스 접미사와 설정 파일을 따로 둡니다.

소스 범위가 생기기 전에 기록된 레지스트리를 업그레이드할 때: 범위의 첫 실행은
파일이 아직 코퍼스에 있는 이전 방식 레코드를 다시 추출하지 않고 넘겨받습니다.
그러나 범위가 없는 레코드는 삭제 대상으로 보지 않으므로, 업그레이드 전에
코퍼스에서 지운 파일의 레코드는 넘겨받지도 지우지도 않고, 그 산출물도 저장소에
남습니다. 이를 지우려면 `indexing.reset: true`로 한 번 실행합니다. 재구축은
해당 네임스페이스의 레코드뿐 아니라 비우는 접미사의 범위 없는 레코드도 모두
지웁니다(이런 레코드는 접미사마다 네임스페이스가 하나였던 시기에 기록됨). 해당 레지스트리
항목(`registry_scope` 속성이 없고 파일이 없는 항목)을 직접 지우면 레지스트리만
정리되고, 산출물은 재구축 전까지 남습니다.

증분 실행이 갱신하지 않는 것도 있습니다. 여러 문서가 공유하는 엔티티나 관계는
바뀌거나 삭제된 문서가 보탠 설명을 그대로 유지하고, 델타 실행은 그래프 전체를
다시 군집화하지 않고 커뮤니티를 덧붙입니다. 주기적인 전체 재구축
(`indexing.reset: true`)으로 둘 다 새로 고칩니다. 자세한 내용은
[사용자 가이드](./user-guide.ko.md) §5에 있습니다.

레지스트리를 읽을 수 없으면(테이블 없음, 접근 거부, 클라이언트 재시도로도 풀리지
않는 스로틀링) `document_loading` 단계가 테이블 이름과 원인을 담은
`DocStatusRegistryError`로 실패합니다. 모든 문서를 인덱싱하는 경로로 넘어가지
않습니다. 그 경로는 접미사의 인덱스 내용을 이번 실행의 문서만으로 바꾸기
때문입니다. 레지스트리를 고친 뒤 같은 `--pipeline-id`로 다시 실행하거나, 증분
인덱싱 없이 실행하려면 `aws.dynamodb.enabled: false`로 설정합니다.

### 중단된 실행

인덱싱 단계 도중 멈춘 증분 실행(태스크 강제 종료, 저장소나 레지스트리 연결 끊김,
실패 기준 미달)은 따로 손볼 필요 없이 다시 실행하면 됩니다. 인덱싱 단계는 무엇이든
지우거나 쓰기 전에 새 문서와 바뀐 문서를 `PENDING`으로 기록하며, 이 레코드에는
문서가 원래 가진 산출물과 이번 실행이 쓸 산출물이 모두 들어갑니다. 레코드는 쓰기가
끝난 뒤에야 `PROCESSED`(또는 `FAILED`)로 바뀝니다. 그래서 다음 실행은

- 코퍼스에 남아 있는 `PENDING` 문서를 다시 추출합니다. 파일을 이전에 인덱싱한
  버전으로 되돌렸어도 마찬가지입니다.
- 코퍼스에서 사라진 `PENDING` 문서에 대해 중단된 실행이 썼을 수 있는 것을 모두
  지웁니다.

중단은 실패로 세지 않지만, `indexing.max_document_failures`에 쓰는 연속 실패
횟수를 초기화합니다. `PENDING` 레코드의 `failure_count`는 0이므로, 이전에 실패한
문서도 다음 실패부터 다시 셉니다. 실패한 실행 뒤에 `PENDING` 레코드(`status`와
`content_hash`가 `pending`이고 `error_info`가 선기록 레코드임을 알림)가 남는 것은
정상입니다. 직접 지우지 마세요. 다음 실행은 이 레코드의 계보로 중단된 실행이 쓴
것을 찾습니다. 큰 문서의 이전 산출물 ID와 새 산출물 ID를 합치면 DynamoDB 항목
하나에 들어가지 않을 때는, 레코드에 들어가지 않은 ID를 같은 테이블의 초과 항목
(키 `<doc_id>#pending#<n>`, `record_kind`는 `lineage_overflow`)에 둡니다. 이
항목도 그대로 두세요. 실행이 문서를 커밋하거나 제거할 때 지웁니다. 인덱싱 단계는 레지스트리를 일괄로
쓰므로, 직접 작성한 정책을 쓰는 역할에는 `BatchGetItem`과 함께
`dynamodb:BatchWriteItem` 권한이 필요합니다(CDK 스택의 `grant_read_write_data`는
둘 다 포함합니다).

## 실패한 문서

번역, 그래프 추출, gleaning, 주장 추출 중 하나가 문서의 텍스트 단위 일부에서
실패하거나, 문서의 산출물(텍스트 단위, 엔티티, 관계, 주장, 커뮤니티, 보고서)을
OpenSearch나 Neptune에 쓰는 데 실패하면 그 문서는 `FAILED`로 기록됩니다. 다음
실행은 이 문서를 바뀐 문서로 취급해 실패한 실행이 쓴 내용을 지우고 다시
처리합니다.

- 내용이 그대로인 채로 `indexing.max_document_failures`(기본값 `3`)번 연속
  실패하면 더 이상 재시도하지 않습니다. 문서는 `FAILED`로 남아 인덱싱된 내용을
  유지하고, 실행할 때마다 파일 이름을 담은 WARNING이 남습니다.
- 다시 시도하려면 원인(대개 너무 크거나 형식이 잘못된 파일)을 고치거나, 파일을
  수정해 콘텐츠 해시를 바꾸거나, `indexing.max_document_failures`를 올립니다.
- 한 문서의 산출물 ID가 DynamoDB 항목 하나(400 KB, 약 10,000개 ID)를 넘으면
  인덱싱 단계가 델타에 대해 아무것도 쓰기 전에, 파일 이름과 ID 개수를 담은
  오류로 실패합니다. 파일을 나눕니다.

쓰기 실패는 따로 판정합니다. 재시도할 수 있는 상태(429, 502, 503, 504)로
거부된 OpenSearch bulk 항목은 먼저 백오프를 두고 최대 네 번 다시 보냅니다. 그래도
한 산출물 유형의 쓰기 중 실패 비율이 `indexing.max_failure_rate`(기본값 `0.2`)를
넘으면 인덱싱 단계가 실패하고 문서를 기록하지 않으므로, 다음 실행이 다시
시도합니다. 그 이하이면 실행은 성공하고, 실패한 산출물을 가진 문서만 `FAILED`로
기록됩니다. 백엔드가 항목 ID 없이 보고한 실패가 있으면 그 실행의 모든 문서를
`FAILED`로 기록합니다. CDK 스택에서는 다음 세 경보가 SNS 토픽으로 문제를
알립니다.

| 경보 | 발생 조건 |
|---|---|
| `PipelineFailures` | Step Functions 실행이 실패함 |
| `IndexingFailures` | 실행이 성공했더라도 인덱싱에서 실패한 항목이 있음 |
| `ExtractionFailures` | 추출, gleaning, 주장 추출이 일부 텍스트 단위에서 실패함 |

경보를 받으려면 `-c alarm_email=<address>`로 배포해 구독자를 등록해야 합니다.

## 캐시와 접두사

| 캐시 | 위치 | 참고 |
|---|---|---|
| 단계 체크포인트(로컬) | `<--cache-directory>/<pipeline_id>/` (기본값 `cache/`) | 재개할 때 재사용하며, `--force-rebuild`는 무시합니다 |
| 단계 체크포인트(S3) | `s3://<bucket>/<--s3-prefix>/<pipeline_id>/` (기본 접두사 `pipeline-runs`) | `--s3-sync` 사용 시. Step Functions 단계 사이의 인계에 쓰이며, 동기화가 실패하면 `CacheSyncError`로 실행이 실패합니다 |
| 임베딩 캐시 | `s3://<aws.s3.bucket_name>/embedding-cache/cache.json` (`indexing.opensearch.embedding_cache_s3_key`) | `indexing.opensearch.persist_embedding_cache: true`일 때만 |
| 시각화 데이터 | `graph.visualization.outputs_directory`가 없으면 `<cache directory>/<pipeline_id>/visualization/` | 단계 캐시와 함께 동기화(`.json`만) |

CDK 캐시 버킷은 `pipeline-runs/`와 `embedding-cache/` 아래의 객체만 30일 뒤에
만료합니다. 코퍼스는 다른 접두사(예: `corpus/`)에 둡니다. 만료된 코퍼스 파일은
다음 증분 실행에서 삭제된 문서로 보입니다. 기존 버킷을 재사용하면
(`cache_bucket_name`) 그 버킷의 수명 주기 규칙이 그대로 적용되므로 확인합니다.

`pipeline_id`는 캐시 디렉터리와 S3 접두사 이름이 되므로 영문 소문자, 숫자,
하이픈, 밑줄만 쓸 수 있습니다.

## 시작 시 엔드포인트 검사

CLI는 첫 유료 모델 호출 전에 실행에 필요한 저장소 엔드포인트가 설정되어 있는지
검사하고, 없으면 종료 코드 1로 끝납니다.

```text
Error: Missing endpoint configuration for the indexing stage: aws.neptune.endpoint (env NEPTUNE_ENDPOINT), aws.opensearch.endpoint (env OPENSEARCH_ENDPOINT). Set them in the config file or the environment.
```

- `run-ingestion`은 `indexing` 단계가 켜져 있으면 두 엔드포인트를 모두
  검사합니다(`--verify-metadata` / `--repair-metadata`는 제외).
- `run-rag`와 `run-eval`은 선택한 전략의 검색기가 쓰는 엔드포인트를 검사합니다.
  모든 전략에 OpenSearch가 필요하고, `local`, `drift`, `mix`, `hybrid`는
  Neptune도 필요합니다. `auto`는 `search.auto_routable_strategies`의 모든 전략을
  기준으로 검사합니다.

이 검사는 값이 설정되어 있는지만 봅니다. 값이 틀렸거나 연결할 수 없는
엔드포인트는 첫 연결에서 실패하며, 검색기는 인증·설정·연결 오류를 빈 결과로
감추지 않고 그대로 보고합니다.

## 자주 보는 오류

| 메시지(요약) | 의미 | 조치 |
|---|---|---|
| `Missing endpoint configuration for ...` | 실행에 필요한 엔드포인트가 비어 있음 | 메시지에 나온 키나 환경 변수를 설정 |
| `Configuration validation error: ...` | 설정값의 타입이나 값이 잘못됨 | 메시지에 나온 키를 수정 |
| `Unknown config key '<path>' is ignored` (WARNING) | 오타이거나 없어진 키이며, 실행은 그 키 없이 계속됨 | 키를 고치거나 삭제 |
| `No valid AWS credentials for Amazon Bedrock in region ...` | 자격 증명이 없거나 만료됨 | 자격 증명을 갱신하고 `AWS_PROFILE` / `aws.profile_name` 확인 |
| `Model '<id>' is only available through a cross-region inference profile, but none resolved ...` | 온디맨드 처리량이 없는 모델인데 추론 프로파일을 찾지 못함 | `aws.bedrock.enable_global_profile: true`를 유지하고 `bedrock:ListInferenceProfiles`를 허용하거나 다른 리전 사용 |
| `Reranking failed: ...` (질의마다 ERROR) | Rerank 호출이 거부되었거나 해당 리전에 모델이 없음 | `bedrock:Rerank`를 `Resource: "*"`로 허용하거나, rerank 모델이 있는 리전으로 Bedrock을 옮기거나, `search.reranking.enabled: false` 설정 |
| `No indices found for suffix '<suffix>' ...` | 그 인덱스 접미사로 수집한 데이터가 없음 | 수집을 실행하거나, 코퍼스를 수집할 때 쓴 접미사로 질의 |
| `InvalidFilterError` | 전략이 읽는 어떤 저장소도 그 필터 키를 선언하지 않음 | 메시지에 나온 키 목록에서 선택 |
| `Skipping N '.md' file(s) ...` / `No supported source files found in '<dir>'` | `unstructured` 추가 패키지가 설치되지 않음 | 설치(Python 3.11 이상)하거나 파일 형식을 변환 |
| `DocStatusRegistryError` | 문서 상태 레지스트리 테이블이 없거나, 접근할 수 없거나, 스로틀링됨 | 테이블을 만들거나 `aws.dynamodb.table_name`을 고치고, 메시지에 나온 DynamoDB 권한을 부여한 뒤 같은 `--pipeline-id`로 재개 |
| `Incremental indexing is enabled but no document delta was computed` | 레지스트리 비교 없이 인덱싱 단계가 실행됨(`continue_on_error`로 로딩 단계 실패 뒤 진행) | 레지스트리에 접근할 수 있게 한 뒤 `document_loading`부터 다시 실행 |
| `CacheSyncError` | S3 단계 캐시 다운로드나 업로드가 실패함 | 버킷 권한과 `aws.s3.encryption`을 확인한 뒤 같은 `--pipeline-id`로 재개 |
| `Pipeline failed at stage(s): ...` | 단계가 실패해 CLI가 1로 종료함 | 해당 단계 로그를 확인해 고친 뒤 같은 `--pipeline-id`로 재개 |
| 프라이빗 VPC에서 Bedrock 호출이 멈춤 | VPC 엔드포인트가 처리하지 않는 리전으로 Bedrock을 호출함 | `aws.bedrock.region_name`을 비우거나 VPC 리전과 같게 설정 |

다른 메시지와 해결 방법은 [사용자 가이드](./user-guide.ko.md) §10에 있습니다.
