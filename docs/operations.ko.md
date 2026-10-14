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
아직 돌고 있을 수 있는 동안(상태 머신 제한 시간 `sfn_timeout_hours`, 기본 6시간까지)에는 실행되지 않게
설정합니다.

## 다시 수집하기

비슷해 보이지만 건드리는 상태가 다른 작업이 세 가지 있습니다.

| 목적 | 방법 | 단계 캐시 | 저장소와 레지스트리 |
|---|---|---|---|
| 중단된 실행 마무리 | 같은 `--pipeline-id`로 다시 실행합니다(CDK 스택에서는 같은 `pipeline_id`로 새 실행을 시작). 처음으로 실패했거나 끝나지 않은 단계부터 재개하며, Step Functions의 각 단계 묶음은 이미 완료된 단계를 건너뜁니다. `--resume-from-stage <stage>`로 단계를 지정할 수 있습니다. | 재사용 | 실행되는 단계가 기록 |
| 모든 단계 다시 계산 | `--force-rebuild` | 무시하고 새로 기록 | 평소대로 기록(활성화 시 증분) |
| 저장소를 처음부터 재구축 | 한 번의 실행에만 `indexing.reset: true` | 변경 없음 | 비운 뒤 코퍼스 전체를 인덱싱하고 모든 문서를 다시 기록 |

저장된 데이터에 반영되는 설정을 바꾼 뒤에는 `indexing.reset: true`를 쓰고, 단계
결과도 바뀌어야 하면 `--force-rebuild`를 함께 씁니다.

- 임베딩 모델 또는 `indexing.opensearch.embedding_dimension`
- `processing.translation.source_language` 또는 언어 분석기(텍스트 단위 매핑이
  바뀝니다)
- 증분 실행이 갱신하지 않는 설명과 커뮤니티를 새로 고칠 때
  ([증분 실행](#증분-실행) 참고)
- 레지스트리에 추출 지문이 없던 시기에 쓴 레코드가 있을 때의 추출 설정(프롬프트,
  `entity_types`, 추출·청킹 모델, 청크 크기 등) 변경, 또는 내장 프롬프트 템플릿
  (코드) 수정. 그 밖의 경우 증분 실행은 저장된 지문이 현재 설정과 다른 미변경
  문서를 스스로 다시 추출하고 그 수를 로그로 남깁니다
  ([사용자 가이드 §5](./user-guide.ko.md#작업-흐름) 참고). 지문이 없는 레코드는
  콘텐츠로만 비교하므로 초기화하기 전까지 이전 추출 결과를 유지합니다.

끝나면 `indexing.reset`을 다시 `false`로 돌립니다. 그대로 두면 실행할 때마다
재구축합니다.

`--verify-metadata`(`--pipeline-id` 필요)는 실행의 단계 메타데이터를 검사하고,
메타데이터가 없거나 손상되었으면 0이 아닌 종료 코드를 반환합니다.

## 증분 실행

`aws.dynamodb.enabled: true`(컨테이너 이미지의 기본값)이면 `run-ingestion`은
실행할 때마다 콘텐츠 해시로 코퍼스와 문서 상태 레지스트리를 비교하고, 새 문서와
바뀐 문서만 인덱싱합니다. 비교 범위는 실행의
인덱스 접미사(사용자 가이드 §3)와 소스 범위입니다. 소스 범위는
`processing.document_parsing.source_scope`이며 기본값은 해석된 소스
디렉터리이고, 컨테이너 엔트리포인트는 `s3://` 소스 URI로 설정합니다. 둘 다 문서의
레지스트리 키에 들어가므로, 한 접미사를 쓰는 두 코퍼스는 상대 경로가 같은 파일이
있어도 레코드를 공유하지 않습니다. 끝에 슬래시가 있는 `s3://` URI와 없는 URI
(`s3://bucket/corpus/`, `s3://bucket/corpus`)는 같은 범위입니다. 레지스트리에 실행의
표기는 없고 다른 표기만 있으면 저장된 표기를 쓰고 WARNING 로그를 남깁니다. 소스에서
사라진 파일은 삭제된 것으로 보고, 그 문서에만 속한 산출물을 지웁니다. 따라서 다음을
지킵니다.

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

정식 릴리스 전 빌드가 기록한 레지스트리를 업그레이드할 때는
[CHANGELOG: Upgrading from pre-release builds](../CHANGELOG.md#upgrading-from-pre-release-builds)를
참고하세요.

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
항목도 그대로 두세요. 실행이 문서를 커밋하거나 제거할 때 지웁니다. 커밋과 삭제
사이에 멈춘 실행이 커밋된 레코드 옆에 남긴 초과 항목은 문서가 바뀌지 않았어도
같은 네임스페이스의 다음 증분 실행이 지웁니다. 인덱싱 단계는 레지스트리를 일괄로
쓰므로, 직접 작성한 정책을 쓰는 역할에는 `BatchGetItem`과 함께
`dynamodb:BatchWriteItem` 권한이 필요합니다(CDK 스택의 `grant_read_write_data`는
둘 다 포함합니다).

## 실패한 문서

번역, 그래프 추출, gleaning, 주장 추출 중 하나가 문서의 텍스트 단위 일부에서
실패하거나, `--continue-on-error` 실행에서 문서의 산출물을 만드는 단계(번역부터
커뮤니티 탐지까지)가 비정상 종료하거나, 문서의 산출물(텍스트 단위, 엔티티, 관계, 주장, 커뮤니티, 보고서)을
OpenSearch나 Neptune에 쓰는 데 실패하면 그 문서는 `FAILED`로 기록됩니다. 다음
실행은 이 문서를 바뀐 문서로 취급해 실패한 실행이 쓴 내용을 지우고 다시
처리합니다. `--continue-on-error` 실행에서 파싱, 로딩, 청킹 단계가 비정상
종료하면 인덱싱 단계를 아예 실행하지 않으므로 저장소와 레지스트리는 이전 상태를
유지합니다. `--continue-on-error` 없이 어떤 단계든 실패해도 마찬가지입니다. 실행은
인덱싱 전에 멈추고 아무것도 기록하지 않습니다. 단계 실패만으로는 문서 하나가
문제인지 백엔드(스로틀링, 장애, 만료된 자격 증명)가 문제인지 알 수 없으므로, 다음
실행이 같은 새 문서와 바뀐 문서를 다시 처리합니다.

- 내용이 그대로인 채로 `indexing.max_document_failures`(기본값 `3`)번 연속
  실패하면 더 이상 재시도하지 않습니다. 문서는 `FAILED`로 남아 인덱싱된 내용을
  유지하고, 실행할 때마다 파일 이름을 담은 WARNING이 남습니다.
- 다시 시도하려면 원인(대개 너무 크거나 형식이 잘못된 파일)을 고치거나, 파일을
  수정해 콘텐츠 해시를 바꾸거나, `indexing.max_document_failures`를 올립니다.
- 한 문서의 산출물 ID가 DynamoDB 항목 하나(400 KB, 약 10,000개 ID)를 넘으면
  인덱싱 단계가 델타에 대해 아무것도 쓰기 전에, 파일 이름과 ID 개수를 담은
  오류로 실패합니다. 파일을 나눕니다.

쓰기 실패는 따로 판정합니다. 재시도할 수 있는 상태(429, 502, 503, 504)로
거부된 OpenSearch bulk 항목과, 클라이언트 자체 재시도 후에도 그런 상태를 받은
bulk 요청 전체의 항목은 먼저 백오프를 두고 최대 네 번 다시 보냅니다. 실행 중
Neptune 요청(쓰기, 삭제, 제거와 병합을 위한 읽기)은 모두 새 연결로
`indexing.neptune.max_attempts`회(기본값 `4`)까지 재시도하며, 어느 요청이든
반복해도 안전하므로 응답을 잃은 요청이 두 번 적용되지 않습니다. 그래도
한 산출물 유형의 쓰기 중 실패 비율이 `indexing.max_failure_rate`(기본값 `0.2`)를
넘으면 인덱싱 단계가 실패하고 문서를 기록하지 않으므로, 다음 실행이 다시
시도합니다. 그 이하이면 실행은 성공하고, 실패한 산출물을 가진 문서만 `FAILED`로
기록됩니다. 백엔드가 항목 ID 없이 보고한 실패가 있으면 그 실행의 모든 문서를
`FAILED`로 기록합니다. CDK 스택에서는 단계가 실패하면 실행 ID와 오류를 담은
메시지를 SNS 토픽에 게시하고, 다음 네 경보도 같은 토픽으로 문제를 알립니다.
단계가 실패하면 `PipelineFailures`도 발생하므로 같은 실패가 두 번 오며, 두 번째
알림에는 실행 ID와 오류가 없습니다. 이 경보를 두는 이유는 메시지로 알릴 수 없는
실패도 잡기 때문입니다. 단계의 오류 처리기가 잡지 못하는 오류(예: `pipeline_id`가
없는 시작 입력)나 재시도 후에도 실패한 게시가 여기에 해당합니다.

| 경보 | 발생 조건 |
|---|---|
| `PipelineFailures` | Step Functions 실행이 실패함 |
| `PipelineTimedOutOrAborted` | 실행이 상태 머신 제한 시간을 넘겼거나 중지됨 |
| `IndexingFailures` | 실행이 성공했더라도 인덱싱에서 실패한 항목이 있음 |
| `ExtractionFailures` | 추출, gleaning, 주장 추출이 일부 텍스트 단위에서 실패함 |

경보를 받으려면 `-c alarm_email=<address>`로 배포해 구독자를 등록해야 합니다.

## 캐시와 접두사

| 캐시 | 위치 | 참고 |
|---|---|---|
| 단계 체크포인트(로컬) | `<--cache-directory>/<pipeline_id>/` (기본값 `cache/`) | 재개할 때 재사용하며, `--force-rebuild`는 읽지 않고 다시 계산한 출력을 씁니다 |
| 단계 체크포인트(S3) | `s3://<bucket>/<--s3-prefix>/<pipeline_id>/` (기본 접두사 `pipeline-runs`) | `--s3-sync` 사용 시. Step Functions 단계 사이의 인계에 쓰이며, 동기화가 실패하면 `CacheSyncError`로 실행이 실패합니다 |
| 임베딩 캐시 | `s3://<aws.s3.bucket_name>/embedding-cache/cache.json` (`indexing.opensearch.embedding_cache_s3_key`) | `indexing.opensearch.persist_embedding_cache: true`일 때만 |
| 시각화 데이터 | `graph.visualization.outputs_directory`가 없으면 `<cache directory>/<pipeline_id>/visualization/` | 단계 캐시와 함께 동기화(`.json`만) |

CDK 캐시 버킷은 `pipeline-runs/`와 `embedding-cache/` 아래의 객체만 30일 뒤에
만료합니다. 코퍼스는 다른 접두사(예: `corpus/`)에 둡니다. 만료된 코퍼스 파일은
다음 증분 실행에서 삭제된 문서로 보입니다. 기존 버킷을 재사용하면
(`cache_bucket_name`) 그 버킷의 수명 주기 규칙이 그대로 적용되므로 확인합니다.

`pipeline_id`는 캐시 디렉터리와 S3 접두사 이름이 되므로 영문 소문자, 숫자,
하이픈, 밑줄만 쓸 수 있습니다.

## OpenSearch 복제본

CDK 스택에서 다중 노드 도메인(`opensearch_count` > 1, dev가 아닌 환경의 기본값)은
인덱스마다 복제본을 하나 두므로 노드나 AZ 하나를 잃어도 모든 샤드를 쓸 수
있습니다. 단일 노드인 dev 도메인은 복제본이 없습니다. 컴퓨트 스택이 설정하는
`OPENSEARCH_NUMBER_OF_REPLICAS`가 인덱스를 만들 때
`indexing.opensearch.index_settings.number_of_replicas`를 덮어씁니다. 그 전에
만든 인덱스는 복제본 수가 그대로이므로 VPC 안에서 한 번 설정합니다.

```bash
curl -X PUT "https://<OpenSearch endpoint>/_all/_settings" \
  --aws-sigv4 "aws:amz:<region>:es" --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  -H "x-amz-security-token: $AWS_SESSION_TOKEN" \
  -H 'Content-Type: application/json' -d '{"index": {"number_of_replicas": 1}}'
```

## 백업과 복원

세 저장소는 백업 주기가 다르고 내용이 서로를 참조합니다(레지스트리가 문서별
그래프·인덱스 산출물을 기록함). 따라서 세 저장소를 같은 시점으로 함께 복원하거나
아예 복원하지 않습니다.

| 저장소 | 백업 | 보존 기간 |
|---|---|---|
| Neptune | 자동 백업, 특정 시점 복원 | `backup_retention_days`(기본 7일) |
| OpenSearch | 매시간 자동 스냅샷(`cs-automated-enc` 저장소) | 14일 |
| DynamoDB 문서 상태 테이블 | 특정 시점 복구(PITR) | 35일 |
| S3 캐시 버킷 | 없음(버전 관리 없음, 단계 체크포인트는 다시 만들 수 있음) | — |

S3의 코퍼스가 원본이므로 모든 저장소는 코퍼스에서 다시 만들 수 있습니다.

**Neptune 보존 기간 안이면** 세 저장소를 같은 시점으로 복원합니다.

1. 복원하는 동안 쓰기가 없도록 수집을 멈춥니다(`RUNNING` 실행 없음).
2. 마지막 수집 실행이 끝난 뒤 만들어진 OpenSearch 스냅샷을 고르고
   (`GET _snapshot/cs-automated-enc/_all`), 그 시작 시각을 복원 시점으로 씁니다.
3. Neptune은 `aws neptune restore-db-cluster-to-point-in-time`, 테이블은
   `aws dynamodb restore-table-to-point-in-time`으로 그 시점에 복원합니다. 둘 다
   기본 설정의 *새* 리소스를 만듭니다(Neptune 클러스터에 인스턴스를 추가하고,
   테이블의 PITR과 삭제 방지를 다시 켭니다).
4. 도메인의 현재 인덱스를 삭제한 뒤 스냅샷의 인덱스와 별칭을 복원합니다
   (`POST _snapshot/cs-automated-enc/<snapshot>/_restore`).
5. 데이터 플레인이 복원한 클러스터 엔드포인트와 테이블을 쓰도록 바꿉니다
   (`NEPTUNE_ENDPOINT`, `GRAPHRAG_DOC_STATUS_TABLE`). 태스크 역할, DynamoDB
   게이트웨이 엔드포인트 정책과 경보는 스택의 클러스터와 테이블로 범위가
   정해져 있으므로 복원한 리소스에도 권한을 줍니다.

**Neptune 보존 기간을 지났거나** 5단계를 할 만한 가치가 없으면 공통 시점이 없으므로
`indexing.reset: true`로 전체를 한 번 다시 만듭니다([다시 수집하기](#다시-수집하기)
참고). 이 인덱스 네임스페이스의 저장소와 레지스트리 레코드를 지우고 코퍼스 전체를
다시 인덱싱합니다. 다른 네임스페이스가 테이블을 함께 쓸 때만 테이블을 먼저
복원합니다.

## 시작 시 엔드포인트 검사

CLI는 첫 유료 모델 호출 전에 실행에 필요한 저장소 엔드포인트가 설정되어 있는지
검사하고, 없으면 종료 코드 1로 끝납니다.

```text
Error: Missing endpoint configuration for the indexing stage: aws.neptune.endpoint (env NEPTUNE_ENDPOINT), aws.opensearch.endpoint (env OPENSEARCH_ENDPOINT). Set them in the config file or the environment.
```

- `run-ingestion`은 `indexing` 단계가 켜져 있으면 두 엔드포인트를 모두
  검사합니다(`--verify-metadata`는 제외).
- `run-rag`와 `run-eval`은 선택한 전략의 검색기가 쓰는 엔드포인트를 검사합니다.
  모든 전략에 OpenSearch가 필요하고, `local`, `drift`, `mix`, `hybrid`는
  Neptune도 필요합니다. `auto`는 `search.auto_routable_strategies`의 모든 전략을
  기준으로 검사합니다.

이 검사는 값이 설정되어 있는지만 봅니다. 값이 틀렸거나 연결할 수 없는
엔드포인트는 첫 연결에서 실패하며, 검색기는 인증·설정·연결 오류를 빈 결과로
감추지 않고 그대로 보고합니다.

## 관측성

추적은 기본적으로 꺼져 있습니다(`observability.enabled: false`). 4318 포트로 OTLP를
받는 로컬 Jaeger로 트레이스를 보내려면 다음과 같이 합니다.

```bash
docker run --rm -d --name jaeger -p 16686:16686 -p 4318:4318 \
  jaegertracing/all-in-one:latest
uv sync --extra otel
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

```yaml
observability:
  enabled: true
  backend: "otlp"
```

질의를 실행하고(`run-rag --config-path config.yaml --query "..."`)
<http://localhost:16686>을 열면 `unified-kg-rag` 서비스에서 질의마다
`kg_rag.query` 트레이스, 수집 실행마다 `kg_rag.ingestion.run` 트레이스를 볼 수
있습니다. OpenTelemetry Collector나 상용 엔드포인트를 쓸 때는
`OTEL_EXPORTER_OTLP_ENDPOINT`를 그 주소로 지정하고, 자격 증명은 설정 파일이 아니라
Secrets Manager나 태스크 시크릿에서 `OTEL_EXPORTER_OTLP_HEADERS`로 넘기세요. gRPC
엔드포인트(4317 포트)는 `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`를 설정합니다.

동작 확인:

- 시작 로그에 `Observability enabled (backend: 'otlp', ...)`가 나옵니다.
- 스팬 안에서 남긴 로그 레코드에는 `trace_id`와 `span_id`가 있습니다. 이 ID로
  트레이스 백엔드를 검색하세요.
- 질의 트레이스 아래에 `kg_rag.retrieve`, `kg_rag.fusion`, `kg_rag.context_build`,
  `chat <model>` 스팬이 있고, 모델 스팬의 `gen_ai.usage.input_tokens`가 해당 실행의
  토큰 사용량입니다.
- `kg_rag.rerank` 스팬의 `kg_rag.rerank.status`는 Rerank 호출이 실패해 질의가 융합
  순서로 대체되었으면 `error`, 모델이 쓸 수 있는 결과를 돌려주지 않았으면
  `empty`입니다.

### Langfuse

로컬 Langfuse(v3)는 저장소의 Docker Compose 파일로 실행합니다.

```bash
git clone --depth 1 https://github.com/langfuse/langfuse.git
cd langfuse && docker compose up -d     # UI: http://localhost:3000
```

UI에서 프로젝트와 API 키 쌍을 만든 뒤 다음을 설정합니다.

```bash
uv sync --extra otel
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-...   # 배포 환경: Secrets Manager에서
export LANGFUSE_SECRET_KEY=sk-lf-...   # 태스크 시크릿으로 주입, 파일에 두지 않음
```

```yaml
observability:
  enabled: true
  backend: "langfuse"
```

질의마다 `kg_rag.query` 트레이스가 생기고, Sessions에서는 대화별로, Users에서는
`RAGInput.user_id`별로 묶입니다. 모델 호출은 토큰 사용량과 함께 generation으로
표시됩니다. Langfuse Cloud는 `LANGFUSE_HOST`를 리전 URL(예:
`https://cloud.langfuse.com`)로 지정합니다. Langfuse는 HTTP OTLP만 받습니다. 두 키
대신 `OTEL_EXPORTER_OTLP_HEADERS`를 직접 지정할 때는
`x-langfuse-ingestion-version=4`를 함께 넣어야 트레이스가 몇 분씩 늦게 보이지
않습니다.

### Arize Phoenix

```bash
docker run --rm -d --name phoenix -p 6006:6006 -p 4317:4317 \
  arizephoenix/phoenix:latest            # UI: http://localhost:6006
uv sync --extra otel
```

```yaml
observability:
  enabled: true
  backend: "phoenix"
```

다른 설정이 없으면 스팬은 `http://localhost:6006/v1/traces`로 가고
`service_name` 이름의 프로젝트에 쌓입니다(`PHOENIX_PROJECT_NAME`이 우선). 원격
또는 인증이 필요한 Phoenix는 `PHOENIX_COLLECTOR_ENDPOINT`(서버 루트, 예:
`https://phoenix.example.com`)와 `PHOENIX_API_KEY`를 설정합니다. 스팬에
OpenInference 종류(`LLM`, `EMBEDDING`, `RETRIEVER`, `RERANKER`, `CHAIN`)가
붙으므로 모델 호출의 토큰 수가 보이고 Sessions 화면에서 대화의 질의가 묶입니다.

### ADOT 수집기 사이드카를 거친 CloudWatch

CloudWatch는 `https://xray.<region>.amazonaws.com/v1/traces`에서 SigV4로 서명한
OTLP 트레이스를 받으므로 키가 필요 없습니다. 앱의 OTLP 익스포터는 요청에 서명하지
않으므로 옆에 둔 수집기가 서명해서 전달합니다. CDK 스택에서는
`-c enable_cloudwatch_tracing=true`를 주면 태스크에 AWS Distro for
OpenTelemetry(ADOT) 수집기 사이드카가 추가되고, 앱이 그 수집기를 가리킵니다
(`backend: otlp`, `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318`). 서명은
수집기가 태스크 역할로 합니다. IAM과 프라이빗 모드의 `xray` 엔드포인트를 포함해
이 옵션이 바꾸는 내용은
[iac/README.md](../iac/README.md#cloudwatch-tracing-optional)에 있습니다.

1. **Transaction Search 켜기**(계정과 리전마다 한 번). 켜기 전에는 트레이스
   엔드포인트가 스팬을 거부합니다. 계정 전체 설정이라 이 스택만이 아니라 그
   리전의 모든 X-Ray 스팬 수집을 CloudWatch Logs 수집(`aws/spans` 로그 그룹,
   CloudWatch Logs 수집 요금)으로 옮기고, 스팬의 1%를 무료로 트레이스 요약으로
   색인합니다. 다른 워크로드가 현재 설정에 기대고 있을 수 있으니 먼저 확인하세요.

   ```bash
   aws xray get-trace-segment-destination --region "$REGION"
   ```

   `"Destination": "CloudWatchLogs", "Status": "ACTIVE"`이면 이미 켜져 있습니다.
   아니면 CloudWatch 콘솔(**Application Signals, Transaction Search, Enable
   Transaction Search**)에서 켜세요. 콘솔은 X-Ray에 필요한 로그 그룹 리소스 정책도
   함께 설정합니다. API로 켜려면
   [Transaction Search 안내서](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Enable-TransactionSearch.html)의
   단계(`xray.amazonaws.com`용 CloudWatch Logs 리소스 정책을 만든 뒤
   `aws xray update-trace-segment-destination --destination CloudWatchLogs`)를
   따르세요. 켠 뒤 스팬을 검색할 수 있을 때까지 10분 정도 걸릴 수 있습니다.
2. **`otel` extra를 넣어 이미지 빌드**:
   `docker build --build-arg UV_EXTRAS="--extra otel" -f docker/Dockerfile ...`.
   빠지면 시작할 때
   `observability.enabled is true but OpenTelemetry is not installed`로 실패합니다.
3. **배포**: `cdk deploy --all -c enable_cloudwatch_tracing=true`. 이후 배포에도
   이 플래그를 주거나 `cdk.json`에 넣으세요. 빠뜨리면 다음 배포에서 사이드카가
   제거됩니다.
4. **수집기 이미지를 한 번 pull**: 첫 실행 전에 작업용 컴퓨터에서 받아 두세요
   ([CloudWatch tracing](../iac/README.md#cloudwatch-tracing-optional)의 4단계).
   이미지는 ECR pull-through 캐시를 거쳐 오는데, ECR에 인터페이스 엔드포인트로
   접근하면(`private` 모드) 캐시를 거치는 첫 pull에 인터넷 경로가 필요할 수
   있습니다. ECS는 이미지 중 하나라도 받지 못하면 태스크를 실패시킵니다.

**확인할 곳.** 수집이나 질의를 실행한 뒤 CloudWatch 콘솔의 **Application Signals,
Transaction Search**에서 서비스 `unified-kg-rag`로 거르세요. 수집 실행마다
`kg_rag.ingestion.run` 트레이스 하나, 질의마다 `kg_rag.query` 트레이스 하나가
생기며 자식 스팬은 [관측성](#관측성)에 나온 그대로입니다. 스팬은 `aws/spans` 로그
그룹에도 있어 Logs Insights로 볼 수 있습니다(필드가 비어 나오면 먼저
`fields @message | limit 5`로 정확한 필드 경로를 확인하세요).

```text
fields @timestamp, name, durationNano, attributes.gen_ai.usage.input_tokens, traceId
| filter resource.attributes.service.name = "unified-kg-rag"
| sort @timestamp desc
| limit 50
```

트레이스가 오지 않으면 태스크 로그 그룹(`/<prefix>/tasks`)의 `otel-collector/`
로그 스트림을 읽으세요. `403`이면 X-Ray 권한이 없거나 SigV4 서비스 이름이 틀린
것이고, 대상(destination)을 언급하며 거부되면 Transaction Search가 꺼진 것이며,
`private` 모드에서 연결 시간 초과가 나면 `xray` 인터페이스 엔드포인트가 없는
것입니다(재사용하는 VPC는 직접 준비해야 합니다). 수집기 컨테이너는 필수
컨테이너가 아니므로 수집기가 죽거나 내보내지 못해도 실행은 실패하지 않습니다.
다만 ECS가 수집기 이미지를 받지 못하면 실행이 실패하므로(중지된 태스크 사유의
`CannotPullContainerError`) 4단계처럼 미리 한 번 받아 두세요.

**비용.** Transaction Search는 스팬을 `aws/spans`의 CloudWatch Logs 수집과 저장으로
과금하고, 무료 1%를 넘는 색인 스팬에도 요금을 매깁니다. 수집 실행은 스테이지,
배치, 모델 호출마다 스팬을 만들므로 코퍼스가 크면 스팬도 많습니다. 이미지 설정의
`observability.sample_rate`를 낮춰 루트 트레이스를 표본 추출하고, 트레이스 요약을
검색하지 않는다면 색인 비율은 1%로 두세요. `xray` 인터페이스 엔드포인트(프라이빗
모드)도 다른 엔드포인트처럼 AZ-시간 단위로 과금됩니다.

**CDK 스택 없이** 쓰려면 같은 수집기 설정을 직접 실행하세요. CloudWatch OTLP
안내서를 따라 스택이 만드는 설정은 다음과 같습니다.

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:4318
exporters:
  otlphttp/traces:
    compression: gzip
    traces_endpoint: https://xray.us-east-1.amazonaws.com/v1/traces
    auth:
      authenticator: sigv4auth/traces
extensions:
  sigv4auth/traces:
    region: us-east-1
    service: xray
service:
  extensions: [sigv4auth/traces]
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlphttp/traces]
```

수집기 역할에는 X-Ray 쓰기 권한(`xray:PutTraceSegments`, `xray:PutSpans`,
`xray:PutSpansForIndexing`)이 필요합니다. CloudWatch 안내서는 관리형 정책
`AWSXrayWriteOnlyAccess`를 안내합니다.

**계정 수준 보조 수단: Bedrock 모델 호출 로깅.** 이 패키지의 추적과 관계없이 리전의
모든 `InvokeModel` / `Converse` 요청과 응답(임베딩 포함)을 CloudWatch Logs나 S3에
기록합니다. 감사 기록은 이 로깅으로, 질의별 분석은 트레이스로 하세요. Rerank
호출(`bedrock-agent-runtime`)은 기록하지 않고, 프롬프트와 응답 전문을 담으므로
저장 위치는 수집한 본문처럼 보호하세요(읽기 권한 제한, CMK, 보존 기간). 계정과
리전마다 하나뿐인 설정이라 그곳의 모든 워크로드가 함께 쓰므로 CDK 스택은 관리하지
않습니다. 스택에서 켜고 끄면 계정 안의 다른 모든 로깅이 바뀝니다. Bedrock 런타임
리전(`bedrock_region`)에서 설정하세요.

```bash
# 먼저 확인: 아래 put은 기존 설정을 대체합니다.
aws bedrock get-model-invocation-logging-configuration --region "$BEDROCK_REGION"

aws bedrock put-model-invocation-logging-configuration --region "$BEDROCK_REGION" \
  --logging-config '{
    "cloudWatchConfig": {
      "logGroupName": "/aws/bedrock/model-invocation-logs",
      "roleArn": "arn:aws:iam::<account>:role/<bedrock-logging-role>"
    },
    "textDataDeliveryEnabled": true,
    "embeddingDataDeliveryEnabled": false
  }'
```

로그 그룹이 있어야 하고, 역할은 `bedrock.amazonaws.com`을 신뢰하며 그 로그 그룹에
`logs:CreateLogStream`과 `logs:PutLogEvents`를 허용해야 합니다. 콘솔(**Amazon
Bedrock, Settings, Model invocation logging**)에서 둘 다 만들 수 있습니다.
[Monitor model invocation using CloudWatch Logs and Amazon S3](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html)를
참고하세요. 끄려면 `aws bedrock delete-model-invocation-logging-configuration`을
실행합니다. CloudWatch 콘솔의 **GenAI Observability, Model Invocations**는 이 로그를
Bedrock 자체의 호출, 토큰, 지연 시간 지표와 함께 보여 줍니다.

### 전송 제한, 프라이빗 VPC, 개인정보

스팬은 백그라운드 스레드에서 내보내므로 수집기에 연결할 수 없어도 실행이 느려지지
않습니다. 전송 요청 하나는 `observability.export_timeout_seconds`(10초,
`OTEL_EXPORTER_OTLP_TIMEOUT`이 있으면 우선)로 제한되고, 큐가 가득 차면 막히지
않고 스팬을 버립니다. 익스포터는 실패를 `ERROR` 로그로 남깁니다(재시도마다 나오는
경고는 `logging.library_levels`가 숨깁니다). 종료 시에는 큐에 남은 스팬을 최대
512개씩 묶어 배치마다 요청 한 번으로 보냅니다. 따라서 수집기에 연결할 수 없으면
남은 배치마다 제한 시간 한 번 정도가 걸리고, 배치 프로세서의 종료 대기 시간인
약 30초가 지나면 나머지를 버리고 프로세스가 끝납니다. 프라이빗 VPC에서는 태스크가 엔드포인트에 연결할 수 있어야 합니다.
`localhost`의 수집기 사이드카, VPC 안에서 운영하는 Langfuse나 Phoenix, 또는
수집기의 CloudWatch 전송을 위한 X-Ray 인터페이스 VPC 엔드포인트를 쓰세요. 호스팅형
Langfuse나 Phoenix는 인터넷 송신이 필요한데, 프라이빗 모드 스택에는 없습니다. 데이터
플레인 이미지에는 `--build-arg UV_EXTRAS="--extra otel"`로 빌드했을 때만 extra가
들어갑니다.

자격 증명(`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`, `PHOENIX_API_KEY`,
`OTEL_EXPORTER_OTLP_HEADERS`)은 환경 변수에서만 읽습니다. Secrets Manager 값을 ECS
태스크 시크릿으로 주입하세요. 시작 로그 `Observability exporter for backend '...'`에는
자격 증명을 뺀 엔드포인트와 헤더 이름만 나오고 값은 나오지 않습니다.

개인정보 기본값: 스팬에는 개수, ID, 모델 이름, 토큰 사용량만 들어갑니다.
`capture_content: true`로 설정하면 모델 스팬에 프롬프트와 응답 본문이
(`redact_patterns` 적용 후) 추가되므로, 트레이스 백엔드가 코퍼스 데이터를 보관해도
되는 환경에서만 켜세요. `otlp`, `langfuse`, `phoenix`에서는 내보낼 때도 모든 문자열
속성, 이벤트 속성(예외 메시지에 질의가 들어갈 수 있음), 상태 메시지에
`redact_patterns`를 적용하며 `session.id`와 `user.id`도 대상입니다. `none`에서는
호스트 파이프라인이 기록된 그대로 내보내므로 그쪽 수집기에서 가리세요.
`RAGInput.user_id`는 받은 그대로 내보내므로 불투명한 ID를 넘기세요. LangSmith
추적(`LANGSMITH_TRACING=true`)도 켜져 있으면 LangChain이 이 설정과 관계없이 본문을
LangSmith로 업로드합니다.

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
| `OpenSearch rejected the search on <index>: ...` | 그 인덱스에 맞지 않는 요청(HTTP 400). 필드와 형식이 다른 필터 값(`query_shard_exception`)이거나, 클러스터에 없는 대상을 가리키는 설정(예: `indexing.opensearch.hybrid_search_pipeline_name`, `Pipeline ... is not defined`). `too_many_nested_clauses` / `too_many_clauses`(HTTP 500)는 질의의 절이 클러스터 허용치보다 많은 경우로, 보통 `indices.query.bool.max_clause_count`가 기본값 1,024보다 낮음 | 필터 값이나 설정을 고침. 하이브리드 검색 파이프라인은 수집 단계가 만듦. 절 한도는 `indices.query.bool.max_clause_count`를 1,024 이상으로 둠 |
| `StoreConnectionError: ... Cannot connect to host ...` / `Neptune is not reachable ...` / `... status=429 ... (still failing after the client's retries)` | OpenSearch 또는 Neptune이 연결을 거부했거나 주소를 확인할 수 없음, Neptune이 연결을 닫음(재시작, 장애 조치), 또는 OpenSearch가 `search.store_max_attempts`번 시도 후에도 429/502/503/504로 응답함(과부하, 재시작 중). 컨텍스트 없이 답하지 않고 질의가 실패함. 끊긴 Neptune 연결은 버리고 다음 질의나 수집의 다음 재시도가 다시 연결함 | 엔드포인트와 포트, 저장소 실행 여부, 네트워크 경로(보안 그룹, VPC 엔드포인트) 확인 |
| `OpenSearch did not answer a request on <index> within N s in M attempt(s)` / `Neptune did not answer within N s` | 저장소가 연결은 받았지만 `search.store_timeout_seconds` 안에 응답하지 않음(OpenSearch는 `search.store_max_attempts`번 시도). 질의가 실패함 | 저장소 상태와 부하(CPU, 대기열, 진행 중인 장애 조치) 확인. 원래 느린 클러스터라면 `search.store_timeout_seconds`를 늘림 |
| `OpenSearch denied the search on <index>: ... status=403 ...` (또는 `401`) | 질의 주체가 인덱스를 검색할 권한이 없음. 세분화된 액세스 제어에서 `indices:data/read/search`가 없는 역할이거나 자격 증명이 잘못됨 | 역할에 해당 인덱스의 `indices:data/read/search`(임베딩 확인용 `indices:admin/mappings/get` 포함)를 부여하거나 자격 증명을 고침 |
| 질의 중 `ClientError: ... (ValidationException) ... InvokeModel ...` / `(ResourceNotFoundException)` / `NoCredentialsError` | 질의 임베딩 호출이 성공할 수 없음. 잘못된 모델 ID, 그 리전에서 온디맨드 처리량을 지원하지 않는 모델, 부여되지 않은 모델 액세스, 자격 증명 없음. 컨텍스트 없이 답하지 않고 질의가 실패함 | `indexing.opensearch.embedding_model_id`, `aws.bedrock.region_name`, 계정의 모델 액세스 또는 자격 증명 확인 |
| `EmbeddingModelError: Query embedding failed: ...` | 클라이언트 재시도 후에도 질의 임베딩 모델 호출이 실패함(멈춘 Ollama 러너, 죽은 vLLM 엔진, 계속되는 Bedrock `InternalServerException`). 질의의 모든 벡터 검색에 필요하므로 "찾을 수 없음"으로 답하지 않고 질의가 실패함 | 임베딩 서버나 모델(`providers.embedding`, `indexing.opensearch.embedding_model_id`)과 그 로그를 확인한 뒤 다시 시도 |
| `Neptune refused the connection (HTTP 403)` | IAM 인증이 웹소켓 연결을 거부함. 정책에 클러스터의 `neptune-db:connect`가 없거나 다른 리전으로 서명함 | `neptune-db:connect`(실행에 필요한 `neptune-db:*` 데이터 작업 포함)를 부여하고 `aws.region_name`을 클러스터의 리전으로 설정 |
| `Skipping N '.md' file(s) ...` / `No supported source files found in '<dir>'` | `unstructured` 추가 패키지가 설치되지 않음 | 설치(Python 3.11 이상)하거나 파일 형식을 변환 |
| `DocStatusRegistryError` | 문서 상태 레지스트리 테이블이 없거나, 접근할 수 없거나, 스로틀링됨 | 테이블을 만들거나 `aws.dynamodb.table_name`을 고치고, 메시지에 나온 DynamoDB 권한을 부여한 뒤 같은 `--pipeline-id`로 재개 |
| `Incremental indexing is enabled but no document delta was computed` | 레지스트리 비교 없이 인덱싱 단계가 실행됨(`continue_on_error`로 로딩 단계 실패 뒤 진행) | 레지스트리에 접근할 수 있게 한 뒤 `document_loading`부터 다시 실행 |
| `CacheSyncError` | S3 단계 캐시 다운로드나 업로드가 실패함 | 버킷 권한과 `aws.s3.encryption`을 확인한 뒤 같은 `--pipeline-id`로 재개 |
| `Pipeline failed at stage(s): ...` | 단계가 실패해 CLI가 1로 종료함 | 해당 단계 로그를 확인해 고친 뒤 같은 `--pipeline-id`로 재개. `--verify-metadata`로 실행 메타데이터를 확인하거나 `--force-rebuild`로 처음부터 시작 |
| `--source-directory is required for pipeline execution` | `--source-directory`도 `$GRAPHRAG_SOURCE_DIRECTORY`도 없음 | 플래그를 넘김. `--verify-metadata`에는 대신 `--pipeline-id`가 필요 |
| `--s3-bucket-name must be specified for S3 sync` | 버킷 없이 `--s3-sync`를 지정함 | `--s3-bucket-name`을 넘김 |
| `--pipeline-id is required for --resume-from-stage` | 재개하려면 이전 실행이 필요함 | 그 실행의 `--pipeline-id`를 넘김 |
| `Invalid stage names provided: ...` | `--enabled-stages`에 없는 단계 이름이 있음 | CLI가 출력하는 목록의 이름을 사용 |
| `argument --resume-from-stage: Unknown pipeline stage ...`(종료 코드 2) | `--resume-from-stage`에 없는 단계 이름을 넘김 | 메시지에 나온 목록의 이름을 사용(대소문자 무관) |
| `... needs confirmation, but stdin is not a terminal` | 다시 실행할 때 완료된 단계의 출력이 현재 입력 기준으로 캐시에 없는데, 물어볼 터미널이 없음 | `--yes`를 넘겨 계속(재개하면서 그 단계를 다시 계산)하거나 새 `--pipeline-id`를 사용 |
| `Stage '<stage>', upstream of this run's enabled stages, has not completed` | `--enabled-stages` 범위(Step Functions 단계 묶음)를 재개했는데 앞 단계가 실패했거나 캐시된 출력이 없음 | 앞 단계를 먼저 실행. CDK 스택에서는 같은 `pipeline_id`로 새 실행을 시작 |
| `The doc-status registry lists indexed documents of this namespace whose entities the graph store does not hold` | 레지스트리에 기록된 내용을 그래프가 잃음(Gremlin Server 재시작, `docker compose ... down -v`, 복원하거나 교체한 그래프) | `indexing.reset: true`로 다시 실행하거나 해당 네임스페이스의 레지스트리 레코드를 삭제(로컬 모드에서는 SQLite 파일) |
| 내장 그래프·벡터 저장소의 `... stayed locked ...`(`database is locked`)(로컬 모드) | 다른 프로세스(두 번째 수집 등)가 같은 SQLite 파일에 쓰면서 `search.store_timeout_seconds`보다 오래 잠금을 잡음. SQLite는 한 번에 한 프로세스만 씀 | 다른 실행이 끝난 뒤 다시 실행하거나, 저장소 파일마다 수집을 한 번에 하나만 실행하거나, `search.store_timeout_seconds`를 늘림. 파일은 로컬 디스크에 둠(네트워크 파일 시스템은 지원하지 않음) |
| 로그에 `Connection refused`와 함께 `... processed N inputs but produced 0 outputs`(로컬 모드) | `providers.llm.base_url` / `providers.embedding.base_url`에서 Ollama나 OpenAI 호환 서버에 연결할 수 없음 | 서버를 띄우고(`ollama serve` 또는 `docker compose -f docker/compose.local.yaml --profile models up -d --wait`), `base_url`(또는 `--model-base-url`)을 확인하고, 모델을 내려받음 |
| 질의가 `AuthenticationError` / `NotFoundError`(`Error code: 401` / `404`), `APIConnectionError` / `APITimeoutError`, `ResponseError: model ... not found`로 실패함(로컬 모드) | OpenAI 호환 서버나 Ollama가 키를 거부했거나 그 모델을 제공하지 않음, 또는 클라이언트 재시도 후에도 연결할 수 없거나 응답하지 않음. 이후 질의도 모두 같은 이유로 실패하므로 "찾을 수 없음"으로 답하지 않고 질의를 실패시킴 | `providers.llm` / `providers.embedding`(`base_url`, `api_key`, `model_id`), 서버 실행 여부를 확인하고 모델을 내려받음 |
| `observability.enabled is true but OpenTelemetry is not installed` | 추적을 켰지만 `otel` extra가 설치되지 않음 | `uv sync --extra otel`을 실행하거나 `observability.enabled: false` 설정 |
| 프라이빗 VPC에서 Bedrock 호출이 멈춤 | VPC 엔드포인트가 처리하지 않는 리전으로 Bedrock을 호출함 | `aws.bedrock.region_name`을 비우거나 VPC 리전과 같게 설정 |
| `use_iam: false`에서 OpenSearch 인증 실패 | 기본 인증 자격 증명이 없음 | `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` 설정(예: `.env`) |
| LightRAG `mix` / `hybrid` 결과가 없음 | 관계 인덱스가 없거나 키워드 추출이 키워드를 만들지 못함 | 수집에서 관계 인덱스를 만들었는지 확인. 아주 짧은 질의는 `raw_query_fallback_max_len` 이하일 때만 원래 질의를 씀 |
