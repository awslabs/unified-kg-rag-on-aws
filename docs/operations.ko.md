# 운영 런북

> 🇬🇧 English version: [docs/operations.md](./operations.md)

용어는 [용어집](./glossary.ko.md)을 따릅니다.

이미 배포된 환경에서 운영자가 하는 작업과, 실행이나 질의가 실패했을 때 할 일을
정리합니다. 모든 설정 키는 [설정 레퍼런스](./configuration.md)(영문, 설정을 읽는
방식은 [사용자 가이드 §2](./user-guide.ko.md#2-설정))에, 배포 스택과 명령은
[`iac/README.md`](../iac/README.md#after-deploy)에 있습니다.

| 하려는 일 | 볼 곳 |
|---|---|
| 첫 실행 전에 배포를 점검하기 | [실행 전 확인](#실행-전-확인) |
| 수집을 재개하거나, 다시 계산하거나, 다시 구축하기 | [다시 수집하기](#다시-수집하기) |
| 코퍼스를 최신 상태로 유지하기 | [증분 실행](#증분-실행) |
| 멈춘 실행을 복구하기 | [중단된 실행](#중단된-실행) |
| 문서가 `FAILED`나 `QUARANTINED`인 이유 찾기 | [실패한 문서](#실패한-문서), [격리된 문서](#격리된-문서) |
| 수집을 둘 실행하거나 `NamespaceLeaseError` 이해하기 | [동시 실행](#동시-실행) |
| CloudWatch 경보에 대응하기 | [경보](#경보) |
| 저장소 복원하기 | [백업과 복원](#백업과-복원) |
| 서비스 계층에 질의 권한을 주거나 테넌트를 격리하기 | [질의 권한과 테넌트 격리](#질의-권한과-테넌트-격리) |
| 추적 켜기(Jaeger, Langfuse, Phoenix, CloudWatch) | [관측성](#관측성) |
| 오류 메시지 찾아보기 | [자주 보는 오류](#자주-보는-오류) |

## 실행 전 확인

1. **리전과 자격 증명.** 설정 파일의 `aws.region_name`이 우선합니다. `AWS_REGION`은
   파일이 지정하지 않았을 때만 값을 채우고, `UKR__AWS__REGION_NAME`은 파일보다
   우선합니다(`aws.bedrock.region_name`이 비어 있으면 Bedrock도 이 리전을 따릅니다).
   `env | grep -E '^(AWS_REGION|UKR__AWS__)'`와 `aws sts get-caller-identity`로
   확인합니다.
2. **모델.** [필수 모델](./user-guide.ko.md#필수-모델)이 Bedrock 리전에서 활성화되어
   있어야 합니다.
3. **엔드포인트**가 설정되어 있어야 합니다([시작 시 엔드포인트 검사](#시작-시-엔드포인트-검사)).
4. **같은 저장소에서 다른 수집이 실행 중이지 않아야 합니다**([동시 실행](#동시-실행)).
   CDK 스택에서는
   `aws stepfunctions list-executions --state-machine-arn <arn> --status-filter RUNNING`으로
   확인합니다.

## 일상 운영

### 다시 수집하기

| 목적 | 방법 | 단계 캐시 | 저장소와 레지스트리 |
|---|---|---|---|
| 중단된 실행 마무리 | 같은 `--pipeline-id`로 다시 실행(CDK: 같은 `pipeline_id`로 새 실행). 처음으로 실패했거나 끝나지 않은 단계, 또는 `--resume-from-stage <stage>`부터 재개 | 재사용 | 실행되는 단계가 기록 |
| 모든 단계 다시 계산 | `--force-rebuild` | 무시하고 새로 기록 | 평소대로 기록(활성화 시 증분) |
| 저장소 재구축 | 한 번의 실행에만 `indexing.reset: true` | 변경 없음 | 서비스를 멈추지 않고 재구축하며, 마지막에 네임스페이스의 나머지 레코드를 지움([설계 §5.2](./design.ko.md#52-전체-재구축-indexingreset)) |

저장된 데이터에 반영되는 설정, 곧 임베딩 모델이나 `models.embedding.dimension`,
`processing.translation.source_language`나 언어 분석기, 내장 프롬프트 템플릿(코드)을
바꾼 뒤에는 `indexing.reset: true`를 쓰고, 단계 결과도 바뀌어야 하면
`--force-rebuild`를 함께 씁니다. reset은 증분 실행이 덧붙인 커뮤니티를 새로 고치고
엔티티에 현재 이름의 ID를 줍니다. 그 밖의 추출 설정은 초기화할 필요가 없습니다. 다음
증분 실행이 해당 문서를 다시 추출하고 그 수를 로그로 남깁니다
([사용자 가이드 §5](./user-guide.ko.md#작업-흐름)).

reset 동안 벡터 질의는 새 인덱스로 하나씩 바뀔 때까지 이전 인덱스를 읽고, 그래프 질의는
마지막 정리 전까지 이전 항목과 다시 만든 항목을 섞어 봅니다. 격리된 문서는 건너뛰고 레코드를
그대로 둡니다. 실패한 reset은 아무것도 지우지 않으므로
다시 실행하면 됩니다. 그래프 쓰기가 일부 실패한 reset은(`indexing.max_failure_rate`
이내라도) 역시 실패하며(인덱싱 결과의 `neptune_sweep`), 이전 그래프에 남은 것은 다음
실행이 지웁니다. 인덱스를 별칭에 연결하기 전에 정지되거나 종료된 reset은 그 인덱스
(`<alias>-<timestamp>`, 별칭 없음)를 남기며, 다음 reset이 지울 때까지 남습니다. 증분
실행은 지우지 않으니 직접 지우거나 reset을 다시 실행하세요.
reset 파이프라인 ID를 다시 실행하면 다른 실행처럼 캐시된 단계부터 재개합니다. 그 사이
다른 파이프라인 ID가 이 파이프라인의 텍스트 단위가 가리키는 추출 레코드를 지웠으면(다른
설정의 증분 실행은 다시 추출하는 단위의 레코드를 지움) 그 레코드를 저장하는 단계부터
재개해(WARNING `Extraction records this pipeline stored are gone ...`) 해당 단위를 다시
추출합니다. LLM 응답 캐시가 답하며, `cache.llm_responses.enabled: false`이면 재실행
비용은 처음 reset과 같습니다.
**끝나면 `indexing.reset`을 `false`로 되돌리세요.** 그러지 않으면
실행할 때마다 재구축합니다.

### 증분 실행

`providers.doc_status.enabled: true`(컨테이너 이미지가 설정)이면 실행마다 새 문서와
바뀐 문서만 인덱싱하고, 없어진 파일만 언급하던 내용은 지웁니다. 비교 범위는 실행의
[인덱스 접미사](./user-guide.ko.md#인덱스-접미사)와 소스 범위
(`processing.document_parsing.source_scope`: 해석한 소스 디렉터리, 컨테이너에서는
`s3://` URI. 끝 슬래시가 있든 없든 같은 범위이며, 쓰인 표기를 WARNING으로 알림)입니다.
따라서 다음을 지킵니다.

- **코퍼스는 한 소스에 둡니다**(또는 고정한 `source_scope`). 기본 범위로 코퍼스를
  옮기면 이전 범위의 내용이 인덱스에 남습니다. 옮기기 전에 `source_scope`를 정하거나
  새 위치에서 `--retire-source-scope <old source directory>`로 한 번 실행하세요
  ([삭제 범위](./user-guide.ko.md#삭제-범위)). 새 문서가 있는 실행은 같은 접미사의 다른
  로컬 소스 디렉터리를 WARNING으로 알립니다(거부되거나 실패한 실행만 기록한 문서도 새
  문서로 셉니다).
- **코퍼스 파일이 만료되거나 사라지지 않게 합니다.** 다음 실행이 그 내용을 지웁니다.
- **테넌트나 코퍼스 버전마다 인덱스 접미사와 설정 파일을 따로 둡니다.**

**주기적으로 재구축합니다.** 델타 실행은 커뮤니티를 다시 묶지 않고 덧붙이며, 저장된
커뮤니티와 리포트는 엔티티가 현재 이름의 ID를 받은 뒤에도 이전 ID를 가리킵니다.
`indexing.reset: true`가 둘 다 새로 고칩니다. 인덱싱 단계의 `community_drift` 지표를
지켜보세요: `delta_communities`, `clustered_entities`, `overlapping_entities`,
`stale_communities`, `added_roots`(전역 검색의 모든 단면에 리포트를 하나씩 더함),
`overlap_share`. `indexing.community_drift_threshold`(기본값 `0.5`)에 이르면 WARNING을
남기고 경보가 지켜볼 수 있는 `community_rebuild_recommended: 1`을 보고합니다(수치는
상한으로 읽으세요). 텍스트 단위를 지우는 실행은 지운 단위를 아직 인용하는 저장된
리포트 수(`outdated_reports`)도 보고합니다. 이런 리포트는 지운 내용을 담고 있을 수
있으므로 하나라도 있으면 역시 재구축을 권합니다. `unclustered_entities`는 실행이 묶은
작은 델타 연결 요소와 이어졌지만 커뮤니티가 없는 저장된 엔티티 수입니다. 전체 빌드는 이들을
묶지만 델타 실행은 델타만 묶습니다.

**레지스트리를 읽을 수 없으면**(테이블 없음, 접근 거부, 계속되는 스로틀링)
`document_loading`이 테이블과 원인을 밝힌 `DocStatusRegistryError`로 실패하며, 전체를
다시 인덱싱하지 않습니다. 레지스트리를 고친 뒤 같은 `--pipeline-id`로 다시 실행하거나
`providers.doc_status.enabled: false`로 설정하세요.

사전 출시 빌드에서 올라온다면
[CHANGELOG: Upgrading from pre-release builds](../CHANGELOG.md#upgrading-from-pre-release-builds)를
참고하세요.

### 캐시와 접두사

| 캐시 | 위치 | 참고 |
|---|---|---|
| 단계 체크포인트(로컬) | `<--cache-directory>/<pipeline_id>/`(기본값 `cache/`) | 재개할 때 재사용 |
| 단계 체크포인트(S3) | `s3://<bucket>/<--s3-prefix>/<pipeline_id>/`(기본값 `pipeline-runs`) | `--s3-sync`일 때 단계마다, 그리고 SIGTERM을 받으면 올리므로 새 태스크가 이어서 재개함. 다운로드나 마지막 업로드가 실패하면 `CacheSyncError`로 실행이 실패함. 단계마다 또는 멈출 때 올리는 업로드가 실패하면 경고만 남기고(다음 업로드가 다시 시도), 쓰기 임대를 잃은 뒤 멈추면 올리지 않음. `cache.ttl_seconds`가 아니라 버킷 수명 주기로 만료 |
| LLM 응답 캐시 | `<stage checkpoint dir>/llm-cache/` | 같은 파이프라인 ID로 다시 실행할 때 재사용. `cache.llm_responses.enabled: false`로 끄고, `--force-rebuild`는 무시함 |
| 추출 저장소 | `graphrag-extraction-records-<suffix>`(`indexing.opensearch.extraction_records_index_prefix`) | 저장된 텍스트 단위가 가리키는 동안 이후 실행이 재사용하는 청크별 추출 결과와, 문서 제거(`pending-strip`)나 커밋(`pending-commit`)이 끝날 때까지 남겨 두는 기록(재개한 실행이 필요로 함). `clear`와 `--force-rebuild`는 유지하며, 통째로 버리려면 성공한 실행 뒤에만 인덱스를 삭제하거나 이어서 `indexing.reset: true`로 실행 |
| 임베딩 캐시 | `s3://<aws.s3.bucket_name>/embedding-cache/cache.json` | `indexing.opensearch.persist_embedding_cache: true`일 때만 |
| 시각화 데이터 | `graph.visualization.outputs_directory`가 없으면 `<cache directory>/<pipeline_id>/visualization/` | 단계 캐시와 함께 동기화(`.json`만) |

CDK 버킷은 `pipeline-runs/`와 `embedding-cache/` 아래만 30일 뒤 만료합니다. 만료된
파일은 삭제된 것으로 보이므로 코퍼스는 다른 접두사(`corpus/`)에 두세요. 재사용한
버킷은 자체 수명 주기 규칙을 확인하세요.

### OpenSearch 복제본

CDK 스택에서 다중 노드 도메인(`opensearch_count` > 1)은 인덱스마다 복제본 하나를, 단일
노드 도메인은 복제본 없이 만듭니다. 인덱스를 만들 때 적용하는
`UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS`로 정하므로, 그 전에
만든 인덱스는 VPC 안에서 한 번 설정합니다.

```bash
curl -X PUT "https://<OpenSearch endpoint>/_all/_settings" \
  --aws-sigv4 "aws:amz:<region>:es" --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  -H "x-amz-security-token: $AWS_SESSION_TOKEN" \
  -H 'Content-Type: application/json' -d '{"index": {"number_of_replicas": 1}}'
```

### 백업과 복원

저장소들은 서로를 가리키므로(레지스트리는 문서별 텍스트 단위를 기록하고, 그래프와
인덱스 산출물은 그 텍스트 단위를 인용함) 모두 같은 시점으로 복원하거나 아예 복원하지
않습니다. S3의 코퍼스가 원본입니다.

| 저장소 | 백업 | 기간 |
|---|---|---|
| Neptune | 자동 백업, 특정 시점 복원 | `backup_retention_days`(기본값 7일) |
| OpenSearch | 매시간 자동 스냅샷(`cs-automated-enc`) | 14일 |
| DynamoDB 문서 상태 테이블 | 특정 시점 복구(PITR) | 35일 |
| S3 캐시 버킷 | 없음(체크포인트는 다시 만들 수 있음) | — |

**Neptune 보존 기간 안이면**

1. 수집을 멈춥니다(`RUNNING` 실행이 없어야 함).
2. 마지막 실행이 끝난 뒤 찍힌 OpenSearch 스냅샷을 고르고
   (`GET _snapshot/cs-automated-enc/_all`) 그 시작 시각을 복원 시점으로 씁니다.
3. Neptune(`aws neptune restore-db-cluster-to-point-in-time`)과 테이블
   (`aws dynamodb restore-table-to-point-in-time`)을 그 시점으로 복원합니다. 둘 다 기본
   설정의 새 리소스를 만들므로 Neptune 인스턴스를 추가하고 테이블의 PITR과 삭제 보호를
   다시 켭니다.
4. 도메인의 현재 인덱스를 지운 뒤 스냅샷의 인덱스와 별칭을 복원합니다
   (`POST _snapshot/cs-automated-enc/<snapshot>/_restore`).
5. 데이터 플레인이 복원한 클러스터와 테이블을 가리키게 하고
   (`UKR__AWS__NEPTUNE__ENDPOINT`, `UKR__AWS__DYNAMODB__TABLE_NAME`), 태스크 역할,
   DynamoDB 게이트웨이 엔드포인트 정책, 경보도 그 리소스에 맞게 부여합니다(모두 스택
   자체 리소스로 범위가 정해져 있음).

**그 밖의 경우**에는 `indexing.reset: true`로 전체 재구축을 한 번 실행합니다
([다시 수집하기](#다시-수집하기)). 다른 네임스페이스가 테이블을 함께 쓸 때만 테이블을
먼저 복원합니다.

## 실행이 실패할 때

### 중단된 실행

인덱싱 도중 멈춘 증분 실행(태스크 종료, 저장소나 레지스트리 연결 불가, 실패 한도
초과)은 손으로 고칠 필요 없이 다시 실행하면 됩니다. 인덱싱은 먼저 새 문서와 바뀐
문서를 이전·새 텍스트 단위와 함께, 삭제된 문서는 제거하기 전에 `PENDING`으로 기록하고,
쓰기가 끝난 뒤에야 `PROCESSED`나 `FAILED`로 바꿉니다(삭제된 문서는 레코드를 지움). 그래서
다음 실행은 코퍼스에 남은 `PENDING` 문서를, 그사이 되살린 삭제 문서까지 다시 추출하고,
사라진 문서에 쓰인 내용은 지웁니다
([설계 §5.3](./design.ko.md#53-크래시-복구선기록-프로토콜)).

- **`PENDING` 레코드는 그대로 두세요**(`status` `pending`, `content_hash` `pending`).
  다음 실행은 그 텍스트 단위로 중단된 쓰기를 찾습니다.
- 중단은 실패로 세지 않습니다. 추출 중에 멈추거나 죽으면 `failure_count`와 `retry_count`가
  그대로 남고, 인덱싱 중 `PENDING` 레코드를 쓴 뒤라면 둘 다 초기화합니다. 죽은 실행은
  `attempt_count`로 셉니다([격리된 문서](#격리된-문서)).
- 사용자 정의 레지스트리 정책에는 `dynamodb:BatchWriteItem`과 `BatchGetItem`이
  필요합니다(CDK 권한에는 둘 다 있음).

### 실패한 문서

다음 다섯 경우에 문서가 `FAILED`로 기록됩니다.

- 텍스트 단위 하나에서 번역, 그래프 추출, 글리닝, 주장 추출이 실패함
- 텍스트 단위를 포함하는 기본 수준 커뮤니티가 리포트를 받지 못함(상위 수준 리포트
  실패는 WARNING과 `reports_failed` 지표로만 남음. 리포트 없는 커뮤니티가
  `indexing.max_failure_rate`를 넘으면 커뮤니티 탐지 단계가 실패함)
- `--continue-on-error` 실행에서 산출물 단계(번역부터 커뮤니티 탐지까지)가 비정상
  종료해 그 실행의 모든 문서를 `FAILED`로 기록함
- 산출물 하나를 저장소에 쓰는 데 실패함
- 텍스트 단위의 추출 레코드를 추출 저장소에 쓰지 못함(그래프 해석과 주장 추출의
  `records_not_stored`)

다음 실행은 `FAILED` 문서를 바뀐 문서로 보고, 실패한 실행이 쓴 것을 지운 뒤 다시
처리합니다. 같은 파이프라인 ID로 `--resume-from-stage` 없이 다시 실행하면, 이번 실행에서
켜진 완료 단계 중 실패한 텍스트 단위를 기록한 가장 앞선 단계부터 재개해 그 단위를 다시
시도합니다(WARNING `Stage '...' failed on N text units; resuming from it ...`). 성공한
단위는 추출 저장소에 남아 있습니다. 그보다 뒤를 가리키는 `--resume-from-stage`는 WARNING과
함께 그대로 따르며, 인덱싱 단계는 그 문서를 다시 시도하지 않고 다시 `FAILED`로 기록합니다. 이 실행도
`indexing.max_document_failures`에 한 번 더 세므로, 한도를 넘으면 실패한 단계부터 재개해도 다시
시도하지 않고 제외합니다.
재실행에서 켜지지 않은 단계의 실패는 그 단계가 재개 지점이나 그 뒤에 있으면 복원한 상태에서
버립니다. 재개 지점 앞의 꺼진 단계는 실패를 유지합니다. 인덱싱만 실행하는 단계 묶음은 그
실패를 한 번 기록하고, 이미 끝난 그 묶음을 다시 실행하면 거부합니다(`PipelineResumeError`:
다시 시도하려면 메시지에 나온 단계를 켜세요. 명시한 `--resume-from-stage`는 WARNING과 함께
따르며, 그 실패를 다시 셉니다). 재시도 한도에 이른 문서가
있으면 재실행이 코퍼스를 다시 비교해 그 문서를 건너뜁니다.

그 밖의 단계 실패는 인덱싱 전에 실행을 멈추므로 저장소는 이전 상태로 남습니다.
레지스트리에는 단위 자체의 오류가 아닌 오류로 실패한 문서의 `retry_count`를 하나 올리는
것만 기록하며(아래 참고), 다음 실행이 같은 델타를 다시 처리합니다. 예외로 그래프
추출이 모든 단위에서 실패하면(`Critical stage 'graph_extraction' ... produced 0 outputs.
It failed on N files: ...`) 단위 자체의 오류로 실패한 문서를 `FAILED`로 기록해 아래 재시도
한도가 적용됩니다.

이 경로에서도, 인덱싱 단계가 문서를 `FAILED`로 기록할 때도 재시도 한도에는 단위 자체의
오류만 셉니다. 파싱할 수 없거나 잘린 응답, 그리고 모델이 내용 때문에 거부한 요청(Bedrock
`ValidationException`, 로컬 서버의 HTTP 400/413/422)입니다. 단, 모델이나 설정을 거부한
경우(잘못된 모델 ID, 추론 프로필이 필요한 모델, 지원하지 않는 파라미터)는 제외합니다. 그
밖의 오류는 세지 않습니다. 스로틀링, 5xx, 시간 초과, 연결, 자격 증명(만료된 SSO 토큰,
`NoCredentialsError`), 리전 누락, 할당량(`ServiceQuotaExceededException`), 스트림 오류와
버그, 저장소 쓰기 실패는 모든 문서를 똑같이 실패시킵니다. 이런 문서는 원인을 고칠 때까지 세지
않은 채 `PENDING`(또는 다음 실행이 다시 시도하는 `FAILED`)으로 남으므로, 장애나 잘못된 설정
때문에 제외되지는 않습니다. 같은 내용으로 연속 실패한 실행 수는 `retry_count`에 남고(내용이
바뀌거나, 성공하거나, 문서 자체의 오류로 실패하면 처음부터 다시 셉니다. 추출 중에 멈추거나
죽으면 그대로 두고, 인덱싱 중 `PENDING` 레코드를 쓴 뒤라면 처음부터 다시 셉니다), 두
번째부터는 실행이 파일과 횟수를 경고하고 `documents_retrying`에 셉니다. 보낸 모든 단위가
실패하고 그 자체의 오류가 원인 하나이면(메시지와 관계없이 오류 유형, 코드, HTTP 상태가 같거나,
모두 파싱할 수 없거나 잘린 출력. 함께 스로틀링되거나 시간 초과된 단위는 빼고 봄), 여러 문서의
단위에서 난 출력 오류는 정상 요청 없이 각 문서 자체의 오류로 셉니다. 서버 쪽 출력 한도는 그 출력을
모두 자르지만, 짧은 정상 요청으로는 이를 각 문서 자체의 출력과 구별할 수 없기 때문입니다. 실패한
실행마다 한 번씩 세므로(첫 실행부터 `FAILED`, `1/N`. 사이에 낀 장애 실행은 세지도 끊지도 않음), 늘
실패하는 배치는 `indexing.max_document_failures`번 뒤 제외되어 델타의 삭제와 함께 모든 실행을 막지
못합니다. 대가로, 그만큼의 실행 동안 이어지는 서버 쪽 출력 한도도 내용이나 추출 설정이 바뀔 때까지
그 문서들을 제외합니다(한도를 고친 뒤 파일을 수정하거나 설정을 바꾸세요).
그 밖에는 같은 체인으로 보낸
정상 요청 하나(LLM 응답 캐시를 쓰지 않음)로 판단합니다. 이 요청도 실패하면(출력 오류일 때는
파싱할 수 없는 응답도 실패) 아무것도 세지 않고, 응답이 오면 오류를 세므로 모델이 거부하는
문서만 있는 델타도, 출력이 끝내 파싱되지 않는 문서 하나도 결국 제외됩니다. 그래서 아무 변경
없이 나중에 풀리는 내용 오류로도 문서가 제외될 수 있습니다. 엔티티를 나열했지만 이름이 있는
엔티티가 하나도 없는 그래프 추출 출력(중지 사유 없이 잘림)은 잘린 출력으로 봅니다(토큰 한도에서
잘린 출력처럼 다시 시도하지 않음). 더 뒤에서 잘린 출력은 잘린 곳 앞의 엔티티를 유지하고
WARNING(`XML output ends with <entities> unclosed`)을 남깁니다. OpenAI 호환 서버가 요청한
`max_tokens`까지 세어 컨텍스트 길이로 거부한 경우("you requested N tokens", "'max_tokens' or
'max_completion_tokens' is too large")는 설정 문제로 보고 세지 않습니다.

- **재시도 한도.** 내용과 추출 설정이 그대로인 채 `indexing.max_document_failures`(기본값
  `3`)번 연속으로 문서 자체의 오류나 여러 문서에서 똑같이 실패한 응답(위 설명 참고)으로
  `FAILED`가 기록되면(추출 설정을 바꾸면 다시 셈) 문서는
  `FAILED`로 남고 인덱싱한 내용을 유지한 채 건너뜁니다. 실행마다 파일 이름을 밝힌
  WARNING을 남기고 삭제를 포함한 나머지 델타를 처리합니다. 원인(흔히 너무 크거나 잘못된
  파일)을 고치거나, 파일을 수정하거나, 한도를 올리세요.
- **레지스트리에 담기에 너무 큼.** 텍스트 단위 ID가 DynamoDB 항목 하나(400KB, 약
  10,000개, 수정한 문서는 그 절반)를 넘는 문서는 아무것도 쓰기 전에 파일 이름을 밝힌
  오류로 인덱싱을 실패시킵니다. 파일을 나누세요.
- **쓰기 실패**는 먼저 재시도합니다. 429, 502, 503, 504로 거부된 OpenSearch 벌크 항목은
  백오프하며 최대 네 번, 모든 Neptune 요청은 새 연결로 최대
  `indexing.neptune.max_attempts`(기본값 `4`)번 시도합니다. 한 산출물 유형의 쓰기가
  `indexing.max_failure_rate`(기본값 `0.2`)를 넘게 실패하면 인덱싱이 실패하고 문서를
  기록하지 않습니다. 그 이하면 실패한 산출물을 가진 문서만 `FAILED`가 됩니다(백엔드가
  항목을 밝히지 않으면 실행의 모든 문서). 이 문서는 다음 실행이 재시도 한도에 세지 않고 다시
  시도합니다. `IndexingFailures`와 `ExtractionFailures`
  [경보](#경보)가 두 경우를 알립니다.

### 격리된 문서

처리할 때마다 태스크를 죽이는 문서(메모리 부족, 프로세스를 죽이는 파서)는 이후 실행도
모두 멈추게 하므로, 실행은 죽은 횟수를 세고 원인 문서를 가려냅니다.

- **죽은 경우만 셉니다.** 실행은 처리하는 문서마다 시도 표시를 열고, 프로세스 안에서
  어떻게 끝나든(완료, 실패, SIGTERM, Ctrl-C) 닫습니다. 쓰기 임대를 잃은 실행은 다른
  실행이 임대를 가지고 있지 않으면(아무도 없거나 그 임대가 만료됨) 임대를 다시 얻어
  닫습니다. 다른 실행이 가지고 있으면 열린 채 두므로 죽은 것으로 한 번 셉니다. 태스크가
  죽어 열린 채 남은 시도는 다음 실행이 셉니다(`attempt_count`, 배치는 `crash_batch`).
  `indexing.reset`은 세지 않습니다.
- **격리.** `indexing.max_document_attempts`(기본값 `3`)번 연달아 죽고 마지막이 그
  문서만 처리한 실행이었다면, 다음 실행이 파일 이름을 밝힌 WARNING을 남기고 시도들이
  쓴 것을 지운 뒤 `QUARANTINED`로 기록하고(이유는 `error_info`) 나머지를 인덱싱합니다
  (`total_documents_quarantined`, CDK 경보 `DocumentsQuarantined`).
- **이분 탐색.** 여러 문서를 함께 처리하다 죽었다면 다음 실행은 가장 작은 죽은 배치의
  절반만 처리하고 나머지 새 문서와 바뀐 문서는 미룹니다(WARNING이 그 절반을 알리고,
  `total_documents_deferred`가 미룬 문서를 세며, 미룬 문서는 인덱싱된 버전을 유지).
  의심 문서 `n`개 중 원인 하나는 약 `log2(n) + 1`번의 실행 뒤 격리됩니다.
  **`total_documents_deferred`가 0이 될 때까지 새 파이프라인 ID로 실행하세요.**
- **재시도.** 격리된 문서는 파일이 바뀌거나 `indexing.max_document_attempts`를
  `attempt_count`보다 높이면 다시 시도합니다(추출 설정을 바꿔도 다시 시도하지 않음).
  `indexing.reset`도 이 문서를 건너뛰고 레코드를 남깁니다. 전체 재구축은 시도를 세지 않으므로,
  처음 보는 문서가 태스크를 죽이면 모든 전체 재구축이 죽습니다. 그 문서가 격리될 때까지 증분
  실행을 돌리세요. 인덱싱
  전에 죽은 바뀐 문서는 인덱싱된 버전을 유지하고, 인덱싱 중에 죽은 문서는 지워집니다.
- **한계.** 무엇을 처리하든 죽는 태스크(망가진 이미지)는 문서를 하나씩 격리합니다.
  배포를 고친 뒤 파일을 수정하거나 한도를 올리세요. 의심 문서 하나만 처리하는 실행에서
  우연히 한 번 죽어도 격리됩니다. SIGTERM 정지 제한 시간(ECS에서 120초) 안에 시도를 닫은
  실행은 세지 않지만, 저장소 요청 하나가 180초까지 걸릴 수 있어 제한 시간이 지나 요청 도중
  종료된 실행은 죽음으로 셀 수 있습니다. 텍스트 단위를 만들지 못하는 문서는 `PENDING`으로
  남고 같은 방식으로 격리됩니다.
- **재개한 실행**은 같은 파이프라인 ID의 델타를 다시 쓰지만, 레지스트리가 그 델타와
  맞지 않거나 원인 문서를 가려낼 수 있으면 `document_loading`부터 재개합니다(WARNING
  `... resuming from 'document_loading' ...`). 델타를 계산한 뒤 `document_loading`의
  입력(초기화, 재시도, 말뭉치 설정)이 바뀐 경우에도 마찬가지입니다(WARNING
  `Rewinding resume point from ... to 'document_loading'`). 끝난 파이프라인 ID를 다시
  실행해도 재개로 취급합니다. 그 뒤 다른 파이프라인 ID가 델타 문서의 레코드를 썼거나
  (레코드마다 마지막으로 쓴 파이프라인 ID가 남습니다. 이 기록이 생기기 전에 쓴 레코드는
  비어 있으며, `PENDING`이나 `FAILED`이거나 열린 시도가 있으면 다른 파이프라인 ID가 쓴
  것으로 보고 `PROCESSED`이면 다른 검사에 맡깁니다), 델타의 문서를 다른 추출 설정으로
  커밋했거나(쓴 ID가 없는 레코드는 델타가 설정 변경으로 다시 추출하는 문서여도
  해당합니다), 델타 밖의 문서를 추가·변경·삭제했으면 다시 비교합니다. 실행은 자기
  범위(인덱스 네임스페이스와 원본 범위) 안에서만 비교하므로 마지막 검사도 그 범위만
  읽습니다. 델타가 지우는 문서를 그 뒤 다른 파이프라인 ID(또는 쓴 ID가 없는 레코드)가
  다른 범위(델타가 폐기하는 범위가 아닌 곳)에 기록했어도 다시 비교하므로, 재실행이 다른 범위의 살아 있는 문서를
  지우지 않습니다. 자신이 종료되거나 정지된 뒤 재개한 실행은 델타를 그대로 씁니다.

### 동시 실행

레지스트리를 쓰면 한 인덱스 네임스페이스에는 한 번에 한 실행만 씁니다. 실행은 첫 단계
전에 네임스페이스의 쓰기 임대를 얻고 `indexing.lease_seconds`(기본값 600초)의 6분의 1마다
갱신합니다. 두 번째 실행은 저장소를 읽거나 무엇을 쓰기 전에 멈춥니다.

```text
NamespaceLeaseError: Index namespace '<namespace>' is being written by
<host>:<pid>:<id> (its lease runs until <time> unless renewed) ...
```

그 실행이 끝나기를 기다렸다가 다시 시작하세요. 죽은 실행의 임대는 마지막 갱신 뒤
`lease_seconds` 안에 만료되므로 정리할 것이 없습니다. 갱신이 계속 실패했거나 다른
실행이 임대를 가져간 실행은 다음 쓰기 전에
`NamespaceLeaseError: This run lost the write lease ...`로 멈춥니다. 다른 실행이 끝난
뒤 다시 실행하세요.

- **쓰기 차단.** 모든 단계 시작, 체크포인트 업로드, 레지스트리 쓰기, 인덱싱 단계, 저장소
  쓰기 시도(재시도 포함)는 먼저 임대를 확인하고, 마지막으로 성공한 갱신 뒤 남은 기간이
  3분의 1 미만이면 멈춥니다. `lease_seconds`는 가장 긴 쓰기 요청(180초)의 세 배 이상으로
  두세요(더 짧으면 WARNING을 남김) ([설계 §5.6](./design.ko.md#56-동시성)).
- **죽거나 중지된 실행의 임대를 넘겨받을 때.** 쓰기 차단을 이미 지난 쓰기는 임대가 만료된 뒤에
  도착할 수 있지만(프로세스가 멈췄거나 요청이 느림), 그 만료 시각에 가장 긴 요청 시간을 더한
  시각보다 늦지는 않습니다(레지스트리 쓰기도 시도마다 확인함). 그래서 놓은 것이 아니라 만료된
  임대를(보유자가 죽었거나 멈췄거나 중지됨. 중지된 실행은 보낸 요청이 아직 남아 있을 수 있어
  임대를 만료된 상태로 둠) 넘겨받은 실행은 그때까지 첫 쓰기를 미루고(그 앞의 파싱 같은 단계는
  그대로 진행) WARNING
  (`Took over the expired write lease ... the first write waits until ...`)을 남깁니다.
  이전 임대가 만료된 뒤 최대 약 3분(180초에 5초 여유)입니다. 중지된 실행의 메시지가 언제까지인지
  알려 줍니다(`the next run on its namespace waits until ...`). SIGTERM을 받으면 기다리기를
  멈춥니다. 모든 저장소가 내장(`backend: local`, SQLite)이면 중지된 실행은 임대를 놓습니다. 내장
  저장소의 쓰기는 모두 데이터베이스 잠금을 쥔 뒤 임대를 확인하므로, 중지 뒤에 남은 쓰기는 다음
  실행의 쓰기 뒤에 도착하지 않고 실패합니다. 넘겨받기는 저장소 쓰기 한
  번(`search.store_timeout_seconds`)까지만 기다립니다. 요청을 이미 만든 채 그보다 오래 멈춘 프로세스는 여전히 새 보유자 뒤에 쓸 수
  있으며, 이를 막으려면 저장소 자체가 쓰기를 차단해야 합니다. DynamoDB에서 TTL이 이미 지운
  만료된 임대 항목은 놓은 것으로 읽힙니다(TTL 삭제는 보통 3분보다 훨씬 오래 걸림).
- **라이브러리로 쓸 때의 오류.** 실패한 인덱싱 단계는 스레드가 이미 보낸 저장소 요청이 끝날
  때까지 기다린 뒤 예외를 올리므로, 같은 프로세스에서 다시 실행하는 호출자가 그 요청과
  경쟁하지 않습니다. 중지(SIGTERM, Ctrl-C)는 프로세스가 끝나므로 기다리지 않습니다.
- **체크포인트.** 실행은 임대를 얻은 뒤에만 S3 체크포인트를 내려받고, 멈춘 실행은
  임대를 내려놓기 전에 체크포인트를 올리며(저장소와 레지스트리 쓰기만 미루는 넘겨받기 대기
  중에도), 임대를 잃은 실행은 올리지 않습니다.
- **임대 위치.** DynamoDB에서는 `namespace` = `<namespace>`, `doc_id` = `#lease`인
  항목(`lease_owner`, 테이블의 TTL 속성인 `lease_expires_at`), SQLite에서는 `leases`
  테이블의 네임스페이스 행입니다. 그 실행이 없어진 것을 확인한 뒤에만 손으로 지우세요.
- **권한.** 사용자 정의 정책에는 `dynamodb:PutItem`, `GetItem`, `DeleteItem`(라이브러리가
  테이블을 만들면 `UpdateTimeToLive`도)이 필요합니다.
- **네임스페이스**는 인덱스 접미사(`index_value`, 없으면 `default`)와
  `indexing.additional_suffix`에, 기본값이 아닌 `indexing.opensearch.*_index_prefix` /
  `indexing.neptune.*_label_prefix` 값의 태그를 더한 것입니다. 네임스페이스가 다르면
  동시에 실행할 수 있습니다.

**레지스트리가 없으면 아무것도 잠그지 않으며** 한 인덱스 접미사에 대한 동시 실행은
지원하지 않습니다. 전체 재인덱싱이 별칭을 옮기고 이전 인덱스를 지우므로 한 실행이 다른
실행이 쓰는 인덱스를 지울 수 있습니다. `IncrementalIndexer` API를 직접 쓸 때도 임대를
얻지 않습니다.

**CDK 스택에서는** 실행을 예약하는 것이 없고 상태 머신이 두 번째 실행을 막지 않습니다
(임대가 실패시킴). 먼저 `RUNNING` 실행이 있는지 확인하고([실행 전 확인](#실행-전-확인)),
직접 추가한 일정은 이전 실행이 아직 돌 수 없을 때만(최대 `sfn_timeout_hours`, 기본값
6시간) 실행되게 하세요.

### 시작 시 엔드포인트 검사

CLI는 첫 모델 호출 전에 실행에 필요한 저장소 엔드포인트가 설정되어 있는지 검사하고,
없으면 종료 코드 2로 끝납니다.

```text
run-ingestion: error: Missing endpoint configuration for the indexing stage: aws.neptune.endpoint (env UKR__AWS__NEPTUNE__ENDPOINT), aws.opensearch.endpoint (env UKR__AWS__OPENSEARCH__ENDPOINT). Set them in the config file or the environment.
```

- `run-ingestion`은 `indexing` 단계가 켜져 있으면 두 엔드포인트를 모두 검사합니다
  (`--verify-metadata`는 제외).
- `run-rag`와 `run-eval`은 모든 전략에서 OpenSearch를, `local`, `drift`, `mix`,
  `hybrid`에서는 Neptune도 검사합니다. `auto`는 라우팅할 수 있는 모든 전략을
  검사합니다(`search.auto_routable_strategies`에서
  `graph.community_detection.enabled`가 false면 `global`과 `drift`를 뺌).

설정은 되어 있지만 잘못되었거나 연결할 수 없는 엔드포인트는 첫 연결에서 빈 결과가 아닌
오류로 실패합니다.

## 경보

CDK 스택에서는 수집 태스크가 실패하면 실행 ID와 오류를 SNS 토픽에 게시하고, 아래
경보도 같은 토픽으로 알립니다. 파이프라인 실행 경보 다섯 개, 저장소 상태 경보 일곱
개(Neptune Serverless에서는 여덟 개), 용량 경보 두 개이며, `-c alarm_email=<address>`로
배포해야 구독자가 생깁니다. 태스크가 실패하면 `PipelineFailures`도 발생해 맥락 없는
알림이 한 번 더 오는데, 이 경보는 메시지로 알릴 수 없는 실패(태스크의 오류 처리기가
잡지 못하는 오류, 예를 들어 `pipeline_id`가 없는 입력이나 실패한 게시)도 잡습니다.

**파이프라인 실행**

| 경보 | 발생 조건 |
|---|---|
| `PipelineFailures` | Step Functions 실행이 실패함 |
| `PipelineTimedOutOrAborted` | 실행이 상태 머신 제한 시간을 넘겼거나 중지됨 |
| `IndexingFailures` | 실행이 성공했더라도 인덱싱에서 실패한 항목이 있음 |
| `ExtractionFailures` | 추출, gleaning, 주장 추출이 일부 텍스트 단위에서 실패함 |
| `DocumentsQuarantined` | 실행이 문서를 격리함([격리된 문서](#격리된-문서)) |

뒤의 세 경보는 `run-ingestion --metrics-sink cloudwatch`가 단계가 끝날 때마다 내보내는
EMF 지표를 읽습니다(중지된 실행도 끝낸 단계는 보고함). 태스크는 지표에
`Environment` 차원(`env_name`)을 붙이고 경보와 대시보드는 그 시계열만 읽으므로 dev
실행이 prod 경보를 울리지 않습니다. `UKR_METRICS_ENVIRONMENT`가 없는 실행은 어떤 경보도
읽지 않는 시계열을 내보냅니다.

**저장소 상태**

실행이 없을 때도 발생하므로 실행이 실패하기 전에 성능 저하를 알 수 있습니다.

| 경보 | 발생 조건 | 조치 |
|---|---|---|
| `OpenSearchClusterRed` | 할당되지 않은 프라이머리 샤드가 있음(5분) | 클러스터 상태와 실패한 노드를 확인합니다. 해당 인덱스 검색에서 데이터가 빠지므로 green이나 yellow가 될 때까지 수집을 멈춥니다 |
| `OpenSearchFreeStorageLow` | 가장 많이 찬 데이터 노드의 여유 공간이 `opensearch_volume_gib`의 40% 이하 | 쓰기 차단 기준에 이르기 전에 `opensearch_volume_gib`(또는 `opensearch_count`)를 늘려 다시 배포합니다 |
| `OpenSearchJvmPressure` | JVM 메모리 압력이 3분 동안 95% 이상 | 더 큰 `opensearch_instance`를 쓰거나 데이터 노드를 늘립니다 |
| `DocStatusWriteThrottled` | 문서 상태 테이블이 쓰기를 스로틀링함(5분) | 레지스트리 호출은 백오프하며 재시도하고, 그래도 스로틀링되면 `DocStatusRegistryError`로 실행이 실패합니다. 겹친 실행이 있는지 확인하고 해소된 뒤 다시 실행합니다 |
| `NeptuneHighCpu` | 라이터의 평균 CPU가 15분 동안 90% 이상 | 지속적인 포화입니다. 더 큰 `neptune_instance`(또는 `neptune_serverless_ncu` 최댓값)를 쓰거나, 올려 두었다면 `indexing.neptune.index_concurrency`(기본값 `1`)를 낮춥니다 |
| `NeptuneLowFreeableMemory` | 5분 안에 라이터의 가용 메모리가 한 번이라도 512MiB 이하. Serverless에서는 라이터가 최대 용량일 때만 | 메모리 부족 오류가 날 수 있습니다. 더 큰 `neptune_instance`(또는 `neptune_serverless_ncu` 최댓값)를 씁니다 |
| `NeptuneServerlessAtMaxCapacity` | Serverless만: 라이터가 5분 구간 3회 연속 `neptune_serverless_ncu` 최댓값에 있음 | 최댓값을 높입니다(부하 전에 용량이 확보되도록 최솟값을 높여도 됨) |
| `NeptuneGremlinErrors` | 3분 연속 어느 인스턴스에서든 Gremlin 서버 오류가 있음 | 지속적인 실패입니다(재시도되는 쓰기 충돌은 짧게 몰려 이 경보를 울리지 않음). 태스크 로그에서 실패한 요청을 찾습니다 |

**용량**

| 경보 | 발생 조건 | 조치 |
|---|---|---|
| `BedrockThrottling` | 5분 구간 3개 연속으로 스로틀링된 호출이 100회 이상. 계정과 리전의 모든 모델을 합산함(다른 Bedrock 사용량도 포함) | 스로틀링된 호출은 재시도하므로 실행이 실패하지 않고 느려집니다. `processing.max_concurrency`를 낮추거나 할당량 증가를 요청합니다 |
| `IngestionTaskMemory` | 수집 태스크가 3분 동안 메모리의 90% 이상을 씀(Container Insights) | Fargate는 메모리를 넘은 태스크를 종료해 실행 중인 단계를 잃습니다. `fargate_memory`를 늘립니다 |

대시보드에는 단계별 소요 시간, 태스크 메모리와 CPU, 모델별 Bedrock 호출·스로틀링·지연
시간과 경보가 보지 않는 증분 실행의 품질 저하(`total_documents_deferred`,
`community_rebuild_recommended`, `total_records_not_stored`, `underived_items`)도
표시됩니다.

## 질의 권한과 테넌트 격리

질의(`run-rag`, `run-eval`, 내장한 `GraphRAGChain`)는 별도의 태스크 정의와 역할을
씁니다(`QueryTaskDefinitionArn`, `QueryTaskRoleArn`.
[`iac/README.md`](../iac/README.md#4-query-from-inside-the-vpc) 참고). 이 역할은
Neptune `connect`와 `ReadDataViaQuery`, OpenSearch `GET`/`HEAD`와
`<index>/_search`에만 허용한 `POST`(`_search`로 끝나지만 쓰기인 더 깊은 경로는 명시적
거부로 막음. `POST <index>/_doc/_search`는 문서를 만듦), Bedrock 모델·재순위화·가드레일
호출, 캐시 버킷의 `eval/` 접두사만 가집니다. DynamoDB에는 접근할 수 없고 두 저장소
어디에도 쓰거나 지울 수 없습니다. 수집 태스크 정의(역할이 삭제할 수 있음)로 질의하지
말고, 직접 만든 서빙 계층도 같은 권한의 역할을 쓰세요. 세분화된 액세스 제어에서는
이 패키지의 인덱스에 `indices:data/read/search`, `indices:admin/mappings/get`(임베딩
모델 확인), `indices:monitor/settings/get`(살아 있는 인덱스의 `max_result_window`·
`max_terms_count`에 맞춘 검색 크기)을 부여하세요. 뒤의 둘이 없어도 질의는 실행되며
각각 경고를 한 번 남깁니다.

**인덱스 접미사는 이름 분리일 뿐 접근 제어 경계가 아닙니다.** `RAGInput.suffix` /
`--suffix`는 질의가 읽을 인덱스와 레이블을 고르며(대상을 넓히지 못하도록 검증함), 질의
역할을 가진 호출자는 어떤 접미사든 지정할 수 있습니다. 멀티 테넌트 서빙 계층은 접미사를
요청 필드가 아니라 인증된 주체(호출자 토큰의 테넌트 클레임)로 서버에서 정해야 합니다.
인프라로 테넌트를 격리하려면 다음을 쓰세요.

- OpenSearch: 테넌트마다 질의 역할을 두고 `es:ESHttpGet`, `es:ESHttpHead`,
  `es:ESHttpPost`의 리소스를 그 테넌트의 인덱스로 한정합니다
  (`<도메인 ARN>/graphrag-*-<suffix>`, `<도메인 ARN>/graphrag-*-<suffix>/*`. 어떤
  접미사도 `-<다른 접미사>`로 끝나지 않게 정하거나 인덱스 접두사를 다르게 둠). 문서·벌크
  API 거부 구문은 그대로 둡니다. 문서·필드 수준 규칙이 필요하면 세분화된 액세스
  제어(fine-grained access control)를 켜고 테넌트 역할마다 `*-<suffix>*`로 한정한
  OpenSearch 역할을 매핑합니다.
- Neptune: IAM은 클러스터 단위로만 권한을 주므로, 다른 테넌트가 그래프를 읽을 수 없어야
  하면 테넌트마다 클러스터를 따로 둡니다(테넌트마다 배포 하나 또는 `env_name` 하나).
- Bedrock과 가드레일은 공유하며 테넌트 데이터를 보관하지 않습니다.

## 관측성

추적은 기본적으로 꺼져 있으며 설정은 [사용자 가이드](./user-guide.ko.md#관측성)에
있습니다. 4318 포트로 OTLP를 받는 로컬 Jaeger로 트레이스를 보내려면 다음과 같이 합니다.

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

질의를 실행하고 <http://localhost:16686>을 열면 `unified-kg-rag` 서비스에서 질의마다
`kg_rag.query` 트레이스, 수집 실행마다 `kg_rag.ingestion.run` 트레이스를 볼 수
있습니다. 수집기나 상용 엔드포인트를 쓸 때는 `OTEL_EXPORTER_OTLP_ENDPOINT`를 그
주소로 지정하고, 자격 증명은 Secrets Manager나 태스크 시크릿에서
`OTEL_EXPORTER_OTLP_HEADERS`로 넘기며, gRPC 엔드포인트(4317 포트)는
`OTEL_EXPORTER_OTLP_PROTOCOL=grpc`를 설정합니다.

동작 확인: 시작 로그에 `Observability enabled (backend: 'otlp', ...)`가 나오고, 스팬
안에서 남긴 로그 레코드에는 그 스팬의 `trace_id`와 `span_id`가 있습니다. 질의
트레이스 아래에는 `kg_rag.retrieve`, `kg_rag.fusion`, `kg_rag.context_build`,
`chat <model>` 스팬이 있고, 모델 스팬의 `gen_ai.usage.input_tokens`가 해당 실행의
토큰 사용량입니다.

### Langfuse

Langfuse와 Phoenix는 OTLP를 받으므로 `otel` extra만 있으면 되고 벤더 SDK는 필요
없습니다. 설정은 환경 변수에서만 읽으며, 표준 `OTEL_EXPORTER_OTLP_*` 변수가 벤더
변수보다 우선합니다. 로컬 Langfuse는 저장소의 Docker Compose 파일로 실행합니다(Langfuse
v4를 받으며 v3도 같은 방식으로 동작합니다).

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

- `LANGFUSE_HOST`는 기본 호스트가 없으므로 필수이며, Langfuse Cloud는 리전 URL(예:
  `https://cloud.langfuse.com`)을 지정합니다. 익스포터는 HTTP로만
  `<host>/api/public/otel/v1/traces`에 보내며(gRPC면 시작할 때 실패),
  `OTEL_EXPORTER_OTLP_ENDPOINT`를 대신 쓸 때는 `<host>/api/public/otel`을
  지정합니다(호스트 루트만 주면 404).
- 두 키는 필수이며 Basic `Authorization` 헤더와
  `x-langfuse-ingestion-version: 4`로 보냅니다. `OTEL_EXPORTER_OTLP_HEADERS`를 직접
  지정할 때도 이 버전 헤더를 넣으세요. 넣지 않으면 Langfuse Cloud에서 트레이스가 몇
  분씩 늦게 보일 수 있습니다.
- 트레이스는 Sessions에서 `RAGInput.conversation_id`별로, Users에서
  `RAGInput.user_id`별로 묶입니다. Langfuse v4의 기본값인 `events_only` 모드는
  `/api/public/traces`, `/observations`, `/sessions`를 막으므로 스팬은
  `/api/public/v2/observations`나 UI에서 확인합니다.

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

다른 설정이 없으면 스팬은 `http://localhost:6006/v1/traces`(`OTEL_EXPORTER_OTLP_PROTOCOL=grpc`이면
`:4317`)로 가고 `service_name` 이름의 프로젝트에 쌓입니다(`PHOENIX_PROJECT_NAME`이 우선).
원격 Phoenix는 `PHOENIX_COLLECTOR_ENDPOINT`를 서버 루트로 지정하고, 서버가 요구하면
`PHOENIX_API_KEY`(Bearer `authorization` 헤더로 보냄)를 설정합니다. 스팬에
OpenInference 종류(`LLM`, `EMBEDDING`, `RETRIEVER`, `RERANKER`, `CHAIN`)가 붙으며
Sessions 화면에서 대화의 질의가 묶입니다.

### ADOT 수집기 사이드카를 거친 CloudWatch

CloudWatch는 SigV4로 서명한 OTLP 트레이스를 받으므로, 앱 옆의 수집기가 서명해 전달합니다.
CDK 스택에서는 `-c enable_cloudwatch_tracing=true`가 AWS Distro for OpenTelemetry(ADOT)
수집기 사이드카를 추가하고 앱이 그 수집기로 보내게 합니다. 전체 절차(Transaction Search
켜기, 이미지, 수집기 이미지 첫 pull, 트레이스를 볼 곳, 비용, CDK 스택 없이 쓰는 수집기 설정,
계정 수준의 보조 수단인 Bedrock 모델 호출 로깅)는
[iac/README.md: CloudWatch tracing](../iac/README.md#cloudwatch-tracing-optional)(영문)에
있습니다.

### 트레이스 구조

| 스팬 | 부모 | 속성 |
|---|---|---|
| `kg_rag.ingestion.run` | 루트 | 실행이 실패하면 상태 `ERROR` |
| `kg_rag.ingestion.stage` | run | 단계마다 하나. 입력·출력 수, 상태, 단계의 수치 지표(`kg_rag.stage.<metric>`), 예외로 끝나면 `error.type`과 예외 |
| `kg_rag.batch` | stage, `run-eval` 답변 생성에서는 루트 | 항목 수와 실패 수. `run-eval`에서는 실행의 `kg_rag.eval.strategy`와 `kg_rag.eval.repeat`(1부터)도 담으며, 실행의 모든 스팬에 같이 붙음 |
| `kg_rag.query` | 루트 또는 `run-eval` 배치 | `session.id`, `user.id`, `kg_rag.blocked`. `capture_content`이면 마스킹한 질의를 `input.value`로, (`stream` / `astream`이 아니면) 답변을 `output.value`로 기록 |
| `kg_rag.strategy` | query | 질의 전체의 `kg_rag.rerank_status` |
| `kg_rag.retrieve` | strategy | 결과 유형별 적중 수, 최대 50개의 소스 ID |
| `kg_rag.fusion` | strategy | 입력·출력 수, 해당 단계의 `kg_rag.rerank_status` |
| `kg_rag.rerank` | fusion | `kg_rag.rerank.status`(`ok`, `empty`, `error`), 문서 수 |
| `kg_rag.context_build` | query(`drift`: 해당 라운드의 actions 스팬) | 사용한 토큰과 토큰 예산 |
| `kg_rag.global.select`, `kg_rag.global.map` | strategy(`global`) | select는 질의마다 하나: `kg_rag.depth`(설정한 깊이), `kg_rag.cut`(더 얕은 단면으로 내려간 뒤 실제로 map한 단면), `kg_rag.reports`, `kg_rag.dynamic`. map: `kg_rag.reports`, `kg_rag.map_calls`, `kg_rag.report_tokens` |
| `kg_rag.drift.primer`, `kg_rag.drift.actions` | strategy(`drift`) | primer: `kg_rag.reports`, `kg_rag.answers`. 라운드마다 actions 스팬 하나(`kg_rag.depth`, `kg_rag.actions`)이며 그 라운드의 로컬 검색이 아래에 들어감 |
| `kg_rag.guardrail` | query | 검사한 질의나 답변마다 하나: `aws.bedrock.guardrail.id`, `.source`(`input`, `output`), `.action`(`none`, `masked`, `blocked`), `.intervened`, `.policies` |
| `kg_rag.evaluate` | 루트 | 평가기마다 하나: `kg_rag.evaluator`, `kg_rag.items`, `kg_rag.eval.strategy`, `kg_rag.eval.repeat`. 심사 모델의 스팬이 아래에 들어감 |
| `chat <model>`, `embeddings <model>` | 호출한 스팬 | OpenTelemetry GenAI 규약, 스팬 종류 `CLIENT`(Bedrock 재순위화 스팬도 같음): `gen_ai.request.model`, `.max_tokens`, `.temperature`, `gen_ai.usage.input_tokens` / `output_tokens`, Bedrock이 프롬프트 캐시 사용량을 알려 주면 `gen_ai.usage.cache_read.input_tokens` / `cache_creation.input_tokens`, 종료 사유, `error.type`. 비용은 기록하지 않음 |

`capture_content`이면 답변 모델의 `chat <model>` 스팬에는 가드레일이 나중에 차단하거나
마스킹한 답변도 생성된 그대로 남고, `kg_rag.guardrail` 스팬은 Bedrock 가드레일 어댑터만
만듭니다(직접 구현한 `ContentGuardPort`는 만들지 않음).

`kg_rag.rerank.status`는 Rerank 호출 한 번의 결과입니다(`error`: 질의가 융합 순서로
대체됨, `empty`: 쓸 수 있는 결과가 없음). `kg_rag.rerank_status`는 재순위화가 한 일이며
호출이 없었어도 설정됩니다. 값은 `applied`, `disabled`, `unavailable`(재순위화 모델
없음), `skipped`(재순위화할 대상 없음), `failed`이고, strategy 스팬에서는
[`metadata.rerank.status`](./user-guide.ko.md#재순위화-상태)처럼 가장 나쁜 단계의 값입니다.

내보낼 때 스팬은 `gen_ai.*`, `kg_rag.*` 속성을 그대로 두고 벤더 속성이 더해집니다.
Langfuse에는 `langfuse.observation.type`(모델 호출은 `generation`, `embedding`, 검색과
재순위화는 `retriever`, 나머지는 `span`)과 `capture_content`일 때
`langfuse.observation.input` / `output`이, Phoenix에는 `openinference.span.kind`,
`llm.model_name`, `llm.provider`, `llm.token_count.*`와 `capture_content`일 때
`input.value` / `output.value`가 붙습니다.

알려진 제한은 다음과 같습니다.

- 스트리밍 답변(`stream` / `astream`)은 별도 트레이스입니다. 그 모델 스팬은 답변 단계
  실행 메타데이터에서 `session.id` / `user.id`를 가져오므로 같은 세션으로 묶입니다.
- 비동기 배치 경로는 청크가 아니라 배치마다 스팬을 하나 기록합니다.
- `ainvoke` 도중 취소된 모델 호출(태스크 취소, 또는 RAGAS `ragas_timeout`처럼
  `asyncio.wait_for` / `asyncio.timeout`으로 끊긴 호출)은 LangChain이 콜백을 호출하지
  않으며, 그 모델 스팬은 호출을 시작한 태스크가 끝날 때 `ERROR`(`error.type:
  CancelledError`)로 종료됩니다. 호출자가 스트림을 중간에 그만 읽으면 그 스팬은 정상
  종료됩니다.
- `otlp`, `langfuse`, `phoenix`에서는 프로세스가 처음 설치한 백엔드가 OpenTelemetry
  전역 트레이서 공급자도 되는데, 전역 공급자는 한 번만 설정할 수 있습니다. 나중에 다른
  설정으로 다시 구성하면 패키지의 스팬은 새 익스포터로 가지만, 다른 라이브러리가 전역
  API로 남기는 스팬은 버려집니다.

### 전송 제한, 프라이빗 VPC, 개인정보

- **전송이 실행을 막지 않습니다.** 스팬은 백그라운드 스레드에서 내보내고, 요청 하나는
  `observability.export_timeout_seconds`(10초, `OTEL_EXPORTER_OTLP_TIMEOUT`이 우선)로
  제한되며, 큐가 가득 차면 스팬을 버리고 실패는 `ERROR`로 남깁니다. 수집기에 연결할 수
  없으면 종료 시 플러시가 남은 512개 단위 배치마다 제한 시간 한 번 정도, 최대 약 30초
  걸린 뒤 나머지를 버립니다.
- **프라이빗 VPC.** 태스크가 엔드포인트에 연결할 수 있어야 합니다. `localhost`의 수집기
  사이드카, VPC 안의 Langfuse나 Phoenix, 또는 수집기를 위한 X-Ray 인터페이스 엔드포인트를
  쓰세요. 호스팅형 Langfuse나 Phoenix는 인터넷 송신이 필요한데 `private` 모드에는 없습니다.
  이미지는 `--build-arg UV_EXTRAS="--extra otel"`로 빌드합니다.
- **자격 증명**(`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`, `PHOENIX_API_KEY`,
  `OTEL_EXPORTER_OTLP_HEADERS`)은 환경 변수에서만 읽으므로 ECS 태스크 시크릿으로
  주입하세요. 시작 로그 `Observability exporter for backend '...'`에는 엔드포인트와 헤더
  이름만 나오고 값은 나오지 않습니다.

스팬에는 개수, ID, 모델 이름, 토큰 사용량만 들어갑니다. `capture_content: true`로
설정하면 모델 스팬에 프롬프트와 응답 본문이, `kg_rag.query` 스팬에 질의와 답변이
(`redact_patterns` 적용 후) 추가되므로 트레이스 백엔드가 코퍼스 데이터를 보관해도 되는
환경에서만 켜세요. `otlp`, `langfuse`, `phoenix`에서는 내보낼 때도 모든 문자열 속성,
이벤트 속성(예외 메시지에 질의가 들어갈 수 있음), 상태 메시지에 `redact_patterns`를
적용하며 `session.id`와 `user.id`도 대상입니다. `none`에서는 호스트 파이프라인이 기록된
그대로 내보내므로 그쪽 수집기에서 가리세요. `RAGInput.user_id`는 받은 그대로 내보내므로
이메일 주소가 아닌 불투명한 ID를 넘기세요. LangSmith 추적(`LANGSMITH_TRACING=true`)도
켜져 있으면 LangChain이 이 설정과 관계없이 본문을 LangSmith로 업로드합니다.

## 자주 보는 오류

메시지는 요약한 것이며, 로그 줄에 관련 키, 파일, 단계가 나옵니다. 질의 중의 저장소, 모델,
자격 증명 오류는 빈 컨텍스트로 답하지 않고 질의를 실패시킵니다.

### 설정과 시작

| 메시지(요약) | 의미 | 조치 |
|---|---|---|
| `Missing endpoint configuration for ...` / `Configuration validation error: ...` | 실행에 필요한 엔드포인트가 비어 있거나 값의 타입이 잘못됨 | 메시지에 나온 키를 지정하거나 고침 |
| `ConfigurationError: Unknown config keys: '<path>' ...`(종료 코드 2) | 오타이거나 대신할 키 없이 없어진 키(이름이 바뀐 키는 `settings moved; replace <old> -> <new>`로 실패하고, 제거된 키 `indexing.cross_run_merge`, `aws.bedrock.guardrail.apply_to`는 `Config key ... is removed and ignored`만 남김) | 키를 고치거나 삭제함. `--allow-unknown-keys`를 주면 키마다 WARNING을 남기고 읽음 |
| 사용법 오류(종료 코드 2): `--source-directory is required for pipeline execution`, `--s3-sync needs --s3-bucket-name or aws.s3.bucket_name`, `--pipeline-id is required for --resume-from-stage`, `--enabled-stages: invalid stage names ...`, `argument --resume-from-stage: Unknown pipeline stage ...` | 필수 플래그가 없거나 단계 이름이 없음 | 플래그를 넘기거나 메시지에 나온 단계 이름을 씀([CLI 레퍼런스](./cli.md#run-ingestion)(영문)) |
| `... needs confirmation, but stdin is not a terminal` | 완료된 단계의 출력이 현재 입력 기준으로 캐시에 없음 | `--yes`(그 단계를 다시 계산)나 새 `--pipeline-id` |
| `Stage '<stage>', upstream of this run's enabled stages, has not completed` | 앞 단계가 실패했거나 출력이 없음 | 앞 단계를 먼저 실행 |
| `observability.enabled is true but OpenTelemetry is not installed` | `otel` extra가 없음 | `uv sync --extra otel`을 실행하거나 추적을 끔 |
| `Skipping N '.md' file(s) ...` / `No supported source files found in '<dir>'` | `unstructured` extra가 설치되지 않음 | 설치하거나 파일 형식을 변환 |

### 자격 증명과 모델

| 메시지(요약) | 의미 | 조치 |
|---|---|---|
| `No valid AWS credentials for Amazon Bedrock in region ...` | 자격 증명이 없거나 만료됨 | 갱신하고 `AWS_PROFILE` / `aws.profile_name` 확인 |
| `Model '<id>' is only available through a cross-region inference profile, but none resolved ...` | 온디맨드 처리량이 없는데 프로파일을 찾지 못함 | `aws.bedrock.enable_global_profile: true`를 유지하고 `bedrock:ListInferenceProfiles`를 허용하거나 다른 리전 사용 |
| `Reranking failed: ...`(질의마다, `metadata.rerank.status: failed`) / `Reranking initialization failed; queries keep the fused order ...`(`unavailable`) | Rerank 호출이 거부되었거나 리전에 모델이 없음. 또는 재순위화 모델을 만들 수 없음 | `bedrock:Rerank`를 `Resource: "*"`로 허용하거나, 모델이 있는 리전을 쓰거나, `models.rerank` / `providers.rerank`를 확인하거나, `search.reranking.enabled: false` 설정 |
| `RerankModelError: ...` | 위와 같으나 `search.reranking.required: true`임 | 원인을 고치거나 `required: false`로 융합 순서대로 답하게 함 |
| 질의 중 `ClientError: ... (ValidationException) ... InvokeModel ...` / `(ResourceNotFoundException)` / `NoCredentialsError` | 질의 임베딩 호출이 성공할 수 없음 | `models.embedding.model_id`, `aws.bedrock.region_name`, 모델 액세스와 자격 증명 확인 |
| `EmbeddingModelError: Query embedding failed: ...` | 재시도 후에도 임베딩 호출이 실패함(멈춘 Ollama 러너나 vLLM 엔진, 계속되는 `InternalServerException`) | 임베딩 서버와 로그를 확인한 뒤 다시 시도 |
| 프라이빗 VPC에서 Bedrock 호출이 멈춤 | VPC 엔드포인트와 다른 리전으로 Bedrock을 호출함 | `aws.bedrock.region_name`을 비우거나 VPC 리전으로 설정 |

### 저장소: 연결, 권한, 요청

| 메시지(요약) | 의미 | 조치 |
|---|---|---|
| `StoreConnectionError: ... Cannot connect to host ...` / `Neptune is not reachable ...` / `... status=429 ... (still failing after the client's retries)` | 연결 거부나 이름 확인 실패, Neptune이 연결을 닫음(재시작, 장애 조치. 다음에 다시 엶), 또는 OpenSearch가 `search.store_max_attempts`번 모두 429/502/503/504로 답함. 벡터 저장소에 연결할 수 없으면 `run-ingestion`은 모델 호출 전 저장소 검사에서 실패함 | 엔드포인트, 포트, 저장소 실행 여부, 보안 그룹과 VPC 엔드포인트 확인 |
| `OpenSearch did not answer a request on <index> within N s in M attempt(s)` / `Neptune did not answer within N s` | `search.store_timeout_seconds` 안에 답이 없음(수집 중 Neptune은 고정 180초, 항목 재시도) | 저장소 상태와 부하를 확인하거나 제한 시간을 늘림 |
| `OpenSearch denied the search on <index>: ... status=403 ...`(또는 `401`) | 세분화된 액세스 제어에서 역할에 `indices:data/read/search`가 없거나 자격 증명이 잘못됨 | 권한(과 `indices:admin/mappings/get`, `indices:monitor/settings/get`)을 부여하거나 자격 증명을 고침 |
| `Neptune refused the connection (HTTP 403)` | 클러스터에 대한 `neptune-db:connect`가 없거나 다른 리전으로 서명함 | 권한(과 필요한 `neptune-db:*` 데이터 작업)을 부여하고 `aws.region_name`을 클러스터 리전으로 설정 |
| `use_iam: false`에서 OpenSearch 인증 실패 | 기본 인증 자격 증명이 없음 | `UKR__AWS__OPENSEARCH__USERNAME` / `__PASSWORD` 설정(예: `.env`) |
| `OpenSearch rejected the search on <index>: ...` | 400 `query_shard_exception`: 필터 값의 타입이 잘못됨. 500 `too_many_clauses` / `too_many_nested_clauses`: `indices.query.bool.max_clause_count`가 1,024보다 작음 | 값을 고치거나 한도를 1,024 이상으로 유지 |
| `Cannot create the OpenSearch index '<index>': the cluster has no analyzer '<name>'`(`mapper_parsing_exception`) | `indexing.opensearch.language_analyzers`의 분석기가 클러스터에 없음. 보통 Nori 패키지 없는 `ko: nori` | 클러스터에 있는 분석기(`ko: cjk`, CDK 기본값)를 쓰거나 패키지를 연결한 뒤 다시 실행 |
| `The search engine refused to create an index: ...`(종료 코드 2, 모델 호출 전) | `run-ingestion`은 첫 모델 호출 전에 실행이 만들 인덱스마다(전체 구축이나 초기화는 전부, 증분 실행은 없는 것만) 프로브 인덱스 `<extraction-records 별칭>-settings-probe-<16진수 12자리>`를 만들었다가 지움. 저장소가 그중 하나를 거부함(오류의 `(a probe of '<별칭>')`이 대신한 인덱스를 밝힘): `indexing.opensearch.index_settings` 키, `vector_search` 방식이나 분석기, 또는(`maximum shards open`) 새 인덱스의 샤드를 둘 자리가 없음 | 메시지가 가리키는 설정을 고치거나 지우고, 또는 필요 없는 인덱스를 지우거나 `cluster.max_shards_per_node`를 올린 뒤 다시 실행. 중지된 실행이 남긴 프로브(그 이름의 별칭 없는 인덱스)는 네임스페이스의 다음 실행이 지움 |

### 수집 실행과 문서 상태 레지스트리

| 메시지(요약) | 의미 | 조치 |
|---|---|---|
| `Pipeline failed at stage(s): ...` | 단계가 실패해 종료 코드 1 | 단계 로그를 보고 고친 뒤 같은 `--pipeline-id`로 재개(`--force-rebuild`는 처음부터 시작) |
| `Pipeline resume failed: ...`(`PipelineResumeError`, 종료 코드 1, 모델 호출 전) | 요청한 방식으로 파이프라인 ID를 재개할 수 없음: 이번 실행이 다시 실행해야 하는 단계가 꺼져 있거나(메시지에 단계 이름이 나옴), 명시한 `--resume-from-stage`가 캐시 출력이 사라진 단계를 지나치거나, 저장된 상태를 읽지 못함 | 메시지의 단계를 켜거나 그 단계부터 재개(`--resume-from-stage <단계>`). 상태를 읽지 못하면 새 `--pipeline-id`나 `--force-rebuild` |
| `CacheSyncError` | S3 단계 캐시 다운로드나 업로드가 실패함 | 버킷 권한과 `aws.s3.encryption`을 확인한 뒤 재개 |
| `DocStatusRegistryError` | 레지스트리 테이블이 없거나, 접근할 수 없거나, 스로틀링됨 | 테이블을 만들거나 `aws.dynamodb.table_name`을 고치고 메시지에 나온 권한을 부여한 뒤 재개 |
| `Incremental indexing is enabled but no document delta was computed` | 레지스트리 비교 없이 인덱싱이 실행됨(`continue_on_error`로 로딩 실패 뒤 진행) | 레지스트리에 접근할 수 있게 한 뒤 `document_loading`부터 다시 실행 |
| `NamespaceLeaseError: Index namespace '...' is being written by ...` | 다른 실행이 쓰기 임대를 가지고 있음 | 기다림. 죽은 실행의 임대는 `indexing.lease_seconds` 안에 만료됨([동시 실행](#동시-실행)) |
| `NamespaceLeaseError: This run lost the write lease ...` | 갱신이 계속 실패해 임대가 만료됐거나 다른 실행이 가져감 | 다른 실행이 끝난 뒤 다시 실행하고 레지스트리 연결을 확인 |
| `ConfigurationError: Doc-status registry: DynamoDB table '...' has the key schema ... of an earlier release` | `doc_id`만으로 키를 잡은 테이블 | `aws.dynamodb.table_name`을 새 테이블(또는 스택의 새 테이블)로 바꾸고 `indexing.reset: true`로 한 번 실행([증분 실행](#증분-실행)) |
| `ConfigurationError: The OpenSearch index '...' records no embedding settings ...` | 이전 릴리스가 만든 인덱스 | `indexing.reset: true`로 한 번 실행 |
| `ConfigurationError: The OpenSearch index '...' holds embeddings of ...` / `... was built for other language settings ...`(종료 코드 2, 모델 호출 전) | 증분 실행의 임베딩 모델이나 차원, 번역 언어, `language_analyzers`가 현재 인덱스를 만든 설정과 다름(바뀐 문서가 없거나 삭제만 있는 실행도 해당) | `indexing.reset: true`로 한 번 실행(코퍼스 전체를 한 번 다시 임베딩하거나 번역함)하거나 이전 설정으로 되돌림 |
| `The doc-status registry lists indexed documents of this namespace whose entities the graph store does not hold`(또는 `whose text units the vector store does not hold`) | 레지스트리가 기록한 내용을 저장소가 잃음(재시작한 Gremlin Server, `down -v`, 복원한 그래프, 삭제된 인덱스, 옮긴 `vector.sqlite3`), 또는 #383 이전 빌드가 쓴 인덱스에서 제거가 끝나기 전에 삭제 문서를 되살림, 또는 #410 이전 빌드에서 문서를 빼는 reset이 인덱스를 교체한 뒤 중단됨 | `indexing.reset: true`로 다시 실행하거나 네임스페이스의 레코드를 삭제. 로컬에서는 저장소 파일 세 개를 함께 지우거나 옮긴 파일을 되돌림. 되살린 문서는 그 문서의 레지스트리 레코드 하나만 지우면 다음 실행이 다시 인덱싱함 |

### 질의

| 메시지(요약) | 의미 | 조치 |
|---|---|---|
| `No indices found for suffix '<suffix>' ...` | 그 접미사로 수집한 데이터가 없음 | 수집하거나 코퍼스의 접미사로 질의 |
| `InvalidFilterError: Unknown filter key(s) ...` / `... has no value (None)` / `... not a scalar` / `... lists N values ...` | 전략이 읽는 어떤 저장소도 그 필터 키를 선언하지 않음, 필터 값이 `None`임(단독, 목록 안, 범위 경계 모두. OpenSearch는 그런 질의를 거부함), 목록 안에 목록이나 매핑이 있음, 또는 목록 필터의 값이 `terms` 질의 하나가 받는 수(`index.max_terms_count`, 기본값 65,536)보다 많음 | 메시지에 나온 키를 씀, `None`을 빼거나 일치시킬 값을 줌, 문자열·숫자·불리언만 나열함, 목록을 줄임(또는 `indexing.opensearch.index_settings`에서 `max_terms_count`를 올리고 다시 인덱싱) |
| LightRAG `mix` / `hybrid` 결과가 없음 | 관계 인덱스가 없거나 키워드를 추출하지 못함 | 관계 인덱스가 있는지 확인. 짧은 질의는 `raw_query_fallback_max_len` 이하일 때만 원래 질의를 씀 |

### 로컬 모드

| 메시지(요약) | 의미 | 조치 |
|---|---|---|
| `... stayed locked ...`(`database is locked`) | 다른 프로세스가 SQLite 쓰기 잠금을 `search.store_timeout_seconds`보다 오래 쥠 | 저장소 파일마다 수집을 하나씩 실행하거나 제한 시간을 늘림. 로컬 디스크를 씀 |
| 로그에 `Connection refused`와 함께 `... processed N inputs but produced 0 outputs` | `providers.llm.base_url` / `providers.embedding.base_url`에서 Ollama나 OpenAI 호환 서버에 연결할 수 없음 | 서버를 띄우고(`ollama serve` 또는 `docker compose -f docker/compose.local.yaml --profile models up -d --wait`), `base_url`(`--model-base-url`)을 확인하고 모델을 내려받음 |
| `AuthenticationError` / `NotFoundError`(401 / 404), `APIConnectionError` / `APITimeoutError`, `ResponseError: model ... not found` | 키가 거부됨, 모델을 제공하지 않음, 또는 서버에 연결할 수 없음 | `providers.llm` / `providers.embedding`의 `base_url`, `api_key`, 모델 이름을 확인하고 모델을 내려받음 |
