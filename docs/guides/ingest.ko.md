# 수집

> 🇬🇧 English version: [ingest.md](./ingest.md)

수집은 문서 디렉터리를 그래프 저장소와 벡터 저장소의 지식 그래프로 바꿉니다.
`run-ingestion --source-directory <dir>`이 13개 단계([설계 §4](../design.md#4-ingestion-pipeline)(영문)에
그림이 있음)를 실행하며 단계마다 체크포인트를 남기고, 같은 `--pipeline-id`로 다시
실행하면 마지막으로 유효한 단계 다음부터 재개합니다. 문서 상태 레지스트리를 켜면
증분으로 동작합니다([증분 인덱싱](./incremental.ko.md)).

모든 플래그와 기본값은 [CLI 레퍼런스](../reference/cli.md#run-ingestion)(영문)에 있습니다.
실행이 어떤 인덱스에 쓰고 어떤 문서를 삭제할 수 있는지는 인덱스 접미사와 소스 범위로
정해집니다([범위와 네임스페이스](../concepts/scopes-and-namespaces.ko.md)). 한국어나 여러 언어가
섞인 코퍼스는 [다국어 코퍼스](./multilingual.ko.md)를 참고하세요.

<a id="파이프라인-단계"></a>

## 파이프라인 단계 13개

실행 순서대로이며, `--enabled-stages` / `--resume-from-stage`에는 단계 **이름**(대소문자
무관)을 씁니다.

1. **`document_parsing`**: 형식별 텍스트 추출(`.pdf`, `.txt`, `.csv`, `.json`,
   `.md`, `.html`. `docx` extra가 있으면 `.docx`도). [원본 형식](../troubleshooting.md#which-source-formats-are-read)(영문) 참고.
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
10. **`graph_analysis`**: 그래프를 만들고 크기, 밀도, 연결 요소를 기록.
11. **`community_detection`**: Leiden 군집화와 LLM 커뮤니티 리포트(`indexing.methodologies`에
    `graphrag`가 있을 때만).
12. **`indexing`**: OpenSearch와 Neptune에 씀(켜져 있으면 레지스트리에도).
13. **`summary_tree`**(`indexing.methodologies`에 `raptor`가 있을 때만): 저장된 텍스트
    유닛 위에 RAPTOR 요약 트리를 만들며, `tree` 전략이 이를 검색함
    ([아래](#raptor-요약-트리선택-사항)).

실행에는 [방법론 프로필](#방법론-프로필)에 필요한 단계가 들어갑니다. `--enabled-stages`로 단계를
직접 나열하면 그 목록이 우선합니다. 나열한 단계는 실행하고 나열하지 않은 단계는 실행하지 않으며,
나열하지 않은 단계 때문에 인덱스에서 빠지는 방법론이 있으면 WARNING으로 알려 줍니다.

## 예시

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

## 재개와 강제 재구축

`--force-rebuild` 없이 다시 실행하면 완료된 단계는 캐시를 쓰고 건너뜁니다.

- `--pipeline-id <id>`는 그 실행을 처음으로 실패했거나 끝나지 않은 단계부터
  재개합니다(`--enabled-stages`가 있으면 그 범위 안에서). 그 뒤에 켠 단계(글리닝,
  주장)는 끝나지 않은 단계로 봅니다. `--pipeline-id`가 없는 실행은 새 ID를 받습니다.
- 코퍼스나 설정이 바뀌면 CLI는 출력이 더는 캐시에 없는 완료 단계를 나열하고, 다시
  계산하기 전에 묻습니다(터미널이 없으면 `--yes`).
- 증분 인덱싱에서 델타를 계산한 뒤 추출 설정이 바뀐 재개는 `document_loading`부터 다시
  시작하므로 델타가 새 설정을 따릅니다.
- `--force-rebuild`는 모든 단계를 다시 계산하고 캐시를 새로 씁니다. 모든 사전 청크의
  경계, 설명 요약, 커뮤니티 리포트도 모델에 다시 묻고 모든 텍스트를 다시 임베딩합니다
  ([S3 동기화](#s3-동기화) 참고).

## S3 동기화

`--s3-sync`는 단계 캐시를 `s3://<bucket>/<prefix>/...`에 두고 단계가 끝날 때마다
업로드하므로, 새 태스크가 거기서 재개합니다. SIGTERM(ECS `StopTask`, Step Functions 시간
초과)을 받으면 실행 중인 단계의 체크포인트를 업로드하고 143으로 종료합니다. 단계 사이의
업로드 실패는 다음 단계에서 다시 시도하고, 처음 다운로드나 마지막 업로드가 실패하면
`CacheSyncError`로 실행이 실패합니다. 만료와 다른 캐시는
[캐시와 접두사](../operations.md#caches-and-prefixes)(영문)에 있습니다.

실행 사이에는 문서 상태 레지스트리(`providers.doc_status.enabled`)가 바뀌지 않은 문서를
다시 임베딩하지 않게 합니다. 증분 실행은 델타가 바꾼 것만 임베딩합니다. 실행이 쓰는 항목
중에서도 바뀌지 않은 텍스트는 저장된 벡터를 그대로 쓰고, 바뀌지 않은 청크 경계, 번역, 설명 요약,
커뮤니티 리포트는 저장된 모델 응답을 그대로 씁니다. 그래서 문서를 고치거나 바뀌지 않은
코퍼스를 reset해도 바뀐 부분에만 비용이 듭니다([설계 §4.4](../design.md#44-pipeline-infrastructure),
[§5.2](../design.md#52-full-rebuild-indexingreset), 영문).

## LLM 응답 캐시

`cache.llm_responses`(기본값 켜짐). 수집 단계의 모델 응답은 모두 실행의 캐시 디렉터리
`llm-cache/`에 두고 함께 동기화하므로, 같은 pipeline id로 다시 실행하면 중지되거나 비정상
종료된 단계가 하지 않은 호출만 다시 합니다. 키는 렌더링한 프롬프트와 모델 ID, 최대 토큰,
temperature, 추론 설정입니다.

- 이전 실행의 응답만, 실행마다 한 번까지만 씁니다. 출력 토큰 한도에서 잘린 응답은
  보관하지 않습니다. temperature가 0보다 크면 다시 실행할 때 이전 샘플을 재사용합니다.
  모든 시도가 실패한 호출(끝내 파싱되지 않은 응답 등)이나 그래프·주장 레코드를 만들지 못한 호출의 응답은 버리므로,
  다시 실행하면 그 단위만 모델에 다시 묻습니다.
  시간 제한을 넘겨 버려진 호출이 늦게 받은 응답도 보관하지 않습니다.
- `--force-rebuild`는 캐시에서 답하지 않으며, 질의와 평가 호출은 캐시하지 않습니다.
- 캐시는 모델 호출마다 수 KB씩 커지고 실행을 시작할 때 메모리로 읽습니다.
  `cache.llm_responses.flush_every`개 응답 단위의 세그먼트로 쓰므로, 예고 없이 종료된
  태스크가 잃는 것은 세그먼트 하나 이하입니다. 중지하면 실행 중인 모델 호출의 재시도도
  끝납니다.

## 대규모, 이질적인 코퍼스

- **대규모:** Bedrock 할당량이 허용하는 만큼 `processing.max_concurrency`를 올리고,
  `indexing.neptune.index_concurrency`는 효과를 측정하기 전까지 `1`로 두세요(동시 쓰기
  배치는 공유 정점에서 충돌해 재시도함). 문서 상태
  레지스트리(`providers.doc_status.enabled`)를 켜면 다음 실행부터 바뀐 문서에만 비용이
  듭니다. Neptune에서 `indexing.neptune.vertex_ids: custom`을 쓰면 정점을 `id` 속성
  대신 정점 id로 찾습니다. 기존 그래프를 전환하려면 전체 빌드를 한 번 실행해야
  합니다([운영 런북: 사용자 지정 Neptune 정점 id](../operations.md#custom-neptune-vertex-ids), 영문).
  그래프 확장은 허브의 이웃을 id 순서로 가져오므로(`indexing.neptune.deterministic_order:
  true`, 기본값) 다시 빌드해도 결과가 같습니다. 대신 확장하는 허브마다 이웃을 모두
  읽습니다(Gremlin Server에서 이웃 2,500개 허브는 질의당 6.8 ms에서 23.2 ms로 늘었고,
  Neptune에서는 측정하지 않음). 허브가 많은 질의가 너무 느리면 `false`로 두어 Neptune이
  먼저 돌려주는 이웃을 쓰세요.
- **커뮤니티 리포트:** 리포트 프롬프트는 커뮤니티의 관계를 하나씩, 양 끝 엔티티(주장 추출을
  켰으면 그 주장까지)와 함께
  `graph.community_detection.report_generation.max_report_context_tokens`(4000)에
  채웁니다. MS GraphRAG는 8000과 `max_cluster_size` 10을 씁니다. 큰 커뮤니티의 리포트가
  관계를 너무 적게 다루면 예산을 올리거나(리포트 호출당 입력 토큰 증가)
  `max_cluster_size`를 낮추세요(더 작은 커뮤니티가 많아져 리포트 호출도 늘어남).
- **다국어:** [다국어 코퍼스](./multilingual.ko.md)를 참고하세요.
- **이질적인 도메인:** `entity_types`를 여러 도메인의 합집합으로 맞추거나, 도메인마다
  [인덱스 접미사](../concepts/scopes-and-namespaces.ko.md#인덱스-접미사)를 따로 두세요.

## 방법론 프로필

`indexing.methodologies`(기본값 `[graphrag, lightrag]`)가 실행에서 만들 것을 정합니다
([프로필](../concepts/overview.ko.md#방법론-프로필)).

```yaml
indexing:
  methodologies: [lightrag]   # mix, hybrid, naive, ppr, iterative: 커뮤니티도 리포트도 만들지 않음
```

- **건너뛰는 것.** `graphrag`가 없으면 Leiden 단계와 리포트 호출을 하지 않습니다. 나머지
  (추출, 글리닝, 요약, 관계 벡터)는 LightRAG가 읽는 것입니다. 이 설정이 생기기 전의 설정에서
  `graph.community_detection.enabled: false`이면 같은 프로필을 만들고,
  `indexing.opensearch.build_relationship_vector_index: false`이면 `graphrag`만 만듭니다.
  모르는 이름은 올바른 이름을 알려 주며 설정 로드에 실패하고, 이 스위치들 때문에 남는 방법론이
  없는 선택도 마찬가지입니다.
- **리포트 없이.** `graph.community_detection.report_generation.enabled: false`이면 Leiden
  단계는 실행하고 리포트 호출만 건너뜁니다. 네임스페이스는 `local`이 읽는 `graphrag`를 가지며,
  리포트로 답하는 `global`과 `drift`는 이 스위치를 알려 주며 거절하고(설정 로드 때 WARNING으로
  알림) `auto`도 고르지 않습니다.
- **네임스페이스가 가진 것.** 실행마다 빌드 기록(`artifacts`, `methodologies`)에 남깁니다.
  질의한 전략이 네임스페이스에 없는 방법론을 읽으면 모델을 호출하기 전에 거절하고
  (`MethodologyNotIndexedError`, `run-rag`와 `run-eval`은 종료 코드 2), 네임스페이스가 가진 것을 알려 줍니다.
- **나중에 추가하면**(예: `[lightrag]`로 만든 뒤 `graphrag`) 그 방법론이 더하는 것만 만듭니다.
  문서 상태 레지스트리를 켰다면 다음 실행이 커뮤니티 없는 네임스페이스를 보고 한 번 다시
  구축하며(`indexing.reset`, WARNING으로 알림), 저장된 청크 경계, 추출 결과, 설명 요약, 벡터를
  재사용하므로 새 모델 호출은 커뮤니티 리포트뿐입니다. 네임스페이스에 다른 소스 범위의 문서도
  있으면 대신 거절합니다(종료 코드 2). 한 코퍼스로 다시 구축하면 그 문서들이 지워지기 때문입니다.
  레지스트리가 없으면 모든 실행이 전체 구축이고 같은 것을 재사용합니다. 같은 `--pipeline-id`로
  다시 실행하면 단계 캐시에서 `community_detection`과 `indexing`만 실행합니다.
- **빼면** 그 산출물은 더 이상 갱신되지 않습니다. 네임스페이스의 다음 실행이(코퍼스가 바뀌지
  않은 실행도) 빠졌다고 기록하고, 그 뒤로는 해당 전략이 거절됩니다. 아무것도 지우지 않습니다. 지우려면 남길 방법론으로 `indexing.reset: true`를 한 번
  실행하세요(추출 레코드는 남으므로 나중에 그 방법론을 다시 넣어도 다시 추출하지 않습니다). 재구축이 커뮤니티 리포트 인덱스를 빈 인덱스로 바꾸고 그래프의 커뮤니티 정점을
  지우며, `raptor`가 없으면 요약 트리의 인덱스(교체 전에 중단된 구축의 인덱스 포함)를 지웁니다.
  레지스트리가 없으면 모든 실행이 전체 구축이므로 다음 실행이 지웁니다.

## RAPTOR 요약 트리(선택 사항)

`raptor`(기본값 아님)는 인덱싱 뒤에 요약 트리를 만듭니다. 저장된 텍스트 유닛을 벡터로
군집화하고 군집마다 요약(`models.roles.summary_tree`, default 티어)한 뒤, 요약을 다시
군집화하고 요약하는 일을 `indexing.summary_tree.max_levels`(5) 레벨까지 반복해 `tree`
전략이 쓰는 별도 인덱스에 씁니다([검색](../concepts/retrieval.ko.md#검색-전략-선택)).
선택 사항입니다. 멀티홉 질문에서는 `naive`보다 나았지만 그래프 전략보다 크게 낮았고, 한 코퍼스에서
1회 반복한 결과입니다([근거](../concepts/defaults-and-evidence.md#raptor-summary-tree), 영문).

```yaml
indexing:
  methodologies: [raptor]                     # 텍스트 유닛과 트리만
  # methodologies: [graphrag, lightrag, raptor]  # 그래프와 함께 트리도
  summary_tree:
    algorithm: kmeans          # 또는 gmm(RAPTOR의 가우스 혼합, 더 느림)
    partition_by: [category]   # 선택: 필터 값마다 트리 하나
```

- **`raptor`만 고르면 실행되는 단계.** 파싱, 로딩, 청킹, 번역(켠 경우), 인덱싱과 트리뿐이며
  그래프 추출, 글리닝, 병합, 주장, 커뮤니티는 실행하지 않습니다. 그래서 `raptor`만 가진
  네임스페이스는 요약 비용만 냅니다. 텍스트 유닛 5-6개에 호출 한 번 정도이므로 코퍼스 토큰당
  비용은 청크가 작을수록, 추론 수준(effort)이 높을수록 커집니다. MuSiQue 벤치마크 인덱스에서
  fast 티어(`models.roles.summary_tree: fast`)로 만든 트리는 1달러에 한참 못 미쳤고, 같은
  인덱스의 그래프 수집은 수백 달러였습니다([근거](../concepts/defaults-and-evidence.md#raptor-summary-tree),
  default 티어 추정치: [설계 §6.5](../design.md#65-raptor-summary-tree-raptor-tree-opt-in), 영문). 이 네임스페이스는
  `tree`, `naive`, `simple`이 서비스하며 `auto`는 `tree`로 갑니다. 그래프 저장소에 쓰지 않으므로 Neptune 엔드포인트가
  필요 없습니다. 네임스페이스에 있던 그래프는 증분 실행에서는 그대로 두고, 전체 구축
  (`indexing.reset: true`, 또는 레지스트리 없는 모든 실행)은 더 만들지 않는 벡터 인덱스를 비우듯
  그래프 엔드포인트가 설정되어 있으면 그래프를 지웁니다. 엔드포인트가 없으면 그래프가 남으므로,
  지우려면 엔드포인트를 설정하고 그 재설정을 한 번 실행하세요.
- **나중에 `raptor` 추가.** 다음 실행에서(코퍼스가 그대로여도) 또는 `run-ingestion
  --enabled-stages summary_tree`로, 저장된 텍스트 유닛에서 트리만 만듭니다. 다시 추출하거나
  네임스페이스를 재구축하지 않습니다. 이 단계는 저장소만 읽으므로, 업그레이드로 앞 단계의 캐시
  키가 바뀌었어도 네임스페이스를 만든 파이프라인 id로 `--enabled-stages summary_tree`를 실행할
  수 있고 앞 단계는 실행하지 않습니다. 어느 쪽이든 그 실행의 `indexing.methodologies`에 `raptor`가
  있어야 합니다. `raptor` 없이 `--enabled-stages summary_tree`로 실행하면 트리는 쓰지만 빌드 기록에
  남기지 않으므로 `tree`는 계속 거절됩니다.
- **증분 실행.** 저장된 텍스트 유닛으로 트리를 다시 만들어 교체합니다. 자식이 그대로인 군집은
  저장된 요약을 그대로 쓰고(모델 호출 없음), 군집화는 작은 델타로는 거의 바뀌지 않는 노드 id
  표본에서 시작하므로 대부분의 군집이 살아남습니다. 삭제된 문서의 요약과 그 위의 모든 요약은
  트리를 다시 만들기 전에 텍스트 유닛과 함께 지우므로, 실행이 중간에 멈춰도 삭제된 내용을
  요약한 노드가 제공되지 않습니다. `refresh: on_reset`이면 재설정할 때까지 트리를 그대로
  둡니다(삭제 시 요약 삭제는 그대로 함). 다만 요약 일부가 실패한 트리는 다음 실행에서 다시
  만듭니다.
- **필터.** 요약은 모든 문서가 같은 값을 가진 문서 필터 키만 가지므로, 필터를 걸면 모든
  문서가 통과하는 요약만 남습니다. `partition_by`를 쓰면 그 키로 트리를 나눠 각 트리가
  그 키에 대해 섞이지 않게 합니다.
- **언어.** 요약은 인덱스 언어(번역을 켜면 대상 언어, 끄면 원문 언어)로 쓰고, 그 언어의
  분석기로 색인합니다.
- **한도.** 트리 하나에는 텍스트 유닛을 최대 `max_leaves_in_memory`(100,000)개까지 담습니다.
  텍스트와 벡터를 메모리에 올려 군집화하므로 최대 사용 시점에 텍스트 유닛 하나에 태스크 메모리가
  약 50 KiB(1,024차원) 들고, 기본값이면 태스크 8 GiB 중 약 5 GiB를 씁니다([용량
  산정](../operations.md#capacity-sizing), 영문). 더 큰 코퍼스는 `partition_by`나 `scope: document`
  (문서마다 트리 하나, RAPTOR 원래 설정이며 텍스트 유닛이 3개 미만인 문서는 트리가 없음)로
  나누거나 `fargate_memory`와 함께 한도를 올립니다.

## 비용 조절 수단

비용은 대부분 LLM 호출에서 나옵니다. 추출(청크당 1회 이상), 글리닝(청크당 최대
`max_rounds`회), 커뮤니티 리포트, 주장(텍스트 단위당 1회, 기본은 꺼짐), 질의마다의 답변
생성입니다. `graphrag`를 뺀 [방법론 프로필](#방법론-프로필)(커뮤니티 리포트 없음), 기계적인
역할에 Haiku급 모델 쓰기(청킹, 번역, map-reduce, 요약의 기본값), 글리닝 횟수 줄이기, 수집 추론
수준 낮추기, 주장 끄기, 캐시와 증분 인덱싱으로 줄일 수 있습니다.

이 중 가장 큰 것은 글리닝입니다(측정한 수집에서 차지한 비중:
[근거](../concepts/defaults-and-evidence.md#gleaning-rounds), 영문). `processing.gleaning.max_rounds`는
3이며, 글리닝이 더하는 관계는 대부분 1회차에서 나오고 2~3회차는 조금만 더합니다. 1회차만 쓴
그래프와 비교한 답변에서도 어느 쪽이 낫다고 할 수 없었습니다. `max_rounds: 1`(MS GraphRAG 기본값)로
두면 그 효과의 대부분을 글리닝 비용의 약 3분의 1로 얻습니다.

`aws.bedrock.ingestion_effort: low`는 default 티어 수집 호출의 추론 수준만 낮춥니다(질의는
`default_effort`를 그대로 씀). 측정한 일부 코퍼스에서 수집 비용이 약 5분의 1 줄었고 답변에는
측정할 만한 차이가 없었습니다([근거](../concepts/defaults-and-evidence.md#ingestion-effort), 영문).
