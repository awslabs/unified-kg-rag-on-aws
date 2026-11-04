# Unified Knowledge Graph RAG on AWS — 사용자 가이드

> 🇬🇧 English version: [docs/user-guide.md](./user-guide.md)

한국어 문서는 README, 문서 안내, 개요, 시작하기 두 문서, 설정 가이드와 이 사용자 가이드입니다. 설계 문서, 운영 런북, 모델 카탈로그, CLI·설정 레퍼런스와 그 밖의 가이드는 영문으로만 제공합니다.

이 문서는 **unified-kg-rag-on-aws**를 실제로 사용하는 방법을 설명하는 가이드입니다.
unified-kg-rag-on-aws는 대규모 다국어 문서 코퍼스로 지식 그래프를 만들고 그 그래프를
근거로 질문에 답하는 AWS 네이티브 지식 그래프 RAG 프레임워크입니다. 두 가지 검색
방법론인 **Microsoft GraphRAG**(커뮤니티 요약)와 **LightRAG**(이중 레벨 키워드)를
하나의 스택 위에 재구현했으며, 질의마다 방법론을 고를 수 있습니다.

- 프로젝트 소개와 1분 빠른 시작은 [README.ko.md](../README.ko.md)를 참고하세요.
- 모든 문서의 지도는 [문서 안내](./index.ko.md), 개념과 원 논문 대비 확장한 부분은
  [개요](./concepts/overview.ko.md)를 참고하세요.
- 내부 구조와 아키텍처(헥사고날 레이어, 포트와 어댑터, 의존성 규칙)는
  [설계 문서](./design.md)(영문)를 참고하세요.

콘솔 진입점 여섯 개(`pyproject` 스크립트로 정의)는 모두 `application.cli.launch`로
시작해 아래 모듈을 불러옵니다.

| 스크립트 | 모듈 | 용도 |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | 지식 그래프 구축과 갱신 |
| `run-rag` | `application.cli.run_rag_chain` | 그래프 질의 |
| `run-eval` | `application.cli.run_evaluation` | 검색과 생성 평가 |
| `run-visualization` | `application.cli.run_visualization` | 내보낸 그래프 렌더링(수집 없음) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | 도메인에 맞춘 프롬프트 생성 |
| `prepare-benchmark` | `application.cli.prepare_benchmark` | 공개 벤치마크(MuSiQue)를 코퍼스와 `run-eval` 데이터셋으로 변환 |

각 CLI의 `--help` 끝에는 실행 예시와 이 문서의 해당 절 링크가 있으며, 모든 플래그는 [CLI 레퍼런스](./reference/cli.md)(영문)에 있습니다.

## 목차

1. 사전 요구사항과 설치: [AWS에서 시작하기](./getting-started/aws.ko.md)([필수 모델](./getting-started/aws.ko.md#필수-모델) 포함), [로컬에서 시작하기](./getting-started/local.ko.md), [Local mode](./guides/local-mode.md)(영문)로 옮김
2. 설정: [설정](./guides/configure.ko.md)으로 옮김(모든 키: [설정 레퍼런스](./reference/configuration.md))
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

관련 문서: [CLI 레퍼런스](./reference/cli.md)(영문) · [설정 레퍼런스](./reference/configuration.md)(영문) · [모델 카탈로그](./reference/models.md)(영문) · [운영 런북](./operations.md)(영문) ·
[설계 문서](./design.md)(영문) · [`iac/README.md`](../iac/README.md)

---

## 3. 수집 (`run-ingestion`)

수집은 문서 디렉터리를 그래프 저장소와 벡터 저장소의 지식 그래프로 바꿉니다.
`run-ingestion --source-directory <dir>`이 12개 단계([설계 §4](./design.md#4-ingestion-pipeline)에
그림이 있음)를 실행하며 단계마다 체크포인트를 남기고, 같은 `--pipeline-id`로 다시
실행하면 마지막으로 유효한 단계 다음부터 재개합니다. 문서 상태 레지스트리를 켜면
증분으로 동작합니다([§5](#5-증분-인덱싱)).

모든 플래그와 기본값은 [CLI 레퍼런스](./reference/cli.md#run-ingestion)(영문)에 있습니다.

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
- `--force-rebuild`는 모든 단계를 다시 계산하고 캐시를 새로 씁니다. 모든 사전 청크의
  경계, 설명 요약, 커뮤니티 리포트도 모델에 다시 묻고 모든 텍스트를 다시 임베딩합니다(아래
  참고).

**S3 동기화**(`--s3-sync`)는 단계 캐시를 `s3://<bucket>/<prefix>/...`에 두고 단계가
끝날 때마다 업로드하므로, 새 태스크가 거기서 재개합니다. SIGTERM(ECS `StopTask`, Step
Functions 시간 초과)을 받으면 실행 중인 단계의 체크포인트를 업로드하고 143으로
종료합니다. 단계 사이의 업로드 실패는 다음 단계에서 다시 시도하고, 처음 다운로드나
마지막 업로드가 실패하면 `CacheSyncError`로 실행이 실패합니다. 만료와 다른 캐시는
[캐시와 접두사](./operations.md#caches-and-prefixes)에 있습니다. 실행 사이에는 문서 상태
레지스트리(`providers.doc_status.enabled`)가 바뀌지 않은 문서를 다시 임베딩하지 않게
합니다. 증분 실행은 델타가 바꾼 것만 임베딩합니다. 실행이 쓰는 항목 중에서도 같은
임베딩 제공자·모델·차원으로 이미 임베딩한 텍스트는 저장된 벡터를 그대로 씁니다(벡터마다
텍스트의 다이제스트를 함께 저장). 그래서 문서를 고치면 텍스트가 바뀐 청크, 설명,
리포트만 다시 임베딩하고, 바뀌지 않은 코퍼스를 reset하면 다시 생성한 것만 임베딩합니다.
지능형 청커의 경계 응답은 사전 청크(pre-chunk)마다 추출 저장소에 남으므로, reset이나
문서 수정은 본문이나 청킹 설정이 바뀐 사전 청크만 모델에 묻고, 매번 다르게 답하는
모델이라도 바뀌지 않은 사전 청크의 경계는 그대로입니다. 청커가 실제로 쓴 응답만 남기므로,
응답이 없는 줄 번호를 가리켜 크기 기준으로 나눈 사전 청크는 다음 실행에서 모델에 다시
묻습니다. 전체 빌드(reset, 첫 실행, 레지스트리 없는 실행)는 코퍼스 전체에 걸친 응답도
같은 방식으로 다시 씁니다. 요약 임계값을 넘는 설명은 저장된 사본이 같은 텍스트와 같은 요약
설정으로 만든 요약이면 그 요약을 그대로 쓰고, 리포트 입력(묶어 넣은 구성원, 설명, 관계,
주장, 하위 리포트 텍스트와 이름)과 리포트 설정이 바뀌지 않은 커뮤니티는 저장된 응답을
씁니다(추출 저장소에 두고, 리포트의 `attributes.report_record`가 그 키를 가리킵니다).
커뮤니티 이름(`Level 0 Community 1a2b3c4d`)은 구성원에서 만들어지므로, 문서가 커뮤니티를
하나 더해도 다른 커뮤니티의 이름과 저장된 응답은 그대로입니다.

**LLM 응답 캐시**(`cache.llm_responses`, 기본값 켜짐). 수집 단계의 모델 응답은 모두 실행의
캐시 디렉터리 `llm-cache/`에 두고 함께 동기화하므로, 같은 pipeline id로 다시 실행하면
중지되거나 비정상 종료된 단계가 하지 않은 호출만 다시 합니다. 키는 렌더링한 프롬프트와
모델 ID, 최대 토큰, temperature, 추론 설정입니다.

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

### 인덱스 접미사

코퍼스의 인덱스와 그래프 레이블 이름은 모두 `<prefix>-<suffix>`이며
`indexing.additional_suffix`를 지정하면 `-<additional_suffix>`가 붙습니다. 이
`<suffix>`가 **인덱스 접미사**입니다. 수집은
`processing.document_parsing.index_value`에서 가져오고(플래그 없음), `run-rag`와
`run-eval`은 `--suffix`로 다른 값을 지정하지 않으면 설정 파일의 같은 `index_value`를
질의합니다. Python에서는 `RAGInput.suffix`로 고릅니다(체인은 `index_value`를 읽지
않음). 모두 기본값은 `default`입니다.
테넌트나 코퍼스 버전마다 인덱스 접미사와 설정 파일을 하나씩 두고,
`indexing.additional_suffix`는 양쪽에서 같게 하세요. 증분 인덱싱과 삭제는 인덱스 접미사
단위입니다(§5).

접미사는 이름을 나눌 뿐 접근을 막지 않습니다. 질의할 수 있는 호출자는 어떤 접미사든
지정할 수 있으므로, 멀티 테넌트 서비스는 인증된 호출자를 기준으로 서버에서 접미사를
정해야 합니다. [운영 런북: 질의 권한과 테넌트 격리](./operations.md#query-access-and-tenant-isolation)를
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
- **청크 크기.** 청크 크기는 기본값으로 문자 수를 셉니다. 한글·한자·가나 한 글자는
  토큰 하나쯤이므로 8,000자짜리 한국어 청크는 약 6,000토큰이 되어 Cohere Rerank 3.5의
  문서당 한도 4,096토큰을 넘고, 넘은 뒷부분은 재순위화(rerank)되지 않습니다. 크기를
  절반으로 줄이거나, `processing.chunking.length_unit: tokens`로 토큰 수를 세게 하고
  모든 크기를 토큰 단위로 지정하세요. 토큰 수는 로컬 추정치이며 모델을 호출하지 않습니다.

  ```yaml
  processing:
    chunking:
      length_unit: tokens
      min_chunk_size: 250
      max_chunk_size: 2000
      chunk_overlap: 125
      fallback_chunk_size: 1200
      pre_chunk_size: 12500
      pre_chunk_overlap: 125
  ```

  단위를 바꾸면 코퍼스를 다시 청크로 나누고 문서를 다시 추출합니다(증분 실행은 범위의
  모든 문서를 다시 추출합니다).
- **분석기는 클러스터에 있어야** 하며, 없으면 인덱싱 단계가
  `the cluster has no analyzer '<name>'`로 실패합니다. `nori`는 `analysis-nori`
  플러그인에서 오는데, [로컬 OpenSearch 컨테이너](./guides/local-mode.md#local-stores-in-containers)(영문)에는 있고
  Amazon OpenSearch Service 도메인은 Nori 패키지를 연결해야 갖습니다(블루/그린 배포). 그래서
  [CDK 배포](../iac/README.md)는 `ko`를 기본 제공 `cjk` 분석기(문자 바이그램)에
  매핑합니다. `-c opensearch_nori_package=true`로 배포하면 패키지를 연결하고 `ko`를
  `nori`에 매핑합니다([Nori로 한국어 분석하기](./operations.md#korean-with-nori)).
- **실패.** 번역에 실패한 텍스트 단위는 원문 언어로 추출합니다. 대상 언어 하나의 호출
  전체가 실패하면 `processing.ignore_errors: true`가 아닌 한 단계가 실패합니다. 증분
  실행은 `failed_units`의 문서를 실패로 기록하므로 다음 실행이 다시 시도합니다.

---

## 4. 질의 (`run-rag`)

`run-rag --query "..."`는 질문 하나에 답하고, `--interactive`는 대화를 이어 갑니다.
전략은 `--search-strategy`로 고르며(기본값 `auto`, LLM 라우터), 모든 답변은 답변 모델이
실제로 본 출처를 함께 돌려줍니다. 전략별 검색 방식은
[설계 §6](./design.md#6-retrieval-two-methodologies)의 그림에 있습니다.

모든 플래그와 기본값은 [CLI 레퍼런스](./reference/cli.md#run-rag)(영문)에 있습니다.

### 검색 전략 선택

GraphRAG 전략은 코퍼스 전체를 *요약하고 주제별로 종합*하는 데 강하고, LightRAG 전략은
더 빠르며 *이중 레벨 키워드* 검색을 중심으로 동작하므로 키워드 중심 조회와 저비용
기준선에 적합합니다. 두 방법론 모두 같은 하이브리드 스코어러(BM25, 벡터, 그래프, RRF,
재순위화)를 씁니다.

**GraphRAG(커뮤니티 요약):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `simple` | 빠른 사실 조회 | 그래프 순회 없이 OpenSearch 벡터·키워드로 검색하며, 주장이 켜져 있으면 주장 인덱스도 포함합니다. 가장 빠릅니다. |
| `local` | 특정 엔티티에 관한 질문 | 질의 엔티티 → 그래프 순회로 이웃과 관계를 찾아 벡터·키워드 결과와 결합합니다. 켜져 있으면 주장도 넣습니다. 질의 엔티티마다 따로 찾아, 이름이 같은 엔티티가 순회를 시작하고, 없으면 최상위 결과가 가장 좋은 결과 점수의 `search.local_search.seed_score_ratio`(0.6)배 이상일 때만 시작점이 됩니다. 하이브리드 검색에서는 이 비율을 가중치에 맞춰 환산하므로, 벡터 검색만 찾은 결과는 가중치와 상관없이 이름이 일치한 엔티티와 함께 시작점이 되지 않습니다. 벡터 검색은 원래 유사도 점수끼리 차이가 작아 이 필터가 덜 걸러 냅니다. 커뮤니티 보고서는 기본값으로 질의 엔티티와 가장 비슷한 것을 넣습니다. `search.local_search.report_selection: membership`이면 순회한 엔티티가 속한 커뮤니티의 보고서를 넣으며(MS GraphRAG local 방식), 보통 그 수가 더 적습니다. 측정에서 답변이 유의하게 나빠졌으므로 기본값은 similarity입니다([근거](./concepts/defaults-and-evidence.md#local-report-selection), 영문). |
| `global` | "주요 주제는 무엇인가" 같은 넓은 주제 질문 | 계층 한 단면(`search.global_search.community_depth`, 기본값 2. 리포트가 `max_reports_per_level`보다 많은 단면은 들어맞는 가장 깊은 얕은 단면으로 내려가고, 리포트 키 필터가 있으면 가장 비슷한 리포트를 고름)의 커뮤니티 리포트를 fast 모델로 점수 매긴 핵심 내용으로 map하고(리포트 `map_context_tokens`마다 호출 1회), 순위를 매긴 핵심 내용으로 답합니다. 동적 선택은 대신 루트부터 내려갑니다. |
| `drift` | 큰 그림과 구체적 사실이 함께 필요한 여러 부분으로 된 질문 | 가상 답변(HyDE)과 비슷한 리포트로 중간 답변과 후속 질의를 만들고, 후속 질의마다 local search를 실행해 새 후속 질의를 얻는 과정을 `n_depth` 라운드 반복한 뒤 답변 모델이 종합합니다. LLM 호출이 가장 많습니다(약 `primer_folds + k_followups * n_depth`회). |
| `auto` | 일반 용도(기본값) | LLM 라우터(`models.roles.strategy_selection`)가 `search.auto_routable_strategies`(기본 local, mix, global. 질의마다 LLM 호출이 최대 27번인 drift는 추가했을 때만) 중에서 고릅니다. 후보 전략이 읽는 엔티티·키워드 추출과 질의 번역을 라우터와 동시에 실행하므로(`search.auto_speculative_extraction`), local이나 mix로 분기한 `auto`는 그 전략을 직접 고른 질의와 걸리는 시간이 같습니다. 분기한 전략이 읽지 않는 추출도 백그라운드에서 계속 실행되고 과금됩니다(기본 설정에서 질의마다 fast 모델 호출 1회, global로 분기하면 2회). 질의는 이 추출을 기다리지 않고 반환하며, 시작 시점부터 질의 사용량에 넣습니다(실행 중이면 `estimated_calls`). 추출 결과는 추출 캐시를 채우고, `search.query_timeout_seconds`가 지나거나 체인을 닫으면 취소합니다. |

**LightRAG(이중 레벨 키워드):**

| 전략 | 사용 시점 | 동작 방식 |
|---|---|---|
| `mix` | 일반적인 LightRAG 사용 | 저수준 키워드 → 엔티티 인덱스, 고수준 키워드 → 관계 인덱스, 1홉 연결 관계·끝점 엔티티 확장(Neptune 다중 홉은 `search.lightrag_search.enable_graph_expansion`으로 선택)에 **더해** naive 청크 검색 결과를 섞습니다. |
| `hybrid` | 키워드 중심의 그래프 질문 | naive 청크 결과를 섞지 않는 `mix`입니다. |
| `naive` | 빠른 기준선, 비교 평가 | 청크 검색만 합니다(`search.lightrag_search.chunk_stream_top_k`, 기본 20개). 기본값 `--search-type hybrid`는 BM25를 더하며, `--search-type vector`는 upstream LightRAG의 순수 벡터 기준선과 MS GraphRAG의 `basic` 검색에 가깝습니다. |

`mix`와 `hybrid`는 수집이 만드는 관계 인덱스를 읽습니다. 키워드가 나오지 않는 짧은
질의는 원래 질의를 저수준 키워드로 씁니다(`search.lightrag_search.raw_query_fallback_max_len`
이하).

두 전략은 찾은 엔티티가 인용하는 원문 청크와, 별도 스트림으로 찾은 관계가 인용하는
청크를 더합니다(각각 최대 `search.lightrag_search.chunk_stream_top_k`개. 결과 융합(fusion)이
두 스트림을 번갈아 섞고, 합친 청크는 재순위화(rerank) 뒤에 이 개수로 자릅니다).
`search.lightrag_search.kg_chunk_pick_method`는 항목마다 어떤 청크를 고를지 정합니다.
`weight`(기본값)는 찾은 항목마다 감소하는 비율로 `related_chunk_number`개씩 배정하고,
`vector`는 upstream LightRAG의 기본값처럼 인용된 청크 중 질의와 가장 잘 맞는 청크를
엔티티 쪽과 관계 쪽에서 한 번씩 검색해 고릅니다. 기본값은 `weight` 그대로이고 MMR의
`search.fusion.diversity_lambda`도 0.5를 유지합니다. 다른 값은 측정에서 유의한 차이가
없었습니다([근거](./concepts/defaults-and-evidence.md#lightrag-chunk-picking-and-mmr-diversity), 영문).

### 답변 형식

`global`은 기본으로 포괄적(comprehensive) 답변 프롬프트를, 나머지 전략은 간결한 프롬프트를
씁니다. 간결한 프롬프트는 한 문장으로 바로 답한 뒤 근거를 짧게 붙입니다
(`custom_prompts.answer_generation_*`). 포괄적 프롬프트는 MS GraphRAG의 global reduce
프롬프트를 본뜬 것으로, 핵심 내용을 모두 설명하는 여러 문단의 마크다운 답변을
씁니다(`custom_prompts.comprehensive_answer_*`). `search.answer_prompt_style`은 이 기본값
위에 합쳐지므로 한 전략만 지정해도 나머지는 그대로입니다.

```yaml
search:
  answer_prompt_style: {drift: comprehensive}   # global stays comprehensive
  query_timeout_seconds: 300                    # see below
# answer_prompt_style: {global: concise}        # concise global answers
```

근거. 포괄적 global 답변은 간결한 답변보다 더 많은 내용을 다루지만 덜 직접적이고, 맞비교
LLM 판정은 긴 답변을 선호합니다. 짧은 답을 원하면 `{global: concise}`를 유지하세요. `drift`의
기본값이 간결한 형식인 이유는 포괄적 답변을 쓰면 지연 시간이 기본
`search.query_timeout_seconds` 180초에 가까워지기 때문입니다. `{drift: comprehensive}`를
쓰면 이 마감 시간을 늘리세요(300초면 여유가 있습니다.
[운영 런북](./operations.md#query-deadlines-and-model-timeouts)). 측정 결과는
[기본값과 근거](./concepts/defaults-and-evidence.md#answer-style)(영문)에 있습니다.

`auto`는 분기한 전략의 형식을 따릅니다. `answer_match` 점수는 제목이 아닌 첫 줄을 읽으므로(`evaluation.answer_span`)
포괄적 답변은 품질과 관계없이 정확 일치(exact match) 점수가 낮게 나옵니다.

global search는 순위를 매긴 핵심 내용 뒤에 그 내용이 인용한 리포트를 붙여 답변 모델에
넘깁니다. `search.global_search.evidence_tokens`는 이 리포트 본문의 토큰 수를 제한합니다
(가장 점수가 높은 핵심 내용의 리포트부터 리포트 단위로). 기본값은 4000이며, 제한이 없을
때보다 global 질의 비용이 약 40% 줄었고 측정된 품질 차이는 없었으며 답변은 출처를
유지합니다([근거](./concepts/defaults-and-evidence.md#global-evidence-cap), 영문).
`null`(제한 없음)은 컨텍스트 예산이 허용하는 만큼 넣으며, 단면이 크면 global 질의 입력
토큰의 대부분을 차지합니다. `0`(비용이 더 적음)은 MS GraphRAG의 reduce
단계처럼 핵심 내용만 보내며, 이때 답변은 출처를 보고하지 않습니다(핵심
내용은 모델 출력이기 때문입니다). 그래서 `retrieval` 평가기는 global 질의의 순위 지표를 0점
처리하지 않고 건너뜁니다(`no_sources_by_config`).

### 컨텍스트 예산

`search.token_manager.max_context_tokens`(`30000`, upstream LightRAG의 예산)는 답변
프롬프트가 받는 검색 컨텍스트입니다. `models.roles.answer_generation`가 출력 예약분과
함께 받을 수 있는 크기로 줄이며 줄일 때 경고를 남기고, `null`이면 그 모델의 컨텍스트
창에서 도출합니다. 유형별 비율과 `local` 슬롯 할당은
[레퍼런스](./reference/configuration.md#searchtoken_manager)(영문)에 있습니다.

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

### 질의 제한 시간

질의 하나(`run-rag`, `invoke`/`ainvoke`, `search`/`asearch`, 스트리밍)는 전체
`search.query_timeout_seconds`(기본 180초) 안에 끝나야 합니다. 이 시간을 넘기면
`QueryTimeoutError`(`unified_kg_rag`에서 내보냄, 내장 `TimeoutError`이기도 함)로 실패하고,
진행 중인 모델 호출과 저장소 요청은 취소됩니다.

- `run-rag`는 메시지를 출력하고 `1`로 종료합니다. `run-eval`은 그 질의를 실패로 세고
  (`error_message` 기록, 채점 제외) 나머지 질의를 계속 처리합니다.
- `processing.ignore_errors`를 켜면 오류 답변을 대신 돌려주고, 메시지는
  `search_results.metadata.error`에 남습니다.
- 스트리밍 답변은 모델 호출 후 `search.model_timeout_seconds` 안에 첫 청크를 보내야 합니다.
  청크가 오기 시작하면 전체 제한 시간이 적용됩니다.
- `null`로 두면 두 제한을 모두 끕니다.

질의의 모델 호출(답변 생성과 그 밖의 질의 시점 LLM 호출, 질의 임베딩, Bedrock 재순위화)은
전용 클라이언트를 씁니다. 요청마다 응답을 `search.model_timeout_seconds`(기본 90초)까지
기다리고 botocore 시도는 `search.model_max_attempts`(기본 2회)라서, 멈춘 Bedrock 호출은
약 3분 뒤 포기합니다. 수집과 평가 판정 호출은 읽기 제한 시간 330초와 6회 시도를
유지합니다. OpenAI 호환 서버나 Ollama 서버에서는 질의 호출의 요청 제한 시간이
`providers.llm.timeout_seconds` / `providers.embedding.timeout_seconds`이되
`search.model_timeout_seconds`를 넘지 않습니다. 로컬 서버는 요청을 대기열에 쌓으므로
`backend: local`은 모델 제한 300초, 질의 제한 900초를 미리 채웁니다(여기서 `null`은
미리 채운 값으로 바뀌므로 큰 숫자를 지정).

스트리밍하지 않는 답변은 한 번에 도착하므로 `search.model_timeout_seconds`가 모델이 답변을
생성하는 시간의 상한이기도 합니다. 긴 답변이나 높은 추론 강도(effort)에는 이 값과 질의
제한 시간을 함께 올리세요. 기본값은 실제 AWS에서 측정한 질의 시간에서
정했습니다([근거](./concepts/defaults-and-evidence.md#query-deadlines-and-model-timeouts), 영문). 90초는 가장 느린 질의 전체를 요청 하나로 받아도 여유가
있고, 180초는 멈춘 첫 시도를 한 번 다시 시도할 수 있는 시간입니다. 조정 방법은
[운영 런북](./operations.md#query-deadlines-and-model-timeouts)에 있습니다.

### 필터

`--filters key:value ...`(또는 `RAGInput.filters`)는 검색이 순위를 매길 대상을
좁힙니다. 키·`:`·값 중 하나가 없는 쌍, 두 번 준 키, 전략이 읽는 어떤 인덱스도 선언하지
않은 키는 사용법 오류(종료 코드 2)입니다. 필터는 OpenSearch에서는 `term`/`terms`/`range` 절로, Neptune에서는 `has` 단계로
변환됩니다. 저장소마다 해당 인덱서가 쓰는 필드를 받습니다.

| 저장소 | 필터 가능한 필드 |
|---|---|
| 텍스트 단위 | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.filters.<key>` |
| 엔티티 | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `in_schema`, `type_iri`, `iri`, `source_kinds`, `attr_<key>`, `attributes.<path>` |
| 관계 | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids`, `in_schema`, `predicate_iri`, `source_kinds`, `attr_<key>`, `attributes.filters.<key>` |
| 주장 | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `start_date`, `end_date`, `description`, `source_text`, `text_unit_ids`, `attr_<key>`, `attributes.filters.<key>` |
| 커뮤니티 리포트 | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `level`, `depth`, `max_cut_depth`, `parent_community_id`, `child_community_ids`, `attr_<key>`, `attributes.filters.<key>` |
| Neptune 엔티티 정점 | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `in_schema`, `type_iri`, `iri`, `source_kinds`, `attr_<key>`(있는 경우) |

- OpenSearch에서 `attr_<key>`는 문서 `filters` 메타데이터의 `<key>` 항목입니다(여러
  문서에서 나온 항목은 그 값을 모두 가짐). Neptune 엔티티 정점에서는 엔티티 자체의
  속성입니다(`attr_role`).
- 엔티티와 주장의 `type`, 주장의 `status`는 소문자로 저장되고 `term`은 정확히 일치해야
  하므로 `type:PERSON`이 아니라 `type:person`으로 필터링하세요. 관계에는 `type` 필터가
  없습니다(관계 인덱스에 유형을 저장하지 않음). `in_schema`, `type_iri`,
  `predicate_iri`는 [추출 스키마](#c-추출-스키마) 표시이고, `iri`와
  `source_kinds`(`document`, `seed_graph`)는
  [시드 그래프](#d-시드-그래프가지고-있는-지식-그래프-가져오기)에서 옵니다. keyword나 숫자 필드를
  쓰는 편이 좋습니다. `description`처럼 분석되는 필드에 `term`을 걸면 소문자 토큰 하나와만
  일치합니다. 범위 필터는 Python API에서 `{"gte": ..., "lte": ...}`로 넘깁니다.
  `attr_<key>`, `attributes.filters.<key>`와 주장 날짜는 날짜 형태의 값도 keyword라서
  범위 필터는 문자열로 비교하며, ISO 날짜(`YYYY-MM-DD`)는 이 비교로도 올바르게 정렬됩니다.
  달력 날짜로 읽히는 주장 날짜(`2024-01-15`, `2024-1-5`, `January 15th, 2024`, `Sept. 3, 2024`)는 `YYYY-MM-DD`로 저장하고,
  더 넓은 기간은 `start_date`에는 그 첫날로, `end_date`에는 그 마지막 날로 저장합니다(주장 프롬프트가 연말을
  `YYYY-12-31`로 쓰라고 하는 것과 같습니다). ISO 주(`2024-W03`)는 그 주의 월요일 또는 일요일(요일까지 적은
  `2024-W03-2`는 그날), 분기(`Q3 2022`)는 2022-07-01 또는 2022-09-30, 월(`March 2024`, `2024-03`)은
  2024-03-01 또는 2024-03-31, 연도(`2024`)는 2024-01-01 또는 2024-12-31입니다. 따라서 `start_date`의
  `{"gte": "2022-07-01", "lte": "2022-09-30"}`은 `Q3 2022`와 일치하지만 `{"gte": "2022-08-01"}`은 일치하지
  않고, `end_date`의 `{"lte": "2024-06-30"}`은 "2024년까지"와 일치하지 않습니다. 그 밖의 값(일과 월 순서를 알 수
  없는 `10/08/2024`, `early 2024` 등)은 적힌 그대로 저장해 문자열로 비교하므로 정확히 일치하는 필터만 의미가
  있으며, 범위 필터가 잘못 일치할 수 있습니다(`10/08/2024`는 모든 ISO 날짜보다 앞에 정렬되므로
  `{"lte": "2000-01-01"}`과도 일치합니다). 알 수 없는 날짜(`UNKNOWN`, `TBD`, `present`)는 저장하지 않으므로
  범위 필터를 포함해 어떤 필터에도 일치하지 않습니다.
- 불리언·숫자 필드는 그 유형의 값을 받습니다. `in_schema`는 불리언, `rank`·`confidence`·
  `weight`·`rating`은 숫자, `n_tokens`·`level`·`depth`·`max_cut_depth`는 정수입니다. 이
  필드에 문자열을 주면(`--filters`와 `set-filter`의 값은 모두 문자열이고, 데이터셋이나
  `RAGInput.filters`에도 문자열이 들어갈 수 있음) 검색 전에 리스트와 범위 경계까지 그
  유형으로 바꾸므로, 모든 저장소가 유형을 갖춘 값과 똑같이 일치시킵니다. `in_schema:false`는
  `False`(불리언은 대소문자 구분 없이 `true`/`false`, `1`/`0`, `yes`/`no`를 받음),
  `rank:2`는 `2`가 됩니다. 유형에 맞지 않는 문자열(`in_schema:maybe`, `rank:high`,
  `level:1.5`)은 `InvalidFilterError`를 일으킵니다(명령줄에서는 사용법 오류, 종료 코드 2).
  유형이 다른 값도 마찬가지입니다. 불리언 필드의 숫자(`in_schema: 1`), 숫자 필드의
  불리언(`rank: true`), 정수 필드의 소수(`level: 2.5`. `2.0`은 `2`가 됨)는
  `InvalidFilterError`를 일으킵니다.
  그 밖의 필드 값은 `attr_<key>`와 `attributes.<path>`를 포함해 주어진 그대로 비교합니다.
- 필터는 해당 필드를 선언한 저장소에만 적용됩니다(`type:person`은 텍스트 단위를
  거르지 않음). 선택한 전략이 읽는 저장소 어디에도 선언되지 않은 키(예: `category`,
  `naive`나 `global`의 `type`)는 필터 가능한 키 목록과 함께 `InvalidFilterError`를 일으킵니다
  (`unified_kg_rag/adapters/storage/filter_schema.py`). `naive`는 텍스트 단위를, `hybrid`와
  `mix`는 엔티티·관계·텍스트 단위(`search.lightrag_search.enable_graph_expansion`을 켜면
  Neptune도)를, `local`과 `drift`는 파이프라인이 만드는 모든 인덱스와 Neptune을, `global`은
  커뮤니티 리포트(`search.global_search.text_unit_slots`를 쓰면 선택한 리포트의 커뮤니티로
  범위를 좁힌 텍스트 단위도)를, `simple`은 파이프라인이 만드는 모든 인덱스를 읽습니다. `global`은
  자기 검색 범위를 정하는 키를 거부합니다. `depth`와 `max_cut_depth`(`search.global_search.use_dynamic_selection`을
  켜면 `depth`와 `community_id`), `text_unit_slots`를 쓰면 `community_ids`도 해당합니다. `id`,
  `source_id`, `target_id`는 `local`, `drift`, `mix`, `hybrid`가 id로 가져오거나 탐색으로 닿는
  대상을 좁힙니다(값이 목록에 있는 항목만 돌아옴). 이 키에 범위를 주면 어느 전략에서나 거부되며(이 키로 범위를 좁히지 않는 `naive`와 `global`도 거부해, 필터의 유효 여부가 전략에 따라 달라지지 않음), 어느 키든 빈 값 목록도 거부됩니다. `auto`는 라우팅될 수 있는 전략 중 키를 모두
  적용하는 전략 사이에서만 라우팅하고(기본 집합에서 `type:person`이면 `local`과 `mix`), 그런
  전략이 없을 때만 라우팅 전에 필터를 거부합니다. Neptune은 엔티티 정점에만 필터를 적용합니다.
- OpenSearch는 `attr_<key>`와 `attributes.<path>`를 엄격하게(fail-closed) 적용합니다.
  속성이 없는 항목은 제외됩니다. 관계, 주장, 커뮤니티 리포트는 자기 텍스트 단위가 속한
  문서들의 필터 값을 가지므로 그중 하나가 일치하면 일치합니다. 벡터나 하이브리드 검색은 그
  속성을 가진 문서가 하나도 없는 인덱스를 건너뜁니다(OpenSearch 2.13은 인덱스가 매핑하지
  않은 필드에 대한 kNN 필터를 거부함).
- 문서를 제거하면 남은 엔티티, 관계, 주장에서 그 문서의 필터 값을 지웁니다. 남은 문서
  어디에도 없는 키는 null로 저장되고 null은 어떤 필터와도 일치하지 않습니다.
- `attr_<key>`와 `attributes.filters.<key>`는 문서 속성입니다. 그래프 저장소에는 이것이
  없으므로 그래프 확장은 시드를 제외한 확장 엔티티를, 그 엔티티가 나온 모든 문서의 필터를
  가진 엔티티 인덱스 문서가 이를 통과한 경우에만 남깁니다. 키가 둘 이상이면 그 엔티티의
  텍스트 단위(한 문서의 것) 중 하나도 모든 키를 통과해야 합니다. 또한 필터를 통과한 관계만
  따라 시드에서 `max_hops` 안에 닿아야 하며(제외된 문서만 진술한 관계는 따라가지 않음),
  `Path:`도 그 관계를 따릅니다. 이때 걷기는 관계 인덱스에서 한 홉씩 진행하므로 이 확인을 마친
  뒤에 자르며, 정점 필터(`type`)가 결과에서 뺀 엔티티도 거쳐 갑니다. 그 인덱스가
  없으면(`indexing.opensearch.build_relationship_vector_index: false`, 또는 질의 시점에 인덱스가
  없어 WARNING을 남김) 그래프를 걷고 엔티티 인덱스 확인만으로 정합니다. 텍스트 단위에 걸리는 필터가
  있으면 local과 mix는 상위 순위 청크만이 아니라 엔티티와 연결된 청크 중 필터를 통과한 것으로
  청크 할당량을 채웁니다. 그 밖의 `attributes.<path>` 키는 항목 자체의 속성이며 엔티티 인덱스에만
  적용됩니다. 엔티티 정점에 없는 이 키와 `name.keyword`는 그래프 걷기가 닿은 엔티티를
  LightRAG의 시드까지 포함해 엔티티 인덱스로 확인합니다(`local`과 `drift`의 시드는 필터로 찾은 것).
  이 확인은 걷기가 홉별 상한과 `top_k`로 자른 뒤 남은 엔티티만 보므로, 필터를 통과하지만 그
  아래 순위인 이웃에는 닿지 못합니다. Neptune에서 `attr_<key>`는 엔티티 정점 속성(`attr_role`)이기도 해서 "있는 경우"
  기준으로 적용되지만(그 속성이 없는 정점은 통과) 위의 엔티티 인덱스 확인도 적용되므로, 문서에
  `role` 필터가 없으면 `attr_role:buyer`는 아무것도 고르지 않습니다. 그 밖의 키는 두 저장소
  모두 엄격하게 적용합니다.

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
`set-filter key:value`, `clear-filters`, `show-config`, `quit` / `exit`. 명령은 줄 전체여야
하므로(인자를 받는 명령은 `set-filter`뿐) `help me find ...`는 질문으로 처리합니다.
`set-filter`는 `--filters`처럼 검색 전략의 어느 인덱스에도 없는 키를 거부합니다(`--interactive`와
함께 준 `--filters`는 첫 질문 전에 검사하며, 거부하면 종료 코드 2). `auto`에서는 이미 설정한
필터와 함께, 라우팅될 수 있는 어떤 전략도 모두 적용하지 못할 때만 키를 거부합니다.

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
파일입니다(`providers.doc_status.kind: sqlite`. [로컬 모드](./guides/local-mode.md)(영문) 참고).

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
  템플릿(코드)을 고친 것은 감지하지 못하므로 고친 뒤에는 초기화하세요. 설명 요약 설정
  (`description_summarization`, 그 `custom_prompts`, `models.roles.description_summary`)은
  추출 설정이 아니므로 바꿔도 다시 추출하지 않습니다. 실행이 병합하는 설명은 새 설정으로
  요약하고, `indexing.reset: true`는 모든 설명을 다시 요약합니다(저장된 요약은 만들 때의
  설정을 기록하므로 같은 설정일 때만 재사용합니다).
- **임베딩 모델이나 차원 변경**(`models.embedding.model_id` / `dimension`):
  OpenSearch 인덱스마다 만들 때 쓴 모델과 차원을 기록하며, 증분 실행은 그 값이 이번
  실행과 다르면 첫 모델 호출 전에 멈춥니다(종료 코드 2, 기록이 없는 인덱스도 거부).
  `indexing.reset: true`로 한 번 다시 실행하세요. 다른 모델이나 차원으로 `run-rag`·`run-eval`을
  실행하면 빈 검색 결과로 답하지 않고 `ConfigurationError`로 실패합니다.
- **코퍼스 언어나 번역 언어 변경**(`processing.translation`의 언어 설정,
  `indexing.opensearch.language_analyzers`): 운영 중인 텍스트 필드는 만들 때의
  분석기를 그대로 쓰므로 증분 실행이 첫 모델 호출 전에 멈춥니다(종료 코드 2).
  `indexing.reset: true`로 한 번 다시 실행하세요. 언어를 뺄 때는 멈추지 않습니다.
- **레지스트리를 끄고 접미사 인덱싱**(`backend: local` 프리셋처럼): 그 접미사의 다음
  증분 실행은 첫 모델 호출 전에 멈춥니다. 레지스트리를 켜고 `indexing.reset: true`로
  한 번 다시 실행하거나, 그 접미사는 레지스트리를 끈 채로 쓰세요.
- **재구축**(`indexing.reset: true`): 접미사를 서비스에서 내리지 않고 코퍼스 전체를
  다시 인덱싱합니다. 격리된 문서는 건너뛰고 레코드를 그대로 둡니다. 실패한 reset은 아무것도
  지우지 않으므로 다시 실행해도 안전합니다
  (동작 방식: [운영 런북: 다시 수집하기](./operations.md#re-ingest)). 바뀌지 않은 텍스트의
  청크 경계, 추출 레코드, 벡터를 다시 쓰고, 입력이 바뀌지 않은 설명 요약과 커뮤니티 리포트
  응답도 저장된 것을 다시 쓰므로, 같은 모델과 설정으로 재구축하면 코퍼스가 바뀐 부분만
  모델에 묻습니다. 증분 실행 뒤의 재구축은 증분 실행과 다르게 군집화한 커뮤니티의 리포트를
  여전히 묻습니다(증분 실행은 델타만 군집화합니다). 요약이나 리포트 설정을 바꾼 뒤에는
  모든 요약이나 리포트를 다시 묻고, 임베딩 모델이나 차원을 바꾼 뒤에는 모든 텍스트를 다시
  임베딩합니다. 전부 다시 묻고 임베딩하려면 `--force-rebuild`를 함께 쓰세요.
- **재구축 시점:** 증분 실행은 델타만 군집화하고 그 커뮤니티를 덧붙입니다. 마지막 전체
  빌드 뒤 증분 실행이 20번 쌓이거나, 접미사의 커뮤니티가 그 빌드가 군집화한 수의 1.5배가
  되면 실행이 `indexing.reset: true`를 권하는 WARNING(`Communities of '<suffix>' are
  due for re-clustering: ...`)을 남깁니다. 자동으로 재구축하지는 않습니다. 임곗값은
  `indexing.community_compaction`에서 정하고(`null`이면 그 검사를 끕니다), 카운터는 빌드
  레코드와 `community_growth_ratio`, `runs_since_clustering` 지표에 있습니다
  ([운영 런북: 증분 실행](./operations.md#incremental-runs)). 빌드 레코드에 카운터가
  없는 접미사는 다음 전체 빌드부터 셉니다.

### 삭제 범위

실행의 **범위**는 인덱스 접미사(`document_parsing.index_value`와
`indexing.additional_suffix`)와 코퍼스 소스(`document_parsing.source_scope`. 기본값은
해석된 소스 디렉터리이며, 컨테이너 엔트리포인트는 동기화 원본인 S3 URI로 설정)입니다.
문서의 레지스트리 키는 이 범위와 소스 디렉터리 기준 상대 경로이며, 실행은 자기 범위의
문서만 비교하고 삭제 대상으로 봅니다. 따라서 다른 테넌트(다른 `index_value`)의 실행은 이
테넌트의 문서를 삭제하지 않고, 하위 폴더를 소스 디렉터리로 따로 실행해도 코퍼스의
나머지는 삭제되지 않으며(그 파일들은 별도 문서로 등록되므로 같은 파일을 두 루트에서 한
접미사로 인덱싱하지 마세요), 소스가 다른 두 코퍼스는 한 접미사에서 상대 경로가 같아도
서로 다른 문서입니다. 문서와 텍스트 단위 ID도 범위에서 만들어지므로, 경로와 머신이 달라도
같은 ID가 나와야 하면 `source_scope`를 지정하세요(지정하지 않으면 같은 코퍼스라도 다른
디렉터리나 호스트에서 인덱싱하면 ID가 달라집니다). 파싱이나 로딩에 실패한 파일과, 그
형식을 읽는 로더가 없어(예: `unstructured` 같은 선택 extra 미설치) 건너뛴 파일은
`failed`로 보고되며, 다음 실행에서 다시 읽을 때까지 인덱싱된 내용을 유지합니다.

**대량 삭제.** 범위가 가진 문서 중 `indexing.max_deletion_share`(기본값 `0.5`)보다 많은
비율을 삭제하게 되는 실행은 쓰기 전에 멈추고 묻습니다(`This run deletes N of the M
documents ...; continue?`). 일부만 동기화된 S3나 마운트되지 않은 볼륨은 의도적으로 지운
파일과 구별되지 않기 때문입니다. `run-ingestion`은 터미널에서 묻고,
`--confirm-deletions`(또는 `UKR_CONFIRM_DELETIONS=1`)는 이 질문에만, `--yes`는 모든 질문에
'예'로 답합니다. '아니요'로 답하거나 터미널이 없으면 실행이 멈춥니다(exit 1). 비율은
문서만 셉니다. 시드 그래프 샤드는 `processing.seed_graphs` 설정을 따르므로 시드 그래프를
빼거나 샤드 수가 바뀌어도 묻지 않고, 샤드가 많아도 비율이 희석되지 않습니다. 문서가
10개 미만인 범위, 은퇴시킨 범위(retired scope)의 레코드, 재설정(reset), 레지스트리를 끈
실행은 검사하지 않으며, `1.0`이나 `null`로 검사를 끕니다. Python에서는
`DataIngestionPipeline(..., confirm_deletions=callable)`을 넘기세요. 넘기지 않거나
'아니요'로 답하면 `run`이 `ConfirmationDeclinedError`를 발생시킵니다.

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
([운영 런북: 증분 실행](./operations.md#incremental-runs)).

### 실패한 문서

텍스트 단위 하나라도 번역, 추출, gleaning, 주장 추출에 실패했거나, 그 문서를 포함한 기본
수준 커뮤니티의 리포트가 생성되지 않았거나, `--continue-on-error` 실행에서 산출물 단계가
비정상 종료했거나, 산출물 하나를 쓰는 데 실패했거나, 텍스트 단위의 추출 레코드를 저장하지
못한 문서는 `FAILED`로 기록됩니다. 다음 실행은 파일이 바뀌지 않았더라도 이 문서를 다시
처리하며, 같은 내용으로 `indexing.max_document_failures`(기본값 `3`)번 연속 실패하면 더
재시도하지 않습니다. 그 뒤에는 파일을 수정하거나 한도를 올리면 다시 시도합니다.
주장은 선택 사항이라, 주장 추출만 단위 자체의 오류로 실패하면 문서는 그 주장 없이 인덱싱되고
실패로 세지 않으며 다음 실행이 그 주장을 한 번 더 추출합니다.
[운영 런북: 실패한 문서](./operations.md#failed-documents)를 참고하세요.

### 격리된 문서

처리하다가 태스크를 죽이는 문서(메모리 부족, 프로세스를 죽이는 파서)는 이후 실행도 같은
식으로 멈추게 합니다. 실행이 이 문서의 바뀌지 않은 내용을 처리하다
`indexing.max_document_attempts`(기본값 `3`)번 연달아 죽고 그 마지막 실행이 이 문서만
처리하고 있었다면, 다음 실행은 시도들이 쓴 내용을 지우고 문서를 `QUARANTINED`로 기록하며
파일 이름이 담긴 WARNING을 남기고 나머지를 인덱싱합니다. 파일이 바뀌거나 한도를 올릴
때까지 이 문서는 reset에서도 건너뜁니다.

- 죽은 실행만 셉니다. 완료하거나 실패하거나 중지된(SIGTERM, Ctrl-C) 실행은 처리 중
  표시를 지우고, 전체 재구축은 시도를 세지 않습니다.
- 여러 문서를 처리하다 죽었다면 다음 실행들이 원인 문서만 남을 때까지 의심 문서를 절반씩
  줄이고, 그동안 다른 새 문서와 변경된 문서는 미룹니다(`total_documents_deferred`).

[운영 런북: 격리된 문서](./operations.md#quarantined-documents)를 참고하세요.

### 동시 실행

레지스트리를 쓰면 실행은 첫 단계부터 끝날 때까지 인덱스 네임스페이스의 쓰기 임대를
유지하며, 같은 네임스페이스의 두 번째 실행은 저장소를 읽거나 무엇이든 쓰기 전에
`NamespaceLeaseError`로 멈춥니다. [운영 런북: 동시 실행](./operations.md#concurrent-runs)을
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
[설계 §5.4](./design.md#54-identity-derived-content-and-merge-rules)에 있습니다.

---

## 6. 평가 (`run-eval`)

`run-eval --data-path <dataset.json>`은 데이터셋의 모든 질문에 답한 뒤 답변과 출처를
채점합니다. 평가기 셋은 LLM 없이 비용 없이 동작하고, 둘은 LLM 심사 모델을 씁니다.
전략을 여럿 주거나 `--repeats`를 주면 짝지은 비교가 됩니다.

모든 플래그와 기본값은 [CLI 레퍼런스](./reference/cli.md#run-eval)(영문)에 있습니다.

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

심사 평가기 둘은 `eval` extra가 필요합니다(`uv sync --extra eval`, [설치](./getting-started/aws.ko.md#설치)). extra가
없으면 `run-eval`은 첫 질의 전에 extra 이름을 알려 주고 상태 코드 `2`로 종료하며,
`processing.ignore_errors: true`여도 같습니다. 위의 심사 모델 없는 실행에는 extra가
필요 없습니다. 그 밖에 켜 둔 평가기를 만들 수 없거나(예: 심사 모델의 Bedrock 접근 권한
없음) 설정을 거부하면 실행이 멈춥니다. `processing.ignore_errors: true`이면 그 평가기를 빼고
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
  exact match보다 정답 여부를 훨씬 잘 따라갑니다. 소유격은 그 단어로 읽습니다(`Depot`은
  `Depot's`와 맞고, `Depots`는 다른 단어). `graph_aware`도 같은 방식으로 맞춥니다.
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
- **`exact_match`와 `token_f1`**은 답변 가운데 `evaluation.answer_span`이 가리키는 부분을
  채점합니다. SQuAD, HotpotQA, MuSiQue는 모델이 예측한 짧은 답(`Depot` 같은 구간)을 정답과
  비교하는데, 답변 프롬프트는 한 문장 답에 근거를 덧붙여 쓰므로 답변 전체를 채점하면 정답
  여부만큼 답변 길이를 재게 됩니다.
  - `first_line`(기본값, 비용 없음): 비어 있지도 않고 Markdown 제목도 아닌 첫 줄, 즉 한
    문장 답입니다(`Depot stores widgets for Vendor.`는 `Depot`에 대해 token F1 1/3. EM은
    여전히 낮게 나옴).
  - `extracted`: `models.roles.answer_span` 모델(fast 계층. 프롬프트
    `custom_prompts.answer_span_extraction_*`)이 답변에서 짧은 구간을 그대로 옮겨 옵니다.
    서로 다른 질문·답변마다 한 번 호출하고 반복과 전략 사이에서 다시 씁니다. 호출이
    실패하거나, 빈 응답이거나, 답변에 없는 구간이면 첫 줄로 대신 채점하고
    `answer_span_statistics.fallbacks`에 셉니다. 논문 수치에 가깝게 보려면 이 설정을
    쓰세요.
  - `full`: 답변 전체.

  리포트마다 `answer_span`과 채점한 `scored_span`이 기록됩니다. 정규화는 SQuAD v1.1
  방식(NFKC, 소문자화, 문장 부호와 영어 관사 제거, 공백 압축)이되 의미를 바꾸는 문장
  부호(`-5`, `1.5`, `C++`)는 남깁니다. `evaluation.answer_normalization: official_squad`는
  공식 스크립트(MuSiQue 평가도 같음)이고, `hotpotqa`는 여기에 `yes`/`no`/`noanswer`인
  구간이나 정답은 둘이 같을 때만 token F1을 받는 HotpotQA 규칙을 더합니다. token F1은
  공백으로 나누므로 중국어와 일본어에서는 exact match와 같아지고, 한국어 조사가 붙으면 두
  지표 모두 낮게 나옵니다.

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
  없는 `search_strategy`, 또는 체인이 거부하는 `filters`: 빈 목록 같은 잘못된 값이나
  항목의 전략이 거부하는 키. 비교 실행이면 비교하는 모든 전략으로 확인)가 있으면
  `Error: invalid evaluation dataset:`과 항목 번호, ID를 출력하고 종료 코드 2로 멈춥니다.

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
- `answer_span_statistics`: exact match와 token F1이 읽은 `answer_span`, 채점한 리포트
  수(`scored`), 추출에 실패해 첫 줄로 채점한 수(`fallbacks`)입니다.
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
  - 모델 ID와 `judge_is_answer_model`(심사 모델이 답변 모델이면 심사 점수가 부풀려집니다.
    Bedrock의 기본 심사 모델인 Opus 5.5는 기본 답변 모델과 다르고, 답변 모델이 Opus 5.5이면
    다른 모델로 바뀝니다. 답변 모델과 같은 `models.roles.evaluation`을 직접 지정하면(`default`
    등) 그대로 쓰고 경고합니다).
    비교의 `--pairwise-judge`도 심사 모델로 세며, `pairwise_judge`에 그 설정이 기록됩니다.
  - `answer_scoring`: exact match와 token F1에 쓴 `answer_span`과
    `answer_normalization`(`answer_match` 사용 시), `models.answer_span`은 구간을 추출한
    모델입니다. 채점 방식이 다른 실행은 이 두 지표로 비교할 수 없습니다.
  - `retrieval_config_sha256`: 답변이나 점수를 바꿀 수 있는 모든 구성 설정의 해시(모델 ID와
    수집 설정 포함, 리전·엔드포인트·자격 증명·경로·시간 제한 같은 운영 설정은 제외)이므로
    같은 평가는 어느 환경에서든 같은 해시가 됩니다. `run-eval`의 질의 플래그(`--suffix`,
    `--top-k`, `--search-type`, `--retrieval-multiplier`, `--enable-thinking`)는 구성이
    아니므로 해시에 들어가지 않고 CLI 인수에 기록됩니다.
  - `index`: 질의한 접미사마다 벡터 인덱스를 마지막으로 쓴 수집 실행(`pipeline_id`,
    `mode`, `built_at`, `documents`, 추출 설정, 증분 실행이면 그 실행이 올라선
    `full_build`, `communities` 카운터. §5 참고)입니다. 최선 노력 방식이며 인덱스 메타데이터가 없으면 `null`입니다.

  `EvaluationManager.evaluate_dataset`이 만들므로 라이브러리 호출자도 받습니다
  (`dataset_path=` / `cli_args=`를 넘김).

질의별 리포트에는 평가기마다 `metrics`가 있고(서로 무관한 지표의 평균은 의미가 없으므로
종합 점수는 없음), 보고된 출처마다 순위 순으로 `retrieved_source_ids`(귀속된 파일 이름,
귀속할 수 없으면 `[]`), `retrieved_source_types`, `retrieved_entities`,
`retrieved_relationships`가 기록됩니다.

### 전략 비교

`run-eval --search-strategy local,mix,global --repeats 3`은 같은 질의를 전략마다
`--repeats`번 답하고 채점하며(지정한 전략이 항목별 `search_strategy`보다 우선. 전략은 플래그 하나에
나열하며, `--search-strategy`를 두 번 주면 사용법 오류), 실행마다
결과 파일을 저장하고 `evaluation_comparison_<timestamp>.json`을 추가로 씁니다.

- `strategies.<s>.metrics.<evaluator>/<metric>`: 질의별 반복 평균을 질의 전체로 평균한
  값과 `repeat_stdev`(실행 간 잡음)입니다. `routed_strategies`는 `auto`가 어떻게
  분기했는지 보여 줍니다. 체인은 질문마다 엔티티와 키워드를 한 번만 추출하므로
  (`search.query_extraction_cache_size`, 질문 1024개), 반복 실행과 이를 읽는 전략(`mix`와
  `hybrid`는 키워드, `local`은 엔티티)은 같은 엔티티·키워드로 검색하고, 실행 간 잡음은
  검색과 답변에서만 생깁니다.
- `deltas_vs_baseline.<s>.<metric>`: 두 전략이 모두 채점한 질의에서 첫 전략과 짝지어
  비교한 값입니다. `mean_delta`, 95% 짝지은 부트스트랩 구간(`ci_low`/`ci_high`. 2,000번
  재표집, 시드 0), `wins`/`ties`/`losses`, 질의별 차이, `unpaired`가 있습니다. 답변
  생성에 실패한 질의는 다른 실행이 채점한 지표 가운데 자기 전략이 어느 질의에서든 채점하는
  지표에서 0점이므로(`answer_length` 제외) 실패로 이득을 볼 수 없고, 그 전략이 만들지 않는
  지표(`global`은 엔터티를 보고하지 않으므로 `retrieved_entity_coverage`가 없음)는 손실로
  지어내지 않습니다. 아무것도 채점하지 못한 전략은 모든 지표에서 0점입니다. 이 규칙은
  데이터로 판단하므로 실패한 질의만 채점할 수 있던 지표(`expected_relationships`가 있는
  항목이 하나뿐인 경우 등)도 0점으로 채우지 않습니다(`failed_answers` / `failed_queries`.
  차단 수는 `guardrail_blocked_counts`. 전략마다 실패하거나 차단된 수가 다르거나, 실행마다
  exact match와 token F1이 읽은 `answer_span`이 다르거나, `answer_span: extracted`에서 추출한
  구간이 첫 줄로 대체된 전략이 있거나(`answer_span_fallbacks`) 그 대체 비율이 전략마다 다르면
  `warnings`에 나옴). 구간이 0을 포함하지 않으면 그 차이가 뒷받침됩니다.
  이진 지표의 10%p 차이는 질의 50개에서 15%, 200개에서 59%의 실행에서만 찾아지므로 수백
  개 질의로 비교하세요.
- `deltas_by_category.<s>.<category>.<metric>`: 같은 차이를 데이터셋 `category`(MuSiQue의
  홉 수 등)별로 나눈 값입니다. 구간은 범주마다 따로 구하고 범주와 지표 수에 따른 다중 비교
  보정을 하지 않으므로, 여러 개 가운데 하나만 0을 벗어난 결과는 탐색적으로 읽으세요.
- `pairwise_vs_baseline.<s>`(`--pairwise-judge` 사용 시): GraphRAG / LightRAG 논문의
  맞비교 LLM 판정으로, `models.roles.evaluation`이 첫 반복의 답변을 포괄성
  (comprehensiveness), 다양성(diversity), 이해 도움(empowerment), 직접성(directness)으로
  판정합니다(`evaluation.pairwise_overall: true`이면 LightRAG의 `overall`도 추가. 프롬프트:
  `custom_prompts.pairwise_comparison_*`). 쌍마다 순서를 바꿔 두 번 판정하고 두 순서가
  일치한 기준만 승리로 셉니다. `position_consistency`(전체와 기준별)는 판정한 쌍 가운데 두
  순서가 일치한 비율이며, 0.5 미만이면 경고합니다. 그런 판정 모델은 위치로 결정하므로
  무승부가 답변에 대해 말해 주는 것이 없습니다. `candidate_win_rate`는 결정된 쌍을 승리 1,
  무승부 0.5, 패배 0으로 매긴 평균이고, 이 질의별 점수에 대한 부트스트랩 구간
  (`ci_low`/`ci_high`)이 함께 나옵니다. 판정 모델이 결정한 쌍이 없으면 `null`입니다. 한쪽
  답변이 실패한 쌍은 판정하지 않고 다른 쪽이 이깁니다(`candidate_errors`,
  `baseline_errors`. 둘 다 실패: `both_errors`). `answerable: false` 항목은 판정하지
  않습니다(`unanswerable_skipped`). `failed`는 판정 모델이 결정하지 못한 쌍(호출 실패, 또는
  `1`, `2`, `tie`가 아닌 판정)의 수이며 `--max-failure-rate`가 적용됩니다. 판정 모델이 답변
  모델과 같으면 경고합니다.
- `cost.strategies.<s>`: 답변의 질의당 모델 사용량을 모든 반복의 모든 답변에 대한
  `mean` / `p50` / `p95`로 보여 줍니다. `llm_calls_per_query`, `input_tokens_per_query`,
  `output_tokens_per_query`, `total_tokens_per_query`, `latency_seconds`, `totals`,
  `cache_hits`가 있고, `evaluation.prices`를 설정하면 `usd_per_query`와 `usd_total`도
  있습니다. `cost.deltas_vs_baseline.<s>`는 지표처럼 기준과 짝지어 비교합니다
  (`usage/llm_calls`, `usage/input_tokens`, `usage/output_tokens`, `usage/total_tokens`,
  `usage/cost_usd`, `latency/response_time_seconds`. 질의별 반복 평균, 부트스트랩 구간
  포함). 실패하거나 시간 초과된 답변은 실패하기 전까지 쓴 만큼 셉니다. 답변마다 사용량 범위와
  타이머를 따로 두므로 그 결과에는 시도들의 호출·토큰과, 첫 시도 시작부터 마지막 시도 끝까지의
  시간이 남습니다(`failed_answers`가 그 수이며 경고로도 알림). 그래서 실패했다고 더 싸거나
  빨라 보이지 않지만, 빨리 실패한 답변은 실제로 비용이 적으므로 비용은 `failed_answers`와 실패
  경고와 함께 읽으세요. 저장된 답변 가운데 사용량 기록이나 지연 시간이 없는 답변은
  `answers_without_usage` / `answers_without_latency`로 세고 경고하며, `usage/*`나
  `latency/*` 측정값의 `--fail-if`는 실패합니다. 한 실행의 전략과 반복은 체인의 질의 추출
  캐시를 공유하므로, 앞선 실행의 추출을 재사용한 뒤쪽 실행은 그 호출 비용을 내지 않으며
  (`cache_hits`에 셈) 비용이 앞서 실행된 것에 따라 달라집니다. 반복 횟수나 답변당 캐시 적중
  수가 기준과 다른 전략은 경고합니다. 캐시 없는 비용을 비교하려면
  `search.query_extraction_cache_size: 0`으로 설정하세요. 품질 차이는 이 설정과 무관합니다.
  `run-eval`은 전략마다 첫 실행 전에 그 전략으로 합성 질의 하나에 답하며, 이 질의는 시간을
  재지 않고 채점하지 않으며 사용량에도 넣지 않습니다(`evaluation.warm_up_query`, 기본값
  `true`). 그래서 첫 전략의 지연 시간에 저장소 연결을 열고 모델을 처음 호출하는 시간이 들어가지
  않습니다. 실행마다 전략당 질의 하나(그 모델 호출)만큼 비용이 더 들며, 이 비용은 실행의
사용량에도 USD 추정치에도 들어가지 않습니다(Bedrock 청구에는 들어감). `false`로 끌 수 있습니다.
- 첫 실행의 `run_manifest`. `--pairwise-judge`를 쓰면 `pairwise_judge_model`과 판정 모델의
  `pairwise_usage`도 있습니다.

비교는 `evaluation_comparison_<timestamp>.csv`(기준이 아닌 전략, 측정값, 질의마다 한 행.
`baseline`, `baseline_value`, `value`, `delta`, `category`)와 `.md`(지표 표와 비용 표, 맞비교
판정의 승률, 게이트, 경고. 풀 리퀘스트나 CI 작업 요약용)로도 저장합니다.

패키지에는 모델 가격이 들어 있지 않습니다. USD 추정치가 필요하면 사용하는 리전의 현재
[Amazon Bedrock 가격](https://aws.amazon.com/bedrock/pricing/)을 100만 토큰 단위로,
사용량 기록이 쓰는 모델 ID(`usage.by_model`. 팩터리가 추론 프로필을 확정했으면
`global.<model>` 같은 프로필 ID)를 키로 채우세요. 가격이 없는 모델을 쓴 답변은 추정치가
없고 비교가 경고합니다. `input_tokens`에는 프롬프트 캐시 읽기·쓰기 토큰이 들어 있습니다.
그 단가(`cache_read_per_million_tokens`, `cache_write_per_million_tokens`)를 주면 따로
계산하고, 없으면 입력 단가로 계산합니다(Claude에서 캐시 읽기는 입력의 10분의 1이므로 상한값).

```yaml
evaluation:
  prices:
    "global.<answer-model-id>":
      input_per_million_tokens: 0.0
      output_per_million_tokens: 0.0
      cache_read_per_million_tokens: 0.0
      cache_write_per_million_tokens: 0.0
    "global.<fast-model-id>": {input_per_million_tokens: 0.0, output_per_million_tokens: 0.0}
```

추정치는 사용량에 기록된 호출만 다룹니다. botocore가 다시 보낸 요청(`request_retries`), 실패한
호출(`failed_calls`), 예열 질의는 가격에 들어가지 않으므로 CloudWatch 호출 수와 청구액이 조금
더 많을 수 있습니다(패키지의 2026년 11월 벤치마크 실행에서 Sonnet 5.5는 기록된 호출보다
호출 수가 13% 많았고 입력 토큰은 4.6%, 출력 토큰은 0.9% 이내였으며, Haiku 5.5는 0.2% 이내).

콘솔에도 같은 요약이 출력되며, 실패 허용 기준은 모든 실행에 적용됩니다.

### 저장한 결과로 비교 다시 만들기, 다시 판정하기, 게이트

`run-eval --from-results <dir|files>`는 이전 실행이 저장한 파일(`evaluation_results_*`와 그 옆의
`evaluation_reports_*`, `evaluation_summary_*`. 기본값인 `evaluation.save_detailed_results:
true`일 때 저장됨)로 질의에 답하거나 저장소를 열지 않고 비교를 다시 만듭니다. 실행을 전략별로
묶어 가장 오래된 실행이 나열한 순서로 비교하므로(`--search-strategy`를 주면 그 전략을 그 순서로
비교), 중간에 멈춘 비교는 빠진 전략만 같은 디렉터리에 같은 `--repeats`로 실행한 뒤 다시 만들면
끝납니다. 이렇게 만든 차이는 중단 없이 실행한 비교와 같습니다. 실행이 모두 사라진 전략은
`--repeats N`으로 다시 실행하고, 반복 일부가 저장된 전략은 빠진 반복만 실행하거나(남은 횟수를
`--repeats`로) 그 전략의 일부 파일을 지운 뒤 N번 모두 실행하세요.

```bash
# naive 실행 중에 멈춤(--repeats 3으로 실행): naive만 같은 디렉터리에 같은 --repeats로
# 실행하고 다시 만들기.
run-eval --data-path eval.json --search-strategy naive --repeats 3 --output-directory out/
run-eval --from-results out/ --output-directory out/

# 저장한 답변을 다른 판정 모델로 다시 판정(답변 호출 없음).
run-eval --from-results out/ --pairwise-judge --judge-model <judge-model-id>

# CI: 이전 실행과 비교해 뒷받침되는 성능 저하가 있으면 실패.
run-eval --data-path eval.json --search-strategy local \
  --baseline-results ci/previous/ --output-directory ci/current/ \
  --fail-if "answer_match/token_f1:ci_high<-0.02" \
  --fail-if "usage/total_tokens:mean_delta>500"
```

- 한 전략의 저장된 실행들은 그 전략의 반복으로 합칩니다. 이때 질의 플래그(`suffix`, `top_k`,
  `search_type`, `retrieval_multiplier`, `enable_thinking`), `retrieval_config_sha256`,
  `git_sha`, `index` 빌드나 답변 모델이 서로 다르면(설정이 다른 실행은 다른 디렉터리에 두거나
  결과 파일을 직접 나열), 전략마다 반복 횟수가 다르면, `evaluation_summary_*` 파일이 없는
  실행이 있으면(매니페스트가 없어 이 확인과 답변 채점·자기 선호 확인을 할 수 없음) 경고합니다.
- `--from-results`와 `--pairwise-judge`를 함께 쓰면 저장한 답변을 다시 판정합니다
  (`--judge-model`은 여기서나 일반 실행에서나 그 실행의 `models.roles.evaluation`을 정함.
  `--pairwise-judge`도 LLM 판정 평가기도 없으면 아무것도 판정하지 않으므로 경고함).
  자기 선호 경고는 그 판정 모델을 저장된 실행의 답변 모델과 비교합니다.
- `--baseline-results`는 이전 실행을 기준으로 삼습니다. 그 실행의 전략마다
  `baseline:<strategy>`라는 이름을 붙이고, 이번 실행이나 `--from-results`의 각 전략을 같은
  전략의 기준과 비교합니다(`deltas_vs_baseline.<s>`와 `cost.deltas_vs_baseline.<s>`는
  `baseline:<s>`와의 차이이며, `baselines`가 전략마다 기준을 알려 줌). 이전 실행에 없던 전략은
  그 실행의 첫 전략과 비교하고 경고합니다. 기준 실행은 비교할 실행과 같은 `--repeats`로
  만드세요. 반복 실행은 질의 추출 캐시를 재사용하므로 횟수가 다르면 사용량 차이가 달라집니다
  (경고로 알림). 실행마다 디렉터리를 따로 두세요. `--output-directory`가 `--baseline-results`
  디렉터리이면, 다음에 그 디렉터리를 기준으로 읽는 실행이 이번 결과를 반복으로 합치므로
  경고합니다. exact match와
  token F1이 답변을 다르게 읽은 실행(실행 매니페스트의 `answer_scoring`:
  `evaluation.answer_span`, `answer_normalization`)은 거부하며(종료 코드 2), 답변을 생성하기
  전에 확인합니다. 데이터셋이 다른 실행은 경고합니다(공통 질의 ID만 짝지음).
- `--fail-if <measure>:<stat><op><value>`(여러 번 지정 가능)는 비교한 전략의 짝지은 차이가
  조건을 만족하면 종료 코드 1로 끝냅니다. `<measure>`는 `deltas_vs_baseline`의 키
  (`<evaluator>/<metric>`)나 `cost.deltas_vs_baseline`의 키이고, `<stat>`은 `mean_delta`,
  `ci_low`, `ci_high` 가운데 하나입니다. `ci_high<-0.02`는 구간 전체가 -0.02 아래일 때, 즉
  질의 표본이 뒷받침하는 성능 저하일 때만 실패합니다. 그 측정값의 차이가 없는 전략도 게이트에서
  실패하며, 전략이나 그 기준의 일부 답변에 값이 없는 비용 측정값(`usage/*`, `latency/*`)도
  실패합니다. 전략과 그 기준의 실패한 답변 수가 다를 때도 비용 측정값 게이트는 실패합니다.
  실패한 답변도 실패 전까지 쓴 비용으로 세므로 그 차이에는 실패의 차이도 섞이기 때문입니다. 결과는 `comparison.gates`와 JSON의 `gate_failures`에 있습니다. 비교가 있어야
  하므로 전략 두 개 이상이나 `--baseline-results`가 필요합니다.

### 벤치마크 (MuSiQue)

`prepare-benchmark musique`는 [MuSiQue](https://github.com/StonyBrookNLP/musique)(답변
가능, 다중 홉, CC BY 4.0. Trivedi 외, TACL 2022)를 코퍼스와 `run-eval` 데이터셋으로
바꿉니다. 데이터는 이 저장소에 포함하지 않습니다. 기본적으로 공개 미러의 고정 리비전에서
답변 가능 dev 분할(질문 2,417개)을 내려받아 SHA-256을 확인하며, `--data-path`로 로컬
`musique_ans_v1.0_*.jsonl`을 대신 읽을 수 있습니다. 전체 파일(`musique_full_v1.0_*.jsonl`)도
읽지만 그 답변 가능한 절반은 MuSiQue-Ans가 아니므로(단락과 답변 별칭이 다름) 행에
`musique` 대신 `metadata.benchmark: musique-full-answerable`을 붙이고, 점수를 MuSiQue-Ans
결과와 비교할 수 없다는 WARNING을 남깁니다. 이 레이블은 출력(콘솔과
`--output-format json`)에 나오며, `run-eval`은 요약(`benchmarks`, 여러 전략이나 반복의 비교 표
아래에도 출력)과 실행 매니페스트(`dataset.benchmarks`)에 레이블별 질문 수를 남깁니다. `corpus/`(서로 다른 단락마다 텍스트
파일 하나. 질문마다 근거 단락 2-4개와 방해 단락 최대 18개)와 `eval_data.json`(`answer`,
`answer_aliases`, 근거 단락인 `reference_sources`, 홉 수인 `category`)을 씁니다.
`corpus/`가 비어 있지 않으면 아무것도 내려받기 전에 사용 오류(종료 코드 2)로 끝납니다.

```bash
# 1. 시드 0으로 질문 100개(단락 약 1,900개) 표집
prepare-benchmark musique --output-directory bench/musique --limit 100

# 2. 단락을 별도 접미사로 수집
run-ingestion --source-directory bench/musique/corpus --config-path musique.yaml

# 3. 두 방법론을 짝지어 비교하고 EM / F1은 짧은 답 구간으로 채점
run-eval --data-path bench/musique/eval_data.json --suffix musique \
  --search-strategy local,mix,global,naive --repeats 2 --config-path musique.yaml
```

`musique.yaml`에는 평소 설정에 다음을 더합니다.

```yaml
processing:
  document_parsing:
    index_value: musique          # 벤치마크를 기존 인덱스와 분리
evaluation:
  answer_span: extracted                 # 답변이 제시한 짧은 구간으로 EM / F1 채점
  answer_normalization: official_squad   # MuSiQue 평가 스크립트의 정규화
  enabled_evaluators: [answer_match, retrieval]   # 판정 모델 없이. 필요하면 langchain / ragas 추가
```

MuSiQue는 HotpotQA, SQuAD처럼 독해(reader) 모델이 예측한 짧은 답을 정답과 별칭에 대해 공식
정규화의 EM과 token F1로 채점합니다. RAG 답변은 한 문장 답과 근거이므로
`answer_span: extracted`(답변마다 fast 모델 호출 한 번)가 먼저 모델로 구간을 옮겨 옵니다.
이렇게 하면 수치가 논문에 가까워지지만, 논문의 독해 모델은 구간을 출력하도록 학습되었으므로
같지는 않습니다. 기본값 `first_line`은 비용이 없지만 한 문장 답을 채점하므로 EM이 논문보다
훨씬 낮습니다. GraphRAG와 LightRAG 논문은 EM / F1을 보고하지 않고 맞비교 LLM 판정
(`--pairwise-judge`)을 씁니다.

`answer_contains` / `token_f1`과 `retrieval`(`category`별, 비교에서는
`deltas_by_category`)을 읽으세요. 수집 비용은 단락
수(질문당 약 19개)에 비례합니다. 결과를 발표할 때는 논문을 인용하세요.

---

## 7. 시각화 (`run-visualization`)

이미 내보낸 시각화 데이터 JSON으로 그림을 그리는 **독립** 렌더러이며, 수집을 다시
실행하거나 AWS에 접근하지 **않습니다**. 모든 플래그와 기본값은
[CLI 레퍼런스](./reference/cli.md#run-visualization)(영문)에 있습니다.

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

`viz` extra가 필요합니다(`uv sync --extra viz`, [설치](./getting-started/aws.ko.md#설치)). 렌더러는
**`interactive`**(pyvis)와 **`static`**(Bokeh) 두 개이며 `graph.visualization`
아래(`interactive.*`, `static.*`, `centrality.*`)에서 설정합니다. 수집 단계는 직접
렌더링하지 않습니다.

- **입력.** `run-ingestion`의 커뮤니티 탐지 단계는 그래프가 코퍼스 전체를 담는
  실행(전체 빌드, 초기화, 모든 문서가 새 문서이거나 변경된 증분 실행)마다
  `visualization_data.json`을 씁니다. 변경되지 않은 문서를 건너뛴 증분 실행은 아무것도
  쓰지 않고(INFO 로그만 남김) 이전 파일을 그대로 둡니다. 전체 그래프를 보려면 전체
  빌드(`indexing.reset: true`)를 한 번 실행하세요.
- **위치.** `graph.visualization.outputs_directory`이며 기본값은
  `<캐시 디렉터리>/<pipeline_id>/visualization/`입니다. S3 캐시 동기화는 `.json`만
  올리므로 HTML은 로컬에서 다시 렌더링하세요.
- **내용.** 노드(레벨 0 `community_id`로 색을 정함), 간선, 커뮤니티 계층이며 벡터는
  뺍니다. 파일을 쓸 때 모델을 호출하지 않으며, `run-visualization`이 렌더링할 때 그래프
  구조로 시드를 고정한 spring 레이아웃과 `centrality` 지표를 계산합니다.
- **크기.** 대화형 그래프는 차수 기준 상위 `interactive.max_nodes`개 노드(기본 2000)만
  남기고 그 노드만 배치합니다.
- 간선 두께와 불투명도는 그래프 자체의 가중치 범위에 맞춰 로그 스케일로 조정합니다.
- 렌더러가 실제로 쓴 파일만 보고하며(같은 이름의 기존 파일은 먼저 지움), 아무것도
  렌더링하지 못하면(예: 빈 그래프) 상태 코드 `1`로 종료합니다.

---

## 8. 프롬프트 튜닝 (`run-prompt-tuning`)

디렉터리에서 문서를 표본으로 뽑아 `models.roles.graph_extraction`으로 코퍼스
특성(도메인, 언어, 페르소나, 엔티티 유형)을 분석하고, 검토해서 `config.yaml`에 합칠
도메인 맞춤 `custom_prompts` YAML 조각을 씁니다. 모든 플래그와 기본값은
[CLI 레퍼런스](./reference/cli.md#run-prompt-tuning)(영문)에 있습니다.

```bash
run-prompt-tuning --source-directory ./source --output-path tuned_prompts.yaml --config-path config.yaml
```

출력에는 `custom_prompts` 블록과, 감지한 도메인과 엔티티 유형이 담긴 `profile`이
있습니다. **검토한 뒤** 원하는 프롬프트를 `custom_prompts:` 아래에 복사하세요.

- 튜닝한 `graph_extraction_system`과 `community_report_system`은 도메인 머리말 뒤에
  내장 규칙과 출력 형식을 그대로 붙인 것이며, 생성한 퓨샷 예시는 마지막에 옵니다.
- 엔티티 범주는 여전히 `processing.graph_extraction.entity_types`(`{entity_types}`)에서
  오므로 프로필의 `entity_types`를 그곳에 복사하세요.
- 퓨샷 예시는 프로필의 엔티티 유형으로 추출하므로 프로필이 찾은 유형을 보여 줍니다.
  설정이 엔티티 유형이나 패턴을 직접 선언하면(인라인 또는 `schema_source`. 튜닝도 수집처럼
  스키마 소스를 읽으며, 읽지 못하면 종료 코드 2로 끝남) 그 스키마를 따릅니다. 이때 예시와
  튜닝한 프롬프트가 안내하는 엔티티 유형은 설정한 유형입니다.
- `--schema-out schema.yaml`을 주면 [추출 스키마](#c-추출-스키마) 초안도 씁니다. 초안에는
  예시를 추출한 엔티티 유형(위 항목), 튜닝 예시가 가장 자주 쓴 관계 유형(최대 12개,
  개수는 주석), 그리고 어떤 관계 유형의 끝점 유형 쌍이 모두 두 번 이상 나왔을 때만 그
  패턴이 들어 있습니다. 설정이 선언한 유형과 패턴은 설명, IRI와 함께 모두 남고, 설정에
  패턴이 있는 관계 유형에는 다른 패턴을 더하지 않습니다. `schema_source: {kind: file}`
  파일이므로 검토한 뒤 `schema_mode: align`으로 읽으세요(주석의 경로는 `--config-path`
  파일의 디렉터리 기준 상대 경로). 모델 답이 같으면 같은 파일이 나옵니다. 사용 중인 설정과
  스키마는 덮어쓰지 않습니다. `--config-path` 파일이나 설정이 읽는 스키마 파일을
  `--schema-out`이나 `--output-path`로 지정하면 `--force`를 주어도 종료 코드 2로 끝나고,
  두 옵션 중 하나가 이미 있는 파일을 가리켜도 `--force`가 없으면 종료 코드 2로 끝납니다.
- 텍스트가 아닌 형식은 수집과 같은 로더(`ParserFactory.register_loader`로 추가한 형식
  포함)를 거치며, 파싱하지 못한 파일은 경고와 함께 건너뜁니다.
- 모델이 JSON 프로필을 돌려주지 않으면 0이 아닌 코드로 끝나고 아무것도 쓰지 않습니다.

---

## 9. 도메인 적응

파이프라인을 의료, 법률, 금융 같은 도메인에 맞추는 방법은 엔티티 유형(A), 프롬프트(B),
추출 스키마(C) 세 가지입니다. 이미 가지고 있는 지식 그래프는 시드 그래프(D)로 추출한
그래프에 합칩니다.

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
씀. 재정의 키가 없는 프롬프트 다섯 개는 이 절 끝에
나옴). `{braces}` 안의 변수는 프레임워크가 채우므로 그대로 두세요.

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
모델. JSON 판정 형식은 유지하고, `evaluation.pairwise_overall`이면 `overall` 판정도 포함),
`answer_span_extraction`(`evaluation.answer_span: extracted`의 구간 추출기), 그리고 프롬프트
튜닝의 `corpus_profile`.

재정의는 설정을 읽을 때 검사하며, 잘못되면 해당 키를 알리고 읽기에 실패합니다.

- 프롬프트 호출에 쓰이지 않는 `{name}`은 오류입니다(호출할 때마다 `KeyError`가 남).
  중괄호를 글자 그대로 쓰려면 `{{` / `}}`로 씁니다: `{{"name": "Vendor"}}`.
  `run-prompt-tuning` 출력은 이미 이스케이프되어 있습니다.
- 각 프롬프트의 데이터 변수는 system이나 human 템플릿에 있어야 합니다
  (`graph_extraction`과 `claim_extraction`은 `{input_text}`, `answer_generation`은
  `{query}`와 `{context}` 등). `{max_entities_per_chunk}` 같은 한도와
  `{entity_types}`, `{relationship_types}`(선언한 관계 유형, 없으면 내장 예시)는
  빼도 됩니다.

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

지능형 청커(`TextChunkingPrompt`), 문서 번역(`TextTranslationPrompt`), 질의 번역
(`TranslationPrompt`), 전역 검색의 커뮤니티 평가(`CommunityRelevancePrompt`)와 축약 단계
(`MapReduceSummaryPrompt`)의 프롬프트를 뺀 모든 프롬프트에는 `custom_prompts.*_system` /
`*_human` 재정의 키가 있습니다(기본값 `null` = `unified_kg_rag/domain/prompts/`의 내장
프롬프트). 이 다섯 개는 재정의할 수 없습니다. 필요한 것만 재정의하고
나머지는 `null`로 두세요. 재정의는 설정을 읽을 때 검사합니다. 알
수 없는 `{variables}`와 빠진 데이터 변수(`{input_text}` 등)는 오류이며, 중괄호를
글자 그대로 쓰려면 `{{` / `}}`로 씁니다.

### C. 추출 스키마

엔티티 유형, 관계 유형, 엔티티 유형 사이에 허용되는 패턴이 추출 스키마를 이룹니다.
기본값에서는 `entity_types` 목록만 있고 추출은 전과 똑같이 동작합니다. 그래프가 정해진
어휘 하나만 쓰게 하려면 관계 유형을(원하면 패턴과 온톨로지 IRI도) 선언하세요. 그러면
`works for`, `Works_For`, `WORKS_FOR`로 쓴 같은 관계가 하나의 에지 레이블이 되고,
저장되는 항목마다 스키마에 맞는지가 기록됩니다.

**모드 고르기.** `schema_mode`는 추출이 스키마로 무엇을 할지 정합니다.

| 모드 | 남기는 것 | 이럴 때 씁니다 |
|---|---|---|
| `off` | 전부, 쓴 그대로 | 스키마로 프롬프트만 바꾸고 싶을 때 |
| `align`(기본값) | 전부, 선언한 유형은 선언한 표기로 | 레이블을 일관되게 하고 싶고, 스키마 밖 항목을 두거나 검토해도 될 때 |
| `strict` | 선언한 유형과 허용한 패턴의 항목만 | 그래프를 쓰는 쪽이 온톨로지 유형만 받을 때 |

`strict`는 재현율을 희생합니다. 모델이 스키마 밖 유형으로 적은 것은 사라지고,
엔티티가 사라지면 그 단위에서 이 엔티티를 가리키는 관계도 함께 사라집니다. `align`으로
시작해 스키마 밖으로 떨어지는 항목을 살펴보고(아래 `filters={"in_schema": False}`),
스키마를 넓힌 다음, 밖으로 떨어지는 것이 적을 때만 `strict`로 바꾸세요. 내 질문에서
비용이 얼마인지 재려면 코퍼스를 모드별로 다른 인덱스 접미사([인덱스 접미사](#인덱스-접미사))에
수집하고 같은 `run-eval` 데이터셋을 각각에 실행하세요.

**스키마 선언.** `processing.graph_extraction`에 직접 씁니다.

```yaml
processing:
  graph_extraction:
    entity_types:
      - "PERSON: People"
      - {label: ORGANIZATION, description: "Companies", iri: "https://schema.org/Organization"}
    relationship_types:
      - "WORKS_FOR: Employment"
      - {label: PARTNERS_WITH, iri: "https://example.org/onto#partnersWith"}
    relationship_patterns:
      - "PERSON -WORKS_FOR-> ORGANIZATION"
      - {source_type: ORGANIZATION, relation: PARTNERS_WITH, target_type: ORGANIZATION}
    schema_mode: align     # off | align(기본값) | strict
```

- 유형은 `"LABEL: description"`(설명은 생략 가능) 또는 `{label, description, iri}`
  매핑입니다. `iri`는 그 유형이 대응하는 온톨로지 클래스(엔티티 유형)나 속성(관계
  유형)입니다. 패턴은 `"SOURCE -RELATION-> TARGET"` 또는
  `{source_type, relation, target_type}` 매핑이며 선언한 유형만 가리켜야 합니다.
  매핑 키를 잘못 쓰면 설정을 읽지 못합니다.
- 레이블은 대소문자, 앞뒤 공백, 단어 사이의 공백·`_` 차이만 무시하고 비교하며
  (`works for` = `WORKS_FOR`) 유사도로는 맞추지 않습니다. `PERSONS`는 `PERSON`이
  아닙니다. 같은 종류의 레이블 두 개가 서로 일치하면 안 됩니다.
- 환경 변수에서는 목록을 JSON으로 씁니다. 예:
  `UKR__PROCESSING__GRAPH_EXTRACTION__RELATIONSHIP_TYPES='["WORKS_FOR: Employment"]'`,
  `UKR__PROCESSING__GRAPH_EXTRACTION__SCHEMA_MODE=strict`
  ([환경 변수로 덮어쓰기](./guides/configure.ko.md#환경-변수로-덮어쓰기)).
- 스키마를 파일로 둘 수도 있습니다. `schema_source: {kind: file, path: schema.yaml}`로
  두면 같은 모양의 YAML 또는 JSON 파일에서 `relationship_types`,
  `relationship_patterns`와 선택적으로 `entity_types`를 읽습니다(상대 경로는 설정 파일
  기준). 이때 설정에서는 그 키를 빼세요. 파일과 설정이 모두 `entity_types`를 정하면
  설정을 읽지 못합니다. 경로는 로컬 파일(`file://` URL도 됨)이나 S3 객체(`s3://bucket/schemas/schema.yaml`,
  최대 16 MiB)이며, S3는 실행의 AWS 세션으로 읽습니다. `https://` 위치는 그 위치용
  스키마 소스를 등록해야 하므로 설정을 읽지 못합니다. `kind: owl`도 같은 방식으로
  OWL 또는 RDFS 온톨로지를 읽습니다. 로컬 파일이나 디렉터리, S3 객체, 또는 디렉터리로
  읽는 S3 접두사(`s3://bucket/ontology/`, 끝의 `/` 필수)를 줄 수 있습니다. 상대 IRI를
  쓰는 온톨로지에는 `@base`(또는 `xml:base`)를 선언하세요. 선언하지 않으면 IRI가 파일
  위치를 기준으로 정해져 파일을 옮기면 이름이 바뀝니다
  ([OWL/RDFS 온톨로지에서 읽기](#owlrdfs-온톨로지에서-읽기)). 다른 형식(SHACL, 스키마
  서비스)은 스키마 소스로 등록해 붙입니다([설계 문서 §15](./design.md#15-extension-guide),
  영문).
- **AWS에서는** 수집 작업이 이미지에 들어 있는 설정을 읽습니다(`docker/Dockerfile`은
  `docker/config.yaml`만 `/app/config.yaml`로 복사). 따라서 로컬 `kind: file` 스키마(또는
  `kind: owl` 온톨로지)도 이미지에 넣어야 합니다. 파일을 설정 옆에 복사하는 `COPY`를 추가해 이미지를 다시
  빌드하거나, 스키마를 S3에 두거나(작업 역할 권한은
  [운영 가이드](./operations.md#schema-sources-in-s3), 영문), 그 설정에 인라인으로
  선언하거나, 위의 `UKR__PROCESSING__GRAPH_EXTRACTION__*` 환경 변수로 지정하세요.

**추출이 하는 일.**

- **프롬프트.** 모든 모드에서 선언한 관계 유형과 패턴이 추출 프롬프트의 관계 유형
  예시를 대신합니다. 관계 유형을 하나도 선언하지 않으면 프롬프트는 예시를 그대로
  씁니다.
- **`align`**은 엔티티와 관계를 모두 남깁니다. 선언한 유형을 가리키는 유형은 id를
  만들기 전에 선언한 표기로 바꿔 쓰므로(`works for`는 `WORKS_FOR`), 여러 표기가
  하나의 에지 레이블과 하나의 관계 id가 됩니다. 스키마나 패턴을 벗어난 항목은 개수만
  셉니다. 엔티티 유형만 있으면(기본 설정처럼 관계 유형이 없으면) 레이블은 쓴
  그대로입니다.
- **`strict`**는 같은 일을 하고 텍스트 단위마다 다음을 버립니다.
  - 유형이 비었거나 선언되지 않은 엔티티와, 그 단위에서 이 엔티티를 가리키는 관계.
    단위가 거부를 기록하므로 글리닝이 관계를 통해 이 엔티티를 되살리지 못합니다.
    같은 엔티티를 선언한 유형으로도 언급하면 남습니다.
  - 선언되지 않은 유형의 관계.
  - 단위가 양 끝 엔티티를 모두 선언한 유형으로 나열했는데 패턴이 허용하지 않는 관계.
    단위가 나열하지 않은 끝점은 통과하고, 방향이 거꾸로인 관계는 뒤집지 않고
    버립니다. 패턴이 없는 관계 유형은 제한하지 않습니다.

  글리닝도 같은 규칙을 따릅니다. 스키마를 벗어난 엔티티나 관계를 더하지 않고,
  선언되지 않은 유형이나 패턴이 허용하지 않는 방향으로 바꾸는 수정은 무시합니다.
- 끝점의 유형은 단위 안에서 그 엔티티에 병합된 유형(가장 많이 나온 유형)입니다.
  선언된 유형이라고 우선하지는 않습니다. `align`에서 `VENDOR`로 두 번,
  `ORGANIZATION`으로 한 번 언급된 엔티티는 `vendor`로 남습니다.
- 버린 것은 실패가 아닙니다. 항목을 모두 버린 텍스트 단위도 (빈 결과로) 추출된
  것이며 성공으로 세고 다시 시도하지 않습니다.
- 클레임 자체는 스키마로 검사하지 않지만, 클레임의 주어는 그 텍스트 단위가 언급한
  엔티티여야 합니다. 그래서 `strict`에서 그 단위에서 스키마 밖이라 버린 엔티티에 대한
  클레임은 함께 버려집니다. 그래프에 없는 목적어는 스키마가 없을 때처럼 클레임의 값으로
  남습니다.
- 그래프 추출 단계는 `extraction_stats`에 `entities_off_schema`,
  `entities_dropped_off_schema`, `relationships_off_schema`,
  `relationships_dropped_off_schema`, `relationships_dropped_off_schema_endpoint`,
  `relationships_off_pattern`, `relationships_dropped_off_pattern`을 보고합니다.
  글리닝 단계는 글리닝이 제안한 항목에 대해 같은 카운터를(`align`에서는 남기고 세며,
  `strict`에서는 버림) `corrections_ignored_off_schema`와 함께 `gleaning_stats`에
  보고합니다. 둘 다 `<cache directory>/<pipeline_id>/pipeline_metadata.json`의 단계
  지표에 있고, 단계마다 요약 로그를 한 줄 남깁니다. 스키마를 벗어난 레이블 자체는
  DEBUG 로그에만 남깁니다.
- 선언되지 않은 유형의 관계는 끝점과 상관없이 `relationships_off_schema`에 세므로,
  이 개수는 `align`과 `strict`에서 같습니다. strict에서 버린 관계는 한 번만 셉니다.
  선언되지 않은 유형을 먼저, 그다음 스키마 밖이라 버린 끝점
  (`relationships_dropped_off_schema_endpoint`), 그다음 패턴 순입니다. 카운터는
  텍스트 단위별 언급을 세고, `in_schema: false` 목록은 병합된 항목(엔티티나 관계마다
  하나)을 보여 줍니다.

**저장되는 것.** 인덱싱은 그래프 저장소와 벡터 저장소의 엔티티와 관계에 다음 필드를
씁니다.

| 필드 | 대상 | 값 | 쓰는 때 |
|---|---|---|---|
| `in_schema` | 엔티티, 관계 | 저장된 유형이 그 종류의 선언한 유형인지(빈 유형은 아님). 선언한 유형이 없는 종류의 항목은 모두 `true`(그 종류에는 스키마 밖 항목이 없음) | `strict`, 그리고 관계 유형을 선언한 `align`. 두 종류 모두에 씀 |
| `type_iri` | 엔티티 | 저장된 유형이 가리키는 선언한 엔티티 유형의 `iri` | 그 유형에 `iri`가 있고 모드가 `off`가 아니면 언제나(엔티티 유형만 있어도) |
| `predicate_iri` | 관계 | 선언한 관계 유형의 `iri` | `type_iri`와 같음 |

- 값은 저장된(병합된) 유형을 따르므로 전체 빌드와 증분 실행이 같은 값을 저장합니다.
  관계가 패턴에 맞는지는 저장하지 않습니다. 끝점 유형에 달려 있고, 다른 문서가 끝점
  유형을 바꿀 수 있기 때문입니다.
- 관계의 끝점으로만 나온 엔티티는 유형이 없으므로, 엔티티 유형을 선언했다면
  `strict`에서도 스키마 밖입니다.
- 해당하지 않는 필드는 아예 쓰지 않으므로, 관계 유형도 IRI도 없는 설정(기본값 포함)은
  이 필드들 없이 항목을 저장합니다. 이 필드 때문에 id가 바뀌는 일은
  없습니다.

**필터.** 이 필드는 다른 필드처럼 엔티티·관계 인덱스와 Neptune 엔티티 정점(그래프
확장)의 필터 키입니다([필터](#필터) 참고).

```python
RAGInput(query="...", search_strategy="local", filters={"in_schema": False})
RAGInput(query="...", search_strategy="mix",
         filters={"type_iri": "https://schema.org/Organization"})
```

첫 번째는 스키마 밖 엔티티만(`mix`, `hybrid`, `local`, `drift`에서는 스키마 밖 관계도)
돌려주므로, `align`이 스키마 밖으로 남긴 항목을 검토하는 목록이 됩니다. 필드 없이
저장된 항목은 그 필드의 어떤 값과도 일치하지 않고, 아직 그 필드를 매핑하지 않은 인덱스도
마찬가지입니다(다음 수집 실행이 매핑합니다). 이 때문에 실패하는 쿼리는 없습니다.

설정한 스키마가 어떤 항목에도 쓰지 않는 필드로 거르면 결과를 비우기만 하므로, 모델을
호출하기 전에 거부합니다(`InvalidFilterError`). `off`이거나 엔티티 유형만 있는
`align`에서의 `in_schema`, `iri`를 선언한 엔티티 유형이 없을 때(또는 모드가 `off`일
때)의 `type_iri`, 관계 유형에 대해 같은 경우의 `predicate_iri`가 그렇습니다. 이 검사는
스키마 소스를 포함해 쿼리의 설정을 읽습니다. 소스를 읽을 수 없으면 키를 검사하지 않고
경고를 한 번 남깁니다.

**스키마 바꾸기.** 다음 증분 실행이 무엇을 하는지는 모델이 그 변경을 보는지에 달려
있습니다.

- **그 범위를 다시 추출**: 프롬프트나 스키마 검사가 읽는 변경입니다. 유형의 레이블이나
  설명, 패턴, `strict` 켜고 끄기, 관계 유형을 선언한 상태에서 `off`와 `align` 사이의
  전환이 여기에 해당합니다. 다른 인덱스 범위의 항목은 그 범위가 다시 실행되거나
  `indexing.reset: true`로 코퍼스를 다시 빌드할 때까지 각자의 실행이 추출한 내용(모델이
  쓴 그대로의 레이블, `strict`라면 버렸을 스키마 밖 항목)을 유지합니다.
- **표시만 다시 씀**: IRI를 더하거나 바꾸거나 지울 때, 엔티티 유형만 있는 상태에서
  `off`와 `align` 사이를 전환할 때, 같은 항목을 다른 형식으로 쓸 때(`"GENE: A gene"`와
  `{label: GENE, description: A gene}`, 문자열 패턴과 매핑 패턴)입니다. 다시 추출하거나
  임베딩하지 않습니다. 실행은 인덱스 접미사의 모든 범위에 저장된 엔티티와 관계의
  `in_schema`, `type_iri`, `predicate_iri`를 저장된 유형으로 다시 계산해 쓰고, 개수를
  INFO 로그 한 줄로 남깁니다. 그러면 저장소에는 전체 빌드가 쓰는 것과 같은 내용이
  남습니다. 레지스트리는 인덱스 네임스페이스마다 저장소가 어떤 표시를 담고 있는지를 모든
  범위에 대해 한 번만 기록합니다. 그래서 어느 범위의 실행이 스키마 변경을 발견하든 다시
  쓰기는 네임스페이스마다 한 번이고, 다른 범위의 실행 뒤에 한 범위에서 스키마를 되돌려도
  다시 씁니다. 표시가 기록된 네임스페이스에 새 범위를 더하면 아무것도 읽지 않습니다. 이
  기록이 없는 네임스페이스는 한 번 읽습니다. 저장소는 크기가 제한된 페이지로
  읽습니다. 다시 쓰기가 끝나지 않으면(실행 중단, 쓰기 실패, 저장소 읽기 실패) 경고를
  남기고 실행은 계속되며, 다음 실행이 어떤 스키마로든 다시 씁니다. 항목 목록을 읽지 못하는
  사용자 정의 저장소 어댑터는 경고와 함께 건너뛰며, `indexing.reset: true`가 그 저장소도
  다시 씁니다.
- **아무것도 바뀌지 않음**: 스키마 파일을 고치지 않고 옮기기만 할 때입니다(경로가 아니라
  내용으로 키를 정함). 기본값이거나 IRI 없이 문자열로 쓴 엔티티 유형만 있으면
  업그레이드해도 캐시 키, 추출 레코드, 저장된 항목이 그대로입니다. 증분 실행은 첫 쓰기
  전에 기존 OpenSearch 인덱스에 새 필드를 매핑합니다.

**프롬프트 튜닝으로 스키마 초안 만들기.** `run-prompt-tuning --schema-out schema.yaml`은
스키마 파일 초안도 씁니다([§8](#8-프롬프트-튜닝-run-prompt-tuning) 참고).

1. 코퍼스를 대표하는 표본으로 실행합니다.
2. 초안을 검토합니다. 초안에는 프로필의 엔티티 유형(설정이 선언하면 설정한 유형)과,
   튜닝 예시가 가장 자주 쓴 관계 유형이 각각의 개수와 함께 있고, 패턴은 예시가
   뒷받침할 때만 있습니다. 동의어를 합치고, 잡음을 지우고, 설명과 IRI를 더하세요.
   설정이 이미 읽는 스키마를 다듬을 때는 초안을 다른 파일에 쓰세요(명령은 읽는 파일을
   덮어쓰지 않음). 초안에는 선언한 유형과 패턴이 설명, IRI와 함께 모두 남으므로 초안이
   더한 것만 합치면 됩니다.
3. `schema_source: {kind: file, path: schema.yaml}`과 `schema_mode: align`으로
   읽습니다(초안은 청크 몇 개에서 나오므로 `strict`는 그 청크에 없던 것을 버림). 초안이
   `entity_types`를 정하면 설정에서 `entity_types`를 빼세요.
4. 수집한 뒤 `filters={"in_schema": False}`와 단계 카운터로 스키마 밖 항목을 검토하고,
   파일을 넓혀 다시 실행합니다.

**로컬 백엔드.** `backend: local`도 똑같이 동작합니다. SQLite 그래프 저장소와 내장
벡터 저장소가 같은 필드를 저장하고, 같은 필터가 (그래프 확장 포함) 적용되며, 기존
내장 인덱스도 같은 방식으로 새 필드를 얻습니다.

#### OWL/RDFS 온톨로지에서 읽기

`schema_source: {kind: owl}`은 OWL 또는 RDFS 온톨로지로 스키마를 채웁니다. 클래스는
엔티티 유형, 객체 속성(object property)은 관계 유형, 정의역(domain)과 치역(range)은
패턴이 됩니다. 유형마다 IRI가 붙으므로 `type_iri`와 `predicate_iri`가 온톨로지의 용어를
가리킵니다. 매핑은 추론하지 않고 옮기기만 합니다. 정해진 RDFS 방식 규칙을 시작할 때 한
번 적용하며, 모델도 OWL 추론기(reasoner)도 호출하지 않습니다.

**설치.** 이 소스에는 `ontology` extra(rdflib)가 필요합니다.

```bash
uv sync --extra ontology    # 또는: pip install -e '.[ontology]'
```

extra가 없으면 이런 설정으로 실행한 CLI(`run-ingestion`, `run-rag`, `run-eval`,
`run-prompt-tuning`)는 모델을 호출하기 전에 extra 이름을 알려 주고 종료 코드 2로
끝납니다. 쿼리도 필터 키를 검사하려고 스키마를 읽으므로 `run-rag`에도 필요합니다.
배포용 이미지에는 이 extra가 빠져 있습니다. `--build-arg UV_EXTRAS="--extra ontology"`로
빌드하고 온톨로지를 이미지 안 설정 파일 옆에 복사하거나
([운영 런북: Ontology schema in the image](./operations.md#ontology-schema-in-the-image), 영문),
S3에 두세요([Schema sources in S3](./operations.md#schema-sources-in-s3), 영문).

**설정.**

```yaml
processing:
  graph_extraction:
    schema_source:
      kind: owl
      path: ontology/procure.ttl   # 파일 또는 온톨로지 파일이 든 디렉터리
    schema_mode: align
```

- 설정에서 `entity_types`는 빼세요. 온톨로지가 정합니다.
- `path`는 로컬 파일이나 디렉터리이며 설정 파일 기준 상대 경로입니다(`file://` URL도
  됨). `s3://` 객체나 디렉터리로 읽는 `s3://bucket/prefix/`도 됩니다
  ([Schema sources in S3](./operations.md#schema-sources-in-s3), 영문). 디렉터리는 하위까지 읽어 온톨로지 하나로 다룹니다(파일은 정렬하고 `.`으로
  시작하는 이름과 Protégé의 `catalog-v001.xml` 같은 OASIS XML 카탈로그는 건너뜀). 그 밖의 URL이나 `secret_arn`을 주면 설정을 읽지 못합니다.
- 형식은 확장자로 정합니다. `.ttl`은 Turtle, `.rdf` / `.owl` / `.xml`은 RDF/XML,
  `.jsonld` / `.json-ld`는 JSON-LD, `.nt`는 N-Triples, `.nq`는 N-Quads, `.trig`는
  TriG입니다. 디렉터리의 다른 파일은 건너뜁니다. `options.format`을 주면 확장자와
  상관없이 모든 파일을 그 형식으로 읽습니다(디렉터리도 마찬가지이므로 온톨로지 파일만
  두세요).

**옵션**은 `schema_source.options` 아래에 둡니다. 모르는 옵션이나 잘못된 값은
`schema_source.options.<name>`을 알려 주며 설정을 읽지 못합니다. 용어는 전체 IRI나
CURIE(`ex:Vendor`)로 씁니다. CURIE 접두사는 `prefixes`, 잘 알려진 `rdf`, `rdfs`,
`owl`, `xsd`, `skos`, `schema`, `dcterms`, 온톨로지 자체의 접두사 선언 순으로 찾습니다
(두 네임스페이스에 묶인 접두사는 `prefixes`에 정해야 함). 온톨로지에 없는 용어를 쓰면
설정을 읽지 못하므로, 오타 때문에 아무것도 선택되지 않은 채 지나가는 일은 없습니다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `namespaces` | `[]`: RDF, RDFS, OWL, XSD, SKOS, Dublin Core를 뺀 모든 네임스페이스 | 매핑할 클래스와 속성의 IRI 접두사 |
| `classes` | `[]`: 모든 클래스 | 루트 클래스. 각각 하위 클래스와 함께 매핑 |
| `exclude_classes` | `[]` | 하위 클래스와 함께 뺄 클래스 |
| `max_depth` | `null`: 모든 단계 | 루트 아래로 매핑할 하위 클래스 단계 수(`0`: 루트만. `classes`가 없으면 최상위 클래스가 루트) |
| `properties` | `[]`: 모든 객체 속성 | 루트 속성. 각각 하위 속성과 함께 매핑 |
| `exclude_properties` | `[]` | 하위 속성과 함께 뺄 속성 |
| `include_deprecated` | `false` | `owl:deprecated true`인 용어도 매핑 |
| `prefixes` | `{}` | 이 옵션들의 CURIE에 쓸 접두사 → 네임스페이스 IRI |
| `label_source` | `auto` | `auto`: `skos:prefLabel`, 다음 `rdfs:label`. `label`: `rdfs:label`만. `local_name`: IRI의 로컬 이름만. 레이블이 없는 용어는 로컬 이름을 씀 |
| `label_case` | `upper_snake` | `upper_snake`(`supplyChain` → `SUPPLY_CHAIN`, `공급 업체` → `공급_업체`) 또는 `preserve`(쓴 그대로, 문장 부호는 공백으로) |
| `languages` | `[en, ""]` | 레이블과 설명에서 선호하는 언어 태그, 순서대로(`""`: 태그 없음. `en`은 정확히 `en`인 값 다음으로 `en-US`도 맞춤). 그 밖의 태그는 이 뒤 |
| `labels` | `{}` | IRI 또는 CURIE → 쓸 레이블(쓴 그대로) |
| `on_collision` | `error` | 같은 종류의 두 용어가 같은 레이블일 때 실패하거나, `qualify`면 각 레이블 뒤에 네임스페이스 접두사를 붙임(`BANK_EX`, `BANK_GEO`). 접두사는 `prefixes`가 묶은 것, 없으면 잘 알려진 접두사, 없으면 파일이 묶은 것, 그것도 없으면 네임스페이스의 짧은 해시. 충돌하는 네임스페이스는 `prefixes`에 고정하세요([*바꾸면 다시 실행되는 것*](#ontology-what-a-change-re-runs) 참고) |
| `description_properties` | `[skos:definition, rdfs:comment]` | 설명을 읽을 주석 속성. 선호 언어에서 처음 찾은 값. 기본값 외에 적은 속성은 온톨로지의 속성이거나 어떤 용어의 리터럴 주석이어야 하며, 아니면 읽기가 실패함 |
| `max_description_chars` | `300` | 더 긴 설명은 단어 경계에서 잘라 "…"를 붙임(`0`: 온톨로지에서 설명을 가져오지 않음) |
| `hierarchy_hints` | `true` | 설명 뒤에 "Kind of X.", "Narrower than P.", "Inverse of Q."를 덧붙임 |
| `patterns` | `closure` | `closure`: 정의역과 치역에서 패턴을 만듦. `none`: 패턴 없음 |
| `max_entity_types` | `64` | 선택된 클래스가 더 많으면 실패 |
| `max_relationship_types` | `64` | 매핑된 속성이 더 많으면 실패 |
| `max_patterns` | `256` | 패턴이 더 많이 나오면 실패 |
| `format` | `auto` | `auto`(확장자로), `turtle`, `xml`, `json-ld`, `nt`, `nquads`, `trig` |
| `max_bytes` | `67108864`(64 MiB) | 파일 전체 크기가 더 크면 실패([*읽기 비용*](#ontology-load-cost) 참고) |

**읽는 것.** 파일에 있는 문장 중 이름 있는 용어만 읽습니다.

- 클래스: `owl:Class`나 `rdfs:Class`로 선언한 것, `rdfs:subClassOf`와
  `owl:equivalentClass`의 양 끝, 관계 유형의 정의역이나 치역에 나온 클래스. 데이터 타입
  속성의 정의역에만 나온 클래스는 클래스로 읽지 않습니다.
- 상위 클래스: 클래스의 이름 있는 `rdfs:subClassOf` 대상, 그리고 클래스가 하위이거나
  동치인 `owl:intersectionOf`의 이름 있는 구성원입니다(`ex:Vendor owl:equivalentClass
  [owl:intersectionOf (ex:Organization [a owl:Restriction ...])]`이면 `ex:Vendor`는
  `ex:Organization` 아래). `owl:unionOf`와 동치인 클래스는 그 합집합을 덮으므로, 합집합의
  이름 있는 구성원이 그 클래스 아래에 놓입니다.
- 관계 유형: 객체 속성(`owl:ObjectProperty`나 `owl:SymmetricProperty` 같은 그 특성,
  `owl:inverseOf`의 양 끝, 객체 속성의 하위·상위·동치 속성), 리터럴 유형이 아닌 치역이
  있는 `rdf:Property`, 그런 `rdf:Property`의 하위·상위·동치 속성. 리터럴 유형은 XSD 유형,
  `rdfs:Literal`, `rdf:langString` 같은 RDF 리터럴 유형, `rdfs:Datatype`, 그리고
  schema.org 데이터 유형(`http`와 `https`의 `schema:Text`, `URL`, `Number`, `Integer`,
  `Float`, `Boolean`, `Date`, `DateTime`, `Time`, `CssSelectorType`,
  `PronounceableText`, `XPathType`, `schema:DataType`으로 선언한 것)이며 엔터티 유형이
  되지 않습니다.
- 빼는 것: 데이터 타입 속성(`owl:DatatypeProperty`, 또는 치역이 모두 리터럴 유형인
  `rdf:Property`)과 주석 속성(건너뛴 것으로 보고), `owl:FunctionalProperty`로만 선언했거나
  정의역·치역만 있는 속성(INFO 로그 `N properties not read ...`가 이름을 알려 줌),
  개체(individual), RDF·RDFS·OWL·XSD 어휘, 위의 클래스 식을 뺀 익명 클래스. 레이블과
  설명은 문자열 리터럴만 읽습니다.

**매핑 규칙.**

- **폐포(closure).** `rdfs:subClassOf`는 선택하지 않은 클래스를 포함해 온톨로지 전체에서
  추이적입니다. `subClassOf` 순환이나 `owl:equivalentClass`는 유형 하나가 되고, 그
  `iri`는 선택된 IRI 중 가장 작은 것입니다(속성도 `owl:equivalentProperty`로 같음).
  속성의 정의역은 자신의 `rdfs:domain` 문장, 상위 속성의 정의역, 역속성의 치역이 모두
  허용하는 것이고, 치역은 그 반대입니다. 문장 하나는 클래스, 클래스의 `owl:unionOf`,
  또는 그 속성의 `schema:domainIncludes` 전체이며, 그 안의 클래스 중 어느 것이든
  허용합니다. `owl:Thing`과 `rdfs:Resource`는 무엇이든 허용합니다. 대칭 속성(또는
  자기 자신이 역속성인 속성)은 정의역과 치역을 양쪽 끝에서 모두 허용합니다. 한쪽
  끝에만 문장이 있으면 다른 쪽도 같은 문장을 쓰고, 하위 속성과 역속성은 양쪽을 모두
  물려받습니다. 이 판단은 폐포 전체를 본 뒤에 하므로, 역속성이 채워 주는 치역을
  덮어쓰지 않습니다. 추이성, 함수성, 카디널리티, 제약(restriction)은 무시합니다.
- **선택.** 먼저 네임스페이스와 폐기 여부를 보고, `classes` 하위 트리를 `max_depth`까지
  고른 뒤 `exclude_classes` 하위 트리를 뺍니다. 속성도 같은 방식입니다. 클래스를 골라도
  그 상위 클래스는 선택되지 않지만, 폐포는 선택하지 않은 클래스를 거쳐 계산합니다.
- **평평한 계층, 내려보내기와 올려보내기.** 선택된 클래스는 모두 독립된 유형이며, 추출할
  때 다른 유형의 패턴을 물려받지 않습니다. 대신 패턴을 클래스마다 풀어 씁니다. 속성의
  출발 유형은 모든 정의역 문장이 허용하는 클래스이거나 그 하위인 선택된 클래스입니다
  (내려보내기, push-down). 그런 클래스가 없으면 허용된 클래스의 가장 가까운 선택된
  상위 클래스입니다(올려보내기, lift). 둘 다 없으면 그 속성은 빠지고 로그에 이유가
  남습니다. 도착 유형도 치역으로 같은 방식으로 정하고, 패턴은 모든 출발 유형과 도착
  유형의 조합입니다. 정의역도 치역도 없는 속성은 패턴이 없으므로 `strict`에서도 제한되지
  않습니다.
- **역속성과 대칭.** 역속성 한 쌍은 관계 유형 두 개이고, 각자 자기 방향의 패턴을
  가집니다. 방향을 뒤집지 않습니다. 그래프에 한 방향만 두려면 `exclude_properties`로
  하나를 빼세요. 클래스 n개 사이의 대칭 속성은 n × n개의 패턴을 만듭니다.
- **순서.** 유형은 선택된 계층을 깊이 우선으로 돌며 정하고, 형제는 레이블 다음 IRI
  순입니다. 패턴은 관계 유형, 출발 유형, 도착 유형 순으로 정렬합니다. 파일이나 직렬화
  형식이 아니라 문장만으로 정해집니다.

**계층과 저장되는 표시.** 유형이 평평하므로 다음과 같습니다.

- `in_schema`는 저장된 유형이 선택된 클래스인 엔티티에서 참입니다.
  `classes: ["ex:Organization"]`과 `max_depth: 0`이면 모델이 `VENDOR`로 적은 엔티티는
  스키마 밖입니다(`strict`에서는 버림).
- `type_iri`는 저장된 유형이 가리키는 클래스, 즉 모델이 적은 클래스의 IRI이며 상위
  클래스가 아닙니다. `VENDOR`로 저장된 엔티티의 `type_iri`는
  `https://example.org/procure#Vendor`이고 `#Organization`이 아닙니다. 하위 트리 전체를
  가져오려면 그 IRI 목록으로 거르세요.

  ```python
  EX = "https://example.org/procure#"
  RAGInput(query="...", search_strategy="local",
           filters={"type_iri": [EX + "Organization", EX + "Vendor", EX + "Buyer"]})
  ```

  명령줄로는 이렇게 거를 수 없습니다. `run-rag --filters`는 키 하나에 값 하나만,
  `set-filter`도 값 하나만 받고, 쉼표는 값의 일부로 읽습니다. 목록이 필요하면
  `RAGInput.filters`나 `run-eval` 데이터셋 항목의 `filters`를 쓰세요.

- 패턴 검사는 계층을 모릅니다. `strict`에서 `ORGANIZATION` 패턴은 `VENDOR` 끝점을
  허용하지 않으므로, 매핑이 하위 클래스 패턴을 풀어 씁니다. 엔티티에 저장되는 유형은
  여전히 가장 많이 나온 유형입니다. `VENDOR`로 두 번, `ORGANIZATION`으로 한 번
  언급되면 vendor로 남습니다.

**예제.** `tests/fixtures/ontologies/procure.ttl`(`ex:`는
`https://example.org/procure#`)은 `ex:Organization`과 그 하위 클래스 `ex:Vendor`,
`ex:Buyer`, 그리고 `ex:Person`을 선언합니다. 속성은 Organization → Organization인 대칭
속성 `ex:tradesWith`, `tradesWith`의 하위 속성이며 Vendor → Buyer인 `ex:supplies`,
`supplies`의 역속성 `ex:suppliedBy`, Organization → Person인 `ex:employs`, `employs`의
역속성 `ex:worksFor`, 데이터 타입 속성 `ex:name`입니다. 기본 옵션으로는 다음과 같이
매핑됩니다.

| 종류 | 유형(설명) |
|---|---|
| 엔티티 유형 | `ORGANIZATION`; `BUYER`("Kind of ORGANIZATION."); `VENDOR`("An organization that sells goods. Kind of ORGANIZATION."); `PERSON` |
| 관계 유형 | `EMPLOYS`("Inverse of WORKS_FOR."); `SUPPLIED_BY`("Inverse of SUPPLIES."); `TRADES_WITH`; `SUPPLIES`("Narrower than TRADES_WITH. Inverse of SUPPLIED_BY."); `WORKS_FOR`("Inverse of EMPLOYS.") |

패턴은 17개입니다.

- `EMPLOYS`: `ORGANIZATION`, `BUYER`, `VENDOR` → `PERSON`(3개, Organization에서
  내려보냄).
- `SUPPLIED_BY`: `BUYER` → `VENDOR`(1개, 역속성의 정의역과 치역에서).
- `TRADES_WITH`: `ORGANIZATION`, `BUYER`, `VENDOR`의 모든 쌍(9개, 대칭).
- `SUPPLIES`: `VENDOR` → `BUYER`(1개).
- `WORKS_FOR`: `PERSON` → `ORGANIZATION`, `BUYER`, `VENDOR`(3개).

`ex:name`은 건너뜁니다. `strict`에서 `VENDOR -SUPPLIES-> BUYER`는 통과하고
`BUYER -SUPPLIES-> VENDOR`는 버립니다. 선택을 좁히면 다음과 같습니다.

```yaml
    schema_source:
      kind: owl
      path: ontology/procure.ttl
      options:
        classes: ["ex:Organization"]   # ORGANIZATION, BUYER, VENDOR
        max_depth: 0                   # ORGANIZATION만
```

- `classes: ["ex:Organization"]`만 두면 `ORGANIZATION`, `BUYER`, `VENDOR`가 매핑되고,
  `PERSON`이 선택되지 않으므로 `EMPLOYS`와 `WORKS_FOR`는 빠져 패턴이 11개 남습니다.
- `max_depth: 0`까지 두면 `ORGANIZATION`만 남고 `SUPPLIES`와 `SUPPLIED_BY`는
  `ORGANIZATION -> ORGANIZATION`으로 올려보내집니다(패턴 3개). 그래서 `strict`에서도
  두 조직 사이라면 어느 방향이든 허용합니다.

**프롬프트 예산과 상한.** 유형과 패턴은 모든 그래프 추출 호출의 프롬프트에 들어갑니다.
유형 하나는 약 5토큰에 설명이 더해지고(300자 설명은 약 75토큰), 패턴 하나는 약
13토큰입니다. 기본 상한이면 패턴만 최대 약 3,300토큰이고, 300자 설명이 달린 유형
128개는 약 10,000토큰이 더 듭니다. 패턴은 곱으로 늘어납니다. 출발 클래스 m개와 도착
클래스 n개 사이의 속성은 m × n개를 만들므로, 넓은 대칭 속성 하나가 대부분을 차지할 수
있습니다. 상한을 넘는 선택은 잘라 내지 않고 설정을 읽지 못하게 하며(종료 코드 2),
잘라 내면 프롬프트가 임의로 바뀌기 때문입니다. 메시지는 무엇을 좁힐지 알려 주며,
패턴이 넘치면 패턴을 가장 많이 만드는 속성 다섯 개를 보여 줍니다.

```text
schema_source.options: the selection yields 17 patterns, more than max_patterns (10); the most come from https://example.org/procure#tradesWith (9), ... Narrow classes or properties, exclude some, set patterns: none, or raise max_patterns
```

`namespaces`, `classes`, `max_depth`, `exclude_*` 옵션으로 좁히거나, `patterns: none`으로
두거나(유형만 남고 `strict`는 유형만 검사하며 패턴은 검사하지 않음), 프롬프트가 커지는
것을 감안하고 상한을 올리세요.

<a id="ontology-load-cost"></a>**읽기 비용.** 파일은 프로세스마다 한 번, 시작할 때(`run-rag`는 필터를 쓴 첫
질의 때) 메모리로 파싱하며, 이어지는 매핑은 계층이 아무리 깊어도 온톨로지 크기에
거의 비례하는 시간과 메모리를 씁니다. 비용 대부분은 파싱입니다. 클래스 110,000개와
속성 500개인 14 MB Turtle 파일은 약 45초, 최대 메모리 1.2 GB가 들었습니다(파싱 30초,
클래스와 속성 읽기 10초, 매핑 5초). Turtle 1 MB당 약 3초, 85 MB인 셈이며, 깊이
20,000인 `subClassOf` 사슬은 1초 안에 매핑됩니다. 기본 `max_bytes`(64 MiB)라면 약 5 GB로,
기본 8 GiB 태스크에 들어가지만 큰 수집과 함께라면 부족할 수 있습니다. `max_bytes`를
실제로 읽을 크기로 낮추고 태스크 메모리를 그에 맞추거나, 온톨로지 전체 대신 필요한
모듈(그 파일들)만 매핑하세요.

**한국어 레이블.** 한국어로 레이블을 단 온톨로지라면 다음처럼 씁니다.

```yaml
    schema_source:
      kind: owl
      path: ontology/procure.ttl
      options:
        label_source: label        # rdfs:label만
        languages: [ko]            # 한국어 먼저, 다음 태그 없음, 다음 그 밖의 태그
        label_case: preserve       # 레이블을 쓴 그대로
        labels:
          "ex:suppliedBy": 공급받는다  # 온톨로지에 한국어 레이블이 없음
```

예제는 `사람`, `조직`, `공급업체`, `구매자`와 `거래한다`, `공급한다`, `고용한다`,
`공급받는다`, `근무한다`로 매핑되고, 패턴은 같은 17개입니다.

- 한국어 레이블이 없는 용어는 태그 없는 레이블, 다른 언어 레이블, 로컬 이름 순으로
  씁니다. 로컬 이름은 퍼센트 인코딩을 풀므로 한글 IRI는 한글 레이블이 됩니다.
  `preserve`에서는 로컬 이름도 쓴 그대로 남으므로(`suppliedBy`. `upper_snake`라면
  `SUPPLIED_BY`) 이런 용어는 `labels`에 적어 주세요.
- 한글에는 대소문자가 없으므로 한국어 레이블에서 두 방식은 구분자만 다릅니다
  (`공급 업체` → `공급_업체` 또는 `공급 업체`). 레이블 비교는 이 차이를 무시합니다.
  라틴 문자 레이블과 로컬 이름에서는 결과가 달라집니다.
- 설명도 `languages`를 따릅니다(이 예제는 영어 주석 하나뿐이라 그것을 씀). 계층
  힌트는 항상 영어입니다. 예: "Kind of 조직."

<a id="ontology-what-a-change-re-runs"></a>**바꾸면 다시 실행되는 것.** 파일이 아니라 매핑된 스키마로 키를 정하므로, 증분 실행은
매핑 결과에 따라 움직입니다([스키마 바꾸기](#c-추출-스키마)).

- **다시 추출**: 레이블, 설명, 패턴이나 유형 순서가 달라질 때입니다. 선택된 용어의
  레이블이나 주석 수정, 선택에 클래스나 속성이 들어오거나 나가는 경우, 이런 결과를
  바꾸는 옵션 변경(`hierarchy_hints` 포함)이 해당합니다. 첫 전체 수집 전에 옵션을
  정해 두세요. 온톨로지 편집자가 레이블을 자주 바꾼다면 `label_source: local_name`을
  쓰거나 `labels`로 고정하세요.
- **표시만 다시 씀**: 레이블은 그대로인데 IRI만 바뀔 때입니다(`ex:Vendor`를 레이블은
  두고 `ex:Supplier`로 바꾼 경우). IRI가 더 작은 동치 클래스가 생겨 유형의 IRI가 되는
  경우도 같습니다(`ex:Supplier owl:equivalentClass ex:Vendor`를 더하면 `type_iri`가
  `ex:Supplier`가 됨). 단, 레이블이 그대로일 때만입니다. 합쳐진 유형은 구성원 전체 중
  가장 알맞은 레이블을, `label_source: local_name`이거나 레이블이 없으면 가장 작은
  IRI의 로컬 이름을 쓰므로, 자기 레이블이 있는 동치 클래스나 `local_name`에서 IRI가 더
  작은 동치 클래스가 생기면 유형의 레이블이 바뀌어 다시 추출합니다. 그런 유형은
  `labels`로 고정하세요.
- **아무것도 바뀌지 않음**: 다른 형식으로 다시 저장(Turtle에서 RDF/XML로), 파일 이동이나
  이름 변경(절대 IRI일 때, 아래 참고), `owl:versionInfo`, `rdfs:seeAlso`, 데이터 타입
  속성의 레이블이나 주석, 폐포가 쓰지 않는 선택 밖 용어의 수정, 폐포가 이미 추론하는
  문장 추가입니다. `on_collision: qualify`를 쓴다면 충돌하는 네임스페이스를 모두
  `options.prefixes`에 묶으세요. 그러지 않으면 접미사가 파일이 선언한 접두사에서 올 수
  있는데, 다시 저장하면 그 접두사가 바뀌거나 사라질 수 있고(`BANK_EX`가 해시로 바뀜),
  상대 IRI의 네임스페이스와 그 해시는 파일 위치를 따릅니다.

**기준 IRI 선언.** 절대 IRI를 쓰거나 기준 IRI를 선언하세요. Turtle과 TriG는 `@base`,
RDF/XML은 `xml:base`, JSON-LD는 컨텍스트의 `@base`입니다. 상대 IRI(`<#Vendor>`)는
파일 자신의 위치(`file:///app/ontology/procure.ttl#Vendor`, 또는 `s3://` URL)를 기준으로
풀리므로, 파일을 옮기거나(경로 사이든 경로와 S3 사이든) 다른 경로에서 읽으면 그런 용어의 이름이 모두 바뀝니다. `type_iri`와
`predicate_iri`가 달라지고 다음 실행이 저장된 모든 항목의 표시를 다시 씁니다.
디렉터리의 두 파일이 모두 `<#Vendor>`를 쓰면 서로 다른 클래스 두 개가 됩니다. 이럴 때
읽기 단계에서 `N terms have IRIs relative to their file` 경고를 남깁니다.

**따라가지 않는 것, 읽지 않는 것.**

- `owl:imports`는 따라가지 않습니다. 가져올 온톨로지 파일을 같은 디렉터리에 두세요
  (디렉터리는 온톨로지 하나). 어떤 파일도 선언하지 않는 가져오기는 경고로 한 번
  남깁니다.
- `path` 외에는 아무것도 내려받지 않습니다. 문서를 가리키는 JSON-LD `@context`나 `@import`는
  거부합니다(컨텍스트를 파일 안에 넣거나 Turtle로 변환). XML 외부 엔티티는 풀지 않으며
  내부 엔티티는 펼칩니다.
- OWL/XML(`.owx`, 또는 루트 요소가 `owl:Ontology`인 RDF/XML 확장자 파일), OWL 함수형
  문법(`.ofn`), 맨체스터 문법(`.omn`), OBO(`.obo`), Notation3(`.n3`)은 거부합니다. 먼저
  변환하세요. ROBOT은 `robot convert --input onto.owx --output onto.ttl`, Protégé는
  *File > Save as > Turtle*입니다.
- 파서가 이차 시간(quadratic time)이 걸리는 모양은 파싱 전에 한도를 알려 주며
  거부합니다. 엔티티가 XML 파서의 증폭 한도를 넘게 펼쳐지는 RDF/XML, 텍스트 조각이
  10,000개를 넘는 RDF/XML 요소(`MAX_XML_TEXT_CHUNKS`. 줄바꿈과 엔티티 참조가 각각 한
  조각), 파일 하나의 `rdf:parseType="Literal"` 값 안에 요소가 1,000개를 넘는 경우
  (`MAX_XML_LITERAL_ELEMENTS`), 256 KiB를 넘는 N-Triples·N-Quads 줄(`MAX_LINE_BYTES`)입니다.
  이런 파일은 Turtle로 변환하세요. RDF/XML이 통과시키는 공백 있는 IRI와 순환하는
  `rdf:List`도 설정을 읽지 못하게 합니다. 파싱 오류는 줄 번호와 무엇이 잘못됐는지만
  알려 주고 파일 내용은 보여 주지 않습니다.

<a id="ontology-preview"></a>**미리 보기.** 모델을 호출하지 않고 추출이 쓸 스키마를 봅니다.

```python
import yaml

from unified_kg_rag import get_config
from unified_kg_rag.application.bootstrap import resolve_extraction_schema

extraction = resolve_extraction_schema(get_config("config.yaml")).processing.graph_extraction
print(yaml.safe_dump(
    extraction.model_dump(
        mode="json",
        include={"entity_types", "relationship_types", "relationship_patterns"},
        exclude_none=True,
    ),
    allow_unicode=True,
    sort_keys=False,
))
```

읽을 때마다 INFO 로그도 한 줄 남깁니다. `Ontology '<path>' (N files): ... entity types,
... relationship types and ... patterns from ... classes and ... properties; ...
properties dropped, ... ends lifted, ...` 형식이며, DEBUG에는 빠지거나 올려보내지거나
합쳐지거나 건너뛰거나 접미사가 붙은 용어가 하나씩 나옵니다. 종류를 정할 수 없는
속성(정의역, 치역, `owl:FunctionalProperty`만 있음)은 따로 INFO 로그
`Ontology: N properties not read ...`에 앞의 다섯 개까지 이름이 나옵니다.

**매핑 고정하기.** 온톨로지 수정이 그래프에 닿지 않게 하는 방법은 두 가지입니다.

- 미리 보기 출력을 `schema.yaml`로 저장하고 `schema_source: {kind: file, path: schema.yaml}`로
  읽습니다. 같은 스키마이므로 다시 추출하지 않고, extra도 더는 필요 없습니다.
- `kind: owl` 설정으로 `run-prompt-tuning --schema-out draft.yaml`을 실행하면
  ([프롬프트 튜닝으로 스키마 초안 만들기](#c-추출-스키마)) 온톨로지의 유형, 패턴, IRI에
  표본이 언급했지만 온톨로지에 없는 관계 유형을 더해 쓰고, 머리말에 그 유형과 그
  유형에 더한 패턴 수를 적습니다. 온톨로지가 선언한 유형은 온톨로지의 패턴만
  그대로 가지므로(`patterns: none`이면 패턴 없음), 더한 것이 없으면 초안은 같은
  스키마로 읽히고 다시 추출하지 않습니다. 새 유형을 온톨로지에 더하거나(`kind: owl`
  유지), 초안을 검토해 `kind: file`로 읽으세요(더한 유형 때문에 스키마가 바뀌므로
  다시 추출함).

### D. 시드 그래프(가지고 있는 지식 그래프 가져오기)

시드 그래프(seed graph)는 이미 가지고 있는 지식 그래프(공급업체 마스터, 제품 카탈로그,
온톨로지의 인스턴스 등)를 문서에서 추출한 그래프에 합친 것입니다. 시드마다 실행 자신의
코퍼스에 속한 가상 문서(`seed_graphs/<name>/<k>-of-<n>.jsonl`)로 나뉘므로, 증분 실행,
삭제, 재개와 저장소가 시드를 다른 파일과 똑같이 다룹니다. 시드는 어떤 모델도 읽지
않습니다. 시드의 진술(statement)은 파싱할 뿐이며, 모델로 청킹하거나 번역, 추출,
글리닝(gleaning), 주장 추출을 하지 않습니다.

문서가 언급하는 엔티티를 정리된 원천이 이미 설명하고 있고 답변이 그 설명, 속성, IRI를
쓰게 하고 싶을 때, 또는 텍스트에 없는 에지(조직도, 누가 누구에게 공급하는지)가 있어야
로컬 검색과 그래프 확장이 따라갈 수 있을 때 씁니다. 시드는 문서를 덮어쓰지 않으며(아래
[*우선순위*](#seed-precedence)), 유형이 아니라 인스턴스를 가져옵니다. 유형 어휘는 추출 스키마(C)가 정합니다.

**설정.** 시드는 `processing.seed_graphs` 목록에 선언합니다.

```yaml
processing:
  graph_extraction:
    entity_types:
      - {label: VENDOR, description: "Companies that sell goods", iri: "https://example.org/procure#Vendor"}
      - {label: BUYER, description: "Companies that buy goods", iri: "https://example.org/procure#Buyer"}
    relationship_types:
      - {label: SUPPLIES, iri: "https://example.org/procure#supplies"}
    relationship_patterns:
      - "VENDOR -SUPPLIES-> BUYER"
    schema_mode: align
  seed_graphs:
    - name: procure
      kind: jsonl
      path: seed/procure.jsonl         # 이 파일 기준 상대 경로
      filters: {department: purchasing}
      description_properties: [country]
```

| 키 | 기본값 | 의미 |
|---|---|---|
| `name` | 필수 | 샤드 문서의 이름과, 시드가 진술한 모든 항목의 `seed_graph` 문서 필터 값이 됩니다. 소문자, 숫자, `-`, `_`로 64자까지이며 문자나 숫자로 시작하고 시드끼리 겹치면 안 됩니다. 이름을 바꾸면 시드를 다시 가져옵니다 |
| `kind` | 필수 | `jsonl`, `csv`, `rdf`(`ontology` extra 필요), 또는 배포에서 등록한 종류([설계 문서 §15](./design.md#15-extension-guide), 영문) |
| `path` | 필수 | 로컬 파일이나 디렉터리(설정 파일 기준 상대 경로이며 환경 변수로 지정해도 같음. `file://` URL도 됨), S3 객체(`s3://bucket/key`), 또는 디렉터리로 읽는 S3 접두사(`s3://bucket/seed/`, 끝의 `/` 필수)입니다. 실행의 AWS 세션으로 읽습니다. 그 밖의 URL은 실패합니다 |
| `options` | `{}` | 종류별 설정(아래). 모르는 옵션이 있으면 가져오기가 실패합니다 |
| `secret_arn` | `null` | 등록한 서비스 소스가 읽는 Secrets Manager 비밀. `jsonl`, `csv`, `rdf`는 받지 않습니다 |
| `filters` | `{}` | 시드 항목의 문서 필터로, 문서의 메타데이터 필터와 같습니다(문자열, 숫자, 불리언 값). `seed_graph: <name>`이 항상 더해지므로 `seed_graph` 키는 쓸 수 없습니다 |
| `on_name_collision` | `error` | 이름으로 보아 한 엔티티인 시드 노드 두 개를 다루는 방법: `error`, `merge`, `qualify`([*우선순위*](#seed-precedence)) |
| `description_properties` | `[]` | 설명 뒤에 `key: value` 줄로 적을 속성 키. 답변이 이 값을 쓸 수 있습니다 |
| `confidence` | `1.0` | 시드 진술의 신뢰도. 노드에 `confidence`가 있으면 그 값을 씁니다 |
| `shards` | `16`(1~1,024) | 노드 id의 안정적인 해시로 시드를 나눌 가상 문서 수 |
| `max_items_per_unit` | `50`(1~500) | 텍스트 단위 하나에 담는 노드·에지 진술 수. 계보(lineage)의 단위입니다 |
| `max_nodes` | `100000` | 노드가 이보다 많으면 실패 |
| `max_edges` | `500000` | 에지가 이보다 많으면 실패 |
| `max_bytes` | `268435456`(256 MiB), `rdf`는 `33554432`(32 MiB) | 파일 크기 합이 이보다 크면 실패(읽기 전과 읽는 중에 확인). RDF 파서는 `max_nodes`를 확인하기 전에 그래프 전체를 메모리에 두며, 파일 크기의 약 25배(N-Triples)에서 최대 약 150배(조밀한 Turtle)를 씁니다(32 MiB면 최대 약 5 GB). `rdf`에서 늘리려면 먼저 태스크 메모리를 맞추세요 |
| `max_property_keys` | `64`(0~512) | 서로 다른 속성 키가 이보다 많으면 실패(키마다 `attributes.seed_properties`의 키가 됨) |
| `max_description_chars` | `1000`(50 이상) | 설명을 단어 경계에서 이 길이로 자르고 "…"를 붙입니다. 시드만 진술한 항목은 설명 요약기가 필요 없게 됩니다 |

- 한도를 넘으면 가져오기가 실패하며 아무것도 잘라내지 않습니다. 잘라내면 그래프가
  임의로 바뀌기 때문입니다. 텍스트 단위가 2,000개를 넘는 샤드도 가장 큰 노드를 알려 주며
  실패합니다. `shards`를 늘리거나, 허브(노드와 그 나가는 에지는 한 샤드에 있으므로 샤드를
  늘려도 나뉘지 않음)라면 `max_items_per_unit`을 늘리세요. 1 MiB를 넘는 JSONL 줄이나 CSV
  행도 실패합니다.
- 저장소가 키워드(UTF-8 최대 32,766바이트)로 색인할 수 있도록 이름, IRI, 유형의 길이를
  제한합니다. 노드 `name`(한정어를 붙인 이름 포함)은 1,000자, `iri`·`type_iri`·
  `predicate_iri`는 4,000자, `type`은 256자까지입니다. 올바른 유니코드가 아닌 텍스트(JSON
  `\ud800` 이스케이프 같은 짝 없는 서로게이트(lone surrogate))는 어느 필드에서든 파일과
  줄, 또는 RDF 주어를 알려 주며 실패합니다.
- 모든 종류가 옵션 두 개를 받습니다. `dangling`(없는 노드를 가리키는 에지:
  기본값 `error`, 또는 그 에지를 빼고 개수만 세는 `skip`)과
  `qualifier_property`(`on_name_collision: qualify`용)입니다.
- 환경 변수에서는 목록을 JSON으로 씁니다.
  `UKR__PROCESSING__SEED_GRAPHS='[{"name": "procure", "kind": "jsonl", "path": "seed/procure.jsonl"}]'`
- `run-ingestion`은 시작할 때 추출 스키마를 읽은 다음, 모델을 만들기 전에 시드마다 한 번
  읽습니다. 가져올 수 없는 시드는 시드 이름과, 잘못된 레코드라면 파일과 줄을 알려 주고
  종료 코드 2로 끝납니다. 설정한 종류에 필요한 extra(`rdf`)가 없을 때도 같습니다. 다른
  CLI는 시드를 읽지 않으므로(`run-prompt-tuning`은 문서만 분석) 시드 종류의 extra도
  필요 없습니다. 질의 이미지는 `ontology` extra 없이도 `rdf` 시드를 가져온 인덱스에
  답합니다.
- 시드를 설정하면 코퍼스가 비어 있어도 됩니다. 레지스트리에 문서가 없는 소스 범위(새 범위나
  시드 전용 범위)라면 빈 `--source-directory`는 시드만 가져오는 실행입니다([*수명 주기*](#seed-lifecycle)).
  문서가 있는 범위(표기만 다른 범위 포함: `s3://bkt/corpus`와 `s3://bkt/corpus/`는 한
  범위)에서 빈 코퍼스는 볼륨이 아직 마운트되지 않았거나 S3 동기화가 아무것도 복사하지
  않은 경우일 가능성이 훨씬 높으므로, 실행은 문서를 지우는 대신 쓰기 전에
  `document_loading`에서 실패합니다(`No source document reached this run, only its seed
  graphs, but the doc-status registry holds 2 documents of source scope ...`, 종료 코드 1).
  레지스트리를 끈 경우는 [*수명 주기*](#seed-lifecycle)를 보세요.
- **AWS에서는** 시드 파일을 이미지 안 설정 파일 옆에 두거나 수집 태스크 역할이 읽을 수
  있는 S3 위치에 두어야 합니다
  ([운영 런북: Seed graphs in the image or S3](./operations.md#seed-graphs-in-the-image-or-s3), 영문).
  수집 원본 접두사에는 두지 마세요. 그곳의 `.csv`는 문서로도 파싱됩니다.

**JSONL**(`kind: jsonl`): 파일 하나(이름은 상관없음), 또는 `*.jsonl` 파일을 하위까지
정렬된 순서로 읽는 디렉터리입니다(`.`으로 시작하는 이름은 건너뜀). 빈 줄이 아닌 줄은
`"kind"`가 `node`나 `edge`인 JSON 객체입니다.

```json
{"kind": "node", "id": "v1", "name": "Acme Supplies", "type": "VENDOR", "iri": "https://example.org/procure#acme", "description": "Industrial valve vendor.", "properties": {"country": "DE"}}
{"kind": "node", "id": "b1", "name": "Globex Buyer", "type_iri": "https://example.org/procure#Buyer"}
{"kind": "edge", "source": "v1", "target": "b1", "type": "SUPPLIES", "description": "Acme supplies Globex with valves.", "weight": 2}
```

- 노드는 `id`(필수, 중복 불가), `name`(기본값: id), `type`(레이블), `type_iri`(스키마가
  대응하는 클래스 IRI), `description`, `iri`(노드 자신의 IRI), `confidence`,
  `properties`를 받습니다.
- 에지는 `source`와 `target`(노드 id), `type` 또는 관계 유형이 선언한 `predicate_iri`(둘
  중 하나는 필수), `description`, `weight`(기본값 1), `properties`를 받습니다.
- 속성 값은 문자열, 숫자, 불리언이며 `null`은 버립니다. 모르는 키, 중복된 노드 id,
  스칼라가 아닌 속성, `NaN`, JSON 객체가 아닌 줄, 디코딩할 수 없을 만큼 깊이 중첩된 줄은
  파일과 줄을 알려 주며 실패합니다.
- 자기 자신을 가리키는 에지는 버립니다(전체 빌드도 그리지 않음). 양 끝과 유형이 같은
  에지는 합칩니다. 가중치는 더하고 설명은 합집합으로 모읍니다.

**CSV**(`kind: csv`): `path`는 `nodes.csv`와 `edges.csv`가 든 디렉터리(또는 `/`로
끝나는 S3 접두사)입니다. 첫 행이 열 이름이며, 위의 노드·에지 키가 열이 되고 그 밖의
열은 문자열 속성입니다.

```text
id,name,type,type_iri,description,iri,confidence,country
v1,Acme Supplies,VENDOR,,Industrial valve vendor.,https://example.org/procure#acme,,DE
b1,Globex Buyer,,https://example.org/procure#Buyer,,,,
```

```text
source,target,type,predicate_iri,description,weight
v1,b1,SUPPLIES,,Acme supplies Globex with valves.,2
```

```yaml
  seed_graphs:
    - name: procure
      kind: csv
      path: seed/procure               # nodes.csv와 edges.csv가 든 디렉터리
      filters: {department: purchasing}
      description_properties: [country]
      options:
        property_columns: [country]
```

이 파일들은 JSONL 예시와 같은 그래프를 진술합니다. 빈 칸은 값이 없는 것입니다.
`options.nodes`와 `options.edges`로 디렉터리 안의 다른 파일을 지정하고(`edges: null`이면
노드만 읽음), `options.property_columns`는 나열한 추가 열만 남깁니다. UTF-8이며 바이트
순서 표시(BOM)는 있어도 됩니다. 열보다 칸이 많은 행, 반복된 열, `id`(`source`,
`target`) 열이 없는 파일, Python CSV 한도(131,072자)를 넘는 필드는 실패합니다.

**RDF**(`kind: rdf`): Turtle, N-Triples, N-Quads, TriG, RDF/XML, JSON-LD로 쓴
온톨로지의 인스턴스(A-Box)입니다. `ontology` extra가 필요하며(이미지도
`--build-arg UV_EXTRAS="--extra ontology"`로 빌드), `owl` 스키마 소스와 똑같이 파일을
읽습니다(파일, 디렉터리, `s3://` 객체나 접두사, `options.format`, 아무것도 가져오지 않고
`owl:imports`를 따라가지 않으며 같은 형식을 거부:
[가져오지 않는 것, 읽지 않는 것](#owlrdfs-온톨로지에서-읽기)). 크기는 시드의
`max_bytes`를 따르며, 설정하지 않으면 32 MiB입니다(파싱한 그래프를 메모리에 두며 파일
크기의 최대 약 150배, 약 5 GB를 씀). procure 온톨로지를 스키마로 쓰면 다음과 같습니다.

```yaml
processing:
  graph_extraction:
    schema_source: {kind: owl, path: ontology/procure.ttl}
    schema_mode: strict
  seed_graphs:
    - name: procure
      kind: rdf
      path: seed/procure-abox.ttl
      options:
        languages: [en, ""]            # 기본값. [ko, en, ""]이면 Acme 이름이 한국어로 됨
```

```turtle
@prefix ex: <https://example.org/procure#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:acme a ex:Organization, ex:Vendor ;
    rdfs:label "Acme Supplies"@en, "애크미 서플라이"@ko ;
    skos:definition "Industrial valve vendor."@en ;
    ex:country "DE" ;
    ex:rating "4"^^xsd:integer ;
    ex:supplies ex:globex ;
    ex:headquarters ex:berlin .

ex:globex a ex:Buyer ;
    rdfs:label "Globex Buyer" .

ex:berlin rdfs:label "Berlin"@en .
```

이 파일은 노드 두 개와 에지 하나가 됩니다. `Acme Supplies`(두 클래스 중 가장 구체적인
`VENDOR`, 속성 `ex:country` = `DE`, `ex:rating` = 4), `Globex Buyer`(`BUYER`), 그리고
`predicate_iri`가 `ex:supplies`인 `Acme Supplies -SUPPLIES-> Globex Buyer`입니다.
`ex:berlin`은 유형이 없어 인스턴스가 아니므로 `ex:headquarters` 에지는 건너뜁니다
(`objects_not_instances`로 셈).

- **인스턴스**는 RDF, RDFS, OWL, XSD, SKOS 어휘 밖의 `rdf:type`이 있는 이름 있는 주어
  가운데 그 자신이 클래스나 속성이 아닌 것입니다. 노드의 `id`와 `iri`는 그 IRI입니다.
- **이름**은 `name_properties`(기본값 `skos:prefLabel`, `rdfs:label`, `schema:name`)에서
  `languages`(기본값 `[en, ""]`, `""`는 태그 없음, 그 밖의 태그는 그 뒤) 순서로 고르고,
  없으면 IRI의 로컬 이름을 퍼센트 디코딩해 그대로 씁니다. **설명**도 같은 방식으로
  `description_properties`(기본값 `skos:definition`, `rdfs:comment`,
  `schema:description`)에서 읽습니다.
- **유형.** `rdf:type`마다 스키마의 엔티티 유형 IRI에서 찾고, 여러 개가 대응하면 스키마가
  마지막에 선언한 유형을 씁니다. `owl` 스키마는 클래스를 하위 클래스보다 먼저 나열하므로
  그것이 가장 구체적인 클래스입니다(위에서는 `ORGANIZATION`이 아니라 `VENDOR`). 어떤
  유형도 선언하지 않은 클래스는 그 IRI를 `type_iri`로 두고 레이블은 `labels`가 준 값,
  없으면 로컬 이름을 `UPPER_SNAKE`로 바꾼 값입니다. 그러면 `strict`는 그 노드를 버리고
  `align`과 `off`는 남깁니다.
- **에지**는 목적어가 인스턴스인 트리플이며, 술어를 IRI로 가진 관계 유형, 없으면
  `labels`, 없으면 술어의 로컬 이름을 `UPPER_SNAKE`로 바꾼 값이 유형입니다. 인스턴스가
  아닌 목적어는 건너뛰고 셉니다(`dangling`은 적용되지 않음).
- **속성**은 나머지 리터럴이며, 키는 `options.prefixes`, 잘 알려진 접두사, 파일 자신의
  접두사 순으로 만든 CURIE입니다. 어느 것도 그 이름공간을 묶지 않으면 전체 IRI가
  키입니다(N-Triples는 접두사를 선언하지 않으므로 `options.prefixes`로 묶지 않으면 키가
  전체 IRI). 숫자와 불리언 타입 리터럴은 숫자와 불리언이 됩니다. 값이 여러 개인 술어는
  언어 기준으로 선호하는 문자열, 없으면 가장 작은 값을 남깁니다. `properties`는 나열한
  술어만 남기고 `exclude_properties`는 일부를 뺍니다.
- **읽지 않는 것**: 빈 노드(blank node), 구체화(reification)한 진술, RDF-star,
  `owl:sameAs`는 건너뛰고 셀 뿐 따라가지 않습니다. 아무것도 추론하지 않습니다. 역관계,
  대칭, 하위 속성 에지를 만들지 않고 하위 클래스 진술이 함의하는 유형도 붙이지 않습니다.
  시드마다 INFO 로그 한 줄에 개수가 나옵니다.
- **파일 기준 상대 IRI**(`@base` 없음)는 경고합니다. 파일을 옮기면 노드 이름이 바뀌어
  다시 가져오게 됩니다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `prefixes` | `{}` | 이 옵션들과 속성 키의 CURIE에 쓸 접두사 → 이름공간 |
| `name_properties` | `skos:prefLabel`, `rdfs:label`, `schema:name`(`https`와 `http`) | 이름을 읽을 리터럴 |
| `description_properties` | `skos:definition`, `rdfs:comment`, `schema:description`(`https`와 `http`) | 설명을 읽을 리터럴 |
| `languages` | `[en, ""]` | 이름, 설명, 문자열 속성의 선호 언어 태그 |
| `labels` | `{}` | 스키마 유형이 선언하지 않은 클래스나 속성(IRI 또는 CURIE) → 쓸 유형 레이블 |
| `properties` | `null`: 나머지 리터럴 전부 | 속성으로 남길 리터럴 술어 |
| `exclude_properties` | `[]` | 남기지 않을 리터럴 술어 |
| `format` | `auto` | `auto`(확장자), `turtle`, `xml`, `json-ld`, `nt`, `nquads`, `trig` |

**스키마와 맞추기.** 시드는 실행이 읽은 추출 스키마(C)에 맞춰 시드 전체를 한 번에
변환하므로, 패턴 검사는 언제나 양 끝 유형을 압니다. `on_name_collision: qualify`에서는
이름이 같은 노드를 서로 다른 엔티티로 보고 각자의 유형으로 검사하고, `merge`에서는 전체
빌드처럼 한 엔티티로 검사합니다.

- `type_iri`(또는 `predicate_iri`)는 그 `iri`를 가진 선언 유형을 가리킵니다. 어떤
  유형도 선언하지 않은 IRI는 개수를 세고(`nodes_unmapped_type_iri`) 대신 `type`
  레이블을 씁니다. `type` 레이블은 추출과 같은 방식으로 씁니다(관계 유형을 선언하면
  `vendor`가 `VENDOR`가 됨).
- `off`는 아무것도 검사하지 않습니다. `align`은 전부 남기고 스키마 밖 항목을 셉니다.
  `strict`는 스키마 밖 노드와 그 에지, 선언하지 않은 유형의 에지, 패턴이 허용하지 않는
  에지를 버립니다. JSONL 예시에 `DISTRIBUTOR` 노드와 그 `SUPPLIES` 에지,
  `BUYER -SUPPLIES-> VENDOR` 에지를 더하면 `align`에서는 `entities_off_schema=1`,
  `relationships_off_pattern=1`을 세고, `strict`에서는 그 노드와 두 에지도 버립니다.
- 개수는 INFO 로그 `Seed graph 'procure': ...`에 추출과 같은 카운터 이름으로 나옵니다.
  시드 항목도 추출한 항목처럼 `in_schema`, `type_iri`, `predicate_iri`를 받습니다.
  `iri`는 노드 자신의 IRI입니다.

<a id="seed-precedence"></a>**우선순위: 문서가 정합니다.** 이름이 같은 엔티티로 보이는(정규화 후 같거나, 추출이
합치는 것처럼 법인 형태만 다른) 시드 노드와 문서 속 언급은 한 엔티티가 됩니다.

| 대상 | 규칙 |
|---|---|
| 유형, 이름, 신뢰도 | 그 엔티티를 진술한 문서가 있으면 문서의 진술이 정합니다. 시드의 진술은 빈 곳만 채웁니다. 문서가 진술하지 않은 엔티티, 또는 문서가 관계의 끝점으로만 언급해 유형이 없는 엔티티입니다 |
| 설명 | 합집합이며 문서의 줄이 먼저, 시드의 줄이 그 뒤에 옵니다 |
| 속성 | `attributes.seed_properties` 아래에 두며, 추출한 속성을 덮어쓰지 않습니다. 저장만 하고 색인하지 않으므로 어떤 키와 문자열·숫자·불리언 값도 쓸 수 있고, 어떤 필터나 검색도 이 값과 일치하지 않습니다 |
| IRI | 진술된 IRI 중 가장 작은 값. 여러 IRI로 진술된 엔티티는 셉니다(`entities_with_conflicting_iris`, INFO) |
| 에지 | 더합니다. 양 끝과 유형이 같은 시드 에지와 문서 에지는 한 관계이며 가중치를 더합니다 |
| 순위(rank) | 문서가 뒷받침하는 에지만 셉니다(인용한 텍스트 단위가 모두 시드의 것이면 시드 전용 에지). 시드는 엔티티를 더 중요하게 만들지 않으며, 시드 에지로만 이어진 엔티티의 순위는 0입니다. Leiden은 여전히 모든 에지를 원래 가중치로 군집화합니다. `indexing.neptune.entity_importance_source: degree`는 질의 시점에 에지를 세므로 시드 에지도 셉니다 |
| 커뮤니티 | 시드만 진술한 엔티티로 이루어진 커뮤니티는 리포트 없이 저장합니다(단계 지표 `communities_seed_only`). 섞인 커뮤니티의 리포트는 문서 기준 순위가 높은 엔티티부터 나열합니다. 증분 실행에서는 실행의 그래프가 시드로만 진술하더라도(변경 없는 문서 옆에 더한 시드) 저장된 구성원 중 문서가 뒷받침하는 것이 있으면 섞인 커뮤니티로 봅니다 |

- **동음이의어.** 정체성은 이름입니다. 관계없는 시드 노드 "Mercury"(행성)와 문서의
  "Mercury"(원소)는 한 엔티티가 되어 시드의 설명 줄과 IRI를 얻지만 유형은 문서의 것을
  유지합니다. `on_name_collision`은 한 시드 안의 이름만 다룹니다. 이름을 다듬거나 시드
  안에서 한정어를 붙이세요.
- **철자 변형.** 위 규칙은 이름 단위로 성립합니다. 같지는 않지만 한 엔티티로 해석되는
  이름(문서의 "Acme", 시드의 "Acme Inc")에서는 시드도 영향을 줍니다. 문서 철자 두 개가
  동률이면 시드의 단위가 이름을 정할 수 있고, 변형들의 설명 줄이 섞입니다.
  `processing.fuzzy_entity_resolution`을 켜면 시드 철자가 문서 철자 두 개를 이어 붙일 수도
  있습니다.
- **`on_name_collision`**은 이름으로 보아 한 엔티티이지만 id도 IRI도 같지 않은 한 시드의
  노드 두 개에 적용됩니다. `error`(기본값)는 그 노드를 나열하며 가져오기를 실패시킵니다
  (`1 names are shared by distinct nodes (ids v1, v2); set on_name_collision to merge or qualify`).
  `merge`는 한 엔티티로 합칩니다(가장 작은 IRI 유지). `qualify`는 id가 가장 작은 노드의 이름은 그대로
  두고 나머지 이름에 ` (<qualifier>)`를 붙입니다. 한정어는 `qualifier_property` 값, 없으면 IRI의 로컬 이름,
  없으면 id입니다(`ACME supplies (v2)`). 한정어를 붙인 이름은 문서 속 언급과 더는 이어지지
  않습니다.

**검색.** 시드의 사실은 엔티티와 관계를 읽는 모든 전략(벡터·키워드 검색, 시드 에지를
따라가는 그래프 확장)에서 엔티티와 관계로 답변에 들어가며, 텍스트 청크로는 들어가지
않습니다. 시드의 텍스트 단위는 계보 때문에 저장하지만 모든 텍스트 검색이 이를 빼고,
`local`과 `mix`는 청크 할당량을 여기에 쓰지 않습니다. `global`과 `drift`의 프라이머는
커뮤니티 리포트를 읽습니다. 시드만 진술한 커뮤니티에는 리포트가 없지만, 섞인 커뮤니티의
리포트는 시드 구성원도 설명하므로(그 설명이 리포트 프롬프트에 들어감) 시드의 사실은 문서도
진술한 커뮤니티를 통해 이 전략에 닿습니다. `naive`는 텍스트 단위만 읽습니다.

- **인용.** 시드가 진술한 출처의 메타데이터에는 `provenance`가 있고 청크는 없습니다
  (`chunk_id`는 `None`).

  ```python
  {"kind": "seed_graph",          # 또는 "document+seed_graph"
   "seed_graphs": ["procure"],
   "iri": "https://example.org/procure#acme"}
  ```

  시드가 진술하지 않은 출처에는 `provenance` 키가 없습니다. `local`(과 이를
  실행하는 `drift`, `auto`)은 엔티티 인덱스가 찾은 엔티티를 그래프의 사본으로 보고하는데,
  이 사본은 그 검색 결과의 문서 필터를 받으므로 시드 그래프 이름이 나옵니다. 그래프
  탐색으로만 도달한 엔티티는 `"seed_graphs": []`입니다. 그래프 정점에는 문서 필터가 없기
  때문입니다. 메타데이터의 `source_kinds`는 모든 저장소에서 목록입니다.
- **필터**([필터](#필터)):
  - `source_kinds`(엔티티, 관계, Neptune 엔티티 정점): `seed_graph`는 시드가 진술한 것,
    `document`는 문서가 진술한 것(둘 다 진술한 엔티티도 시드 설명 줄과 함께 포함)을
    남깁니다. 목록은 둘 중 하나와 일치합니다. 대소문자는 상관없습니다. 커뮤니티 보고서에는
    출처 종류가 없고 시드에만 있는 구성원까지 커뮤니티 전체를 요약하므로, 한 종류만 남기는
    필터에서는 `local`, `drift`, `simple`이 보고서를 읽지 않습니다(DRIFT는 질문 자체로 local
    검색을 한 번 실행하고, `simple`은 나머지 인덱스를 모두 검색하며, `auto`는 이 중 하나로
    보냄). 보고서만 읽는 `global`은 이 키를 거부합니다. `mix`와 `hybrid`는 보고서를 읽지 않고,
    텍스트 청크만 읽는 `naive`도 이 키를 거부합니다.
  - `iri`(엔티티, Neptune 엔티티 정점): 정확한 IRI.
  - `attr_seed_graph: procure`는 모든 인덱스가 가진 문서 필터로 그 시드가 진술한 것을
    남깁니다. `attr_department: legal` 같은 테넌트 필터는 시드의 `filters`에 그 키가 없으면
    시드만 진술한 것을 빼며, 문서와 같은 규칙입니다.
  - 시드 속성 키로는 필터링할 수 없습니다. 벡터 저장소가 `attributes.seed_properties`를
    매핑하지 않으므로 타입, 길이, 키 이름과 관계없이 어떤 속성도 쓰기를 실패시키지 않습니다.
    값으로 필터링하려면 시드의 `filters`에 넣거나(시드 전체에 값 하나), 키를
    `description_properties`에 넣어 텍스트·벡터 검색이 보게 하세요.

```python
RAGInput(query="...", search_strategy="local", filters={"source_kinds": "document"})
RAGInput(query="...", search_strategy="mix", filters={"attr_seed_graph": "procure"})
```

**무엇을 다시 인덱싱하나.** 파일이 아니라 변환된 진술을 키로 씁니다.

| 변경 | 다음 증분 실행 |
|---|---|
| 노드나 에지 수정 | 진술이 바뀐 샤드만 다시 가져옵니다(삭제 후 추가, 모델 호출 없음). 문서는 다시 추출하지 않습니다 |
| `shards` 변경 | 시드 전체를 다시 가져옵니다. 샤드 경로가 모두 바뀝니다 |
| 시드 이름이나 `filters` 변경 | 시드 전체를 다시 가져옵니다. 샤드마다 이 값을 적습니다 |
| `description_properties`, `confidence`, `max_description_chars`, `max_items_per_unit`, `on_name_collision` 변경 | 진술이 바뀌는 샤드를 다시 가져옵니다 |
| 변환된 진술을 바꾸는 스키마 변경(레이블, `schema_mode`, `type_iri`, `predicate_iri`, RDF 클래스가 대응하는 IRI) | 영향받는 샤드를 다시 가져옵니다(문서는 다른 스키마 변경과 같이 다시 추출) |
| 모델이나 추출 설정 | 문서는 다시 추출하고, 시드 샤드는 그대로 둡니다. 어떤 설정도 시드가 진술하는 내용을 바꾸지 않습니다 |
| 파일 이동이나 재직렬화, 시드가 이미 지키는 한도 변경 | 아무것도 하지 않습니다 |

충돌 후 시드가 바뀐 상태로 재개한 실행은 단계를 다시 계산합니다. 시드 내용이 모든 단계의
캐시 키에 들어가기 때문입니다.

<a id="seed-lifecycle"></a>**수명 주기.** 아래 동작에는 문서 상태 레지스트리(`providers.doc_status.enabled`)가
필요합니다. 레지스트리가 없으면 매 실행이 코퍼스와 설정한 시드의 전체 빌드이며, 빈
코퍼스와 없는 코퍼스를 구분할 수 없습니다. 그래서 레지스트리를 끄면 빈 코퍼스는 인덱스에
엔티티, 관계, 텍스트 단위가 아직 없을 때만(첫 시드 전용 빌드, 잃을 것이 없음) 묻지 않고 시드만으로
빌드합니다. 무언가 있으면 아래 재설정처럼 먼저 묻습니다(`... rebuild the index 'default'
from the seed graphs alone, replacing what it holds; continue?`). `--confirm-reset`,
`UKR_CONFIRM_RESET=1`, `--yes`가 '예'로 답하고, '아니요'로 답하거나 터미널이 없으면 쓰기
전에 멈춥니다(종료 코드 1). 판단할 수 없거나 읽지 못한 저장소는 항목이 있는 것으로 봅니다.

- **삭제.** 설정에서 시드를 빼면 다음 증분 실행이 그 샤드를 삭제합니다. 그 시드만 진술한
  것은 사라지고, 문서도 진술한 엔티티는 그 시드의 설명 줄, 속성, IRI, 에지를 잃습니다.
- **재설정.** `indexing.reset: true`는 설정한 시드를 포함한 실행의 코퍼스로 네임스페이스를
  다시 만듭니다. 재설정이 모르는 사이 잃기 쉬운 것을 지우게 되면 실행은 먼저 묻습니다. 다른
  소스 범위의 시드 문서(아래 시드 전용 범위), 또는 시드만으로 된 코퍼스에서 다른 범위의
  문서가 그렇습니다(시드 전용 범위의 재설정은 네임스페이스 전체를 그 시드로 다시
  만듭니다). `run-ingestion`은 터미널에서 묻고, `--confirm-reset`(또는
  `UKR_CONFIRM_RESET=1`)은 이 질문에만, `--yes`는 모든 질문에 '예'로 답합니다. '아니요'로
  답하거나 터미널이 없으면 쓰기 전에 멈춥니다(`Pipeline stopped before it wrote anything:
  ...`, 종료 코드 1). 빈 코퍼스로 하는 재설정 중 두 가지는 답과 관계없이 실패합니다.
  문서가 있는 범위의 재설정(위)과, 범위에 기록이 하나도 없는데 다른 범위에 문서가 있는
  재설정(`... holds no record of source scope ...`)입니다. 후자는 시드 전용 범위가 아니라
  새로 만들었거나 옮겼거나 잘못 적은 접두사입니다. 시드 전용 범위라면 이전 실행이 시드를
  기록해 두었습니다. 네임스페이스를 시드만으로 다시 만들려면 시드 전용 범위를
  `indexing.reset` 없이 한 번 실행한 뒤 재설정하세요. Python에서는
  `DataIngestionPipeline(..., confirm=callable)`을 넘기세요. 없거나 '아니요'면 `run`이
  `ConfirmationDeclinedError`(`unified_kg_rag`에서 가져올 수 있음)를 던집니다.
- **시드 전용 범위.** 시드를 설정하고 빈 `--source-directory`와 고정한
  `processing.document_parsing.source_scope`(예: `seeds-procure`)로 실행하면 시드가 자기
  범위에 따로 있고 모델 호출도 없습니다. 같은 네임스페이스의 다른 범위 실행은 이를
  건드리지 않으며, 재설정만 예외입니다.
- **시드 하나, 코퍼스 둘.** 시드는 그것을 설정한 실행의 범위에 속합니다. 한 네임스페이스의
  두 범위에 설정하면 두 번 가져옵니다. 엔티티는 같지만 에지 가중치가 두 번 셉니다. 공유
  시드는 한 범위에만 설정하거나 시드 전용 범위를 주세요. 범위의 첫 실행은 다른 범위가 같은
  이름의 시드를 이미 가지고 있으면 경고합니다(`Seed graph 'procure' is already imported by
  source scope(s) ...`). 소스 디렉터리를 옮겼거나 잘못 적었을 때도 이렇게 보입니다.

<a id="seed-cost-and-limits"></a>**비용과 한계.**

- 추출 호출(추출, 글리닝, 클레임, 번역)은 시드의 텍스트 단위를 읽지 않지만, 시드 설명은
  세 곳에서 모델에 닿습니다. 문서도 진술한 엔티티의 합친 설명이 임계값을 넘을 때의 설명
  요약기, 시드 구성원도 설명하는 문서도 진술한 커뮤니티의 리포트, 검색한 엔티티와 관계를
  담는 답변 프롬프트입니다. 시드 설명은 모델이 읽을 수 있는 데이터로 쓰세요. 시드만 진술한
  커뮤니티에는 리포트가 없고, 시드만 진술한 항목은 `max_description_chars`에서 잘리므로
  요약기가 필요 없습니다.
- 항목은 임베딩합니다. 노드마다 벡터 두 개(이름, 설명), 에지마다
  하나(`indexing.opensearch.build_relationship_vector_index`, 기본값 켜짐), 텍스트
  단위마다 하나로, 기본값에서 대략 2 × 노드 + 에지 + (노드 + 에지) / 50개입니다. 노드
  100,000개, 에지 500,000개 한도에서는 임베딩 약 712,000개입니다. 시드 전체를 메모리와
  단계 캐시에 둡니다.
- **증분 실행은 델타만 군집화합니다**(§5). 증분 실행으로 더한 시드는 재설정으로 그래프
  전체를 다시 군집화할 때까지 자기 샤드의 엔티티끼리만 군집화됩니다. 이 커뮤니티는 저장된
  구성원 중 문서가 뒷받침하는 것이 있으면 저장된 구성원의 설명 뒤에 시드의 설명 줄을
  붙여, 실행이 인덱싱하는 설명 그대로 리포트를 만듭니다. 시드를 고치면 이전 버전의 줄은
  빠지고, `description_summarization.force_summary_threshold_tokens`를 넘는 설명은
  먼저 요약합니다(한 번만: 인덱싱이 그 요약을 재사용합니다). 전체 빌드는 구성원의 모든
  줄을 한 번에 요약하므로 증분 실행의 설명은 같은 요약이 아니라 같은 내용을 담은
  요약입니다. 요약을 끄면 둘 다 줄 전체를 읽습니다.
- **번역.** 시드 진술은 번역하지 않지만 문서의 엔티티는 번역된 텍스트에서 추출합니다.
  시드 이름이 문서와 이어지려면 `processing.translation.target_language`로 쓰세요.
- **로컬 백엔드.** SQLite 그래프와 임베디드 벡터 저장소도 같은 필드를 두고 같은 필터를
  적용합니다.

**미리 보기.** 모델 호출도 저장소도 없이 시드가 무엇으로 변환되는지 봅니다.

```python
from unified_kg_rag import get_config
from unified_kg_rag.application.bootstrap import load_seed_graphs, resolve_extraction_schema

config = resolve_extraction_schema(get_config("config.yaml"))
for conversion in load_seed_graphs(config):
    print(conversion.name, conversion.stats.model_dump(exclude_defaults=True))
```

JSONL 예시에서는 `procure {'nodes_read': 2, 'edges_read': 1, 'nodes': 2, 'edges': 1,
'units': 2, 'documents': 2}`가 출력됩니다.

---

## 10. 운영과 문제 해결

[IAM 권한](./getting-started/aws.ko.md#iam-권한)은 AWS에서 시작하기에, 오류 메시지와 해결 방법은
[운영 런북: 자주 보는 오류](./operations.md#common-errors)에 있습니다.

### CLI 종료 코드와 JSON 출력

모든 CLI(`run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
`run-prompt-tuning`, `prepare-benchmark`)는 다음 코드로 종료합니다.

| 코드 | 의미 |
|---|---|
| `0` | 성공 |
| `1` | 실행 실패: 단계나 질의 실패(`run-ingestion`은 단계에서 발생한 `ConfigurationError`와 연결할 수 없는 저장소도 포함), `--max-failure-rate` 초과, `run-eval --fail-if` 게이트 실패, 업로드 실패, 렌더링한 파일 없음 |
| `2` | 사용법·설정 오류: 잘못된 플래그, 없거나 잘못된 입력(`run-eval` 데이터셋, `run-visualization` 데이터 파일, `prepare-benchmark` 데이터 파일), 잘못된 설정 파일이나 덮어쓰기, 설정하지 않은 저장소 엔드포인트, 선택한 제공자에 필요한데 설치하지 않은 extra(`local`)(모두 작업 시작 전), 또는 실행 중 단계 밖에서 발견한 `ConfigurationError`. 예: `run-ingestion`이 첫 모델 호출 전에 실행을 거부한 경우(실행에 필요한 인덱스를 검색 엔진이 만들지 않거나, 삭제만 있는 실행을 포함해 증분 실행의 현재 인덱스가 다른 임베딩·언어·레지스트리 설정으로 만들어짐) |
| `130` | Ctrl-C(SIGINT)로 중단 |
| `143` | SIGTERM으로 중지(ECS `StopTask`, Step Functions 시간 초과). `run-ingestion`은 끝난 단계를 먼저 체크포인트로 저장하고, `run-eval`은 끝난 실행의 결과를 `s3://` 출력에 업로드함(실패하면 로컬에 남기고 경로를 로그로 남김) |

`130`이나 `143`으로 중지된 실행은 프로세스를 바로 끝냅니다. 진행 중인 모델 호출은 기다리지 않고 버리며, 그 전에 로그, 추적 데이터, 표준 출력을 비우고 프로세스 풀의 작업자도 함께 끝냅니다. SIGTERM과 Ctrl-C는 질의 CLI의 작업을 다음 await 지점에서 취소하므로 정리 작업(저장소 연결 닫기, `run-eval`의 업로드)은 실행됩니다(대화형 프롬프트에서 기다리는 중이면 SIGTERM은 세션을, Ctrl-C는 대화를 바로 끝냄). SIGTERM이나 이벤트 루프 실행 중의 Ctrl-C로 중지된 질의 CLI가 100초 뒤에도 남아 있으면 그때 끝내며(앱 컨테이너의 StopTimeout 120초 안), 출력한 문서가 없으면 상태 문서를 출력합니다. 두 번째 신호는 바로 끝냅니다. `run-ingestion`은 대신 SIGTERM이나 Ctrl-C를 받으면 진행 중인 체크포인트나 메타데이터 쓰기를 마친 뒤 멈추며, StopTimeout이 그 상한입니다. 끝난 CLI는 종료 처리기를 실행한 즉시 자기 종료 코드로 프로세스를 끝내므로, 종료 중에 받은 신호가 `0`, `1`, `2`를 `-15`나 `-2`로 바꾸지 못합니다.

플래그는 전체 이름으로 써야 하며, `--output`처럼 앞부분만 쓰면 거부합니다.

`--output-format json`은 JSON 문서 하나를 표준 출력에 쓰고 배너·진행 상황·로그는 표준
오류로 보내므로 출력을 `jq`로 바로 넘길 수 있습니다(표준 출력에 함께 쓰는
`--metrics-sink cloudwatch`와는 같이 쓸 수 없음). 담기는 내용은 답변이나 검색
결과(`run-rag`), 파이프라인 ID와 상태, 실행이 `FAILED`로 기록한 문서 수(`failed_documents`. 모든
단위를 실패시킨 모델 장애는 아무것도 기록하지 않고, 일부 단위만 실패시킨 장애는 그 문서들을 기록하며
다음 실행이 `indexing.max_document_failures`에 세지 않고 다시 처리함), 단계가 실패한 텍스트 단위 수(`units_failed`. 그 단계의 출력 없이 인덱싱되고 다음 실행이 다시 추출함. 레지스트리가 문서를 `FAILED`로 기록하지 않아 `failed_documents`가 0인 전체 빌드에서도 셈. 실행 끝에 단계와 원인별 개수를 WARNING으로 알림), 문서 자체의 것이 아닌 오류로 연속 실행에서
또 실패한 문서 수(`documents_retrying`. 장애나 설정 때문이면 제외되지 않음. 파일과 실행 횟수를 경고로도 알림), 일부 텍스트 단위의 주장 추출이 단위 자체의 오류로 실패해 그 주장 없이 인덱싱된 문서 수(`documents_without_claims`. 실패로 세지 않음. 요약에도 표시), 내용과 추출 설정이 그대로인 채
`indexing.max_document_failures`번 연속 실패해 이번 실행에서 빠진 문서 수(`documents_excluded`. 설정을 바꾸면 다시 셈. 요약에도 표시),
이번 실행이 수행한 단계의 모델 호출과 토큰 수(`token_usage`),
단계별 결과(`run-ingestion`), 실행별 요약과 비교,
실패 예산 초과 내역과 실패한 게이트(`gate_failures`)(`run-eval`), 쓴 파일(`run-visualization`), 출력 경로와 프로필
(`run-prompt-tuning`), 쓴 항목 수(`prepare-benchmark`)입니다. 문서를 출력하기 전에 신호로 중지되면
대신 `{"status": "interrupted", "exit_code": 143}`(Ctrl-C는 `130`)을, 출력하기 전에
실패하면 `{"status": "failed", "exit_code": 1}`(또는 `2`)에 실패 원인 `error`(argparse 오류 포함)를
붙여 출력하므로 표준 출력은 항상 JSON입니다. 시작 직후에도 같습니다. CLI가 아직 로드 중일 때나 결과를
출력한 뒤 프로세스가 끝나는 중에 받은 SIGTERM도 `143`으로, 로드 중의 Ctrl-C는 `130`으로 끝납니다(상태
문서는 로드 중에만 출력. `nohup`처럼 부모가 무시하게 한 SIGINT는 계속 무시함). 단계가 실패한 `run-ingestion` 실행의 결과에도
같은 `error`와 `exit_code`가 담깁니다. `run-ingestion`은 재개할 때 쓸 `pipeline_id`를
(실행이 정한 뒤라면) 덧붙이고, 중지된 실행 다음에 같은 네임스페이스의 다음 실행이 아직 진행
중일 수 있는 쓰기(이 실행의 쓰기, 또는 이 실행이 만료된 임대를 넘겨받은 실행의 쓰기)를
기다려야 하면 `next_run_writes_from`(UTC, ISO 8601)도 덧붙입니다.

### 대규모, 다국어, 이질적인 코퍼스

- **대규모:** Bedrock 할당량이 허용하는 만큼 `processing.max_concurrency`를 올리고,
  `indexing.neptune.index_concurrency`는 효과를 측정하기 전까지 `1`로 두세요(동시 쓰기
  배치는 공유 정점에서 충돌해 재시도함). 문서 상태
  레지스트리(`providers.doc_status.enabled`)를 켜면 다음 실행부터 바뀐 문서에만 비용이
  듭니다. Neptune에서 `indexing.neptune.vertex_ids: custom`을 쓰면 정점을 `id` 속성
  대신 정점 id로 찾습니다. 기존 그래프를 전환하려면 전체 빌드를 한 번 실행해야
  합니다([운영 런북: 사용자 지정 Neptune 정점 id](operations.md#custom-neptune-vertex-ids)).
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
- **다국어:** [다국어 수집](#다국어-수집)을 참고하세요.
- **이질적인 도메인:** `entity_types`를 여러 도메인의 합집합으로 맞추거나, 도메인마다
  [인덱스 접미사](#인덱스-접미사)를 따로 두세요.

### 비용 참고

비용은 대부분 LLM 호출에서 나옵니다. 추출(청크당 1회 이상), gleaning(청크당 최대
`max_rounds`회), 커뮤니티 리포트, 주장(텍스트 단위당 1회, 기본은 꺼짐), 질의마다의 답변
생성입니다. 기계적인 역할에 Haiku급 모델 쓰기(청킹, 번역, map-reduce, 요약의 기본값),
gleaning 횟수 줄이기, 주장 끄기, 캐시와 증분 인덱싱으로 줄일 수 있습니다.

이 중 가장 큰 것은 gleaning입니다. `processing.gleaning.max_rounds`는 3을 유지합니다. 3회는
1회보다 그래프를 더 많이 찾았고 수집 비용은 약 38% 늘었으며, 답변에는 유의한 차이가
없었습니다([근거](./concepts/defaults-and-evidence.md#gleaning-rounds), 영문). 그래프 재현율보다 수집 비용이 중요하면
`max_rounds: 1`(MS GraphRAG 기본값)로 두세요.

---

## 11. Python 라이브러리로 사용하기

CLI는 공개 클래스 두 개를 감싼 래퍼입니다. 이 클래스와 그 밖에 쓰는 이름은 모두 패키지
최상위에서 import하세요. `unified_kg_rag.__all__`의 이름이 공개 API이고, 그 밖의 모듈
경로는 내부 구현이라 바뀔 수 있습니다. `get_config`는 CLI와 같은 설정 파일·환경 변수
단계를 적용합니다([설정](./guides/configure.ko.md#설정을-읽는-순서)).

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
- 구분해 처리할 오류는 패키지 최상위에서 내보냅니다. `ConfigurationError`(고칠 설정.
  `StoreRequestRejectedError` 포함), `InvalidFilterError`, `IndexNotFoundError`,
  `StoreConnectionError`, `NamespaceLeaseError`, `PipelineExecutionError`,
  `PipelineInterrupted`(`KeyboardInterrupt`)이며, 마지막을 빼면 모두
  `GraphRAGException`입니다. `DataIngestionPipeline.run`은 `ConfigurationError`,
  `StoreConnectionError`, `NamespaceLeaseError`는 그대로 올리고, 단계 밖의 다른
  오류는 `PipelineExecutionError`로 감쌉니다. 모델 클라이언트 자체의 전송 오류(로컬 모델 서버의
  `httpx.ConnectError` 등)는 그대로 전달됩니다. `DataIngestionPipeline(...)`은 가져올 수
  없는 시드 그래프에 `SeedGraphError`(`ConfigurationError`의 하위 클래스)를 던집니다.
- `DataIngestionPipeline(..., confirm=callable)`은 실행이 묻는 질문에 답합니다
  (`callable(question) -> bool`). 지금은 다른 소스 범위의 시드 그래프 문서를 지우게 되는
  재설정만 묻습니다. 넘기지 않으면 그런 실행은 실패합니다
  ([시드 그래프](#d-시드-그래프가지고-있는-지식-그래프-가져오기)).
- `RAGInput`은 `run-rag` 옵션을 필드로 받습니다(`search_strategy`, `search_type`,
  `top_k`, `suffix`([인덱스 접미사](#인덱스-접미사)), `filters`, `conversation_id`,
  `use_memory`, `target_language`). 일반 딕셔너리도 됩니다.
- `chain.search(...)`(`await chain.asearch(...)`)는 답변 없이 검색만 해서
  `SearchOutput`(`search_results`, `processed_query`, `metadata`)을 반환합니다.
- 비동기 코드에서는 `await chain.ainvoke(...)`와 `await chain.aclose()`를 쓰고, 시작할
  때 이벤트 루프의 실행기 크기를 한 번 지정하세요:
  `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`.
- 만든 객체는 항상 `close()` / `aclose()`로 닫으세요. 두 클래스 모두 처음 쓸 때 연결을
  열고, 닫지 않으면 가비지 컬렉션 때까지 유지합니다. 체인을 닫을 때 다른 스레드에서
  실행 중인 동기 호출은 `StoreConnectionError`를 일으키고, 그 뒤의 호출은 다시 연결합니다.
- 패키지를 import해도 로깅 핸들러는 설정하지 않습니다. CLI와 같은 로깅은
  `unified_kg_rag.setup_logging(config)`로 켭니다. 호출하지 않으면 구조화 필드
  (`structlog.contextvars`로 바인딩한 값 포함)가 레코드 속성으로 호스트 핸들러에
  전달되며, `LogRecord` 속성과 이름이 같은 필드(`name`, `msg` 등)에는 끝에 `_`가
  붙습니다. `setup_logging`은 프로세스 전체 상태를 바꾸므로 패키지가 로깅을 맡는
  곳에서만 호출하세요. structlog 설정을 교체하고, 루트 로거의 레벨을
  `logging.level`로, `logging.library_levels`에 있는 로거의 레벨을 각 값으로 정하고,
  루트 로거에 stdout 핸들러(`logging.log_to_file`이면 날짜가 붙은 파일 핸들러도)를
  추가합니다. 루트 로거에 이미 있던 핸들러는 그대로 남아 모든 레코드를 함께 받습니다.
  그중 하나가 콘솔에 쓰면 같은 줄이 두 번 나오며, 패키지 레코드에 대해서는 구조화 필드
  없이 메시지와 예외만 받습니다. 다시 호출하면 이전 호출이 추가한 핸들러만 교체합니다.

확장 코드도 패키지 최상위에서 import합니다. 포트(`LLMFactoryPort`, `GraphIndexer`,
`DocStatusPort` 등), `Providers`, `register_provider`, `BaseSearchStrategy`와
`register_strategy`, `ParserFactory`, `register_renderer`가 여기에 있습니다. 두 생성자
모두 백엔드를 `providers=Providers(config, ...)` 번들이나 개별 인자(`doc_status=`,
`vector_indexer=`, `model_factory=` 등) 중 하나로 주입받으며, 둘을 함께 쓸 수는
없습니다. [설계 문서 §15](./design.md#15-extension-guide)를 참고하세요.
`vector_indexer=` / `graph_indexer=`로 넘긴 인덱서는 호출한 쪽이 닫습니다. 파이프라인은
닫지 않습니다.

---

## 12. 제한 사항

- **추출 단계에서 환각이 생길 수 있습니다.** 원문이 뒷받침하지 않는 엔티티나 관계가
  만들어져 검색까지 이어질 수 있습니다.
  `processing.graph_extraction.entity_grounding.enabled`(기본은 꺼짐)는 인용한
  `source_text`가 청크에 없는 항목을 버리거나 가중치를 낮춥니다.
- **실행 결과를 재현할 수 없습니다.** LLM 출력(추출, 리포트, `auto` 라우터, 답변)은
  달라지므로 설정은 여러 번 실행해 비교하세요. 모델 응답과 소스 범위가 같으면 나머지
  처리는 어느 프로세스에서든(`PYTHONHASHSEED`와 관계없이) 결정적입니다. 문서 상태
  레지스트리를 켜면 텍스트 단위 ID가 소스 범위에서 나오며, 소스 범위의 기본값은 소스
  디렉터리의 절대 경로이므로 같은 코퍼스도 다른 디렉터리에서 수집하면 ID가 달라집니다.
  코퍼스 위치와 관계없는 ID가 필요하면 `processing.document_parsing.source_scope`를
  설정하세요.
- **재순위화는 리전에 따라 다릅니다**([필수 모델](./getting-started/aws.ko.md#필수-모델)). 없으면 융합 순서를 쓰며,
  `metadata.rerank`에 나타납니다([재순위화 상태](#재순위화-상태)).
- **기본 제공 백엔드.** 모델: Bedrock(기본값), Ollama, OpenAI 호환 서버. 재순위화:
  Bedrock이나 sentence-transformers 크로스 인코더([로컬 모드](./guides/local-mode.md), 영문).
  저장소: Neptune(또는 Gremlin Server)과 OpenSearch, 또는 내장 SQLite 저장소(호스트 한
  대, 인덱스당 문서 약 100,000개). 그 밖의 것은 `register_provider` 빌더나 사용자 정의
  어댑터가 필요합니다([설계 문서 §15](./design.md#15-extension-guide)).
- **서빙 계층이 없습니다.** HTTP API, 최종 사용자 인증, 요청 속도 제한은 없습니다.
- **필터는 접근 제어가 아닙니다**(§4). 테넌트는 인덱스 접미사로 분리하세요.
- **가벼운 수집은 쓸 수 있는 전략을 제한합니다.** 커뮤니티 탐지를 끄면
  (`graph.community_detection.enabled: false`) `global`과 `drift`가 쓸 리포트가 없고,
  관계 벡터 인덱스를 끄면(`indexing.opensearch.build_relationship_vector_index: false`)
  `mix`와 `hybrid`가 고수준 키워드를, `local`이 관계 섹션을 쓰지 못하고, 문서 필터가 걸린 그래프
  확장은 엔티티에 닿는 관계가 아니라 엔티티 인덱스 확인만으로 엔티티를 남깁니다.
- **증분 인덱싱은 근사치입니다**(§5). 증분 실행은 다시 군집화하지 않고 커뮤니티를
  덧붙이며, 엔티티가 현재 이름의 id를 받은 뒤에도 저장된 커뮤니티와 리포트에는 이전
  id가 남습니다. 둘 다 전체 재구축(`indexing.reset: true`)으로 갱신됩니다. DynamoDB에서는
  문서 하나의 청크 ID가 400KB 항목 하나에 들어가야 합니다([*문서 크기 한도*](#문서-크기-한도)).
- **동시 실행에는 레지스트리가 필요합니다.** 레지스트리의 쓰기 임대가 한
  네임스페이스의 두 번째 실행을 멈춥니다. 레지스트리가 없거나 `IncrementalIndexer`
  API를 직접 쓰면 잠그는 장치가 없어, 같은 접미사의 수집 두 개가 서로의 인덱스를 지울
  수 있습니다([동시 실행](./operations.md#concurrent-runs)).
- **태스크를 죽이는 문서를 가려내려면 여러 번 실행해야 합니다**([*격리된 문서*](#격리된-문서)). 함께
  죽은 문서 `n`개 가운데 원인 문서 하나에 대략 `log2(n) + 1`번이 걸리고, 그동안 미룬
  문서는 기다립니다.
- **Markdown과 HTML**에는 `unstructured` extra가 필요하며, 컨테이너
  이미지에는 기본으로 빠져 있습니다.
- **시드 그래프는 이름으로만 이어집니다**([*우선순위*](#seed-precedence),
  [§9 D](#d-시드-그래프가지고-있는-지식-그래프-가져오기)). 시드 노드와 이름이 같은 관계없는
  문서 엔티티는 한 엔티티가 됩니다. IRI는 저장하고 필터로 쓸 수 있지만 정체성이 아니며,
  `owl:sameAs`는 따라가지 않고 아무것도 추론하지 않습니다.
- **임베딩 모델(또는 차원)을 바꾸면** `indexing.reset: true`가 필요합니다. 증분 실행은
  다른 모델의 인덱스에 쓰지 않습니다([작업 흐름](#작업-흐름)).

---

*함께 보기: 개요와 빠른 시작은 [README.ko.md](../README.ko.md), 아키텍처와 내부 구조는
[설계 문서](./design.md)를 참고하세요.*

## 옮긴 절

이 가이드의 1절과 2절은 별도 문서로 옮겼습니다. 아래 제목은 이전 버전의 `--help`
출력을 비롯한 기존 링크가 계속 동작하도록 남겨 둔 것입니다.

### 1. 사전 요구사항과 설치

[AWS에서 시작하기](./getting-started/aws.ko.md)와
[로컬에서 시작하기](./getting-started/local.ko.md)로 옮겼습니다.

### 런타임

[AWS에서 시작하기: 런타임](./getting-started/aws.ko.md#런타임)으로 옮겼습니다.

### AWS 서비스

[AWS에서 시작하기: AWS 서비스](./getting-started/aws.ko.md#aws-서비스)로 옮겼습니다.

### 필수 모델

[AWS에서 시작하기: 필수 모델](./getting-started/aws.ko.md#필수-모델)로 옮겼습니다.

### 설치

[AWS에서 시작하기: 설치](./getting-started/aws.ko.md#설치)로 옮겼습니다.

### 인증

[AWS에서 시작하기: 인증](./getting-started/aws.ko.md#인증)으로 옮겼습니다.

### IAM 권한

[AWS에서 시작하기: IAM 권한](./getting-started/aws.ko.md#iam-권한)으로 옮겼습니다.

### 로컬 모드 (AWS 없이)

[로컬에서 시작하기](./getting-started/local.ko.md)(첫 실행)와
[Local mode](./guides/local-mode.md)(영문. 공급자, AWS 모드와 다른 점, 컨테이너)로 옮겼습니다.

### AWS 모드와 다른 점

[Local mode: Differences from AWS mode](./guides/local-mode.md#differences-from-aws-mode)(영문)로 옮겼습니다.

### 공급자 플래그

[Local mode: Provider flags](./guides/local-mode.md#provider-flags)(영문)로 옮겼습니다.

### 컨테이너로 띄운 로컬 저장소

[Local mode: Local stores in containers](./guides/local-mode.md#local-stores-in-containers)(영문)로 옮겼습니다.

### 로컬 저장소와 Bedrock 모델

[Local mode: Local stores with Bedrock models](./guides/local-mode.md#local-stores-with-bedrock-models)(영문)로 옮겼습니다.

### 2. 설정

[설정](./guides/configure.ko.md)으로 옮겼습니다.

### 설정을 읽는 순서

[설정: 설정을 읽는 순서](./guides/configure.ko.md#설정을-읽는-순서)로 옮겼습니다.

### 대부분의 실행에서 바꾸는 설정

[설정: 대부분의 실행에서 바꾸는 설정](./guides/configure.ko.md#대부분의-실행에서-바꾸는-설정)으로 옮겼습니다.

### 모델 선택 참고 사항

[Model Catalog: Model selection notes](./reference/models.md#model-selection-notes)(영문)로 옮겼습니다.

### 가드레일

[Guardrail](./guides/guardrail.md)(영문)로 옮겼습니다.

### S3 캐시 암호화

[설정: S3 캐시 암호화](./guides/configure.ko.md#s3-캐시-암호화)로 옮겼습니다.

### 로그에 남는 내용

[설정: 로그에 남는 내용](./guides/configure.ko.md#로그에-남는-내용)으로 옮겼습니다.

### 사용자 정의 프롬프트

[B. `custom_prompts` 재정의](#b-custom_prompts-재정의)로 옮겼습니다.

### 환경 변수로 덮어쓰기

[설정: 환경 변수로 덮어쓰기](./guides/configure.ko.md#환경-변수로-덮어쓰기)로 옮겼습니다.

### 관측성

[Observability](./guides/observability.md)(영문. 설정과 절차)와
[Telemetry reference](./reference/telemetry.md)(영문. 트레이스 구조, 사용량 집계)로 옮겼습니다.
