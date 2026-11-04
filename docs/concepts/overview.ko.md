# 개요

> 🇬🇧 English version: [docs/concepts/overview.md](./overview.md)

`unified-kg-rag-on-aws`는 대규모 다국어 문서 코퍼스를 Amazon Neptune과 Amazon OpenSearch
Service 위의 지식 그래프로 만들고, Amazon Bedrock으로 그 그래프에 대한 질문에 답합니다.
두 그래프 RAG 방법론인 Microsoft GraphRAG(커뮤니티 요약)와 LightRAG(이중 레벨 키워드)를
하나의 공통 스택 위에 구현했기 때문에, 방법론을 배포 단위가 아니라 질의 단위로 고릅니다.
모든 답변은 실제로 모델 컨텍스트에 들어간 출처를 함께 돌려줍니다.

이 페이지는 구성 요소와 그 관계를 설명합니다. 실행은 [README](../../README.ko.md#빠른-시작)나
[문서 안내](../index.ko.md)에서 시작하세요.

## 아키텍처 한눈에 보기

![아키텍처 한눈에 보기](../../assets/architecture-overview.drawio.png)

- **수집**(`run-ingestion`)은 코퍼스를 파싱하고, LLM으로 지식 그래프를 추출하고, 그래프를
  커뮤니티로 묶어 LLM이 쓴 리포트를 붙인 뒤, 모두 그래프 저장소와 검색 저장소에 씁니다.
- **저장소.** AWS에서는 그래프에 Neptune, BM25와 벡터 검색에 OpenSearch, 수집을 증분으로
  만드는 문서 상태 레지스트리에 DynamoDB, 단계 캐시에 S3를 씁니다. 로컬 모드에서는 내장
  SQLite 저장소나 컨테이너로 띄운 Gremlin Server와 OpenSearch를 씁니다.
- **모델.** 기본으로 채팅, 임베딩, 재순위화(rerank) 모델 모두 Bedrock을 쓰고, 로컬 모드에서는
  Ollama나 OpenAI 호환 서버를 씁니다. 모든 LLM 역할이 두 계층 중 하나에서 실행되므로 설정
  하나로 파이프라인 전체를 바꿀 수 있습니다.
- **검색**(`run-rag` 또는 Python 체인)은 같은 저장소와 같은 하이브리드 점수 계산기 위에서
  전략 8개 중 하나를 실행하고, 결과를 답변 모델의 토큰 예산에 맞춰 컨텍스트로 담습니다.
- **평가**(`run-eval`)는 LLM 없이 계산하는 지표와 LLM 판정 모델로 답변을 채점하고, 전략이나
  실행을 짝지어 비교하며, 질의마다 토큰, 비용, 지연 시간을 측정합니다.

코드는 헥사고날(포트와 어댑터) 구조입니다. 알고리즘은 인터페이스에만 의존하고 AWS와 로컬
백엔드는 그 뒤에 있으므로, 저장소, 모델 공급자, 전략, 파서, 렌더러를 분기 코드를 고치지
않고 레지스트리에 등록해 추가합니다([설계 문서 §2](../design.md#2-hexagonal-architecture-ports--adapters),
[§15 확장 가이드](../design.md#15-extension-guide), 영문).

## 파이프라인

수집은 체크포인트를 남기는 12단계 파이프라인이며, 다시 실행하면 마지막으로 유효한 단계
다음부터 이어 갑니다.

1. 문서를 파싱하고(PDF, TXT, CSV, JSON. Markdown과 HTML은 `unstructured` 추가 패키지 필요)
   이번 실행의 코퍼스를 불러옵니다.
2. 텍스트를 텍스트 단위로 나누고, 필요하면 번역합니다.
3. 텍스트 단위마다 엔티티와 관계를 추출하고, 첫 추출에서 빠진 것을 글리닝(gleaning)으로
   보충하고, 이름 변형을 하나의 엔티티로 합칩니다.
4. 필요하면 사실 주장(claim)을 추출하고 정리합니다.
5. 그래프를 만들고, Leiden 커뮤니티 계층을 찾아 커뮤니티마다 구조화된 리포트를 씁니다.
6. 인덱싱합니다. 엔티티, 관계, 텍스트 단위, 주장, 리포트는 OpenSearch(BM25와 벡터)에,
   그래프는 Neptune에, 문서별 상태는 레지스트리에 씁니다.

문서 상태 레지스트리를 켜면 새 문서와 바뀐 문서만 다시 인덱싱하고 지운 문서는 제거합니다.
단계 목록과 플래그는 [수집](../guides/ingest.ko.md#파이프라인-단계-12개), 내부 구조는
[설계 문서 §4](../design.md#4-ingestion-pipeline)와 [§5](../design.md#5-incremental-indexing)(영문)에
있습니다.

## 두 방법론

두 방법론은 같은 인덱스를 읽고 검색 알고리즘만 다릅니다.

| | GraphRAG(커뮤니티 요약) | LightRAG(이중 레벨 키워드) |
|---|---|---|
| 전략 | `local`, `global`, `drift`, `simple`, `auto` 라우터 | `mix`, `hybrid`, `naive` |
| 읽는 것 | 엔티티와 그 그래프 이웃, 커뮤니티 리포트, 텍스트 단위, 주장 | 엔티티(저수준 키워드), 관계(고수준 키워드), 그 원문 청크 |
| 잘하는 질문 | 코퍼스 전체에 걸친 주제·의미 파악 질문(`global`, `drift`), 특정 엔티티 질문(`local`) | 키워드 중심 조회, 빠르고 저렴한 기준선 |

모든 전략은 같은 하이브리드 점수 계산기를 거칩니다. BM25, 벡터 검색, 그래프 순회(`simple`과
`naive`는 제외)를 RRF(Reciprocal Rank Fusion)로 합치고, 재순위화하고, 다양성을 맞춥니다.
기본값인 `auto`는 LLM 라우터가 질의마다 `local`, `mix`, `global` 중 하나를 고릅니다. 언제 어떤
전략을 쓸지는 [검색](./retrieval.ko.md#검색-전략-선택), 전략별 검색 방식은
[설계 문서 §6](../design.md#6-retrieval-two-methodologies)(영문)에 있습니다.

## 원 논문을 넘어서

이 프레임워크는 각 방법론의 검색 알고리즘을 유지하고, AWS에서 실제 코퍼스를 운영하는 데
필요한 부분을 그 주변에 더했습니다. upstream 열은 참조 라이브러리인
[microsoft/graphrag](https://github.com/microsoft/graphrag) 3.1.0과
[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 1.5.4의 소스 코드를 읽고 정리한
내용입니다. 칸에 적은 파일이 확인할 위치이며, 저장소 루트에서 시작하는 경로(`docs/`,
`reproduce/`)가 아니면 라이브러리의 Python 패키지(`graphrag/`, `lightrag/`) 기준입니다.

| 영역 | GraphRAG | LightRAG | 이 프레임워크 | 문서 |
|---|---|---|---|---|
| 3중 하이브리드 검색 | 엔티티 설명, 텍스트 단위, 리포트 내용에 대한 벡터 검색(`config/embeddings.py`)과 인덱싱된 테이블에서 읽는 그래프 컨텍스트. 질의 코드에 어휘(lexical) 검색이나 재순위화 모델은 없음 | 엔티티, 관계, 청크 인덱스에 대한 벡터 검색과 그래프 저장소를 라운드 로빈으로 합치고, 재순위화 모델은 선택(`operate.py`, `rerank.py`) | BM25(OpenSearch), 벡터 검색(Bedrock 임베딩), Neptune 그래프 순회를 RRF로 합치고, Bedrock rerank 모델로 재순위화하고, MMR로 다양성을 맞춤. 두 방법론의 모든 전략이 점수 계산기 하나를 씀(`simple`과 `naive`는 그래프 제외) | [설계 문서 §7](../design.md#7-hybrid-scoring-and-token-management)(영문) |
| 한 인덱스에서 두 방법론 | 커뮤니티 리포트. 관계 벡터 인덱스(`config/embeddings.py`)와 키워드 수준 검색은 없음 | 엔티티와 관계 인덱스에 대한 이중 레벨 키워드. 커뮤니티 탐지와 리포트는 없음 | 한 번의 수집이 두 방법론에 필요한 것을 모두 만들고, 방법론은 질의마다 직접 고르거나 `auto` 라우터가 고름 | [설계 문서 §6](../design.md#6-retrieval-two-methodologies)(영문) |
| 삭제 계보(lineage)를 갖춘 증분 인덱싱 | `graphrag update`는 제목이 새로운 문서를 인덱싱함. 계산한 차이에 삭제된 문서도 들어 있지만 갱신 워크플로에는 새 문서만 전달됨(`index/update/incremental_index.py`, `index/workflows/load_update_documents.py`) | 문서 상태 저장소는 파일 이름이나 내용 해시가 이미 등록된 문서의 삽입을 건너뜀(`pipeline.py`). `adelete_by_doc_id`는 문서를 지우고, 다른 문서가 여전히 뒷받침하는 엔티티와 관계는 캐시된 추출 결과로 다시 만듦(`lightrag.py`) | 레지스트리(DynamoDB, 로컬에서는 SQLite)가 내용 해시로 코퍼스를 비교해 새 문서와 바뀐 문서를 다시 인덱싱하고, 지운 문서는 각 항목이 인용한 텍스트 단위를 따라 제거해 남은 텍스트 단위가 기술하지 않는 것만 지움. 선기록(write-ahead) 장애 복구, 네임스페이스마다 쓰기 실행 하나. 속성 기반 테스트로 증분 실행이 전체 구축과 같은 결과에 수렴하는지 확인 | [사용자 가이드 §5](../user-guide.ko.md#5-증분-인덱싱), [설계 문서 §5](../design.md#5-incremental-indexing)(영문) |
| 다국어 | `graphrag prompt-tune --language`가 튜닝한 프롬프트의 언어를 정함(`cli/main.py`) | `SUMMARY_LANGUAGE`가 추출과 요약의 출력 언어를 정함(`addon_params.py`) | 인덱싱과 질의 시점의 선택적 번역 단계(추가 대상 언어 포함), 언어별 OpenSearch 분석기(한국어는 `nori`), 다국어 키워드 추출, 한국어·CJK를 고려한 답변 일치 채점을 두 방법론 모두에 적용 | [다국어 코퍼스](../guides/multilingual.ko.md), [설계 문서 §8.3](../design.md#83-multilingual-processing)(영문) |
| 온톨로지와 시드 그래프 가져오기 | Bring Your Own Graph: 추출 대신 엔티티와 관계 테이블을 넣고 커뮤니티 워크플로를 실행(`docs/index/byog.md`) | `insert_custom_kg`가 주어진 청크, 엔티티, 관계를 삽입(`lightrag.py`) | 직접 선언하거나 OWL/RDFS 온톨로지에서 읽은 추출 스키마(유형, 패턴, IRI). 시드 그래프(JSONL, CSV, RDF)를 코퍼스의 가상 문서로 추출 그래프에 병합하므로 증분 실행과 삭제가 파일처럼 다루고, 문서가 기준(canonical)으로 남음 | [추출 스키마](../user-guide.ko.md#c-추출-스키마), [온톨로지](../user-guide.ko.md#owlrdfs-온톨로지에서-읽기), [시드 그래프](../user-guide.ko.md#d-시드-그래프가지고-있는-지식-그래프-가져오기) |
| 측정으로 정한 기본값과 평가 도구 | 논문은 답변을 맞비교로 판정함(포괄성, 다양성, 역량 강화, 직접성). 라이브러리 패키지에는 평가 명령이 없음 | RAGAS 평가 스크립트(`evaluation/eval_rag_quality.py`)와 논문의 맞비교 판정 프롬프트(`reproduce/batch_eval.py`) | LangChain·RAGAS 판정 모델과 함께 쓰는 LLM 없는 평가기, 반복 실행과 부트스트랩 신뢰 구간을 갖춘 짝지은 비교, 순서를 바꿔 판정하고 위치 일관성을 보고하는 맞비교 판정, 측정한 비용과 지연 시간, CI 게이트, MuSiQue 벤치마크. 기본값의 근거가 되는 측정 결과를 공개함 | [사용자 가이드 §6](../user-guide.ko.md#6-평가-run-eval), [기본값과 근거](./defaults-and-evidence.md)(영문) |

소스 코드로 확인한 범위에서는 두 라이브러리 모두 OWL이나 RDF를 읽지 않고, 인덱싱
파이프라인에 문서 번역 단계가 없습니다.

### 의도적으로 다르게 한 부분

upstream의 기본값이나 알고리즘과 일부러 다르게 한 부분입니다.

- **관계 식별자.** 관계 유형이 ID에 포함되고 에지에 방향이 있으므로, 한 쌍 사이의 유형 두
  개는 에지 두 개입니다. MS GraphRAG는 관계를 출발점과 도착점으로 묶고(`extract_graph.py`),
  LightRAG는 정렬한 양 끝점 쌍을 에지 키로 씁니다(`operate.py`)
  ([설계 문서 §4.2](../design.md#42-graph-identity-and-merging-stages-5-7), 영문).
- **커뮤니티 계층.** 레벨 0은 그래프 전체에 대한 Leiden 분할 하나이고, 그보다 상위 레벨은
  커뮤니티 그래프를 다시 분할하며, 구성원이 20개를 넘는 커뮤니티는 다시 나눕니다.
  `min_community_size`보다 작은 연결 요소만 커뮤니티를 받지 않습니다. MS GraphRAG는
  `max_cluster_size` 10으로 계층적 Leiden을 실행하고, 기본값(`use_lcc`)으로는 가장 큰 연결
  요소만 군집화합니다(`config/defaults.py`)
  ([설계 문서 §4.3](../design.md#43-communities-stages-10-11), 영문).
- **글리닝.** 기본 라운드는 MS GraphRAG의 1회(`max_gleanings`)가 아니라 3회이고, 텍스트
  단위는 직전 답변이 새로운 것을 더했을 때만 다시 묻습니다. 그래프 재현율을 측정해 정했습니다
  ([근거](./defaults-and-evidence.md#gleaning-rounds), 영문).
- **컨텍스트 몫.** 토큰 예산을 `max_context_tokens`(30,000, LightRAG의 전체 예산)에 대한
  유형별 몫으로 나눕니다. LightRAG는 엔티티를 6,000, 관계를 8,000 토큰으로 제한하고
  (`constants.py`), 여기서는 `mix`와 `hybrid`가 텍스트 18,072, 엔티티 7,228, 관계 4,698을
  받습니다([설계 문서 §7](../design.md#7-hybrid-scoring-and-token-management), 영문).
- **LightRAG 청크 선택.** `kg_chunk_pick_method` 기본값은 `weight`이고, upstream LightRAG의
  기본값은 `VECTOR`입니다(`constants.py`). 두 방식은 측정에서 차이가 없었습니다
  ([근거](./defaults-and-evidence.md#lightrag-chunk-picking-and-mmr-diversity), 영문).
- **local 검색의 리포트.** `local`은 질의 엔티티와 가장 비슷한 리포트를 읽습니다. MS
  GraphRAG local은 엔티티가 속한 커뮤니티의 리포트를 읽으며(`mixed_context.py`), 여기서는
  `report_selection: membership`에 해당하고 측정 결과가 더 나빴습니다
  ([근거](./defaults-and-evidence.md#local-report-selection), 영문).
- **global 답변.** 답변 모델은 순위를 매긴 핵심 내용과, 그 내용이 인용한 리포트를 최대
  4,000 토큰까지 읽습니다. MS GraphRAG의 reduce 단계는 핵심 내용만 읽으며
  (`global_search/search.py`), 여기서는 `evidence_tokens: 0`에 해당합니다
  ([근거](./defaults-and-evidence.md#global-evidence-cap), 영문).
- **전략 이름.** `local`과 `global`은 GraphRAG의 전략입니다. LightRAG 자체의 `local`과
  `global` 모드(`base.py`)는 따로 제공하지 않으며, `hybrid`는 upstream과 마찬가지로 두 키워드
  수준을 함께 씁니다.

## 제한 사항

주요 제한 사항을 짧게 정리하면 다음과 같습니다.

- 추출 과정에서 엔티티나 관계를 지어낼 수 있고, LLM 출력이 매번 달라 실행 결과를 그대로
  재현할 수 없습니다. 설정은 여러 번 실행해 비교하세요.
- 서빙 계층(HTTP API, 최종 사용자 인증, 요청 속도 제한)이 없고, 필터는 접근 제어가
  아닙니다. 테넌트는 인덱스 접미사로 나누세요.
- 증분 인덱싱은 주기적으로 재구축하기 전까지 커뮤니티를 전체 구축과 근사하게만 맞추며,
  동시 실행에는 레지스트리의 쓰기 임대(lease)가 필요합니다.
- 수집을 가볍게 하면(커뮤니티 없음, 관계 인덱스 없음) 실행할 수 있는 전략이 줄어듭니다.

전체 목록은 [사용자 가이드 §12](../user-guide.ko.md#12-제한-사항)에 있습니다.
