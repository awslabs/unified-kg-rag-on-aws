# 개요

> 🇬🇧 English version: [docs/concepts/overview.md](./overview.md)

`unified-kg-rag-on-aws`는 대규모 다국어 문서 코퍼스를 Amazon Neptune과 Amazon OpenSearch
Service 위의 지식 그래프로 만들고, Amazon Bedrock으로 그 그래프에 대한 질문에 답합니다.
네 가지 그래프 RAG 방법론, 즉 Microsoft GraphRAG(커뮤니티 요약), LightRAG(이중 레벨 키워드),
HippoRAG 2(엔티티와 패시지에 대한 개인화 PageRank), 그리고 선택 사항인 RAPTOR(요약 트리)를
하나의 공통 스택 위에 구현했습니다. 한 번의 수집으로 GraphRAG와 LightRAG의 인덱스(HippoRAG 2도
이것을 읽음)를 함께 만들어 질의마다 방법론을 고를 수도 있고, 고른 방법론만 만들어 나머지
방법론의 비용을 아낄 수도 있습니다. 모든 답변은 실제로 모델 컨텍스트에 들어간 출처를 함께 돌려줍니다.

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
- **검색**(`run-rag` 또는 Python 체인)은 같은 저장소 위에서 전략 11개 중 하나나 `auto`
  라우터를 실행하고, 결과를 답변 모델의 토큰 예산에 맞춰 컨텍스트로 담습니다.
- **평가**(`run-eval`)는 LLM 없이 계산하는 지표와 LLM 판정 모델로 답변을 채점하고, 전략이나
  실행을 짝지어 비교하며, 질의마다 토큰, 비용, 지연 시간을 측정합니다([평가](../guides/evaluate.ko.md)).

코드는 헥사고날(포트와 어댑터) 구조입니다. 알고리즘은 인터페이스에만 의존하고 AWS와 로컬
백엔드는 그 뒤에 있으므로, 저장소, 모델 공급자, 전략, 파서, 렌더러를 분기 코드를 고치지
않고 레지스트리에 등록해 추가합니다([설계 문서 §2](../design/architecture.md#2-hexagonal-architecture-ports--adapters),
[§15 확장 가이드](../guides/extend.md#15-extension-guide), 영문).

## 파이프라인

수집은 체크포인트를 남기는 13단계 파이프라인이며, 다시 실행하면 마지막으로 유효한 단계
다음부터 이어 갑니다.

1. 문서를 파싱하고(PDF, TXT, CSV, JSON, Markdown, HTML. Word는 `docx` 추가 패키지 필요)
   이번 실행의 코퍼스를 불러옵니다.
2. 텍스트를 텍스트 단위로 나누고, 필요하면 번역합니다.
3. 텍스트 단위마다 엔티티와 관계를 추출하고, 첫 추출에서 빠진 것을 글리닝(gleaning)으로
   보충하고, 이름 변형을 하나의 엔티티로 합칩니다.
4. 필요하면 사실 주장(claim)을 추출하고 정리합니다.
5. 그래프를 만들고, Leiden 커뮤니티 계층을 찾아 커뮤니티마다 구조화된 리포트를 씁니다.
6. 인덱싱합니다. 엔티티, 관계, 텍스트 단위, 주장, 리포트는 OpenSearch(BM25와 벡터)에,
   그래프는 Neptune에, 문서별 상태는 레지스트리에 씁니다.
7. 선택 사항인 `raptor` 방법론이면 저장된 텍스트 단위 위에 요약 트리(RAPTOR)를 만듭니다.
   군집화하고 레벨마다 요약하며, `tree` 전략이 검색합니다.

문서 상태 레지스트리를 켜면 새 문서와 바뀐 문서만 다시 인덱싱하고 지운 문서는 제거합니다.
단계 목록과 플래그는 [수집](../guides/ingest.ko.md#파이프라인-단계), 내부 구조는
[설계 문서 §4](../design/ingestion.md#4-ingestion-pipeline)와 [§5](../design/incremental.md#5-incremental-indexing)(영문)에
있습니다.

## 방법론

방법론들은 같은 인덱스를 읽고 검색 알고리즘만 다릅니다.

| | GraphRAG(커뮤니티 요약) | LightRAG(이중 레벨 키워드) | HippoRAG 2(개인화 PageRank) | RAPTOR(요약 트리, 선택 사항) |
|---|---|---|---|---|
| 전략 | `local`, `global`, `drift`, `simple` | `mix`, `hybrid`, `naive` | `ppr` | `tree` |
| 읽는 것 | 엔티티와 그 그래프 이웃, 커뮤니티 리포트, 텍스트 단위, 주장 | 엔티티(저수준 키워드), 관계(고수준 키워드), 그 원문 청크 | 관계 검색 결과, 패시지 검색 결과, 그 엔티티와 패시지 사이의 그래프. LightRAG가 만드는 것 | 텍스트 단위와 그 위의 모든 레벨 요약 |
| 잘하는 질문 | 코퍼스 전체에 걸친 주제·의미 파악 질문(`global`, `drift`), 특정 엔티티 질문(`local`) | 키워드 중심 조회, 빠르고 저렴한 기준선 | 가장 적은 비용의 멀티홉 질문. 질의마다 LLM 호출 한 번 | 그래프 추출 없는 저렴한 인덱스. 멀티홉 질문에서는 `naive`보다 낫고 그래프 전략보다 낮은 답을 냄([근거](./defaults-and-evidence.md#raptor-summary-tree), 영문) |

`iterative`는 `mix`나 `local`을 감싸 멀티홉 질문이 기대는 연결(bridge) 엔티티로 정해진 횟수만큼
후속 라운드를 실행하며, 기반 전략이 읽는 것을 읽습니다. `local`은 기본으로 HippoRAG 2의
PageRank로 텍스트 단위의 순서를 정합니다.

`global`을 뺀 모든 전략은 같은 하이브리드 점수 계산기를 거칩니다. BM25, 벡터 검색, 그래프
신호(`simple`과 `naive`는 그래프를 읽지 않음)를 RRF(Reciprocal Rank Fusion)로 합치고,
재순위화하고, 다양성을 맞춥니다. `global`은 대신 커뮤니티 리포트를 핵심 요점으로 map합니다
(텍스트 단위는 `search.global_search.text_unit_slots`를 설정할 때만 합침).
기본값인 `auto`는 LLM 라우터가 질의마다 `local`, `mix`, `global`, `iterative`(다중 홉 질문) 중 하나를 고릅니다. 언제 어떤
전략을 쓸지는 [검색](./retrieval.ko.md#검색-전략-선택), 전략별 검색 방식은
[설계 문서 §6](../design/retrieval-graphrag-lightrag.md#6-retrieval-methodologies)(영문)에 있습니다.

### 방법론 프로필

`indexing.methodologies`가 수집에서 만들 것을 정합니다. 기본값 `[graphrag, lightrag]`는 한
인덱스에 두 방법론을 모두 만들고, 하나만 고르면 그 전략들이 읽는 것만큼만 비용을 냅니다.

| 프로필 | 만드는 것 | 전략 | 수집 단계의 모델 호출 |
|---|---|---|---|
| `[graphrag, lightrag]`(기본값) | 텍스트 단위, 엔티티, 관계와 그 벡터, 그래프, 커뮤니티와 리포트 | `tree`를 뺀 모든 전략 | 청크 경계, 텍스트 단위마다 추출과 글리닝, 설명 요약, 커뮤니티마다 리포트 하나 |
| `[lightrag]`나 `[hipporag]` | 커뮤니티와 리포트를 뺀 나머지 | `mix`, `hybrid`, `naive`, `simple`, `ppr`, `iterative`(`mix` 기반), `auto` | Leiden 단계와 리포트 호출을 뺀 나머지 |
| `[graphrag]` | 기본값과 같음. `local`이 읽는 관계 벡터가 `lightrag`와 `hipporag`에도 쓰임 | `tree`를 뺀 모든 전략. `indexing.opensearch.build_relationship_vector_index: false`이면 `local`, `global`, `drift`, `simple`, `naive`, `iterative`(`search.iterative_search.base: local`로 `local` 기반), `auto` | 기본값과 같음. 관계 벡터 인덱스를 끄면 관계 임베딩을 아끼는 대신 `local`의 관계 섹션과 `lightrag`·`hipporag`의 모든 전략이 빠짐 |
| `[raptor]`(선택 사항) | 텍스트 단위와 그 위의 요약 트리(RAPTOR) | `tree`, `naive`, `simple`, `auto`(`tree`를 씀) | 청크 경계와 텍스트 단위 5-6개마다 요약 하나 정도. 그래프 추출 없음([요약 트리](../guides/ingest.ko.md#raptor-요약-트리선택-사항)) |

다른 프로필에 `raptor`를 더하면 트리도 함께 만듭니다.
`hipporag`는 `lightrag`가 만드는 것과 같은 것을 가리키며, 그것을 읽는 전략(`ppr`)에 씁니다.
`iterative`는 기반 전략(`local`이나 `mix`)이 읽는 것을 읽습니다. `naive`는 텍스트 단위만,
`simple`은 네임스페이스가 가진 인덱스를 읽으므로 모든 프로필에서 쓸 수 있습니다.
인덱스 네임스페이스는 자신이 가진 것을 기록합니다. 질의한 전략이 네임스페이스에 없는 방법론을
읽으면 모델을 호출하기 전에 실패하고, `auto`는 가진 방법론의 전략 중에서만 고릅니다. 나중에
방법론을 추가하거나 뺄 때 무엇을 다시 만들고, 무엇을 재사용하며, 비용이 얼마나 드는지는
[수집: 방법론 프로필](../guides/ingest.ko.md#방법론-프로필)에 있습니다(내부 동작:
[설계 문서 §4.6](../design/ingestion.md#46-methodology-profiles), 영문).

## Amazon Bedrock Knowledge Bases GraphRAG와 비교

[Amazon Bedrock Knowledge Bases GraphRAG](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-graphs.html)(영문)는
AWS에서 그래프 RAG를 관리형으로 운영하는 방법입니다. Amazon S3의 문서로 Amazon Neptune
Analytics에 그래프를 만들고 유지하며, 직접 운영할 그래프 인프라가 없습니다. 이 프레임워크는
자기 계정에서 직접 실행하고 운영하는 코드이고, 그 대가로 그래프를 만들고 검색하는 방식을
직접 정할 수 있습니다. 관리형 서비스 열은 Amazon Bedrock 문서(2026-11-07 기준)를 옮긴 것이므로,
달라진 점은 링크한 문서에서 확인하세요.

| | Amazon Bedrock Knowledge Bases GraphRAG | 이 프레임워크 |
|---|---|---|
| 운영 | 완전 관리형. 지식 기반이 엔티티, 사실, 관계를 추출해 그래프를 만들고 갱신합니다([Neptune Analytics 그래프로 지식 기반 만들기](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-graphs-build.html), 영문) | 자기 Neptune 클러스터와 OpenSearch 도메인, 또는 번들 CDK 스택 위에서 CLI나 Python 라이브러리를 직접 실행합니다. 서빙 계층은 들어 있지 않습니다([제한 사항](#제한-사항)) |
| 저장소 | 그래프와 벡터 모두 Neptune Analytics. Neptune Analytics 그래프의 자동 확장은 지원하지 않습니다([고려 사항과 제한](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-graphs.html), 영문) | 그래프는 Neptune, BM25와 벡터는 OpenSearch, 문서 상태 레지스트리는 DynamoDB. 로컬 모드에서는 내장 SQLite 저장소 |
| 검색 | 한 가지 방법. 벡터 검색 후 찾은 청크에 연결된 그래프 노드를 가져와 그래프 순회로 넓힙니다([GraphRAG 동작 방식](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-graphs.html), 영문) | 네 가지 방법(GraphRAG, LightRAG, HippoRAG 2, RAPTOR)과 `iterative`를 질의마다 고르거나 `auto` 라우터에 맡깁니다([검색](./retrieval.ko.md)) |
| 키워드 검색 | 벡터와 원문을 함께 검색하는 하이브리드 검색은 필터링 가능한 텍스트 필드가 있는 Amazon RDS, OpenSearch Serverless, MongoDB 벡터 저장소에서만 지원하며, 다른 저장소는 의미 검색을 씁니다([검색 유형](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html), 영문) | `global`을 뺀 모든 전략에서 BM25, 벡터 검색, 그래프 신호를 RRF로 합치고 재순위화합니다 |
| 그래프 구축 제어 | 그래프 구축을 바꾸는 설정은 지원하지 않으며, 그래프 구축 모델만 고릅니다 | 추출 스키마나 OWL/RDFS 온톨로지, 도메인에 맞춘 프롬프트, 이미 가진 그래프의 시드 그래프, 만들 방법론을 정합니다([Extraction schema](../guides/extraction-schema.md), [Seed graphs](../guides/seed-graphs.md), 영문) |
| 데이터 소스 | Amazon S3만 지원하며, 데이터 소스당 파일 1,000개까지(한도 상향 요청 시 10,000개) | 디렉터리의 파일(CDK 수집 파이프라인은 코퍼스를 S3에서 복사). PDF, TXT, CSV, JSON, 그리고 `unstructured` extra로 Markdown과 HTML |
| 갱신 | 동기화는 증분 방식이어서 추가, 수정, 삭제된 문서만 다시 수집하거나 지웁니다([데이터 동기화](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-data-source-sync-ingest.html), 영문) | 내용 해시로 증분 인덱싱하며, 삭제한 문서는 남은 텍스트 단위가 기술하지 않는 내용만 지웁니다([증분 인덱싱](../guides/incremental.ko.md)) |
| 답변의 근거 | `RetrieveAndGenerate`가 응답 구간과 그 근거 문서를 묶은 인용(citation)을 돌려줍니다([Citation](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Citation.html), 영문) | 모든 답변이 모델 컨텍스트에 들어간 근거 문단을 함께 돌려줍니다 |
| 측정 | — | LLM 없는 지표, LLM 심사, 짝 비교를 갖춘 `run-eval`. 기본값은 공개한 측정 결과로 정했습니다([기본값과 근거](./defaults-and-evidence.md), 영문) |
| 리전 | 7개 리전(미국 동부(버지니아 북부), 미국 서부(오레곤), 유럽(프랑크푸르트, 런던, 아일랜드), 아시아 태평양(도쿄, 싱가포르)) | 설정한 Bedrock 모델과 Neptune, OpenSearch가 있는 모든 리전. 저장소와 다른 리전의 Bedrock도 쓸 수 있습니다([필수 모델](../getting-started/aws.ko.md#필수-모델)) |

S3에 있는 코퍼스, 한 가지 검색 방법, 운영할 인프라가 없는 구성으로 충분하면 관리형 서비스를
고르세요. 방법을 고르거나 비교해야 할 때, 벡터와 함께 키워드 검색이 필요할 때, 그래프에서
추출할 내용을 직접 정해야 할 때, 코퍼스가 S3에 있지 않을 때는 이 프레임워크가 맞습니다.

## 원 논문을 넘어서

이 프레임워크는 각 방법론의 검색 알고리즘을 유지하고, AWS에서 실제 코퍼스를 운영하는 데
필요한 부분을 그 주변에 더했습니다. upstream 열은 참조 라이브러리인
[microsoft/graphrag](https://github.com/microsoft/graphrag) 3.1.0과
[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 1.5.4의 소스 코드를 읽고 정리한
내용입니다. 칸에 적은 파일이 확인할 위치이며, 저장소 루트에서 시작하는 경로(`docs/`,
`reproduce/`)가 아니면 라이브러리의 Python 패키지(`graphrag/`, `lightrag/`) 기준입니다.

| 영역 | GraphRAG | LightRAG | 이 프레임워크 | 문서 |
|---|---|---|---|---|
| 3중 하이브리드 검색 | 엔티티 설명, 텍스트 단위, 리포트 내용에 대한 벡터 검색(`config/embeddings.py`)과 인덱싱된 테이블에서 읽는 그래프 컨텍스트. 질의 코드에 어휘(lexical) 검색이나 재순위화 모델은 없음 | 엔티티, 관계, 청크 인덱스에 대한 벡터 검색과 그래프 저장소를 라운드 로빈으로 합치고, 재순위화 모델은 선택(`operate.py`, `rerank.py`) | BM25(OpenSearch), 벡터 검색(Bedrock 임베딩), Neptune 그래프 순회를 RRF로 합치고, Bedrock rerank 모델로 재순위화하고, MMR로 다양성을 맞춤. 커뮤니티 리포트를 map하는 `global`을 뺀 모든 전략이 점수 계산기 하나를 씀(`simple`과 `naive`는 그래프 제외) | [설계 문서 §7](../design/scoring-and-citations.md#7-hybrid-scoring-and-token-management)(영문) |
| 한 인덱스에서 여러 방법론 | 커뮤니티 리포트. 관계 벡터 인덱스(`config/embeddings.py`)와 키워드 수준 검색은 없음 | 엔티티와 관계 인덱스에 대한 이중 레벨 키워드. 커뮤니티 탐지와 리포트는 없음 | 한 번의 수집이 고른 방법론(기본값은 GraphRAG와 LightRAG이며, 이것으로 HippoRAG 2의 `ppr`도 씀)에 필요한 것을 모두 만들고, 방법론은 질의마다 직접 고르거나 `auto` 라우터가 고르며, 네임스페이스에 없는 방법론은 거절함 | [방법론 프로필](#방법론-프로필), [설계 문서 §6](../design/retrieval-graphrag-lightrag.md#6-retrieval-methodologies)(영문) |
| 삭제 계보(lineage)를 갖춘 증분 인덱싱 | `graphrag update`는 제목이 새로운 문서를 인덱싱함. 계산한 차이에 삭제된 문서도 들어 있지만 갱신 워크플로에는 새 문서만 전달됨(`index/update/incremental_index.py`, `index/workflows/load_update_documents.py`) | 문서 상태 저장소는 파일 이름이나 내용 해시가 이미 등록된 문서의 삽입을 건너뜀(`pipeline.py`). `adelete_by_doc_id`는 문서를 지우고, 다른 문서가 여전히 뒷받침하는 엔티티와 관계는 캐시된 추출 결과로 다시 만듦(`lightrag.py`) | 레지스트리(DynamoDB, 로컬에서는 SQLite)가 내용 해시로 코퍼스를 비교해 새 문서와 바뀐 문서를 다시 인덱싱하고, 지운 문서는 각 항목이 인용한 텍스트 단위를 따라 제거해 남은 텍스트 단위가 기술하지 않는 것만 지움. 선기록(write-ahead) 장애 복구, 네임스페이스마다 쓰기 실행 하나. 속성 기반 테스트로 증분 실행이 전체 구축과 같은 결과에 수렴하는지 확인 | [증분 인덱싱](../guides/incremental.ko.md), [설계 문서 §5](../design/incremental.md#5-incremental-indexing)(영문) |
| 다국어 | `graphrag prompt-tune --language`가 튜닝한 프롬프트의 언어를 정함(`cli/main.py`) | `SUMMARY_LANGUAGE`가 추출과 요약의 출력 언어를 정함(`addon_params.py`) | 인덱싱과 질의 시점의 선택적 번역 단계(추가 대상 언어 포함), 언어별 OpenSearch 분석기(한국어는 `nori`), 다국어 키워드 추출, 한국어·CJK를 고려한 답변 일치 채점을 모든 방법론에 적용 | [다국어 코퍼스](../guides/multilingual.ko.md), [설계 문서 §8.3](../design/aws-integration.md#83-multilingual-processing)(영문) |
| 온톨로지와 시드 그래프 가져오기 | Bring Your Own Graph: 추출 대신 엔티티와 관계 테이블을 넣고 커뮤니티 워크플로를 실행(`docs/index/byog.md`) | `insert_custom_kg`가 주어진 청크, 엔티티, 관계를 삽입(`lightrag.py`) | 직접 선언하거나 OWL/RDFS 온톨로지에서 읽은 추출 스키마(유형, 패턴, IRI). 시드 그래프(JSONL, CSV, RDF)를 코퍼스의 가상 문서로 추출 그래프에 병합하므로 증분 실행과 삭제가 파일처럼 다루고, 문서가 기준(canonical)으로 남음 | [Extraction schema](../guides/extraction-schema.md), [Ontology](../guides/ontology.md), [Seed graphs](../guides/seed-graphs.md)(영문) |
| 측정으로 정한 기본값과 평가 도구 | 논문은 답변을 맞비교로 판정함(포괄성, 다양성, 역량 강화, 직접성). 라이브러리 패키지에는 평가 명령이 없음 | RAGAS 평가 스크립트(`evaluation/eval_rag_quality.py`)와 논문의 맞비교 판정 프롬프트(`reproduce/batch_eval.py`) | LangChain·RAGAS 판정 모델과 함께 쓰는 LLM 없는 평가기, 반복 실행과 부트스트랩 신뢰 구간을 갖춘 짝지은 비교, 순서를 바꿔 판정하고 위치 일관성을 보고하는 맞비교 판정, 측정한 비용과 지연 시간, CI 게이트, MuSiQue 벤치마크. 기본값의 근거가 되는 측정 결과를 공개함 | [평가](../guides/evaluate.ko.md), [기본값과 근거](./defaults-and-evidence.md)(영문) |

소스 코드로 확인한 범위에서는 두 라이브러리 모두 OWL이나 RDF를 읽지 않고, 인덱싱
파이프라인에 문서 번역 단계가 없습니다.

### 의도적으로 다르게 한 부분

upstream의 기본값이나 알고리즘과 일부러 다르게 한 부분입니다.

- **관계 식별자.** 관계 유형이 ID에 포함되고 에지에 방향이 있으므로, 한 쌍 사이의 유형 두
  개는 에지 두 개입니다. MS GraphRAG는 관계를 출발점과 도착점으로 묶고(`extract_graph.py`),
  LightRAG는 정렬한 양 끝점 쌍을 에지 키로 씁니다(`operate.py`)
  ([설계 문서 §4.2](../design/ingestion.md#42-graph-identity-and-merging-stages-5-7), 영문).
- **커뮤니티 계층.** 그래프 전체에 대한 Leiden 분할 하나에서 구성원이 20개를 넘는
  커뮤니티는 다시 나누고, 나눈 커뮤니티는 조각들의 부모로 남깁니다(MS GraphRAG의 계층적
  Leiden과 같음). 그 위의 상위 레벨은 커뮤니티 그래프를 다시 분할합니다.
  `min_community_size`보다 작은 연결 요소만 커뮤니티를 받지 않습니다. MS GraphRAG는 첫
  분할에서 멈추고, `max_cluster_size` 10을 쓰며, 기본값(`use_lcc`)으로는 가장 큰 연결
  요소만 군집화합니다(`config/defaults.py`)
  ([설계 문서 §4.3](../design/ingestion.md#43-communities-stages-10-11), 영문).
- **글리닝.** 기본 라운드는 MS GraphRAG의 1회(`max_gleanings`)가 아니라 3회이고, 텍스트
  단위는 직전 답변이 새로운 것을 더했을 때만 다시 묻습니다. 그래프 재현율을 측정해 정했습니다
  ([근거](./defaults-and-evidence.md#gleaning-rounds), 영문).
- **컨텍스트 몫.** 토큰 예산을 `max_context_tokens`(30,000, LightRAG의 전체 예산)에 대한
  유형별 몫으로 나눕니다. LightRAG는 엔티티를 6,000, 관계를 8,000 토큰으로 제한하고
  (`constants.py`), 여기서는 `mix`와 `hybrid`가 텍스트 18,072, 엔티티 7,228, 관계 4,698을
  받습니다([설계 문서 §7](../design/scoring-and-citations.md#7-hybrid-scoring-and-token-management), 영문).
- **LightRAG 청크 선택.** `kg_chunk_pick_method` 기본값은 `weight`이고, upstream LightRAG의
  기본값은 `VECTOR`입니다(`constants.py`). 두 방식은 측정에서 차이가 없었습니다
  ([근거](./defaults-and-evidence.md#lightrag-chunk-picking-and-mmr-diversity), 영문).
- **local 검색의 리포트.** `local`은 질의 엔티티와 가장 비슷한 리포트를 읽습니다. MS
  GraphRAG local은 엔티티가 속한 커뮤니티의 리포트를 읽으며(`mixed_context.py`), 여기서는
  `report_selection: membership`이 이를 따르되 업스트림의 `community_level` 단면 대신 최하위
  커뮤니티를 읽습니다. 측정 결과는 더 나빴습니다
  ([근거](./defaults-and-evidence.md#local-report-selection), 영문).
- **global 답변.** 답변 모델은 순위를 매긴 핵심 내용과, 그 내용이 인용한 리포트를 최대
  4,000 토큰까지 읽습니다. MS GraphRAG의 reduce 단계는 핵심 내용만 읽으며
  (`global_search/search.py`), 여기서는 `evidence_tokens: 0`에 해당합니다
  ([근거](./defaults-and-evidence.md#global-evidence-cap), 영문).
- **전략 이름.** `local`과 `global`은 GraphRAG의 전략입니다. LightRAG 자체의 `local`과
  `global` 모드(`base.py`)는 따로 제공하지 않으며, `hybrid`는 upstream과 마찬가지로 두 키워드
  수준을 함께 씁니다.

## 제한 사항

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
- **서빙 계층이 없습니다.** HTTP API, 최종 사용자 인증, 요청 속도 제한은 없습니다.
- **필터는 접근 제어가 아닙니다**
  ([범위와 네임스페이스](./scopes-and-namespaces.ko.md#접근-제어-경계가-아님)).
  테넌트는 인덱스 접미사로 분리하세요.
- **기본 제공 백엔드.** 모델: Bedrock(기본값), Ollama, OpenAI 호환 서버. 재순위화:
  Bedrock이나 sentence-transformers 크로스 인코더([Local mode](../guides/local-mode.md), 영문).
  저장소: Neptune(또는 Gremlin Server)과 OpenSearch, 또는 내장 SQLite 저장소(호스트 한
  대, 인덱스당 문서 약 100,000개). 그 밖의 것은 `register_provider` 빌더나 사용자 정의
  어댑터가 필요합니다([설계 문서 §15](../guides/extend.md#15-extension-guide), 영문).
- **재순위화는 리전에 따라 다릅니다**([필수 모델](../getting-started/aws.ko.md#필수-모델)). 없으면 융합 순서를 쓰며,
  `metadata.rerank`에 나타납니다([재순위화 상태](../guides/query.ko.md#재순위화-상태)).
- **가벼운 수집은 쓸 수 있는 전략을 제한합니다.** 커뮤니티가 없는 네임스페이스
  (`indexing.methodologies: [lightrag]`나 `graph.community_detection.enabled: false`)에는
  `graphrag`가 없으므로 `local`, `global`, `drift`가 거절되고(`simple`은 가진 인덱스만 훑음), 관계 벡터 인덱스를 끄면(`indexing.opensearch.build_relationship_vector_index: false`)
  네임스페이스에 `lightrag`도 `hipporag`도 없어 `mix`, `hybrid`, `ppr`이 거절되고, `local`이 관계 섹션을 쓰지 못하며, 문서 필터가 걸린 그래프
  확장은 엔티티에 닿는 관계가 아니라 엔티티 인덱스 확인만으로 엔티티를 남깁니다.
  `graph.community_detection.report_generation.enabled: false`이면 커뮤니티는 탐지하지만 리포트는
  쓰지 않습니다. `local`은 리포트 섹션 없이 답하고, 리포트를 읽는 `global`과 `drift`는 이 스위치를
  알려 주는 메시지와 함께 거절됩니다.
- **Markdown과 HTML**은 기본 설치에서 텍스트로 읽습니다(Markdown은 쓰인 그대로, HTML은 표준
  라이브러리 파서로). `unstructured` extra는 둘을 더 정교하게 파싱하고, Word(`.docx`)에는 `docx`
  extra가 필요합니다. 컨테이너 이미지에는 둘 다 기본으로 빠져 있습니다([Troubleshooting](../troubleshooting.md#which-source-formats-are-read), 영문).
- **증분 인덱싱은 전체 구축의 근사치입니다**([재구축 신호](../guides/incremental.ko.md#재구축-신호)). 증분 실행은 다시 군집화하지 않고 커뮤니티를
  덧붙이며, 엔티티가 현재 이름의 id를 받은 뒤에도 저장된 커뮤니티와 리포트에는 이전
  id가 남습니다. 둘 다 전체 재구축(`indexing.reset: true`)으로 갱신됩니다. DynamoDB에서는
  문서 하나의 청크 ID가 400KB 항목 하나에 들어가야 합니다([*문서 크기 한도*](../guides/incremental.ko.md#문서-크기-한도)).
- **동시 실행에는 레지스트리가 필요합니다.** 레지스트리의 쓰기 임대(lease)가 한
  네임스페이스의 두 번째 실행을 멈춥니다. 레지스트리가 없거나 `IncrementalIndexer`
  API를 직접 쓰면 잠그는 장치가 없어, 같은 접미사의 수집 두 개가 서로의 인덱스를 지울
  수 있습니다([동시 실행](../operations.md#concurrent-runs), 영문).
- **태스크를 죽이는 문서를 가려내려면 여러 번 실행해야 합니다**([*격리된 문서*](../guides/incremental.ko.md#격리된-문서)). 함께
  죽은 문서 `n`개 가운데 원인 문서 하나에 대략 `log2(n) + 1`번이 걸리고, 그동안 미룬
  문서는 기다립니다.
- **임베딩 모델(또는 차원)을 바꾸면** `indexing.reset: true`가 필요합니다. 증분 실행은
  다른 모델의 인덱스에 쓰지 않습니다([작업 흐름](../guides/incremental.ko.md#작업-흐름)).
- **시드 그래프는 이름으로만 이어집니다**
  ([Seed graphs: Precedence](../guides/seed-graphs.md#precedence-documents-decide), 영문).
  시드 노드와 이름이 같은 관계없는 문서 엔티티는 한 엔티티가 됩니다. IRI는 저장하고
  필터로 쓸 수 있지만 정체성이 아니며, `owl:sameAs`는 따라가지 않고 아무것도 추론하지
  않습니다.
