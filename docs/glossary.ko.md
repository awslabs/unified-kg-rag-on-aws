# 용어집 (한국어 ↔ English)

한국어 문서([README.ko.md](../README.ko.md), [사용자 가이드](./user-guide.ko.md),
[설계 문서](./design.ko.md), [모델 카탈로그](./models.ko.md),
[운영 런북](./operations.ko.md))에서 쓰는 용어와 영문 문서·코드의 용어를 대응시킨
표입니다. 설정 키, CLI 플래그, 클래스 이름처럼 코드에 그대로 나오는 이름은
번역하지 않습니다.

| 한국어 | English | 설명 |
|---|---|---|
| 수집 | ingestion | 문서를 파싱해 지식 그래프와 인덱스를 만드는 과정. CLI는 `run-ingestion` |
| 파이프라인 단계 | pipeline stage | 수집 파이프라인의 12개 처리 단계(`document_parsing` 등) |
| 체크포인트 | checkpoint | 단계별 결과 캐시. 재개(resume)에 사용 |
| 증분 인덱싱 | incremental indexing | 새 문서와 바뀐 문서만 다시 인덱싱하는 방식 |
| 문서 상태 레지스트리 | doc-status registry | 증분 인덱싱용 DynamoDB 테이블 |
| 전체 재구축 | full rebuild | `indexing.reset: true`로 저장소를 비우고 다시 인덱싱 |
| 인덱스 접미사 | index suffix | OpenSearch 인덱스와 Neptune 레이블 이름 뒤에 붙는 값. 수집 시 `processing.document_parsing.index_value`, 질의 시 `--suffix` / `RAGInput.suffix` |
| 테넌트 | tenant | 데이터를 분리해야 하는 사용자 또는 조직 단위 |
| 소스 범위 | source scope | 증분 삭제 판단에 쓰는 코퍼스 식별값(`source_scope`) |
| 코퍼스 | corpus | 수집 대상 문서 전체 |
| 청크 | chunk | 문서를 나눈 조각 |
| 텍스트 단위 | text unit | 인덱싱된 청크. 엔티티와 관계가 이를 출처로 참조 |
| 엔티티 | entity | 그래프의 노드(사람, 조직, 개념 등) |
| 관계 | relationship | 두 엔티티를 잇는 간선 |
| 주장 | claim (covariate) | 선택적으로 추출하는 사실 진술 |
| 커뮤니티 | community | Leiden 군집화로 묶인 엔티티 집합 |
| 커뮤니티 리포트 | community report | 커뮤니티마다 LLM이 작성한 요약 |
| 계보 | lineage | 산출물이 어느 텍스트 단위와 문서에서 왔는지에 대한 기록 |
| 추가 추출(gleaning) | gleaning | 놓친 엔티티와 관계를 찾는 추가 추출 패스 |
| 엔티티 해석 | entity resolution | 같은 대상을 가리키는 엔티티를 병합하는 단계 |
| 검색 | retrieval, search | 질의에 맞는 근거를 찾는 단계 |
| 검색 전략 | search strategy | `auto`, `local`, `global`, `drift`, `simple`, `mix`, `hybrid`, `naive` |
| 방법론 | methodology | GraphRAG(커뮤니티 요약)와 LightRAG(이중 레벨 키워드) |
| 이중 레벨 키워드 | dual-level keywords | LightRAG의 고수준(hl)·저수준(ll) 키워드 |
| 그래프 확장 | graph expansion | Neptune에서 이웃 엔티티를 따라가는 순회 |
| 어휘 검색 | lexical search | BM25 기반 키워드 검색 |
| 벡터 검색 | vector search | 임베딩 유사도 검색 |
| 하이브리드 검색 | hybrid search | 어휘 검색과 벡터 검색의 결합 |
| 융합 | fusion | 여러 검색 결과의 순위를 합치는 단계(RRF 등) |
| 재순위화 | reranking | Bedrock rerank 모델로 후보 순위를 다시 매기는 단계 |
| 토큰 예산 | token budget | 답변 프롬프트에 넣을 컨텍스트의 토큰 상한 |
| 임베딩 | embedding | 텍스트의 벡터 표현 |
| 기본 계층 / 빠른 계층 | default tier / fast tier | `default_model_id` / `fast_model_id`를 쓰는 모델 역할 묶음 |
| 추론 강도 | reasoning effort | `effort` 설정값(`low`~`max`) |
| 추론 프로파일 | inference profile | Bedrock 교차 리전 호출 경로 |
| 가드레일 | guardrail | Amazon Bedrock Guardrails 정책 |
| 환각 | hallucination | 원문에 근거 없는 내용을 모델이 만들어 내는 현상 |
| 평가기 | evaluator | `run-eval`의 지표 계산 모듈 |
| 판정 모델 | judge (LLM judge) | 답변을 채점하는 LLM |
| 포트 / 어댑터 | port / adapter | 헥사고날 아키텍처의 추상 인터페이스와 구현 |
| 레지스트리 | registry | 데코레이터로 구현체를 등록하는 목록 |
| 스택 출력값 | stack output | CloudFormation 스택의 Outputs |
| 배포 토폴로지 | deployment topology | 배포된 리소스의 배치와 연결 구조 |
| 비용 요인 | cost driver | 비용을 늘리는 사용량 요소 |
| 운영 런북 | operator runbook | 운영 작업 절차 문서 |
