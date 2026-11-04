# Unified Knowledge Graph RAG on AWS — 사용자 가이드

> 🇬🇧 English version: [docs/user-guide.md](./user-guide.md)

한국어 문서는 README, 문서 안내, 개요, 검색, 범위와 네임스페이스, 시작하기 두 문서, 설정·수집·다국어 코퍼스·질의·증분 인덱싱·도메인 적응·평가 가이드와 이 사용자 가이드입니다. 설계 문서, 운영 런북, 모델 카탈로그, CLI·설정 레퍼런스와 그 밖의 가이드는 영문으로만 제공합니다.

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

각 CLI의 `--help` 끝에는 실행 예시와 해당 가이드 링크가 있으며, 모든 플래그는 [CLI 레퍼런스](./reference/cli.md)(영문)에 있습니다.

## 목차

1. 사전 요구사항과 설치: [AWS에서 시작하기](./getting-started/aws.ko.md)([필수 모델](./getting-started/aws.ko.md#필수-모델) 포함), [로컬에서 시작하기](./getting-started/local.ko.md), [Local mode](./guides/local-mode.md)(영문)로 옮김
2. 설정: [설정](./guides/configure.ko.md)으로 옮김(모든 키: [설정 레퍼런스](./reference/configuration.md))
3. 수집: [수집](./guides/ingest.ko.md)과 [다국어 코퍼스](./guides/multilingual.ko.md)로 옮김
4. 질의: [질의](./guides/query.ko.md), [검색](./concepts/retrieval.ko.md), [필터 레퍼런스](./reference/filters.md)(영문)로 옮김
5. 증분 인덱싱: [증분 인덱싱](./guides/incremental.ko.md)으로 옮김
6. 평가: [평가](./guides/evaluate.ko.md), [Evaluation output 레퍼런스](./reference/evaluation-output.md)(영문), [Benchmarks](./guides/benchmarks.md)(영문)로 옮김
7. 시각화: [Visualization](./guides/visualize.md)(영문)으로 옮김
8. 프롬프트 튜닝: [도메인 적응: 프롬프트 튜닝](./guides/domain-adaptation.ko.md#프롬프트-튜닝-run-prompt-tuning)으로 옮김
9. 도메인 적응: [도메인 적응](./guides/domain-adaptation.ko.md), [Extraction schema](./guides/extraction-schema.md)(영문), [Ontology](./guides/ontology.md)(영문), [Seed graphs](./guides/seed-graphs.md)(영문)로 옮김
10. [운영과 문제 해결](#10-운영과-문제-해결)
11. [Python 라이브러리로 사용하기](#11-python-라이브러리로-사용하기)
12. [제한 사항](#12-제한-사항)

관련 문서: [CLI 레퍼런스](./reference/cli.md)(영문) · [설정 레퍼런스](./reference/configuration.md)(영문) · [모델 카탈로그](./reference/models.md)(영문) · [운영 런북](./operations.md)(영문) ·
[설계 문서](./design.md)(영문) · [`iac/README.md`](../iac/README.md)

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
  `top_k`, `suffix`([인덱스 접미사](./concepts/scopes-and-namespaces.ko.md#인덱스-접미사)), `filters`, `conversation_id`,
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
  `metadata.rerank`에 나타납니다([재순위화 상태](./guides/query.ko.md#재순위화-상태)).
- **기본 제공 백엔드.** 모델: Bedrock(기본값), Ollama, OpenAI 호환 서버. 재순위화:
  Bedrock이나 sentence-transformers 크로스 인코더([로컬 모드](./guides/local-mode.md), 영문).
  저장소: Neptune(또는 Gremlin Server)과 OpenSearch, 또는 내장 SQLite 저장소(호스트 한
  대, 인덱스당 문서 약 100,000개). 그 밖의 것은 `register_provider` 빌더나 사용자 정의
  어댑터가 필요합니다([설계 문서 §15](./design.md#15-extension-guide)).
- **서빙 계층이 없습니다.** HTTP API, 최종 사용자 인증, 요청 속도 제한은 없습니다.
- **필터는 접근 제어가 아닙니다**
  ([범위와 네임스페이스](./concepts/scopes-and-namespaces.ko.md#접근-제어-경계가-아님)).
  테넌트는 인덱스 접미사로 분리하세요.
- **가벼운 수집은 쓸 수 있는 전략을 제한합니다.** 커뮤니티 탐지를 끄면
  (`graph.community_detection.enabled: false`) `global`과 `drift`가 쓸 리포트가 없고,
  관계 벡터 인덱스를 끄면(`indexing.opensearch.build_relationship_vector_index: false`)
  `mix`와 `hybrid`가 고수준 키워드를, `local`이 관계 섹션을 쓰지 못하고, 문서 필터가 걸린 그래프
  확장은 엔티티에 닿는 관계가 아니라 엔티티 인덱스 확인만으로 엔티티를 남깁니다.
- **증분 인덱싱은 근사치입니다**([재구축 신호](./guides/incremental.ko.md#재구축-신호)). 증분 실행은 다시 군집화하지 않고 커뮤니티를
  덧붙이며, 엔티티가 현재 이름의 id를 받은 뒤에도 저장된 커뮤니티와 리포트에는 이전
  id가 남습니다. 둘 다 전체 재구축(`indexing.reset: true`)으로 갱신됩니다. DynamoDB에서는
  문서 하나의 청크 ID가 400KB 항목 하나에 들어가야 합니다([*문서 크기 한도*](./guides/incremental.ko.md#문서-크기-한도)).
- **동시 실행에는 레지스트리가 필요합니다.** 레지스트리의 쓰기 임대가 한
  네임스페이스의 두 번째 실행을 멈춥니다. 레지스트리가 없거나 `IncrementalIndexer`
  API를 직접 쓰면 잠그는 장치가 없어, 같은 접미사의 수집 두 개가 서로의 인덱스를 지울
  수 있습니다([동시 실행](./operations.md#concurrent-runs)).
- **태스크를 죽이는 문서를 가려내려면 여러 번 실행해야 합니다**([*격리된 문서*](./guides/incremental.ko.md#격리된-문서)). 함께
  죽은 문서 `n`개 가운데 원인 문서 하나에 대략 `log2(n) + 1`번이 걸리고, 그동안 미룬
  문서는 기다립니다.
- **Markdown과 HTML**에는 `unstructured` extra가 필요하며, 컨테이너
  이미지에는 기본으로 빠져 있습니다.
- **시드 그래프는 이름으로만 이어집니다**([*우선순위*](#seed-precedence),
  [§9 D](#d-시드-그래프가지고-있는-지식-그래프-가져오기)). 시드 노드와 이름이 같은 관계없는
  문서 엔티티는 한 엔티티가 됩니다. IRI는 저장하고 필터로 쓸 수 있지만 정체성이 아니며,
  `owl:sameAs`는 따라가지 않고 아무것도 추론하지 않습니다.
- **임베딩 모델(또는 차원)을 바꾸면** `indexing.reset: true`가 필요합니다. 증분 실행은
  다른 모델의 인덱스에 쓰지 않습니다([작업 흐름](./guides/incremental.ko.md#작업-흐름)).

---

*함께 보기: 개요와 빠른 시작은 [README.ko.md](../README.ko.md), 아키텍처와 내부 구조는
[설계 문서](./design.md)를 참고하세요.*

## 옮긴 절

이 가이드의 1절부터 9절까지와 10절 일부는 별도 문서로 옮겼습니다. 아래 제목은 이전 버전의 `--help`
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

[도메인 적응: 사용자 정의 프롬프트](./guides/domain-adaptation.ko.md#사용자-정의-프롬프트)로 옮겼습니다.

### 환경 변수로 덮어쓰기

[설정: 환경 변수로 덮어쓰기](./guides/configure.ko.md#환경-변수로-덮어쓰기)로 옮겼습니다.

### 관측성

[Observability](./guides/observability.md)(영문. 설정과 절차)와
[Telemetry reference](./reference/telemetry.md)(영문. 트레이스 구조, 사용량 집계)로 옮겼습니다.

### 3. 수집 (`run-ingestion`)

[수집](./guides/ingest.ko.md)으로 옮겼습니다.

### 파이프라인 단계 12개

[수집: 파이프라인 단계 12개](./guides/ingest.ko.md#파이프라인-단계-12개)로 옮겼습니다.

### 인덱스 접미사

[범위와 네임스페이스: 인덱스 접미사](./concepts/scopes-and-namespaces.ko.md#인덱스-접미사)로 옮겼습니다.

### 다국어 수집

[다국어 코퍼스](./guides/multilingual.ko.md)로 옮겼습니다.

### 4. 질의 (`run-rag`)

[질의](./guides/query.ko.md)와 [검색](./concepts/retrieval.ko.md)으로 옮겼습니다.

### 검색 전략 선택

[검색: 검색 전략 선택](./concepts/retrieval.ko.md#검색-전략-선택)으로 옮겼습니다.

### 답변 형식

[검색: 답변 형식](./concepts/retrieval.ko.md#답변-형식)으로 옮겼습니다.

### 컨텍스트 예산

[검색: 컨텍스트 예산](./concepts/retrieval.ko.md#컨텍스트-예산)으로 옮겼습니다.

### 재순위화 상태

[질의: 재순위화 상태](./guides/query.ko.md#재순위화-상태)로 옮겼습니다.

### 질의 제한 시간

[질의: 질의 제한 시간](./guides/query.ko.md#질의-제한-시간)으로 옮겼습니다.

### 필터

[질의: 필터](./guides/query.ko.md#필터)(사용 방법)와
[필터 레퍼런스](./reference/filters.md)(영문. 모든 필드, 값의 유형, 전략별 적용)로 옮겼습니다.

### 대화형 모드와 대화 메모리

[질의: 대화형 모드와 대화 메모리](./guides/query.ko.md#대화형-모드와-대화-메모리)로 옮겼습니다.

### 5. 증분 인덱싱

[증분 인덱싱](./guides/incremental.ko.md)으로 옮겼습니다.

### 켜기

[증분 인덱싱: 켜기](./guides/incremental.ko.md#켜기)로 옮겼습니다.

### 작업 흐름

[증분 인덱싱: 작업 흐름](./guides/incremental.ko.md#작업-흐름)으로 옮겼습니다(재구축은
[재구축](./guides/incremental.ko.md#재구축-indexingreset)과
[재구축 신호](./guides/incremental.ko.md#재구축-신호) 참고).

### 삭제 범위

[증분 인덱싱: 삭제 범위](./guides/incremental.ko.md#삭제-범위)로 옮겼습니다.

### 실패한 문서

[증분 인덱싱: 실패한 문서](./guides/incremental.ko.md#실패한-문서)로 옮겼습니다.

### 격리된 문서

[증분 인덱싱: 격리된 문서](./guides/incremental.ko.md#격리된-문서)로 옮겼습니다.

### 동시 실행

[증분 인덱싱: 동시 실행](./guides/incremental.ko.md#동시-실행)으로 옮겼습니다.

### 문서 크기 한도

[증분 인덱싱: 문서 크기 한도](./guides/incremental.ko.md#문서-크기-한도)로 옮겼습니다.

### 실행 간 병합

[증분 인덱싱: 실행 간 병합](./guides/incremental.ko.md#실행-간-병합)으로 옮겼습니다.

### 대규모, 다국어, 이질적인 코퍼스

[수집: 대규모, 이질적인 코퍼스](./guides/ingest.ko.md#대규모-이질적인-코퍼스)로 옮겼습니다.

### 비용 참고

[수집: 비용 조절 수단](./guides/ingest.ko.md#비용-조절-수단)으로 옮겼습니다.

### 6. 평가 (`run-eval`)

[평가](./guides/evaluate.ko.md)(평가기, 데이터 형식, 비교, 다시 만들기와 게이트),
[Evaluation output 레퍼런스](./reference/evaluation-output.md)(영문: 결과 파일, 요약, 실행
매니페스트, 비교 필드, 평가기별 채점 방식),
[Benchmarks](./guides/benchmarks.md)(영문)로 옮겼습니다.

### 평가기

[평가: 평가기](./guides/evaluate.ko.md#평가기)로 옮겼습니다.

### `ragas` 컨텍스트

[Evaluation output: `ragas` contexts](./reference/evaluation-output.md#ragas-contexts)(영문)로 옮겼습니다.

### `graph_aware`

[Evaluation output: `graph_aware`](./reference/evaluation-output.md#graph_aware)(영문)로 옮겼습니다.

### `retrieval`

[Evaluation output: `retrieval`](./reference/evaluation-output.md#retrieval)(영문)로 옮겼습니다.

### `answer_match`

[Evaluation output: `answer_match`](./reference/evaluation-output.md#answer_match)(영문)로 옮겼습니다.

### 평가 데이터 형식

[평가: 평가 데이터 형식](./guides/evaluate.ko.md#평가-데이터-형식)으로 옮겼습니다.

### 예시

[평가: 예시](./guides/evaluate.ko.md#예시)로 옮겼습니다(결과 파일의 내용:
[Evaluation output](./reference/evaluation-output.md), 영문).

### 전략 비교

[평가: 전략 비교](./guides/evaluate.ko.md#전략-비교)로 옮겼습니다(모든 필드:
[Evaluation output: Comparison](./reference/evaluation-output.md#comparison), 영문).

### 저장한 결과로 비교 다시 만들기, 다시 판정하기, 게이트

[평가: 저장한 결과로 비교 다시 만들기, 다시 판정하기, 게이트](./guides/evaluate.ko.md#저장한-결과로-비교-다시-만들기-다시-판정하기-게이트)로 옮겼습니다.

### 벤치마크 (MuSiQue)

[Benchmarks](./guides/benchmarks.md)(영문)로 옮겼습니다.

### 7. 시각화 (`run-visualization`)

[Visualization](./guides/visualize.md)(영문)으로 옮겼습니다.

### 8. 프롬프트 튜닝 (`run-prompt-tuning`)

[도메인 적응: 프롬프트 튜닝](./guides/domain-adaptation.ko.md#프롬프트-튜닝-run-prompt-tuning)으로 옮겼습니다.

### A. `entity_types`(비용이 가장 적고 효과가 가장 큼)

[도메인 적응: 엔티티 유형](./guides/domain-adaptation.ko.md#엔티티-유형)으로 옮겼습니다.

### B. `custom_prompts` 재정의

[도메인 적응: 사용자 정의 프롬프트](./guides/domain-adaptation.ko.md#사용자-정의-프롬프트)로
옮겼습니다(재정의 키가 없는 프롬프트 다섯 개:
[재정의할 수 없는 프롬프트](./guides/domain-adaptation.ko.md#재정의할-수-없는-프롬프트)).

### 9. 도메인 적응

[도메인 적응](./guides/domain-adaptation.ko.md)(엔티티 유형, 프롬프트, 프롬프트 튜닝),
[Extraction schema](./guides/extraction-schema.md)(영문),
[Ontology](./guides/ontology.md)(영문), [Seed graphs](./guides/seed-graphs.md)(영문)로
옮겼습니다.

### C. 추출 스키마

[Extraction schema](./guides/extraction-schema.md)(영문)로 옮겼습니다.

### OWL/RDFS 온톨로지에서 읽기

[Ontology](./guides/ontology.md)(영문: 설치, 설정, 미리 보기, 매핑 고정)와
[Ontology mapping 레퍼런스](./reference/ontology-mapping.md)(영문: 옵션, 읽는 것, 매핑 규칙,
예제)로 옮겼습니다.

- <a id="ontology-load-cost"></a>읽기 비용: [Ontology: Load cost](./guides/ontology.md#load-cost)(영문).
- <a id="ontology-what-a-change-re-runs"></a>바꾸면 다시 실행되는 것: [Ontology: What a change re-runs](./guides/ontology.md#what-a-change-re-runs)(영문).
- <a id="ontology-preview"></a>미리 보기: [Ontology: Preview](./guides/ontology.md#preview)(영문).

### D. 시드 그래프(가지고 있는 지식 그래프 가져오기)

[Seed graphs](./guides/seed-graphs.md)(영문)와
[Seed graph formats 레퍼런스](./reference/seed-graph-formats.md)(영문: JSONL, CSV, RDF)로
옮겼습니다.

- <a id="seed-precedence"></a>우선순위: [Seed graphs: Precedence](./guides/seed-graphs.md#precedence-documents-decide)(영문).
- <a id="seed-lifecycle"></a>수명 주기: [Seed graphs: Lifecycle](./guides/seed-graphs.md#lifecycle)(영문).
- <a id="seed-cost-and-limits"></a>비용과 한계: [Seed graphs: Cost and limits](./guides/seed-graphs.md#cost-and-limits)(영문).
