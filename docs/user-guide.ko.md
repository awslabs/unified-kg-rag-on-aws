# Unified Knowledge Graph RAG on AWS — 사용자 가이드

> 🇬🇧 English version: [docs/user-guide.md](./user-guide.md)

사용자 가이드는 작업, 개념, 레퍼런스별 문서로 나누었으며, 전체 목록은
[문서 안내](./index.ko.md)에 있습니다. 이 문서는 0.2.3 릴리스 사용자 가이드의 절 제목을 그대로
남겨, 그 절을 가리키는 링크가 계속 동작하게 합니다. 아래 각 절은 원래 절의 제목 ID(heading
id)를 유지하고, 그 내용을 다루는 문서를 가리킵니다.

## 목차

1. 사전 요구사항과 설치: [AWS에서 시작하기](./getting-started/aws.ko.md)([필수 모델](./getting-started/aws.ko.md#필수-모델) 포함), [로컬에서 시작하기](./getting-started/local.ko.md), [Local mode](./guides/local-mode.md)(영문)
2. 설정: [설정](./guides/configure.ko.md)(모든 키: [설정 레퍼런스](./reference/configuration.md), 영문), [Model selection notes](./reference/models.md#model-selection-notes)(영문), [Guardrail](./guides/guardrail.md)(영문), [Observability](./guides/observability.md)(영문), [Telemetry reference](./reference/telemetry.md)(영문)
3. 수집: [수집](./guides/ingest.ko.md), [다국어 코퍼스](./guides/multilingual.ko.md), [범위와 네임스페이스](./concepts/scopes-and-namespaces.ko.md)
4. 질의: [질의](./guides/query.ko.md), [검색](./concepts/retrieval.ko.md), [필터 레퍼런스](./reference/filters.md)(영문)
5. 증분 인덱싱: [증분 인덱싱](./guides/incremental.ko.md)
6. 평가: [평가](./guides/evaluate.ko.md), [Evaluation output 레퍼런스](./reference/evaluation-output.md)(영문), [Benchmarks](./guides/benchmarks.md)(영문)
7. 시각화: [Visualization](./guides/visualize.md)(영문)
8. 프롬프트 튜닝: [도메인 적응: 프롬프트 튜닝](./guides/domain-adaptation.ko.md#프롬프트-튜닝-run-prompt-tuning)
9. 도메인 적응: [도메인 적응](./guides/domain-adaptation.ko.md), [Extraction schema](./guides/extraction-schema.md)(영문), [Ontology](./guides/ontology.md)(영문), [Seed graphs](./guides/seed-graphs.md)(영문)
10. 운영과 문제 해결: [Troubleshooting](./troubleshooting.md)(영문), [CLI exit codes and JSON output](./reference/exit-codes-and-json.md)(영문), [운영 런북](./operations.md)(영문)
11. Python 라이브러리로 사용하기: [Using the library from Python](./guides/python-library.md)(영문)
12. 제한 사항: [개요: 제한 사항](./concepts/overview.ko.md#제한-사항)

## 옮긴 절

### 1. 사전 요구사항과 설치

<a id="1-사전-요구사항--설치"></a>[AWS에서 시작하기](./getting-started/aws.ko.md),
[로컬에서 시작하기](./getting-started/local.ko.md),
[Local mode](./guides/local-mode.md)(영문)로 옮겼습니다.

### 런타임

[AWS에서 시작하기: 런타임](./getting-started/aws.ko.md#런타임)으로 옮겼습니다.

### AWS 서비스

[AWS에서 시작하기: AWS 서비스](./getting-started/aws.ko.md#aws-서비스)로 옮겼습니다(활성화할
모델: [필수 모델](./getting-started/aws.ko.md#필수-모델)).

### 설치

[AWS에서 시작하기: 설치](./getting-started/aws.ko.md#설치)로 옮겼습니다.

### 인증

[AWS에서 시작하기: 인증](./getting-started/aws.ko.md#인증)으로 옮겼습니다.

### 2. 설정

[설정](./guides/configure.ko.md)(설정을 읽는 순서, 대부분의 실행에서 바꾸는 설정, 환경 변수로
덮어쓰기)과 [설정 레퍼런스](./reference/configuration.md)(영문. 모든 키)로 옮겼습니다.

### 2.1 `aws`: 서비스 엔드포인트와 자격 증명

<a id="21-aws--서비스-엔드포인트--자격증명"></a>[설정 레퍼런스: `aws`](./reference/configuration.md#aws)(영문)로 옮겼습니다.

### 2.2 `fixing`: 잘못된 형식의 모델 출력 자동 복구

<a id="22-fixing--잘못된-형식의-모델-출력-자동-복구"></a>출력 복구에는 설정 절이 없습니다. 항상 켜져 있고, 모델은 `models.roles.fixing`이
고릅니다([설정 레퍼런스: `models.roles`](./reference/configuration.md#modelsroles), 영문.
예전 키: [Removed keys](./reference/configuration.md#removed-keys), 영문).

### 2.3 `processing`: 동시성, 청킹, 번역, 추출

<a id="23-processing--동시성-청킹-번역-추출"></a>[설정 레퍼런스: `processing`](./reference/configuration.md#processing)(영문)으로 옮겼습니다.

### 2.4 `graph`: 분석, 커뮤니티 탐지, 시각화

<a id="24-graph--분석-커뮤니티-탐지-시각화"></a>[설정 레퍼런스: `graph`](./reference/configuration.md#graph)(영문)로 옮겼습니다.

### 2.5 `indexing`: OpenSearch와 Neptune 쓰기

<a id="25-indexing--opensearch--neptune-쓰기-측"></a>[설정 레퍼런스: `indexing`](./reference/configuration.md#indexing)(영문)으로 옮겼습니다.

### 2.6 `search`: 검색, 융합, 재순위화, 전략별 설정

<a id="26-search--검색-융합-리랭킹-전략별-항목"></a>[설정 레퍼런스: `search`](./reference/configuration.md#search)(영문)로 옮겼습니다.

### 2.7 `memory`, `cache`, `logging`

설정 레퍼런스(영문)의 [`memory`](./reference/configuration.md#memory),
[`cache`](./reference/configuration.md#cache),
[`logging`](./reference/configuration.md#logging)으로 옮겼습니다(로그에 남는 내용:
[설정](./guides/configure.ko.md#로그에-남는-내용)).

### 2.8 `evaluation`

[설정 레퍼런스: `evaluation`](./reference/configuration.md#evaluation)(영문)으로 옮겼습니다.

### 2.9 `custom_prompts`

[도메인 적응: 사용자 정의 프롬프트](./guides/domain-adaptation.ko.md#사용자-정의-프롬프트)로
옮겼습니다(모든 키: [설정 레퍼런스: `custom_prompts`](./reference/configuration.md#custom_prompts), 영문).

### 3. 수집 (`run-ingestion`)

<a id="3-인제스천-run-ingestion"></a>[수집](./guides/ingest.ko.md)으로 옮겼습니다(인덱스 네임스페이스:
[범위와 네임스페이스](./concepts/scopes-and-namespaces.ko.md)).

### CLI 플래그

<a id="cli-플래그-검증됨"></a>[CLI Reference: `run-ingestion`](./reference/cli.md#run-ingestion)(영문)으로 옮겼습니다.

### 파이프라인 단계 12개

<a id="12개-파이프라인-스테이지"></a>[수집: 파이프라인 단계 12개](./guides/ingest.ko.md#파이프라인-단계-12개)로 옮겼습니다.

### 예시

[수집: 예시](./guides/ingest.ko.md#예시)로 옮겼습니다.

### 다국어 수집

<a id="다국어-인제스천"></a>[다국어 코퍼스](./guides/multilingual.ko.md)로 옮겼습니다.

### 4. 질의 (`run-rag`)

[질의](./guides/query.ko.md)와 [검색](./concepts/retrieval.ko.md)으로 옮겼습니다(모든 필터
필드: [필터 레퍼런스](./reference/filters.md), 영문).

### CLI 플래그

<a id="cli-플래그-검증됨-1"></a>[CLI Reference: `run-rag`](./reference/cli.md#run-rag)(영문)로 옮겼습니다.

### 검색 전략 선택

<a id="검색-전략--어느-것을-언제-쓸까"></a>[검색: 검색 전략 선택](./concepts/retrieval.ko.md#검색-전략-선택)으로 옮겼습니다.

### 예시

[질의: 예시](./guides/query.ko.md#예시)로 옮겼습니다.

### 대화형 모드와 대화 메모리

<a id="인터랙티브-모드--대화-메모리"></a>[질의: 대화형 모드와 대화 메모리](./guides/query.ko.md#대화형-모드와-대화-메모리)로 옮겼습니다.

### 5. 증분 인덱싱

[증분 인덱싱](./guides/incremental.ko.md)으로 옮겼습니다.

### 켜기

<a id="활성화"></a>[증분 인덱싱: 켜기](./guides/incremental.ko.md#켜기)로 옮겼습니다.

### 작업 흐름

<a id="워크플로"></a>[증분 인덱싱: 작업 흐름](./guides/incremental.ko.md#작업-흐름)으로 옮겼습니다(재구축은
[재구축](./guides/incremental.ko.md#재구축-indexingreset)과
[재구축 신호](./guides/incremental.ko.md#재구축-신호) 참고).

### 실행 간 병합

<a id="실행-간-병합-cross-run-merge"></a>[증분 인덱싱: 실행 간 병합](./guides/incremental.ko.md#실행-간-병합)으로 옮겼습니다.

### 6. 평가 (`run-eval`)

[평가](./guides/evaluate.ko.md)(평가기, 데이터 형식, 비교, 다시 만들기와 게이트),
[Evaluation output 레퍼런스](./reference/evaluation-output.md)(영문: 결과 파일, 요약, 실행
매니페스트, 비교 필드, 평가기별 채점 방식),
[Benchmarks](./guides/benchmarks.md)(영문)로 옮겼습니다.

### CLI 플래그

<a id="cli-플래그-검증됨-2"></a>[CLI Reference: `run-eval`](./reference/cli.md#run-eval)(영문)로 옮겼습니다.

### 평가기

<a id="평가자"></a>[평가: 평가기](./guides/evaluate.ko.md#평가기)로 옮겼습니다.

### 평가 데이터 형식

<a id="평가-데이터-포맷"></a>[평가: 평가 데이터 형식](./guides/evaluate.ko.md#평가-데이터-형식)으로 옮겼습니다.

### 예시

[평가: 예시](./guides/evaluate.ko.md#예시)로 옮겼습니다(결과 파일의 내용:
[Evaluation output](./reference/evaluation-output.md), 영문).

### 7. 시각화 (`run-visualization`)

[Visualization](./guides/visualize.md)(영문)으로 옮겼습니다.

### CLI 플래그

<a id="cli-플래그-검증됨-3"></a>[CLI Reference: `run-visualization`](./reference/cli.md#run-visualization)(영문)으로 옮겼습니다.

### 8. 프롬프트 튜닝 (`run-prompt-tuning`)

[도메인 적응: 프롬프트 튜닝](./guides/domain-adaptation.ko.md#프롬프트-튜닝-run-prompt-tuning)으로 옮겼습니다.

### CLI 플래그

<a id="cli-플래그-검증됨-4"></a>[CLI Reference: `run-prompt-tuning`](./reference/cli.md#run-prompt-tuning)(영문)으로 옮겼습니다.

### 9. 도메인 적응

[도메인 적응](./guides/domain-adaptation.ko.md)(엔티티 유형, 프롬프트, 프롬프트 튜닝),
[Extraction schema](./guides/extraction-schema.md)(영문),
[Ontology](./guides/ontology.md)(영문), [Seed graphs](./guides/seed-graphs.md)(영문)로
옮겼습니다.

### A. `entity_types`(비용이 가장 적고 효과가 가장 큼)

<a id="a-entity_types-가장-저렴하고-영향력-큰-레버"></a>[도메인 적응: 엔티티 유형](./guides/domain-adaptation.ko.md#엔티티-유형)으로 옮겼습니다.

### B. `custom_prompts` 재정의

<a id="b-custom_prompts-오버라이드"></a>[도메인 적응: 사용자 정의 프롬프트](./guides/domain-adaptation.ko.md#사용자-정의-프롬프트)로
옮겼습니다(재정의 키가 없는 프롬프트 다섯 개:
[재정의할 수 없는 프롬프트](./guides/domain-adaptation.ko.md#재정의할-수-없는-프롬프트)).

### 10. 운영과 문제 해결

<a id="10-운영--트러블슈팅"></a>[Troubleshooting](./troubleshooting.md)(영문: FAQ와 오류 메시지별 의미와 조치),
[CLI exit codes and JSON output](./reference/exit-codes-and-json.md)(영문),
[운영 런북](./operations.md)(영문. 대응 절차)으로 옮겼습니다.

### IAM 권한

[AWS에서 시작하기: IAM 권한](./getting-started/aws.ko.md#iam-권한)으로 옮겼습니다.

### 자주 발생하는 오류

<a id="흔한-에러"></a>[Troubleshooting: Common errors](./troubleshooting.md#common-errors)(영문)로 옮겼습니다.

### 대규모, 다국어, 이질적인 코퍼스

<a id="대규모--다국어--이종heterogeneous-코퍼스"></a>[수집: 대규모, 이질적인 코퍼스](./guides/ingest.ko.md#대규모-이질적인-코퍼스)로 옮겼습니다.

### 비용 참고

[수집: 비용 조절 수단](./guides/ingest.ko.md#비용-조절-수단)으로 옮겼습니다.
