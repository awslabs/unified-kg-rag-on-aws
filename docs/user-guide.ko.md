# Unified Knowledge Graph RAG on AWS — 사용자 가이드

> 🇬🇧 English version: [docs/user-guide.md](./user-guide.md)

사용자 가이드는 작업, 개념, 레퍼런스별 문서로 나누었으며, 전체 목록은
[문서 안내](./index.ko.md)에 있습니다. 이 문서는 각 절이 옮겨 간 곳을 알려 주는 목록으로
남겨 두어, 이전 버전의 `--help` 출력을 비롯한 기존 링크가 계속 동작합니다. 아래 각 제목은
원래 절의 제목 ID(heading id)를 그대로 유지하고 새 문서를 가리킵니다.

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

3절, 4절, 6절의 예시는 각 가이드로 옮겼습니다.

- 수집: [수집: 예시](./guides/ingest.ko.md#예시).
- <a id="예시-1"></a>질의: [질의: 예시](./guides/query.ko.md#예시).
- <a id="예시-2"></a>평가: [평가: 예시](./guides/evaluate.ko.md#예시)(결과 파일의 내용:
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

### 10. 운영과 문제 해결

[Troubleshooting](./troubleshooting.md)(영문: FAQ와 오류 메시지별 의미와 조치)과
[CLI exit codes and JSON output](./reference/exit-codes-and-json.md)(영문)으로
옮겼습니다. [IAM 권한](./getting-started/aws.ko.md#iam-권한)은 AWS에서 시작하기에, 대응
절차는 [운영 런북](./operations.md)(영문)에 있습니다.

### CLI 종료 코드와 JSON 출력

[CLI exit codes and JSON output](./reference/exit-codes-and-json.md)(영문)으로 옮겼습니다.

### 대규모, 다국어, 이질적인 코퍼스

[수집: 대규모, 이질적인 코퍼스](./guides/ingest.ko.md#대규모-이질적인-코퍼스)로 옮겼습니다.

### 비용 참고

[수집: 비용 조절 수단](./guides/ingest.ko.md#비용-조절-수단)으로 옮겼습니다.

### 11. Python 라이브러리로 사용하기

[Using the library from Python](./guides/python-library.md)(영문)으로 옮겼습니다.

### 12. 제한 사항

[개요: 제한 사항](./concepts/overview.ko.md#제한-사항)으로 옮겼습니다.
