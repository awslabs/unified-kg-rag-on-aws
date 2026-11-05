# 문서 안내

> 🇬🇧 English version: [docs/index.md](./index.md)

`unified-kg-rag-on-aws` 문서를 하려는 일에 따라 정리했습니다. 처음이라면
[개요](./concepts/overview.ko.md)부터 읽으세요. 한국어 문서는 README, 이 페이지, 개요,
검색, 범위와 네임스페이스, 시작하기 두 문서, 설정·수집·다국어 코퍼스·질의·증분 인덱싱·도메인 적응·평가
가이드와 옮긴 절 목록인 사용자 가이드이며, "(영문)"으로 표시한 문서는 영문으로만 제공합니다.

## 여기서 시작하기

| 독자 | 목표 | 읽는 순서 | 완료 기준 |
|---|---|---|---|
| **처음 써 보는 평가자** | AWS 없이 동작 확인 | [로컬에서 시작하기](./getting-started/local.ko.md) → [개요](./concepts/overview.ko.md) → [검색 전략 선택](./concepts/retrieval.ko.md#검색-전략-선택) → [평가기](./guides/evaluate.ko.md#평가기) | 예제 코퍼스에서 두 방법론의 답변을 받고, 판정 모델 없이 `run-eval` 실행 |
| **AWS에서 구축하는 개발자** | 실제 코퍼스를 수집하고 질의 | [AWS에서 시작하기](./getting-started/aws.ko.md)(기존 저장소, [필수 모델](./getting-started/aws.ko.md#필수-모델)) 또는 [`iac/README.md`](../iac/README.md#quick-start)(CDK, 영문) → [설정](./guides/configure.ko.md) → [수집](./guides/ingest.ko.md) → [질의](./guides/query.ko.md) → [증분 인덱싱](./guides/incremental.ko.md) → [도메인 적응](./guides/domain-adaptation.ko.md) | 증분 수집을 켜고, 필터로 질의하고, 스키마나 프롬프트를 도메인에 맞춤 |
| **운영자** | 실행, 모니터링, 복구, 업그레이드 | [`iac/README.md`](../iac/README.md)(영문) → [운영 런북](./operations.md)(영문: [실행 전](./operations.md#before-a-run), [실행이 실패하면](./operations.md#when-a-run-fails), [경보](./operations.md#alarms), [문제 해결](./troubleshooting.md)) → [CHANGELOG](../CHANGELOG.md) | 모든 경보와 오류에서 대응 절차를 찾을 수 있음 |
| **평가자·연구자** | 품질과 비용 측정, 기본값 재현, 논문과 비교 | [평가](./guides/evaluate.ko.md) → [전략 비교](./guides/evaluate.ko.md#전략-비교) → [Evaluation output](./reference/evaluation-output.md)(영문) → [Benchmarks](./guides/benchmarks.md)(영문) → [기본값과 근거](./concepts/defaults-and-evidence.md)(영문) → [원 논문을 넘어서](./concepts/overview.ko.md#원-논문을-넘어서) | MuSiQue나 자체 데이터로 신뢰 구간과 게이트가 있는 짝지은 비교 실행 |
| **확장 개발자** | 백엔드, 파서, 렌더러, 스키마, 시드 소스를 플러그인으로 추가하거나 전략, 평가기를 저장소 안에서 추가 | [개요](./concepts/overview.ko.md) → [설계 문서 §2](./design.md#2-hexagonal-architecture-ports--adapters)(영문) → [§15 확장 가이드](./design.md#15-extension-guide) → [Extraction schema](./guides/extraction-schema.md) / [Seed graphs](./guides/seed-graphs.md)(영문) | 등록한 플러그인이나 저장소 안에 추가한 전략·평가기가 가짜 포트(fake port) 적합성 테스트를 통과 |
| **기여자** | 품질 게이트를 깨지 않고 코드와 문서 수정 | [CONTRIBUTING.md](../CONTRIBUTING.md)(영문) → [설계 문서 §2](./design.md#2-hexagonal-architecture-ports--adapters)와 [§13](./design.md#13-testing-strategy) → [문서 규칙](../CONTRIBUTING.md#documentation) | 문서 검사를 포함한 CI를 통과하는 풀 리퀘스트 |

## 문서 지도

| 유형 | 문서 | 내용 |
|---|---|---|
| 튜토리얼 | [README 빠른 시작](../README.ko.md#빠른-시작) | 실행 방법 선택, 설치, 첫 수집, 질의, 평가 |
| 튜토리얼 | [로컬에서 시작하기](./getting-started/local.ko.md), [AWS에서 시작하기](./getting-started/aws.ko.md) | 단계별 첫 실행. AWS 없이 예제 코퍼스로 두 방법론 질의와 판정 모델 없는 평가, 기존 Neptune·OpenSearch를 쓸 때의 사전 요구사항, 필수 모델, 설치, 인증, IAM |
| 설명 | [개요](./concepts/overview.ko.md) | 아키텍처, 파이프라인, 두 방법론, 원 논문 대비 확장한 부분, 제한 사항 |
| 설명 | [기본값과 근거](./concepts/defaults-and-evidence.md)(영문) | 측정으로 정한 기본값, 측정 방법, 날짜를 밝힌 가격, 재현 방법 |
| 설명 | [설계 문서](./design.md)(영문) | 헥사고날 아키텍처, 알고리즘, 데이터 모델, 확장 가이드, 참고 자료 |
| 설명 | [검색](./concepts/retrieval.ko.md), [범위와 네임스페이스](./concepts/scopes-and-namespaces.ko.md) | 검색 전략과 언제 어떤 전략을 쓸지, 하이브리드 점수와 재순위화, 답변 형식, 컨텍스트 예산. 인덱스 접미사, 소스 범위, 네임스페이스, 접미사와 필터가 접근 제어가 아닌 이유 |
| 사용 방법 | [`iac/README.md`](../iac/README.md)(영문) | CDK 스택, 배포 토폴로지, 비용 요인, `-c key=value` 옵션, Guardrail 배포, 배포 후 명령 |
| 사용 방법 | [Local mode](./guides/local-mode.md)(영문), [설정](./guides/configure.ko.md) | 로컬 공급자, AWS 모드와 다른 점, 컨테이너로 띄운 저장소. 설정을 읽는 순서, 대부분의 실행에서 바꾸는 설정, 환경 변수, 로그에 남는 내용 |
| 사용 방법 | [수집](./guides/ingest.ko.md), [다국어 코퍼스](./guides/multilingual.ko.md), [질의](./guides/query.ko.md) | 12개 단계, 재개와 강제 재구축, S3 동기화, 응답 캐시, 대규모 코퍼스와 비용 조절 수단. 번역, 분석기, CJK 텍스트의 청크 크기. 질의 실행, 필터, 재순위화 상태, 질의 제한 시간, 대화형 모드와 메모리 |
| 사용 방법 | [증분 인덱싱](./guides/incremental.ko.md) | 레지스트리 켜기, 코퍼스나 설정을 바꿀 때 일어나는 일, 재구축과 두 가지 재구축 신호, 삭제 범위와 대량 삭제, 코퍼스 옮기기, 실패한 문서와 격리된 문서, 동시 실행 |
| 사용 방법 | [도메인 적응](./guides/domain-adaptation.ko.md) | 엔티티 유형, 프롬프트 튜닝, 사용자 정의 프롬프트 재정의와 그 검사, 재정의할 수 없는 프롬프트 |
| 사용 방법 | [Extraction schema](./guides/extraction-schema.md), [Ontology](./guides/ontology.md), [Seed graphs](./guides/seed-graphs.md)(영문) | 스키마 모드, 유형과 패턴 선언, 저장하는 스키마 필드와 그 필터, 스키마를 바꾸면 다시 실행되는 것. OWL/RDFS 온톨로지에서 스키마 읽기. 이미 가진 지식 그래프 합치기, 시드가 스키마와 문서를 만날 때의 규칙, 수명 주기와 비용 |
| 사용 방법 | [평가](./guides/evaluate.ko.md), [Benchmarks](./guides/benchmarks.md)(영문), [Visualization](./guides/visualize.md)(영문) | 평가기와 필요한 필드, 데이터 형식, 전략의 짝지은 비교, 비용과 가격, 다시 만들기·다시 판정하기·CI 게이트. MuSiQue를 코퍼스와 데이터셋으로 변환. 내보낸 그래프 렌더링 |
| 사용 방법 | [Guardrail](./guides/guardrail.md), [Observability](./guides/observability.md)(영문) | 질의와 답변 검사, 컨텍스트 근거 검사. Jaeger, Langfuse, Phoenix, CloudWatch로 추적, 전송 제한과 개인정보 |
| 사용 방법 | [Using the library from Python](./guides/python-library.md)(영문) | Python에서 수집과 질의, 구분해 처리할 결과와 오류, 비동기 코드, 닫기, 로깅, 백엔드 주입 |
| 레퍼런스 | [CLI 레퍼런스](./reference/cli.md)(영문) | 모든 CLI의 모든 플래그(파서에서 생성) |
| 레퍼런스 | [CLI exit codes and JSON output](./reference/exit-codes-and-json.md)(영문) | 종료 코드의 의미, 신호를 받았을 때의 동작, `--output-format json` 문서 |
| 레퍼런스 | [설정 레퍼런스](./reference/configuration.md)(영문) | 모든 설정 키의 타입, 기본값, 설명(코드에서 생성) |
| 레퍼런스 | [모델 카탈로그](./reference/models.md)(영문) | 지원하는 Bedrock 모델, 공급자별 요청 차이, 모델 선택 참고 사항(계층, 역할, 판정 모델, 출력 상한) |
| 레퍼런스 | [필터 레퍼런스](./reference/filters.md)(영문) | 저장소별로 필터할 수 있는 모든 필드, 값의 유형과 날짜, 전략별 적용, OpenSearch의 엄격한 적용, 필터를 건 그래프 확장 |
| 레퍼런스 | [Ontology mapping reference](./reference/ontology-mapping.md), [Seed graph formats](./reference/seed-graph-formats.md)(영문) | `owl` 소스의 옵션, 읽는 것, 매핑 규칙과 예제. JSONL, CSV, RDF 시드 형식과 옵션 |
| 레퍼런스 | [Evaluation output reference](./reference/evaluation-output.md)(영문) | 결과 파일, 실행 요약과 실행 매니페스트, 모든 비교 필드, 다시 만들기와 게이트 규칙, 평가기별 채점 방식 |
| 레퍼런스 | [Telemetry reference](./reference/telemetry.md)(영문) | 모든 스팬과 속성, 벤더 속성 이름, 알려진 제한, 모델 사용량 집계 |
| 런북 | [운영 런북](./operations.md)(영문) | 용량, 재수집, 증분 실행, 캐시, 백업과 복원, 업그레이드, 실패한 문서와 격리된 문서, 경보, 질의 마감 시간, 테넌트 격리 |
| 문제 해결 | [Troubleshooting](./troubleshooting.md)(영문) | 자주 묻는 질문(Markdown 파일, 리전, 재순위화, Nori, 추론 프로필, 쓰기 임대, 확인 질문, 제한 시간)과 오류 메시지별 의미와 조치 |
| 이전 문서 | [사용자 가이드](./user-guide.ko.md) | 0.2.3 릴리스 사용자 가이드의 각 절이 옮겨 간 곳 |
| 프로젝트 | [CONTRIBUTING.md](../CONTRIBUTING.md), [CHANGELOG.md](../CHANGELOG.md), [SECURITY.md](../SECURITY.md)(영문) | 개발 절차, 릴리스 노트, 보안 이슈 신고 |

## 하려는 일별 안내

| 하려는 일 | 문서 |
|---|---|
| AWS 계정 없이 써 보기 | [로컬에서 시작하기](./getting-started/local.ko.md) |
| AWS에 전체 스택 배포 | [`iac/README.md`](../iac/README.md#quick-start)(영문) |
| 필요한 Bedrock 모델 활성화 | [필수 모델](./getting-started/aws.ko.md#필수-모델) |
| 모델이나 설정 변경 | [설정](./guides/configure.ko.md), [모델 선택 참고 사항](./reference/models.md#model-selection-notes)(영문), [설정 레퍼런스](./reference/configuration.md)(영문) |
| 질의와 답변 검사 | [Guardrail](./guides/guardrail.md)(영문) |
| 질의와 수집 실행 추적 | [Observability](./guides/observability.md)(영문) |
| 코퍼스 수집, 재개, 재구축 | [수집](./guides/ingest.ko.md) |
| 한국어나 여러 언어가 섞인 코퍼스 인덱싱 | [다국어 코퍼스](./guides/multilingual.ko.md) |
| 검색 전략 고르기 | [검색 전략 선택](./concepts/retrieval.ko.md#검색-전략-선택) |
| 문서나 속성으로 결과 필터링 | [질의: 필터](./guides/query.ko.md#필터), [필터 레퍼런스](./reference/filters.md)(영문) |
| 문서가 바뀌거나 지워질 때 인덱스 갱신 | [증분 인덱싱](./guides/incremental.ko.md) |
| 추출을 도메인에 맞추기 | [도메인 적응](./guides/domain-adaptation.ko.md), [프롬프트 튜닝](./guides/domain-adaptation.ko.md#프롬프트-튜닝-run-prompt-tuning) |
| 온톨로지나 기존 지식 그래프 가져오기 | [Ontology](./guides/ontology.md)(영문), [Seed graphs](./guides/seed-graphs.md)(영문) |
| 답변과 검색 평가 | [평가](./guides/evaluate.ko.md), [Benchmarks](./guides/benchmarks.md)(영문) |
| 전략 비교, CI에서 변경 검증 | [전략 비교](./guides/evaluate.ko.md#전략-비교), [비교 다시 만들기, 다시 판정하기, 게이트](./guides/evaluate.ko.md#저장한-결과로-비교-다시-만들기-다시-판정하기-게이트) |
| 기본값을 그렇게 정한 이유 확인 | [기본값과 근거](./concepts/defaults-and-evidence.md)(영문) |
| Python에서 사용 | [Using the library from Python](./guides/python-library.md)(영문) |
| 스크립트에서 CLI 실행, JSON 결과 읽기 | [CLI exit codes and JSON output](./reference/exit-codes-and-json.md)(영문) |
| 오류나 실패한 실행 해결 | [Troubleshooting](./troubleshooting.md)(영문), [실행이 실패하면](./operations.md#when-a-run-fails)(영문) |
| 백엔드, 전략, 파서 추가 | [설계 문서 §15](./design.md#15-extension-guide)(영문) |

## 용어집

한국어 문서는 아래 용어를 씁니다. 괄호 안은 영문 문서와 코드에서 쓰는 용어입니다.

| 용어 | 뜻 |
|---|---|
| 수집(ingestion) | 문서를 그래프와 인덱스로 바꾸는 12단계 파이프라인(`run-ingestion`) |
| 텍스트 단위(text unit) | 문서를 나눈 청크(chunk). 추출, 인용, 삭제가 모두 텍스트 단위로 이뤄짐 |
| 엔티티(entity), 관계(relationship) | 추출한 지식 그래프의 노드와 에지 |
| 주장(claim) | 텍스트 단위마다 선택적으로 추출하는 사실 진술(MS GraphRAG의 covariate) |
| 글리닝(gleaning) | 첫 추출에서 빠진 것을 모델에 다시 묻는 추가 추출 라운드 |
| 커뮤니티(community), 커뮤니티 리포트(community report) | 엔티티를 Leiden으로 묶은 군집과, `global`과 `drift`가 읽는 LLM 작성 요약 |
| 방법론(methodology), 전략(strategy) | GraphRAG 또는 LightRAG, 그리고 질의마다 고르는 실제 검색 알고리즘(`local`, `mix` 등) |
| 인덱스 접미사(index suffix) | 모든 인덱스와 그래프 레이블 이름의 `<suffix>` 부분(`processing.document_parsing.index_value`. 지정하면 뒤에 `-<indexing.additional_suffix>`가 붙음). 테넌트나 코퍼스 버전마다 하나 |
| 문서 상태 레지스트리(doc-status registry) | 문서마다 해시와 텍스트 단위를 기록해 증분 수집을 가능하게 하는 DynamoDB(또는 SQLite) 테이블 |
| 네임스페이스(namespace) | 인덱스 접미사에 `indexing.additional_suffix`와 기본값이 아닌 인덱스·레이블 접두사의 태그를 더한 것, 그리고 그 이름이 가리키는 인덱스 묶음. 한 번에 한 실행만 쓸 수 있음(쓰기 임대(lease)). 메시지에서는 index namespace라고 부름 |
| 소스 범위(source scope) | 실행이 비교하고 삭제하는 코퍼스 출처(기본값은 소스 디렉터리). 문서 ID와 텍스트 단위 ID가 여기서 나옴 |
| 격리(quarantine) | 처리하다 실행이 거듭 죽은 문서를 따로 둔 상태. 파일이 바뀌거나 한도를 올릴 때까지 건너뜀 |
| 시드 그래프(seed graph) | 이미 가진 지식 그래프. 가상 문서로 추출 그래프에 병합 |
| 추출 스키마(extraction schema) | 추출이 따르는 엔티티 유형, 관계 유형, 패턴. 직접 선언하거나 온톨로지에서 읽음 |
