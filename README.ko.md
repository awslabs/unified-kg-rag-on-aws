# Unified Knowledge Graph RAG on AWS

🇬🇧 **[English README](./README.md)** · 🤝 **[기여 가이드](./CONTRIBUTING.md)**

![Knowledge Graph](./assets/interactive_graph.jpg)

대규모 다국어 문서 코퍼스를 지식 그래프로 변환하고, 멀티홉 그래프 순회로 질의에 답하는 **AWS 네이티브 Knowledge Graph RAG 프레임워크**입니다.

두 검색 방법론 — Microsoft GraphRAG(*"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"*)와 LightRAG(*"Simple and Fast Retrieval-Augmented Generation"*) — 를 단일 AWS 네이티브 스택(Bedrock, Neptune, OpenSearch, S3, DynamoDB) 위에 재구현했습니다. 두 방법론은 질의마다 선택 가능하며 동일한 인제스천·인덱싱·캐싱·다국어·하이브리드 검색 인프라를 공유합니다.

---

## ✨ 핵심 특징

### 🔍 하나의 인프라, 두 가지 선택형 방법론

질의마다 `search_strategy`로 선택합니다. 두 방법론은 **동일한 인제스천·인덱싱·캐싱·다국어·하이브리드 스코어링 인프라**를 공유하며, 검색 알고리즘 레이어만 다릅니다.

- **GraphRAG (커뮤니티 요약 방식)**: `simple`(직접), `local`(엔티티 중심), `global`(커뮤니티 기반), `drift`(점진 탐색), `auto`(LLM 라우터)
- **LightRAG (이중 레벨 키워드 방식)**: `mix` / `hybrid` / `naive` — 고수준·저수준 키워드를 추출해 엔티티 인덱스 + 관계 벡터 인덱스 + 그래프 확장으로 검색

### 🚀 트리플 하이브리드 검색

- **시맨틱 검색**: Bedrock 임베딩 모델 기반 고품질 벡터 검색
- **렉시컬 검색**: BM25 알고리즘 기반 정밀 키워드 매칭
- **그래프 검색**: Neptune 지식 그래프 순회를 통한 연결성 분석
- **결과 최적화**: RRF 융합 + Bedrock 리랭킹 모델

### ♻️ 증분 인덱싱

- **콘텐츠 해시 델타 감지**: DynamoDB 문서-상태 레지스트리가 신규/변경 문서만 재인덱싱하고 라이브 그래프에 멱등(idempotent) 병합
- **삭제 계보(lineage)**: 문서 삭제 시 해당 문서만 *독점적으로* 소유한 아티팩트만 제거 (공유 엔티티는 보존)

### 🧠 고급 지식 그래프 처리

- 퍼지 매칭 기반 엔티티 해석(중복 통합), Leiden 알고리즘 커뮤니티 탐지
- gleaning(반복 정제), 멀티홉 추론, 출처 투명성
- claim(사실 주장) 추출(옵트인, 기본 off): 활성화 시 전용 인덱스에 임베딩되어
  `local`/`simple` 검색 컨텍스트에 covariate로 주입

### 🎯 종합 평가 프레임워크

- **LangChain 평가자**: 정확성/부분정확성
- **RAGAS 지표**: 답변 충실도·관련성·컨텍스트 정확도
- **그래프 인식 평가**: 정답 기대치(`expected_entities`/`expected_relationships`) 대비 엔티티·관계 커버리지(= recall) (결정적·LLM 불필요, 단어 경계 매칭; precision/F1은 자유 텍스트 답변에서 엔티티 열거가 불가해 의도적으로 미산출)

### 🌍 다국어 지원

인덱싱·검색 시 번역, 언어별 분석기(analyzer), 다국어 키워드 추출 — **두 방법론 모두에 적용**됩니다.

### 📊 시각화 & 분석

- Node2Vec + UMAP 인터랙티브 그래프, 중심성(centrality) 지표, 그래프 통계
- 독립 CLI: 재인제스천 없이 내보낸 그래프 데이터로 시각화 (`run-visualization`)

### 🔧 사용자 지원

- 프롬프트별 커스텀 오버라이드, **자동 프롬프트 튜닝**(코퍼스 도메인 프로파일링 → 도메인 적응 프롬프트, `run-prompt-tuning`)
- YAML 설정 파일, 구조화 로깅(structlog)

### 🧱 헥사고날 아키텍처 (포트 & 어댑터)

스토리지/검색 백엔드를 교체 가능하게 하고, 검색 전략·평가자·렌더러를 레지스트리로 확장합니다 — 디스패치 코드를 수정하지 않고 확장 가능. 자세한 내용은 [`CLAUDE.md`](./CLAUDE.md)와 [기술 문서](./docs/design.md) 참고.

---

## 🏛️ 아키텍처 개요

### 인제스천 파이프라인 (12단계)

문서 파싱 → 로딩 → 청킹 → (번역) → 그래프 추출 → (gleaning) → 그래프 해석 → (claim 추출/해석) → 그래프 분석 → 커뮤니티 탐지 → 인덱싱

- **핵심 기능**: 증분 인덱싱(콘텐츠 해시 델타+병합), 재개 가능 파이프라인(스테이지 체크포인트), S3 캐시 동기화, 병렬 처리
- **인텔리전트 청킹**: simple/intelligent(LLM 시맨틱) 전략

### 검색 파이프라인

전략 해석(AUTO 라우팅) → 질의 처리(번역·엔티티/키워드 추출) → 대화 메모리 → 검색(전략별) → 컨텍스트 빌드 → 답변 생성

- **융합/재랭킹**: RRF, 다양성 필터링, Bedrock 리랭킹, 토큰 예산 관리
- **검색 전략**은 추상 역할(GRAPH/DOCUMENT)로 백엔드를 주입받아 백엔드 무관하게 동작

자세한 컴포넌트·데이터 흐름·알고리즘은 **[기술 문서](./docs/design.md)**를 참고하세요.

---

## 🚀 설치

### 사전 요구사항

- **Python 3.10 – 3.12** (uv 권장)
- 적절한 권한으로 구성된 **AWS CLI**
- 배포·접근 가능한 **AWS 서비스**: Amazon Bedrock(모델 액세스 활성화), Neptune 클러스터, OpenSearch 도메인, S3 버킷, (증분 인덱싱 시) DynamoDB

### 빠른 시작

```bash
# 저장소 클론
git clone <repository-url>
cd unified-kg-rag-on-aws

# 설치 (uv 권장)
uv sync --extra dev        # 또는: pip install -e .

# 설정 복사 및 편집
cp config-template.yaml config.yaml
# config.yaml에 AWS 서비스 엔드포인트 입력

# (OpenSearch에 username/password 인증을 쓰는 경우)
cp .env-template .env
# .env에 OpenSearch 자격증명 입력 (IAM 인증 use_iam: true면 불필요)
```

---

## 📖 사용법

모든 동작은 `config.yaml`(스키마: `config-template.yaml`)로 제어합니다. 5개 CLI(pyproject 스크립트)가 전체 워크플로를 커버합니다:

```bash
# 1) 코퍼스 인덱싱 (12단계 풀 파이프라인; DynamoDB 활성화 시 증분)
run-ingestion --source-directory ./source --config-path config.yaml

# 2) 질의 — GraphRAG(커뮤니티 요약) 또는 LightRAG(이중 레벨 키워드)
run-rag --query "문서의 주요 주제는?" --search-strategy global --config-path config.yaml
run-rag --query "Alice와 Acme의 관계는?" --search-strategy mix --config-path config.yaml
run-rag --interactive --use-memory --conversation-id my-session --config-path config.yaml

# 3) 평가 (langchain + ragas + graph-aware)
run-eval --eval-data-path eval_data.json --config-path config.yaml

# 4) 시각화 (재인제스천 불필요, 내보낸 그래프 데이터에서 렌더)
run-visualization --data-path visualization_data.json --output-dir ./viz --config-path config.yaml

# 5) 도메인 코퍼스에 프롬프트 자동 튜닝
run-prompt-tuning --source-dir ./source --output tuned_prompts.yaml --config-path config.yaml
```

**전략 선택** — GraphRAG: `simple`(직접 벡터/렉시컬), `local`(엔티티 중심), `global`(커뮤니티 요약, map-reduce), `drift`(점진 탐색), `auto`(LLM 라우터). LightRAG: `mix` / `hybrid` / `naive`(이중 레벨 키워드).

📘 **전체 설정 레퍼런스, 모든 CLI 플래그, Python API, 증분 추가/수정/삭제, 도메인 적응, 트러블슈팅은 [사용자 가이드](./docs/user-guide.ko.md)(영문: [User Guide](./docs/user-guide.md))를 참고하세요.** 아키텍처·알고리즘·구현 내부는 [설계 문서](./docs/design.md)(영문: [Design Doc](./docs/design.md))를 참고하세요.

---

## 🧪 테스트 & 품질

```bash
uv run pytest -m "not aws"                       # AWS 불필요 테스트 (단위/통합/프로퍼티)
uv run pytest -m "not aws" --cov=unified_kg_rag    # 커버리지 포함
uv run ruff check unified_kg_rag tests
uv run mypy unified_kg_rag
```

- `aws` 마커는 실제 AWS 서비스가 필요한 테스트를 분리하며 CI에서 제외됩니다.
- DynamoDB/S3는 `moto`, Neptune/OpenSearch는 포트 기반 in-memory fake로 테스트합니다.
- CI(`.github/workflows/`): ruff/black/isort/mypy + pytest+coverage 게이트, ASH 보안 스캔.

---

## 🔒 보안 & 면책

본 프로젝트는 **교육·예시 목적의 참조 프레임워크**입니다. 어떠한 보증도 없이
"있는 그대로(AS IS)" 제공됩니다([LICENSE](./LICENSE) 참고). **별도의 보안
테스트·위협 모델링·하드닝 없이 프로덕션 환경에 배포해서는 안 됩니다.**

- 본인의 AWS 계정에서 본인의 리소스에 대해 실행하세요. 배포 환경의 IAM 정책,
  네트워크 구성, 데이터 분류, 최종 사용자 인증은 사용자 책임입니다.
- 선택적 CDK 스택(`iac/`)은 보안 기본값(프라이빗 VPC 격리, KMS 저장 시 암호화,
  TLS 강제, 최소 권한 IAM, PII/프롬프트 공격 필터링용 선택적 Bedrock
  Guardrail)을 제공하지만, 배포 책임과 환경 검토는 사용자에게 있습니다.
- 프로덕션 사용 전 Bedrock Guardrail(`aws.bedrock.guardrail`)을 활성화하고,
  용도에 맞는 레이트 리밋·모니터링을 적용하세요.
- 보안 이슈 신고는 [SECURITY.md](./SECURITY.md)를 참고하세요(공개 이슈로
  올리지 마세요).

## 🤝 기여

확장 방법(새 검색 전략/스토리지 백엔드/평가자/렌더러 추가)은 [`CONTRIBUTING.md`](./CONTRIBUTING.md)와 [`CLAUDE.md`](./CLAUDE.md)를 참고하세요. 대부분의 확장은 레지스트리 등록만으로 가능하며 디스패치 코드 수정이 필요 없습니다.

## 📄 라이선스

Apache-2.0. [`LICENSE`](./LICENSE) 참고.

## 📚 참고문헌

- Microsoft GraphRAG: [*From Local to Global: A Graph RAG Approach to Query-Focused Summarization*](https://arxiv.org/abs/2404.16130) · [라이브러리](https://github.com/microsoft/graphrag)
- LightRAG: [*Simple and Fast Retrieval-Augmented Generation*](https://arxiv.org/abs/2410.05779) · [라이브러리](https://github.com/HKUDS/LightRAG)
