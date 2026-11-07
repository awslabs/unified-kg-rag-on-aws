# 로컬에서 시작하기

> 🇬🇧 English version: [local.md](./local.md)

로컬 모드는 AWS 계정 없이 호스트 한 대에서 수집, 질의, 평가를 실행하는 개발·평가용
모드입니다(기본값은 여전히 AWS). 기본 설정에서는 그래프 저장소, 벡터·어휘 저장소,
레지스트리가 모두 SQLite 파일이므로 실행할 서비스는
`http://localhost:11434`의 [Ollama](https://ollama.com) 하나뿐입니다. 이 문서에서는 파일 두
개로 된 예제 코퍼스를 인덱싱하고, GraphRAG 전략과 LightRAG 전략으로 질의한 뒤, LLM 판정 모델 없이 답변을
채점합니다.

로컬 모드는 개발과 평가 전용입니다. 내장 저장소는 호스트 한 대와 인덱스당 문서 약
100,000개를 기준으로 하고, 컨테이너에는 인증도 TLS도 없으며, 작은 로컬 모델은 그래프를
더 약하게 추출합니다. 공급자, AWS 모드와 다른 점, 컨테이너로 띄운 저장소는
[Local mode](../guides/local-mode.md)(영문) 가이드에 있습니다.

## 설치

Python 3.11–3.12, [uv](https://docs.astral.sh/uv/), Ollama가 필요합니다. `local` extra를
설치하고 기본 모델 두 개를 내려받으세요.

```bash
uv sync --extra local
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
```

- **CPU 전용 PyTorch.** Linux에서 `uv sync --extra local`은 잠금 파일이 고정한 CUDA
  빌드(수 GB)를 설치합니다. CPU 빌드를 쓰려면 `uv sync` 뒤에
  `uv pip install --torch-backend cpu -e '.[local]'`을 실행하세요(나중에 그냥
  `uv sync`를 실행하면 지워짐).

## 예제 코퍼스 수집

짧은 텍스트 파일 두 개를 만듭니다(이름, 회사, 수치는 모두 지어낸 것입니다).

```bash
mkdir -p sample-corpus
cat > sample-corpus/acme.txt <<'EOF'
Acme Robotics is a manufacturer of warehouse robots based in Springfield.
Alice Chen is the chief engineer at Acme Robotics and leads its motion-control team.
In 2024 Acme Robotics signed a supply agreement with Globex Sensors.
EOF
cat > sample-corpus/globex.txt <<'EOF'
Globex Sensors makes lidar sensors for mobile robots.
Bob Martin runs the sensor division of Globex Sensors.
Under its agreement with Acme Robotics, Globex Sensors supplies the lidar units
for the Acme warehouse robots.
EOF
```

인덱싱합니다.

```bash
uv run run-ingestion --backend local --source-directory ./sample-corpus
```

- **파일.** 설정 파일이 없으면 저장소가 작업 디렉터리의 `.unified-kg-rag/`에
  생기므로(`graph.sqlite3`, `vector.sqlite3`, 레지스트리를 켜면 `doc-status.sqlite3`도),
  모든 명령을 같은 디렉터리에서 실행하거나 `backend: local`을 적은 설정 파일을
  쓰세요(상대 경로는 그 파일이 있는 디렉터리 기준으로 해석함). 처음부터 다시 하려면
  `.unified-kg-rag/` 디렉터리를 지우거나 `indexing.reset: true`로 한 번 실행합니다.
- **증분 인덱싱**은 프리셋에서 꺼져 있으며 `providers.doc_status.enabled: true`로
  켭니다(실행 로그에 레지스트리의 절대 경로가 남음). 이전 실행이 만든 저장소에는 새
  레지스트리가 모르는 문서가 들어 있으므로, 레지스트리를 켠 첫 실행은 종료 코드 2("was
  built by a run with the doc-status registry off")로 멈춥니다. 이 실행만
  `indexing.reset: true`로(예: `UKR__INDEXING__RESET=true`) 한 번 돌리면 저장소를 다시
  만들고 레지스트리를 채우며, 그다음 실행부터 증분으로 처리합니다.

## GraphRAG와 LightRAG로 질의

`local`은 특정 엔티티와 그 관계를 묻는 질문에 쓰는 GraphRAG 전략이고, `mix`는 LightRAG의
범용 전략입니다.

```bash
uv run run-rag --backend local --search-strategy local --query "Who is the chief engineer at Acme Robotics?"
uv run run-rag --backend local --search-strategy mix --query "How are Acme Robotics and Globex Sensors related?"
```

첫 질의는 기본 재순위화 모델인 크로스 인코더 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`을
Hugging Face에서 내려받으므로 한 번은 인터넷 연결이 필요하고 시간이 더 걸립니다. 이후
질의는 캐시된 모델을 씁니다. 오프라인 환경에서는 미리 내려받거나
`providers.rerank.kind: none`으로 설정하세요([Local mode](../guides/local-mode.md), 영문).

`--search-strategy`를 생략하면 LLM 라우터가 질의마다 전략을 고릅니다(`auto`). 어떤
전략을 언제 쓰는지는 [검색 전략 선택](../concepts/retrieval.ko.md#검색-전략-선택)에 있습니다.

## LLM 판정 모델 없이 평가

작은 데이터셋을 만듭니다. `answer`는 `answer_match`가, `reference_sources`는
`retrieval`이, `expected_entities`는 `graph_aware`가 씁니다
([평가 데이터 형식](../guides/evaluate.ko.md#평가-데이터-형식)).

```bash
cat > sample-eval.json <<'EOF'
[
  {
    "id": "q1",
    "question": "Who is the chief engineer at Acme Robotics?",
    "answer": "Alice Chen",
    "reference_sources": ["acme.txt"],
    "expected_entities": ["Alice Chen", "Acme Robotics"]
  },
  {
    "id": "q2",
    "question": "Which company supplies the lidar units for the Acme warehouse robots?",
    "answer": "Globex Sensors",
    "reference_sources": ["globex.txt"],
    "expected_entities": ["Globex Sensors", "Acme Robotics"]
  }
]
EOF
```

평가기 네 개가 기본으로 켜져 있고, 그중 하나(`ragas`)는 LLM 판정 모델을 호출합니다. LLM이
필요 없는 세 개만 남기고(여기서는 환경 변수로 지정.
[환경 변수로 덮어쓰기](../guides/configure.ko.md#환경-변수로-덮어쓰기) 참고) 같은 질문으로
두 전략을 비교합니다.

```bash
UKR__EVALUATION__ENABLED_EVALUATORS='["answer_match", "retrieval", "graph_aware"]' \
  uv run run-eval --backend local --data-path sample-eval.json --search-strategy local,mix
```

전략마다 결과, 보고서, 요약 파일과 `evaluation_comparison_<timestamp>.json`(`mix`의 지표를
`local`과 짝지어 비교)이 `outputs/evaluation/`에 생깁니다. 지표의 뜻은
[평가기](../guides/evaluate.ko.md#평가기)와 [전략 비교](../guides/evaluate.ko.md#전략-비교)에 있습니다.
질문 두 개로는 동작만 확인할 수 있으므로, 차이를 믿으려면 질문 수백 개로 측정하세요.

## 다음 단계

- [Local mode](../guides/local-mode.md)(영문): 다른 모델 서버, 컨테이너로 띄운 저장소,
  로컬 저장소와 Bedrock 모델.
- [설정](../guides/configure.ko.md): 설정을 읽는 순서와 대부분의 실행에서 바꾸는 설정.
- [AWS에서 시작하기](./aws.ko.md): 같은 흐름을 Amazon Neptune, Amazon OpenSearch, Amazon
  Bedrock에서 실행.
