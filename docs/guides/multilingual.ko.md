# 다국어 코퍼스

> 🇬🇧 English version: [multilingual.md](./multilingual.md)

한국어나 다른 언어, 또는 여러 언어로 된 코퍼스는 번역, 검색 분석기, 청크 크기에 맞게
언어를 지정해야 합니다. 내부 동작은 [설계 §8.3](../design.md#83-multilingual-processing)(영문)에
있습니다.

## 번역

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

**실패.** 번역에 실패한 텍스트 단위는 원문 언어로 추출합니다. 대상 언어 하나의 호출
전체가 실패하면 `processing.ignore_errors: true`가 아닌 한 단계가 실패합니다. 증분
실행은 `failed_units`의 문서를 실패로 기록하므로 다음 실행이 다시 시도합니다.

## 분석기

- **분석기**는 `indexing.opensearch.language_analyzers`에서 정합니다. 지원하는 언어는
  모두 기본값이 있습니다. `de: german`, `en: english`, `es: spanish`, `fr: french`,
  `it: italian`, `pt: portuguese`, `ru: russian`은 OpenSearch 언어 분석기(불용어 제거와
  어간 추출), `ja: cjk`와 `zh: cjk`는 겹치는 두 글자 단위(플러그인 불필요, "首都"가
  "日本の首都は" 안에서 맞음), `ko: nori`입니다. 항목이 없는 언어는 `default_analyzer`를
  씁니다. 청크 `text`는 `source_language`로
  분석하므로 번역하지 않더라도 코퍼스 언어로 지정하세요. `source_language: en`이면 한국어
  코퍼스가 조사가 붙은 단어("홍길동으로부터")를 토큰 하나로 둡니다. 바꾸면 매핑이
  바뀌므로 텍스트 단위를 다시 인덱싱하세요.
- **유니코드.** 텍스트와 질의는 NFC로 정규화하므로 분해형 한글(NFD)도 조합형과 맞습니다.
- **분석기는 클러스터에 있어야** 하며, 없으면 인덱싱 단계가
  `the cluster has no analyzer '<name>'`로 실패합니다. `nori`는 `analysis-nori`
  플러그인에서 오는데, [로컬 OpenSearch 컨테이너](./local-mode.md#local-stores-in-containers)(영문)에는 있고
  Amazon OpenSearch Service 도메인은 Nori 패키지를 연결해야 갖습니다(블루/그린 배포). 그래서
  [CDK 배포](../../iac/README.md)는 `ko`를 기본 제공 `cjk` 분석기(문자 바이그램)에
  매핑합니다. `-c opensearch_nori_package=true`로 배포하면 패키지를 연결하고 `ko`를
  `nori`에 매핑합니다([Nori로 한국어 분석하기](../operations.md#korean-with-nori), 영문).

## 청크 크기

청크 크기는 기본값으로 문자 수를 셉니다. 한글·한자·가나 한 글자는 토큰 하나쯤이므로
8,000자짜리 한국어 청크는 약 6,000토큰이 되어 Cohere Rerank 3.5의 문서당 한도 4,096토큰을
넘고, 넘은 뒷부분은 재순위화(rerank)되지 않습니다. 크기를 절반으로 줄이거나,
`processing.chunking.length_unit: tokens`로 토큰 수를 세게 하고 모든 크기를 토큰 단위로
지정하세요. 토큰 수는 로컬 추정치이며 모델을 호출하지 않습니다.

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

## 한국어·CJK 답변 평가

`answer_match` 평가기는 한국어 조사와 단위를 허용하고, CJK 문자 옆의 띄어쓰기를 무시하며,
한국어·중국어·일본어의 예/아니요 답을 읽습니다. 이 언어들에서는 token F1이 낮게 나옵니다.
규칙은 [`answer_match`](../reference/evaluation-output.md#answer_match)(영문)에 있습니다.
