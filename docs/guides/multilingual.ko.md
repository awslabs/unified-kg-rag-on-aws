# 다국어 코퍼스

> 🇬🇧 English version: [multilingual.md](./multilingual.md)

한국어나 다른 언어, 또는 여러 언어로 된 코퍼스는 번역, 검색 분석기, 청크 크기에 맞게
언어를 지정해야 합니다. 내부 동작은 [설계 §8.3](../design/aws-integration.md#83-multilingual-processing)(영문)에
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
번역 단계를 비용 없이 건너뜁니다. `enabled: false`이면 코퍼스를 원문 그대로 색인하고,
질의도 번역하지 않고 `source_language`로 처리합니다. 질의에 `target_language`를 지정하면
그 언어를 따릅니다.

**색인 언어는 하나입니다.** 번역을 켜면 `target_language`, 끄면 `source_language`로
색인하며 모든 구성 요소가 이 언어를 따릅니다. 추출, 글리닝, 설명 요약, 커뮤니티 보고서,
요약 트리를 이 언어로 쓰고, 텍스트 단위를 이 언어로 임베딩하며, 엔티티·관계·클레임·보고서
인덱스를 이 언어의 분석기로 분석합니다. 질의의 엔티티와 키워드, DRIFT와 iterative의 후속
질의, global 검색의 맵 포인트도 이 언어로 씁니다. 질의가 지정한 언어(`RAGInput.target_language`)는
질의 번역에만 쓰입니다. 텍스트 단위는 추가 대상 언어마다 `translated_text_<language>` 필드를 가져
어휘 검색이 이를 읽지만, 그래프 인덱스는 색인 언어로만 저장되기 때문입니다. 답변은 질의의 언어로
씁니다. 따라서 `enabled: false`이면
`target_language`는 번역 텍스트 필드의 매핑 외에는 영향이 없습니다.

**실패.** 번역에 실패한 텍스트 단위는 원문 언어로 추출합니다. 대상 언어 하나의 호출
전체가 실패하면 `processing.ignore_errors: true`가 아닌 한 단계가 실패합니다. 증분
실행은 `failed_units`의 문서를 실패로 기록하므로 다음 실행이 다시 시도합니다.

**저장된 번역.** 텍스트 단위의 번역은 텍스트와 번역 설정을 키로, 모든 언어를 한 레코드에
저장하므로 재설정이나 문서 수정은 새 텍스트만 번역합니다. `additional_target_languages`(또는
원문 언어, 번역 모델, 번역 프롬프트)를 바꾸면 모든 레코드의 키가 바뀝니다. 다음 구축은 모든
단위를 나열한 모든 언어로 다시 번역하고, 대상 언어 번역이 달라진 단위는 다시 추출합니다. 어떤
단위도 더는 갖지 않는 텍스트의 레코드는 다음 전체 구축(`indexing.reset: true`)이 지우며,
증분 실행은 그대로 둡니다.

## 분석기

- **분석기**는 `indexing.opensearch.language_analyzers`에서 정합니다. 언어 프로필이 있는
  언어는 모두 기본값이 있습니다. `ar: arabic`, `de: german`, `en: english`,
  `es: spanish`, `fr: french`, `hi: hindi`, `it: italian`, `pt: portuguese`,
  `ru: russian`, `th: thai`는 OpenSearch 언어 분석기(불용어 제거와 어간 추출, 태국어는
  사전 기반 단어 분할), `ja: cjk`와 `zh: cjk`는 겹치는 두 글자 단위(플러그인 불필요,
  "首都"가 "日本の首都は" 안에서 맞음), `ko: nori`이며, `he: standard`와
  `vi: standard`는 OpenSearch에 히브리어·베트남어 기본 분석기가 없어서입니다(플러그인
  분석기를 대신 지정할 수 있음). 항목이 없는 언어는 `default_analyzer`를 씁니다.
  `backend: local`의 내장 엔진은 어간 추출 언어를 Snowball(`[local]` extra)로 처리하고
  태국어 연속 문자열을 두 글자 단위로 나누며, 불용어는 두지 않습니다. 청크 `text`는 `source_language`로, 그래프 인덱스는 색인 언어로
  분석하므로 번역하지 않더라도 코퍼스 언어로 지정하세요. `source_language: en`이면 한국어
  코퍼스가 조사가 붙은 단어("홍길동으로부터")를 토큰 하나로 둡니다. 바꾸면 매핑이
  바뀌므로 `indexing.reset: true`로 한 번 실행할 때까지 증분 실행이
  `was built for other language settings`로 멈춥니다. 문서 로딩 단계는 코퍼스 표본(처음
  20개 문서의 앞 2,000자)을 보고, 표본의 문자 체계가 설정과 맞지 않으면
  `processing.translation.source_language`를 지목하는 WARNING을 남깁니다. 다른 문자
  체계의 언어로 설정했는데 한글·가나·한자가 대부분이면(`ko`, `ja`, `zh` 중 하나를 제안),
  `ko`, `ja`, `zh`로 설정했는데 그런 문자가 적으면 경고합니다. 같은 문자 체계의 두
  언어(영어와 독일어)는 구별하지 못하며, 실행은 계속됩니다.
- **유니코드.** 텍스트와 질의는 NFC로 정규화하므로 분해형 한글(NFD)도 조합형과 맞습니다.
- **다른 언어.** `source_language`와 `target_language`에는 언어 프로필이 있는 코드(`ar`,
  `de`, `en`, `es`, `fr`, `he`, `hi`, `it`, `ja`, `ko`, `pt`, `ru`, `th`, `vi`, `zh`)나
  다른 BCP-47 언어 코드(`sw`, `fa`, 지역은 버림: `pt-BR`은 `pt`)를 쓸 수 있습니다.
  프로필이 없는 코드는 WARNING을 한 번 남기고 기본값을 씁니다. 분석기는 OpenSearch에 그
  언어의 내장 분석기가 있으면 그것(`nl`: `dutch`, `sv`: `swedish`, `tr`: `turkish` 등,
  기본 `language_analyzers`에 있음), 없으면 `default_analyzer`이고, 거절 응답은 영어,
  프롬프트는 언어를 코드로 부릅니다. 프로필
  (`unified_kg_rag/domain/languages/profiles/`)은 모듈 하나로, 언어 이름, 문자 체계,
  분석기, 거절 응답, 답변을 읽는 데 쓰는 단어를 담습니다.
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
한국어·중국어·일본어의 예/아니요 답을 읽습니다. 기본 정규화에서 token F1은 한자·한글·가나
한 글자를 토큰 하나로 셉니다(문자 단위 F1). exact match는 조사가 붙은 답을 여전히 낮게
셉니다. 독일어·스페인어·프랑스어·이탈리아어·포르투갈어 답변(질문의 언어 기준)은 그 언어의
관사도 지우고 그 언어의 수사를 읽습니다.
규칙은 [`answer_match`](../reference/evaluation-output.md#answer_match)(영문)에 있습니다.
