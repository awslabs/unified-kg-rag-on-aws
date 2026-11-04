# 도메인 적응

> 🇬🇧 English version: [domain-adaptation.md](./domain-adaptation.md)

파이프라인을 의료, 법률, 금융 같은 도메인에 맞추는 수단은 네 가지입니다. 엔티티 유형과
프롬프트는 이 문서에서, 추출 스키마와 시드 그래프는
[사용자 가이드](../user-guide.ko.md#9-도메인-적응)에서 다룹니다. `run-prompt-tuning`은
코퍼스 표본에서 앞의 두 가지 초안을 만듭니다. 엔티티 유형이나 추출 프롬프트를 바꾸면 다음
증분 실행이 다른 설정으로 추출했던 문서를 다시 추출합니다
([증분 인덱싱: 작업 흐름](./incremental.ko.md#작업-흐름)). 프롬프트와 튜닝의 내부 동작은
[설계 §11](../design.md#11-prompts-and-prompt-tuning)(영문)에 있습니다.

## 권장 흐름

`run-prompt-tuning` → 검토 → 쓸 만한 프롬프트를 합치고 `entity_types`를 조정 → 다시 수집.

## 엔티티 유형

`processing.graph_extraction.entity_types`는 비용이 가장 적고 효과가 가장 큰 수단입니다.
프롬프트를 다시 쓰지 않고 추출 프롬프트가 찾을 범주를 정합니다.

```yaml
processing:
  graph_extraction:
    entity_types:
      - "GENE: Genes, gene products, loci"
      - "DISEASE: Disorders, syndromes, conditions"
      - "DRUG: Medications, compounds, dosages"
      - "TRIAL: Clinical trials, studies, cohorts"
```

## 프롬프트 튜닝 (`run-prompt-tuning`)

디렉터리에서 문서를 표본으로 뽑아 `models.roles.graph_extraction`으로 코퍼스
특성(도메인, 언어, 페르소나, 엔티티 유형)을 분석하고, 검토해서 `config.yaml`에 합칠
도메인 맞춤 `custom_prompts` YAML 조각을 씁니다. 모든 플래그와 기본값은
[CLI 레퍼런스](../reference/cli.md#run-prompt-tuning)(영문)에 있습니다.

```bash
run-prompt-tuning --source-directory ./source --output-path tuned_prompts.yaml --config-path config.yaml
```

출력에는 `custom_prompts` 블록과, 감지한 도메인과 엔티티 유형이 담긴 `profile`이
있습니다. **검토한 뒤** 원하는 프롬프트를 `custom_prompts:` 아래에 복사하세요.

- 튜닝한 `graph_extraction_system`과 `community_report_system`은 도메인 머리말 뒤에
  내장 규칙과 출력 형식을 그대로 붙인 것이며, 생성한 퓨샷 예시는 마지막에 옵니다.
- 엔티티 범주는 여전히 `processing.graph_extraction.entity_types`(`{entity_types}`)에서
  오므로 프로필의 `entity_types`를 그곳에 복사하세요.
- 퓨샷 예시는 프로필의 엔티티 유형으로 추출하므로 프로필이 찾은 유형을 보여 줍니다.
  설정이 엔티티 유형이나 패턴을 직접 선언하면(인라인 또는 `schema_source`. 튜닝도 수집처럼
  스키마 소스를 읽으며, 읽지 못하면 종료 코드 2로 끝남) 그 스키마를 따릅니다. 이때 예시와
  튜닝한 프롬프트가 안내하는 엔티티 유형은 설정한 유형입니다.
- `--schema-out schema.yaml`을 주면 [추출 스키마](../user-guide.ko.md#c-추출-스키마)
  초안도 씁니다. 초안에는 예시를 추출한 엔티티 유형(위 항목), 튜닝 예시가 가장 자주 쓴 관계
  유형(최대 12개, 개수는 주석), 그리고 어떤 관계 유형의 끝점 유형 쌍이 모두 두 번 이상
  나왔을 때만 그 패턴이 들어 있습니다. 설정이 선언한 유형과 패턴은 설명, IRI와 함께 모두
  남고, 설정에 패턴이 있는 관계 유형에는 다른 패턴을 더하지 않습니다.
  `schema_source: {kind: file}` 파일이므로 검토한 뒤 `schema_mode: align`으로
  읽으세요(주석의 경로는 `--config-path` 파일의 디렉터리 기준 상대 경로). 모델 답이 같으면
  같은 파일이 나옵니다. 사용 중인 설정과 스키마는 덮어쓰지 않습니다. `--config-path` 파일이나
  설정이 읽는 스키마 파일을 `--schema-out`이나 `--output-path`로 지정하면 `--force`를
  주어도 종료 코드 2로 끝나고, 두 옵션 중 하나가 이미 있는 파일을 가리켜도 `--force`가
  없으면 종료 코드 2로 끝납니다. 초안을 검토하고 읽는 방법은
  [추출 스키마](../user-guide.ko.md#c-추출-스키마)의 *프롬프트 튜닝으로 스키마 초안
  만들기*에 있습니다.
- 텍스트가 아닌 형식은 수집과 같은 로더(`ParserFactory.register_loader`로 추가한 형식
  포함)를 거치며, 파싱하지 못한 파일은 경고와 함께 건너뜁니다.
- 모델이 JSON 프로필을 돌려주지 않으면 0이 아닌 코드로 끝나고 아무것도 쓰지 않습니다.

## 사용자 정의 프롬프트

`custom_prompts` 아래에서 프롬프트의 `*_system` / `*_human` 텍스트를 재정의합니다(기본값
`null`이면 `unified_kg_rag/domain/prompts/`의 내장 프롬프트를 씀. 재정의 키가 없는
프롬프트 다섯 개는 이 문서 끝에 나옴). 필요한 것만 재정의하고 나머지는 `null`로 두세요.
`{braces}` 안의 변수는 프레임워크가 채우므로 그대로 두세요.

```yaml
custom_prompts:
  # human만 오버라이드하면 내장 system 프롬프트와 출력 형식이 그대로 유지됩니다.
  graph_extraction_human: |
    Extract medical entities and relationships from this clinical text:
    {input_text}
    Extraction Limits:
    - Maximum Entities: {max_entities_per_chunk}
    - Maximum Relationships: {max_relationships_per_chunk}

  entity_extraction_system: |
    You are a financial expert. Extract companies, instruments, markets, and metrics
    from user queries.
```

### 재정의 키

키마다 `_system`과 `_human` 필드가 있습니다: `graph_extraction`,
`description_summarization`, `claim_extraction`, `graph_refinement`,
`community_report`, `answer_generation`, `query_rewrite`, `entity_extraction`,
`drift_hyde`, `drift_primer`, `drift_action`, `strategy_selection`,
`keywords_extraction`, `global_map`, `pairwise_comparison`(`--pairwise-judge` 판정
모델. JSON 판정 형식은 유지하고, `evaluation.pairwise_overall`이면 `overall` 판정도 포함),
`answer_span_extraction`(`evaluation.answer_span: extracted`의 구간 추출기), 그리고 프롬프트
튜닝의 `corpus_profile`. 모든 키와 기본값은
[설정 레퍼런스](../reference/configuration.md#custom_prompts)(영문)에 있습니다.

### 읽을 때 하는 검사

재정의는 설정을 읽을 때 검사하며, 잘못되면 해당 키를 알리고 읽기에 실패합니다.

- 프롬프트 호출에 쓰이지 않는 `{name}`은 오류입니다(호출할 때마다 `KeyError`가 남).
  중괄호를 글자 그대로 쓰려면 `{{` / `}}`로 씁니다: `{{"name": "Vendor"}}`.
  `run-prompt-tuning` 출력은 이미 이스케이프되어 있습니다.
- 각 프롬프트의 데이터 변수는 system이나 human 템플릿에 있어야 합니다
  (`graph_extraction`과 `claim_extraction`은 `{input_text}`, `answer_generation`은
  `{query}`와 `{context}` 등). `{max_entities_per_chunk}` 같은 한도와
  `{entity_types}`, `{relationship_types}`(선언한 관계 유형, 없으면 내장 예시)는
  빼도 됩니다.

### system 프롬프트 바꾸기

`*_system` 오버라이드는 출력 지시를 포함한 내장 시스템 프롬프트 전체를 대체합니다.
`graph_extraction`과 `community_report`에서는 파서가 읽는 XML 형식(추출에서는 환각 방지
검사가 확인하는 원문 그대로의 `<source_text>` 규칙 포함)이 여기에 있으므로 남겨야
합니다. `run-prompt-tuning` 출력에서 시작하거나
`unified_kg_rag/domain/prompts/graph_extraction.py`의 형식을 복사하세요. 두 프롬프트의
오버라이드에 파서가 읽는 태그(`<entities>`, `<relationships>`, `<source_text>`,
`<community_name>`, `<summary>`, `<rating>`, `<rating_explanation>`, `<findings>`)가
빠지면 빠진 태그를 알리는 경고를 남깁니다.

### 재정의할 수 없는 프롬프트

지능형 청커(`TextChunkingPrompt`), 문서 번역(`TextTranslationPrompt`), 질의 번역
(`TranslationPrompt`), 전역 검색의 커뮤니티 평가(`CommunityRelevancePrompt`)와 축약 단계
(`MapReduceSummaryPrompt`)의 프롬프트를 뺀 모든 프롬프트에는 `custom_prompts.*_system` /
`*_human` 재정의 키가 있습니다. 이 다섯 개는 재정의할 수 없습니다.

## 추출 스키마와 시드 그래프

추출이 따르는 관계 유형과 패턴(직접 선언하거나 OWL/RDFS 온톨로지에서 읽음)은
[사용자 가이드: 추출 스키마](../user-guide.ko.md#c-추출-스키마)에 있습니다. 이미 가지고
있는 지식 그래프는 시드 그래프로 추출한 그래프에 합칩니다:
[사용자 가이드: 시드 그래프](../user-guide.ko.md#d-시드-그래프가지고-있는-지식-그래프-가져오기).
