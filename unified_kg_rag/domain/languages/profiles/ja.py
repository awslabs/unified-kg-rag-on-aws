# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Japanese."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.JA,
    name="Japanese",
    script="kana",
    # Built in: overlapping Han and Kana bigrams, no dictionary plugin
    # (kuromoji).
    analyzer="cjk",
    decline=(
        "利用可能なデータには、この質問に答えるための関連情報が見つかりませんでした。"
    ),
    claim_statuses={
        "true": ("真", "真実", "事実", "確認済み", "検証済み"),
        "false": ("偽", "虚偽", "否定", "否認"),
        "disputed": ("係争", "争い", "疑い", "不確か", "不確実", "未確認"),
        "unknown": ("不明", "なし"),
    },
    no_value_words=(
        "不明",
        "不詳",
        "未詳",
        "無",
        "なし",
        "無し",
        "該当なし",
        "不明確",
    ),
    open_date_words=("未定", "現在", "進行中"),
    # Kabushiki kaisha romanized.
    legal_forms={"kk": "kk"},
    # Kabushiki, yugen, godo, goshi and gomei kaisha (with the traditional
    # 會).
    dense_legal_forms=(
        "(株)",
        "株式会社",
        "株式會社",
        "(有)",
        "有限会社",
        "合同会社",
        "合資会社",
        "合名会社",
    ),
    # Han numerals before a counter ("三社", "十二件").
    counter_numerals={
        "〇": 0,
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    },
    numeral_multipliers={
        "十": 10,
        "百": 100,
        "千": 1000,
        "万": 10**4,
        "億": 10**8,
    },
    year_suffixes=("年", "以降", "以来", "以前", "から", "まで"),
    counters=(
        "社",
        "つ",
        "件",
        "人",
        "名",
        "個",
        "箇所",
        "か所",
        "カ所",
        "ヶ所",
        "台",
        # A sum of money ("一億円").
        "円",
    ),
    polarity_words={"はい": "yes", "いいえ": "no"},
    quote_pairs=(("「", "」"), ("『", "』")),
    chars_per_token={CLAUDE_TOKENIZER: 1.03},
)
