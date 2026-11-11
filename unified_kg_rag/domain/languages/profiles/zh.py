# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chinese, simplified and traditional."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.ZH,
    name="Chinese",
    script="han",
    # Built in: overlapping Han bigrams, no dictionary plugin (smartcn).
    analyzer="cjk",
    decline="在现有数据中未找到回答此问题的相关信息。",
    claim_statuses={
        "true": (
            "真",
            "真实",
            "真實",
            "事实",
            "事實",
            "已确认",
            "已確認",
            "已证实",
            "已證實",
        ),
        "false": (
            "假",
            "虚假",
            "虛假",
            "否定",
            "否認",
            "反駁",
            "已否认",
            "已否認",
            "已驳斥",
            "已駁斥",
        ),
        "disputed": ("争议", "爭議", "有争议", "有爭議"),
        "suspected": (
            "可疑",
            "涉嫌",
            "不确定",
            "不確定",
            "未证实",
            "未證實",
            "未确认",
        ),
        "unknown": ("不明", "未知", "无", "無"),
    },
    no_value_words=("不明", "未知", "不详", "不詳", "无", "無"),
    open_date_words=("未定", "待定", "现在", "現在", "至今", "进行中", "進行中"),
    # Company limited by shares, limited liability company, limited company.
    dense_legal_forms=(
        "股份有限公司",
        "有限责任公司",
        "有限責任公司",
        "有限公司",
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
        # Two before a counter ("两家", "兩個").
        "两": 2,
        "兩": 2,
    },
    numeral_multipliers={
        "十": 10,
        "百": 100,
        "千": 1000,
        "万": 10**4,
        "萬": 10**4,
        "亿": 10**8,
        "億": 10**8,
    },
    # And a sum of money ("三千万元").
    counters=("家", "个", "個", "件", "人", "名", "台", "元"),
    polarity_words={
        "是": "yes",
        "是的": "yes",
        "对": "yes",
        "对的": "yes",
        "否": "no",
        "不是": "no",
        "不是的": "no",
        "没有": "no",
    },
    quote_pairs=(("「", "」"), ("『", "』")),
    chars_per_token={CLAUDE_TOKENIZER: 0.95},
)
