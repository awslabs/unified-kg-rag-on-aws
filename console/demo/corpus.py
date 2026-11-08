# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console demo's synthetic corpus and evaluation dataset.

Every company, person and figure here is made up: a small procurement story
(a robot maker buys lidar units, a carrier ships them to a depot, a lab
inspects them) in English, with a Korean and a Japanese note on the same
deal, so the graph, the citations and the multilingual paths have something
to show. The files mix the formats the core install parses: PDFs of several
pages with headings and ruled tables (written by ``pdf_writer.py`` when the
demo starts, so no binary is committed), Markdown, HTML, CSV, JSON and text.
The dataset's answers are stated in the corpus.
"""

from __future__ import annotations

import json

from pdf_writer import Block, Heading, Paragraph, Table, render

# The PDFs, page by page. English only: the base-14 fonts are Latin-only.
PDF_DOCUMENTS: dict[str, list[list[Block]]] = {
    "supply-agreement.pdf": [
        [
            Heading("Supply Agreement", 18),
            Paragraph(
                "Between Acme Robotics (the Buyer) and Globex Sensors (the Supplier)"
            ),
            Heading("1. Parties and Recitals", 12),
            Paragraph(
                "Acme Robotics signed a supply agreement with Globex Sensors on "
                "3 March 2026.\n"
                "Acme Robotics builds warehouse robots that find their way with lidar.\n"
                "Globex Sensors supplies every lidar unit for the Acme Robotics "
                "warehouse robots.\n"
                "Alice Chen, the chief engineer at Acme Robotics, approved the "
                "lidar unit specification."
            ),
            Table(
                (
                    ("Party", "Role", "Contact"),
                    ("Acme Robotics", "Buyer", "Alice Chen, chief engineer"),
                    ("Globex Sensors", "Supplier", "Ravi Patel, logistics lead"),
                    ("Umbra Freight", "Carrier", "Named by the Supplier"),
                )
            ),
        ],
        [
            Heading("2. Delivery and Penalties", 12),
            Paragraph(
                "Globex Sensors ships each lidar unit with Umbra Freight.\n"
                "Umbra Freight carries the lidar units to Kestrel Depot within "
                "10 business days.\n"
                "The 10 business days run from each call-off by Acme Robotics.\n"
                "The supply agreement sets a late-delivery penalty of $1,000 per "
                "day, capped at $50,000."
            ),
            Table(
                (
                    ("Term", "Figure", "Applies to"),
                    ("Delivery window", "10 business days", "Each call-off"),
                    ("Late-delivery penalty", "$1,000 per day", "Each late call-off"),
                    ("Penalty cap", "$50,000", "The contract year"),
                    ("First order", "400 lidar units", "May 2026"),
                    ("Unit price", "$1,250", "Each lidar unit"),
                )
            ),
        ],
        [
            Heading("3. Warranty and Signatures", 12),
            Paragraph(
                "Globex Sensors warrants each lidar unit for 24 months from "
                "delivery.\n"
                "The warranty terms set out what the warranty covers.\n"
                "The agreement runs for two years and renews by written consent."
            ),
            Table(
                (
                    ("Party", "Signed by", "Title", "Date"),
                    (
                        "Acme Robotics",
                        "Morgan Lee",
                        "Head of Procurement",
                        "3 March 2026",
                    ),
                    ("Globex Sensors", "Dana Brooks", "Sales Director", "3 March 2026"),
                )
            ),
        ],
    ],
    "quality-report.pdf": [
        [
            Heading("Incoming Inspection Report", 18),
            Paragraph("Prepared by Orion Labs for Acme Robotics, May 2026"),
            Heading("1. Inspection Results", 12),
            Paragraph(
                "Orion Labs inspected 400 lidar units from Globex Sensors in May "
                "2026.\n"
                "Orion Labs rejected 12 units for calibration drift."
            ),
            Table(
                (
                    ("Lot", "Units inspected", "Passed", "Rejected", "Finding"),
                    ("L-0501", "150", "146", "4", "Calibration drift"),
                    ("L-0502", "150", "145", "5", "Calibration drift"),
                    ("L-0503", "100", "97", "3", "Calibration drift"),
                    ("Total", "400", "388", "12", ""),
                )
            ),
        ],
        [
            Heading("2. Corrective Action", 12),
            Paragraph(
                "Alice Chen asked Globex Sensors for a corrective action plan.\n"
                "Globex Sensors replaced the 12 rejected units within two weeks.\n"
                "Orion Labs found the replacement units within tolerance."
            ),
            Table(
                (
                    ("Step", "Owner", "Completed"),
                    ("Root-cause analysis", "Globex Sensors", "20 May 2026"),
                    ("Replacement units shipped", "Umbra Freight", "27 May 2026"),
                    ("Re-inspection", "Orion Labs", "29 May 2026"),
                )
            ),
        ],
    ],
}

_LOGISTICS_PLAN = """\
# Logistics Plan: Lidar Units for Acme Robotics

## Route

Umbra Freight carries each lidar unit from the Globex Sensors plant to Kestrel Depot.
Kestrel Depot stores the units until Acme Robotics calls them off.

## Service Levels

Umbra Freight must deliver to Kestrel Depot within 10 business days of a call-off.

- Booking: two business days before pickup
- Transit insurance: Umbra Freight, door to door
- Proof of delivery: signed at Kestrel Depot

## Roles

Ravi Patel, the logistics lead at Globex Sensors, books every shipment with Umbra Freight.
"""

_WARRANTY_TERMS = """\
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Warranty Terms</title></head>
<body>
<h1>Warranty Terms for Lidar Units</h1>
<p>Globex Sensors warrants each lidar unit for 24 months from delivery to Kestrel Depot.</p>
<p>A lidar unit that fails inside the warranty is replaced at no cost to Acme Robotics.</p>
<h2>Exclusions</h2>
<p>The warranty does not cover damage in transit, which Umbra Freight insures.</p>
<table>
<tr><th>Item</th><th>Term</th></tr>
<tr><td>Warranty period</td><td>24 months from delivery</td></tr>
<tr><td>Replacement</td><td>Within 15 business days of a claim</td></tr>
<tr><td>Damage in transit</td><td>Not covered; insured by the carrier</td></tr>
</table>
</body>
</html>
"""

_SHIPMENTS = """\
shipment,booked_by,carrier,origin,destination,lidar_units,called_off,delivered
UF-2026-031,Ravi Patel,Umbra Freight,Globex Sensors plant,Kestrel Depot,150,2026-04-06,2026-04-14
UF-2026-032,Ravi Patel,Umbra Freight,Globex Sensors plant,Kestrel Depot,150,2026-04-20,2026-04-29
UF-2026-033,Ravi Patel,Umbra Freight,Globex Sensors plant,Kestrel Depot,100,2026-05-04,2026-05-12
UF-2026-041,Ravi Patel,Umbra Freight,Globex Sensors plant,Kestrel Depot,12,2026-05-25,2026-05-27
"""

_STEERING_MINUTES = {
    "meeting": "Acme Robotics steering group",
    "date": "2026-06-02",
    "summary": "The steering group of Acme Robotics met on 2 June 2026.",
    "attendees": ["Alice Chen", "Morgan Lee"],
    "items": [
        {
            "topic": "Quality",
            "note": "Alice Chen reported that Orion Labs found the replacement "
            "units within tolerance.",
        },
        {
            "topic": "Supply",
            "decision": "Acme Robotics decided to extend the supply agreement "
            "with Globex Sensors to a second warehouse.",
        },
        {
            "topic": "Storage",
            "decision": "Kestrel Depot will add a second storage bay for Acme "
            "Robotics.",
        },
    ],
}

_CONTRACT_SUMMARY_KO = """\
# 라이다 공급 계약 요약

애크미 로보틱스는 2026년 3월 글로벡스 센서스와 라이다 공급 계약을 맺었다.

## 주요 조건

지연 납품 위약금은 하루 1,000달러이며 상한은 50,000달러다.
엄브라 프레이트는 케스트럴 창고까지 운송을 맡는다.
"""

_QUALITY_MEMO_JA = (
    "オリオン・ラボはグローベックス・センサーズのライダー400台を検査した。\n"
    "そのうち12台を較正ずれで不合格とした。\n"
)

# The files that are text, as written.
TEXT_DOCUMENTS: dict[str, str] = {
    "logistics-plan.md": _LOGISTICS_PLAN,
    "warranty-terms.html": _WARRANTY_TERMS,
    "shipments.csv": _SHIPMENTS,
    "steering-minutes.json": json.dumps(_STEERING_MINUTES, indent=2) + "\n",
    "contract-summary.ko.md": _CONTRACT_SUMMARY_KO,
    "quality-memo.ja.txt": _QUALITY_MEMO_JA,
}


def documents() -> dict[str, bytes]:
    """Every corpus file's name and bytes (the PDFs rendered here, the same
    bytes on every call)."""
    files = {
        name: render(pages, title=name.removesuffix(".pdf").replace("-", " ").title())
        for name, pages in PDF_DOCUMENTS.items()
    }
    files.update({name: text.encode("utf-8") for name, text in TEXT_DOCUMENTS.items()})
    return dict(sorted(files.items()))


def page_texts(name: str) -> list[str]:
    """A PDF's text, page by page, as laid out (what a parser extracts)."""
    pages = []
    for blocks in PDF_DOCUMENTS[name]:
        lines: list[str] = []
        for block in blocks:
            if isinstance(block, Table):
                lines.extend(" ".join(c for c in row if c) for row in block.rows)
            else:
                lines.extend(block.text.split("\n"))
        pages.append("\n".join(lines))
    return pages


# Earlier demo corpora's file names, removed from a workspace's corpus
# directory when the demo rewrites it, so a kept workspace does not ingest
# both the old and the new copy of a document.
RETIRED_FILES: tuple[str, ...] = (
    "supply-agreement.txt",
    "logistics-plan.txt",
    "quality-report.txt",
    "warranty-terms.txt",
    "steering-minutes.txt",
    "contract-summary.ko.txt",
)

# A run-eval dataset over the corpus (docs/guides/evaluate.md#eval-data-format):
# answers for answer_match, reference documents for retrieval, entities for
# graph_aware, and one question the corpus cannot answer.
DATASET: list[dict[str, object]] = [
    {
        "id": "penalty",
        "question": "What is the late-delivery penalty in the supply agreement?",
        "answer": "$1,000 per day, capped at $50,000",
        "reference_sources": ["supply-agreement.pdf"],
        "expected_entities": ["supply agreement", "Globex Sensors"],
        "metadata": {"answer_aliases": ["$1,000 per day"]},
    },
    {
        "id": "carrier",
        "question": "Who carries the lidar units to Kestrel Depot?",
        "answer": "Umbra Freight",
        "reference_sources": ["logistics-plan.md"],
        "expected_entities": ["Umbra Freight", "Kestrel Depot"],
    },
    {
        "id": "rejected",
        "question": "How many lidar units did Orion Labs reject?",
        "answer": "12",
        "reference_sources": ["quality-report.pdf"],
        "expected_entities": ["Orion Labs", "Globex Sensors"],
        "metadata": {"answer_aliases": ["12 units"]},
    },
    {
        "id": "warranty",
        "question": "How long does Globex Sensors warrant each lidar unit?",
        "answer": "24 months",
        "reference_sources": ["warranty-terms.html"],
        "expected_entities": ["Globex Sensors", "lidar unit"],
    },
    {
        "id": "engineer",
        "question": "Who approved the lidar unit specification at Acme Robotics?",
        "answer": "Alice Chen",
        "reference_sources": ["supply-agreement.pdf"],
        "expected_entities": ["Alice Chen", "Acme Robotics"],
    },
    {
        "id": "ko-carrier",
        "question": "케스트럴 창고까지 운송은 누가 맡나요?",
        "answer": "엄브라 프레이트",
        "reference_sources": ["contract-summary.ko.md"],
        "expected_entities": ["엄브라 프레이트", "케스트럴 창고"],
    },
    {
        "id": "unanswerable",
        "question": "What did Acme Robotics pay for the second warehouse?",
        "metadata": {"answerable": False},
    },
]

# Questions the Ask page suggests for this corpus.
SAMPLE_QUESTIONS: list[str] = [
    "Who carries the lidar units to Kestrel Depot, and how fast?",
    "What happened to the units Orion Labs rejected?",
    "How are Acme Robotics and Globex Sensors related?",
    "지연 납품 위약금은 얼마인가요?",
]
