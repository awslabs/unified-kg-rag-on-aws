# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The data references of a community report's findings (pure).

The report prompt numbers the entities and relationships it packs and asks
for MS GraphRAG's grounding syntax at the end of each finding:
``[Data: Entities (1, 3); Relationships (2, +more)]``. The numbers mean
something only beside the report's own input, so they are taken out of the
prose (which is embedded and searched) and kept, resolved to record ids, on
the finding. The syntax is ASCII in every report language; a model writing
CJK text may still use full-width brackets, parentheses, colons or commas,
which are read the same.

A stored report keeps its findings, cited ids included, in an unindexed
field (:data:`REPORT_FINDINGS_FIELD`), so a retrieved report (its
metadata, a reported source's) names the records each finding rests on:
:func:`report_cited_ids` reads them.
"""

import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "DATASET_ENTITIES",
    "DATASET_RELATIONSHIPS",
    "REPORT_FINDINGS_FIELD",
    "report_cited_ids",
    "split_data_references",
]

DATASET_ENTITIES = "entities"
DATASET_RELATIONSHIPS = "relationships"

# The stored community report's field holding its findings
# (``CommunityFinding.model_dump()`` each), not searched or filtered.
REPORT_FINDINGS_FIELD = "findings"

_REFERENCE = re.compile(
    r"\s*[\[［]\s*Data\s*[:：](?P<body>[^\]］]*)[\]］]", re.IGNORECASE
)
_DATASET = re.compile(r"(?P<name>[A-Za-z]+)\s*[(（](?P<ids>[^)）]*)[)）]")
_NUMBER = re.compile(r"\d+")


def _dataset(name: str) -> str | None:
    name = name.lower()
    if name.startswith("entit"):
        return DATASET_ENTITIES
    if name.startswith("relation"):
        return DATASET_RELATIONSHIPS
    return None


def split_data_references(text: str) -> tuple[str, dict[str, list[int]]]:
    """``text`` without its ``[Data: ...]`` references, and the record
    numbers they cite by dataset (:data:`DATASET_ENTITIES`,
    :data:`DATASET_RELATIONSHIPS`; others, such as ``Claims``, are
    dropped), each in first-cited order without repeats. ``+more`` is no
    number."""
    cited: dict[str, list[int]] = {}

    def take(match: re.Match[str]) -> str:
        for dataset in _DATASET.finditer(match.group("body")):
            name = _dataset(dataset.group("name"))
            if name is None:
                continue
            numbers = cited.setdefault(name, [])
            for number in _NUMBER.findall(dataset.group("ids")):
                if int(number) not in numbers:
                    numbers.append(int(number))
        return ""

    return _REFERENCE.sub(take, text).strip(), cited


def report_cited_ids(metadata: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The entity and relationship ids the findings of a stored community
    report cite (``metadata`` is the report's stored fields, as a retrieved
    report or a reported source carries them), each in first-cited order
    without repeats. A report stored before its findings were kept, or one
    whose findings cite nothing, cites none."""
    entity_ids: dict[str, None] = {}
    relationship_ids: dict[str, None] = {}
    findings = metadata.get(REPORT_FINDINGS_FIELD)
    for finding in findings if isinstance(findings, list) else []:
        if not isinstance(finding, Mapping):
            continue
        for key, ids in (
            ("entity_ids", entity_ids),
            ("relationship_ids", relationship_ids),
        ):
            cited = finding.get(key)
            for cited_id in cited if isinstance(cited, list) else []:
                if isinstance(cited_id, str) and cited_id:
                    ids.setdefault(cited_id)
    return list(entity_ids), list(relationship_ids)
