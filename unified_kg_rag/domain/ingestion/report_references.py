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
:func:`report_cited_ids` reads them. Global search and DRIFT cite a
report by the text units of those records (:func:`report_cited_records`),
not by its community's membership.
"""

import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "DATASET_ENTITIES",
    "DATASET_RELATIONSHIPS",
    "FINDINGS_DOCUMENTS_FIELD",
    "FINDINGS_RECORD_IDS_FIELD",
    "FINDINGS_TEXT_UNITS_FIELD",
    "REPORT_FINDINGS_FIELD",
    "findings_record_ids",
    "repointed_findings",
    "report_cited_ids",
    "report_cited_records",
    "split_data_references",
]

DATASET_ENTITIES = "entities"
DATASET_RELATIONSHIPS = "relationships"

# The stored community report's field holding its findings
# (``CommunityFinding.model_dump()`` each), not searched or filtered.
REPORT_FINDINGS_FIELD = "findings"

# The stored community report's flat keyword list of every record id its
# findings cite (entities and relationships alike: their ids do not
# collide), so the stores can find the reports citing an id an incremental
# run re-keyed and point them at its new id (``repointed_findings``).
FINDINGS_RECORD_IDS_FIELD = "findings_record_ids"

# Set by the search strategies on a retrieved report whose findings cite
# stored records: the text units and documents of those records, which its
# citation names in place of its community's membership (``text_unit_ids``,
# ``document_ids``, which the report keeps).
FINDINGS_TEXT_UNITS_FIELD = "findings_text_unit_ids"
FINDINGS_DOCUMENTS_FIELD = "findings_document_ids"

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


def report_cited_records(metadata: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The records the findings of a stored community report cite, as
    ``(dataset, id)`` pairs finding by finding (a finding's relationships,
    the narrower evidence, before its entities), each once. A report stored
    before its findings were kept, or one whose findings cite nothing,
    cites none."""
    records: dict[tuple[str, str], None] = {}
    findings = metadata.get(REPORT_FINDINGS_FIELD)
    for finding in findings if isinstance(findings, list) else []:
        if not isinstance(finding, Mapping):
            continue
        for key, dataset in (
            ("relationship_ids", DATASET_RELATIONSHIPS),
            ("entity_ids", DATASET_ENTITIES),
        ):
            cited = finding.get(key)
            for cited_id in cited if isinstance(cited, list) else []:
                if isinstance(cited_id, str) and cited_id:
                    records.setdefault((dataset, cited_id))
    return list(records)


def repointed_findings(
    findings: Any, id_remap: Mapping[str, str]
) -> list[dict[str, Any]] | None:
    """The stored ``findings`` (a report's :data:`REPORT_FINDINGS_FIELD`)
    with each cited id ``id_remap`` replaces by its new id (repeats
    dropped), or ``None`` when none is replaced. The findings of a stored
    report cite the ids of the run that made it: an incremental run that
    re-keys a stored item (its name changed) points them at its new one."""
    if not isinstance(findings, list):
        return None
    changed = False
    repointed: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            repointed.append(finding)
            continue
        finding = dict(finding)
        for key in ("entity_ids", "relationship_ids"):
            cited = finding.get(key)
            if not isinstance(cited, list):
                continue
            if any(i in id_remap for i in cited):
                changed = True
                finding[key] = list(dict.fromkeys(id_remap.get(i, i) for i in cited))
        repointed.append(finding)
    return repointed if changed else None


def findings_record_ids(findings: Any) -> list[str]:
    """Every record id ``findings`` cite (:data:`FINDINGS_RECORD_IDS_FIELD`),
    in first-cited order without repeats."""
    entity_ids, relationship_ids = report_cited_ids({REPORT_FINDINGS_FIELD: findings})
    return list(dict.fromkeys([*entity_ids, *relationship_ids]))
