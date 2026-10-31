# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claims of a full build and of incremental runs converge (embedded stores).

One claim stated with two spellings of its subject ("Acme Inc", "Acme") in
two documents is one claim in a full build. Incremental runs that index the
documents one after the other must store that same single claim (and remove
the right text unit when a document is deleted), not one claim per spelling.
"""

from __future__ import annotations

import re
import shutil
from itertools import pairwise
from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import (
    DIM,
    FakeCrossEncoder,
    _config,
    _errors,
    _ingest,
    _providers,
    _stored,
)
from unified_kg_rag.domain.models import PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws

NAMES = ("Acme Inc", "Acme", "Buyer", "Depot", "Globex")


def _names_in(text: str) -> list[str]:
    found, rest = [], text
    for name in sorted(NAMES, key=len, reverse=True):
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, rest):
            found.append(name)
            rest = re.sub(pattern, " ", rest)
    return sorted(found, key=text.find)


def _respond(system: str, human: str) -> str:
    if "knowledge graph extraction expert" in system:
        text = stages._source_text(human)
        names = _names_in(text)
        entities = "".join(
            f"<entity><name>{n}</name><type>ORGANIZATION</type>"
            f"<description>{n} desc</description><confidence>9</confidence>"
            f"<source_text>{n}</source_text></entity>"
            for n in names
        )
        relationships = "".join(
            f"<relationship><source>{a}</source><target>{b}</target>"
            f"<type>WORKS_WITH</type><description>{a} works with {b}"
            f"</description><strength>8</strength>"
            f"<source_text>{text.strip()}</source_text></relationship>"
            for a, b in pairwise(names)
        )
        return f"<entities>{entities}</entities><relationships>{relationships}</relationships>"
    if "claim extraction specialist" in system:
        names = _names_in(stages._source_text(human))
        if len(names) < 2:
            return "<claims></claims>"
        subject, obj = names[:2]
        return (
            f"<claims><claim><subject>{subject}</subject><object>{obj}</object>"
            "<claim_type>OBLIGATION</claim_type><claim_status>TRUE</claim_status>"
            f"<description>{subject} must deliver to {obj}</description>"
            f"<source_text>{subject} supplies {obj}.</source_text></claim></claims>"
        )
    return stages._respond(system, human)


def _claims_after(tmp: Path, corpora: list[dict[str, str]]) -> list[tuple]:
    """(subject name, object name, cited text units) of every stored claim.

    Compared by the names of the entities the claim points at: the entity
    ids themselves may differ between the two builds (an incremental run
    keeps the stored entity's id).
    """
    config = _config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
    providers = _providers(
        config, FakeOllama(_respond, "x", DIM), FakeCrossEncoder("r")
    )
    source = tmp / "corpus"
    for run, corpus in enumerate(corpora):
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        for name, text in corpus.items():
            (source / name).write_text(text)
        context = _ingest(config, providers, source, tmp / f"run-{run}")
        assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    names = {
        hit["_id"]: hit["_source"]["name"]
        for hit in _stored(
            config, providers, config.indexing.opensearch.entities_index_prefix
        )
    }
    hits = _stored(config, providers, config.indexing.opensearch.claims_index_prefix)
    return sorted(
        (
            names[hit["_source"]["subject_id"]].removesuffix(" Inc"),
            names[hit["_source"]["object_id"]],
            len(hit["_source"].get("text_unit_ids") or []),
        )
        for hit in hits
    )


FIRST = {"a.txt": "Acme Inc supplies Buyer.", "c.txt": "Globex pays Depot."}
FINAL = {"a.txt": "Acme Inc supplies Buyer.", "d.txt": "Acme supplies Buyer."}


def test_incremental_claims_match_a_full_build(tmp_path: Path) -> None:
    full = _claims_after(tmp_path / "full", [FINAL])
    incremental = _claims_after(tmp_path / "inc", [FIRST, FINAL])

    assert full == [("Acme", "Buyer", 2)]
    assert incremental == full


def test_deleting_a_document_drops_its_text_unit_from_the_claim(
    tmp_path: Path,
) -> None:
    after_delete = {"a.txt": FINAL["a.txt"]}
    full = _claims_after(tmp_path / "full", [after_delete])
    incremental = _claims_after(tmp_path / "inc", [FIRST, FINAL, after_delete])

    assert full == [("Acme", "Buyer", 1)]
    assert incremental == full


def _dated(system: str, human: str) -> str:
    """``_respond`` with the claim dated."""
    answer = _respond(system, human)
    return answer.replace(
        "<claim_status>", "<start_date>2024-01-01</start_date><claim_status>"
    )


def test_a_dated_claim_keeps_its_dates_through_incremental_runs(
    tmp_path: Path,
) -> None:
    # The claims index did not store a claim's dates, which are part of its
    # id (resolved_claim_id): a stored claim read back for a merge or a
    # re-key lost them, and shown it read like an undated one.
    from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id
    from unified_kg_rag.domain.models import Claim

    def stored(tmp: Path, corpora: list[dict[str, str]]) -> list[tuple]:
        config = _config(tmp / "stores", tmp / "viz")
        config.processing.claim_extraction.enabled = True
        providers = _providers(
            config, FakeOllama(_dated, "x", DIM), FakeCrossEncoder("r")
        )
        source = tmp / "corpus"
        for run, corpus in enumerate(corpora):
            if source.exists():
                shutil.rmtree(source)
            source.mkdir(parents=True)
            for name, text in corpus.items():
                (source / name).write_text(text)
            _ingest(config, providers, source, tmp / f"run-{run}")
        hits = _stored(
            config, providers, config.indexing.opensearch.claims_index_prefix
        )
        for hit in hits:
            # The id is the one the claim's stored fields give.
            fields = {
                k: hit["_source"].get(k)
                for k in ("subject_id", "object_id", "object_name", "type")
            }
            claim = Claim(
                id=hit["_id"],
                subject_name="",
                start_date=hit["_source"].get("start_date"),
                end_date=hit["_source"].get("end_date"),
                **{k: v or "" for k, v in fields.items()},
            )
            assert resolved_claim_id(claim) == hit["_id"], hit["_source"]
        return sorted(
            (h["_source"]["start_date"], len(h["_source"]["text_unit_ids"]))
            for h in hits
        )

    full = stored(tmp_path / "full", [FINAL])
    assert full == [("2024-01-01", 2)]
    assert stored(tmp_path / "inc", [FIRST, FINAL]) == full


_COARSE_DATES = (
    ("Q3 2022", "2024"),
    ("Sept. 2023", "UNKNOWN"),
    ("2023-w05", "Q4-2024"),
)


def _coarsely_dated(system: str, human: str) -> str:
    """``_respond`` with claims of several types, dated by periods coarser
    than a day, in each text unit."""
    answer = _respond(system, human)
    if "<claim>" not in answer:
        return answer
    claim = answer[answer.index("<claim>") : answer.index("</claims>")]
    return "<claims>{}</claims>".format(
        "".join(
            claim.replace("OBLIGATION", f"OBLIGATION{i}").replace(
                "<claim_status>",
                f"<start_date>{start}</start_date><end_date>{end}</end_date>"
                "<claim_status>",
            )
            for i, (start, end) in enumerate(_COARSE_DATES)
        )
    )


def _as_written_coarse_dates(value: str | None, *, end: bool = False) -> str | None:
    """Claim dates as builds before #451 read them: a period coarser than a
    day as written."""
    from datetime import datetime

    text = " ".join((value or "").split())
    if text.casefold() in {"", "unknown"}:
        return None
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return text


@pytest.mark.parametrize(
    ("stated", "restated"),
    [
        pytest.param(
            "Acme Inc supplies Buyer.",
            ["Acme Inc supplies Buyer twice."],
            id="same-words",
        ),
        pytest.param(
            "Acme Inc supplies Buyer.", ["Acme supplies Buyer."], id="other-spelling"
        ),
        # "Acme Inc", now in more units, becomes the entity's name: the
        # stored "Acme" entity is re-keyed, and its claims with it.
        pytest.param(
            "Acme supplies Buyer.",
            ["Acme Inc supplies Buyer twice.", "Acme Inc supplies Buyer again."],
            id="rekeyed",
        ),
    ],
)
@pytest.mark.parametrize("legacy", ["as written", "end as first day"])
def test_a_claim_stored_under_an_earlier_date_reading_merges_with_its_restatement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy: str,
    stated: str,
    restated: list[str],
) -> None:
    # Regression: a claim id keys on its dates. A claim an earlier build
    # stored under another reading of its dates ("Q3 2022" as written, a
    # coarse end date as its first day) kept that id, and a document
    # restating it after the upgrade stored a second copy beside it.
    from tests.fixtures.fakes import stores
    from unified_kg_rag.adapters.storage import opensearch_indexer
    from unified_kg_rag.domain.ingestion import claim_resolver

    current = claim_resolver.claim_date_value

    def first_day_end(value: str | None, *, end: bool = False) -> str | None:
        return current(value)

    reading = _as_written_coarse_dates if legacy == "as written" else first_day_end

    def stored(tmp: Path, corpora: list[dict[str, str]], upgrade_at: int) -> list:
        config = _config(tmp / "stores", tmp / "viz")
        config.processing.claim_extraction.enabled = True
        providers = _providers(
            config, FakeOllama(_coarsely_dated, "x", DIM), FakeCrossEncoder("r")
        )
        source = tmp / "corpus"
        for run, corpus in enumerate(corpora):
            for module in (claim_resolver, opensearch_indexer, stores):
                monkeypatch.setattr(
                    module,
                    "claim_date_value",
                    reading if run < upgrade_at else current,
                )
            if source.exists():
                shutil.rmtree(source)
            source.mkdir(parents=True)
            for name, text in corpus.items():
                (source / name).write_text(text)
            context = _ingest(config, providers, source, tmp / f"run-{run}")
            assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
        hits = _stored(
            config, providers, config.indexing.opensearch.claims_index_prefix
        )
        return sorted(
            (
                h["_id"],
                h["_source"].get("start_date"),
                h["_source"].get("end_date"),
                len(h["_source"]["text_unit_ids"]),
            )
            for h in hits
        )

    # Restated in the same words: no entity is re-keyed, so nothing else
    # derives the stored claim again. Restated by another spelling, the
    # entity keeps its id ("Acme Inc" and "Acme" in one unit each: the
    # smaller name) or is re-keyed with its claims ("rekeyed").
    first = {"a.txt": stated}
    final = {**first, **{f"d{i}.txt": text for i, text in enumerate(restated)}}
    full = stored(tmp_path / "full", [final], upgrade_at=0)
    units = 1 + len(restated)
    assert sorted(row[1:] for row in full) == [
        ("2022-07-01", "2024-12-31", units),
        ("2023-01-30", "2024-12-31", units),
        ("2023-09-01", None, units),
    ]
    # Indexed by the earlier build, then the restating document by this one.
    assert stored(tmp_path / "inc", [first, final], upgrade_at=1) == full


def _end_dated(system: str, human: str) -> str:
    """``_respond`` with the claim ending in 2024: "2024" in one document,
    "2024-01-01" in the one saying "twice"."""
    answer = _respond(system, human)
    end = "2024-01-01" if "twice" in human else "2024"
    return answer.replace(
        "<claim_status>",
        f"<start_date>2023-01-01</start_date><end_date>{end}</end_date>"
        "<claim_status>",
    )


def test_an_earlier_reading_id_that_is_another_claims_id_keeps_them_apart(
    tmp_path: Path,
) -> None:
    # "until 2024" read its end as 2024-01-01 before #454 (as a start date
    # reads): that earlier-reading id is the current id of the claim ending
    # on 2024-01-01, a distinct fact. The earlier reading must not take it.
    def stored(tmp: Path, corpora: list[dict[str, str]]) -> list[tuple]:
        config = _config(tmp / "stores", tmp / "viz")
        config.processing.claim_extraction.enabled = True
        providers = _providers(
            config, FakeOllama(_end_dated, "x", DIM), FakeCrossEncoder("r")
        )
        source = tmp / "corpus"
        for run, corpus in enumerate(corpora):
            if source.exists():
                shutil.rmtree(source)
            source.mkdir(parents=True)
            for name, text in corpus.items():
                (source / name).write_text(text)
            context = _ingest(config, providers, source, tmp / f"run-{run}")
            assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
        hits = _stored(
            config, providers, config.indexing.opensearch.claims_index_prefix
        )
        return sorted(
            (
                h["_id"],
                h["_source"].get("end_date"),
                len(h["_source"]["text_unit_ids"]),
            )
            for h in hits
        )

    b = {"b.txt": "Acme Inc supplies Buyer twice."}
    final = {**b, "a.txt": "Acme Inc supplies Buyer."}
    full = stored(tmp_path / "full", [final])
    assert sorted(row[1:] for row in full) == [("2024-01-01", 1), ("2024-12-31", 1)]
    assert stored(tmp_path / "inc", [b, final]) == full
    assert stored(tmp_path / "inc2", [{"a.txt": final["a.txt"]}, final]) == full
