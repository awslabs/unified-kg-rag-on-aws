# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prepare-benchmark musique --unanswerable/--hops`` on synthetic MuSiQue-Full
data: twin exclusion, the exact-text contamination filter, hop sampling."""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.evaluation import EvaluationManager
from unified_kg_rag.evaluation.benchmarks import musique

pytestmark = pytest.mark.unit


def _pair(
    qid: str, supporting: list[str], distractors: list[str], removed: int = 1
) -> list[dict[str, Any]]:
    """An answerable item and its unanswerable twin: the twin has the first
    ``removed`` supporting paragraphs replaced by fresh distractors."""

    def paragraph(i: int, text: str, is_supporting: bool) -> dict[str, Any]:
        return {
            "idx": i,
            "title": text.split()[0],
            "paragraph_text": text,
            "is_supporting": is_supporting,
        }

    texts = [(t, True) for t in supporting] + [(t, False) for t in distractors]
    answerable = {
        "id": qid,
        "question": f"Which depot serves {qid}?",
        "answer": f"Depot {qid}",
        "answer_aliases": [f"The depot of {qid}"],
        "answerable": True,
        "paragraphs": [paragraph(i, t, s) for i, (t, s) in enumerate(texts)],
    }
    replaced = [
        (f"Filler {qid} {n} about nothing.", False) for n in range(removed)
    ] + texts[removed:]
    twin = {
        **answerable,
        "answerable": False,
        "paragraphs": [paragraph(i, t, s) for i, (t, s) in enumerate(replaced)],
    }
    return [answerable, twin]


def _write(path: Path, items: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    return path


def _full_file(tmp_path: Path) -> Path:
    items: list[dict[str, Any]] = []
    for n, prefix in enumerate(["2hop", "2hop", "3hop1", "3hop2", "4hop1", "4hop3"]):
        qid = f"{prefix}__{n}"
        items += _pair(
            qid,
            [f"Vendor {qid} ships to Buyer.", f"Buyer {qid} uses the depot."],
            [f"Noise {qid} {k}." for k in range(3)],
        )
    return _write(tmp_path / "full.jsonl", items)


def _rows(out: Path) -> list[dict[str, Any]]:
    return json.loads((out / "eval_data.json").read_text(encoding="utf-8"))


def _unanswerable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["metadata"].get("answerable") is False]


def test_unanswerable_rows_are_marked_and_load_as_a_dataset(tmp_path: Path) -> None:
    out = tmp_path / "out"
    counts = musique.prepare(_full_file(tmp_path), out, limit=2, unanswerable=2, seed=1)
    assert counts["answerable"] == 2 and counts["unanswerable"] == 2
    assert counts["questions"] == 4
    assert counts["unanswerable_filter"]["kept"] == 2
    rows = _unanswerable(_rows(out))
    assert len(rows) == 2
    for row in rows:
        assert row["query_id"].endswith("__unanswerable")
        assert row["answer"] == "" and row["reference_sources"] == []
        assert row["metadata"]["benchmark"] == "musique-full-unanswerable"
        base = row["query_id"].removesuffix("__unanswerable")
        assert row["metadata"]["twin_answer"] == f"Depot {base}"
        assert row["category"] == f"{musique.hop_count(base)}hop"
    queries, truths = EvaluationManager.load_data(out / "eval_data.json")
    assert sum(q.metadata.get("answerable") is False for q in queries) == 2


def test_the_twin_of_a_sampled_question_is_never_pooled(tmp_path: Path) -> None:
    # Every answerable item sampled: each unanswerable one has its twin in
    # the pool, which would give its removed paragraph back.
    with pytest.raises(musique.InvalidBenchmarkDataError, match="only 0 of the 1"):
        musique.prepare(_full_file(tmp_path), tmp_path / "out", unanswerable=1)
    out = tmp_path / "ok"
    counts = musique.prepare(_full_file(tmp_path), out, limit=3, unanswerable=3)
    sampled = {r["query_id"] for r in _rows(out) if "answerable" not in r["metadata"]}
    kept = {
        r["query_id"].removesuffix("__unanswerable") for r in _unanswerable(_rows(out))
    }
    assert sampled.isdisjoint(kept)
    assert counts["unanswerable_filter"]["excluded_twin"] == (
        counts["unanswerable_filter"]["examined"] - 3
    )


def test_a_removed_paragraph_elsewhere_in_the_pool_is_contamination(
    tmp_path: Path,
) -> None:
    # The removed paragraph of "2hop__u" is also a distractor of the
    # answerable "2hop__a" (whitespace differs): pooled, it answers it.
    leaked = "Vendor u ships to Buyer."
    items = _pair("2hop__a", ["A one.", "A two."], ["Vendor  u ships\nto Buyer."])
    items += _pair("2hop__u", [leaked, "Buyer u uses the depot."], ["U noise."])
    items += _pair("2hop__c", ["C one.", "C two."], ["C noise."])
    source = _write(tmp_path / "full.jsonl", items)
    hits = []
    for seed in range(8):
        out = tmp_path / f"s{seed}"
        try:
            counts = musique.prepare(source, out, limit=1, unanswerable=1, seed=seed)
        except musique.InvalidBenchmarkDataError:
            continue
        kept = _unanswerable(_rows(out))[0]["query_id"]
        sampled = [
            r["query_id"] for r in _rows(out) if "answerable" not in r["metadata"]
        ]
        if sampled == ["2hop__a"]:
            assert kept != "2hop__u__unanswerable"
            hits.append(counts["unanswerable_filter"]["excluded_contaminated"])
    # Some seed sampled 2hop__a and visited 2hop__u before keeping another.
    assert hits and max(hits) >= 1


def test_a_kept_item_never_leaks_into_another_kept_item(tmp_path: Path) -> None:
    # "2hop__y" carries the removed paragraph of "2hop__x" as a distractor:
    # the two are never kept together, whichever is visited first.
    removed_x = "X fact one."
    items = _pair("2hop__x", [removed_x, "X fact two."], ["X noise."])
    items += _pair("2hop__y", ["Y fact one.", "Y fact two."], [removed_x])
    items += _pair("2hop__z", ["Z fact one.", "Z fact two."], ["Z noise."])
    twins = {i["id"]: i for i in items if i["answerable"]}
    candidates = [i for i in items if not i["answerable"]]
    for seed in range(20):
        kept, counts = musique.select_unanswerable(
            candidates, twins, [], 2, random.Random(seed)
        )
        ids = {item["id"] for item, _ in kept}
        assert ids != {"2hop__x", "2hop__y"}
        if counts["examined"] == 3:
            assert counts["excluded_contaminated"] == 1
        # The pool of what is kept never holds a removed paragraph.
        pool = {
            " ".join(p["paragraph_text"].split())
            for item, _ in kept
            for p in item["paragraphs"]
        }
        for item, twin in kept:
            assert not musique.removed_paragraphs(twin, item) & pool


def test_an_unanswerable_item_without_its_twin_is_left_out(tmp_path: Path) -> None:
    answerable, twin = _pair("2hop__1", ["One.", "Two."], ["Noise."])
    source = _write(
        tmp_path / "full.jsonl", [twin, *_pair("2hop__2", ["A.", "B."], ["C."])]
    )
    with pytest.raises(musique.InvalidBenchmarkDataError) as error:
        musique.prepare(source, tmp_path / "out", limit=1, unanswerable=2, seed=0)
    assert "excluded_no_twin" in str(error.value)


def test_unanswerable_needs_a_full_file(tmp_path: Path) -> None:
    answerable, _ = _pair("2hop__1", ["One.", "Two."], ["Noise."])
    source = _write(tmp_path / "ans.jsonl", [answerable])
    with pytest.raises(musique.InvalidBenchmarkDataError, match="full file"):
        musique.prepare(source, tmp_path / "out", unanswerable=1)


def test_an_incomplete_unanswerable_item_is_refused_when_sampled(
    tmp_path: Path,
) -> None:
    answerable, _ = _pair("2hop__1", ["One.", "Two."], ["Noise."])
    source = _write(
        tmp_path / "full.jsonl",
        [answerable, {"id": "2hop__1", "answerable": False, "paragraphs": []}],
    )
    # Not sampled: skipped as before.
    assert musique.prepare(source, tmp_path / "a")["questions"] == 1
    with pytest.raises(musique.InvalidBenchmarkDataError, match="complete"):
        musique.prepare(source, tmp_path / "b", limit=1, unanswerable=1)


def test_the_answerable_sample_does_not_depend_on_unanswerable(
    tmp_path: Path,
) -> None:
    source = _full_file(tmp_path)
    musique.prepare(source, tmp_path / "a", limit=3, seed=5)
    musique.prepare(source, tmp_path / "b", limit=3, seed=5, unanswerable=2)
    answerable = [r for r in _rows(tmp_path / "b") if "answerable" not in r["metadata"]]
    assert answerable == _rows(tmp_path / "a")
    # And it is the sample of earlier releases: random.Random(seed).sample.
    every = [json.loads(line) for line in source.read_text().splitlines()]
    expected = random.Random(5).sample([i for i in every if i["answerable"]], 3)
    assert [r["query_id"] for r in answerable] == [i["id"] for i in expected]


def test_the_selection_is_deterministic_per_seed(tmp_path: Path) -> None:
    source = _full_file(tmp_path)
    musique.prepare(source, tmp_path / "a", limit=2, unanswerable=3, seed=2)
    musique.prepare(source, tmp_path / "b", limit=2, unanswerable=3, seed=2)
    assert _rows(tmp_path / "a") == _rows(tmp_path / "b")
    assert sorted(p.name for p in (tmp_path / "a" / "corpus").iterdir()) == sorted(
        p.name for p in (tmp_path / "b" / "corpus").iterdir()
    )


def test_hop_filter_applies_to_both_pools(tmp_path: Path) -> None:
    out = tmp_path / "out"
    counts = musique.prepare(
        _full_file(tmp_path),
        out,
        limit=2,
        unanswerable=2,
        seed=0,
        hops=musique.parse_hops("3,4"),
    )
    assert {r["category"] for r in _rows(out)} <= {"3hop", "4hop"}
    assert sum(counts["hops"].values()) == 2


def test_hop_quotas_draw_exactly_that_many_per_hop(tmp_path: Path) -> None:
    source = _full_file(tmp_path)
    hops = musique.parse_hops("2=1,4=2")
    counts = musique.prepare(source, tmp_path / "a", hops=hops, seed=3)
    assert counts["hops"] == {"2hop": 1, "4hop": 2}
    assert [r["category"] for r in _rows(tmp_path / "a")] == ["2hop", "4hop", "4hop"]
    musique.prepare(source, tmp_path / "b", hops=hops, seed=3)
    assert _rows(tmp_path / "a") == _rows(tmp_path / "b")
    with pytest.raises(musique.InvalidBenchmarkDataError, match="3 2-hop"):
        musique.prepare(source, tmp_path / "c", hops=musique.parse_hops("2=3"))
    with pytest.raises(musique.InvalidBenchmarkDataError, match="drop --limit"):
        musique.prepare(source, tmp_path / "d", hops=hops, limit=2)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("5", "not one of"),
        ("2,2", "twice"),
        ("2=0", "at least 1"),
        ("2=5,3", "every hop count or for none"),
        ("two", "not a hop count"),
    ],
)
def test_parse_hops_rejects(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        musique.parse_hops(value)


def test_download_full_checks_its_own_digest(tmp_path: Path, mocker) -> None:
    body = b'{"id": "2hop__1"}\n'
    urlopen = mocker.patch.object(
        musique.urllib.request, "urlopen", return_value=io.BytesIO(body)
    )
    with pytest.raises(musique.EvaluationException, match="SHA-256"):
        musique.download_dev(tmp_path / "full.jsonl", full=True)
    assert urlopen.call_args.args[0] == musique.FULL_DEV_URL
    mocker.patch.object(musique, "FULL_DEV_SHA256", hashlib.sha256(body).hexdigest())
    mocker.patch.object(
        musique.urllib.request, "urlopen", return_value=io.BytesIO(body)
    )
    assert musique.download_dev(tmp_path / "full.jsonl", full=True).read_bytes() == body


def test_cli_reports_the_selection(tmp_path: Path, capsys) -> None:
    from unified_kg_rag.application.cli import prepare_benchmark

    prepare_benchmark.main(
        [
            "musique",
            "--output-directory",
            str(tmp_path / "b"),
            "--data-path",
            str(_full_file(tmp_path)),
            "--limit",
            "2",
            "--unanswerable",
            "2",
            "--output-format",
            "json",
        ]
    )
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["unanswerable"] == 2
    assert result["unanswerable_filter"]["kept"] == 2
    assert "Unanswerable: 2 kept" in " ".join(out.err.split())


@pytest.mark.parametrize(
    "flags",
    [
        ["--hops", "5"],
        ["--hops", "2=1", "--limit", "1"],
        # An answerable-only file as the unanswerable source.
        ["--unanswerable", "1", "--unanswerable-data-path", "{source}"],
        ["--unanswerable-data-path", "{source}"],  # without --unanswerable
        ["--unanswerable", "1", "--unanswerable-data-path", "{missing}"],
    ],
)
def test_cli_usage_errors(tmp_path: Path, flags: list[str]) -> None:
    from unified_kg_rag.application.cli import prepare_benchmark

    answerable, _ = _pair("2hop__1", ["One.", "Two."], ["Noise."])
    source = _write(tmp_path / "ans.jsonl", [answerable])
    flags = [f.format(source=source, missing=tmp_path / "missing.jsonl") for f in flags]
    with pytest.raises(SystemExit) as exit_info:
        prepare_benchmark.main(
            [
                "musique",
                "--output-directory",
                str(tmp_path / "b"),
                "--data-path",
                str(source),
                *flags,
            ]
        )
    assert exit_info.value.code == 2


def _ans_file(full: Path, tmp_path: Path) -> Path:
    """The MuSiQue-Ans counterpart of ``full``: its answerable items with
    another distractor each (the Ans and Full files differ so)."""
    items = []
    for line in full.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["answerable"]:
            extra = {
                "idx": len(item["paragraphs"]),
                "title": "Extra",
                "paragraph_text": f"Extra {item['id']} distractor.",
                "is_supporting": False,
            }
            items.append({**item, "paragraphs": [*item["paragraphs"], extra]})
    return _write(tmp_path / "ans.jsonl", items)


def test_the_answerable_half_comes_from_the_answerable_file(tmp_path: Path) -> None:
    """Regression: with --unanswerable the answerable questions were drawn
    from the full file's answerable half, another sample of other
    paragraphs ("musique-full-answerable"), not the MuSiQue-Ans sample the
    seed draws without it."""
    full = _full_file(tmp_path)
    ans = _ans_file(full, tmp_path)
    musique.prepare(ans, tmp_path / "a", limit=3, seed=5)
    counts = musique.prepare(
        ans, tmp_path / "b", limit=3, seed=5, unanswerable=2, unanswerable_source=full
    )
    assert counts["benchmark"] == "musique"
    rows = _rows(tmp_path / "b")
    answerable = [r for r in rows if "answerable" not in r["metadata"]]
    assert answerable == _rows(tmp_path / "a")
    unanswerable = _unanswerable(rows)
    assert len(unanswerable) == 2
    # Twins of the sampled questions stay out.
    sampled = {r["query_id"] for r in answerable}
    assert not {r["query_id"].removesuffix("__unanswerable") for r in unanswerable} & (
        sampled
    )
    assert "Extra.txt" in {p.name for p in (tmp_path / "b" / "corpus").iterdir()}


def test_cli_downloads_both_splits_for_unanswerable(tmp_path: Path, mocker) -> None:
    from unified_kg_rag.application.cli import prepare_benchmark

    full = _full_file(tmp_path)
    ans = _ans_file(full, tmp_path)

    def fake_download(destination: Path, *, full: bool = False) -> Path:
        return full_file if full else ans

    full_file = full
    download = mocker.patch.object(musique, "download_dev", side_effect=fake_download)
    prepare_benchmark.main(
        [
            "musique",
            "--output-directory",
            str(tmp_path / "b"),
            "--limit",
            "2",
            "--unanswerable",
            "1",
        ]
    )
    calls = [(c.args[0].name, c.kwargs) for c in download.call_args_list]
    assert calls == [
        ("musique_ans_v1.0_dev.jsonl", {}),
        (musique.FULL_DEV_FILE, {"full": True}),
    ]
    rows = _rows(tmp_path / "b")
    assert {r["metadata"]["benchmark"] for r in rows} == {
        "musique",
        "musique-full-unanswerable",
    }
    # A full file given as --data-path supplies both halves, no download.
    download.reset_mock()
    prepare_benchmark.main(
        [
            "musique",
            "--output-directory",
            str(tmp_path / "c"),
            "--data-path",
            str(full),
            "--limit",
            "2",
            "--unanswerable",
            "1",
        ]
    )
    download.assert_not_called()
    assert "musique-full-answerable" in {
        r["metadata"]["benchmark"] for r in _rows(tmp_path / "c")
    }
