# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prepare-benchmark assertions`` (``evaluation.benchmarks.autoq``) with
scripted models: the pure steps, the whole flow, its determinism given a
seed, its provenance and its Korean / Japanese / Chinese handling."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.cli import prepare_benchmark
from unified_kg_rag.domain.models import Config
from unified_kg_rag.evaluation.benchmarks import autoq
from unified_kg_rag.evaluation.evaluation_manager import EvaluationManager
from unified_kg_rag.shared import EvaluationException

pytestmark = pytest.mark.unit

_TOPICS = ("supply terms", "storage", "freight")


def _ids(human: str) -> list[str]:
    return re.findall(r"Source (\d+) \(id \1\)", human)


def respond(system: str, human: str) -> str:
    """A scripted generator: a reply per prompt, from the prompt's own ids."""
    if "invent" in system and "personas" in system:
        return json.dumps(
            {
                "description": "Synthetic supply contracts.",
                "personas": [
                    {"persona": "Auditor", "tasks": ["check penalties"]},
                    {"persona": "Buyer", "tasks": ["compare delivery terms"]},
                ],
            }
        )
    if "Question class:" in system:
        klass = re.search(r"Question class: (\w+)", system).group(1)
        count = int(re.search(r"Write (\d+) distinct", system).group(1))
        ids = _ids(human)
        # The first source's first word names what the question is about.
        first = re.search(r"Source 1 \(id 1\):\n(\S+)", human)
        about = first.group(1) if first else "the collection"
        return json.dumps(
            {
                "questions": [
                    {
                        "question": f"[{klass}] What does {about} state, part {n}?",
                        "topic": _TOPICS[n % len(_TOPICS)],
                        "entities": ["Vendor", about],
                    }
                    for n in range(count)
                ]
                + (
                    # A duplicate is dropped.
                    [{"question": f"[{klass}] What does {about} state, part 0?"}]
                    if ids
                    else []
                )
            }
        )
    if "factual claims that help answer" in system:
        ids = _ids(human)
        return json.dumps(
            {
                "claims": [
                    {"statement": f"Claim from source {i}.", "sources": [i], "score": 7}
                    for i in ids[:3]
                ]
                + [
                    # An id the prompt never gave, and a zero score: dropped.
                    {"statement": "Invented.", "sources": ["99"], "score": 9},
                    {"statement": "Irrelevant.", "sources": [ids[0]], "score": 0},
                ]
            }
        )
    if "assertions for grading" in system:
        claims = re.findall(r"ID: (claim_\d+)", human)
        return json.dumps(
            {
                "assertions": [
                    {"statement": f"Assertion on {c}.", "sources": [c], "score": 8}
                    for c in claims[:4]
                ]
                + [
                    {"statement": "Hallucinated.", "sources": ["claim_99"], "score": 9},
                    {"statement": "Bad score.", "sources": [claims[0]], "score": 0},
                ]
            }
        )
    if "check assertions written" in system:
        numbers = re.findall(r"^(\d+)\. ", human, flags=re.MULTILINE)
        return json.dumps(
            {
                "scores": [
                    {
                        "id": int(n),
                        # The second assertion is not grounded: dropped.
                        "grounding": 2 if n == "2" else 5,
                        "relevance": 4,
                        "verifiability": 4,
                    }
                    for n in numbers
                ]
            }
        )
    raise AssertionError(f"unexpected prompt: {system[:80]}")


def _corpus(root: Path) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    for name, words in {
        "contract.txt": "Vendor supplies widgets to Buyer under a fixed price.",
        "storage.md": "Depot stores gadgets for Vendor in a bonded warehouse.",
        "freight.txt": "Carrier moves gadgets for Depot every week by rail.",
        "penalty.txt": "Buyer pays a penalty of 100 credits per late day.",
    }.items():
        (corpus / name).write_text(
            "\n\n".join(f"{words} Clause {n}." for n in range(6)), encoding="utf-8"
        )
    (corpus / ".hidden.txt").write_text("never read", encoding="utf-8")
    return corpus


def _providers(config: Config, responder=respond) -> Providers:
    return Providers(
        config,
        llm_factory=ScriptedLLMFactory(responder),
        embedding_factory=HashingEmbeddingFactory(),
    )


def _settings(**overrides) -> autoq.GenerationSettings:
    return autoq.GenerationSettings(
        **{
            "questions_per_class": 3,
            "clusters": 3,
            "units_per_cluster": 3,
            "unit_tokens": 30,
            "seed": 7,
            **overrides,
        }
    )


def _generate(corpus: Path, settings=None, responder=respond) -> dict:
    config = Config()
    generator = autoq.AssertionBenchmarkGenerator(
        config, settings or _settings(), providers=_providers(config, responder)
    )
    return asyncio.run(generator.agenerate(corpus))


class TestTextUnits:
    def test_paragraphs_are_packed_and_offsets_point_into_the_file(self) -> None:
        text = "Alpha one.\n\nBeta two.\n\nGamma three is longer than the rest."
        units = autoq.split_units("a.txt", text, max_tokens=9)
        assert [u.id for u in units] == ["a.txt#1", "a.txt#2"]
        for unit in units:
            assert text[unit.start : unit.end].strip() == unit.text
        assert units[0].text == "Alpha one.\n\nBeta two."

    @pytest.mark.parametrize(
        "text",
        [
            "벤더는 구매자에게 위젯을 공급한다" * 20,  # Korean, no blank line
            "ベンダーは買い手にウィジェットを供給する" * 20,  # Japanese
            "供应商向买方供应小部件并按固定价格结算" * 20,  # Chinese
        ],
        ids=["ko", "ja", "zh"],
    )
    def test_a_dense_script_paragraph_is_cut_by_the_token_estimate(
        self, text: str
    ) -> None:
        # Whitespace words would see one or two words and never cut.
        units = autoq.split_units("doc.txt", text, max_tokens=50)
        assert len(units) > 1
        assert all(estimate_token_count(u.text) <= 50 for u in units)
        # Only the whitespace at a cut is dropped.
        assert "".join(u.text for u in units).replace(" ", "") == text.replace(" ", "")

    def test_the_corpus_skips_hidden_files_and_other_suffixes(
        self, tmp_path: Path
    ) -> None:
        corpus = _corpus(tmp_path)
        (corpus / "image.png").write_bytes(b"\x89PNG")
        units = autoq.load_corpus(corpus, unit_tokens=30)
        assert {u.file for u in units} == {
            "contract.txt",
            "freight.txt",
            "penalty.txt",
            "storage.md",
        }
        with pytest.raises(EvaluationException, match="no .txt or .md"):
            autoq.load_corpus(_empty(tmp_path), 30)

    def test_sampling_is_seeded_and_keeps_corpus_order(self, tmp_path: Path) -> None:
        units = autoq.load_corpus(_corpus(tmp_path), unit_tokens=12)
        one = autoq.sample_units(units, 5, seed=1)
        assert one == autoq.sample_units(units, 5, seed=1)
        assert one != autoq.sample_units(units, 5, seed=2)
        assert [units.index(u) for u in one] == sorted(units.index(u) for u in one)


def _empty(root: Path) -> Path:
    empty = root / "empty"
    empty.mkdir()
    return empty


class TestClustering:
    def test_kmeans_recovers_separated_groups_and_is_seeded(self) -> None:
        rng = np.random.default_rng(0)
        vectors = np.vstack(
            [rng.normal(center, 0.05, size=(10, 2)) for center in (0, 5, 10)]
        )
        labels = autoq.kmeans(vectors, 3, seed=3)
        groups = {tuple(labels[i : i + 10]) for i in (0, 10, 20)}
        assert all(len(set(g)) == 1 for g in groups) and len(groups) == 3
        assert np.array_equal(labels, autoq.kmeans(vectors, 3, seed=3))

    def test_cluster_samples_take_the_centroid_member_and_its_neighbours(
        self,
    ) -> None:
        angle = np.deg2rad
        vectors = np.asarray(
            [[np.cos(angle(a)), np.sin(angle(a))] for a in (0, 10, 20, 85, 90, 45, 30)]
        )
        labels = np.asarray([0, 0, 0, 1, 1, 2, 0])
        samples = autoq.cluster_samples(vectors, labels, per_cluster=2)
        # Largest cluster first; its centroid member is 10 or 20 degrees, and
        # its neighbour the nearest untaken unit by angle, from any cluster.
        assert [label for label, _ in samples] == [0, 1, 2]
        assert samples[0][1][0] in (1, 2)
        taken = [i for _, idx in samples for i in idx]
        assert len(taken) == len(set(taken)) == 6
        assert autoq.cluster_samples(vectors, labels, per_cluster=1) == [
            (label, idx[:1]) for label, idx in samples
        ]


class TestParsing:
    def test_claims_keep_only_scored_claims_citing_given_units(self) -> None:
        output = {
            "claims": [
                {"statement": "A.", "sources": [1, "2"], "score": 5},
                {"statement": "B.", "sources": ["9"], "score": 5},
                {"statement": "", "sources": ["1"], "score": 5},
                {"statement": "C.", "sources": ["1"], "score": 0},
                {"statement": "D.", "sources": ["1"], "score": "7"},
            ]
        }
        claims = autoq.parse_claims(output, {"1": "f#1", "2": "f#2"})
        assert [(c.statement, c.sources) for c in claims] == [("A.", ["f#1", "f#2"])]

    def test_assertions_map_claims_to_units_and_drop_hallucinated_ones(
        self,
    ) -> None:
        claims = [
            autoq.Claim(statement="x", sources=["f#1"], score=5),
            autoq.Claim(statement="y", sources=["f#2", "f#1"], score=5),
        ]
        output = {
            "assertions": [
                {"statement": "S1", "sources": ["claim_1", "claim_2"], "score": 9},
                {"statement": "S2", "sources": ["claim_7"], "score": 9},
                {"statement": "S3", "sources": ["claim_1"]},  # score 5
                {"statement": "S4", "sources": [], "score": 9},
                {"statement": "S5", "sources": ["claim_1"], "score": 11},
            ]
        }
        parsed = autoq.parse_assertions(output, claims)
        assert [(a.statement, a.sources, a.score) for a in parsed] == [
            ("S1", ["f#1", "f#2"], 9),
            ("S3", ["f#1"], 5),
        ]

    def test_ranking_fuses_importance_and_source_count(self) -> None:
        make = autoq.GeneratedAssertion
        assertions = [
            make(statement="few sources, top score", sources=["a"], score=10),
            make(statement="many sources", sources=["a", "b", "c"], score=6),
            make(statement="middle", sources=["a", "b"], score=8),
        ]
        ranked = autoq.rank_assertions(assertions, limit=2)
        # Source count weighs 0.7: the three-source assertion first.
        assert [a.statement for a in ranked] == ["many sources", "middle"]
        assert autoq.competition_ranks([10, 8, 8, 6]) == [1, 2, 2, 4]

    def test_validation_needs_every_score_and_a_reply_for_each_assertion(
        self,
    ) -> None:
        output = {
            "scores": [
                {"id": 1, "grounding": 5, "relevance": 3, "verifiability": 9},
                {"id": 2, "grounding": 2, "relevance": 5, "verifiability": 5},
                {"id": 3, "grounding": "high", "relevance": 5, "verifiability": 5},
            ]
        }
        assert autoq.parse_validation(output, 4, 3) == [True, False, False, False]
        assert autoq.parse_validation("not json", 2, 3) == [False, False]


class TestGeneration:
    def test_it_writes_a_dataset_run_eval_loads_with_provenance(
        self, tmp_path: Path
    ) -> None:
        generated = _generate(_corpus(tmp_path))
        rows = generated["rows"]
        classes = {r["category"] for r in rows}
        assert classes == set(autoq.QUESTION_CLASSES)
        units = {u.id: u for u in generated["units"]}
        for row in rows:
            assert row["metadata"]["benchmark"] == autoq.BENCHMARK_NAME
            if row["category"] in autoq.DATA_CLASSES:
                assert row["assertions"], row
                for assertion in row["assertions"]:
                    # Every assertion cites text units written for review.
                    assert assertion["kind"] == "positive"
                    assert assertion["source"]
                    assert set(assertion["source"]) <= set(units)
                    assert "Hallucinated" not in assertion["text"]
                assert set(row["reference_sources"]) == {
                    Path(units[u].file).name
                    for u in row["metadata"]["source_text_units"]
                }
            else:
                assert "assertions" not in row
                assert row["metadata"]["persona"]
        # The ungrounded second assertion of each question was dropped.
        report = generated["report"]
        assert report["dropped"]["data_local_assertions_invalid"] > 0
        assert report["seed"] == 7
        out = tmp_path / "out"
        autoq.write_outputs(out, generated)
        queries, truths = EvaluationManager.load_data(out / "eval_data.json")
        assert len(queries) == len(rows)
        assert sum(len(t.assertions) for t in truths) == report["assertions"]
        lines = (out / "text_units.jsonl").read_text("utf-8").splitlines()
        unit = json.loads(lines[0])
        source = (tmp_path / "corpus" / unit["file"]).read_text("utf-8")
        assert source[unit["start"] : unit["end"]].strip() == unit["text"]
        with pytest.raises(EvaluationException, match="already holds"):
            autoq.check_output_directory(out)

    def test_the_same_seed_and_replies_write_the_same_files(
        self, tmp_path: Path
    ) -> None:
        corpus = _corpus(tmp_path)
        settings = _settings(max_units=10)
        one, two = _generate(corpus, settings), _generate(corpus, settings)
        assert one["rows"] == two["rows"]
        assert one["units"] == two["units"]
        other = _generate(corpus, _settings(max_units=10, seed=8))
        assert other["rows"] != one["rows"]

    def test_a_failed_call_drops_its_item_and_is_counted(self, tmp_path: Path) -> None:
        def flaky(system: str, human: str) -> str:
            if "factual claims" in system and "Clause 0" in human:
                raise RuntimeError("throttled")
            return respond(system, human)

        generated = _generate(
            _corpus(tmp_path), _settings(classes=["data_local"]), flaky
        )
        assert generated["report"]["dropped"].get("claims_call_failed", 0) > 0
        assert {r["category"] for r in generated["rows"]} <= {"data_local"}

    @pytest.mark.parametrize(
        ("code", "name"), [("ko", "Korean"), ("ja", "Japanese"), ("zh", "Chinese")]
    )
    def test_every_prompt_names_the_language(
        self, tmp_path: Path, code: str, name: str
    ) -> None:
        seen: list[str] = []

        def recording(system: str, human: str) -> str:
            seen.append(system)
            return respond(system, human)

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.txt").write_text(
            "\n\n".join(
                [
                    "벤더는 구매자에게 위젯을 공급한다.",
                    "倉庫は商品を保管する。",
                    "承运人每周运输货物。",
                ]
                * 4
            ),
            encoding="utf-8",
        )
        _generate(
            corpus,
            _settings(language=code, classes=["data_local", "activity_global"]),
            recording,
        )
        assert seen and all(name in system for system in seen)

    def test_an_unknown_class_is_refused(self) -> None:
        with pytest.raises(EvaluationException, match="unknown question class"):
            autoq.AssertionBenchmarkGenerator(
                Config(), _settings(classes=["data_everything"]), providers=None
            )


def test_the_cli_writes_the_files_and_refuses_a_used_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(
        "unified_kg_rag.application.cli.assertion_benchmark.build_providers",
        lambda cfg: _providers(cfg),
    )
    out = tmp_path / "bench"
    argv = [
        "assertions",
        "--source-directory",
        str(corpus),
        "--output-directory",
        str(out),
        "--questions-per-class",
        "2",
        "--clusters",
        "2",
        "--classes",
        "data_local,data_linked",
        "--unit-tokens",
        "50",
        "--output-format",
        "json",
    ]
    prepare_benchmark.main(argv)
    printed = json.loads(capsys.readouterr().out)
    assert printed["questions"]
    assert {p.name for p in out.iterdir()} == set(autoq.OUTPUT_FILES)
    with pytest.raises(SystemExit) as exit_info:
        prepare_benchmark.main(argv)
    assert exit_info.value.code == 2


def _cli_argv(corpus: Path, out: Path) -> list[str]:
    return [
        "assertions",
        "--source-directory",
        str(corpus),
        "--output-directory",
        str(out),
        "--questions-per-class",
        "2",
        "--clusters",
        "2",
        "--classes",
        "data_local",
        "--unit-tokens",
        "50",
        "--output-format",
        "json",
    ]


def test_the_cli_fails_when_no_question_was_generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # Regression (rv23 surfaces 3): every call failing printed "Generated 0
    # questions", exited 0 and wrote an empty dataset run-eval refused.
    def failing(system: str, human: str) -> str:
        raise RuntimeError("AccessDeniedException")

    monkeypatch.setattr(
        "unified_kg_rag.application.cli.assertion_benchmark.build_providers",
        lambda cfg: _providers(cfg, failing),
    )
    out = tmp_path / "bench"
    with pytest.raises(SystemExit) as exit_info:
        prepare_benchmark.main(_cli_argv(_corpus(tmp_path), out))
    assert exit_info.value.code == 1
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "failed" and status["exit_code"] == 1
    assert "generated no question" in status["error"]
    assert "data_local_questions_call_failed 2" in status["error"]
    # Nothing written: the directory takes the next run.
    assert not out.exists() or not any(out.iterdir())


def test_an_unreachable_model_stops_generation_naming_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # The client's raw "Failed to connect to Ollama" was counted per call;
    # it is the ModelUnreachableError run-rag gives, at the first call.
    calls: list[str] = []

    def unreachable(system: str, human: str) -> str:
        calls.append(system)
        raise ConnectionError("Failed to connect to Ollama.")

    monkeypatch.setattr(
        "unified_kg_rag.application.cli.assertion_benchmark.build_providers",
        lambda cfg: _providers(cfg, unreachable),
    )
    with pytest.raises(SystemExit) as exit_info:
        prepare_benchmark.main(_cli_argv(_corpus(tmp_path), tmp_path / "bench"))
    assert exit_info.value.code == 1
    status = json.loads(capsys.readouterr().out)
    assert status["error_type"] == "ModelUnreachableError"
    assert "A model server could not be reached" in status["error"]
