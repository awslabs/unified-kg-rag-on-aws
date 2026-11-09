# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A synthetic structural benchmark for the ``structured`` strategy
(docs/design.md §6.6, §9).

``prepare`` generates, from a seed, a made-up procurement world (vendors,
buyers, products, countries) and writes it as

- ``seed/structural.jsonl``: a seed graph of every node with its properties
  (``country``, ``founded``, ``employees``) and every ``SUPPLIES`` and
  ``MAKES`` edge;
- ``corpus/``: one text document per vendor and buyer stating part of the
  same facts and some facts only text states (each vendor's chief
  executive);
- ``structural-config.yaml``: the extraction schema and the seed graph
  entry to ingest them with, and ``evaluation.answer_span: extracted`` to
  score them with;
- ``eval_data.json``: questions in the ``run-eval`` format.

The questions come from templates: counts by relationship and by a seed
property, the top supplier and the top three, a two-hop listing, the buyer
two vendors share (a path through it), yes/no existence, a numeric maximum
and text-only controls (the cost of misrouting). Every gold answer of a
graph template is computed by the reference executor
(``domain.retrieval.graph_query.reference_execute``) over the generated
graph from the template's plan, kept as ``metadata.gold_plan``; numbers get
their digit and word forms as ``answer_aliases``. Nothing it writes is
committed: the data exists only at run time.
"""

from __future__ import annotations

import copy
import itertools
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from unified_kg_rag.domain.models.graph_query import (
    GraphEdgeRecord,
    GraphQueryPlan,
    PropertyGraph,
    QueryAnswer,
)
from unified_kg_rag.domain.retrieval.graph_query import (
    SEED_PROPERTIES_KEY,
    build_profile,
    reference_execute,
    validate_plan,
)
from unified_kg_rag.shared import EvaluationException, get_logger
from unified_kg_rag.shared.utils.files import atomic_write_text

logger = get_logger(__name__)

BENCHMARK_NAME = "structural"
SEED_GRAPH_NAME = "structural"
Size = Literal["small", "medium", "large"]

# (vendors, buyers, products, questions per template)
SIZES: dict[str, tuple[int, int, int, int]] = {
    "small": (8, 6, 6, 4),
    "medium": (40, 30, 20, 25),
    # Past search.structured.max_scan_rows (5,000): a vendor supplies up to
    # half the buyers, so a country's supply pairs outnumber the cap and a
    # seed-filtered count is a lower bound.
    "large": (400, 300, 30, 25),
}
COUNTRIES = ("Norland", "Sudmark", "Westria", "Ostavia", "Kaldera", "Brimor")
_PREFIXES = ("Al", "Bri", "Cor", "Dun", "El", "Far", "Gal", "Hol", "Is", "Jor")
_ROOTS = ("dex", "van", "mont", "ric", "sor", "tal", "wyn", "zor")
_VENDOR_KINDS = ("Supplies", "Works", "Industries", "Trading", "Components")
_BUYER_KINDS = ("Retail", "Markets", "Clinics", "Logistics", "Foods")
_MATERIALS = ("Steel", "Brass", "Copper", "Polymer", "Ceramic")
_ITEMS = ("Valve", "Pump", "Gasket", "Sensor", "Bearing", "Filter")
_GIVEN = ("Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jules")
_FAMILY = ("Lindqvist", "Moreau", "Okafor", "Tanaka", "Novak", "Silva", "Brandt")
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()

TEMPLATES = (
    "count_suppliers",
    "count_by_country",
    "top_supplier",
    "top_three",
    "two_hop_products",
    "shared_buyer",
    "supplies_yes_no",
    "max_employees",
    "text_only_ceo",
    "absent_entity",
    "multilingual",
)


@dataclass
class _Node:
    id: str
    name: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class World:
    """The generated graph: nodes by id, edges, and the text-only facts."""

    nodes: dict[str, _Node]
    edges: list[tuple[str, str, str]]
    ceo: dict[str, str]

    def of_type(self, kind: str) -> list[_Node]:
        return [n for n in self.nodes.values() if n.type == kind]

    def targets(self, source: str, kind: str) -> list[str]:
        return [t for s, k, t in self.edges if s == source and k == kind]

    def sources(self, target: str, kind: str) -> list[str]:
        return [s for s, k, t in self.edges if t == target and k == kind]

    def property_graph(self) -> PropertyGraph:
        """The world as a graph store holds it (value maps, typed edges)."""
        vertices: dict[str, dict[str, list[Any]]] = {}
        for node in self.nodes.values():
            vertex: dict[str, list[Any]] = {
                "id": [node.id],
                "name": [node.name],
                "type": [node.type],
            }
            if node.properties:
                vertex[SEED_PROPERTIES_KEY] = [json.dumps(node.properties)]
            vertices[node.id] = vertex
        edges = [
            GraphEdgeRecord(
                id=f"{source}-{kind}-{target}",
                type=kind,
                source_id=source,
                target_id=target,
            )
            for source, kind, target in self.edges
        ]
        return PropertyGraph(vertices=vertices, edges=edges)


def _names(rng: random.Random, count: int, kinds: tuple[str, ...]) -> list[str]:
    pool = [
        f"{prefix}{root} {kind}"
        for prefix, root, kind in itertools.product(_PREFIXES, _ROOTS, kinds)
    ]
    rng.shuffle(pool)
    return pool[:count]


def generate_world(seed: int = 0, size: Size = "small") -> World:
    """The procurement world of ``seed`` at ``size`` (deterministic)."""
    vendors_n, buyers_n, products_n, _ = SIZES[size]
    rng = random.Random(f"structural-{seed}-{size}")
    countries = COUNTRIES[: 4 if size == "small" else len(COUNTRIES)]
    nodes: dict[str, _Node] = {}
    for index, name in enumerate(_names(rng, vendors_n, _VENDOR_KINDS), start=1):
        nodes[f"v{index}"] = _Node(
            f"v{index}",
            name,
            "VENDOR",
            {
                "country": rng.choice(countries),
                "founded": rng.randint(1950, 2020),
                "employees": rng.randint(10, 5000),
            },
        )
    for index, name in enumerate(_names(rng, buyers_n, _BUYER_KINDS), start=1):
        nodes[f"b{index}"] = _Node(
            f"b{index}", name, "BUYER", {"country": rng.choice(countries)}
        )
    products = [f"{m} {i}" for m, i in itertools.product(_MATERIALS, _ITEMS)]
    rng.shuffle(products)
    for index, name in enumerate(products[:products_n], start=1):
        nodes[f"p{index}"] = _Node(f"p{index}", name, "PRODUCT")
    vendors = [n.id for n in nodes.values() if n.type == "VENDOR"]
    buyers = [n.id for n in nodes.values() if n.type == "BUYER"]
    product_ids = [n.id for n in nodes.values() if n.type == "PRODUCT"]
    edges: list[tuple[str, str, str]] = []
    for vendor in vendors:
        for buyer in rng.sample(buyers, rng.randint(0, max(1, len(buyers) // 2))):
            edges.append((vendor, "SUPPLIES", buyer))
        for product in rng.sample(product_ids, rng.randint(1, 3)):
            edges.append((vendor, "MAKES", product))
    ceo = {vendor: f"{rng.choice(_GIVEN)} {rng.choice(_FAMILY)}" for vendor in vendors}
    return World(nodes=nodes, edges=edges, ceo=ceo)


# --- questions -----------------------------------------------------------------------


def _number_aliases(value: float) -> list[str]:
    number = int(value) if float(value).is_integer() else value
    aliases = [str(number)]
    if isinstance(number, int) and 0 <= number < len(_NUMBER_WORDS):
        aliases.append(_NUMBER_WORDS[number])
    if isinstance(number, int) and number >= 1000:
        aliases.append(f"{number:,}")
    return aliases


@dataclass
class _Question:
    template: str
    question: str
    answer: str
    aliases: list[str] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    sources: list[str] = field(default_factory=list)
    # How answer_match reads the answer (``metadata.answer_type``: "number",
    # "set" with ``items``) and whether a decline is the right answer.
    answer_type: str | None = None
    items: list[str] = field(default_factory=list)
    # The names a set answer picks ``items`` from (``answer_candidates``).
    candidates: list[str] = field(default_factory=list)
    decline_is_correct: bool = False


class _Gold:
    """Gold answers of plans over the world's graph."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.graph = world.property_graph()
        types = {v: (m.get("type") or [""])[0] for v, m in self.graph.vertices.items()}
        self.profile = build_profile(
            self.graph.vertices.values(),
            [
                (types[e.source_id], e.type, types[e.target_id])
                for e in self.graph.edges
            ],
            sampled=False,
        )

    # Questions a template generates at most (its candidates are drawn
    # shuffled until it has them, or it has tried ten times as many).
    budget = 1_000_000

    def candidates(self, items: list[Any], rng: random.Random) -> list[Any]:
        shuffled = list(items)
        rng.shuffle(shuffled)
        return shuffled[: self.budget * 10]

    def answer(
        self, plan: dict[str, Any], linked: dict[str, list[str]] | None = None
    ) -> QueryAnswer:
        validated = validate_plan(
            GraphQueryPlan.model_validate(plan), self.profile, max_hops=4
        )
        return reference_execute(validated, self.graph, linked=linked)

    def name(self, vertex_id: str) -> str:
        return self.world.nodes[vertex_id].name


def _count_suppliers(gold: _Gold, rng: random.Random) -> list[_Question]:
    out = []
    for buyer in gold.candidates(gold.world.of_type("BUYER"), rng)[: gold.budget]:
        plan = {
            "nodes": [
                {"var": "v", "types": ["VENDOR"]},
                {"var": "b", "mention": buyer.name},
            ],
            "edges": [{"types": ["SUPPLIES"]}],
            "returns": {"kind": "count_distinct", "vars": ["v"]},
        }
        count = gold.answer(plan, {"b": [buyer.id]}).count or 0
        out.append(
            _Question(
                "count_suppliers",
                f"How many vendors supply {buyer.name}?",
                str(count),
                _number_aliases(count),
                plan,
                answer_type="number",
            )
        )
    return out


def _count_by_country(gold: _Gold, rng: random.Random) -> list[_Question]:
    out = []
    countries = sorted({n.properties["country"] for n in gold.world.of_type("VENDOR")})
    for country in countries:
        plan = {
            "nodes": [
                {
                    "var": "v",
                    "types": ["VENDOR"],
                    "where": [{"prop": "seed.country", "op": "eq", "value": country}],
                },
                {"var": "b", "types": ["BUYER"]},
            ],
            "edges": [{"types": ["SUPPLIES"]}],
            "returns": {"kind": "count_distinct", "vars": ["v"]},
        }
        count = gold.answer(plan).count or 0
        out.append(
            _Question(
                "count_by_country",
                f"How many vendors based in {country} supply at least one buyer?",
                str(count),
                _number_aliases(count),
                plan,
                answer_type="number",
            )
        )
        buyers = {
            "nodes": [
                {"var": "v", "types": ["VENDOR"]},
                {
                    "var": "b",
                    "types": ["BUYER"],
                    "where": [{"prop": "seed.country", "op": "eq", "value": country}],
                },
            ],
            "edges": [{"types": ["SUPPLIES"]}],
            "returns": {"kind": "count_distinct", "vars": ["b"]},
        }
        count = gold.answer(buyers).count or 0
        out.append(
            _Question(
                "count_by_country",
                f"How many buyers based in {country} are supplied by at least one "
                "vendor?",
                str(count),
                _number_aliases(count),
                buyers,
                answer_type="number",
            )
        )
    return out


def _ranking(
    gold: _Gold, country: str | None, top: int
) -> tuple[dict[str, Any], list[str]] | None:
    vendor: dict[str, Any] = {"var": "v", "types": ["VENDOR"]}
    if country is not None:
        vendor["where"] = [{"prop": "seed.country", "op": "eq", "value": country}]
    returns: dict[str, Any] = {
        "kind": "group_count",
        "group_by": {"var": "v"},
        "vars": ["b"],
        "limit": top + 1,
    }
    plan = {
        "nodes": [vendor, {"var": "b", "types": ["BUYER"]}],
        "edges": [{"types": ["SUPPLIES"]}],
        "returns": returns,
    }
    groups = gold.answer(plan).groups
    if len(groups) < top or (
        len(groups) > top and groups[top].count == groups[top - 1].count
    ):
        return None  # a tie at the cut: no single gold answer
    returns["limit"] = top
    return plan, [gold.name(str(g.key)) for g in groups[:top]]


def _top_supplier(gold: _Gold, rng: random.Random) -> list[_Question]:
    out = []
    countries = [
        None,
        *sorted({n.properties["country"] for n in gold.world.of_type("VENDOR")}),
    ]
    for country in countries:
        found = _ranking(gold, country, 1)
        if found is None:
            continue
        plan, names = found
        where = f" based in {country}" if country else ""
        out.append(
            _Question(
                "top_supplier",
                f"Which vendor{where} supplies the most buyers?",
                names[0],
                [],
                plan,
            )
        )
    return out


def _top_three(gold: _Gold, rng: random.Random) -> list[_Question]:
    out = []
    countries = [
        None,
        *sorted({n.properties["country"] for n in gold.world.of_type("VENDOR")}),
    ]
    for country in countries:
        found = _ranking(gold, country, 3)
        if found is None:
            continue
        plan, names = found
        where = f" based in {country}" if country else ""
        orders = [list(p) for p in itertools.permutations(names)]
        aliases = [", ".join(o) for o in orders] + [
            f"{o[0]}, {o[1]} and {o[2]}" for o in orders
        ]
        out.append(
            _Question(
                "top_three",
                f"Which three vendors{where} supply the most buyers?",
                ", ".join(names),
                aliases,
                plan,
                answer_type="set",
                items=list(names),
                # Every vendor: one of another country is a wrong pick too.
                candidates=sorted(n.name for n in gold.world.of_type("VENDOR")),
            )
        )
    return out


def _two_hop_products(gold: _Gold, rng: random.Random) -> list[_Question]:
    """The vendor that supplies a buyer and makes a product (two hops
    through it), where exactly one does."""
    out: list[_Question] = []
    pairs = gold.candidates(
        list(
            itertools.product(
                gold.world.of_type("BUYER"), gold.world.of_type("PRODUCT")
            )
        ),
        rng,
    )
    for buyer, product in pairs:
        if len(out) >= gold.budget:
            break
        plan = {
            "nodes": [
                {"var": "b", "mention": buyer.name},
                {"var": "v", "types": ["VENDOR"]},
                {"var": "p", "mention": product.name},
            ],
            "edges": [
                {"types": ["SUPPLIES"], "direction": "in"},
                {"types": ["MAKES"]},
            ],
            "returns": {"kind": "rows", "vars": ["v"], "limit": 10},
        }
        rows = gold.answer(plan, {"b": [buyer.id], "p": [product.id]}).rows
        if len(rows) != 1:
            continue  # one vendor: a single gold span
        out.append(
            _Question(
                "two_hop_products",
                f"Which vendor that supplies {buyer.name} makes {product.name}?",
                gold.name(rows[0]["v"]),
                [],
                plan,
            )
        )
    return out


def _shared_buyer(gold: _Gold, rng: random.Random) -> list[_Question]:
    out: list[_Question] = []
    vendors = gold.world.of_type("VENDOR")
    pairs = gold.candidates(list(itertools.combinations(vendors, 2)), rng)
    for first, second in pairs:
        if len(out) >= gold.budget:
            break
        plan = {
            "nodes": [
                {"var": "a", "mention": first.name},
                {"var": "z", "mention": second.name},
            ],
            "edges": [{"types": ["SUPPLIES"], "direction": "both", "hops": [2, 2]}],
            "returns": {"kind": "path", "limit": 10},
        }
        paths = gold.answer(plan, {"a": [first.id], "z": [second.id]}).paths
        if len(paths) != 1:
            continue  # one shared buyer: one path through it
        out.append(
            _Question(
                "shared_buyer",
                f"Which buyer do both {first.name} and {second.name} supply?",
                gold.name(paths[0].vertex_ids[1]),
                [],
                plan,
            )
        )
    return out


def _supplies_yes_no(gold: _Gold, rng: random.Random) -> list[_Question]:
    out = []
    pairs = gold.candidates(
        list(
            itertools.product(gold.world.of_type("VENDOR"), gold.world.of_type("BUYER"))
        ),
        rng,
    )
    for vendor, buyer in pairs[: gold.budget * 4]:
        plan = {
            "nodes": [
                {"var": "v", "mention": vendor.name},
                {"var": "b", "mention": buyer.name},
            ],
            "edges": [{"types": ["SUPPLIES"]}],
            "returns": {"kind": "exists"},
        }
        exists = bool(gold.answer(plan, {"v": [vendor.id], "b": [buyer.id]}).exists)
        out.append(
            _Question(
                "supplies_yes_no",
                f"Does {vendor.name} supply {buyer.name}?",
                "yes" if exists else "no",
                [],
                plan,
            )
        )
    # Balanced: as many yes as no.
    yes = [q for q in out if q.answer == "yes"]
    no = [q for q in out if q.answer == "no"]
    keep = min(len(yes), len(no))
    return [q for pair in zip(yes[:keep], no[:keep], strict=True) for q in pair]


def _max_employees(gold: _Gold, rng: random.Random) -> list[_Question]:
    out = []
    countries = sorted({n.properties["country"] for n in gold.world.of_type("VENDOR")})
    for country in countries:
        plan = {
            "nodes": [
                {
                    "var": "v",
                    "types": ["VENDOR"],
                    "where": [{"prop": "seed.country", "op": "eq", "value": country}],
                }
            ],
            "returns": {
                "kind": "aggregate",
                "aggregate": {"fn": "max", "var": "v", "prop": "seed.employees"},
            },
        }
        for fn, word in (("max", "largest"), ("min", "smallest")):
            plan["returns"]["aggregate"]["fn"] = fn  # type: ignore[index]
            value = gold.answer(plan).value
            if value is None:
                continue
            out.append(
                _Question(
                    "max_employees",
                    f"What is the {word} number of employees of a vendor based "
                    f"in {country}?",
                    str(int(value)),
                    _number_aliases(value),
                    copy.deepcopy(plan),  # the next fn edits this one
                    answer_type="number",
                )
            )
    return out


def _text_only_ceo(gold: _Gold, rng: random.Random) -> list[_Question]:
    vendors = gold.candidates(gold.world.of_type("VENDOR"), rng)[: gold.budget]
    return [
        _Question(
            "text_only_ceo",
            f"Who is the chief executive of {vendor.name}?",
            gold.world.ceo[vendor.id],
            [],
            None,
            [_document_name(vendor)],
        )
        for vendor in vendors
    ]


def _absent_name(world: World, rng: random.Random) -> str:
    """A vendor name no node of ``world`` has."""
    taken = {node.name for node in world.nodes.values()}
    while True:
        name = f"Zor{rng.choice(_ROOTS)}{rng.randint(10, 99)} Holdings"
        if name not in taken:
            return name


def _absent_entity(gold: _Gold, rng: random.Random) -> list[_Question]:
    """Questions about a vendor no data names: the answer is no (or none),
    and a strategy that links the name to a similar entity gets it wrong."""
    out = []
    for buyer in gold.candidates(gold.world.of_type("BUYER"), rng)[: gold.budget]:
        absent = _absent_name(gold.world, rng)
        exists = {
            "nodes": [
                {"var": "v", "mention": absent},
                {"var": "b", "mention": buyer.name},
            ],
            "edges": [{"types": ["SUPPLIES"]}],
            "returns": {"kind": "exists"},
        }
        assert gold.answer(exists, {"v": [], "b": [buyer.id]}).exists is False
        if rng.random() < 0.5:
            out.append(
                _Question(
                    "absent_entity",
                    f"Does {absent} supply {buyer.name}?",
                    "no",
                    [],
                    exists,
                    decline_is_correct=True,
                )
            )
        else:
            count = {
                **exists,
                "returns": {"kind": "count_distinct", "vars": ["b"]},
                "nodes": [
                    {"var": "v", "mention": absent},
                    {"var": "b", "types": ["BUYER"]},
                ],
            }
            out.append(
                _Question(
                    "absent_entity",
                    f"How many buyers does {absent} supply?",
                    "0",
                    ["zero", "none", "no buyers"],
                    count,
                    answer_type="number",
                    decline_is_correct=True,
                )
            )
    return out


# Korean and Japanese forms of two templates (answers as answer_match reads
# yes/no in those languages).
_LOCALIZED = {
    "ko": (
        "{buyer}에 공급하는 벤더는 몇 곳입니까?",
        "{vendor}은(는) {buyer}에 공급합니까?",
        ("예", "아니요"),
    ),
    "ja": (
        "{buyer}に供給しているベンダーは何社ですか？",
        "{vendor}は{buyer}に供給していますか？",
        ("はい", "いいえ"),
    ),
}


# Korean native numerals in their form before a counter ("세 곳"), 1-20.
_KOREAN_NATIVE = (
    "한 두 세 네 다섯 여섯 일곱 여덟 아홉 열 열한 열두 열세 열네 열다섯 열여섯 "
    "열일곱 열여덟 열아홉 스무"
).split()
_HAN_NUMERALS = "〇一二三四五六七八九"


def _han_numeral(number: int) -> str:
    """``number`` (0-99) in Han numerals: 3 三, 12 十二, 40 四十."""
    if number < 10:
        return _HAN_NUMERALS[number]
    tens, ones = divmod(number, 10)
    return (
        ("" if tens == 1 else _HAN_NUMERALS[tens])
        + "十"
        + (_HAN_NUMERALS[ones] if ones else "")
    )


def _native_count_aliases(language: str, count: int) -> list[str]:
    """A vendor count as the localized question's answer writes it: Korean
    "3곳" and "세 곳", Japanese "3社" and "三社"."""
    if language == "ko":
        native = [f"{_KOREAN_NATIVE[count - 1]} 곳"] if 1 <= count <= 20 else []
        return [f"{count}곳", *native]
    if language == "ja":
        han = [f"{_han_numeral(count)}社"] if 0 <= count < 100 else []
        return [f"{count}社", *han]
    return []


def _multilingual(gold: _Gold, rng: random.Random) -> list[_Question]:
    out = []
    counts = _count_suppliers(gold, rng)
    yes_no = _supplies_yes_no(gold, rng)
    by_name = {node.name: node for node in gold.world.nodes.values()}
    for index, language in enumerate(sorted(_LOCALIZED)):
        count_text, yes_no_text, (yes, no) = _LOCALIZED[language]
        for q in counts[index :: len(_LOCALIZED)]:
            buyer = q.question.removeprefix("How many vendors supply ").rstrip("?")
            assert buyer in by_name
            out.append(
                _Question(
                    f"multilingual_{language}",
                    count_text.format(buyer=buyer),
                    q.answer,
                    [
                        *(a for a in q.aliases if a.isdigit()),
                        *_native_count_aliases(language, int(q.answer)),
                    ],
                    q.plan,
                    answer_type="number",
                )
            )
        for q in yes_no[index :: len(_LOCALIZED)]:
            vendor, buyer = (
                q.question.removeprefix("Does ").rstrip("?").split(" supply ")
            )
            out.append(
                _Question(
                    f"multilingual_{language}",
                    yes_no_text.format(vendor=vendor, buyer=buyer),
                    yes if q.answer == "yes" else no,
                    [],
                    q.plan,
                )
            )
    rng.shuffle(out)
    return out


_GENERATORS: dict[str, Callable[[_Gold, random.Random], list[_Question]]] = {
    "count_suppliers": _count_suppliers,
    "count_by_country": _count_by_country,
    "top_supplier": _top_supplier,
    "top_three": _top_three,
    "two_hop_products": _two_hop_products,
    "shared_buyer": _shared_buyer,
    "supplies_yes_no": _supplies_yes_no,
    "max_employees": _max_employees,
    "text_only_ceo": _text_only_ceo,
    "absent_entity": _absent_entity,
    "multilingual": _multilingual,
}


def questions(world: World, seed: int = 0, size: Size = "small") -> list[_Question]:
    """The benchmark's questions over ``world``: at most the size's count
    per template, sampled by ``seed``."""
    per_template = SIZES[size][3]
    gold = _Gold(world)
    gold.budget = per_template
    found: list[_Question] = []
    for template in TEMPLATES:
        rng = random.Random(f"structural-{seed}-{size}-{template}")
        made = _GENERATORS[template](gold, rng)
        if template == "supplies_yes_no":
            # Shuffled, yes and no alternating: the first even count of them
            # stays balanced.
            made = made[: per_template - per_template % 2]
        elif len(made) > per_template:
            made = sorted(rng.sample(made, per_template), key=made.index)
        found.extend(made)
    return found


# --- writing ---------------------------------------------------------------------------


def _document_name(node: _Node) -> str:
    return f"{node.id}-{node.name.lower().replace(' ', '-')}.txt"


def _documents(world: World, rng: random.Random) -> dict[str, str]:
    """One document per vendor and buyer: each states part of the graph's
    facts (some supply edges only the seed graph holds) and the chief
    executive, which only the text states."""
    docs: dict[str, str] = {}
    for vendor in world.of_type("VENDOR"):
        made = [world.nodes[p].name for p in world.targets(vendor.id, "MAKES")]
        supplied = world.targets(vendor.id, "SUPPLIES")
        stated = [world.nodes[b].name for b in supplied if rng.random() < 0.6]
        lines = [
            f"{vendor.name} is a vendor based in {vendor.properties['country']}.",
            f"Its chief executive is {world.ceo[vendor.id]}.",
            f"{vendor.name} makes {', '.join(made)}.",
        ]
        if stated:
            lines.append(f"{vendor.name} supplies {', '.join(stated)}.")
        docs[_document_name(vendor)] = f"{vendor.name}\n\n" + " ".join(lines) + "\n"
    for buyer in world.of_type("BUYER"):
        docs[_document_name(buyer)] = (
            f"{buyer.name}\n\n{buyer.name} is a buyer based in "
            f"{buyer.properties['country']}. It buys industrial parts from "
            "several vendors.\n"
        )
    return docs


def _seed_lines(world: World) -> list[str]:
    lines = [
        json.dumps(
            {
                "kind": "node",
                "id": node.id,
                "name": node.name,
                "type": node.type,
                **({"properties": node.properties} if node.properties else {}),
            },
            ensure_ascii=False,
        )
        for node in world.nodes.values()
    ]
    lines += [
        json.dumps({"kind": "edge", "source": s, "target": t, "type": k})
        for s, k, t in world.edges
    ]
    return lines


CONFIG_TEMPLATE = """\
# The structural benchmark's ingestion settings (generated; merge them into
# your config, or pass this file with --config-path).
processing:
  graph_extraction:
    entity_types:
      - "VENDOR: A company that makes and sells industrial parts"
      - "BUYER: A company that buys industrial parts"
      - "PRODUCT: An industrial part"
    relationship_types: [SUPPLIES, MAKES]
    relationship_patterns:
      - "VENDOR -SUPPLIES-> BUYER"
      - "VENDOR -MAKES-> PRODUCT"
    schema_mode: align
  seed_graphs:
    - name: {name}
      kind: jsonl
      path: seed/structural.jsonl
      description_properties: [country]
# Exact match and token F1 on the answer's short span (one fast-model call
# per scored answer), not on the whole answer.
evaluation:
  answer_span: extracted
"""


def check_output_directory(output_dir: Path) -> None:
    """Raise unless ``output_dir``'s ``corpus/`` and ``seed/`` are new or
    empty: files of another world would be ingested with this one."""
    for part in ("corpus", "seed"):
        path = output_dir / part
        if path.exists() and any(path.iterdir()):
            raise EvaluationException(
                f"'{path}' is not empty; choose a new --output-directory"
            )


def prepare(output_dir: Path, *, seed: int = 0, size: Size = "small") -> dict[str, Any]:
    """Generate the world of ``seed`` at ``size`` and write the corpus, the
    seed graph, the config snippet and ``eval_data.json`` under
    ``output_dir`` (module docstring). Returns the counts."""
    check_output_directory(output_dir)
    world = generate_world(seed, size)
    rng = random.Random(f"structural-{seed}-{size}-documents")
    corpus, seed_dir = output_dir / "corpus", output_dir / "seed"
    corpus.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)
    for name, text in _documents(world, rng).items():
        atomic_write_text(corpus / name, text)
    atomic_write_text(
        seed_dir / "structural.jsonl", "\n".join(_seed_lines(world)) + "\n"
    )
    atomic_write_text(
        output_dir / "structural-config.yaml",
        CONFIG_TEMPLATE.format(name=SEED_GRAPH_NAME),
    )
    found = questions(world, seed, size)
    dataset = [
        {
            "query_id": f"structural-{seed}-{index:04d}",
            "question": q.question,
            "answer": q.answer,
            "answer_aliases": [a for a in q.aliases if a != q.answer],
            "reference_sources": q.sources,
            "category": q.template,
            "metadata": {
                "benchmark": BENCHMARK_NAME,
                "template": q.template,
                "language": (
                    q.template.rsplit("_", 1)[1]
                    if q.template.startswith("multilingual_")
                    else "en"
                ),
                "graph_question": q.plan is not None,
                **({"gold_plan": q.plan} if q.plan is not None else {}),
                **({"answer_type": q.answer_type} if q.answer_type else {}),
                **({"answer_items": q.items} if q.items else {}),
                **({"answer_candidates": q.candidates} if q.candidates else {}),
                **({"decline_is_correct": True} if q.decline_is_correct else {}),
            },
        }
        for index, q in enumerate(found)
    ]
    atomic_write_text(
        output_dir / "eval_data.json", json.dumps(dataset, ensure_ascii=False, indent=2)
    )

    def family(template: str) -> str:
        return "multilingual" if template.startswith("multilingual_") else template

    per_template = {
        t: sum(1 for q in found if family(q.template) == t) for t in TEMPLATES
    }
    logger.info("Structural benchmark: %s questions %s", len(found), per_template)
    return {
        "benchmark": BENCHMARK_NAME,
        "seed": seed,
        "size": size,
        "questions": len(dataset),
        "documents": len(list(corpus.iterdir())),
        "nodes": len(world.nodes),
        "edges": len(world.edges),
        "templates": per_template,
    }
