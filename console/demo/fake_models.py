# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A deterministic model server for the console demo (no model, no network).

It speaks the two Ollama endpoints the local adapters call (``/api/chat``
and ``/api/embed``), so ``providers.llm.kind: ollama`` pointed at it runs
ingestion, every search strategy and the LLM-free evaluators end to end
offline. Answers are built from the prompt itself: the extraction prompt
gets the demo corpus' entities that its passage names, the answer prompt
the context sentences that share the most words with the question. It is
for screenshots, recordings and tests of the console, not a model: real
quality needs a real model (``--models bedrock`` or ``--models ollama``).

Run it on its own::

    uv run python console/demo/fake_models.py --port 11500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# The demo corpus' entities (corpus.py), with a type each. The extractor
# finds them by name in a passage, so the graph is the same on every run.
ENTITIES: dict[str, str] = {
    "Acme Robotics": "ORGANIZATION",
    "Globex Sensors": "ORGANIZATION",
    "Umbra Freight": "ORGANIZATION",
    "Kestrel Depot": "LOCATION",
    "Orion Labs": "ORGANIZATION",
    "Alice Chen": "PERSON",
    "Ravi Patel": "PERSON",
    "lidar unit": "PRODUCT",
    "supply agreement": "CONTRACT",
    "애크미 로보틱스": "ORGANIZATION",
    "글로벡스 센서스": "ORGANIZATION",
    "엄브라 프레이트": "ORGANIZATION",
    "케스트럴 창고": "LOCATION",
    "オリオン・ラボ": "ORGANIZATION",
    "グローベックス・センサーズ": "ORGANIZATION",
}

# A relationship type per ordered pair the corpus states; others relate.
RELATION_TYPES: dict[tuple[str, str], str] = {
    ("Acme Robotics", "Globex Sensors"): "CONTRACTS_WITH",
    ("Globex Sensors", "lidar unit"): "SUPPLIES",
    ("Umbra Freight", "Kestrel Depot"): "DELIVERS_TO",
    ("Globex Sensors", "Umbra Freight"): "SHIPS_WITH",
    ("Orion Labs", "Globex Sensors"): "INSPECTS",
    ("Alice Chen", "Acme Robotics"): "WORKS_FOR",
    ("Ravi Patel", "Globex Sensors"): "WORKS_FOR",
    ("Kestrel Depot", "Acme Robotics"): "STORES_FOR",
    ("Acme Robotics", "supply agreement"): "SIGNED",
    ("애크미 로보틱스", "글로벡스 센서스"): "CONTRACTS_WITH",
    ("엄브라 프레이트", "케스트럴 창고"): "DELIVERS_TO",
    ("オリオン・ラボ", "グローベックス・センサーズ"): "INSPECTS",
}

_WORD = re.compile(r"[\w$%.,']+", re.UNICODE)
_DENSE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
# The marker that opens a numbered context section ("[3] ...").
_SECTION = re.compile(r"\[([0-9]{1,4})\] ")
# A numbered line of a report prompt's input ("- [3] A -> B (TYPE): ...").
_NUMBERED_LINE = re.compile(r"-\s*\[([0-9]+)\]\s*(.+)")
# The heading of each report in a global map prompt.
_MAP_REPORT = re.compile(r"--- Report \(id: ([0-9]+)\) ---")
# A JSON line of one string value (``"key": "value",``), as the JSON loader
# writes a record.
_JSON_VALUE = re.compile(r'^"[\w-]+": "(.*)",?$')
_STOP = frozenset(
    "a an and are as at be by did do does for from has have how in is it its "
    "of on or the to was were what when where which who whom why with".split()
)


def tokens(text: str) -> list[str]:
    """Lower-cased words, and character bigrams of Hangul, Kana and Han
    runs (which have no spaces to split on)."""
    found: list[str] = []
    for word in _WORD.findall(text.lower()):
        word = word.strip(".,'")
        if not word or word in _STOP:
            continue
        if _DENSE.search(word) and len(word) > 1:
            found.extend(word[i : i + 2] for i in range(len(word) - 1))
        else:
            found.append(word)
    return found


def embed(text: str, dimensions: int) -> list[float]:
    """A unit hashing vector of ``text``'s tokens: texts that share words
    are close, so the vector side of hybrid search ranks sensibly."""
    vector = [0.0] * dimensions
    for token in tokens(text) or ["<empty>"]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _between(text: str, start: str, end: str) -> str:
    head, sep, rest = text.partition(start)
    if not sep:
        return ""
    return rest.partition(end)[0]


def _names_in(text: str) -> list[str]:
    """The known entities ``text`` names, in the order it first names them."""
    lowered = text.lower()
    hits = [(lowered.find(n.lower()), n) for n in ENTITIES if n.lower() in lowered]
    return [name for _, name in sorted(hits)]


def _sentences(text: str) -> list[str]:
    """``text``'s sentences, without the context's "Description:" labels or
    the key and quotes of a JSON line (``"note": "..."``)."""
    parts = (
        _JSON_VALUE.sub(r"\1", s.strip().removeprefix("Description:").strip()).strip()
        for s in _SENTENCE.split(text)
    )
    return [s for s in parts if len(s) > 2]


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def extraction(passage: str) -> str:
    """The graph extraction answer: each known entity the passage names, and
    a relationship between each pair named in one sentence."""
    names = _names_in(passage)
    entities = []
    for name in names:
        described = next(
            (s for s in _sentences(passage) if name.lower() in s.lower()), name
        )
        entities.append(
            f"<entity><name>{_xml(name)}</name><type>{ENTITIES[name]}</type>"
            f"<description>{_xml(described)}</description>"
            f"<confidence>9</confidence><source_text>{_xml(name)}</source_text>"
            "</entity>"
        )
    relationships = []
    seen: set[tuple[str, str]] = set()
    for sentence in _sentences(passage):
        in_sentence = _names_in(sentence)
        for a, b in zip(in_sentence, in_sentence[1:], strict=False):
            pair = (a, b) if (a, b) in RELATION_TYPES else (b, a)
            if pair in seen or a == b:
                continue
            seen.add(pair)
            kind = RELATION_TYPES.get(pair, "RELATED_TO")
            relationships.append(
                f"<relationship><source>{_xml(pair[0])}</source>"
                f"<target>{_xml(pair[1])}</target><type>{kind}</type>"
                f"<description>{_xml(sentence)}</description><strength>8"
                f"</strength><source_text>{_xml(sentence)}</source_text>"
                "</relationship>"
            )
    return (
        f"<entities>{''.join(entities)}</entities>"
        f"<relationships>{''.join(relationships)}</relationships>"
    )


def community_report(human: str) -> str:
    """A report naming the community's entities, from their descriptions:
    a finding per numbered relationship (up to three), citing it as the
    report prompt asks (``[Data: Relationships (n)]``)."""
    data = _between(human, "<entity_data>", "</entity_data>") or human
    names = _names_in(data)[:4] or ["Demo corpus"]
    numbered = [
        (int(found.group(1)), found.group(2).strip())
        for line in _between(
            human, "<relationship_data>", "</relationship_data>"
        ).splitlines()
        if (found := _NUMBERED_LINE.match(line.strip()))
    ]
    sentences = [text for _, text in numbered]
    summary = " ".join(sentences[:2]) or f"{', '.join(names)} appear together."
    cited = [
        (text, f" [Data: Relationships ({number})]") for number, text in numbered[:3]
    ]
    findings = "".join(
        f"<finding><summary>{_xml(text[:80])}</summary>"
        f"<explanation>{_xml(text + reference)}</explanation></finding>"
        for text, reference in (cited or [(summary, "")])
    )
    return (
        f"<community_name>{_xml(' and '.join(names[:2]))} network</community_name>"
        f"<summary>{_xml(summary)}</summary><rating>6.0</rating>"
        "<rating_explanation>A small group of related parties.</rating_explanation>"
        f"<findings>{findings}</findings>"
    )


def _marked_sentences(text: str) -> list[tuple[str, int | None]]:
    """``text``'s sentences, each with the context marker of the numbered
    section it is in (``[n] `` opens a section, which runs to the next one
    or the next block header; ``None`` outside one)."""
    marked: list[tuple[str, int | None]] = []
    marker: int | None = None
    for line in text.splitlines():
        opened = _SECTION.match(line)
        if opened:
            marker, line = int(opened.group(1)), line[opened.end() :]
        elif line.startswith("#"):
            marker = None  # a block header ("##### Entities")
        marked.extend((sentence, marker) for sentence in _sentences(line))
    return marked


def best_sentences(
    question: str, text: str, limit: int
) -> list[tuple[str, int | None]]:
    """The ``limit`` sentences of ``text`` that share the most tokens with
    ``question`` (ties keep their order), skipping lines of markup, each with
    the marker of its context section."""
    wanted = set(tokens(question))
    scored = []
    for position, (sentence, marker) in enumerate(_marked_sentences(text)):
        if sentence.startswith(("<", "#", "|", "-", "[", "{", "**")):
            continue
        overlap = len(wanted & set(tokens(sentence)))
        if overlap:
            scored.append((-overlap, position, sentence, marker))
    picked = sorted(scored)[:limit]
    unique: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for _, _, sentence, marker in sorted(picked, key=lambda item: item[1]):
        if sentence not in seen:
            seen.add(sentence)
            unique.append((sentence, marker))
    return unique


def _map_report(human: str, sentence: str) -> list[str]:
    """The id of the report of a global map prompt that holds
    ``sentence``, as the map prompt asks a point to name it."""
    reports = _MAP_REPORT.split(human)
    return [
        reports[i] for i in range(1, len(reports) - 1, 2) if sentence in reports[i + 1]
    ][:1]


def _cited(sentence: str, marker: int | None) -> str:
    """``sentence`` citing its section as the answer prompt asks ([n])."""
    return sentence if marker is None else f"{sentence} [{marker}]"


def answer(human: str) -> str:
    """The answer step: the context sentences closest to the question, each
    citing the numbered section it comes from."""
    question = _between(human, 'Query: "', '"\n') or human[:200]
    context = _between(human, "<context>", "</context>")
    picked = [_cited(*item) for item in best_sentences(question, context, 3)]
    if not picked:
        return (
            "The provided context does not say. It covers the documents "
            "retrieved for this question, none of which answers it."
        )
    return picked[0] + (
        "\n\n" + "\n".join(f"- {s}" for s in picked[1:]) if picked[1:] else ""
    )


def plan(question: str) -> str:
    """The structured strategy's plan: the paths between two named
    entities, the entities one name matches, else how many entities of
    each type the graph holds."""
    names = _names_in(question)
    if len(names) >= 2:
        body: dict[str, Any] = {
            "nodes": [
                {"var": "a", "mention": names[0]},
                {"var": "z", "mention": names[1]},
            ],
            "edges": [{"direction": "both", "hops": [1, 2]}],
            "returns": {"kind": "path", "limit": 5},
        }
    elif names:
        body = {
            "nodes": [{"var": "x", "mention": names[0]}],
            "returns": {"kind": "rows"},
        }
    else:
        body = {
            "nodes": [{"var": "x"}],
            "returns": {
                "kind": "group_count",
                "group_by": {"var": "x", "prop": "type"},
                "limit": 10,
            },
        }
    return json.dumps({**body, "entities": names[:2], "keywords": tokens(question)[:5]})


def rewrite(human: str) -> str:
    """The follow-up rewrite: the latest message, naming the entities the
    conversation names when the message names none."""
    message = _between(human, "<query>\n", "\n</query>").strip()
    if _names_in(message):
        return message
    named = _names_in(
        _between(human, "<conversation_history>", "</conversation_history>")
    )
    return f"{message} ({', '.join(named)})" if named else message


def judge(human: str) -> str:
    """A pairwise verdict: the longer answer is more comprehensive, the
    shorter more direct, the rest tie (deterministic, for the demo)."""
    first = _between(human, "Answer 1:", "Answer 2:")
    second = human.partition("Answer 2:")[2]
    longer = (
        "1" if len(first) > len(second) else "2" if len(second) > len(first) else "tie"
    )
    shorter = {"1": "2", "2": "1", "tie": "tie"}[longer]
    verdict = {
        "comprehensiveness": longer,
        "diversity": "tie",
        "empowerment": "tie",
        "directness": shorter,
        "overall": "tie",
    }
    return json.dumps(
        {k: {"winner": v, "reason": "Demo verdict."} for k, v in verdict.items()}
    )


def route(question: str) -> str:
    lowered = question.lower()
    if any(
        w in lowered for w in ("overall", "themes", "summar", "main ", "전체", "全体")
    ):
        return "global"
    if len(_names_in(question)) >= 2:
        return "mix"
    return "local"


def respond(system: str, human: str) -> tuple[str, bool]:
    """The reply to one chat request, and whether it is an answer (streamed
    in small chunks when the client streams)."""
    if "knowledge graph extraction expert" in system:
        return extraction(_between(human, "<input_text>\n", "\n</input_text>")), False
    if "graph refinement specialist" in system:
        return (
            "<refinement_plan><identified_issues></identified_issues></refinement_plan>",
            False,
        )
    if "claim extraction specialist" in system:
        return "<claims></claims>", False
    if "deciding whether the provided information is useful" in system:
        return json.dumps({"reason": "It names the parties.", "rating": 4}), False
    if "community analysis" in system:
        return community_report(human), False
    if "summaries of a summary tree" in system:
        # RAPTOR's cluster summary: the passages' sentences, each once.
        passages = _between(human, "<passages>\n", "\n</passages>")
        return " ".join(dict.fromkeys(_sentences(passages)))[:600], False
    if "knowledge-graph editor" in system:
        descriptions = _between(human, "<descriptions>", "</descriptions>")
        return " ".join(dict.fromkeys(_sentences(descriptions)))[:600], False
    if "translate a question into a query plan" in system:
        return plan(_between(human, "<question>\n", "\n</question>")), False
    if "rewrite the latest user message" in system:
        return json.dumps({"question": rewrite(human)}, ensure_ascii=False), False
    if "search strategy selector" in system:
        strategy = route(_between(human, 'Query: "', '"') or human)
        return json.dumps({"strategy": strategy}), False
    if "professional technical translator" in system:
        query = _between(human, "<query>\n", "\n</query>")
        return json.dumps({"translation": query}, ensure_ascii=False), False
    if "specialized entity extraction expert" in system:
        return ", ".join(_names_in(human)) or "none", False
    if "expert keyword extractor" in system:
        names = _names_in(human)
        return (
            json.dumps(
                {
                    "high_level_keywords": ["supply chain", "delivery terms"],
                    "low_level_keywords": names or ["lidar"],
                }
            ),
            False,
        )
    if "KEY POINTS from the reports" in system:
        question = _between(human, 'User Question: "', '"')
        reports = _between(human, "<community_reports>", "</community_reports>")
        points = [
            {
                "description": s,
                "score": 80 - 10 * i,
                "report_ids": _map_report(reports, s),
            }
            for i, (s, _) in enumerate(best_sentences(question, reports, 3))
        ]
        return json.dumps({"points": points}), False
    if "DRIFT search primer" in system:
        return (
            json.dumps(
                {
                    "intermediate_answer": "See the reports.",
                    "score": 50,
                    "follow_up_queries": [],
                }
            ),
            False,
        )
    if "DRIFT action step" in system:
        return json.dumps({"response": "", "score": 0, "follow_up_queries": []}), False
    if "hypothetical answer" in system:
        return "A hypothetical report about the parties the question names.", False
    if "You control the retrieval" in system:
        return json.dumps({"sufficient": True, "known_facts": []}), False
    if "answers questions from knowledge graph context" in system or (
        "questions about a dataset by synthesizing" in system
    ):
        return answer(human), True
    if "expert judge comparing two answers" in system:
        return judge(human), False
    if "from your own knowledge" in system:
        return "I do not know that without the documents.", True
    if not system:
        # The output fixer: it cannot repair what it is shown.
        return "still not the requested structure", False
    return "", False


class FakeModels:
    """The responder plus the counters the tests read."""

    def __init__(
        self,
        dimensions: int = 64,
        chunk_delay: float = 0.03,
        request_delay: float = 0.0,
        restyle: Callable[[str, str], str] | None = None,
    ) -> None:
        self.dimensions = dimensions
        # (system prompt, reply) -> the reply as another model family writes
        # it (a reasoning block, a Python dict, prose around it): the tests
        # of how the pipeline reads other families' output.
        self.restyle = restyle
        self.chunk_delay = chunk_delay
        # Seconds every chat request waits first (a slow model, for tests
        # that stop a run part-way).
        self.request_delay = request_delay
        self.calls = 0
        self.lock = threading.Lock()

    def chat(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        messages = body.get("messages") or []
        system = "".join(
            m.get("content", "") for m in messages if m.get("role") == "system"
        )
        human = "".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        reply, streamed = respond(system, human)
        if self.restyle is not None:
            reply = self.restyle(system, reply)
        if self.request_delay:
            time.sleep(self.request_delay)
        with self.lock:
            self.calls += 1
        prompt_tokens = max(1, len(tokens(system + " " + human)))
        pieces = (
            re.findall(r"\S+\s*", reply)
            if streamed and body.get("stream", True)
            else [reply]
        )
        for piece in pieces[:-1] if len(pieces) > 1 else []:
            yield {
                "model": body.get("model"),
                "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": piece},
                "done": False,
            }
            time.sleep(self.chunk_delay)
        yield {
            "model": body.get("model"),
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": pieces[-1] if pieces else ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": prompt_tokens,
            "eval_count": max(1, len(tokens(reply))),
        }


def make_handler(models: FakeModels) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return None

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/api/version"):
                self._json(200, {"version": "0.0.0-demo"})
            elif self.path == "/api/tags":
                self._json(200, {"models": []})
            else:
                self._json(404, {"error": self.path})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/embed":
                inputs = body.get("input") or []
                inputs = inputs if isinstance(inputs, list) else [inputs]
                self._json(
                    200,
                    {
                        "model": body.get("model"),
                        "embeddings": [
                            embed(str(t), models.dimensions) for t in inputs
                        ],
                    },
                )
                return
            if self.path == "/api/show":
                self._json(200, {"model_info": {}, "capabilities": ["completion"]})
                return
            if self.path != "/api/chat":
                self._json(404, {"error": self.path})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in models.chat(body):
                line = (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
                self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    dimensions: int = 64,
    chunk_delay: float = 0.03,
    request_delay: float = 0.0,
    restyle: Callable[[str, str], str] | None = None,
) -> tuple[ThreadingHTTPServer, FakeModels]:
    """Start the server on a daemon thread; ``port`` 0 picks a free one
    (read it from ``server.server_address``)."""
    models = FakeModels(dimensions, chunk_delay, request_delay, restyle)
    server = ThreadingHTTPServer((host, port), make_handler(models))
    server.daemon_threads = True
    threading.Thread(
        target=server.serve_forever, name="fake-models", daemon=True
    ).start()
    return server, models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=11500)
    parser.add_argument("--dimensions", type=int, default=64)
    args = parser.parse_args()
    server, _ = serve(port=args.port, dimensions=args.dimensions)
    print(f"fake models on http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
