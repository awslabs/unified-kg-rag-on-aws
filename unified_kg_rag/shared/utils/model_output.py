# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading model output: one leading reasoning block, one JSON object, and
one name of a closed set.

Pure helpers (no LangChain, no provider): the adapters' chain normalizer and
JSON output parser call them, so every prompt of every strategy reads model
output by the same rules whichever model family wrote it.
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections.abc import Collection
from typing import Any, TypeVar, overload

from pydantic import BaseModel, ValidationError

from unified_kg_rag.shared.exceptions import ModelOutputCutOffError, ModelOutputError
from unified_kg_rag.shared.utils.scripts import word_spans

_M = TypeVar("_M", bound=BaseModel)

# The tags reasoning models put their reasoning in, when the server leaves
# it in the answer (an Ollama or vLLM server without a reasoning parser,
# DeepSeek-R1 and Qwen3 distills, a model told to think step by step).
REASONING_TAGS: tuple[str, ...] = ("think", "thinking", "reasoning")
_REASONING_NAMES = "|".join(REASONING_TAGS)
_LEADING_REASONING = re.compile(
    rf"\A\s*<({_REASONING_NAMES})>.*?</\1\s*>\s*", re.DOTALL | re.IGNORECASE
)
_OPENS_REASONING = re.compile(rf"\A\s*<(?:{_REASONING_NAMES})>", re.IGNORECASE)
_REASONING_OPENINGS = tuple(f"<{tag}>" for tag in REASONING_TAGS)


def leading_reasoning_end(text: str) -> int | None:
    """Where the closed reasoning block ``text`` starts with ends (the
    whitespace after it included), or None when it starts with none."""
    match = _LEADING_REASONING.match(text)
    return match.end() if match is not None else None


def strip_leading_reasoning(text: str) -> str:
    """``text`` without the reasoning block it starts with, if any.

    Only a block at the very start counts (whitespace before it aside): a
    tag named later is part of the answer. A block never closed (the output
    ran out while reasoning) leaves no answer: ``""``, which every parser
    reads as a model that returned nothing.
    """
    if (end := leading_reasoning_end(text)) is not None:
        return text[end:]
    if _OPENS_REASONING.match(text):
        return ""
    return text


# The most words a label before a name holds ("Strategy: local", "Selected
# strategy: local").
_LABEL_WORDS = 2


def named_choice(text: str, names: Collection[str]) -> str | None:
    """The one of ``names`` (casefolded words) an answer that should be a
    name alone names, or ``None``, after any leading reasoning block
    (:func:`strip_leading_reasoning`), case, markdown and punctuation
    aside: the name of every line that is a name alone ("local",
    "**Local**", "`local`", "Sure, here is my choice:\n\nlocal",
    "local\n\nReason: ...") or a label of at most :data:`_LABEL_WORDS` words, a
    colon and the name ("Strategy: local.", ``{"strategy": "local"}``),
    when they all name the same one. Lines that name others ("Fallback:
    global", an alternative listed after the answer) leave the answer to
    the JSON reader, and prose is ``None``, not guessed from, though a name
    is one of its words ("This is a simple lookup", "Not global.", "None
    of these; local is best")."""
    named = {
        name
        for line in strip_leading_reasoning(text).splitlines()
        if (name := _line_name(line, names)) is not None
    }
    return named.pop() if len(named) == 1 else None


def _line_name(line: str, names: Collection[str]) -> str | None:
    """The one of ``names`` ``line`` is (the name alone) or labels (at most
    :data:`_LABEL_WORDS` words that name none, a colon, the name), read
    NFKC-normalized: a full-width colon ("戦略：local") is a colon."""
    line = unicodedata.normalize("NFKC", line)
    spans = word_spans(line)
    if not spans or len(spans) > _LABEL_WORDS + 1:
        return None
    name = line[slice(*spans[-1])].casefold()
    if name not in names:
        return None
    if len(spans) == 1:
        return name
    label = line[spans[-2][1] : spans[-1][0]]
    if ":" in label and not {line[slice(*w)].casefold() for w in spans[:-1]} & set(
        names
    ):
        return name
    return None


def may_open_reasoning(text: str) -> bool:
    """Whether ``text`` (a streamed answer's start, leading whitespace
    included) is empty or could still grow into a reasoning block's opening
    tag: a stream must hold it until it can tell."""
    head = text.lstrip().lower()
    return any(opening.startswith(head) for opening in _REASONING_OPENINGS) or (
        _OPENS_REASONING.match(text) is not None
    )


def opens_reasoning(text: str) -> bool:
    """Whether ``text`` starts with a reasoning block's opening tag."""
    return _OPENS_REASONING.match(text) is not None


# Repairs of a span ``json.loads`` refuses, each one habit of a model
# family: comments (Llama, Mistral), a trailing comma (Llama, Gemma), an
# invalid escape such as ``\'`` or ``\_`` (MS GraphRAG's map outputs), and
# typographic double quotes (tried last: inside a string they are text).
_STRING = r'("(?:[^"\\]|\\.)*")'
_LINE_COMMENT = re.compile(_STRING + r"|//[^\n]*|#[^\n]*")
_BLOCK_COMMENT = re.compile(_STRING + r"|/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA = re.compile(_STRING + r"|,(\s*[}\]])")
_ESCAPE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})?')
_SMART_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u201e": '"'})
# A ``{`` that starts an object: a key's quote next, or the end of the text.
_OPENS_JSON = re.compile(r"\{\s*(?:[\"']|\Z)")
_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n?(.*?)```", re.DOTALL)


def _kept_string(match: re.Match[str]) -> str:
    """A string the pattern matched kept whole, anything else dropped."""
    return match.group(1) or ""


def _repaired(text: str) -> str:
    """``text`` without comments, trailing commas and invalid escapes."""
    text = _LINE_COMMENT.sub(_kept_string, _BLOCK_COMMENT.sub(_kept_string, text))
    text = _TRAILING_COMMA.sub(lambda m: m.group(1) or m.group(2), text)
    return _ESCAPE.sub(lambda m: m.group(0) if m.group(1) else "", text)


def _readings(span: str) -> list[Any]:
    """``span`` parsed each way that accepts it, first the one that changes
    least: JSON with raw line breaks in strings, JSON repaired, repaired with
    typographic quotes made ASCII, and a Python literal (``{'a': True}``,
    what a model answering in Python's ``repr`` writes)."""
    repaired = _repaired(span)
    for text in (span, repaired, _repaired(span.translate(_SMART_QUOTES))):
        try:
            return [json.loads(text, strict=False)]
        except ValueError:
            continue
    try:
        # literal_eval evaluates literals only: no names, calls or code.
        return [ast.literal_eval(span.strip())]
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return []


def _balanced_objects(text: str) -> list[str]:
    """The top-level ``{...}`` spans of ``text``, in order. Braces inside a
    string (double- or single-quoted: JSON or a Python literal) do not
    count; an unclosed last span is left out."""
    return _scan_objects(text)[0]


def _scan_objects(text: str) -> tuple[list[str], int | None]:
    """``_balanced_objects`` of ``text``, and where an unclosed last span
    starts (None when every span closes)."""
    spans: list[str] = []
    depth, quote, escaped, start = 0, "", False, -1
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif depth and char in "\"'":
            quote = char
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start : index + 1])
    return spans, (start if depth else None)


# A `{` that plausibly starts a JSON object: a key's quote, or the end of a
# response cut right after it.
_RE_OPENS_OBJECT = re.compile(r'\{\s*(?:"|\Z)')
# The message of the strict parse's error for a response cut inside its
# last object (``parse_model_object`` tells it from other parse errors).
_UNCLOSED_JSON_MESSAGE = "The last JSON object is unclosed (cut off?)"


def _strict_object(
    raw: str, *, strict: bool = False, answer_keys: Collection[str] = ()
) -> dict[str, Any]:
    """The strict reading of a JSON object in a noisy model response, the
    first ``parse_model_object`` takes.

    LLMs wrap JSON in markdown fences or surround it with prose. The span
    from the first ``{`` to the last ``}`` is parsed. When it does not parse
    because it holds several objects, as a model that corrects itself writes
    (an answer, "Correction: ... final output:", the answer again), the last
    top-level object that parses is the answer: the one the model ends on.
    With ``answer_keys``, the last one holding any of them is, so an example
    object the model writes before or after its answer is not taken for it.
    Non-object JSON yields ``{}``. Unparseable text yields ``{}`` so callers
    degrade gracefully, or raises ``json.JSONDecodeError`` when ``strict`` is
    set, for callers that must surface a broken response. Under ``strict`` a
    response whose last object is unclosed (cut off) raises too, even when
    an earlier object parses: a self-correction ends on a complete object.
    An unclosed ``{`` counts only when it starts JSON (``{"`` or a ``{`` the
    response ends on): prose after the answer may hold a lone brace.
    """
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            pass
        else:
            # A `{` after the last `}` that starts JSON (a key, or the cut)
            # opens an object left unclosed; prose may hold a lone brace.
            if not (strict and _RE_OPENS_OBJECT.search(text, end + 1)):
                return parsed if isinstance(parsed, dict) else {}
        objects, cut_at = _parsed_objects(text)
        if cut_at is not None and not _RE_OPENS_OBJECT.search(text, cut_at):
            # Lone braces in prose after the answer, no cut-off object.
            cut_at = None
        if strict and cut_at is not None:
            raise json.JSONDecodeError(_UNCLOSED_JSON_MESSAGE, text, cut_at)
        if objects:
            return next(
                (o for o in reversed(objects) if any(k in o for k in answer_keys)),
                objects[-1],
            )
    # Strip a leading/trailing markdown code fence if present.
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    # Isolate the outermost JSON object.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except ValueError:
        if strict:
            raise
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parsed_objects(text: str) -> tuple[list[dict[str, Any]], int | None]:
    """The top-level ``{...}`` spans of ``text`` (``_scan_objects``) that
    parse as JSON objects, in order, and where an unclosed last one starts.
    An object nested in one that does not parse is not top-level, so it is
    never taken for the answer, nor is one inside an object left unclosed."""
    spans, cut_at = _scan_objects(text)
    objects: list[dict[str, Any]] = []
    for span in spans:
        try:
            parsed = json.loads(span)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects, cut_at


def _objects(value: Any, answer_keys: Collection[str]) -> list[dict[str, Any]]:
    """The objects ``value`` (a parsed span) offers as the answer: itself,
    or, a list, its objects; a bare list stands for the one answer key that
    takes a list (``[...]`` for ``{"points": [...]}``)."""
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    objects = [item for item in value if isinstance(item, dict)]
    if len(answer_keys) == 1 and not any(
        key in o for o in objects for key in answer_keys
    ):
        return [{next(iter(answer_keys)): value}]
    return objects


def _candidates(raw: str, answer_keys: Collection[str]) -> list[dict[str, Any]]:
    """The objects ``raw`` holds, in order (the answer is the last).

    The strict reading (``_strict_object``) is taken when it holds an answer
    key (or none is asked for). Else the fenced blocks, the outermost
    bracketed span and the top-level ``{...}`` spans are read in that order
    (``_readings``): the first group with an object holding an answer key
    is taken, else the first with any object.

    Raises:
        ModelOutputError: The response was cut inside its last object.
    """
    try:
        strict = _strict_object(raw, strict=True, answer_keys=answer_keys)
    except json.JSONDecodeError as e:
        if e.msg == _UNCLOSED_JSON_MESSAGE:
            raise ModelOutputCutOffError(
                "the answer's last JSON object is unclosed (cut off?)"
            ) from e
        strict = {}
    if strict and (not answer_keys or any(k in strict for k in answer_keys)):
        return [strict]
    start = min((i for i in (raw.find("{"), raw.find("[")) if i != -1), default=-1)
    end = max(raw.rfind("}"), raw.rfind("]"))
    groups = (
        [m.group(1) for m in _FENCE.finditer(raw)],
        [raw[start : end + 1]] if start != -1 and end > start else [],
        _balanced_objects(raw),
    )
    fallback: list[dict[str, Any]] = [strict] if strict else []
    for spans in groups:
        found = [
            obj
            for span in spans
            for value in _readings(span)
            for obj in _objects(value, answer_keys)
        ]
        if any(key in obj for obj in found for key in answer_keys):
            return found
        fallback = fallback or found
    cut_at = _scan_objects(raw)[1]
    if not fallback and cut_at is not None and _OPENS_JSON.match(raw, cut_at):
        raise ModelOutputCutOffError("the answer's JSON object is unclosed (cut off?)")
    return fallback


@overload
def parse_model_object(
    raw: str, schema: None = None, answer_keys: Collection[str] = ()
) -> dict[str, Any]: ...


@overload
def parse_model_object(
    raw: str, schema: type[_M], answer_keys: Collection[str] = ()
) -> _M: ...


def parse_model_object(
    raw: str,
    schema: type[BaseModel] | None = None,
    answer_keys: Collection[str] = (),
) -> dict[str, Any] | BaseModel:
    """The JSON object a model answered with, read however its family
    writes one.

    Reads the strict JSON first (``_strict_object``: fences and prose
    around it, a self-correction ending on the answer, a response cut off
    inside its last object raises). Else each candidate span (a fenced
    block, the outermost braces) is repaired: comments, typographic
    quotes, trailing commas, invalid escapes and raw line breaks inside
    strings; or read as a Python literal (single quotes, ``True``,
    ``None``). A bare list is the answer of the one ``answer_keys`` entry
    when there is one. Of several objects, the last holding an answer key
    is the answer (an example object before it is not).

    With ``schema`` (a Pydantic model; its fields are the answer keys when
    none are given) the answer is the last candidate that validates,
    returned as the model; otherwise the object.

    Raises:
        ModelOutputError: No object (or none ``schema`` accepts), naming
            what was missing but never quoting the output.
    """
    if schema is not None and not answer_keys:
        answer_keys = tuple(schema.model_fields)
    candidates = _candidates(raw or "", answer_keys)
    if answer_keys:
        holding = [c for c in candidates if any(k in c for k in answer_keys)]
        candidates = holding or candidates
    if not candidates:
        raise ModelOutputError("the answer holds no JSON object")
    if schema is None:
        return candidates[-1]
    error: ValidationError | None = None
    for candidate in reversed(candidates):
        try:
            return schema.model_validate(candidate)
        except ValidationError as e:
            error = error or e
    raise ModelOutputError(
        f"the answer's JSON object is not a {schema.__name__}: "
        + "; ".join(
            f"{'.'.join(str(p) for p in item['loc']) or 'object'}: {item['msg']}"
            for item in (error.errors() if error else [])[:5]
        )
    )
