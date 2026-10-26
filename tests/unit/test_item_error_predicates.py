# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which batch-item errors count against a document, per provider.

``indexing.max_document_failures`` excludes a document after that many
counted failures, so only errors the item causes count: output that does not
parse or was cut off, and a request the model rejects for its content. Any
other error (an outage, credentials, a quota, a setting, a bug) fails every
document alike and counts for none. The errors are real client exception
types built without a network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from botocore import exceptions as boto_exc
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import Generation

from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.ingestion import claim_extractor as claim_module
from unified_kg_rag.adapters.ingestion import gleaner as gleaner_module
from unified_kg_rag.adapters.ingestion import graph_extractor as graph_module
from unified_kg_rag.adapters.llm.response_cache import LLMResponseCache
from unified_kg_rag.adapters.llm.retry import (
    item_error_predicate,
    item_failed_indices,
    known_good_unit,
)
from unified_kg_rag.adapters.local.models import (
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
)
from unified_kg_rag.domain.models import Config, TextUnit, TextUnitExtraction
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError
from unified_kg_rag.shared.utils.langchain import BatchProcessor

openai = pytest.importorskip("openai")
ollama = pytest.importorskip("ollama")

pytestmark = pytest.mark.unit

_REQUEST = httpx.Request("POST", "http://model-server.test/v1/chat/completions")


def _openai(cls: type[Exception], status: int, message: str) -> Exception:
    return cls(message, response=httpx.Response(status, request=_REQUEST), body=None)


def _bedrock(code: str, status: int, message: str = "x") -> boto_exc.ClientError:
    return boto_exc.ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "InvokeModel",
    )


_OUTPUT_ERRORS = [OutputParserException("bad"), LLMOutputTruncatedError("cut")]
_NOT_ITEM_ERRORS: list[BaseException] = [
    TimeoutError(),
    RuntimeError("model down"),
    TypeError("unexpected keyword argument"),
    KeyError("x"),
    KeyboardInterrupt(),
    boto_exc.NoCredentialsError(),
    boto_exc.NoRegionError(),
    boto_exc.TokenRetrievalError(provider="sso", error_msg="expired"),
    boto_exc.CredentialRetrievalError(provider="container-role", error_msg="x"),
    boto_exc.ParamValidationError(report="bad"),
]

_BEDROCK_ITEM = [
    _bedrock("ValidationException", 400, "Input is too long for requested model."),
    _bedrock(
        "ValidationException",
        400,
        "The model returned the following errors: prompt is too long: "
        "210000 tokens > 200000 maximum",
    ),
    _bedrock(
        "ValidationException",
        400,
        "input length and `max_tokens` exceed context limit: 190000 + 20000 "
        "> 200000, decrease input length or `max_tokens` and try again",
    ),
]
_BEDROCK_NOT_ITEM = [
    _bedrock("ServiceQuotaExceededException", 400, "exceeds the service quota"),
    _bedrock("ModelStreamErrorException", 424),
    _bedrock("ThrottlingException", 429),
    _bedrock("AccessDeniedException", 403),
    _bedrock("ExpiredTokenException", 403),
    _bedrock("ResourceNotFoundException", 404),
    _bedrock("ValidationException", 400, "The provided model identifier is invalid."),
    _bedrock(
        "ValidationException",
        400,
        "Invocation of model ID x with on-demand throughput isn't supported. "
        "Retry your request with the ID or ARN of an inference profile",
    ),
    _bedrock("ValidationException", 400, "temperature: Extra inputs are not permitted"),
    _bedrock("ValidationException", 400, "This model doesn't support tool use."),
]

_LOCAL_ITEM = [
    _openai(
        openai.BadRequestError,
        400,
        "This model's maximum context length is 8192 tokens. However, your "
        "messages resulted in 9000 tokens",
    ),
    _openai(openai.UnprocessableEntityError, 422, "bad input"),
    ollama.ResponseError("input rejected", 400),
]
_LOCAL_NOT_ITEM = [
    _openai(openai.BadRequestError, 400, "Invalid model name"),
    _openai(openai.BadRequestError, 400, "max_tokens is too large"),
    _openai(openai.BadRequestError, 400, "Unsupported parameter: 'top_k'"),
    _openai(openai.NotFoundError, 404, "no such model"),
    _openai(openai.AuthenticationError, 401, "bad key"),
    _openai(openai.RateLimitError, 429, "slow down"),
    _openai(openai.InternalServerError, 500, "crashed"),
    ollama.ResponseError("boom", -1),
    httpx.ConnectError("refused", request=_REQUEST),
]


# One message per phrase of the two configuration patterns, each matching
# that phrase alone: a setting the request names, not the item's text.
_BEDROCK_CONFIGURATION_MESSAGES = [
    "The provided model identifier is invalid.",
    "Invocation of model ID vendor.model-v1 with on-demand throughput is unavailable.",
    "Retry your request with the ID or ARN of an inference profile.",
    "temperature: Extra inputs are not permitted",
    "The requested operation is not supported for this model.",
    "Streaming isn't supported for this model.",
    "Unsupported parameter: top_k",
    "This model doesn't support tool use.",
    "Malformed input request: #/messages: expected type: JSONArray",
    "The guardrail identifier or version provided in the request does not exist.",
    "This model version has reached the end of its life.",
    "This model version is marked Legacy by the provider.",
    "You don't have access to the model with the specified model ID.",
    "The maximum tokens you requested exceeds the model limit of 4096.",
    "max_tokens: 64001 > 64000, which is the maximum allowed",
    "temperature and top_p cannot both be specified for this model.",
    "#: extraneous key [top_k]",
    "Invocation of model ID vendor.model-v1 with on-demand throughput isn't "
    "supported.",
]
_LOCAL_CONFIGURATION_MESSAGES = [
    "Invalid model name",
    "model 'llama3:70b' not found, try pulling it first",
    "{'code': 'model_not_found'}",
    "logprobs are not supported with this backend",
    "Unsupported parameter: 'top_k'",
    "This model does not support image input",
    "setting seed is not permitted on this server",
    "unknown field: top_k",
    "Unrecognized request argument supplied: functions",
    "1 validation error: extra fields not allowed: 'seed'",
    "max_tokens is too large: 9000",
    "Invalid API key provided",
    "You exceeded your current quota",
    "max_tokens: 9000 > 4096",
    "temperature and top_p cannot both be specified",
    "#: extraneous key [top_k]",
]


def _ids(errors: list[BaseException]) -> list[str]:
    return [f"{type(e).__name__}:{str(e)[:30]}" for e in errors]


@pytest.mark.parametrize(
    "error",
    [*_OUTPUT_ERRORS, *_BEDROCK_ITEM],
    ids=_ids([*_OUTPUT_ERRORS, *_BEDROCK_ITEM]),
)
def test_bedrock_counts_its_item_errors(error: BaseException) -> None:
    factory = BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)
    assert item_error_predicate(factory)(error)


@pytest.mark.parametrize(
    "error",
    [*_NOT_ITEM_ERRORS, *_BEDROCK_NOT_ITEM],
    ids=_ids([*_NOT_ITEM_ERRORS, *_BEDROCK_NOT_ITEM]),
)
def test_bedrock_counts_no_other_error(error: BaseException) -> None:
    factory = BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)
    assert not item_error_predicate(factory)(error)


@pytest.mark.parametrize(
    "factory_cls", [OllamaLanguageModelFactory, OpenAICompatibleLanguageModelFactory]
)
def test_local_factories_count_their_item_errors_only(factory_cls: Any) -> None:
    predicate = item_error_predicate(factory_cls(Config()))
    assert all(predicate(e) for e in [*_OUTPUT_ERRORS, *_LOCAL_ITEM])
    counted = [e for e in [*_NOT_ITEM_ERRORS, *_LOCAL_NOT_ITEM] if predicate(e)]
    assert counted == []


@pytest.mark.parametrize("message", _BEDROCK_CONFIGURATION_MESSAGES)
def test_bedrock_counts_no_validation_error_of_a_setting(message: str) -> None:
    factory = BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)
    predicate = item_error_predicate(factory)
    assert predicate(_bedrock("ValidationException", 400, "Input is too long."))
    assert not predicate(_bedrock("ValidationException", 400, message))


@pytest.mark.parametrize("message", _LOCAL_CONFIGURATION_MESSAGES)
@pytest.mark.parametrize(
    "factory_cls", [OllamaLanguageModelFactory, OpenAICompatibleLanguageModelFactory]
)
def test_local_factories_count_no_rejected_setting(
    factory_cls: Any, message: str
) -> None:
    predicate = item_error_predicate(factory_cls(Config()))
    assert predicate(_openai(openai.BadRequestError, 400, "input rejected"))
    assert not predicate(_openai(openai.BadRequestError, 400, message))
    assert not predicate(ollama.ResponseError(message, 400))


@pytest.mark.parametrize(
    ("status", "counted"), [(400, True), (413, True), (500, False)]
)
def test_local_factories_read_the_status_of_the_response(
    status: int, counted: bool
) -> None:
    # httpx.HTTPStatusError carries no status_code of its own: the status is
    # read from its response, as for the transient classification.
    error = httpx.HTTPStatusError(
        "rejected", request=_REQUEST, response=httpx.Response(status, request=_REQUEST)
    )
    assert not hasattr(error, "status_code")
    predicate = item_error_predicate(OpenAICompatibleLanguageModelFactory(Config()))
    assert predicate(error) is counted


def test_a_factory_without_a_classifier_counts_http_item_rejections() -> None:
    # A custom factory without is_item_error: its client's HTTP 400/413/422
    # count (read from status_code), not only unparseable output; a document
    # it always rejects is excluded rather than retried forever.
    predicate = item_error_predicate(object())
    assert all(predicate(e) for e in [*_OUTPUT_ERRORS, *_LOCAL_ITEM])
    counted = [
        e for e in [*_NOT_ITEM_ERRORS, *_LOCAL_NOT_ITEM, *_BEDROCK_ITEM] if predicate(e)
    ]
    assert counted == []


_ITEM = _BEDROCK_ITEM[0]
_BEDROCK = BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)


class _Probe:
    """A batch processor whose known-good request is ``answered`` (or not);
    ``sent`` lists the ``parsed`` flag of each request made."""

    def __init__(self, answered: bool = False, call: Any = None) -> None:
        self.result = answered
        self.call = call
        self.sent: list[bool] = []

    def answered(
        self, item: TextUnit, prepare: Any, invoke: Any, task: str, *, parsed: bool
    ) -> bool:
        assert item == known_good_unit() and task == "t (known-good request)"
        self.sent.append(parsed)
        if self.call is not None:
            self.call()
        return self.result


def _indices(
    errors: dict[int, BaseException],
    count: int,
    predicate: Any,
    probe: _Probe,
    documents: int = 1,
) -> list[int]:
    """``item_failed_indices`` over ``count`` units, spread over
    ``documents`` documents in turn."""
    units = [
        TextUnit(id=f"u{i}", text="t", document_ids=[f"d{i % documents}"])
        for i in range(count)
    ]
    return item_failed_indices(
        errors, units, predicate, "t", probe, lambda units: [{}], lambda _: {}  # type: ignore[arg-type]
    )


_PARSE, _CUT = _OUTPUT_ERRORS


@pytest.mark.parametrize("answered", [True, False])
@pytest.mark.parametrize(
    ("errors", "alike"),
    [
        # Every unit, one item error: the known-good request decides.
        ({0: _ITEM, 1: _ITEM}, True),
        ({0: _ITEM}, True),
        # Alike in type, code and status, not in message: one cause too (a
        # message can carry each request's own figures).
        ({0: _ITEM, 1: _bedrock("ValidationException", 400, "y")}, True),
        # Output that does not parse or was cut off on every unit: a
        # server-side output cap, say, or each unit's own.
        ({0: _PARSE, 1: _PARSE}, True),
        ({0: _CUT, 1: _CUT}, True),
        ({0: _PARSE, 1: _CUT}, True),
        # Not every unit failed, or not of one cause.
        ({0: _ITEM, 2: _ITEM}, False),
        ({0: _ITEM, 1: _PARSE}, False),
        # A unit timed out (or was throttled) and the others failed of one
        # cause: the timeout says nothing of what the model did to the
        # others, which the known-good request decides; it counts for none.
        ({0: TimeoutError(), 1: _ITEM}, True),
        ({0: TimeoutError(), 1: _ITEM, 2: _ITEM}, True),
        ({0: TimeoutError(), 1: _PARSE, 2: _CUT}, True),
        ({0: TimeoutError(), 1: _ITEM, 2: _PARSE}, False),
        # No error of an item's own: none counts, nothing is sent.
        ({0: TimeoutError(), 1: TimeoutError()}, False),
    ],
)
def test_one_cause_on_every_unit_counts_as_the_known_good_request_decides(
    errors: dict[int, BaseException], alike: bool, answered: bool
) -> None:
    predicate = item_error_predicate(_BEDROCK)
    count = 1 if len(errors) == 1 else 3 if 2 in errors else 2
    probe = _Probe(answered)
    counted = _indices(errors, count, predicate, probe)
    own = [idx for idx, error in sorted(errors.items()) if predicate(error)]
    assert counted == ([] if alike and not answered else own)
    # Sent only when every unit failed and its own errors are of one cause,
    # a single unit's failure too; for output errors its answer must parse.
    output = all(isinstance(errors[idx], type(_PARSE) | type(_CUT)) for idx in own)
    assert probe.sent == ([output] if alike else [])


@pytest.mark.parametrize("answered", [True, False])
@pytest.mark.parametrize(
    "errors",
    [
        {0: _PARSE, 1: _PARSE},
        {0: _CUT, 1: _CUT},
        {0: _PARSE, 1: _CUT},
        {0: _CUT, 1: _CUT, 2: TimeoutError()},
    ],
)
def test_one_output_cause_on_every_unit_of_several_documents_counts_none(
    errors: dict[int, BaseException], answered: bool
) -> None:
    # A server-side output cap cuts every document's output, but not a
    # known-good request's short answer: that answer cannot tell the cap
    # from each document's own output, so output errors across documents
    # count for none, and nothing is sent.
    probe = _Probe(answered)
    counted = _indices(errors, len(errors), item_error_predicate(_BEDROCK), probe, 2)
    assert (counted, probe.sent) == ([], [])


def test_one_output_cause_on_a_documents_units_and_its_timeout_asks() -> None:
    # The own errors are of one document (the other's unit timed out): the
    # known-good request still decides.
    probe = _Probe(answered=True)
    errors = {0: _CUT, 1: TimeoutError(), 2: _CUT}
    assert _indices(errors, 3, item_error_predicate(_BEDROCK), probe, 2) == [0, 2]
    assert probe.sent == [True]


class _Coded(Exception):
    def __init__(self, message: str, code: str = "A", status: int = 400) -> None:
        super().__init__(message)
        self.response = {"Error": {"Code": code}}
        self.status_code = status


class _OtherType(_Coded):
    pass


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (_Coded("bad", "A"), _Coded("bad", "B")),
        (_Coded("bad", "A"), _OtherType("bad", "A")),
        (_Coded("bad", "A", 400), _Coded("bad", "A", 422)),
    ],
    ids=["code", "type", "status"],
)
def test_errors_differing_in_code_type_or_status_are_two_causes(
    a: Exception, b: Exception
) -> None:
    # Each part of the cause tells two errors apart: no known-good request,
    # both counted.
    probe = _Probe()
    assert _indices({0: a, 1: b}, 2, lambda _: True, probe) == [0, 1]
    assert probe.sent == []


def test_messages_differing_only_in_their_figures_are_one_cause() -> None:
    probe = _Probe()
    errors = {i: _Coded(f"request has {900 + i} tokens") for i in range(3)}
    assert _indices(errors, 3, lambda _: True, probe) == []
    assert probe.sent == [False]


def test_per_request_token_counts_do_not_split_one_cause() -> None:
    # An OpenAI-compatible server's 400s carry each request's own figures:
    # every unit rejected so is still one cause, which the known-good
    # request decides, whatever the configuration patterns name.
    predicate = item_error_predicate(object())
    rejected = {
        i: _openai(openai.BadRequestError, 400, f"input has {900 + i} tokens, too many")
        for i in range(3)
    }
    probe = _Probe(answered=False)
    assert _indices(rejected, 3, predicate, probe) == []
    assert probe.sent == [False]


@pytest.mark.parametrize(
    "message",
    [
        "'max_tokens' or 'max_completion_tokens' is too large: 4000. This "
        "model's maximum context length is 4096 tokens and your request has "
        "{n} input tokens (4000 > 4096 - {n}).",
        "This model's maximum context length is 4096 tokens. However, you "
        "requested {n} tokens ({n} in the messages, 4000 in the completion).",
    ],
)
def test_a_context_length_rejection_is_the_configuration(message: str) -> None:
    # vLLM and OpenAI-compatible servers reject a request whose chunk and
    # max_tokens together pass the model's context: the settings, not the
    # document (chunk size bounds every unit).
    predicate = item_error_predicate(OpenAICompatibleLanguageModelFactory(Config()))
    rejected = {
        i: _openai(openai.BadRequestError, 400, message.format(n=900 + i))
        for i in range(3)
    }
    probe = _Probe(answered=True)
    assert _indices(rejected, 3, predicate, probe) == []
    assert probe.sent == []


def test_the_known_good_request_bypasses_the_response_cache(tmp_path: Path) -> None:
    # A response an earlier run cached for the known-good request would pass
    # for a model that now rejects the setting: it is neither served nor
    # stored while the request is made.
    cache = LLMResponseCache()
    cache.open(tmp_path)
    cache.update("prompt", "llm", [Generation(text="cached")])
    cache.flush()
    cache.open(tmp_path)
    seen: list[object] = []

    def known_good() -> None:
        seen.append(cache.lookup("prompt", "llm"))
        cache.update("other", "llm", [Generation(text="new")])

    _indices({0: _ITEM}, 1, item_error_predicate(_BEDROCK), _Probe(call=known_good))
    assert seen == [None]
    cache.flush()
    cache.open(tmp_path)
    assert cache.lookup("other", "llm") is None
    assert cache.lookup("prompt", "llm") is not None


_KNOWN_GOOD = known_good_unit().text
# An answer every stage parses: an entity for graph extraction, an empty
# claims section (no claims), no refinement plan.
_ANSWER = {"entities": {"entity": {"name": "Vendor Corp"}}, "claims": {}}


class _Rejecting:
    """A chain the model answers with an error for every request: ``error``,
    or the error ``error`` maps the text of the request's unit to (a request
    of no listed text, the known-good one, is answered). With
    ``known_good_answered`` the known-good request is answered either way."""

    def __init__(
        self, error: Exception | dict[str, Exception], known_good_answered: bool
    ) -> None:
        self.error = error
        self.known_good_answered = known_good_answered

    def invoke(self, request: Any, *_: Any, **__: Any) -> Any:
        if self.known_good_answered and _KNOWN_GOOD in str(request):
            return _ANSWER
        if isinstance(self.error, dict):
            for text, error in self.error.items():
                if text in str(request):
                    raise error
            return _ANSWER
        raise self.error

    def batch(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("the extractors call the model per item")


_STAGES = {
    "graph": (graph_module, graph_module.GraphExtractor),
    "claims": (claim_module, claim_module.ClaimExtractor),
    "gleaning": (gleaner_module, gleaner_module.GraphGleaner),
}
# A message no configuration pattern names: only the rule tells it apart.
_TOO_LONG = _bedrock("ValidationException", 400, "Input is too long.")


def _processor(
    stage: str,
    config: Config,
    error: Exception | dict[str, Exception],
    known_good_answered: bool = False,
) -> Any:
    """``stage``'s extractor, its model answering every request with
    ``error`` (see :class:`_Rejecting`)."""
    config.processing.ignore_errors = True
    # Output errors are retried: one attempt keeps the backoff out.
    config.processing.max_attempts = 1
    module, cls = _STAGES[stage]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            bedrock_module, "BedrockLanguageModelFactory", lambda *_, **__: _BEDROCK
        )
        patch.setattr(module, "create_robust_xml_output_parser", lambda **_: None)
        patch.setattr(
            module,
            "setup_chain",
            lambda **_: _Rejecting(error, known_good_answered),
        )
        processor: Any = cls(config)
    processor.show_progress = False
    return processor


def _counted(processor: Any, stage: str, units: list[TextUnit]) -> list[str]:
    """The ids of the units ``processor`` counts against their document."""
    if stage == "gleaning":
        records = [TextUnitExtraction(text_unit_id=u.id) for u in units]
        return sorted(processor.glean(units, records)[1].item_failed_text_unit_ids)
    return sorted(processor.extract_units(units)[1].item_failed_text_unit_ids)


def _run(
    stage: str,
    config: Config,
    units: list[TextUnit],
    error: Exception | dict[str, Exception] = _TOO_LONG,
    known_good_answered: bool = False,
) -> list[str]:
    """The ids of the units ``stage`` counts against their document when the
    model rejects all of them with ``error``."""
    processor = _processor(stage, config, error, known_good_answered)
    return _counted(processor, stage, units)


def _two_documents() -> list[TextUnit]:
    return [
        TextUnit(id="u0", text="alpha", document_ids=["a"]),
        TextUnit(id="u1", text="beta", document_ids=["b"]),
    ]


@pytest.mark.parametrize("stage", list(_STAGES))
def test_every_extraction_stage_asks_a_known_good_request_on_one_cause(
    stage: str, config: Config
) -> None:
    # One error on every unit is the configuration when the model rejects a
    # known-good request too, in every extraction stage, for two documents
    # or one; when it answers that request, the documents' own (so a delta
    # of bad documents alone is excluded in the end).
    two = [TextUnit(id=f"u{i}", text="t", document_ids=[d]) for i, d in enumerate("ab")]
    one = [TextUnit(id=f"u{i}", text="t", document_ids=["a"]) for i in range(2)]
    for units in (two, one):
        assert _run(stage, config, units) == []
        assert _run(stage, config, units, known_good_answered=True) == ["u0", "u1"]


@pytest.mark.parametrize("stage", list(_STAGES))
def test_every_extraction_stage_counts_only_the_items_own_errors(
    stage: str, config: Config
) -> None:
    # u0's request is rejected for its content; u1's for access, which is no
    # document's fault: only u0 counts, in every stage.
    errors = {
        "alpha": _TOO_LONG,
        "beta": _bedrock("AccessDeniedException", 403, "not authorized"),
    }
    assert _run(stage, config, _two_documents(), errors) == ["u0"]


@pytest.mark.parametrize("stage", list(_STAGES))
def test_every_extraction_stage_counts_distinct_errors_of_several_documents(
    stage: str, config: Config
) -> None:
    # Both units rejected for their content, each with its own message: one
    # cause (type, code and status), but the model answers the known-good
    # request, so each counts against its document.
    errors = {
        "alpha": _TOO_LONG,
        "beta": _bedrock("ValidationException", 400, "Input is too long (2)."),
    }
    assert _run(stage, config, _two_documents(), errors) == ["u0", "u1"]


@pytest.mark.parametrize("stage", list(_STAGES))
def test_every_extraction_stage_counts_each_call_anew(
    stage: str, config: Config
) -> None:
    # A second call on the same extractor reports its own failures, not the
    # ones of the call before it.
    errors: dict[str, Exception] = {"alpha": _TOO_LONG, "beta": _TOO_LONG}
    processor = _processor(stage, config, errors)
    one = [
        TextUnit(id=f"u{i}", text=t, document_ids=["a"])
        for i, t in enumerate(("alpha", "beta"))
    ]
    assert _counted(processor, stage, one) == ["u0", "u1"]
    errors["alpha"] = errors["beta"] = _bedrock("AccessDeniedException", 403, "x")
    assert _counted(processor, stage, one) == []


@pytest.mark.parametrize("stage", list(_STAGES))
@pytest.mark.parametrize("error", _OUTPUT_ERRORS, ids=repr)
def test_every_extraction_stage_asks_a_known_good_request_on_uniform_bad_output(
    stage: str, error: Exception, config: Config
) -> None:
    # Every unit's output cut off or unparseable (a server-side output cap):
    # across documents no document counts. For one document the known-good
    # request's answer, cut too, makes it the model's; one that parses makes
    # it the document's own.
    for answered in (False, True):
        assert _run(stage, config, _two_documents(), error, answered) == []
    one = _two_documents()[:1]
    assert _run(stage, config, one, error) == []
    assert _run(stage, config, one, error, known_good_answered=True) == ["u0"]


class _Cut:
    """A graph-extraction chain whose output is cut inside the first entity's
    name for every unit (no stop reason), and for the known-good request
    unless ``known_good_answered``."""

    def __init__(self, known_good_answered: bool) -> None:
        self.known_good_answered = known_good_answered
        self.calls = 0

    def invoke(self, request: Any, *_: Any, **__: Any) -> Any:
        if self.known_good_answered and _KNOWN_GOOD in str(request):
            return _ANSWER
        self.calls += 1
        return {"entities": {"entity": {"name": {}}}}


@pytest.mark.parametrize("known_good_answered", [False, True])
def test_graph_output_with_no_named_entity_is_cut_off_output(
    known_good_answered: bool, config: Config
) -> None:
    # Output cut off without a stop reason parses into entities whose names
    # never arrived: cut-off output (it recorded the unit PROCESSED with
    # nothing extracted), which the known-good request rule then judges.
    # Not retried: as a parse failure, each unit was asked max_attempts
    # times with backoff, about 165 s a unit at the default 5.
    processor = _processor("graph", config, _TOO_LONG)
    processor.batch_processor.max_attempts = 3
    processor.batch_processor.retry_multiplier = 0.001
    cut = processor.graph_extractor = _Cut(known_good_answered)
    records, stats = processor.extract_units(_two_documents()[:1])
    assert cut.calls == 1 + (0 if known_good_answered else 1)
    assert records == []
    assert stats.failed_text_unit_ids == ["u0"]
    assert stats.item_failed_text_unit_ids == (["u0"] if known_good_answered else [])


def test_claims_count_an_answer_that_does_not_parse(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model answered, but its answer does not parse: the unit's own.
    processor = _processor("claims", config, _TOO_LONG)
    # A whole claims section, so the call itself succeeds (_checked).
    processor.claim_extractor = SimpleNamespace(invoke=lambda *_: {"claims": {}})
    monkeypatch.setattr(
        processor,
        "_parse_extraction_result",
        lambda *_: (_ for _ in ()).throw(OutputParserException("bad")),
    )
    assert _counted(processor, "claims", _two_documents()[:1]) == ["u0"]


@pytest.mark.parametrize(
    ("error", "answered"),
    [
        (None, True),
        # The model answered; its output did not parse or was cut off.
        (OutputParserException("bad"), True),
        (LLMOutputTruncatedError("cut"), True),
        (_ITEM, False),
        (TimeoutError(), False),
    ],
    ids=repr,
)
def test_a_known_good_request_is_answered_unless_the_call_fails(
    error: Exception | None, answered: bool
) -> None:
    def call(_: dict[str, Any]) -> Any:
        if error is not None:
            raise error
        return {}

    processor = BatchProcessor(max_attempts=1, is_transient_error=lambda _: False)
    assert (
        processor.answered(known_good_unit(), lambda units: [{}], call, "t") is answered
    )
