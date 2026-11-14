# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import numbers
import sys
import threading
import warnings
from collections.abc import Hashable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TextIO

import structlog
from structlog.stdlib import LoggerFactory, ProcessorFormatter

from . import observability
from .exceptions import ErrorClass, classify, error_text, own_text
from .utils.common import error_digest, error_locator, text_digest


class LoggingSettings(Protocol):
    """The settings logging is configured from: the ``logging`` section of
    the domain ``Config`` (``LoggingConfig``). Typed structurally, so the
    kernel imports no other layer."""

    @property
    def level(self) -> str: ...

    @property
    def log_format(self) -> str: ...

    @property
    def log_to_file(self) -> bool: ...

    @property
    def log_file_path(self) -> str: ...

    @property
    def library_levels(self) -> Mapping[str, str]: ...


class HasLoggingSettings(Protocol):
    """A configuration with a ``logging`` section (the domain ``Config``)."""

    @property
    def logging(self) -> LoggingSettings: ...


def add_trace_ids(
    _logger: Any, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Add ``trace_id``/``span_id`` of the active span, so logs join traces.

    One check and nothing added while tracing is off or no span is active.
    """
    ids = observability.current_trace_ids()
    if ids is not None:
        event_dict.setdefault("trace_id", ids[0])
        event_dict.setdefault("span_id", ids[1])
    return event_dict


# Shared by structlog-native and foreign (stdlib: botocore, langchain_aws, ...)
# records so both carry the same keys, including context bound with
# structlog.contextvars (pipeline_id / stage / query_id / conversation_id) and
# the active trace ids.
_SHARED_PROCESSORS: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    add_trace_ids,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="ISO"),
]


# ``extra`` keys that ``logging.Logger.makeRecord`` refuses (KeyError), minus
# the logging kwargs ``render_to_log_kwargs`` passes separately.
_RESERVED_RECORD_KEYS = (
    set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}
) - set(structlog.stdlib.LOG_KWARG_NAMES)


class _FormatterEvent(dict[str, Any]):
    """The event dict a ``setup_logging`` record carries as its ``msg``.

    The package's ``ProcessorFormatter`` handlers copy and render the whole
    dict. Any other handler on the root logger (a host application's own,
    pytest's caplog) formats ``str(record.msg)``: the plain message, not the
    dict repr.
    """

    def __str__(self) -> str:
        return str(self.get("event", ""))


def _to_stdlib(logger: Any, method: str, event_dict: structlog.types.EventDict) -> Any:
    """Hand the event to stdlib logging, in the form its handlers expect.

    Under ``setup_logging`` the CLI's ``ProcessorFormatter`` handlers render
    the whole event dict, and other root handlers get the plain message and
    the exception (see ``_FormatterEvent``). Otherwise (library mode) the
    record carries the message as ``msg`` and the fields as ``extra``, so a
    host handler's own formatter prints the plain message, not a dict repr.
    Decided per call: loggers cached before ``setup_logging`` switch too. A
    field named like a ``LogRecord`` attribute (a host binding ``name`` or
    ``process`` with structlog.contextvars) is passed with a trailing ``_``.
    """
    if LoggingSetup._handlers:
        # As ProcessorFormatter.wrap_for_formatter, plus exc_info on the
        # record: the package's formatters drop the record's copy and render
        # the event's, a host formatter appends the traceback.
        kwargs: dict[str, Any] = {"extra": {"_logger": logger, "_name": method}}
        if exc_info := event_dict.get("exc_info"):
            kwargs["exc_info"] = exc_info
        return (_FormatterEvent(event_dict),), kwargs
    for key in _RESERVED_RECORD_KEYS & event_dict.keys():
        event_dict[f"{key}_"] = event_dict.pop(key)
    return structlog.stdlib.render_to_log_kwargs(logger, method, event_dict)


# The package's own loggers (``get_logger(__name__)``): their records keep to
# the logging policy (``log_failure``); every other logger is a library's.
_PACKAGE_LOGGER = __name__.partition(".")[0]
# Where a library's warnings go under ``setup_logging`` (``_log_warning``).
_warnings_logger = logging.getLogger(f"{_PACKAGE_LOGGER}.warnings")


# Library loggers whose records are request metadata the package's own
# servers log, not text: uvicorn's access line (client, method, path without
# its query string, status) of the MCP server's HTTP transport.
_METADATA_LOGGERS = frozenset({"uvicorn.access"})


def _is_package_record(record: logging.LogRecord) -> bool:
    return record.name == _PACKAGE_LOGGER or record.name.startswith(
        f"{_PACKAGE_LOGGER}."
    )


# The attribute that marks a record ``LibraryRecordFilter`` rewrote: the
# record is shared by every handler, so the next one keeps it as it is (a
# digest of the digest would make the console and the log file disagree).
_FILTERED = "_kg_rag_filtered"


def _without_query(path: object) -> object:
    """An access line's path without its query string (it may hold a key:
    ``/mcp?api_key=...``)."""
    return path.partition("?")[0] if isinstance(path, str) else path


def _safe_arg(value: object) -> object:
    """A library record's argument as the logging policy allows it at INFO
    and above: a number or a boolean as it is, an error by
    :func:`error_digest`, any other value by ``text_digest`` of its text
    (it may be a prompt, a model's reply or an error's message)."""
    if value is None or isinstance(value, numbers.Number):
        return value
    if isinstance(value, BaseException):
        return error_digest(value)
    return text_digest(str(value))


class LibraryRecordFilter(logging.Filter):
    """Keep a library's INFO+ records to the logging policy (SECURITY.md).

    The package logs an error's text at DEBUG only (``log_failure``), but a
    library logs its own: ragas a failed judge job's error at ERROR, the
    ``ChatBedrock`` path a ``ClientError``'s traceback (``logger.exception``),
    either quoting a prompt or an ARN. On the handlers ``setup_logging``
    adds, a record of a logger outside the package at INFO or above keeps
    its logger name, level and message template (the library's code, which
    says what happened) and loses what may hold that text: its arguments
    (:func:`_safe_arg`) and its traceback, of which the error's
    :func:`error_digest` and :func:`error_locator` are appended; a message
    that is no string (``logger.error(exc)``) is digested itself. A record
    is rewritten once, whichever handler meets it first. A library that
    formats the text into the message itself (an f-string) is not caught:
    set its level in ``logging.library_levels``. DEBUG records keep
    everything, as the package's own do. uvicorn's access line keeps its
    arguments but the query string of its path."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, _FILTERED, False) or record.levelno < logging.INFO:
            return True
        if record.name in _METADATA_LOGGERS:
            # uvicorn's access line: client, method, path, version, status.
            if isinstance(record.args, tuple) and len(record.args) == 5:
                client, method, path, version, status = record.args
                record.args = (client, method, _without_query(path), version, status)
            setattr(record, _FILTERED, True)
            return True
        # A structlog event (``get_logger``: the package's own, also a
        # module of it run as ``__main__``) is the package's record.
        if _is_package_record(record) or isinstance(record.msg, _FormatterEvent):
            return True
        if not isinstance(record.msg, str):
            # ``logger.error(exc)``: the message is the object itself.
            record.msg = (
                error_digest(record.msg)
                if isinstance(record.msg, BaseException)
                else text_digest(str(record.msg))
            )
        elif record.name == "py.warnings":
            # ``logging.captureWarnings``: the warning's whole text is the
            # message (``setup_logging``'s own hook names its category).
            record.msg = f"Python warning ({text_digest(record.getMessage())})"
            record.args = None
        if isinstance(record.args, Mapping):
            record.args = {key: _safe_arg(v) for key, v in record.args.items()}
        elif record.args:
            record.args = tuple(_safe_arg(value) for value in record.args)
        error = record.exc_info[1] if record.exc_info else None
        if error is not None:
            where = error_locator(error)
            record.msg = (
                f"{record.msg} [{error_digest(error)}{'; ' + where if where else ''}]"
            )
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        setattr(record, _FILTERED, True)
        return True


def _log_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """``warnings.showwarning`` under ``setup_logging``: a library's warning
    (``warnings.warn``) as a WARNING record of the package's, named by its
    category and the code location that raised it, its text, which may
    quote a value, by ``text_digest`` (the text at DEBUG), as the logging
    policy (SECURITY.md) has every library record."""
    where = "/".join(Path(filename).parts[-2:])
    text = str(message)
    _warnings_logger.warning(
        "%s at %s:%s (%s)", category.__name__, where, lineno, text_digest(text)
    )
    _warnings_logger.debug("%s at %s:%s: %s", category.__name__, where, lineno, text)


def _configure_structlog() -> None:
    """Route structlog through stdlib logging; installs no handler.

    Records then reach whatever handlers the process has: the CLI's (see
    ``setup_logging``) or a host application's own. Levels are the stdlib
    logger levels.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_SHARED_PROCESSORS,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            _to_stdlib,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class LoggingSetup:
    _handlers: list[logging.Handler] = []

    @classmethod
    def setup_logging(
        cls,
        config: HasLoggingSettings,
        config_override: dict[str, Any] | None = None,
        *,
        stream: TextIO | None = None,
    ) -> None:
        """Configure structlog + the root handlers from ``config.logging``.

        Called by the CLIs only (after loading ``--config-path``); importing the
        package never touches the root logger. A repeated call replaces only
        the handlers a previous call added. ``stream`` defaults to stdout.
        """
        log_config = config.logging

        if config_override:
            for key, value in config_override.items():
                setattr(log_config, key, value)

        log_level = getattr(logging, log_config.level.upper(), logging.INFO)
        structured = log_config.log_format == "structured"
        stream = stream or sys.stdout

        _configure_structlog()

        root_logger = logging.getLogger()
        # Re-init replaces only our own handlers (keeps e.g. pytest's caplog).
        for handler in cls._handlers:
            root_logger.removeHandler(handler)
            handler.close()
        cls._handlers = []
        root_logger.setLevel(log_level)

        # Handlers carry no level of their own: logger levels decide, so a
        # CLI --verbose that lowers the 'unified_kg_rag' logger takes effect.
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(
            cls._formatter(
                json=structured and not stream.isatty(), colors=stream.isatty()
            )
        )
        cls._handlers.append(console_handler)

        if log_config.log_to_file and log_config.log_file_path:
            log_file_path = cls._get_log_file_path_with_date(log_config.log_file_path)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(cls._formatter(json=structured, colors=False))
            cls._handlers.append(file_handler)

        library_filter = LibraryRecordFilter()
        for handler in cls._handlers:
            handler.addFilter(library_filter)
            root_logger.addHandler(handler)
        # A library's warnings.warn reaches these handlers, not bare stderr.
        warnings.showwarning = _log_warning

        # Chatty dependencies (e.g. langchain_aws logs INFO per model call and a
        # full traceback per retried attempt) default to WARNING.
        for name, level in log_config.library_levels.items():
            logging.getLogger(name).setLevel(level.upper())

    @staticmethod
    def _formatter(json: bool, colors: bool) -> ProcessorFormatter:
        renderer: list[structlog.types.Processor] = (
            [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(default=str),
            ]
            if json
            # plain_traceback: the default (rich) formatter prints each frame's
            # local variables (prompts, chunk text, model output).
            else [
                structlog.dev.ConsoleRenderer(
                    colors=colors, exception_formatter=structlog.dev.plain_traceback
                )
            ]
        )
        return ProcessorFormatter(
            processors=[ProcessorFormatter.remove_processors_meta, *renderer],
            foreign_pre_chain=_SHARED_PROCESSORS,
        )

    @staticmethod
    def _get_log_file_path_with_date(log_file_path: str) -> Path:
        """Dated variant of ``log_file_path``; a relative path is under the CWD."""
        log_path = Path(log_file_path).expanduser().resolve()
        current_date = datetime.now().strftime("%Y%m%d")
        return log_path.with_name(f"{log_path.stem}_{current_date}{log_path.suffix}")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A structlog logger that emits through stdlib logging.

    Configures structlog on first use unless the host application already
    did; never adds handlers or changes levels.
    """
    if not structlog.is_configured():
        _configure_structlog()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


_warned_once: set[Hashable] = set()
_warned_once_lock = threading.Lock()


def log_failure(
    logger: Any,
    level: int,
    message: str,
    *args: Any,
    error: BaseException,
    traceback: bool = False,
) -> None:
    """Log ``message`` (``%``-formatted, its last ``%s`` the error) at
    ``level`` with :func:`error_digest` of ``error`` (its type and a hash of
    its text, which may quote a model's reply, a query or corpus text), and
    with the error itself at DEBUG, with its traceback when ``traceback``:
    what the logging policy (SECURITY.md) allows at INFO and above. With
    ``traceback`` the ``level`` line also says where it came from
    (:func:`error_locator`: the type chain and ``file:line`` frames, no
    message text), so an unexpected failure is locatable at the default
    level. The one way to log an error whose text may hold model output or
    corpus text."""
    digest = error_digest(error)
    if traceback and (where := error_locator(error)):
        digest = f"{digest}; {where}"
    logger.log(level, message, *args, digest)
    logger.debug(message, *args, error, exc_info=error if traceback else None)


def error_log_text(error: BaseException) -> str:
    """What a record at INFO and above says of ``error``, by its class (the
    rule of :func:`log_error`): an unexpected ``failure`` by
    :func:`error_digest` (its type, code, status and a hash of its text,
    which may quote a model's reply, a query, corpus text or an ARN), a
    failure whose message is the package's own (``own_text``) and any other
    class (an unreachable store or model, the deadline, a refusal) by its
    class and text. For an error kept to be logged later (a write's
    ``IndexingStats.errors``, a worker's outcome): log the error itself at
    DEBUG."""
    error_class = classify(error)
    if error_class is ErrorClass.FAILURE:
        own = own_text(error)
        if own is None:
            return error_digest(error)
        return f"[{error_class.value}] {own}"
    return f"[{error_class.value}] {error_text(error)}"


def log_error(
    logger: Any,
    level: int,
    message: str,
    *args: Any,
    error: BaseException,
) -> None:
    """Log ``error`` (``message``'s last ``%s``) by its class, the one rule
    of every surface's last-resort line (MCP, console, ``run-rag``,
    ``run-eval``, ``run-ingestion``): an unexpected ``failure`` through
    :func:`log_failure` (type, digest and where it came from at ``level``;
    text and traceback at DEBUG), any other class (an unreachable store or
    model, the deadline, a refusal: an expected outcome with its own
    message, whose endpoint is not user data), and a failure whose message
    is the package's own (``has_own_text``), with its text (``error_text``:
    what a surface shows) and class."""
    error_class = classify(error)
    own = own_text(error) if error_class is ErrorClass.FAILURE else None
    if error_class is ErrorClass.FAILURE and own is None:
        log_failure(logger, level, message, *args, error=error, traceback=True)
        return
    logger.log(level, message, *args, error_log_text(error))
    if own is not None and own != error_text(error):
        # The rest of a message only partly the package's own.
        logger.debug(message, *args, error, exc_info=error)


def warn_once(
    logger: Any,
    key: Hashable,
    message: str,
    *args: Any,
    then_debug: bool = False,
    seen: set[Any] | None = None,
) -> bool:
    """Log ``message`` at WARNING on ``logger`` the first time ``key`` is
    seen (a missing optional model or analyzer, a cap a hub entity hits on
    every query); a repeat is not logged, or with ``then_debug`` logged at
    DEBUG. ``seen`` holds the keys warned of (by default this process's):
    a caller's own set scopes them, as one retriever's to a query. Returns
    whether it warned."""
    with _warned_once_lock:
        store = _warned_once if seen is None else seen
        first = key not in store
        store.add(key)
    if first:
        logger.warning(message, *args)
    elif then_debug:
        logger.debug(message, *args)
    return first


def setup_logging(
    config: HasLoggingSettings,
    config_override: dict[str, Any] | None = None,
    stream: TextIO | None = None,
) -> None:
    """(Re-)initialise logging from ``config`` — call after loading a CLI config."""
    LoggingSetup.setup_logging(config, config_override, stream=stream)
    warn_if_langsmith_tracing()


def warn_if_langsmith_tracing() -> None:
    """Warn that LangSmith tracing, when enabled, uploads prompt content.

    ``LANGSMITH_TRACING=true`` (or ``LANGCHAIN_TRACING_V2=true``) makes
    langchain-core send every traced run to LangSmith, including prompts,
    retrieved context and model outputs. That may be intended, so it is
    reported, not disabled.
    """
    # Imported here: langsmith pulls in httpx/requests, and the kernel is
    # imported by the pure domain layer.
    from langsmith.utils import tracing_is_enabled

    if tracing_is_enabled():
        get_logger(__name__).warning(
            "LangSmith tracing is enabled by the environment "
            "(LANGSMITH_TRACING / LANGCHAIN_TRACING_V2): prompts, retrieved "
            "context and model outputs will be sent to LangSmith. Unset the "
            "variable to keep this content local."
        )
