# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console's FastAPI app (docs/design.md#101-the-console-run-console-opt-in).

:func:`create_app` wires the pages' services over one :class:`ConsoleOptions`.
Every path a request names goes through :class:`~.paths.Roots`; requests
that change anything must carry the ``X-Console-Request`` header (a custom
header a cross-site form cannot send, and no CORS header is ever returned),
and on the loopback interface the ``Host`` must be a loopback name (DNS
rebinding).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, ConfigDict, Field

from unified_kg_rag import __version__
from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.application.cli.console import dashboard, evaluations, graph_api
from unified_kg_rag.application.cli.console.ask import (
    AskRequest,
    AskService,
    ChainFactory,
)
from unified_kg_rag.application.cli.console.jobs import (
    JobConflict,
    JobManager,
    NarrowingNotConfirmed,
    dropped_methodologies,
    ingest_config,
)
from unified_kg_rag.application.cli.console.paths import (
    PathRefused,
    Root,
    Roots,
    contained,
    safe_name,
    safe_relative,
    walk_files,
)
from unified_kg_rag.application.cli.console.settings_api import (
    DirectoriesNotConfirmed,
    EndpointNotConfirmed,
    SecretRefused,
    SettingsFile,
    schema,
)
from unified_kg_rag.application.cli.console.store import ConsoleStore
from unified_kg_rag.application.cli.conventions import host_without_port, is_loopback
from unified_kg_rag.application.config import get_config
from unified_kg_rag.application.plugins import load_plugins
from unified_kg_rag.domain.ingestion.cache_keys import is_excluded_source_path
from unified_kg_rag.domain.ingestion.visualization_data import (
    VISUALIZATION_DATA_FILENAME,
)
from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    ModelRole,
    validate_safe_name,
)
from unified_kg_rag.domain.models.retrieval import MAX_SERVED_TOP_K
from unified_kg_rag.domain.prompts.retrieval import STRATEGY_ROUTING_GUIDE
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import (
    INTERNAL_ERROR_MESSAGE,
    ConfigurationError,
    ErrorClass,
    GraphRAGException,
    InvalidArgumentError,
    classify,
    error_text,
)
from unified_kg_rag.shared.utils import event_loop
from unified_kg_rag.shared.utils.files import open_temporary, replace_together

logger = get_logger(__name__)

REQUEST_HEADER = "X-Console-Request"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_UPLOAD_FILES = 10000
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self' "
        "'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self'; "
        "object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}
# Not offered as corpora: the console's and the stores' own directories.
_NOT_CORPORA = {"cache", "outputs", "stores", "node_modules", "datasets"}


@dataclass(frozen=True)
class ConsoleOptions:
    """What ``run-console`` was started with (paths resolved)."""

    workspace: Path
    cache_directory: Path
    config_path: Path | None = None
    allow_unknown_keys: bool = False
    loopback: bool = True
    web_directory: Path | None = None
    trace_url_template: str | None = None
    sample_questions: list[str] = field(default_factory=list)
    # The per-launch access token every request must carry (the cookie the
    # tokenised URL sets, or the X-Console-Token header); None turns the
    # check off (tests of the other checks only).
    token: str | None = None

    @property
    def console_directory(self) -> Path:
        return self.cache_directory / "console"

    @property
    def uploads_directory(self) -> Path:
        return self.workspace / "uploads"


class _Request(BaseModel):
    """A request body of the API: an unknown key is refused (422,
    ``invalid_argument``), not ignored, as ``RAGInput`` refuses one."""

    model_config = ConfigDict(extra="forbid")


# The query parameters any route takes besides its own: the access token.
_ANY_ROUTE_QUERY = frozenset({"token"})


def _query_names(dependant: Any) -> set[str]:
    """The query parameters a route (and its dependencies) declares."""
    names = {param.alias for param in dependant.query_params}
    for dependency in dependant.dependencies:
        names |= _query_names(dependency)
    return names


def _refuse_unknown_query(request: Request) -> None:
    """Refuse a query parameter an API route does not declare (422,
    ``invalid_argument``), as the bodies refuse an unknown key; a page's
    URL keeps its own (``/graph?focus=...``, read by the web client)."""
    dependant = getattr(request.scope.get("route"), "dependant", None)
    if dependant is None or not request.url.path.startswith("/api/"):
        return
    known = _query_names(dependant)
    unknown = sorted(set(request.query_params) - known - _ANY_ROUTE_QUERY)
    if unknown:
        raise RequestValidationError(
            [
                {
                    "loc": ("query", name),
                    "msg": "Extra inputs are not permitted",
                    "type": "extra_forbidden",
                }
                for name in unknown
            ]
        )


class IngestBody(_Request):
    corpus: str = Field(min_length=1, max_length=1024)
    methodologies: list[str] | None = None
    index_value: str | None = Field(default=None, max_length=128)
    force_rebuild: bool = False
    reset: bool = False
    confirm_reset: bool = False
    confirm_deletions: bool = False
    confirm_integrity: bool = False
    confirm_narrowing: bool = False


class EvaluateBody(_Request):
    dataset: str = Field(min_length=1, max_length=1024)
    strategies: list[str] = Field(min_length=1, max_length=12)
    repeats: int = Field(default=1, ge=1, le=20)
    closed_book: bool = False
    pairwise_judge: bool = False
    fail_if: list[str] = Field(default_factory=list, max_length=20)
    max_failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class PreviewBody(_Request):
    text: str | None = Field(default=None, max_length=2_000_000)
    changes: list[dict[str, Any]] | None = Field(default=None, max_length=500)


class SaveBody(_Request):
    text: str = Field(max_length=2_000_000)
    base_sha: str = Field(min_length=64, max_length=64)
    confirm_endpoints: list[str] = Field(default_factory=list, max_length=50)
    confirm_directories: list[str] = Field(default_factory=list, max_length=200)


class UploadTooLarge(InvalidArgumentError):
    """An upload over the size limit (or without a Content-Length)."""


class UnsupportedUpload(InvalidArgumentError):
    """An upload holding a file ingestion does not read: of a type it does
    not parse, or hidden (``is_excluded_source_path``)."""


class UploadConflict(InvalidArgumentError):
    """An upload naming one path as a file and as a folder (two of its
    files, or one of its files and what the folder holds)."""


def _upload_conflicts(target: Path, paths: list[Path]) -> list[str]:
    """The paths of an upload into ``target`` that name a file and a folder
    at once: a file under another of its files, a file where the folder
    holds a folder, or under a file the folder holds. Relative to
    ``target``."""
    planned = set(paths)
    found: list[str] = []
    for path in paths:
        parents = [d for d in path.parents if d == target or d.is_relative_to(target)]
        clash = path.is_dir() or any(
            d in planned or (d.exists() and not d.is_dir()) for d in parents
        )
        if clash:
            found.append(path.relative_to(target).as_posix())
    return sorted(found)


def _remove_created(directories: list[Path]) -> None:
    """Remove the directories a refused upload created, deepest first, each
    only while empty."""
    for directory in sorted(directories, key=lambda d: len(d.parts), reverse=True):
        with contextlib.suppress(OSError):
            directory.rmdir()


def _same_token(offered: str, token: str) -> bool:
    """Whether a client's token is the console's, in constant time.

    Compared as UTF-8 bytes: ``compare_digest`` refuses ``str`` arguments
    holding non-ASCII characters, which would answer 500, not 401.
    """
    return secrets.compare_digest(
        offered.encode("utf-8", "surrogateescape"), token.encode("utf-8")
    )


class ConsoleState:
    """The services of one console process."""

    def __init__(
        self, options: ConsoleOptions, *, chain_factory: ChainFactory | None = None
    ) -> None:
        self.options = options
        self.settings_file = SettingsFile(options.config_path)
        self._config: Config | None = None
        self.config_error: str | None = None
        options.console_directory.mkdir(parents=True, exist_ok=True)
        self.store = ConsoleStore(options.console_directory / "console.sqlite3")
        self.ask = AskService(
            self.load_config,
            self.store,
            trace_url_template=options.trace_url_template,
            chain_factory=chain_factory,
        )
        self.try_config()
        self.jobs = JobManager(
            store=self.store,
            jobs_directory=options.console_directory / "jobs",
            cache_directory=options.cache_directory,
            evaluation_outputs=Path("."),
            config_path=options.config_path,
            load_config=self.load_config,
            on_ingest_end=self.ask.reset,
            read_held=self._held,
        )
        self.graphs = graph_api.GraphBuilder(options.cache_directory)
        self._build_roots()

    async def _held(self, namespace: str) -> list[str] | None:
        """The methodologies ``namespace``'s build record holds (None: no
        record)."""
        held = await self.ask.held_methodologies(namespace)
        return list(held["held"]) if held["recorded"] else None

    def _build_roots(self) -> None:
        """The directories from the options and the config (again after the
        config is saved: its outputs directories may have moved)."""
        config = self._config
        options = self.options
        outputs = (
            Path(config.evaluation.outputs_directory)
            if config
            else Path("outputs/evaluation")
        )
        self.evaluation_outputs = (
            outputs if outputs.is_absolute() else (Path.cwd() / outputs)
        ).resolve()
        viz = config.graph.visualization.outputs_directory if config else None
        roots = [
            Root("workspace", options.workspace),
            Root("uploads", options.uploads_directory, writable=True),
            Root("cache", options.cache_directory, writable=True),
            Root("outputs", self.evaluation_outputs, writable=True),
        ]
        if viz:
            viz_path = Path(viz) if Path(viz).is_absolute() else Path.cwd() / viz
            roots.append(Root("visualization", viz_path))
        self.roots = Roots.build(*roots)
        self.jobs.evaluation_outputs = self.evaluation_outputs

    def load_config(self) -> Config:
        """The config (loaded once, again after Settings saves it).

        Raises:
            ConfigurationError: It does not load.
        """
        if self._config is None:
            # The plugins first, as every CLI loads them before its config:
            # a file type or a kind one registers is known to every page
            # (a plugin that cannot be loaded is the config error shown).
            load_plugins()
            self._config = get_config(
                self.options.config_path,
                allow_unknown_keys=self.options.allow_unknown_keys,
            )
        return self._config

    def try_config(self) -> Config | None:
        try:
            config = self.load_config()
        except (ConfigurationError, FileNotFoundError) as e:
            self.config_error = str(e)
            return None
        self.config_error = None
        return config

    def reload(self) -> None:
        self._config = None
        self.ask.reset()
        self.try_config()
        self._build_roots()


# The HTTP status of an error of each class (docs/reference/exit-codes-and-json.md):
# a request to change is a 4xx, as the CLIs exit 2 for it, a store or model
# server out of reach a 503 and a passed deadline a 504.
HTTP_STATUS: dict[ErrorClass, int] = {
    ErrorClass.INVALID_FILTER: 400,
    ErrorClass.INVALID_ARGUMENT: 400,
    ErrorClass.NOT_INDEXED: 404,
    ErrorClass.INDEX_NOT_FOUND: 404,
    ErrorClass.CONFIGURATION: 400,
    ErrorClass.TIMEOUT: 504,
    ErrorClass.STORE_UNREACHABLE: 503,
    ErrorClass.MODEL_UNREACHABLE: 503,
    ErrorClass.FAILURE: 500,
}


def _error(
    status: int,
    code: str,
    message: str,
    *,
    error_class: ErrorClass | None = None,
    **extra: Any,
) -> JSONResponse:
    """An API error: ``code`` for the page to name it in its language,
    ``error`` the detail, and ``error_class`` the :class:`ErrorClass` every
    surface names a failure by: ``error_class`` when given, else ``code``
    when it is one, else ``invalid_argument`` for a request the caller can
    fix (a 4xx) and ``failure`` for the rest."""
    if error_class is None:
        try:
            error_class = ErrorClass(code)
        except ValueError:
            error_class = (
                ErrorClass.INVALID_ARGUMENT if status < 500 else ErrorClass.FAILURE
            )
    return JSONResponse(
        {"error": message, "code": code, "error_class": error_class.value, **extra},
        status_code=status,
    )


def _classified_error(e: Exception) -> JSONResponse:
    """The API error of ``e`` by its class (:func:`classify`), as every
    surface names it: its status from :data:`HTTP_STATUS`, ``code``
    ``invalid`` for a request the caller can fix (the page's word for a
    refusal) and the class otherwise."""
    error_class = classify(e)
    code = (
        "invalid" if error_class is ErrorClass.INVALID_ARGUMENT else error_class.value
    )
    # An unexpected error's text (botocore's, with ARNs) stays in the log.
    message = (
        INTERNAL_ERROR_MESSAGE if error_class is ErrorClass.FAILURE else error_text(e)
    )
    return _error(HTTP_STATUS[error_class], code, message, error_class=error_class)


TOKEN_COOKIE = "ukr_console_token"
TOKEN_HEADER = "X-Console-Token"
_UNAUTHORIZED_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Console</title></head><body style="font:16px/1.5 system-ui,sans-serif;
max-width:40rem;margin:4rem auto;padding:0 1rem">
<h1>Open the address run-console printed</h1>
<p>The console needs the access token that address carries
(<code>http://127.0.0.1:&lt;port&gt;/?token=&hellip;</code>). It is new at every
start; the terminal that started the console shows it.</p></body></html>"""


def create_app(
    options: ConsoleOptions, *, chain_factory: ChainFactory | None = None
) -> FastAPI:
    """The console app over ``options`` (``chain_factory`` builds the
    ``GraphRAGChain``, for tests)."""
    state = ConsoleState(options, chain_factory=chain_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Sized as every query path's loop is: the model and store calls the
        # chain offloads run in its default executor.
        loop = asyncio.get_running_loop()
        config = state.try_config() or Config()
        executor = event_loop.configure_event_loop(loop, config.processing.io_workers)
        try:
            yield
        finally:
            await state.jobs.shutdown()
            await state.ask.close()
            event_loop.abandon_default_executor(loop, executor)
            state.store.close()

    app = FastAPI(
        title="unified-kg-rag console",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        dependencies=[Depends(_refuse_unknown_query)],
    )
    app.state.console = state

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if options.loopback:
            if not is_loopback(host_without_port(request.headers.get("host") or "")):
                return _error(
                    421,
                    "foreign_host",
                    "the console answers only on its loopback address",
                )
        if options.token:
            offered = request.query_params.get("token")
            if offered is not None and not request.url.path.startswith("/api/"):
                if not _same_token(offered, options.token):
                    return HTMLResponse(_UNAUTHORIZED_PAGE, status_code=401)
                # The token moves from the address to a cookie once.
                redirect = RedirectResponse(request.url.path or "/", status_code=303)
                redirect.set_cookie(
                    TOKEN_COOKIE,
                    options.token,
                    httponly=True,
                    samesite="strict",
                    path="/",
                )
                return redirect
            carried = request.cookies.get(TOKEN_COOKIE) or request.headers.get(
                TOKEN_HEADER
            )
            if not carried or not _same_token(carried, options.token):
                if request.url.path.startswith("/api/"):
                    return _error(
                        401, "unauthorized", "the console's access token is missing"
                    )
                return HTMLResponse(_UNAUTHORIZED_PAGE, status_code=401)
        if request.url.path.startswith("/api/") and request.method not in (
            "GET",
            "HEAD",
            "OPTIONS",
        ):
            if request.headers.get(REQUEST_HEADER) != "1":
                return _error(403, "forbidden", f"missing the {REQUEST_HEADER} header")
            origin = request.headers.get("origin")
            if (
                origin
                and origin.split("://", 1)[-1].lower()
                != (request.headers.get("host") or "").lower()
            ):
                return _error(403, "forbidden", "cross-origin request refused")
        response = await call_next(request)
        for name, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(PathRefused)
    async def _refused(_: Request, e: PathRefused) -> JSONResponse:
        return _error(403, "path_refused", str(e))

    @app.exception_handler(SecretRefused)
    async def _secret(_: Request, e: SecretRefused) -> JSONResponse:
        return _error(422, "credential_refused", str(e))

    @app.exception_handler(EndpointNotConfirmed)
    async def _endpoints(_: Request, e: EndpointNotConfirmed) -> JSONResponse:
        return _error(409, "endpoints", str(e), paths=e.paths)

    @app.exception_handler(DirectoriesNotConfirmed)
    async def _directories(_: Request, e: DirectoriesNotConfirmed) -> JSONResponse:
        return _error(409, "directories", str(e), paths=e.paths)

    @app.exception_handler(RequestValidationError)
    async def _invalid_body(_: Request, e: RequestValidationError) -> JSONResponse:
        # As every other refusal: a message the page can show, not a list.
        details = "; ".join(
            f"{'.'.join(str(p) for p in error.get('loc', ())[1:]) or 'body'}: "
            f"{error.get('msg')}"
            for error in e.errors()
        )
        return _error(422, "invalid", details)

    @app.exception_handler(JobConflict)
    async def _conflict(_: Request, e: JobConflict) -> JSONResponse:
        return _error(409, "job_conflict", str(e), job_id=e.job_id)

    @app.exception_handler(NarrowingNotConfirmed)
    async def _narrowing(_: Request, e: NarrowingNotConfirmed) -> JSONResponse:
        return _error(
            409, "narrowing", str(e), namespace=e.namespace, methodologies=e.dropped
        )

    @app.exception_handler(UploadTooLarge)
    async def _too_large(_: Request, e: UploadTooLarge) -> JSONResponse:
        return _error(413, "too_large", str(e))

    @app.exception_handler(UnsupportedUpload)
    async def _unsupported(_: Request, e: UnsupportedUpload) -> JSONResponse:
        return _error(415, "unsupported", str(e))

    @app.exception_handler(UploadConflict)
    async def _upload_conflict(_: Request, e: UploadConflict) -> JSONResponse:
        return _error(409, "upload_conflict", str(e))

    @app.exception_handler(GraphRAGException)
    async def _classified(_: Request, e: GraphRAGException) -> JSONResponse:
        # Any other error of the package, by its class (ErrorClass) as every
        # surface names it: an unreachable store is a 503 store_unreachable,
        # a namespace without an index a 404 index_not_found.
        if classify(e) is ErrorClass.FAILURE:
            logger.error("Console request failed: %s", e, exc_info=e)
        return _classified_error(e)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, e: Exception) -> JSONResponse:
        # The server logs it with its traceback (it is raised on).
        return _classified_error(e)

    async def _unreadable(_: Request, e: Exception) -> JSONResponse:
        # A file the request named that is not what it should hold.
        return _error(
            422, "unreadable", str(e), error_class=ErrorClass.INVALID_ARGUMENT
        )

    for unreadable in (graph_api.GraphDataError, evaluations.EvaluationFileError):
        app.add_exception_handler(unreadable, _unreadable)

    @app.exception_handler(HTTPException)
    async def _http(_: Request, e: HTTPException) -> JSONResponse:
        code = {404: "not_found", 400: "invalid"}.get(e.status_code, "failed")
        return _error(e.status_code, code, str(e.detail))

    # -- status -----------------------------------------------------------

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        config = state.try_config()
        providers = config.providers if config else None
        return {
            "version": __version__,
            "config_path": str(options.config_path) if options.config_path else None,
            "config_error": state.config_error,
            "namespace": resolve_namespace(config, None) if config else None,
            "backend": (
                {
                    "llm": providers.llm.kind,
                    "embedding": providers.embedding.kind,
                    "rerank": providers.rerank.kind,
                    "graph": providers.graph.kind,
                    "vector": providers.vector.kind,
                    "doc_status": (
                        providers.doc_status.kind
                        if providers.doc_status.enabled
                        else None
                    ),
                }
                if providers
                else None
            ),
            "models": _models(config) if config else None,
            "methodologies": (
                [m.value for m in config.effective_methodologies] if config else []
            ),
            "observability": (
                {
                    "enabled": config.observability.enabled,
                    "backend": config.observability.backend.value,
                }
                if config
                else None
            ),
            "prices_configured": bool(config.evaluation.prices) if config else False,
            # The largest top_k the Ask request takes.
            "max_top_k": MAX_SERVED_TOP_K,
            "jobs": state.jobs.active(),
            "loopback": options.loopback,
            "sample_questions": options.sample_questions,
            "directories": {
                name: str(root.path) for name, root in state.roots.roots.items()
            },
        }

    # -- ask --------------------------------------------------------------

    @app.get("/api/strategies")
    async def strategies(suffix: str | None = None) -> dict[str, Any]:
        found = await state.ask.strategies(_suffix(suffix))
        for item in found:
            guide = next(
                (
                    g
                    for s, g in STRATEGY_ROUTING_GUIDE.items()
                    if s.value == item["name"]
                ),
                None,
            )
            item["router_guide"] = guide
        return {"strategies": found}

    @app.post("/api/ask")
    async def ask(body: AskRequest) -> StreamingResponse:
        async def events() -> AsyncIterator[bytes]:
            async for event in state.ask.ask(body):
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n".encode()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/history")
    async def history(limit: int = 50) -> dict[str, Any]:
        rows = state.store.queries(limit=max(1, min(limit, 500)))
        return {
            "queries": [
                {
                    k: r[k]
                    for k in (
                        "id",
                        "created",
                        "question",
                        "requested_strategy",
                        "strategy",
                        "outcome",
                        "latency_seconds",
                        "cost_usd",
                    )
                }
                for r in rows
            ]
        }

    # -- dashboard --------------------------------------------------------

    @app.get("/api/dashboard")
    async def dashboard_view(days: int = 30) -> dict[str, Any]:
        config = state.load_config()
        since = time.time() - max(1, min(days, 3650)) * 86400
        jobs = state.jobs.all_jobs(since=since, limit=None)

        def summary() -> dict[str, Any]:
            # Up to 100,000 rows: read and summed off the loop.
            queries = state.store.queries(since=since, limit=100000)
            return dashboard.summarize(queries, jobs, config)

        return await asyncio.to_thread(summary)

    @app.delete("/api/history")
    async def clear_history() -> dict[str, Any]:
        return {"deleted": state.store.clear_queries()}

    # -- corpora and ingestion ---------------------------------------------

    @app.get("/api/corpora")
    async def corpora() -> dict[str, Any]:
        return {
            "corpora": await asyncio.to_thread(_corpora, state.roots),
            "extensions": sorted(ParserFactory.get_supported_extensions()),
        }

    @app.post("/api/uploads")
    async def upload(request: Request) -> dict[str, Any]:
        """Files into ``uploads/<name>/`` (form fields ``name`` and
        ``files``, each file named by its path in the folder), all or none:
        refused whole before parsing when over the size limit, every name
        and type checked before a byte is written, and each file written
        beside its target and moved into place only once all are written,
        so a refused upload leaves the folder as it was. The moves are not
        undone: if the folder changes under the request while they run (a
        path made a folder), the files moved in before the refused one
        stay (:func:`replace_together`)."""
        length = request.headers.get("content-length")
        if length is None or not length.isdigit():
            raise UploadTooLarge("an upload needs its Content-Length")
        if int(length) > MAX_UPLOAD_BYTES:
            raise UploadTooLarge(
                f"an upload takes at most {MAX_UPLOAD_BYTES // 2**20} MiB"
            )
        form = await request.form(max_files=MAX_UPLOAD_FILES, max_fields=8)
        staged: dict[Path, Path] = {}
        created: list[Path] = []
        try:
            name = form.get("name")
            files = [f for f in form.getlist("files") if not isinstance(f, str)]
            if not isinstance(name, str) or not files:
                raise PathRefused("an upload names its folder and holds files")
            uploads = state.roots.root("uploads").path
            folder = safe_name(name, "the corpus name", within=uploads)
            target = state.roots.resolve(f"uploads:{folder}", write=True)
            planned = [
                (
                    upload_file,
                    state.roots.resolve(
                        f"uploads:{folder}/"
                        + safe_relative(
                            upload_file.filename or "", within=target
                        ).as_posix(),
                        write=True,
                    ),
                )
                for upload_file in files
            ]
            supported = ParserFactory.get_supported_extensions()
            if unsupported := sorted(
                {p.name for _, p in planned if p.suffix.lower() not in supported}
            ):
                raise UnsupportedUpload(
                    f"ingestion does not read {', '.join(unsupported[:5])}"
                    f"{' and others' if len(unsupported) > 5 else ''}; it reads "
                    f"{' '.join(sorted(supported))}"
                )
            # The loader's own rule: a hidden file or folder is never read.
            if hidden := sorted(
                p.relative_to(target).as_posix()
                for _, p in planned
                if is_excluded_source_path(p, target)
            ):
                raise UnsupportedUpload(
                    f"ingestion skips hidden files and folders (a name starting "
                    f"with '.', __pycache__): {', '.join(hidden[:5])}"
                    f"{' and others' if len(hidden) > 5 else ''}"
                )
            if conflicts := _upload_conflicts(target, [p for _, p in planned]):
                raise UploadConflict(
                    "these paths name a file and a folder at once (in the "
                    f"upload or in the folder): {', '.join(conflicts[:5])}"
                    f"{' and others' if len(conflicts) > 5 else ''}"
                )
            total = 0
            for upload_file, path in planned:
                created.extend(
                    d
                    for d in reversed([path.parent, *path.parent.parents])
                    if (d == target or d.is_relative_to(target)) and not d.exists()
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                if (earlier := staged.pop(path, None)) is not None:
                    earlier.unlink(missing_ok=True)  # the path named twice
                temporary, out = open_temporary(path)
                staged[path] = temporary
                with out:
                    while chunk := await upload_file.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_UPLOAD_BYTES:
                            raise UploadTooLarge(
                                f"an upload takes at most {MAX_UPLOAD_BYTES // 2**20} MiB"
                            )
                        out.write(chunk)
            replace_together(staged)
            staged_count = len(staged)
            staged = {}
        except BaseException:
            for temporary in staged.values():
                temporary.unlink(missing_ok=True)
            _remove_created(created)
            raise
        finally:
            await form.close()
        return {
            "corpus": state.roots.ref(target),
            "files": staged_count,
            "bytes": total,
        }

    @app.post("/api/jobs/ingest")
    async def ingest(body: IngestBody) -> dict[str, Any]:
        corpus = state.roots.resolve(body.corpus)
        if corpus == state.roots.root(
            body.corpus.partition(":")[0]
        ).path and body.corpus.startswith("workspace:"):
            raise PathRefused(
                "pick a folder inside the workspace, not the workspace itself"
            )
        job = await state.jobs.start_ingest(
            corpus=corpus,
            corpus_ref=body.corpus,
            methodologies=body.methodologies,
            index_value=body.index_value,
            force_rebuild=body.force_rebuild,
            reset=body.reset,
            confirm_reset=body.confirm_reset,
            confirm_deletions=body.confirm_deletions,
            confirm_integrity=body.confirm_integrity,
            confirm_narrowing=body.confirm_narrowing,
        )
        return state.jobs.get(job.id)

    @app.post("/api/jobs/evaluate")
    async def evaluate(body: EvaluateBody) -> dict[str, Any]:
        dataset = state.roots.resolve(body.dataset)
        job = await state.jobs.start_evaluate(
            dataset=dataset,
            dataset_ref=body.dataset,
            strategies=body.strategies,
            repeats=body.repeats,
            closed_book=body.closed_book,
            pairwise_judge=body.pairwise_judge,
            fail_if=body.fail_if,
            max_failure_rate=body.max_failure_rate,
        )
        return state.jobs.get(job.id)

    @app.get("/api/jobs")
    async def jobs() -> dict[str, Any]:
        return {"jobs": state.jobs.all_jobs()}

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str) -> dict[str, Any]:
        try:
            return state.jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, f"no job '{job_id}'") from None

    @app.post("/api/jobs/{job_id}/stop")
    async def stop(job_id: str) -> dict[str, Any]:
        try:
            await state.jobs.stop(job_id)
        except KeyError:
            raise HTTPException(404, f"no running job '{job_id}'") from None
        return state.jobs.get(job_id)

    # -- evaluation files ---------------------------------------------------

    @app.get("/api/datasets")
    async def datasets() -> dict[str, Any]:
        return {
            "datasets": await asyncio.to_thread(evaluations.find_datasets, state.roots)
        }

    @app.get("/api/datasets/preview")
    async def dataset_preview(ref: str) -> dict[str, Any]:
        path = state.roots.resolve(ref)
        if not await asyncio.to_thread(path.is_file):
            raise HTTPException(404, f"no dataset '{ref}'")
        return await asyncio.to_thread(evaluations.preview_dataset, path)

    @app.get("/api/evaluations")
    async def runs() -> dict[str, Any]:
        return {"runs": await asyncio.to_thread(evaluations.find_runs, state.roots)}

    @app.get("/api/evaluations/run")
    async def run(ref: str) -> dict[str, Any]:
        directory = state.roots.resolve(ref)
        if not await asyncio.to_thread(directory.is_dir):
            raise HTTPException(404, f"no saved run '{ref}'")
        return await asyncio.to_thread(evaluations.read_run, directory)

    @app.get("/api/evaluations/questions")
    async def run_questions(ref: str) -> dict[str, Any]:
        directory = state.roots.resolve(ref)
        if not await asyncio.to_thread(directory.is_dir):
            raise HTTPException(404, f"no saved run '{ref}'")
        saved = await asyncio.to_thread(evaluations.read_run, directory)
        return await asyncio.to_thread(
            evaluations.question_scores, directory, saved["comparison"]
        )

    # -- graph ------------------------------------------------------------

    @app.get("/api/graphs")
    async def graphs() -> dict[str, Any]:
        # The namespace's graph, built when the page opens and the stores
        # changed since the newest export (no job exports one); ``current``
        # says why there is none (the page's empty state).
        current = await asyncio.to_thread(state.graphs.current, state.load_config)
        names = tuple(
            n for n in ("cache", "visualization", "workspace") if n in state.roots.roots
        )
        found = await asyncio.to_thread(graph_api.find_graphs, state.roots, names)
        graph_api.mark_outdated(found, current)
        return {"graphs": found, "current": current.view()}

    async def graph_file(ref: str) -> Path:
        path = state.roots.resolve(ref)
        if path.name != VISUALIZATION_DATA_FILENAME:
            raise PathRefused(f"'{ref}' is not a visualization_data.json file")
        if not await asyncio.to_thread(path.is_file):
            raise HTTPException(404, f"no graph '{ref}'")
        return path

    @app.get("/api/graphs/data")
    async def graph_data(ref: str) -> Response:
        loaded = await asyncio.to_thread(graph_api.cached_graph, await graph_file(ref))
        return Response(loaded.encoded, media_type="application/json")

    @app.get("/api/graphs/entity")
    async def graph_entity(ref: str, id: str) -> dict[str, Any]:
        found = await asyncio.to_thread(graph_api.entity, await graph_file(ref), id)
        if found is None:
            raise HTTPException(404, f"no entity '{id}' in '{ref}'")
        return found

    # -- settings ---------------------------------------------------------

    @app.get("/api/config")
    async def config_view() -> dict[str, Any]:
        return state.settings_file.view()

    @app.get("/api/config/schema")
    async def config_schema() -> dict[str, Any]:
        return schema()

    @app.post("/api/config/preview")
    async def config_preview(body: PreviewBody) -> dict[str, Any]:
        return state.settings_file.preview(text=body.text, changes=body.changes)

    @app.post("/api/config/save")
    async def config_save(body: SaveBody) -> dict[str, Any]:
        result = state.settings_file.save(
            text=body.text,
            base_sha=body.base_sha,
            confirm_endpoints=body.confirm_endpoints,
            confirm_directories=body.confirm_directories,
        )
        state.reload()
        return result

    @app.get("/api/namespace/methodologies")
    async def namespace_methodologies(
        suffix: str | None = None, methodologies: str | None = None
    ) -> dict[str, Any]:
        """What the index namespace of ``suffix`` (default: the config's)
        holds, as its build record names it, and which of those an
        ingestion of ``methodologies`` (comma-separated; default: the
        config's) would drop: the Ingest page starts from the first and
        confirms the second."""
        config, _ = ingest_config(
            state.load_config(),
            [m for m in (methodologies or "").split(",") if m] or None,
        )
        held = await state.ask.held_methodologies(_suffix(suffix))
        # Without a build record nothing is dropped: the run writes the first.
        dropped = (
            dropped_methodologies(held["held"], config) if held["recorded"] else []
        )
        return {**held, "dropped": dropped}

    @app.get("/api/methodologies")
    async def methodologies() -> dict[str, Any]:
        config = state.try_config()
        return {
            "all": [m.value for m in Methodology],
            "selected": (
                [m.value for m in config.indexing.methodologies] if config else []
            ),
            "effective": (
                [m.value for m in config.effective_methodologies] if config else []
            ),
        }

    _mount_web(app, options.web_directory)
    return app


def _suffix(value: str | None) -> str | None:
    """``value`` if it is a safe index namespace name.

    Raises:
        InvalidArgumentError: It is not.
    """
    if value is None:
        return None
    try:
        return validate_safe_name(value, "suffix")
    except ValueError as e:
        raise InvalidArgumentError(str(e)) from e


def _models(config: Config) -> dict[str, str | None]:
    roles: dict[str, str | None] = {}
    for role in (
        ModelRole.ANSWER_GENERATION,
        ModelRole.ENTITY_EXTRACTION,
        ModelRole.GRAPH_EXTRACTION,
    ):
        try:
            roles[role.value] = config.model_id(role)
        except Exception:  # noqa: BLE001 - shown as unknown
            roles[role.value] = None
    roles["embedding"] = (
        config.models.embedding.model_id if config.models.embedding else None
    )
    return roles


def _corpora(roots: Roots) -> list[dict[str, Any]]:
    """The folders that hold documents: the uploads, and the workspace's
    folders (two levels down) outside the console's own directories."""
    extensions = {e.lower() for e in ParserFactory.get_supported_extensions()}
    candidates: list[Path] = []
    workspace = roots.root("workspace").path
    for child in sorted(workspace.iterdir()) if workspace.is_dir() else []:
        if (
            child.is_dir()
            and not child.is_symlink()
            and not child.name.startswith(".")
            and child.name not in _NOT_CORPORA
        ):
            if child.name == "uploads":
                candidates.extend(
                    sorted(
                        p for p in child.iterdir() if p.is_dir() and not p.is_symlink()
                    )
                )
            else:
                candidates.append(child)
    found = []
    for directory in candidates:
        files = walk_files(directory, limit=20000)
        supported = [f for f in files if f.suffix.lower() in extensions]
        if not supported:
            continue
        ref = roots.ref(directory)
        if ref is None:
            continue
        counts: dict[str, int] = {}
        for f in supported:
            counts[f.suffix.lower()] = counts.get(f.suffix.lower(), 0) + 1
        found.append(
            {
                "ref": ref,
                "name": directory.relative_to(workspace).as_posix(),
                "files": len(supported),
                "unsupported": len(files) - len(supported),
                "bytes": sum(f.stat().st_size for f in supported),
                "extensions": counts,
                "modified": max(f.stat().st_mtime for f in supported),
            }
        )
    return found


_NOT_BUILT = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Console not built</title><style>body{{font:16px/1.5 system-ui,sans-serif;
max-width:40rem;margin:4rem auto;padding:0 1rem;color:#17212B;background:#EEF2F3}}
code{{background:#fff;padding:.1rem .3rem;border-radius:4px}}</style></head><body>
<h1>The console's web client is not built</h1>
<p>The server runs (its API is at <code>/api/status</code>), but
<code>{path}</code> holds no build. Build it once, then reload:</p>
<pre><code>cd console/web
npm ci
npm run build</code></pre>
<p>Or set <code>UKR_CONSOLE_WEB_DIRECTORY</code> to a built copy (a wheel
install carries none: build it in a source checkout).</p>
</body></html>"""


def _mount_web(app: FastAPI, directory: Path | None) -> None:
    """Serve the built client (``index.html`` for every page route), or a
    page saying how to build it."""
    index = directory / "index.html" if directory else None

    @app.get("/{path:path}", include_in_schema=False)
    async def web(path: str) -> Response:
        if path.startswith("api/"):
            raise HTTPException(404, f"no API '{path}'")
        if directory is None or index is None or not index.is_file():
            return HTMLResponse(
                _NOT_BUILT.format(path=directory or "console/web/dist"), status_code=503
            )
        if path:
            try:
                file = contained(directory, path)
            except PathRefused:
                raise HTTPException(404) from None
            if file.is_file():
                cache = (
                    "public, max-age=31536000, immutable"
                    if "/assets/" in f"/{path}"
                    else "no-cache"
                )
                return FileResponse(file, headers={"Cache-Control": cache})
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
