# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SQLite plumbing the embedded stores share.

The graph store, the search engine and the doc-status registry each keep one
connection per database file behind a lock, in WAL mode, waiting up to
``search.store_timeout_seconds`` for another writer's lock. This module opens
that connection, runs their transactions and tells a lock timeout apart from
other SQLite errors; each store words its own typed errors.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from unified_kg_rag.shared import GraphRAGException

# SQLITE_BUSY / SQLITE_LOCKED (the primary codes of their extended ones).
_LOCKED_CODES = {5, 6}


def is_locked(error: sqlite3.Error) -> bool:
    """Whether ``error`` is another connection holding the lock."""
    code = getattr(error, "sqlite_errorcode", None)
    if code is not None:
        return (int(code) & 0xFF) in _LOCKED_CODES
    message = str(error).lower()
    return "locked" in message or "busy" in message


def connect(
    path: str,
    *,
    timeout_seconds: float,
    schema: Iterable[str] = (),
    migrate: Callable[[sqlite3.Connection], None] | None = None,
) -> sqlite3.Connection:
    """A connection to ``path`` (created with its directory; ``":memory:"``
    for a private in-memory one) in autocommit and WAL mode, with ``schema``
    applied and then ``migrate``; closed again when either fails.

    Statements wait ``timeout_seconds`` for another connection's lock.
    """
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        path,
        timeout=timeout_seconds,
        check_same_thread=False,
        isolation_level=None,
    )
    try:
        connection.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
        connection.execute("PRAGMA journal_mode=WAL")
        for statement in schema:
            connection.execute(statement)
        if migrate is not None:
            migrate(connection)
    except BaseException:
        connection.close()
        raise
    return connection


@contextmanager
def typed_errors(
    locked: Callable[[sqlite3.Error], Exception],
    failed: Callable[[Exception], Exception],
) -> Iterator[None]:
    """Re-raise a lock timeout as ``locked(error)`` and any other SQLite or
    file error as ``failed(error)``; a framework error passes unchanged (a
    nested block already typed it)."""
    try:
        yield
    except GraphRAGException:
        # StoreTimeoutError is an OSError: the OSError branch would wrap it.
        raise
    except sqlite3.Error as e:
        raise (locked(e) if is_locked(e) else failed(e)) from e
    except OSError as e:
        raise failed(e) from e


@contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """A write transaction, rolled back when the block raises."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


@contextmanager
def read_snapshot(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One read transaction: every statement of the block reads the same
    snapshot (in WAL mode a reader does not block the writer). Inside an open
    transaction, that transaction is the snapshot."""
    own = not connection.in_transaction
    if own:
        connection.execute("BEGIN")
    try:
        yield connection
    finally:
        if own and connection.in_transaction:
            connection.execute("ROLLBACK")
