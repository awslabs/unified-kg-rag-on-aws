# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SQLite plumbing of the embedded stores (``adapters.local.sqlite_util``)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from unified_kg_rag.adapters.local.sqlite_util import connect, is_locked, typed_errors
from unified_kg_rag.shared.exceptions import StoreConfigurationError, StoreTimeoutError

pytestmark = pytest.mark.unit


def test_connect_sets_wal_busy_timeout_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "db.sqlite3"
    connection = connect(
        str(path), timeout_seconds=1.5, schema=["CREATE TABLE t (x TEXT)"]
    )
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1500
    assert connection.execute("SELECT count(*) FROM t").fetchone()[0] == 0
    connection.close()


def test_connect_closes_the_connection_when_the_schema_fails(tmp_path: Path) -> None:
    opened: list[sqlite3.Connection] = []

    def migrate(connection: sqlite3.Connection) -> None:
        opened.append(connection)
        raise sqlite3.OperationalError("cannot migrate")

    with pytest.raises(sqlite3.OperationalError):
        connect(str(tmp_path / "db.sqlite3"), timeout_seconds=1, migrate=migrate)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_lock_errors_are_told_apart(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    waiter = sqlite3.connect(path, timeout=0.05)
    with pytest.raises(sqlite3.OperationalError) as locked:
        waiter.execute("CREATE TABLE t (x TEXT)")
    holder.close()
    waiter.close()
    assert is_locked(locked.value)
    assert not is_locked(sqlite3.OperationalError("no such table: t"))


def test_typed_errors_maps_and_passes_typed_errors_through() -> None:
    def errors() -> object:
        return typed_errors(
            lambda e: StoreTimeoutError(f"locked: {e}"),
            lambda e: StoreConfigurationError(f"failed: {e}"),
        )

    with pytest.raises(StoreTimeoutError, match="locked"), errors():
        raise sqlite3.OperationalError("database is locked")
    with pytest.raises(StoreConfigurationError, match="failed"), errors():
        raise sqlite3.DatabaseError("file is not a database")
    with pytest.raises(StoreConfigurationError, match="failed"), errors():
        raise PermissionError("read-only")
    # A nested block's StoreTimeoutError (an OSError) is not wrapped again.
    with pytest.raises(StoreTimeoutError, match="inner"), errors():
        raise StoreTimeoutError("inner")
