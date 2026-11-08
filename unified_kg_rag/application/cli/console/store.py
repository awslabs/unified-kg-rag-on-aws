# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console's own records: the questions asked through it and the jobs
it ran, in one SQLite file under its cache directory (``console.sqlite3``).

The dashboard reads it. It holds the question text (the Ask page lists
recent questions), never an answer or a source; delete the file, or use
the dashboard's Clear, to forget them.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    question TEXT NOT NULL,
    requested_strategy TEXT NOT NULL,
    strategy TEXT,
    suffix TEXT,
    latency_seconds REAL,
    outcome TEXT NOT NULL,
    abstained INTEGER NOT NULL DEFAULT 0,
    rerank_status TEXT,
    usage TEXT,
    cost_usd REAL,
    trace_id TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS queries_created ON queries (created);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    namespace TEXT NOT NULL,
    created REAL NOT NULL,
    record TEXT NOT NULL
);
"""

QUERY_COLUMNS = (
    "id",
    "created",
    "question",
    "requested_strategy",
    "strategy",
    "suffix",
    "latency_seconds",
    "outcome",
    "abstained",
    "rerank_status",
    "usage",
    "cost_usd",
    "trace_id",
    "error",
)


class ConsoleStore:
    """Thread-safe access to ``console.sqlite3`` (one connection, one lock:
    the console writes a row per question and per job update)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def record_query(self, row: dict[str, Any]) -> None:
        values = {name: row.get(name) for name in QUERY_COLUMNS}
        values["usage"] = json.dumps(row["usage"]) if row.get("usage") else None
        values["abstained"] = int(bool(row.get("abstained")))
        values.setdefault("created", time.time())
        if values["created"] is None:
            values["created"] = time.time()
        with self._lock:
            self._db.execute(
                f"INSERT OR REPLACE INTO queries ({', '.join(QUERY_COLUMNS)}) "  # nosec B608 - fixed column names
                f"VALUES ({', '.join('?' for _ in QUERY_COLUMNS)})",
                [values[name] for name in QUERY_COLUMNS],
            )
            self._db.commit()

    def queries(self, *, since: float = 0.0, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM queries WHERE created >= ? ORDER BY created DESC LIMIT ?",
                (since, limit),
            ).fetchall()
        return [_query_row(row) for row in rows]

    def clear_queries(self) -> int:
        with self._lock:
            count = self._db.execute("DELETE FROM queries").rowcount
            self._db.commit()
        return int(count)

    def save_job(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO jobs (id, kind, namespace, created, record) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record["id"],
                    record["kind"],
                    record["namespace"],
                    record["created"],
                    json.dumps(record, default=str),
                ),
            )
            self._db.commit()

    def jobs(
        self,
        *,
        limit: int | None = 200,
        since: float | None = None,
        statuses: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        """The newest job records (``limit``; ``None``: all), created at or
        after ``since`` and in one of ``statuses`` when given."""
        where = ["created >= ?"]
        params: list[float | str] = [since or 0.0]
        if statuses is not None:
            marks = ", ".join("?" for _ in statuses)
            where.append(f"json_extract(record, '$.status') IN ({marks})")
            params.extend(statuses)
        with self._lock:
            rows = self._db.execute(
                f"SELECT record FROM jobs WHERE {' AND '.join(where)} "  # nosec B608 - fixed clauses, values bound
                "ORDER BY created DESC LIMIT ?",
                (*params, -1 if limit is None else limit),
            ).fetchall()
        return [json.loads(row["record"]) for row in rows]

    def job(self, job_id: str) -> dict[str, Any] | None:
        """The record of job ``job_id``, or ``None``."""
        with self._lock:
            row = self._db.execute(
                "SELECT record FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return json.loads(row["record"]) if row else None


def _query_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["usage"] = json.loads(data["usage"]) if data.get("usage") else None
    data["abstained"] = bool(data.get("abstained"))
    return data
