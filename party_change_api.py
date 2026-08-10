"""Small HTTP API for storing and querying detected party-change events."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


DEFAULT_DB_PATH = Path("runtime_data/party_changes.sqlite3")
EVENT_FIELDS = (
    "event_id",
    "node_id",
    "from_party",
    "to_party",
    "previous_slot",
    "current_slot",
    "slot_mode",
    "topic",
    "detected_at",
)
REQUIRED_FIELDS = set(EVENT_FIELDS)
QUERY_FIELDS = {
    "node_id",
    "from_party",
    "to_party",
    "slot_mode",
    "topic",
}


class PartyChangeStore:
    def __init__(self, db_path: str | Path, retention_days: int = 365,
                 max_events: int = 100_000):
        self.db_path = Path(db_path)
        self.retention_days = max(0, retention_days)
        self.max_events = max(0, max_events)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    def initialize(self):
        with closing(self.connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS party_change_events (
                    event_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    from_party TEXT NOT NULL,
                    to_party TEXT NOT NULL,
                    previous_slot TEXT NOT NULL,
                    current_slot TEXT NOT NULL,
                    slot_mode TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    received_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_party_changes_node
                    ON party_change_events(node_id);
                CREATE INDEX IF NOT EXISTS idx_party_changes_topic_slot
                    ON party_change_events(topic, current_slot);
                CREATE INDEX IF NOT EXISTS idx_party_changes_detected
                    ON party_change_events(detected_at DESC);
                """
            )
            connection.commit()

    def upsert_many(self, events: list[dict[str, Any]]) -> int:
        received_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [
            tuple(str(event[field]) for field in EVENT_FIELDS)
            + (received_at,)
            for event in events
        ]
        columns = EVENT_FIELDS
        placeholders = ", ".join("?" for _ in range(len(columns) + 1))
        updates = ", ".join(
            f"{column}=excluded.{column}"
            for column in columns
            if column != "event_id"
        )
        sql = (
            "INSERT INTO party_change_events "
            f"({', '.join(columns)}, received_at) VALUES ({placeholders}) "
            f"ON CONFLICT(event_id) DO UPDATE SET {updates}"
        )
        with closing(self.connect()) as connection:
            connection.executemany(sql, rows)
            self._rotate(connection)
            connection.commit()
        return len(rows)

    def _rotate(self, connection):
        if self.retention_days:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            ).isoformat(timespec="seconds")
            connection.execute(
                "DELETE FROM party_change_events WHERE detected_at < ?",
                (cutoff,),
            )
        if self.max_events:
            connection.execute(
                """
                DELETE FROM party_change_events
                WHERE event_id IN (
                    SELECT event_id FROM party_change_events
                    ORDER BY detected_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_events,),
            )

    def get(self, event_id: str):
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM party_change_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def list(self, filters: dict[str, str], limit: int, offset: int):
        clauses = []
        values: list[Any] = []
        for field, value in filters.items():
            if field in QUERY_FIELDS and value:
                clauses.append(f"{field} = ?")
                values.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self.connect()) as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM party_change_events {where}",
                values,
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT * FROM party_change_events
                {where}
                ORDER BY detected_at DESC, event_id
                LIMIT ? OFFSET ?
                """,
                [*values, limit, offset],
            ).fetchall()
        return total, [dict(row) for row in rows]


def validate_events(payload):
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list) or not events:
        raise ValueError("'events' must be a non-empty array")
    if len(events) > 1000:
        raise ValueError("A batch can contain at most 1000 events")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"events[{index}] must be an object")
        missing = sorted(REQUIRED_FIELDS - event.keys())
        if missing:
            raise ValueError(
                f"events[{index}] is missing: {', '.join(missing)}"
            )
        if event["from_party"] == event["to_party"]:
            raise ValueError(
                f"events[{index}] does not contain a party transition"
            )
    return events


def build_handler(store: PartyChangeStore, api_token: str = ""):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PartyChangeAPI/1.0"

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if not api_token:
                return True
            return self.headers.get("Authorization") == f"Bearer {api_token}"

        def do_POST(self):
            if self.path != "/api/v1/party-changes":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self.authorized():
                self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    raise ValueError("request body size is invalid")
                payload = json.loads(self.rfile.read(length))
                events = validate_events(payload)
                count = store.upsert_many(events)
                self.send_json(
                    HTTPStatus.OK,
                    {"accepted": count, "status": "ok"},
                )
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except sqlite3.Error:
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "database operation failed"},
                )

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if not self.authorized():
                self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            prefix = "/api/v1/party-changes/"
            if parsed.path.startswith(prefix):
                event = store.get(unquote(parsed.path[len(prefix):]))
                if event:
                    self.send_json(HTTPStatus.OK, event)
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if parsed.path != "/api/v1/party-changes":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            params = parse_qs(parsed.query)
            try:
                limit = min(500, max(1, int(params.get("limit", ["100"])[0])))
                offset = max(0, int(params.get("offset", ["0"])[0]))
            except ValueError:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "limit and offset must be integers"},
                )
                return
            filters = {
                field: params.get(field, [""])[0]
                for field in QUERY_FIELDS
            }
            total, events = store.list(filters, limit, offset)
            self.send_json(
                HTTPStatus.OK,
                {
                    "events": events,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                },
            )

        def log_message(self, message, *args):
            print(f"[party-change-api] {self.address_string()} - {message % args}")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("PARTY_CHANGE_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PARTY_CHANGE_API_PORT", "8765")))
    parser.add_argument("--db", default=os.getenv("PARTY_CHANGE_DB_PATH", str(DEFAULT_DB_PATH)))
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.getenv("PARTY_CHANGE_RETENTION_DAYS", "365")),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=int(os.getenv("PARTY_CHANGE_MAX_EVENTS", "100000")),
    )
    args = parser.parse_args()
    store = PartyChangeStore(args.db, args.retention_days, args.max_events)
    token = (os.getenv("PARTY_CHANGE_API_TOKEN") or "").strip()
    server = ThreadingHTTPServer((args.host, args.port), build_handler(store, token))
    print(
        f"Party-change API listening on http://{args.host}:{args.port}; "
        f"database={store.db_path}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
