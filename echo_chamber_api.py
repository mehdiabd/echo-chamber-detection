"""HTTP API for pipeline runs, reports, dashboards, and human review.

Party-change storage is reused from party_change_api.py and served on the
same process so a single service covers all implemented endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from cleanup_root import VISUALIZATION_RE
from evaluate_echo_labels import _parse_binary_label, evaluate_label_rows
from party_change_api import (
    DEFAULT_DB_PATH as DEFAULT_PARTY_DB_PATH,
    PartyChangeStore,
    QUERY_FIELDS as PARTY_QUERY_FIELDS,
    validate_events,
)
from sample_echo_review import _rows_for_method, _sample_balanced


DEFAULT_DB_PATH = Path("runtime_data/echo_chamber.sqlite3")
DEFAULT_REPORT_GLOB = "communities/hybrid_report_*_to_*.json"
VALID_SLOT_MODES = ("hourly", "daily", "weekly", "monthly", "quarterly")
VALID_METHODS = ("hybrid", "louvain", "both")
STATIC_ALLOW = {
    "timeline_dashboard.html",
    "community_names.txt",
}
MEMBER_LIMIT = 500
LOG_TAIL_LINES = 80
LOG_TAIL_BYTES = 96_000
STOPPABLE_STATUSES = ("queued", "running")
PRELOADED_TOPICS = (
    {"label": "جنگ جمهوری اسلامی و آمریکا", "query": "جنگ جمهوری اسلامی و آمریکا"},
    {"label": "#همکاری_ملی", "query": "#همکاری_ملی"},
    {"label": "#اعتراضات_سراسری", "query": "#اعتراضات_سراسری"},
    {"label": "#معیشت", "query": "معیشت OR گرانی OR تورم"},
    {"label": "#بازنشستگان", "query": "بازنشستگان OR حقوق_بازنشستگان"},
    {"label": "#دانشجویان", "query": "دانشجو OR دانشجویان OR دانشگاه"},
    {"label": "#انتخابات", "query": "انتخابات OR رای_گیری"},
    {"label": "#تحریم", "query": "تحریم OR sanctions"},
    {"label": "#مهاجرت", "query": "مهاجرت OR خروج_از_کشور"},
    {"label": "#محیط_زیست", "query": "محیط زیست OR آلودگی هوا OR گرد و خاک"},
    {"label": "#سلامت", "query": "سلامت OR بهداشت OR درمان"},
)
PROGRESS_STEPS = (
    ("queued", "در صف"),
    ("fetch", "دریافت از الستیک"),
    ("detect", "تشخیص جوامع"),
    ("ready", "آماده‌سازی داشبورد"),
)
SLOT_PROGRESS_RE = re.compile(
    r"\[slot\s+(\d+)/(\d+)\s*\|\s*([^\]]+)\](?:\s+Processing:\s*(\S+)\s+to\s+(\S+))?"
)
SCAN_PROGRESS_RE = re.compile(
    r"\[scan-progress\]\s+(\w+)\s+docs=(\d+)(?:\s+total=(\d+))?(?:\s+elapsed_s=(\d+))?"
)
COUNT_PROGRESS_RE = re.compile(r"\[count\]\s+query_and_date:\s+(\d+)")

PipelineExecutor = Callable[[Path, dict[str, Any], Path, str], dict[str, Any]]


class PipelineCancelled(Exception):
    """Raised when a pipeline run is stopped by the user."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def safe_basename(value: str) -> str:
    name = unquote(str(value or "")).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("invalid name")
    return name


def parse_limit_offset(params: dict[str, list[str]], default_limit: int = 100,
                       max_limit: int = 500) -> tuple[int, int]:
    try:
        limit = min(max_limit, max(1, int(params.get("limit", [str(default_limit)])[0])))
        offset = max(0, int(params.get("offset", ["0"])[0]))
    except ValueError as exc:
        raise ValueError("limit and offset must be integers") from exc
    return limit, offset


def parse_bool_query(raw: str | None) -> bool | None:
    if raw is None or raw == "":
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("boolean query parameter is invalid")


def normalize_slot_modes(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        modes = [item.strip() for item in re.split(r"[,،;\s]+", value) if item.strip()]
    elif isinstance(value, list):
        modes = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError("slot_modes must be a string or array")
    invalid = [mode for mode in modes if mode not in VALID_SLOT_MODES]
    if invalid:
        raise ValueError(f"unsupported slot_modes: {', '.join(invalid)}")
    if not modes:
        raise ValueError("slot_modes must not be empty")
    return modes


def parse_iso_date(value: Any, field: str) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD") from exc
    return text[:10]


def topic_key(label: str) -> str:
    clean = (label or "").strip()
    if not clean:
        return "__topic__"
    key = re.sub(r"\s+", "_", clean)
    key = re.sub(r"[^\w#آ-ی\u200c_-]+", "_", key, flags=re.UNICODE).strip("_")
    return key or "__topic__"


def parse_topic_labels(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,،;]+", value) if item.strip()]
    if not isinstance(value, list):
        return []
    labels = []
    for item in value:
        if isinstance(item, str) and item.strip():
            labels.append(item.strip())
        elif isinstance(item, dict):
            label = str(item.get("label") or item.get("topic_label") or item.get("key") or "").strip()
            if label:
                labels.append(label)
    return labels


def preloaded_topic_map() -> dict[str, dict[str, str]]:
    return {topic_key(item["label"]): dict(item) for item in PRELOADED_TOPICS}


def resolve_topic_selection(payload: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    raw_topics = payload.get("topics")
    if isinstance(raw_topics, list):
        source_items = raw_topics
    else:
        source_items = parse_topic_labels(
            payload.get("topic_labels") or payload.get("topic_label")
        )
    presets = preloaded_topic_map()
    for item in source_items:
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("topic_label") or item.get("key") or "").strip()
            query = str(item.get("query") or item.get("topic_query") or "").strip()
        else:
            label = str(item).strip()
            query = ""
        if not label:
            continue
        key = topic_key(label)
        if key in seen:
            continue
        preset = presets.get(key)
        items.append(
            {
                "key": key,
                "label": str(preset["label"]) if preset else label,
                "query": query or (str(preset["query"]) if preset else label),
            }
        )
        seen.add(key)
    return items


def config_topic_keys(config: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for item in config.get("topics") or []:
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("key") or "").strip()
            if label:
                keys.add(topic_key(label))
    for part in parse_topic_labels(config.get("topic_label")):
        keys.add(topic_key(part))
    keys.discard("__topic__")
    return keys


def topics_are_covered(requested: list[dict[str, str]], config: dict[str, Any]) -> bool:
    wanted = {item["key"] for item in requested if item.get("key")}
    if not wanted:
        return True
    return wanted <= config_topic_keys(config)


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_remaining_label(seconds: int | None, *, calculating: bool = False) -> str | None:
    if seconds is None:
        return "در حال محاسبه" if calculating else None
    if seconds <= 0:
        return "کمتر از یک دقیقه"
    if seconds < 60:
        return "کمتر از یک دقیقه"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"حدود {minutes} دقیقه"
    hours, rest = divmod(minutes, 60)
    if rest == 0:
        return f"حدود {hours} ساعت"
    return f"حدود {hours} ساعت و {rest} دقیقه"


def run_actions(status: str) -> dict[str, bool]:
    active = status in STOPPABLE_STATUSES
    return {"can_stop": active, "can_restart": True}


def read_text_tail(path: Path, max_bytes: int = LOG_TAIL_BYTES) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    if not data:
        return ""
    if size > max_bytes:
        newline = data.find("\n")
        if newline >= 0:
            data = data[newline + 1 :]
    return data


def read_log_tail_lines(path: Path | None, max_lines: int = LOG_TAIL_LINES) -> list[str]:
    if not path:
        return []
    log_file = Path(path)
    if not log_file.is_file():
        return []
    text = read_text_tail(log_file)
    if not text:
        return []
    return text.splitlines()[-max_lines:]


def read_pid_file(path: Path | None) -> int | None:
    if not path:
        return None
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
        pid = int(raw)
    except (OSError, TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def terminate_pid(pid: int | None, process: subprocess.Popen[Any] | None = None) -> None:
    if process is not None and process.poll() is not None:
        process = None
    target = pid or (process.pid if process is not None else None)
    if not target:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(target, sig)
        except (ProcessLookupError, PermissionError, OSError):
            if process is not None:
                try:
                    process.send_signal(sig)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        if process is not None and process.poll() is not None:
            return


def progress_payload(
    status: str,
    percent: int = 0,
    stage: str = "",
    message: str = "",
    remaining_seconds: int | None = None,
    elapsed_seconds: int | None = None,
    docs_done: int | None = None,
    docs_total: int | None = None,
) -> dict[str, Any]:
    status = str(status or "queued")
    calculating = False
    if status == "queued":
        percent, stage, message = 0, "queued", message or "در صف"
        remaining_seconds = None
        calculating = True
    elif status == "done":
        percent, stage, message = 100, "ready", message or "تکمیل شد"
        remaining_seconds = 0
    elif status == "failed":
        percent = max(0, min(100, int(percent)))
        stage = stage or "fetch"
        message = message or "خطا در اجرا"
        remaining_seconds = None
    elif status == "cancelled":
        percent = max(0, min(100, int(percent)))
        stage = stage or "fetch"
        message = message or "متوقف شد"
        remaining_seconds = None
    else:
        percent = max(0, min(100, int(percent)))
        stage = stage or "fetch"
        message = message or "در حال اجرا"
        calculating = remaining_seconds is None
    reached = False
    steps = []
    for step_id, label in PROGRESS_STEPS:
        if status == "done":
            state = "done"
        elif step_id == stage:
            if status == "failed":
                state = "failed"
            elif status == "cancelled":
                state = "cancelled"
            else:
                state = "done" if percent >= 100 else "running"
            reached = True
        elif not reached:
            state = "done"
        else:
            state = "pending"
        steps.append({"id": step_id, "label": label, "status": state})
    payload: dict[str, Any] = {
        "percent": percent,
        "remaining_percent": max(0, 100 - percent),
        "remaining_seconds": remaining_seconds,
        "remaining_label": format_remaining_label(
            remaining_seconds, calculating=calculating and status in STOPPABLE_STATUSES
        ),
        "elapsed_seconds": elapsed_seconds,
        "stage": stage,
        "message": message,
        "steps": steps,
    }
    if docs_done is not None:
        payload["docs_done"] = docs_done
    if docs_total is not None:
        payload["docs_total"] = docs_total
    return payload


def write_progress(log_file: Path, percent: int, stage: str, message: str) -> None:
    path = Path(log_file).with_suffix(".progress.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "percent": percent,
                "stage": stage,
                "message": message,
                "updated_at": utc_now(),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def infer_live_progress(log_text: str) -> dict[str, Any]:
    percent = 0
    stage = "queued"
    message = ""
    extra: dict[str, Any] = {}
    if "elastic.py" in log_text:
        percent, stage, message = 15, "fetch", "در حال دریافت از الستیک"
        last_count = None
        for match in COUNT_PROGRESS_RE.finditer(log_text):
            last_count = match
        if last_count:
            extra["docs_total"] = int(last_count.group(1))
            message = f"آماده‌سازی اسکن — {extra['docs_total']} سند"
            percent = max(percent, 18)
        if "Starting initial scan" in log_text:
            percent, message = 20, "اسکن اولیه الستیک"
        last_scan = None
        for match in SCAN_PROGRESS_RE.finditer(log_text):
            last_scan = match
        if last_scan:
            label = last_scan.group(1)
            docs = int(last_scan.group(2))
            total = int(last_scan.group(3)) if last_scan.group(3) else extra.get("docs_total")
            elapsed_s = int(last_scan.group(4)) if last_scan.group(4) else None
            extra["docs_done"] = docs
            extra["scan_label"] = label
            extra["scan_elapsed_seconds"] = elapsed_s
            if total:
                extra["docs_total"] = total
            total_label = f"{docs}/{total}" if total else str(docs)
            if label == "interaction_scan":
                if total:
                    percent = 35 + int(15 * min(1.0, docs / max(1, total)))
                else:
                    percent = 38
                message = f"جمع‌آوری تعاملات از الستیک — {total_label} سند"
            else:
                if total:
                    percent = 20 + int(12 * min(1.0, docs / max(1, total)))
                else:
                    percent = 25
                message = f"اسکن اولیه الستیک — {total_label} سند"
        if "Total hits:" in log_text:
            percent, message = max(percent, 32), "جمع‌آوری تعاملات از الستیک"
        if "Starting scan to collect interactions" in log_text:
            percent, message = max(percent, 35), "جمع‌آوری تعاملات از الستیک"
        if "Finished full pipeline." in log_text:
            percent, message = 50, "دریافت الستیک تمام شد"
            extra.pop("scan_elapsed_seconds", None)
    if "community_detection.py" in log_text:
        percent, stage, message = max(percent, 60), "detect", "در حال تشخیص جوامع"
        last_slot = None
        for match in SLOT_PROGRESS_RE.finditer(log_text):
            last_slot = match
        if last_slot:
            current = int(last_slot.group(1))
            total = max(1, int(last_slot.group(2)))
            mode = last_slot.group(3).strip()
            start = (last_slot.group(4) or "").rstrip(".")
            percent = 60 + int(35 * current / total)
            extra["slots_done"] = current
            extra["slots_total"] = total
            message = f"تشخیص جوامع: اسلات {current} از {total} ({mode})"
            if start:
                message += f" — {start}"
        if "[done] Community detection completed." in log_text:
            percent, message = 95, "آماده‌سازی داشبورد"
    extra.update({"percent": percent, "stage": stage, "message": message})
    return extra


def estimate_remaining_seconds(
    status: str,
    percent: int,
    started_at: str | None,
    finished_at: str | None = None,
    docs_done: int | None = None,
    docs_total: int | None = None,
    scan_label: str = "",
    scan_elapsed_seconds: int | None = None,
    slots_done: int | None = None,
    slots_total: int | None = None,
) -> tuple[int | None, int | None]:
    started = parse_utc(started_at)
    ended = parse_utc(finished_at) or datetime.now(timezone.utc)
    elapsed = max(0, int((ended - started).total_seconds())) if started else None
    if status == "done":
        return 0, elapsed
    if status not in STOPPABLE_STATUSES:
        return None, elapsed
    remaining = None
    if docs_total and docs_done and (scan_elapsed_seconds or 0) > 0:
        if scan_label == "interaction_scan":
            remaining_docs = max(0, docs_total - docs_done)
        else:
            remaining_docs = max(0, docs_total - docs_done) + docs_total
        rate = docs_done / max(1, scan_elapsed_seconds or 0)
        if rate > 0:
            remaining = int(remaining_docs / rate)
    elif slots_total and slots_done and elapsed:
        remaining = int(elapsed * max(0, slots_total - slots_done) / max(1, slots_done))
    elif percent > 15 and elapsed and elapsed >= 20:
        remaining = int(elapsed * (100 - percent) / percent)
    return remaining, elapsed


def command_failure_message(log_file: Path, command: list[str], returncode: int) -> str:
    prefix = f"command failed ({returncode}): {' '.join(command)}"
    try:
        lines = Path(log_file).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return prefix
    for line in reversed(lines):
        text = line.strip()
        if not text or text.startswith("$ ") or text.startswith("[error]"):
            continue
        if "Error" in text or "Exception" in text:
            return f"{prefix}: {text}"
    return prefix


def read_progress(
    log_path: str | Path | None,
    status: str,
    error: str = "",
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    percent = 0
    stage = "queued"
    message = ""
    extra: dict[str, Any] = {}
    if log_path:
        progress_file = Path(log_path).with_suffix(".progress.json")
        if progress_file.is_file():
            try:
                loaded = json.loads(progress_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    percent = int(loaded.get("percent") or 0)
                    stage = str(loaded.get("stage") or stage)
                    message = str(loaded.get("message") or "")
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        log_file = Path(log_path)
        if log_file.is_file() and status in STOPPABLE_STATUSES:
            inferred = infer_live_progress(read_text_tail(log_file))
            inferred_percent = int(inferred.get("percent") or 0)
            if inferred_percent:
                percent = max(percent, inferred_percent)
                stage = str(inferred.get("stage") or stage)
                message = str(inferred.get("message") or message)
            extra = inferred
    if status == "failed" and error:
        message = error
    remaining_seconds, elapsed_seconds = estimate_remaining_seconds(
        status,
        percent,
        started_at,
        finished_at,
        docs_done=extra.get("docs_done"),
        docs_total=extra.get("docs_total"),
        scan_label=str(extra.get("scan_label") or ""),
        scan_elapsed_seconds=extra.get("scan_elapsed_seconds"),
        slots_done=extra.get("slots_done"),
        slots_total=extra.get("slots_total"),
    )
    return progress_payload(
        status,
        percent,
        stage,
        message,
        remaining_seconds=remaining_seconds,
        elapsed_seconds=elapsed_seconds,
        docs_done=extra.get("docs_done"),
        docs_total=extra.get("docs_total"),
    )


def format_slot_token(raw: str) -> str:
    if len(raw) == 8:
        dt = datetime.strptime(raw, "%y%m%d%H")
        return dt.strftime("%Y-%m-%d %H:00")
    dt = datetime.strptime(raw, "%y%m%d")
    return dt.strftime("%Y-%m-%d")


def load_party_focus(payload: dict[str, Any]) -> dict[str, Any]:
    party_focus = payload.get("partyFocus")
    if party_focus is None:
        party_focus = payload.get("party_focus")
    if not isinstance(party_focus, dict):
        return {"parties": []}
    parties = party_focus.get("parties")
    if not isinstance(parties, list):
        return {"parties": []}
    return {"parties": [item for item in parties if isinstance(item, dict)]}


def load_graph_json(path: Path | None) -> dict[str, Any]:
    """Read the nodes/edges written by the pipeline next to the hybrid graph."""
    payload: Any = None
    if path is not None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
    if not isinstance(payload, dict):
        payload = {}
    nodes = [item for item in payload.get("nodes") or [] if isinstance(item, dict)]
    edges = [item for item in payload.get("edges") or [] if isinstance(item, dict)]
    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "partyFocus": load_party_focus(payload),
    }


def slot_tokens_from_timeframe(start: str, end: str) -> tuple[str, str] | None:
    start = str(start or "").strip()
    end = str(end or "").strip()
    if not start or not end:
        return None
    try:
        start_dt = datetime.strptime(start[:16], "%Y-%m-%d %H:%M") if len(start) >= 16 else datetime.strptime(start[:10], "%Y-%m-%d")
        end_dt = datetime.strptime(end[:16], "%Y-%m-%d %H:%M") if len(end) >= 16 else datetime.strptime(end[:10], "%Y-%m-%d")
    except ValueError:
        return None
    if len(start) >= 16 or len(end) >= 16:
        return start_dt.strftime("%y%m%d%H"), end_dt.strftime("%y%m%d%H")
    return start_dt.strftime("%y%m%d"), end_dt.strftime("%y%m%d")


def method_summary(block: dict[str, Any] | None) -> dict[str, Any]:
    block = block or {}
    metrics = block.get("echo_metrics") or {}
    echo_count = 0
    for info in metrics.values():
        if isinstance(info, dict) and info.get("is_echo_chamber"):
            echo_count += 1
    return {
        "community_count": len(metrics) if isinstance(metrics, dict) else 0,
        "echo_count": echo_count,
        "modularity": block.get("modularity"),
        "silhouette": block.get("silhouette"),
    }


def lookup_metric(metrics: dict[str, Any], community_id: str) -> tuple[str, dict[str, Any]] | None:
    if community_id in metrics and isinstance(metrics[community_id], dict):
        return community_id, metrics[community_id]
    for key, info in metrics.items():
        if str(key) == str(community_id) and isinstance(info, dict):
            return str(key), info
    return None


class PipelineStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
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
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    fetch INTEGER NOT NULL,
                    detect INTEGER NOT NULL,
                    topic_label TEXT,
                    topic_query TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    slot_modes TEXT,
                    params_json TEXT NOT NULL,
                    error TEXT,
                    log_path TEXT,
                    reports_json TEXT,
                    dashboards_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_pipeline_runs_created
                    ON pipeline_runs(created_at DESC);
                """
            )
            connection.commit()

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        data["fetch"] = bool(data.get("fetch"))
        data["detect"] = bool(data.get("detect"))
        data["slot_modes"] = json_loads(data.get("slot_modes"), None)
        data["params"] = json_loads(data.get("params_json"), {})
        data["reports"] = json_loads(data.get("reports_json"), [])
        data["dashboards"] = json_loads(data.get("dashboards_json"), [])
        data.pop("params_json", None)
        data.pop("reports_json", None)
        data.pop("dashboards_json", None)
        return data

    def _with_runtime(
        self,
        data: dict[str, Any],
        log_tail: int = LOG_TAIL_LINES,
        include_log: bool = True,
    ) -> dict[str, Any]:
        log_path = data.get("log_path")
        if include_log:
            data["log_tail"] = read_log_tail_lines(log_path, log_tail)
        data["progress"] = read_progress(
            log_path,
            str(data.get("status") or ""),
            str(data.get("error") or ""),
            data.get("started_at"),
            data.get("finished_at"),
        )
        data["actions"] = run_actions(str(data.get("status") or ""))
        return data

    def create(self, params: dict[str, Any], log_path: str | Path | None = None) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        created_at = utc_now()
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO pipeline_runs (
                    run_id, status, fetch, detect, topic_label, topic_query,
                    start_date, end_date, slot_modes, params_json, log_path,
                    created_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(bool(params.get("fetch", True))),
                    int(bool(params.get("detect", True))),
                    params.get("topic_label") or None,
                    params.get("topic_query") or None,
                    params.get("start_date") or None,
                    params.get("end_date") or None,
                    json_dumps(params.get("slot_modes")) if params.get("slot_modes") else None,
                    json_dumps(params),
                    str(log_path) if log_path else None,
                    created_at,
                ),
            )
            connection.commit()
        run = self.get(run_id)
        assert run is not None
        return run

    def has_active(self) -> bool:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM pipeline_runs
                WHERE status IN ('queued', 'running')
                """
            ).fetchone()
        return bool(row[0])

    def get(self, run_id: str, log_tail: int = LOG_TAIL_LINES) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM pipeline_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        data = self._row(row)
        if not data:
            return None
        return self._with_runtime(data, log_tail=log_tail, include_log=True)

    def list(self, limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
        with closing(self.connect()) as connection:
            total = connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
            rows = connection.execute(
                """
                SELECT * FROM pipeline_runs
                ORDER BY created_at DESC, run_id
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        runs = []
        for row in rows:
            data = self._row(row)
            if data:
                runs.append(self._with_runtime(data, include_log=False))
        return total, runs

    def mark_running(self, run_id: str, log_path: str | Path | None = None):
        with closing(self.connect()) as connection:
            connection.execute(
                """
                UPDATE pipeline_runs
                SET status = 'running', started_at = ?, log_path = COALESCE(?, log_path)
                WHERE run_id = ?
                """,
                (utc_now(), str(log_path) if log_path else None, run_id),
            )
            connection.commit()

    def mark_done(self, run_id: str, artifacts: dict[str, Any] | None = None):
        artifacts = artifacts or {}
        with closing(self.connect()) as connection:
            connection.execute(
                """
                UPDATE pipeline_runs
                SET status = 'done', finished_at = ?, error = NULL,
                    reports_json = ?, dashboards_json = ?
                WHERE run_id = ? AND status IN ('queued', 'running')
                """,
                (
                    utc_now(),
                    json_dumps(artifacts.get("reports") or []),
                    json_dumps(artifacts.get("dashboards") or []),
                    run_id,
                ),
            )
            connection.commit()

    def mark_failed(self, run_id: str, error: str):
        with closing(self.connect()) as connection:
            connection.execute(
                """
                UPDATE pipeline_runs
                SET status = 'failed', finished_at = ?, error = ?
                WHERE run_id = ? AND status IN ('queued', 'running')
                """,
                (utc_now(), error[:2000], run_id),
            )
            connection.commit()

    def mark_cancelled(self, run_id: str, error: str = "stopped by user"):
        with closing(self.connect()) as connection:
            connection.execute(
                """
                UPDATE pipeline_runs
                SET status = 'cancelled', finished_at = ?, error = ?
                WHERE run_id = ? AND status IN ('queued', 'running')
                """,
                (utc_now(), error[:2000], run_id),
            )
            connection.commit()


class ReviewStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
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
                CREATE TABLE IF NOT EXISTS review_batches (
                    batch_id TEXT PRIMARY KEY,
                    report_query TEXT,
                    method TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    min_size INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_samples (
                    sample_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    report_file TEXT NOT NULL,
                    time_start TEXT,
                    time_end TEXT,
                    method TEXT NOT NULL,
                    community_id TEXT NOT NULL,
                    size INTEGER,
                    ei_index TEXT,
                    conductance TEXT,
                    content_homogeneity TEXT,
                    predicted_is_echo INTEGER NOT NULL,
                    stance_target TEXT,
                    stance_label TEXT,
                    stance_pos TEXT,
                    stance_neg TEXT,
                    stance_neu TEXT,
                    stance_total TEXT,
                    human_label INTEGER,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    labeled_at TEXT,
                    FOREIGN KEY(batch_id) REFERENCES review_batches(batch_id)
                );
                CREATE INDEX IF NOT EXISTS idx_review_samples_batch
                    ON review_samples(batch_id);
                CREATE INDEX IF NOT EXISTS idx_review_samples_label
                    ON review_samples(human_label);
                """
            )
            connection.commit()

    def _sample_row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        if data.get("predicted_is_echo") is not None:
            data["predicted_is_echo"] = int(data["predicted_is_echo"])
        return data

    def create_batch(
        self,
        rows: list[dict[str, Any]],
        report_query: str,
        method: str,
        sample_size: int,
        min_size: int,
        seed: int,
    ) -> dict[str, Any]:
        batch_id = uuid.uuid4().hex
        created_at = utc_now()
        samples = []
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO review_batches (
                    batch_id, report_query, method, sample_size, min_size, seed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (batch_id, report_query, method, sample_size, min_size, seed, created_at),
            )
            for row in rows:
                sample_id = uuid.uuid4().hex
                predicted = _parse_binary_label(row.get("predicted_is_echo"))
                if predicted is None:
                    predicted = 0
                size_raw = row.get("size")
                try:
                    size = int(size_raw) if str(size_raw).strip() else None
                except (TypeError, ValueError):
                    size = None
                connection.execute(
                    """
                    INSERT INTO review_samples (
                        sample_id, batch_id, report_file, time_start, time_end, method,
                        community_id, size, ei_index, conductance, content_homogeneity,
                        predicted_is_echo, stance_target, stance_label, stance_pos,
                        stance_neg, stance_neu, stance_total, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        batch_id,
                        str(row.get("report_file") or ""),
                        str(row.get("time_start") or ""),
                        str(row.get("time_end") or ""),
                        str(row.get("method") or method),
                        str(row.get("community_id") or ""),
                        size,
                        str(row.get("ei_index") or ""),
                        str(row.get("conductance") or ""),
                        str(row.get("content_homogeneity") or ""),
                        predicted,
                        str(row.get("stance_target") or ""),
                        str(row.get("stance_label") or ""),
                        str(row.get("stance_pos") or ""),
                        str(row.get("stance_neg") or ""),
                        str(row.get("stance_neu") or ""),
                        str(row.get("stance_total") or ""),
                        created_at,
                    ),
                )
                samples.append(sample_id)
            connection.commit()
        stored = [self.get_sample(sample_id) for sample_id in samples]
        return {
            "batch_id": batch_id,
            "count": len(stored),
            "method": method,
            "sample_size": sample_size,
            "min_size": min_size,
            "seed": seed,
            "report_query": report_query,
            "created_at": created_at,
            "samples": stored,
        }

    def get_sample(self, sample_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM review_samples WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
        return self._sample_row(row)

    def list_samples(
        self,
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> tuple[int, list[dict[str, Any]]]:
        clauses = []
        values: list[Any] = []
        if filters.get("batch_id"):
            clauses.append("batch_id = ?")
            values.append(filters["batch_id"])
        if filters.get("method"):
            clauses.append("method = ?")
            values.append(filters["method"])
        labeled = filters.get("labeled")
        if labeled is True:
            clauses.append("human_label IS NOT NULL")
        elif labeled is False:
            clauses.append("human_label IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self.connect()) as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM review_samples {where}",
                values,
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT * FROM review_samples
                {where}
                ORDER BY created_at DESC, sample_id
                LIMIT ? OFFSET ?
                """,
                [*values, limit, offset],
            ).fetchall()
        return total, [self._sample_row(row) for row in rows if self._sample_row(row)]

    def apply_labels(self, labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated = []
        labeled_at = utc_now()
        with closing(self.connect()) as connection:
            for item in labels:
                sample_id = item["sample_id"]
                existing = connection.execute(
                    "SELECT sample_id FROM review_samples WHERE sample_id = ?",
                    (sample_id,),
                ).fetchone()
                if existing is None:
                    raise KeyError(sample_id)
                connection.execute(
                    """
                    UPDATE review_samples
                    SET human_label = ?, notes = ?, labeled_at = ?
                    WHERE sample_id = ?
                    """,
                    (
                        item["human_label"],
                        item.get("notes") or "",
                        labeled_at,
                        sample_id,
                    ),
                )
                updated.append(sample_id)
            connection.commit()
        return [self.get_sample(sample_id) for sample_id in updated]

    def all_samples(self, batch_id: str | None = None) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            if batch_id:
                rows = connection.execute(
                    "SELECT * FROM review_samples WHERE batch_id = ? ORDER BY sample_id",
                    (batch_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM review_samples ORDER BY sample_id"
                ).fetchall()
        return [self._sample_row(row) for row in rows if self._sample_row(row)]

    def metrics(self, batch_id: str | None = None) -> dict[str, Any]:
        samples = self.all_samples(batch_id)
        payload = evaluate_label_rows(samples)
        payload["batch_id"] = batch_id
        payload["sample_count"] = len(samples)
        return payload


class ArtifactIndex:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def communities_dir(self) -> Path:
        return self.root / "communities"

    def pipeline_config(self) -> dict[str, Any]:
        path = self.root / "pipeline_config.json"
        if not path.exists():
            return {"exists": False, "config": {}, "path": "pipeline_config.json"}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except json.JSONDecodeError:
            data = {}
        return {"exists": True, "config": data, "path": "pipeline_config.json"}

    def current_topic(self) -> str:
        return str(self.pipeline_config().get("config", {}).get("topic_label") or "")

    def report_paths(self) -> list[Path]:
        directory = self.communities_dir()
        if not directory.is_dir():
            return []
        return sorted(directory.glob("hybrid_report_*.json"))

    def load_report(self, report_id: str) -> tuple[Path, dict[str, Any]]:
        name = safe_basename(report_id)
        if name.endswith(".json"):
            name = name[:-5]
        path = self.communities_dir() / f"{name}.json"
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(name)
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("report is not a JSON object")
        return path, data

    def summarize_report(self, path: Path, data: dict[str, Any]) -> dict[str, Any]:
        timeframe = data.get("timeframe") if isinstance(data.get("timeframe"), dict) else {}
        return {
            "id": path.stem,
            "file": str(path.relative_to(self.root)) if path.is_relative_to(self.root) else path.name,
            "topic": self.current_topic(),
            "timeframe": {
                "start": timeframe.get("start", ""),
                "end": timeframe.get("end", ""),
            },
            "hybrid": method_summary(data.get("hybrid") if isinstance(data.get("hybrid"), dict) else {}),
            "louvain": method_summary(data.get("louvain") if isinstance(data.get("louvain"), dict) else {}),
            "comparison": data.get("comparison") if isinstance(data.get("comparison"), dict) else {},
        }

    def list_reports(self, filters: dict[str, str], limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
        summaries = []
        for path in self.report_paths():
            try:
                with path.open(encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            summary = self.summarize_report(path, data)
            start = str(summary["timeframe"].get("start") or "")[:10]
            end = str(summary["timeframe"].get("end") or "")[:10]
            if filters.get("topic") and summary.get("topic") != filters["topic"]:
                continue
            if filters.get("start") and start and start < filters["start"]:
                continue
            if filters.get("end") and end and end > filters["end"]:
                continue
            summaries.append(summary)
        summaries.sort(key=lambda item: (item["timeframe"].get("start") or "", item["id"]), reverse=True)
        total = len(summaries)
        return total, summaries[offset:offset + limit]

    def get_report(self, report_id: str) -> dict[str, Any]:
        path, data = self.load_report(report_id)
        summary = self.summarize_report(path, data)
        summary["report"] = data
        return summary

    def get_community(self, report_id: str, community_id: str, method: str) -> dict[str, Any]:
        path, data = self.load_report(report_id)
        block = data.get(method) if isinstance(data.get(method), dict) else {}
        metrics = block.get("echo_metrics") if isinstance(block.get("echo_metrics"), dict) else {}
        found = lookup_metric(metrics, community_id)
        if found is None:
            raise FileNotFoundError(community_id)
        resolved_id, info = found
        timeframe = data.get("timeframe") if isinstance(data.get("timeframe"), dict) else {}
        members = self._members_for_community(timeframe, resolved_id)
        payload = {
            "report_id": path.stem,
            "community_id": resolved_id,
            "method": method,
            "timeframe": timeframe,
            "metrics": info,
            "member_count": len(members),
            "members": members[:MEMBER_LIMIT],
            "members_truncated": len(members) > MEMBER_LIMIT,
        }
        return payload

    def _members_for_community(self, timeframe: dict[str, Any], community_id: str) -> list[str]:
        tokens = slot_tokens_from_timeframe(
            str(timeframe.get("start") or ""),
            str(timeframe.get("end") or ""),
        )
        if not tokens:
            return []
        start_token, end_token = tokens
        members: list[str] = []
        for dashboard in self.list_dashboards_raw():
            if dashboard["start_token"] != start_token or dashboard["end_token"] != end_token:
                continue
            groups = ((dashboard.get("legends") or {}).get("hybrid") or {}).get("groups") or {}
            if not isinstance(groups, dict):
                continue
            # legend may be node->community or label->{color,count}
            sample_value = next(iter(groups.values()), None)
            if isinstance(sample_value, dict):
                continue
            for node, cid in groups.items():
                if str(cid) == str(community_id):
                    members.append(str(node))
        return members

    def list_dashboards_raw(self) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            if not path.is_file():
                continue
            match = VISUALIZATION_RE.match(path.name)
            if not match:
                continue
            if match.group("prefix") != "dashboard":
                continue
            if match.group("legend") or match.group("ext") != "html":
                continue
            mode = match.group("mode") or "legacy"
            start_token = match.group("start")
            end_token = match.group("end")
            slot_id = f"{mode}_{start_token}_to_{end_token}"
            legend_path = self.root / path.name.replace(".html", "_legend.json")
            hybrid_graph = self.root / path.name.replace("dashboard_", "hybrid_graph_")
            graph_json = hybrid_graph.with_suffix(".json")
            legends = {"hybrid": {"groups": {}}}
            if legend_path.exists():
                try:
                    loaded = json.loads(legend_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        legends = {"hybrid": loaded.get("hybrid", {"groups": {}})}
                except json.JSONDecodeError:
                    pass
            try:
                start = format_slot_token(start_token)
                end = format_slot_token(end_token)
            except ValueError:
                start, end = start_token, end_token
            items.append(
                {
                    "id": slot_id,
                    "slot_mode": mode,
                    "start": start,
                    "end": end,
                    "start_token": start_token,
                    "end_token": end_token,
                    "dashboard": path.name,
                    "legend": legend_path.name if legend_path.exists() else None,
                    "hybrid_graph": hybrid_graph.name if hybrid_graph.exists() else None,
                    "graph_json": graph_json.name if graph_json.exists() else None,
                    "has_legend": legend_path.exists(),
                    "legends": legends,
                }
            )
        return items

    def list_dashboards(self, filters: dict[str, str], limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
        requested = parse_topic_labels(filters.get("topic"))
        config = self.pipeline_config().get("config") or {}
        if requested and not topics_are_covered(
            [{"key": topic_key(label)} for label in requested],
            config,
        ):
            return 0, []
        items = []
        for item in self.list_dashboards_raw():
            if filters.get("slot_mode") and item["slot_mode"] != filters["slot_mode"]:
                continue
            start = str(item.get("start") or "")[:10]
            end = str(item.get("end") or "")[:10]
            if filters.get("start") and start and start < filters["start"]:
                continue
            if filters.get("end") and end and end > filters["end"]:
                continue
            public = {key: value for key, value in item.items() if key != "legends"}
            items.append(public)
        total = len(items)
        return total, items[offset:offset + limit]

    def get_dashboard(self, slot_id: str) -> dict[str, Any]:
        wanted = safe_basename(slot_id)
        for item in self.list_dashboards_raw():
            if item["id"] == wanted or item["dashboard"] == wanted:
                payload = dict(item)
                graph_name = item.get("graph_json")
                graph = load_graph_json(
                    self.root / graph_name if graph_name else None
                )
                payload["partyFocus"] = graph.pop("partyFocus")
                payload["graph"] = graph
                payload["files"] = {
                    "dashboard": f"/api/v1/files/{item['dashboard']}",
                    "legend": f"/api/v1/files/{item['legend']}" if item.get("legend") else None,
                    "hybrid_graph": (
                        f"/api/v1/files/{item['hybrid_graph']}"
                        if item.get("hybrid_graph")
                        else None
                    ),
                    "graph": (
                        f"/api/v1/files/{item['graph_json']}"
                        if item.get("graph_json")
                        else None
                    ),
                }
                return payload
        raise FileNotFoundError(wanted)

    def list_topics(self) -> list[dict[str, Any]]:
        topics: dict[str, dict[str, Any]] = {}

        def add(
            label: str,
            query: str = "",
            source: str = "",
            has_data: bool = False,
            preloaded: bool = False,
        ):
            clean = (label or "").strip()
            if not clean:
                return
            key = topic_key(clean)
            current = topics.get(key)
            if current:
                current["has_data"] = current["has_data"] or has_data
                current["preloaded"] = current["preloaded"] or preloaded
                if query and not current.get("query"):
                    current["query"] = query
                if source and source not in current["sources"]:
                    current["sources"].append(source)
                return
            topics[key] = {
                "key": key,
                "label": clean,
                "query": (query or clean).strip(),
                "has_data": has_data,
                "preloaded": preloaded,
                "sources": [source] if source else [],
            }

        for item in PRELOADED_TOPICS:
            add(item["label"], item["query"], source="preload", preloaded=True)

        config = self.pipeline_config().get("config") or {}
        config_has_data = bool(self.list_dashboards_raw())
        if config.get("topic_label"):
            add(str(config["topic_label"]), source="pipeline_config", has_data=config_has_data)
        for item in config.get("topics") or []:
            if isinstance(item, dict):
                add(
                    str(item.get("label") or ""),
                    str(item.get("query") or ""),
                    source="pipeline_config",
                    has_data=config_has_data,
                )

        csv_path = self.root / "topics_10.csv"
        if csv_path.exists():
            try:
                with csv_path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        add(
                            str(row.get("topic_label") or ""),
                            str(row.get("topic_query") or ""),
                            source="topics_10.csv",
                        )
            except OSError:
                pass

        outputs = self.root / "topic_outputs"
        if outputs.is_dir():
            for name in sorted(os.listdir(outputs)):
                manifest_path = outputs / name / "topic_manifest.json"
                if not manifest_path.exists():
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(manifest, dict):
                    continue
                add(
                    str(manifest.get("label") or name),
                    str(manifest.get("query") or ""),
                    source="topic_outputs",
                    has_data=True,
                )

        return list(topics.values())

    def resolve_static_file(self, filename: str) -> Path:
        name = safe_basename(filename)
        if name in STATIC_ALLOW:
            path = self.root / name
            if path.is_file():
                return path
            raise FileNotFoundError(name)
        match = VISUALIZATION_RE.match(name)
        if not match:
            raise PermissionError(name)
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def resolve_reports(self, report_query: str | None) -> list[Path]:
        query = (report_query or DEFAULT_REPORT_GLOB).strip() or DEFAULT_REPORT_GLOB
        if any(char in query for char in "*?["):
            matches = sorted(self.root.glob(query))
            if not matches and not Path(query).is_absolute():
                matches = sorted(Path().glob(str(self.root / query)))
            paths = [path for path in matches if path.is_file()]
        else:
            path = Path(query)
            if not path.is_absolute():
                path = self.root / path
            paths = [path] if path.is_file() else []
        if not paths:
            raise FileNotFoundError(query)
        return paths


def validate_pipeline_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("request body must be an object")
    fetch = payload.get("fetch", True)
    detect = payload.get("detect", True)
    if not isinstance(fetch, bool) or not isinstance(detect, bool):
        raise ValueError("fetch and detect must be booleans")
    if not fetch and not detect:
        raise ValueError("at least one of fetch or detect must be true")
    auth = str(payload.get("auth") or "").strip()
    if auth and auth not in {"1", "2"}:
        raise ValueError("auth must be '1' or '2'")
    include_secondary = payload.get("include_secondary", False)
    if not isinstance(include_secondary, bool):
        raise ValueError("include_secondary must be a boolean")
    max_scan_docs = payload.get("max_scan_docs", 0)
    try:
        max_scan_docs = int(max_scan_docs)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_scan_docs must be an integer") from exc
    if max_scan_docs < 0:
        raise ValueError("max_scan_docs must be >= 0")
    days = payload.get("days")
    if days not in (None, ""):
        try:
            days = int(days)
        except (TypeError, ValueError) as exc:
            raise ValueError("days must be an integer") from exc
        if days < 1:
            raise ValueError("days must be >= 1")
    else:
        days = None
    explicit_multi = bool(parse_topic_labels(payload.get("topic_labels"))) or (
        isinstance(payload.get("topics"), list) and bool(payload.get("topics"))
    )
    selected = resolve_topic_selection(payload) if explicit_multi else []
    topic_label = str(payload.get("topic_label") or "").strip()
    topic_query = str(payload.get("topic_query") or "").strip()
    if selected:
        topic_label = topic_label or "، ".join(item["label"] for item in selected)
        topic_query = topic_query or " OR ".join(
            item["query"] for item in selected if item.get("query")
        )
    return {
        "fetch": fetch,
        "detect": detect,
        "topic_label": topic_label,
        "topic_query": topic_query,
        "topics": selected,
        "start_date": parse_iso_date(payload.get("start_date"), "start_date"),
        "end_date": parse_iso_date(payload.get("end_date"), "end_date"),
        "days": days,
        "slot_modes": normalize_slot_modes(payload.get("slot_modes")),
        "auth": auth,
        "include_secondary": include_secondary,
        "max_scan_docs": max_scan_docs,
    }


def validate_review_sample_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("request body must be an object")
    method = str(payload.get("method") or "both").strip() or "both"
    if method not in VALID_METHODS:
        raise ValueError("method must be hybrid, louvain, or both")
    try:
        sample_size = int(payload.get("sample_size", 20))
        min_size = int(payload.get("min_size", 1))
        seed = int(payload.get("seed", 42))
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_size, min_size, and seed must be integers") from exc
    if sample_size < 0:
        raise ValueError("sample_size must be >= 0")
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    return {
        "report": str(payload.get("report") or DEFAULT_REPORT_GLOB).strip(),
        "method": method,
        "sample_size": sample_size,
        "min_size": min_size,
        "seed": seed,
    }


def validate_label_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be an object")
    items = payload.get("labels")
    if items is None and payload.get("sample_id"):
        items = [payload]
    if not isinstance(items, list) or not items:
        raise ValueError("'labels' must be a non-empty array")
    if len(items) > 1000:
        raise ValueError("A batch can contain at most 1000 labels")
    labels = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"labels[{index}] must be an object")
        sample_id = str(item.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"labels[{index}] is missing sample_id")
        parsed = _parse_binary_label(item.get("human_label"))
        if parsed is None:
            raise ValueError(f"labels[{index}] has an invalid human_label")
        labels.append(
            {
                "sample_id": sample_id,
                "human_label": parsed,
                "notes": str(item.get("notes") or ""),
            }
        )
    return labels


def snapshot_names(root: Path, pattern: str) -> set[str]:
    return {path.name for path in root.glob(pattern) if path.is_file()}


def execute_pipeline(
    root: Path,
    params: dict[str, Any],
    log_file: Path,
    python_bin: str,
    cancel_event: threading.Event | None = None,
    on_process: Callable[[subprocess.Popen[Any] | None], None] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    communities = root / "communities"
    before_reports = snapshot_names(communities, "hybrid_report_*.json") if communities.is_dir() else set()
    before_dashboards = snapshot_names(root, "dashboard_*.html")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if params.get("topic_label"):
        env["TOPIC_LABEL"] = params["topic_label"]
        env["TOPIC_LABEL_OVERRIDE"] = params["topic_label"]
    if params.get("topic_query"):
        env["TOPIC_QUERY"] = params["topic_query"]
    if params.get("start_date"):
        env["START_DATE"] = params["start_date"]
    if params.get("end_date"):
        env["END_DATE"] = params["end_date"]

    def check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("stopped by user")

    def run_cmd(command: list[str]) -> None:
        check_cancel()
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ {' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(root),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if on_process:
                on_process(process)
            try:
                returncode = process.wait()
            finally:
                if on_process:
                    on_process(None)
        check_cancel()
        if returncode != 0:
            raise RuntimeError(
                command_failure_message(log_file, command, returncode)
            )

    write_progress(log_file, 5, "queued", "شروع اجرا")
    check_cancel()
    if params.get("fetch", True):
        write_progress(log_file, 15, "fetch", "در حال دریافت از الستیک")
        command = [python_bin, "-B", "elastic.py"]
        auth = params.get("auth") or os.getenv("ELASTIC_AUTH") or "1"
        command.extend(["--auth", str(auth)])
        if params.get("topic_label"):
            command.extend(["--topic-label", params["topic_label"]])
        if params.get("topic_query"):
            command.extend(["--topic-query", params["topic_query"]])
        if params.get("start_date"):
            command.extend(["--start-date", params["start_date"]])
        if params.get("end_date"):
            command.extend(["--end-date", params["end_date"]])
        if params.get("days"):
            command.extend(["--days", str(params["days"])])
        if params.get("include_secondary"):
            command.append("--include-secondary")
        if params.get("max_scan_docs"):
            command.extend(["--max-scan-docs", str(params["max_scan_docs"])])
        run_cmd(command)
        write_progress(log_file, 50, "fetch", "دریافت الستیک تمام شد")

    config_path = root / "pipeline_config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config = loaded
        except json.JSONDecodeError:
            config = {}
    changed = False
    for key in ("topic_label", "topic_query", "start_date", "end_date"):
        if params.get(key):
            config[key] = params[key]
            changed = True
    if params.get("topics"):
        config["topics"] = params["topics"]
        changed = True
    if params.get("slot_modes"):
        config["slot_modes"] = params["slot_modes"]
        changed = True
    if changed or not config_path.exists():
        config.setdefault("generated_at", datetime.now().isoformat(timespec="seconds"))
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    check_cancel()
    if params.get("detect", True):
        if not (root / "interactions.json").exists():
            raise RuntimeError("interactions.json is missing; run fetch first or set fetch=true")
        write_progress(log_file, 60, "detect", "در حال تشخیص جوامع")
        run_cmd([python_bin, "-B", "community_detection.py"])

    write_progress(log_file, 100, "ready", "تکمیل شد")
    after_reports = snapshot_names(communities, "hybrid_report_*.json") if communities.is_dir() else set()
    after_dashboards = snapshot_names(root, "dashboard_*.html")
    return {
        "reports": sorted(after_reports - before_reports),
        "dashboards": sorted(
            name for name in after_dashboards - before_dashboards if "_legend" not in name
        ),
    }


def sample_review_rows(root: Path, params: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    artifacts = ArtifactIndex(root)
    paths = artifacts.resolve_reports(params["report"])
    methods = ["hybrid", "louvain"] if params["method"] == "both" else [params["method"]]
    rng = random.Random(params["seed"])
    pooled: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if not isinstance(report, dict):
            continue
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
        for method in methods:
            pooled[method].extend(_rows_for_method(report, rel, method, params["min_size"]))
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows = pooled.get(method) or []
        if not method_rows:
            continue
        rows.extend(_sample_balanced(method_rows, params["sample_size"], rng))
    if not rows:
        raise RuntimeError("No communities matched the sampling filters")
    query = ", ".join(str(path.relative_to(root)) if path.is_relative_to(root) else path.name for path in paths)
    return query, rows


class UTF8JSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, default=str).encode("utf-8")


class PartyChangeEventIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_id: str
    node_id: str
    from_party: str
    to_party: str
    previous_slot: str
    current_slot: str
    slot_mode: str
    topic: str
    detected_at: str


class PartyChangeIngest(BaseModel):
    model_config = ConfigDict(extra="allow")
    events: list[dict[str, Any]] = Field(..., min_length=1)


class PipelineRunRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    fetch: bool = True
    detect: bool = True
    topic_label: str = ""
    topic_query: str = ""
    topic_labels: list[str] | None = None
    topics: Any = None
    start_date: str | None = None
    end_date: str | None = None
    days: int | None = None
    slot_modes: Any = None
    auth: str = ""
    include_secondary: bool = False
    max_scan_docs: int = 0


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    topics: Any = None
    topic_labels: list[str] | None = None
    topic_label: str = ""
    topic_query: str = ""
    start: str | None = None
    start_date: str | None = None
    end: str | None = None
    end_date: str | None = None
    slot_mode: str = ""
    slot_modes: Any = None
    fetch: bool = True
    detect: bool = True
    auth: str = ""
    max_scan_docs: int = 0


class ReviewSampleRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    report: str = DEFAULT_REPORT_GLOB
    method: str = "both"
    sample_size: int = 20
    min_size: int = 1
    seed: int = 42


class ReviewLabelRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    labels: list[dict[str, Any]] | None = None
    sample_id: str | None = None
    human_label: Any = None
    notes: str = ""


OPEN_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}
FILE_MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


def error_json(status: int, message: str) -> UTF8JSONResponse:
    return UTF8JSONResponse(status_code=status, content={"error": message})


def create_app(
    party_store: PartyChangeStore,
    pipeline_store: PipelineStore,
    review_store: ReviewStore,
    root: str | Path,
    api_token: str = "",
    python_bin: str | None = None,
    pipeline_executor: PipelineExecutor | None = None,
    pipeline_sync: bool = False,
) -> FastAPI:
    root_path = Path(root).resolve()
    artifacts = ArtifactIndex(root_path)
    python_bin = python_bin or sys.executable
    executor = pipeline_executor or execute_pipeline
    pipeline_lock = threading.Lock()
    log_dir = root_path / "runtime_data" / "pipeline_logs"
    active_jobs: dict[str, dict[str, Any]] = {}

    app = FastAPI(
        title="Echo Chamber API",
        version="1.0",
        description=(
            "Pipeline runs, echo-chamber reports, dashboard/timeline metadata, "
            "human review, and party-change events."
        ),
        default_response_class=UTF8JSONResponse,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=86400,
    )

    def pid_path_for(run_id: str) -> Path:
        return log_dir / f"{run_id}.pid"

    def remember_process(run_id: str, process: subprocess.Popen[Any] | None) -> None:
        job = active_jobs.get(run_id)
        if job is not None:
            job["process"] = process
        pid_path = pid_path_for(run_id)
        if process is None:
            pid_path.unlink(missing_ok=True)
            return
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(process.pid), encoding="utf-8")

    def stop_job_process(run_id: str) -> None:
        job = active_jobs.get(run_id)
        process = job.get("process") if job else None
        if job:
            job["cancel"].set()
        terminate_pid(read_pid_file(pid_path_for(run_id)), process)

    def run_pipeline_job(run_id: str) -> None:
        run = pipeline_store.get(run_id)
        if not run:
            return
        log_path = Path(run.get("log_path") or (log_dir / f"{run_id}.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        job = active_jobs.setdefault(run_id, {"cancel": threading.Event(), "process": None})
        if job["cancel"].is_set():
            pipeline_store.mark_cancelled(run_id, "stopped by user")
            active_jobs.pop(run_id, None)
            return
        pipeline_store.mark_running(run_id, log_path)
        try:
            params = run.get("params") or {}
            if executor is execute_pipeline:
                artifacts_found = execute_pipeline(
                    root_path,
                    params,
                    log_path,
                    python_bin,
                    cancel_event=job["cancel"],
                    on_process=lambda process: remember_process(run_id, process),
                )
            else:
                artifacts_found = executor(
                    root_path, params, log_path, python_bin
                )
            if job["cancel"].is_set():
                pipeline_store.mark_cancelled(run_id, "stopped by user")
            else:
                pipeline_store.mark_done(run_id, artifacts_found)
        except PipelineCancelled:
            pipeline_store.mark_cancelled(run_id, "stopped by user")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n[cancelled] stopped by user\n")
        except Exception as exc:
            pipeline_store.mark_failed(run_id, str(exc))
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[error] {exc}\n")
        finally:
            remember_process(run_id, None)
            active_jobs.pop(run_id, None)

    def query_limit_offset(limit: str, offset: str) -> tuple[int, int]:
        return parse_limit_offset({"limit": [str(limit)], "offset": [str(offset)]})

    def start_pipeline_run(params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with pipeline_lock:
            if pipeline_store.has_active():
                return 409, {"error": "a pipeline run is already queued or running"}
            log_dir.mkdir(parents=True, exist_ok=True)
            run = pipeline_store.create(params)
            active_jobs[run["run_id"]] = {"cancel": threading.Event(), "process": None}
        if pipeline_sync:
            run_pipeline_job(run["run_id"])
        else:
            thread = threading.Thread(
                target=run_pipeline_job,
                args=(run["run_id"],),
                name=f"pipeline-{run['run_id'][:8]}",
                daemon=True,
            )
            thread.start()
        started = pipeline_store.get(run["run_id"])
        assert started is not None
        return 202, started

    def stop_pipeline_run(run_id: str) -> tuple[int, dict[str, Any]]:
        with pipeline_lock:
            run = pipeline_store.get(run_id)
            if not run:
                return 404, {"error": "not found"}
            if run["status"] not in STOPPABLE_STATUSES:
                return 409, {"error": "run is not queued or running"}
            stop_job_process(run_id)
            pipeline_store.mark_cancelled(run_id, "stopped by user")
        stopped = pipeline_store.get(run_id)
        assert stopped is not None
        return 200, stopped

    def restart_pipeline_run(run_id: str) -> tuple[int, dict[str, Any]]:
        run = pipeline_store.get(run_id)
        if not run:
            return 404, {"error": "not found"}
        params = dict(run.get("params") or {})
        if not params:
            params = {
                "fetch": run.get("fetch", True),
                "detect": run.get("detect", True),
                "topic_label": run.get("topic_label") or "",
                "topic_query": run.get("topic_query") or "",
                "start_date": run.get("start_date"),
                "end_date": run.get("end_date"),
                "slot_modes": run.get("slot_modes"),
            }
        params = validate_pipeline_payload(params)
        with pipeline_lock:
            current = pipeline_store.get(run_id)
            if not current:
                return 404, {"error": "not found"}
            if current["status"] in STOPPABLE_STATUSES:
                stop_job_process(run_id)
                pipeline_store.mark_cancelled(run_id, "restarted by user")
            elif pipeline_store.has_active():
                return 409, {"error": "a pipeline run is already queued or running"}
            log_dir.mkdir(parents=True, exist_ok=True)
            created = pipeline_store.create(params)
            active_jobs[created["run_id"]] = {"cancel": threading.Event(), "process": None}
            new_id = created["run_id"]
        if pipeline_sync:
            run_pipeline_job(new_id)
        else:
            thread = threading.Thread(
                target=run_pipeline_job,
                args=(new_id,),
                name=f"pipeline-{new_id[:8]}",
                daemon=True,
            )
            thread.start()
        started = pipeline_store.get(new_id)
        assert started is not None
        started["restarted_from"] = run_id
        return 202, started

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        if (
            request.method == "OPTIONS"
            or path in OPEN_PATHS
            or path.startswith("/docs")
            or path.startswith("/redoc")
        ):
            return await call_next(request)
        expected = f"Bearer {api_token}"
        if api_token and request.headers.get("Authorization") != expected:
            return error_json(401, "unauthorized")
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, exc: RequestValidationError
    ):
        errors = exc.errors()
        if not errors:
            return error_json(400, "invalid request")
        err = errors[0]
        loc = ".".join(str(part) for part in err.get("loc", []) if part != "body")
        message = err.get("msg", "invalid request")
        return error_json(400, f"{loc}: {message}" if loc else message)

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError):
        return error_json(400, str(exc))

    @app.exception_handler(json.JSONDecodeError)
    async def json_error_handler(_request: Request, exc: json.JSONDecodeError):
        return error_json(400, str(exc))

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_request: Request, exc: RuntimeError):
        return error_json(400, str(exc))

    @app.exception_handler(FileNotFoundError)
    async def missing_handler(_request: Request, _exc: FileNotFoundError):
        return error_json(404, "not found")

    @app.exception_handler(PermissionError)
    async def forbidden_handler(_request: Request, _exc: PermissionError):
        return error_json(403, "file is not served")

    @app.exception_handler(sqlite3.Error)
    async def database_error_handler(_request: Request, _exc: sqlite3.Error):
        return error_json(500, "database operation failed")

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/party-changes", tags=["party-changes"])
    def list_party_changes(
        limit: str = Query("100"),
        offset: str = Query("0"),
        node_id: str = "",
        from_party: str = "",
        to_party: str = "",
        slot_mode: str = "",
        topic: str = "",
    ):
        limit_n, offset_n = query_limit_offset(limit, offset)
        raw = {
            "node_id": node_id,
            "from_party": from_party,
            "to_party": to_party,
            "slot_mode": slot_mode,
            "topic": topic,
        }
        filters = {field: raw.get(field, "") for field in PARTY_QUERY_FIELDS}
        total, events = party_store.list(filters, limit_n, offset_n)
        return {"events": events, "total": total, "limit": limit_n, "offset": offset_n}

    @app.get("/api/v1/party-changes/{event_id}", tags=["party-changes"])
    def get_party_change(event_id: str):
        event = party_store.get(event_id)
        if not event:
            return error_json(404, "not found")
        return event

    @app.post("/api/v1/party-changes", tags=["party-changes"])
    def post_party_changes(payload: PartyChangeIngest):
        events = validate_events(payload.model_dump())
        count = party_store.upsert_many(events)
        return {"accepted": count, "status": "ok"}

    @app.get("/api/v1/pipeline/config", tags=["pipeline"])
    def pipeline_config():
        return artifacts.pipeline_config()

    @app.get("/api/v1/pipeline/runs", tags=["pipeline"])
    def list_pipeline_runs(limit: str = Query("100"), offset: str = Query("0")):
        limit_n, offset_n = query_limit_offset(limit, offset)
        total, runs = pipeline_store.list(limit_n, offset_n)
        return {"runs": runs, "total": total, "limit": limit_n, "offset": offset_n}

    @app.get("/api/v1/pipeline/runs/{run_id}", tags=["pipeline"])
    def get_pipeline_run(run_id: str):
        run = pipeline_store.get(run_id)
        if not run:
            return error_json(404, "not found")
        return run

    @app.post("/api/v1/pipeline/runs", status_code=202, tags=["pipeline"])
    def post_pipeline_run(payload: PipelineRunRequest | None = None):
        params = validate_pipeline_payload(payload.model_dump() if payload else {})
        status, body = start_pipeline_run(params)
        if status == 409:
            return error_json(409, body["error"])
        return UTF8JSONResponse(status_code=202, content=body)

    @app.post("/api/v1/pipeline/runs/{run_id}/stop", tags=["pipeline"])
    def post_pipeline_run_stop(run_id: str):
        status, body = stop_pipeline_run(run_id)
        if status != 200:
            return error_json(status, str(body.get("error") or "cannot stop run"))
        return body

    @app.post("/api/v1/pipeline/runs/{run_id}/restart", status_code=202, tags=["pipeline"])
    def post_pipeline_run_restart(run_id: str):
        status, body = restart_pipeline_run(run_id)
        if status == 404:
            return error_json(404, str(body.get("error") or "not found"))
        if status == 409:
            return error_json(409, str(body.get("error") or "cannot restart run"))
        return UTF8JSONResponse(status_code=202, content=body)

    @app.get("/api/v1/reports", tags=["reports"])
    def list_reports(
        limit: str = Query("100"),
        offset: str = Query("0"),
        topic: str = "",
        start: str = "",
        end: str = "",
    ):
        limit_n, offset_n = query_limit_offset(limit, offset)
        total, reports = artifacts.list_reports(
            {"topic": topic, "start": start, "end": end},
            limit_n,
            offset_n,
        )
        return {
            "reports": reports,
            "total": total,
            "limit": limit_n,
            "offset": offset_n,
        }

    @app.get("/api/v1/reports/{report_id}", tags=["reports"])
    def get_report(report_id: str):
        return artifacts.get_report(report_id)

    @app.get("/api/v1/communities/{report_id}/{community_id}", tags=["reports"])
    def get_community(report_id: str, community_id: str, method: str = "hybrid"):
        method = method or "hybrid"
        if method not in {"hybrid", "louvain"}:
            raise ValueError("method must be hybrid or louvain")
        return artifacts.get_community(report_id, community_id, method)

    @app.get("/api/v1/dashboards", tags=["dashboards"])
    def list_dashboards(
        limit: str = Query("100"),
        offset: str = Query("0"),
        slot_mode: str = "",
        start: str = "",
        end: str = "",
        topic: str = "",
    ):
        limit_n, offset_n = query_limit_offset(limit, offset)
        total, dashboards = artifacts.list_dashboards(
            {"slot_mode": slot_mode, "start": start, "end": end, "topic": topic},
            limit_n,
            offset_n,
        )
        return {
            "dashboards": dashboards,
            "total": total,
            "limit": limit_n,
            "offset": offset_n,
        }

    @app.get("/api/v1/dashboards/{slot_id}", tags=["dashboards"])
    def get_dashboard(slot_id: str):
        return artifacts.get_dashboard(slot_id)

    @app.get("/api/v1/topics", tags=["dashboards"])
    def list_topics():
        return {
            "selection": "multiple",
            "topics": artifacts.list_topics(),
        }

    @app.post("/api/v1/search", tags=["dashboards"])
    def search_dashboards(payload: SearchRequest | None = None):
        raw = payload.model_dump() if payload else {}
        if raw.get("start") and not raw.get("start_date"):
            raw["start_date"] = raw["start"]
        if raw.get("end") and not raw.get("end_date"):
            raw["end_date"] = raw["end"]
        if raw.get("slot_mode") and not raw.get("slot_modes"):
            raw["slot_modes"] = raw["slot_mode"]
        params = validate_pipeline_payload(raw)
        start = params.get("start_date") or ""
        end = params.get("end_date") or ""
        slot_mode = ""
        if params.get("slot_modes"):
            slot_mode = params["slot_modes"][0]
        selected = params.get("topics") or []
        config = artifacts.pipeline_config().get("config") or {}
        total, dashboards = artifacts.list_dashboards(
            {
                "slot_mode": slot_mode,
                "start": start,
                "end": end,
                "topic": "، ".join(item["label"] for item in selected),
            },
            500,
            0,
        )
        cached = bool(dashboards) and topics_are_covered(selected, config)
        if cached:
            return {
                "status": "ready",
                "cached": True,
                "run_id": None,
                "topics": selected,
                "progress": progress_payload("done"),
                "dashboards": dashboards,
                "total": total,
            }
        status, body = start_pipeline_run(params)
        if status == 409:
            return error_json(409, body["error"])
        return UTF8JSONResponse(
            status_code=202,
            content={
                "status": body.get("status") or "queued",
                "cached": False,
                "run_id": body.get("run_id"),
                "topics": selected,
                "progress": body.get("progress") or progress_payload(str(body.get("status") or "queued")),
                "dashboards": [],
                "total": 0,
                "run": body,
            },
        )

    @app.get("/api/v1/files/{filename}", tags=["dashboards"])
    def get_file(filename: str):
        path = artifacts.resolve_static_file(filename)
        return FileResponse(
            path,
            media_type=FILE_MEDIA_TYPES.get(
                path.suffix.lower(), "application/octet-stream"
            ),
            filename=path.name,
        )

    @app.get("/api/v1/review/samples", tags=["review"])
    def list_review_samples(
        limit: str = Query("100"),
        offset: str = Query("0"),
        batch_id: str = "",
        method: str = "",
        labeled: str = "",
    ):
        limit_n, offset_n = query_limit_offset(limit, offset)
        total, samples = review_store.list_samples(
            {
                "batch_id": batch_id,
                "method": method,
                "labeled": parse_bool_query(labeled),
            },
            limit_n,
            offset_n,
        )
        return {
            "samples": samples,
            "total": total,
            "limit": limit_n,
            "offset": offset_n,
        }

    @app.get("/api/v1/review/samples/{sample_id}", tags=["review"])
    def get_review_sample(sample_id: str):
        sample = review_store.get_sample(sample_id)
        if not sample:
            return error_json(404, "not found")
        return sample

    @app.get("/api/v1/review/metrics", tags=["review"])
    def review_metrics(batch_id: str = ""):
        return review_store.metrics(batch_id or None)

    @app.post("/api/v1/review/samples", status_code=201, tags=["review"])
    def post_review_samples(payload: ReviewSampleRequest | None = None):
        params = validate_review_sample_payload(
            payload.model_dump() if payload else {}
        )
        report_query, rows = sample_review_rows(root_path, params)
        return review_store.create_batch(
            rows,
            report_query=report_query,
            method=params["method"],
            sample_size=params["sample_size"],
            min_size=params["min_size"],
            seed=params["seed"],
        )

    @app.post("/api/v1/review/labels", tags=["review"])
    def post_review_labels(payload: ReviewLabelRequest):
        labels = validate_label_payload(payload.model_dump(exclude_none=True))
        try:
            updated = review_store.apply_labels(labels)
        except KeyError as exc:
            return error_json(404, f"sample not found: {exc}")
        return {"updated": len(updated), "samples": updated}

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema.setdefault("components", {})["securitySchemes"] = {
            "BearerAuth": {"type": "http", "scheme": "bearer"}
        }
        for path, methods in schema.get("paths", {}).items():
            if path in OPEN_PATHS:
                continue
            for operation in methods.values():
                if isinstance(operation, dict):
                    operation.setdefault("security", [{"BearerAuth": []}])
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi
    return app


def env_or(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=env_or("ECHO_API_HOST", "PARTY_CHANGE_API_HOST", default="127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(env_or("ECHO_API_PORT", "PARTY_CHANGE_API_PORT", default="8765")),
    )
    parser.add_argument(
        "--root",
        default=env_or("ECHO_API_ROOT", default="."),
    )
    parser.add_argument(
        "--db",
        default=env_or("ECHO_API_DB_PATH", default=str(DEFAULT_DB_PATH)),
    )
    parser.add_argument(
        "--party-db",
        default=env_or("PARTY_CHANGE_DB_PATH", default=str(DEFAULT_PARTY_DB_PATH)),
    )
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
    root = Path(args.root).resolve()
    party_store = PartyChangeStore(args.party_db, args.retention_days, args.max_events)
    pipeline_store = PipelineStore(args.db)
    review_store = ReviewStore(args.db)
    token = env_or("ECHO_API_TOKEN", "PARTY_CHANGE_API_TOKEN")
    app = create_app(
        party_store=party_store,
        pipeline_store=pipeline_store,
        review_store=review_store,
        root=root,
        api_token=token,
        python_bin=env_or("ECHO_PYTHON_BIN", default=sys.executable),
        pipeline_sync=os.getenv("ECHO_API_SYNC_PIPELINE", "").strip().lower()
        in {"1", "true", "yes"},
    )
    print(
        f"Echo Chamber API listening on http://{args.host}:{args.port}; "
        f"docs=http://{args.host}:{args.port}/docs; "
        f"root={root}; database={Path(args.db)}"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
