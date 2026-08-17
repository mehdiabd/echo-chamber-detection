#!/usr/bin/env python3
"""Create a manual review sample from echo-chamber report metrics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from datetime import datetime
from typing import Dict, List


CSV_COLUMNS = [
    "report_file",
    "time_start",
    "time_end",
    "method",
    "community_id",
    "size",
    "ei_index",
    "conductance",
    "content_homogeneity",
    "predicted_is_echo",
    "stance_target",
    "stance_label",
    "stance_pos",
    "stance_neg",
    "stance_neu",
    "stance_total",
    "human_label",
    "notes",
]


def _latest_report(path_glob: str) -> str:
    matches = sorted(glob.glob(path_glob))
    if not matches:
        raise FileNotFoundError(f"No report matched: {path_glob}")
    return matches[-1]


def _fmt_num(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _rows_for_method(report: Dict, report_file: str, method: str, min_size: int) -> List[Dict]:
    timeframe = report.get("timeframe", {}) or {}
    metrics = report.get(method, {}).get("echo_metrics", {}) or {}
    rows: List[Dict] = []
    for comm_id, info in metrics.items():
        size = int(info.get("size", 0) or 0)
        if size < min_size:
            continue
        stance = info.get("stance", {}) if isinstance(info.get("stance"), dict) else {}
        counts = stance.get("counts", {}) if isinstance(stance.get("counts"), dict) else {}

        rows.append(
            {
                "report_file": report_file,
                "time_start": timeframe.get("start", ""),
                "time_end": timeframe.get("end", ""),
                "method": method,
                "community_id": str(comm_id),
                "size": str(size),
                "ei_index": _fmt_num(info.get("ei_index")),
                "conductance": _fmt_num(info.get("conductance")),
                "content_homogeneity": _fmt_num(info.get("content_homogeneity")),
                "predicted_is_echo": "1" if bool(info.get("is_echo_chamber")) else "0",
                "stance_target": str(stance.get("target", "")) if stance else "",
                "stance_label": str(stance.get("stance", "")) if stance else "",
                "stance_pos": str(counts.get("pos", "")),
                "stance_neg": str(counts.get("neg", "")),
                "stance_neu": str(counts.get("neu", "")),
                "stance_total": str(counts.get("total", "")),
                "human_label": "",
                "notes": "",
            }
        )
    return rows


def _sample_balanced(rows: List[Dict], sample_size: int, rng: random.Random) -> List[Dict]:
    if sample_size <= 0 or sample_size >= len(rows):
        picked = list(rows)
        rng.shuffle(picked)
        return picked

    positives = [r for r in rows if r["predicted_is_echo"] == "1"]
    negatives = [r for r in rows if r["predicted_is_echo"] == "0"]

    if not positives or not negatives:
        return rng.sample(rows, sample_size)

    half = sample_size // 2
    pos_n = min(half, len(positives))
    neg_n = min(sample_size - pos_n, len(negatives))

    picked = rng.sample(positives, pos_n) + rng.sample(negatives, neg_n)
    if len(picked) < sample_size:
        used_ids = {id(r) for r in picked}
        remaining = [r for r in rows if id(r) not in used_ids]
        need = sample_size - len(picked)
        picked.extend(rng.sample(remaining, min(need, len(remaining))))
    rng.shuffle(picked)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manual echo review sample CSV.")
    parser.add_argument(
        "--report",
        default="communities/hybrid_report_*_to_*.json",
        help="Report file path or glob (default: latest hybrid report).",
    )
    parser.add_argument(
        "--method",
        choices=["hybrid", "louvain", "both"],
        default="both",
        help="Which method(s) to sample.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Sample size per selected method (0 = include all).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Minimum community size to include.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling random seed.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path (default: timestamped file in project root).",
    )
    args = parser.parse_args()

    report_path = args.report
    if any(ch in report_path for ch in ["*", "?", "["]):
        report_path = _latest_report(report_path)
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    methods = ["hybrid", "louvain"] if args.method == "both" else [args.method]
    rng = random.Random(args.seed)

    output_rows: List[Dict] = []
    for method in methods:
        rows = _rows_for_method(report, report_path, method, args.min_size)
        if not rows:
            print(f"[skip] No rows found for method={method}.")
            continue
        sampled = _sample_balanced(rows, args.sample_size, rng)
        output_rows.extend(sampled)
        pred_pos = sum(1 for r in sampled if r["predicted_is_echo"] == "1")
        print(
            f"[sample] method={method} total={len(rows)} sampled={len(sampled)} "
            f"pred_pos={pred_pos} pred_neg={len(sampled) - pred_pos}"
        )

    if not output_rows:
        raise RuntimeError("No rows selected for output.")

    output_path = args.output or (
        f"echo_review_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[saved] {output_path}")
    print("[note] Fill `human_label` with 1 (echo) or 0 (not echo), then run evaluate_echo_labels.py.")


if __name__ == "__main__":
    main()
