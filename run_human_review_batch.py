#!/usr/bin/env python3
"""Run/prepare human-review CSVs for multiple topics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List

from sample_echo_review import CSV_COLUMNS, _rows_for_method, _sample_balanced


SUMMARY_COLUMNS = [
    "topic_index",
    "topic_label",
    "topic_query",
    "review_csv",
    "reports_count",
    "row_count",
    "pred_yes",
    "pred_no",
    "human_labeled",
    "human_yes",
    "human_no",
    "agreement",
]


def _slugify(text: str) -> str:
    slug = re.sub(r"\s+", "_", text.strip())
    slug = re.sub(r"[^0-9A-Za-z_\u0600-\u06FF-]+", "", slug)
    slug = slug.strip("_")
    return slug or "topic"


def _parse_binary_label(raw: object):
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    positives = {"1", "true", "t", "yes", "y", "echo", "positive", "pos", "bale", "بله", "اره", "آره", "آری"}
    negatives = {"0", "false", "f", "no", "n", "not_echo", "not-echo", "nonecho", "negative", "neg", "kheir", "خیر", "نه"}
    if text in positives:
        return 1
    if text in negatives:
        return 0
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric == 1.0:
        return 1
    if numeric == 0.0:
        return 0
    return None


def _load_topics(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Topics file not found: {path}")

    topics: List[Dict[str, str]] = []
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON topics file must be a list of objects.")
        for item in data:
            if not isinstance(item, dict):
                continue
            label = str(item.get("topic_label", "")).strip()
            query = str(item.get("topic_query", "")).strip()
            if not label:
                continue
            topics.append({"topic_label": label, "topic_query": query or label})
        return topics

    if ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = str(row.get("topic_label", "")).strip()
                query = str(row.get("topic_query", "")).strip()
                if not label:
                    continue
                topics.append({"topic_label": label, "topic_query": query or label})
        return topics

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if "\t" in raw:
                label, query = raw.split("\t", 1)
            else:
                label, query = raw, raw
            label = label.strip()
            query = query.strip()
            if label:
                topics.append({"topic_label": label, "topic_query": query or label})
    return topics


def _run(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _latest_reports() -> List[str]:
    return sorted(glob.glob("communities/hybrid_report_*_to_*.json"))


def _copy_if_exists(src: str, dst: str) -> None:
    if os.path.exists(src):
        shutil.copy2(src, dst)


def _topic_rows(report_paths: List[str], method: str, min_size: int, sample_size: int, rng: random.Random) -> List[Dict]:
    methods = ["hybrid", "louvain"] if method == "both" else [method]
    pooled_by_method: Dict[str, List[Dict]] = {m: [] for m in methods}
    for report_path in report_paths:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        for m in methods:
            rows = _rows_for_method(report, report_path, m, min_size)
            pooled_by_method[m].extend(rows)

    all_rows: List[Dict] = []
    for m in methods:
        rows = pooled_by_method.get(m, [])
        if not rows:
            continue
        sampled = _sample_balanced(rows, sample_size, rng)
        all_rows.extend(sampled)

    return all_rows


def _summarize_rows(rows: List[Dict]) -> Dict[str, str]:
    pred_yes = 0
    pred_no = 0
    human_yes = 0
    human_no = 0
    human_labeled = 0
    agree = 0

    for row in rows:
        pred = _parse_binary_label(row.get("predicted_is_echo", ""))
        truth = _parse_binary_label(row.get("human_label", ""))
        if pred == 1:
            pred_yes += 1
        elif pred == 0:
            pred_no += 1
        if truth is None:
            continue
        human_labeled += 1
        if truth == 1:
            human_yes += 1
        elif truth == 0:
            human_no += 1
        if pred is not None and pred == truth:
            agree += 1

    agreement = f"{(agree / human_labeled) * 100:.2f}%" if human_labeled else ""
    return {
        "row_count": str(len(rows)),
        "pred_yes": str(pred_yes),
        "pred_no": str(pred_no),
        "human_labeled": str(human_labeled),
        "human_yes": str(human_yes),
        "human_no": str(human_no),
        "agreement": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for multi-topic human echo review.")
    parser.add_argument("--topics", default="topics_10.csv", help="Topics file (CSV/JSON/TXT).")
    parser.add_argument("--python-bin", default="./venv/bin/python", help="Python interpreter path.")
    parser.add_argument("--output-dir", default="human_review_runs", help="Output directory.")
    parser.add_argument("--method", choices=["hybrid", "louvain", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=20, help="Sample size per method for each topic (0=all).")
    parser.add_argument("--min-size", type=int, default=1, help="Minimum community size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--run-elastic", action="store_true", help="Fetch fresh data from Elasticsearch for each topic.")
    parser.add_argument("--auth", choices=["1", "2"], default="1", help="elastic.py --auth value.")
    parser.add_argument("--include-secondary", action="store_true", help="Pass --include-secondary to elastic.py.")
    parser.add_argument("--max-scan-docs", type=int, default=0, help="Pass --max-scan-docs to elastic.py.")
    parser.add_argument("--start-date", default="", help="Pass --start-date to elastic.py.")
    parser.add_argument("--end-date", default="", help="Pass --end-date to elastic.py.")
    parser.add_argument("--skip-detection", action="store_true", help="Skip community_detection.py and reuse latest reports.")
    args = parser.parse_args()

    topics = _load_topics(args.topics)
    if len(topics) != 10:
        print(f"[warn] topics count is {len(topics)} (expected 10).")
    if not topics:
        raise RuntimeError("No topics loaded.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, f"batch_{run_stamp}")
    os.makedirs(run_root, exist_ok=True)

    summary_rows: List[Dict[str, str]] = []
    base_rng = random.Random(args.seed)

    for idx, topic in enumerate(topics, start=1):
        topic_label = topic["topic_label"]
        topic_query = topic["topic_query"]
        topic_slug = _slugify(topic_label)
        topic_dir = os.path.join(run_root, f"{idx:02d}_{topic_slug}")
        os.makedirs(topic_dir, exist_ok=True)

        print(f"\n[topic {idx}/{len(topics)}] {topic_label}")
        with open(os.path.join(topic_dir, "topic_meta.json"), "w", encoding="utf-8") as f:
            json.dump(topic, f, ensure_ascii=False, indent=2)

        reports_before = set(_latest_reports())

        if args.run_elastic:
            elastic_cmd = [args.python_bin, "elastic.py", "--auth", args.auth]
            if args.include_secondary:
                elastic_cmd.append("--include-secondary")
            if args.max_scan_docs:
                elastic_cmd.extend(["--max-scan-docs", str(args.max_scan_docs)])
            if args.start_date:
                elastic_cmd.extend(["--start-date", args.start_date])
            if args.end_date:
                elastic_cmd.extend(["--end-date", args.end_date])
            if topic_query:
                elastic_cmd.extend(["--topic-query", topic_query])
            _run(elastic_cmd)
            _copy_if_exists("interactions.json", os.path.join(topic_dir, "interactions.json"))
            _copy_if_exists("res.json", os.path.join(topic_dir, "res.json"))
            _copy_if_exists("debug.log", os.path.join(topic_dir, "elastic_debug.log"))

        if not args.skip_detection:
            env = os.environ.copy()
            env["TOPIC_LABEL_OVERRIDE"] = topic_label
            _run([args.python_bin, "community_detection.py"], env=env)

        reports_after = set(_latest_reports())
        new_reports = sorted(reports_after - reports_before)
        if not new_reports:
            latest = sorted(reports_after)
            if not latest:
                raise RuntimeError("No hybrid reports found in communities/.")
            new_reports = [latest[-1]]

        topic_rng = random.Random(base_rng.randint(1, 10**9))
        rows = _topic_rows(
            report_paths=new_reports,
            method=args.method,
            min_size=args.min_size,
            sample_size=args.sample_size,
            rng=topic_rng,
        )

        if not rows:
            print("[skip] No review rows generated for this topic.")
            continue

        review_csv = os.path.join(topic_dir, f"echo_review_{topic_slug}.csv")
        with open(review_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {review_csv}")

        stats = _summarize_rows(rows)
        summary_rows.append(
            {
                "topic_index": str(idx),
                "topic_label": topic_label,
                "topic_query": topic_query,
                "review_csv": review_csv,
                "reports_count": str(len(new_reports)),
                **stats,
            }
        )

    summary_path = os.path.join(run_root, "summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    notes_path = os.path.join(run_root, "next_steps.txt")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("1) Fill human_label with 1/0 (or بله/خیر) in each topic CSV.\n")
        f.write("2) Evaluate each CSV using:\n")
        f.write("   ./venv/bin/python evaluate_echo_labels.py --input <path_to_topic_csv>\n")
        f.write("3) Re-run this batch with --skip-detection (optional) to refresh summary agreement.\n")

    print(f"\n[done] Batch outputs: {run_root}")
    print(f"[done] Summary: {summary_path}")


if __name__ == "__main__":
    main()
