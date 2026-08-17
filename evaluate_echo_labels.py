#!/usr/bin/env python3
"""Evaluate echo-chamber predictions against manually labeled samples."""

from __future__ import annotations

import argparse
import csv
import json
from typing import Dict, Optional, Tuple


def _parse_binary_label(raw: object) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None

    positives = {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "echo",
        "positive",
        "pos",
        "bale",
        "بله",
        "اره",
        "آره",
        "آری",
    }
    negatives = {
        "0",
        "false",
        "f",
        "no",
        "n",
        "not_echo",
        "not-echo",
        "nonecho",
        "negative",
        "neg",
        "kheir",
        "خیر",
        "نه",
    }
    if text in positives:
        return 1
    if text in negatives:
        return 0

    # Accept numeric values like 1.0 and 0.0.
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric == 1.0:
        return 1
    if numeric == 0.0:
        return 0
    return None


def _empty_confusion() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}


def _update_confusion(matrix: Dict[str, int], predicted: int, truth: int) -> None:
    if predicted == 1 and truth == 1:
        matrix["tp"] += 1
    elif predicted == 1 and truth == 0:
        matrix["fp"] += 1
    elif predicted == 0 and truth == 0:
        matrix["tn"] += 1
    else:
        matrix["fn"] += 1


def _metrics_from_confusion(matrix: Dict[str, int]) -> Dict[str, float]:
    tp = matrix["tp"]
    fp = matrix["fp"]
    tn = matrix["tn"]
    fn = matrix["fn"]
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_label_rows(
    rows,
    pred_col: str = "predicted_is_echo",
    truth_col: str = "human_label",
    method_col: str = "method",
) -> Dict[str, object]:
    """Compute precision/recall metrics from predicted vs human labels."""
    total_rows = 0
    used_rows = 0
    skipped_unlabeled = 0
    invalid_truth = 0
    invalid_pred = 0
    overall = _empty_confusion()
    by_method: Dict[str, Dict[str, int]] = {}

    for row in rows:
        total_rows += 1
        pred, truth, raw_pred, raw_truth = _parse_row_labels(row, pred_col, truth_col)
        if truth is None:
            if not str(raw_truth).strip():
                skipped_unlabeled += 1
            else:
                invalid_truth += 1
            continue
        if pred is None:
            invalid_pred += 1
            continue
        used_rows += 1
        _update_confusion(overall, pred, truth)
        method = str(row.get(method_col, "") or "unknown").strip() or "unknown"
        if method not in by_method:
            by_method[method] = _empty_confusion()
        _update_confusion(by_method[method], pred, truth)

    return {
        "rows": {
            "total": total_rows,
            "used": used_rows,
            "unlabeled": skipped_unlabeled,
            "invalid_truth": invalid_truth,
            "invalid_pred": invalid_pred,
        },
        "overall": _metrics_from_confusion(overall),
        "by_method": {
            method: _metrics_from_confusion(confusion)
            for method, confusion in sorted(by_method.items())
        },
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"[metrics] {label}: "
        f"n={metrics['n']} "
        f"tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']} "
        f"precision={_fmt_pct(metrics['precision'])} "
        f"recall={_fmt_pct(metrics['recall'])} "
        f"f1={_fmt_pct(metrics['f1'])} "
        f"accuracy={_fmt_pct(metrics['accuracy'])}"
    )


def _parse_row_labels(
    row: Dict[str, str], pred_col: str, truth_col: str
) -> Tuple[Optional[int], Optional[int], str, str]:
    raw_pred = row.get(pred_col, "")
    raw_truth = row.get(truth_col, "")
    if raw_pred is None:
        raw_pred = ""
    if raw_truth is None:
        raw_truth = ""
    raw_pred = str(raw_pred).strip()
    raw_truth = str(raw_truth).strip()
    pred = _parse_binary_label(raw_pred)
    truth = _parse_binary_label(raw_truth)
    return pred, truth, raw_pred, raw_truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted echo labels against human labels in CSV."
    )
    parser.add_argument("--input", required=True, help="Input CSV (from sample_echo_review.py).")
    parser.add_argument(
        "--pred-col",
        default="predicted_is_echo",
        help="Prediction column name (default: predicted_is_echo).",
    )
    parser.add_argument(
        "--truth-col",
        default="human_label",
        help="Human label column name (default: human_label).",
    )
    parser.add_argument(
        "--method-col",
        default="method",
        help="Method grouping column (default: method).",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path for computed metrics.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid non-empty labels instead of skipping them.",
    )
    parser.add_argument(
        "--show-invalid",
        type=int,
        default=5,
        help="Max invalid row examples to print (default: 5).",
    )
    args = parser.parse_args()

    total_rows = 0
    used_rows = 0
    skipped_unlabeled = 0
    invalid_truth = 0
    invalid_pred = 0
    invalid_examples = []

    overall = _empty_confusion()
    by_method: Dict[str, Dict[str, int]] = {}

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in [args.pred_col, args.truth_col] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {', '.join(missing)}")

        for idx, row in enumerate(reader, start=2):
            total_rows += 1
            pred, truth, raw_pred, raw_truth = _parse_row_labels(row, args.pred_col, args.truth_col)

            if truth is None:
                if not raw_truth:
                    skipped_unlabeled += 1
                    continue
                invalid_truth += 1
                if len(invalid_examples) < args.show_invalid:
                    invalid_examples.append(
                        f"line={idx} invalid truth `{raw_truth}` in column `{args.truth_col}`"
                    )
                if args.strict:
                    raise ValueError(
                        f"Invalid non-empty truth label at line {idx}: {raw_truth!r}"
                    )
                continue

            if pred is None:
                invalid_pred += 1
                if len(invalid_examples) < args.show_invalid:
                    invalid_examples.append(
                        f"line={idx} invalid prediction `{raw_pred}` in column `{args.pred_col}`"
                    )
                if args.strict:
                    raise ValueError(
                        f"Invalid prediction label at line {idx}: {raw_pred!r}"
                    )
                continue

            used_rows += 1
            _update_confusion(overall, pred, truth)

            method = str(row.get(args.method_col, "")).strip() if args.method_col else ""
            method = method or "unknown"
            if method not in by_method:
                by_method[method] = _empty_confusion()
            _update_confusion(by_method[method], pred, truth)

    if used_rows == 0:
        raise RuntimeError("No labeled rows were usable. Fill the human labels and retry.")

    overall_metrics = _metrics_from_confusion(overall)
    method_metrics = {
        method: _metrics_from_confusion(confusion)
        for method, confusion in sorted(by_method.items())
    }

    print(f"[rows] total={total_rows} used={used_rows} unlabeled={skipped_unlabeled}")
    print(f"[rows] invalid_truth={invalid_truth} invalid_pred={invalid_pred}")
    for sample in invalid_examples:
        print(f"[invalid] {sample}")

    _print_metrics("overall", overall_metrics)
    for method, metrics in method_metrics.items():
        _print_metrics(f"method={method}", metrics)

    if args.output_json:
        payload = {
            "input_csv": args.input,
            "pred_col": args.pred_col,
            "truth_col": args.truth_col,
            "method_col": args.method_col,
            "rows": {
                "total": total_rows,
                "used": used_rows,
                "unlabeled": skipped_unlabeled,
                "invalid_truth": invalid_truth,
                "invalid_pred": invalid_pred,
            },
            "overall": overall_metrics,
            "by_method": method_metrics,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.output_json}")


if __name__ == "__main__":
    main()
