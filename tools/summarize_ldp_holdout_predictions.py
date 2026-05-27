#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FAILURE_FLAGS = (
    "keypoints_outside_image",
    "low_roi_area",
    "roi_area_too_small",
    "low_roi_dark_fraction",
    "roi_too_bright",
    "too_few_reliable_keypoints",
    "low_keypoint_confidence",
    "low_bbox_keypoint_consistency",
    "implausible_keypoint_angle",
    "keypoints_outside_final_box",
    "no_valid_pose_prediction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ROI predictions on an LDP holdout manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mixed-class-name", default="混杂图片")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Diagnostic reporting threshold only; ROI reject/manual_review actions come from postprocess config.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def stable_key(row: dict[str, Any]) -> str:
    for field in ("source_key", "original_source", "source"):
        value = row.get(field)
        if value:
            return str(Path(str(value)).resolve())
    return ""


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    confidence_threshold: float,
) -> dict[str, Any]:
    total = len(rows)
    actions = Counter(str(row.get("action", "")) for row in rows)
    flags = Counter(flag for row in rows for flag in row.get("flags", []))
    confidences = [float(row.get("final_confidence") or 0.0) for row in rows]
    dark_values = [
        float(row["roi_dark_fraction"])
        for row in rows
        if row.get("roi_dark_fraction") is not None
    ]
    return {
        "count": total,
        "actions": dict(actions),
        "auto_accept_rate": rate(actions["auto_accept"], total),
        "manual_review_rate": rate(actions["manual_review"], total),
        "reject_rate": rate(actions["reject_or_relabel"], total),
        "accepted_or_review_rate": rate(actions["auto_accept"] + actions["manual_review"], total),
        "final_confidence_mean": mean(confidences) if confidences else 0.0,
        "final_confidence_ge_threshold_rate": rate(
            sum(value >= confidence_threshold for value in confidences),
            total,
        ),
        "flags": dict(flags),
        "failure_flag_rates": {
            flag: rate(sum(flag in row.get("flags", []) for row in rows), total)
            for flag in FAILURE_FLAGS
        },
        "roi_dark_fraction_count": len(dark_values),
        "roi_dark_fraction_mean": mean(dark_values) if dark_values else None,
        "roi_dark_fraction_lt_0p20_rate": rate(sum(value < 0.20 for value in dark_values), len(dark_values)),
    }


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    prediction_rows = read_jsonl(args.predictions)
    predictions_by_key = {stable_key(row): row for row in prediction_rows if stable_key(row)}

    matched_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    rows_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest_row in manifest_rows:
        key = stable_key(manifest_row)
        prediction = predictions_by_key.get(key)
        if prediction is None:
            missing.append(manifest_row)
            continue
        class_name = str(manifest_row.get("class_name") or prediction.get("class_name") or "")
        row = dict(prediction)
        row["class_name"] = class_name
        row["holdout_source_key"] = key
        matched_rows.append(row)
        rows_by_class[class_name].append(row)

    summary = {
        "manifest_count": len(manifest_rows),
        "prediction_count": len(prediction_rows),
        "matched_count": len(matched_rows),
        "missing_count": len(missing),
        "confidence_threshold": args.confidence_threshold,
        "overall": summarize_rows(matched_rows, confidence_threshold=args.confidence_threshold),
        "by_class": {
            class_name: summarize_rows(rows, confidence_threshold=args.confidence_threshold)
            for class_name, rows in sorted(rows_by_class.items())
        },
    }
    mixed_rows = rows_by_class.get(args.mixed_class_name, [])
    if mixed_rows:
        mixed_summary = summarize_rows(mixed_rows, confidence_threshold=args.confidence_threshold)
        summary["mixed_class_name"] = args.mixed_class_name
        summary["mixed_false_positive_rate"] = mixed_summary["accepted_or_review_rate"]
        summary["mixed_auto_accept_rate"] = mixed_summary["auto_accept_rate"]
        summary["mixed_final_confidence_ge_threshold_rate"] = mixed_summary[
            "final_confidence_ge_threshold_rate"
        ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "ldp_holdout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = args.out_dir / "ldp_holdout_class_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "class_name",
            "count",
            "auto_accept_rate",
            "manual_review_rate",
            "reject_rate",
            "accepted_or_review_rate",
            "final_confidence_mean",
            "final_confidence_ge_threshold_rate",
            "keypoints_outside_image_rate",
            "low_roi_area_rate",
            "roi_area_too_small_rate",
            "low_roi_dark_fraction_rate",
            "roi_too_bright_rate",
            "roi_dark_fraction_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, item in summary["by_class"].items():
            failure_rates = item["failure_flag_rates"]
            writer.writerow(
                {
                    "class_name": class_name,
                    "count": item["count"],
                    "auto_accept_rate": item["auto_accept_rate"],
                    "manual_review_rate": item["manual_review_rate"],
                    "reject_rate": item["reject_rate"],
                    "accepted_or_review_rate": item["accepted_or_review_rate"],
                    "final_confidence_mean": item["final_confidence_mean"],
                    "final_confidence_ge_threshold_rate": item["final_confidence_ge_threshold_rate"],
                    "keypoints_outside_image_rate": failure_rates["keypoints_outside_image"],
                    "low_roi_area_rate": failure_rates["low_roi_area"],
                    "roi_area_too_small_rate": failure_rates["roi_area_too_small"],
                    "low_roi_dark_fraction_rate": failure_rates["low_roi_dark_fraction"],
                    "roi_too_bright_rate": failure_rates["roi_too_bright"],
                    "roi_dark_fraction_mean": item["roi_dark_fraction_mean"],
                }
            )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
