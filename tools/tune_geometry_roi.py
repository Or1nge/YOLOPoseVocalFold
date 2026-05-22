#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.geometry import angle_bisector_roi_from_three_points, polygon_area, polygon_containment_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune three-keypoint angle-bisector ROI geometry.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/yolo_pose"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--predictions", type=Path, help="Use predicted keypoints from a JSONL file.")
    parser.add_argument("--target-containment", type=float, default=0.87)
    parser.add_argument("--out-dir", type=Path, default=Path("Results/geometry_tuning/glottic_three_point"))
    parser.add_argument("--postprocess-out", type=Path)
    return parser.parse_args()


def read_records(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    path = dataset_dir / "roi_polygons" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"ROI metadata not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        stem = Path(str(payload.get("source", ""))).stem
        if stem:
            records[stem] = payload
    return records


def attach_predicted_keypoints(
    records: list[dict[str, Any]],
    predictions_path: Path | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if predictions_path is None:
        return records, []
    predictions = read_predictions(predictions_path)
    tuned_records = []
    skipped = []
    for record in records:
        prediction = predictions.get(str(record["stem"]))
        keypoints = prediction.get("keypoints") if prediction else None
        if not keypoints:
            skipped.append(str(record["stem"]))
            continue
        updated = dict(record)
        updated["keypoints"] = keypoints
        tuned_records.append(updated)
    return tuned_records, skipped


def score_candidate(
    records: list[dict[str, Any]],
    target_containment: float,
    base_backtrack_fraction: float,
    posterior_margin_fraction: float,
    side_margin_fraction: float,
) -> dict[str, Any]:
    containments = []
    area_ratios = []
    failures = []
    for record in records:
        roi = angle_bisector_roi_from_three_points(
            record["keypoints"],
            base_backtrack_fraction=base_backtrack_fraction,
            posterior_margin_fraction=posterior_margin_fraction,
            side_margin_fraction=side_margin_fraction,
        )
        target_polygon = record["manual_roi_polygon"]
        containment = polygon_containment_rate(target_polygon, roi.polygon)
        target_area = polygon_area(target_polygon)
        pred_area = polygon_area(roi.polygon)
        containments.append(containment)
        area_ratios.append(pred_area / target_area if target_area > 0 else 0.0)
        if containment < target_containment:
            failures.append(
                {
                    "stem": record["stem"],
                    "containment": containment,
                    "area_ratio_to_target": area_ratios[-1],
                }
            )
    return {
        "base_backtrack_fraction": base_backtrack_fraction,
        "posterior_margin_fraction": posterior_margin_fraction,
        "side_margin_fraction": side_margin_fraction,
        "count": len(records),
        "mean_containment": mean(containments) if containments else 0.0,
        "min_containment": min(containments) if containments else 0.0,
        "containment_ge_target_rate": (
            sum(value >= target_containment for value in containments) / len(containments)
            if containments
            else 0.0
        ),
        "mean_area_ratio_to_target": mean(area_ratios) if area_ratios else 0.0,
        "max_area_ratio_to_target": max(area_ratios) if area_ratios else 0.0,
        "failure_count": len(failures),
        "failures": failures[:25],
    }


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -float(candidate["containment_ge_target_rate"]),
        float(candidate["mean_area_ratio_to_target"]),
        float(candidate["max_area_ratio_to_target"]),
        -float(candidate["min_containment"]),
    )


def main() -> None:
    args = parse_args()
    records = read_records(args.dataset_dir, args.split)
    records, skipped = attach_predicted_keypoints(records, args.predictions)
    if not records:
        raise SystemExit("No records available for geometry tuning")
    candidates = []
    for base, posterior, side in itertools.product(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00],
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.30, 1.60, 2.00],
    ):
        candidates.append(
            score_candidate(
                records,
                target_containment=args.target_containment,
                base_backtrack_fraction=base,
                posterior_margin_fraction=posterior,
                side_margin_fraction=side,
            )
        )
    candidates.sort(key=candidate_sort_key)
    best = candidates[0]
    payload = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split": args.split,
        "predictions": str(args.predictions) if args.predictions is not None else None,
        "skipped_without_predictions": skipped,
        "target_containment": args.target_containment,
        "best": best,
        "top_candidates": candidates[:20],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{args.split}_geometry_tuning.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    postprocess_payload = {
        "roi_base_backtrack_fraction": best["base_backtrack_fraction"],
        "roi_posterior_margin_fraction": best["posterior_margin_fraction"],
        "roi_side_margin_fraction": best["side_margin_fraction"],
        "confidence_consistency_weight": 0.25,
        "fusion_mode": "angle_bisector",
    }
    if args.postprocess_out is not None:
        args.postprocess_out.parent.mkdir(parents=True, exist_ok=True)
        args.postprocess_out.write_text(
            yaml.safe_dump(postprocess_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    print(json.dumps({"best": best, "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
