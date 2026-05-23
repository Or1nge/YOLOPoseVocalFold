#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.preprocess import blackpad_image_file  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone a YOLO-Pose dataset and add negative background images with empty labels."
    )
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--negative-source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--all", action="store_true", help="Use all readable negatives after exclusions.")
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="JSONL manifest whose source/original_source/source_key images must be excluded.",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--prefix", default="mixed_negative")
    parser.add_argument(
        "--blackpad-negatives",
        action="store_true",
        help="Apply the V1.1 black-border preprocessing to added negative images.",
    )
    parser.add_argument("--blackpad-fraction", type=float, default=0.30)
    parser.add_argument("--blackpad-min-padding", type=int, default=80)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.width, image.height


def read_excluded_paths(path: Path | None) -> set[Path]:
    if path is None:
        return set()
    excluded: set[Path] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("source_key", "original_source", "source"):
            value = row.get(key)
            if value:
                excluded.add(Path(str(value)).resolve())
                break
    return excluded


def copy_base_dataset(base_dataset: Path, out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {out_dir}")
        shutil.rmtree(out_dir)
    shutil.copytree(base_dataset, out_dir, symlinks=True)


def update_dataset_yaml(out_dir: Path) -> dict[str, Any]:
    yaml_path = out_dir / "vocal_fold_pose.yaml"
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    payload["path"] = str(out_dir.resolve())
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return payload


def add_negatives(args: argparse.Namespace) -> dict[str, Any]:
    if args.count <= 0 and not args.all:
        raise ValueError("--count must be positive")
    if not args.base_dataset.exists():
        raise FileNotFoundError(args.base_dataset)
    if not args.negative_source_dir.exists():
        raise FileNotFoundError(args.negative_source_dir)

    copy_base_dataset(args.base_dataset, args.out_dir, overwrite=args.overwrite)
    dataset_yaml = update_dataset_yaml(args.out_dir)

    excluded_paths = read_excluded_paths(args.exclude_manifest)
    candidates = []
    for image_path in iter_images(args.negative_source_dir):
        if image_path.resolve() in excluded_paths:
            continue
        try:
            width, height = verify_image(image_path)
        except Exception:
            continue
        candidates.append({"path": image_path, "width": width, "height": height})
    if not args.all and len(candidates) < args.count:
        raise ValueError(f"Only {len(candidates)} readable images found, need {args.count}")

    rng = random.Random(args.seed)
    selected = list(candidates) if args.all else rng.sample(candidates, args.count)

    images_dir = args.out_dir / "images" / args.split
    labels_dir = args.out_dir / "labels" / args.split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, item in enumerate(selected, start=1):
        source_path: Path = item["path"]
        dest_name = f"{args.prefix}_{index:03d}__{source_path.stem}{source_path.suffix.lower()}"
        image_dest = images_dir / dest_name
        label_dest = labels_dir / f"{Path(dest_name).stem}.txt"
        preprocess: dict[str, Any] | None = None
        if args.blackpad_negatives:
            preprocess = blackpad_image_file(
                source_path,
                image_dest,
                fraction=args.blackpad_fraction,
                min_padding=args.blackpad_min_padding,
                jpeg_quality=args.jpeg_quality,
            ).to_dict()
        else:
            shutil.copy2(source_path, image_dest)
        label_dest.write_text("", encoding="utf-8")
        output_width = preprocess["padded_width"] if preprocess else item["width"]
        output_height = preprocess["padded_height"] if preprocess else item["height"]
        rows.append(
            {
                "index": index,
                "split": args.split,
                "source_path": str(source_path.resolve()),
                "image_path": str(image_dest.resolve()),
                "label_path": str(label_dest.resolve()),
                "original_width": item["width"],
                "original_height": item["height"],
                "width": output_width,
                "height": output_height,
                "preprocess_type": preprocess["type"] if preprocess else "none",
                "blackpad_padding_px": preprocess["padding_px"] if preprocess else 0,
                "label_type": "empty_negative",
            }
        )

    manifest_csv = args.out_dir / "negative_samples_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "base_dataset": str(args.base_dataset.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "negative_source_dir": str(args.negative_source_dir.resolve()),
        "split": args.split,
        "count": len(rows),
        "requested_count": "all" if args.all else args.count,
        "seed": args.seed,
        "exclude_manifest": str(args.exclude_manifest.resolve()) if args.exclude_manifest else None,
        "excluded_source_count": len(excluded_paths),
        "blackpad_negatives": bool(args.blackpad_negatives),
        "blackpad_fraction": args.blackpad_fraction if args.blackpad_negatives else None,
        "blackpad_min_padding": args.blackpad_min_padding if args.blackpad_negatives else None,
        "label_contract": "empty YOLO label file means background/no glottic ROI",
        "dataset_yaml": dataset_yaml,
        "samples": rows,
    }
    (args.out_dir / "negative_samples_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = add_negatives(args)
    print(json.dumps({"out_dir": manifest["out_dir"], "count": manifest["count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
