#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy an image tree while adding a black border to each image.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.30)
    parser.add_argument("--min-padding", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def save_padded_image(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, quality=max(1, min(int(quality), 100)), subsampling=0)
    else:
        image.save(path)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    counts = {"padded_images": 0, "copied_non_images": 0, "errors": 0}
    for source in iter_files(source_root):
        rel = source.relative_to(source_root)
        dest = out_dir / rel
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            counts["copied_non_images"] += 1
            continue
        try:
            with Image.open(source) as image:
                image = image.convert("RGB")
                original_width, original_height = image.size
                pad = max(int(round(max(original_width, original_height) * args.fraction)), int(args.min_padding))
                padded = ImageOps.expand(image, border=(pad, pad, pad, pad), fill=(0, 0, 0))
                save_padded_image(padded, dest, args.jpeg_quality)
            rows.append(
                {
                    "source_path": str(source),
                    "relative_path": str(rel),
                    "output_path": str(dest),
                    "original_width_before_blackpad": original_width,
                    "original_height_before_blackpad": original_height,
                    "width": original_width + pad * 2,
                    "height": original_height + pad * 2,
                    "blackpad_left": pad,
                    "blackpad_top": pad,
                    "blackpad_right": pad,
                    "blackpad_bottom": pad,
                    "blackpad_fraction": args.fraction,
                }
            )
            counts["padded_images"] += 1
        except Exception as exc:  # noqa: BLE001 - keep per-file failure evidence.
            rows.append(
                {
                    "source_path": str(source),
                    "relative_path": str(rel),
                    "output_path": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            counts["errors"] += 1

    manifest_path = out_dir / "blackpad_manifest.csv"
    fieldnames = [
        "source_path",
        "relative_path",
        "output_path",
        "original_width_before_blackpad",
        "original_height_before_blackpad",
        "width",
        "height",
        "blackpad_left",
        "blackpad_top",
        "blackpad_right",
        "blackpad_bottom",
        "blackpad_fraction",
        "error",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_root": str(source_root),
        "out_dir": str(out_dir),
        "fraction": args.fraction,
        "min_padding": args.min_padding,
        "counts": counts,
        "manifest": str(manifest_path),
    }
    (out_dir / "blackpad_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if counts["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
