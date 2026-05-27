#!/usr/bin/env python3
"""Batch screen-photo laryngoscope pre-cropper with contact-sheet preview.

Uses the reusable detection and cropping logic from
``yoloposevf.screen_photo_crop``.  For integrated ROI workflow use
``tools/predict_roi.py`` which runs this pre-crop automatically.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloposevf.preprocess import IMAGE_EXTENSIONS
from yoloposevf.screen_photo_crop import classify_screen_photo, crop_screen_photo_window


def make_contact_sheet(files: list[Path], output: Path, title: str) -> None:
    thumb_w, thumb_h = 170, 170
    cols = 6
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 34) + 24), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), title, fill=(0, 0, 0))
    for i, path in enumerate(files):
        im = Image.open(path).convert("RGB")
        size = im.size
        im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (i % cols) * thumb_w + (thumb_w - im.width) // 2
        y = 24 + (i // cols) * (thumb_h + 34)
        sheet.paste(im, (x, y))
        draw.text(
            ((i % cols) * thumb_w + 4, y + thumb_h + 2),
            f"{i + 1}:{path.stem[-8:]} {size[0]}x{size[1]}",
            fill=(0, 0, 0),
        )
    sheet.save(output, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch screen-photo laryngoscope pre-cropper."
    )
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--reference-count", type=int, default=0)
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in args.src.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    rows: list[dict[str, object]] = []

    for index, path in enumerate(files, 1):
        out = args.dst / path.name
        with Image.open(path) as img:
            img = img.convert("RGB")
            needs_crop, signals = classify_screen_photo(img)
            if index <= args.reference_count:
                shutil.copy2(path, out)
                box = (0, 0, img.width, img.height)
                mode = "reference_copy"
            elif not needs_crop:
                shutil.copy2(path, out)
                box = (0, 0, img.width, img.height)
                mode = "no_crop"
            else:
                cropped, box = crop_screen_photo_window(img)
                cropped.save(out, quality=95, subsampling=0)
                mode = "cropped"
        rows.append(
            {
                "index": index,
                "file": path.name,
                "mode": mode,
                "x0": box[0],
                "y0": box[1],
                "x1": box[2],
                "y1": box[3],
                "out_width": box[2] - box[0],
                "out_height": box[3] - box[1],
                "stripe_col": round(signals["stripe_col"], 4),
                "stripe_row": round(signals["stripe_row"], 4),
                "blue_col": round(signals["blue_col"], 4),
                "blue_row": round(signals["blue_row"], 4),
            }
        )

    manifest = args.dst / "crop_manifest.csv"
    fieldnames = [
        "index",
        "file",
        "mode",
        "x0",
        "y0",
        "x1",
        "y1",
        "out_width",
        "out_height",
        "stripe_col",
        "stripe_row",
        "blue_col",
        "blue_row",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.preview:
        make_contact_sheet(sorted(args.dst.glob("*.jpg")), args.preview, f"{args.src.name} crop preview")

    print(f"processed={len(files)} dst={args.dst}")


if __name__ == "__main__":
    main()
