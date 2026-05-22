#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


CLASS_ORDER = ["Non-Vocal-Cord", "Normal", "Benign", "Malignant"]
DEFAULT_FOLDER_LABELS = {
    "混杂图片": "Non-Vocal-Cord",
    "正常": "Normal",
    "正常声带": "Normal",
    "喉癌": "Malignant",
    "声带任克": "Benign",
    "声带任克水肿": "Benign",
    "声带囊肿": "Benign",
    "声带小结": "Benign",
    "声带息肉": "Benign",
    "声带白斑": "Benign",
    "声带肉芽": "Benign",
    "声带肉芽肿": "Benign",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a laryngeal 4-class checkpoint on a folder dataset.")
    parser.add_argument("--shared-py", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_shared(shared_py: Path):
    module_name = f"larynx_shared_eval_{abs(hash(shared_py.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, shared_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import shared module from {shared_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    original_set_interop = torch.set_num_interop_threads

    def safe_set_interop_threads(n: int) -> None:
        try:
            original_set_interop(n)
        except RuntimeError as exc:
            if "cannot set number of interop threads" not in str(exc):
                raise

    torch.set_num_interop_threads = safe_set_interop_threads
    try:
        spec.loader.exec_module(module)
    finally:
        torch.set_num_interop_threads = original_set_interop
    return module


def load_state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def iter_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_root)
        if not rel.parts:
            continue
        folder = rel.parts[0]
        label = DEFAULT_FOLDER_LABELS.get(folder)
        if label is None:
            raise ValueError(f"Unknown external label folder: {folder}")
        rows.append(
            {
                "image_path": str(path),
                "relative_path": str(rel),
                "folder": folder,
                "true_label": label,
                "true_id": CLASS_ORDER.index(label),
            }
        )
    if not rows:
        raise RuntimeError(f"No images found under {dataset_root}")
    return rows


class FolderDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
            return self.transform(image), index, ""
        except (OSError, UnidentifiedImageError) as exc:
            blank = Image.new("RGB", (224, 224), color=(0, 0, 0))
            return self.transform(blank), index, str(exc)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shared = load_shared(args.shared_py)
    cfg = shared.load_config(str(args.config), sync_with_split=True)
    shared.init_label_mapping(cfg)
    class_names = [shared.DISPLAY_NAMES[i] for i in range(len(shared.LABEL_DICT))]
    if class_names != CLASS_ORDER:
        raise RuntimeError(f"Class order mismatch: {class_names} != {CLASS_ORDER}")

    rows = iter_rows(args.dataset_root)
    _train_tf, eval_tf = shared.build_transforms(cfg)
    dataset = FolderDataset(rows, eval_tf)
    batch_size = args.batch_size if args.batch_size > 0 else int(cfg.get("eval_batch_size", 128))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_kwargs)

    device = shared.setup_device()
    original_create_model = shared.timm.create_model

    def create_model_no_pretrained(*model_args, **model_kwargs):
        model_kwargs["pretrained"] = False
        return original_create_model(*model_args, **model_kwargs)

    shared.timm.create_model = create_model_no_pretrained
    try:
        model = shared.HierarchicalImageClassifier(num_classes=len(CLASS_ORDER), cfg=cfg).to(device)
    finally:
        shared.timm.create_model = original_create_model
    model.load_state_dict(load_state_dict(args.checkpoint, device), strict=True)
    model.eval()

    probs = np.zeros((len(rows), len(CLASS_ORDER)), dtype=np.float32)
    errors: dict[int, str] = {}
    with torch.inference_mode():
        for images, indices, batch_errors in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            batch_probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            idx_np = indices.detach().cpu().numpy()
            probs[idx_np] = batch_probs
            for idx, err in zip(idx_np.tolist(), batch_errors):
                if err:
                    errors[int(idx)] = str(err)

    y_true = np.array([row["true_id"] for row in rows], dtype=np.int64)
    y_pred = probs.argmax(axis=1)
    for idx, row in enumerate(rows):
        row["pred_label"] = CLASS_ORDER[int(y_pred[idx])]
        row["pred_id"] = int(y_pred[idx])
        row["correct"] = bool(y_pred[idx] == y_true[idx])
        row["error"] = errors.get(idx, "")
        for class_idx, class_name in enumerate(CLASS_ORDER):
            row[f"prob_{class_name}"] = float(probs[idx, class_idx])

    summary = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "n": int(len(rows)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(CLASS_ORDER))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(len(CLASS_ORDER))), average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_ORDER))),
            target_names=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER)))).tolist(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
