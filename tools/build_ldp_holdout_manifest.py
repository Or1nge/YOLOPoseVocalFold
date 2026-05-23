#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic LDP holdout manifest from ROI predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260523)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def record_class(record: dict[str, Any]) -> str:
    source = Path(str(record.get("original_source") or record.get("source") or ""))
    return source.parent.name


def record_key(record: dict[str, Any]) -> str:
    return str(Path(str(record.get("original_source") or record.get("source"))).resolve())


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(args.predictions):
        key = record_key(record)
        if not key:
            continue
        grouped[record_class(record)].append(record)

    selected = []
    counts: Counter[str] = Counter()
    for class_name in sorted(grouped):
        records = sorted(grouped[class_name], key=record_key)
        rng.shuffle(records)
        for record in records[: max(args.per_class, 0)]:
            item = {
                "source": record.get("source"),
                "original_source": record.get("original_source") or record.get("source"),
                "class_name": class_name,
                "source_key": record_key(record),
                "final_confidence": record.get("final_confidence"),
                "action": record.get("action"),
                "flags": record.get("flags", []),
            }
            selected.append(item)
            counts[class_name] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected),
        encoding="utf-8",
    )
    summary = {
        "predictions": str(args.predictions.resolve()),
        "out": str(args.out.resolve()),
        "per_class": args.per_class,
        "seed": args.seed,
        "counts": dict(counts),
        "total": len(selected),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
