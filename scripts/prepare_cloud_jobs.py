#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara job JSONL per cloud agent.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/reorganization_manifest.json"),
        help="Manifest prodotto da reorganize_dataset.py",
    )
    parser.add_argument(
        "--jobs-out",
        type=Path,
        default=Path("output/cloud_jobs.jsonl"),
        help="Output JSONL jobs per coda cloud.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = []

    for item in items:
        cls = item.get("class")
        if cls == "id_card_confirmed":
            job_type = "extract_id_back"
            priority = "high"
        elif cls == "id_card_review":
            job_type = "review_or_vision_fallback"
            priority = "high"
        elif cls == "unreadable":
            job_type = "manual_check_unreadable"
            priority = "medium"
        else:
            continue

        jobs.append(
            {
                "job_id": item["sha256"][:24],
                "job_type": job_type,
                "priority": priority,
                "source_path": item["dest"],
                "source_sha256": item["sha256"],
                "assume_from_filename": False,
                "classification_confidence": item.get("confidence", 0.0),
                "classification_reason": item.get("reason", ""),
            }
        )

    args.jobs_out.parent.mkdir(parents=True, exist_ok=True)
    with args.jobs_out.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    print(f"Creati {len(jobs)} job cloud")
    print(f"File jobs: {args.jobs_out}")


if __name__ == "__main__":
    main()

