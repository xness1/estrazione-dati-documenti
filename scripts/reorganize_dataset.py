#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from id_extractor.classifier import analyze_document_pages
from id_extractor.discovery import iter_candidate_files
from id_extractor.render import load_document_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Riorganizza dataset in modo content-based (no filename assumptions)."
    )
    parser.add_argument("--input", type=Path, default=Path("esempi documenti"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset_pulito"))
    parser.add_argument("--manifest", type=Path, default=Path("output/reorganization_manifest.json"))
    parser.add_argument(
        "--id-threshold",
        type=float,
        default=0.85,
        help="Confidence minima per ID card confermata.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_layout(root: Path) -> None:
    for rel in (
        "id_cards/confirmed",
        "id_cards/review",
        "other_documents",
        "unreadable",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_layout(args.output_root)

    manifest = []
    files = list(iter_candidate_files(args.input))

    for src in files:
        pages = load_document_images(src)
        file_hash = sha256_file(src)
        ext = src.suffix.lower()
        safe_name = f"{file_hash[:16]}{ext}"

        if not pages:
            dest_rel = f"unreadable/{safe_name}"
            _copy(src, args.output_root / dest_rel)
            manifest.append(
                {
                    "source": str(src),
                    "dest": str(args.output_root / dest_rel),
                    "class": "unreadable",
                    "confidence": 0.0,
                    "reason": "Unable to decode/render document pages",
                    "sha256": file_hash,
                }
            )
            continue

        evidence = analyze_document_pages(pages)

        if evidence.is_id_card and evidence.confidence >= args.id_threshold:
            bucket = "id_cards/confirmed"
            cls = "id_card_confirmed"
        elif evidence.is_id_card:
            bucket = "id_cards/review"
            cls = "id_card_review"
        else:
            bucket = "other_documents"
            cls = "other_document"

        dest_rel = f"{bucket}/{safe_name}"
        _copy(src, args.output_root / dest_rel)
        manifest.append(
            {
                "source": str(src),
                "dest": str(args.output_root / dest_rel),
                "class": cls,
                "confidence": evidence.confidence,
                "reason": evidence.reason,
                "sha256": file_hash,
                "content_evidence": evidence.to_dict(),
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "total_files": len(manifest),
        "id_card_confirmed": sum(1 for x in manifest if x["class"] == "id_card_confirmed"),
        "id_card_review": sum(1 for x in manifest if x["class"] == "id_card_review"),
        "other_document": sum(1 for x in manifest if x["class"] == "other_document"),
        "unreadable": sum(1 for x in manifest if x["class"] == "unreadable"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Manifest: {args.manifest}")
    print(f"Output root: {args.output_root}")


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


if __name__ == "__main__":
    main()

