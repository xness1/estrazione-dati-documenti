#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from id_extractor.discovery import iter_candidate_files


TYPE_KEYWORDS = {
    "id_card": ("c.id", "carta id", "carta identit", "documento ident", "ci "),
    "tax_code": ("cf", "codice fiscale", "tessera sanitaria"),
    "license": ("patente",),
    "company_visura": ("visura",),
    "permesso": ("permesso di soggiorno",),
}


def guess_doc_type(name: str) -> str:
    n = name.lower()
    for doc_type, keys in TYPE_KEYWORDS.items():
        if any(k in n for k in keys):
            return doc_type
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indicizza documenti in cartelle disordinate.")
    parser.add_argument("--input", type=Path, default=Path("esempi documenti"))
    parser.add_argument("--output-json", type=Path, default=Path("output/document_manifest.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("output/document_manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = []
    for path in iter_candidate_files(args.input):
        rel = path.relative_to(args.input).as_posix()
        items.append(
            {
                "relative_path": rel,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "doc_type_guess": guess_doc_type(path.name),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["relative_path", "extension", "size_bytes", "doc_type_guess"]
        )
        writer.writeheader()
        writer.writerows(items)

    print(f"Indicizzati {len(items)} file")
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

