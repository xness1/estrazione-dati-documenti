#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from id_extractor.config import PipelineConfig
from id_extractor.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estrazione retro carta identita (MRZ + indirizzo) da cartelle sporche."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("esempi documenti"),
        help="Cartella radice con documenti e sottocartelle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/id_back_extraction.json"),
        help="File JSON di output.",
    )
    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Salva immagini crop MRZ/address in output/debug.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        root_input=args.input,
        output_json=args.output,
        debug_save_images=args.debug_images,
    )
    results = run_pipeline(config)
    print(f"Completato: trovati {len(results)} retro carta ID probabili.")
    print(f"Output: {config.output_json}")


if __name__ == "__main__":
    main()

