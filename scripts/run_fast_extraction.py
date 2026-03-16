#!/usr/bin/env python3
"""
CLI per pipeline veloce estrazione retro carta identità elettronica.
Target: max 3-5 secondi per documento su CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from id_extractor.fast_pipeline import (
    FastConfig,
    TimeoutConfig,
    process_file_fast,
    run_fast_pipeline,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline veloce per estrazione retro carta identità elettronica"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="File o directory da processare"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/fast_extraction.json",
        help="File JSON output per risultati confermati"
    )
    parser.add_argument(
        "--review", "-r",
        default="output/review_queue.json",
        help="File JSON output per risultati da review"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=5.0,
        help="Timeout totale per file in secondi (default: 5.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Abilita debug mode"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Errore: {input_path} non esiste")
        sys.exit(1)
    
    config = FastConfig(
        root_input=input_path if input_path.is_dir() else input_path.parent,
        output_json=Path(args.output),
        review_json=Path(args.review),
        timeouts=TimeoutConfig(
            file_total_sec=args.timeout,
            page_render_sec=2.0,
            ocr_call_sec=1.5,
            mrz_extraction_sec=3.0,
            address_extraction_sec=1.0
        ),
        debug_mode=args.debug
    )
    
    if input_path.is_file():
        print(f"Processing single file: {input_path.name}")
        results = process_file_fast(input_path, config)
        
        for r in results:
            print(f"\nStatus: {r.status}")
            print(f"Time: {r.processing_time_ms}ms")
            
            if r.mrz and r.mrz.is_valid_td1:
                print(f"MRZ Valid: {r.mrz.is_valid_td1} (conf: {r.mrz.confidence:.2f})")
                print(f"  Cognome: {r.mrz.surname}")
                print(f"  Nome: {r.mrz.given_name}")
                print(f"  Data nascita: {r.mrz.birth_date}")
                print(f"  Scadenza: {r.mrz.expiry_date}")
                print(f"  Sesso: {r.mrz.sex}")
                if r.mrz.codice_fiscale:
                    print(f"  Codice Fiscale: {r.mrz.codice_fiscale}")
            
            if r.address and r.address.raw_text:
                print(f"Address (raw): {r.address.raw_text[:100]}...")
            
            if r.error:
                print(f"Error: {r.error}")
    else:
        print(f"Processing directory: {input_path}")
        result = run_fast_pipeline(config)
        
        print(f"\n--- Risultati ---")
        print(f"File processati: {result['stats']['total_files']}")
        print(f"Pagine processate: {result['stats']['total_pages']}")
        print(f"Confermati: {result['stats']['confirmed']}")
        print(f"Da review: {result['stats']['review']}")
        print(f"Saltati: {result['stats']['skipped']}")
        print(f"Errori: {result['stats']['errors']}")
        print(f"\nOutput: {result['confirmed_output']}")
        print(f"Review: {result['review_output']}")


if __name__ == "__main__":
    main()
