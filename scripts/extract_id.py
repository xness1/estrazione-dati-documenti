#!/usr/bin/env python3
"""
Script CLI per estrazione dati da carte d'identità elettroniche italiane.

Uso:
    python scripts/extract_id.py documento.pdf
    python scripts/extract_id.py cartella/ -o risultati.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from id_extractor.hybrid_pipeline import process_document, MrzResult


def result_to_dict(r: MrzResult) -> dict:
    """Converte risultato in dizionario JSON-friendly."""
    return {
        "file": r.source_file,
        "status": r.status,
        "documento": {
            "numero": r.document_number,
            "scadenza": r.expiry_date,
        },
        "anagrafica": {
            "cognome": r.surname,
            "nome": r.given_name,
            "data_nascita": r.birth_date,
            "sesso": r.sex,
            "nazionalita": r.nationality,
        },
        "codice_fiscale": r.codice_fiscale,
        "indirizzo": r.address,
        "validazione": {
            "icao_valid": r.icao_valid,
            "tempo_ms": r.processing_time_ms,
        }
    }


def process_path(path: Path) -> list:
    """Processa file o cartella."""
    results = []
    
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("**/*.pdf")) + list(path.glob("**/*.jpg")) + list(path.glob("**/*.png"))
    
    for f in files:
        print(f"📄 {f.name}...", end=" ", flush=True)
        try:
            for r in process_document(f):
                if r.status == "confirmed":
                    print(f"✅ {r.surname} {r.given_name}")
                    results.append(result_to_dict(r))
                    break
                elif r.status == "not_found":
                    print("❌ non trovato")
                    break
                elif r.status == "review":
                    print("⚠️ review")
                    break
        except Exception as e:
            print(f"❌ errore: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Estrazione dati carte d'identità elettroniche")
    parser.add_argument("input", help="File PDF/immagine o cartella")
    parser.add_argument("-o", "--output", help="File JSON output (default: stdout)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Output solo JSON")
    args = parser.parse_args()
    
    path = Path(args.input)
    if not path.exists():
        print(f"Errore: {path} non esiste", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print(f"\n🔍 Elaborazione: {path}\n")
    
    results = process_path(path)
    
    if not args.quiet:
        print(f"\n📊 Trovati: {len(results)} documenti\n")
    
    output = json.dumps(results, indent=2, ensure_ascii=False)
    
    if args.output:
        Path(args.output).write_text(output)
        if not args.quiet:
            print(f"💾 Salvato in: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
