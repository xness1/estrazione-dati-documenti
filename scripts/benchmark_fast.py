#!/usr/bin/env python3
"""
Benchmark pipeline veloce su documenti reali.
Misura: tempo, precision, recall sul riconoscimento retro tessera.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from id_extractor.fast_pipeline import (
    FastConfig,
    TimeoutConfig,
    process_file_fast,
    quick_detect_back_tessera,
    extract_mrz_full,
    load_image_fast,
)


@dataclass
class BenchmarkCase:
    """Caso di test con ground truth."""
    file_path: str
    expected_is_back: bool
    expected_has_mrz: bool
    description: str


@dataclass
class BenchmarkResult:
    """Risultato singolo benchmark."""
    case: BenchmarkCase
    detected_back: bool
    mrz_valid: bool
    time_ms: int
    mrz_surname: Optional[str] = None
    mrz_birth: Optional[str] = None
    mrz_cf: Optional[str] = None
    error: Optional[str] = None


def run_single_benchmark(case: BenchmarkCase, config: FastConfig) -> BenchmarkResult:
    """Esegue benchmark su singolo file."""
    start = time.time()
    
    path = Path(case.file_path)
    if not path.exists():
        return BenchmarkResult(
            case=case,
            detected_back=False,
            mrz_valid=False,
            time_ms=0,
            error=f"File not found: {case.file_path}"
        )
    
    results = process_file_fast(path, config)
    elapsed_ms = int((time.time() - start) * 1000)
    
    detected_back = False
    mrz_valid = False
    mrz_surname = None
    mrz_birth = None
    mrz_cf = None
    
    for r in results:
        if r.status == "confirmed":
            detected_back = True
            mrz_valid = True
            if r.mrz:
                mrz_surname = r.mrz.surname
                mrz_birth = r.mrz.birth_date
                mrz_cf = r.mrz.codice_fiscale
        elif r.status == "review":
            detected_back = True
            if r.mrz and r.mrz.is_valid_td1:
                mrz_valid = True
    
    return BenchmarkResult(
        case=case,
        detected_back=detected_back,
        mrz_valid=mrz_valid,
        time_ms=elapsed_ms,
        mrz_surname=mrz_surname,
        mrz_birth=mrz_birth,
        mrz_cf=mrz_cf
    )


def calculate_metrics(results: List[BenchmarkResult]) -> dict:
    """Calcola precision, recall, F1."""
    tp = fp = tn = fn = 0
    
    for r in results:
        if r.case.expected_is_back:
            if r.detected_back:
                tp += 1
            else:
                fn += 1
        else:
            if r.detected_back:
                fp += 1
            else:
                tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mrz_tp = mrz_fn = 0
    for r in results:
        if r.case.expected_has_mrz:
            if r.mrz_valid:
                mrz_tp += 1
            else:
                mrz_fn += 1
    
    mrz_recall = mrz_tp / (mrz_tp + mrz_fn) if (mrz_tp + mrz_fn) > 0 else 0.0
    
    return {
        "detection": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1)
        },
        "mrz_extraction": {
            "valid_extractions": mrz_tp,
            "failed_extractions": mrz_fn,
            "recall": round(mrz_recall * 100, 1)
        }
    }


def print_report(results: List[BenchmarkResult], metrics: dict) -> None:
    """Stampa report benchmark."""
    print("\n" + "=" * 70)
    print("BENCHMARK PIPELINE VELOCE - RETRO CARTA IDENTITÀ ELETTRONICA")
    print("=" * 70)
    
    print("\n--- RISULTATI PER FILE ---\n")
    
    total_time = 0
    for r in results:
        status = "✓" if r.detected_back == r.case.expected_is_back else "✗"
        mrz_status = "✓" if r.mrz_valid == r.case.expected_has_mrz else "✗"
        
        print(f"{status} {Path(r.case.file_path).name[:40]:<40}")
        print(f"   Descrizione: {r.case.description}")
        print(f"   Tempo: {r.time_ms:>5} ms | Atteso back: {r.case.expected_is_back} | Rilevato: {r.detected_back}")
        print(f"   MRZ: {mrz_status} Atteso: {r.case.expected_has_mrz} | Valido: {r.mrz_valid}")
        
        if r.mrz_surname:
            print(f"   → Cognome: {r.mrz_surname}")
        if r.mrz_birth:
            print(f"   → Data nascita: {r.mrz_birth}")
        if r.mrz_cf:
            print(f"   → Codice Fiscale: {r.mrz_cf}")
        if r.error:
            print(f"   ⚠ Errore: {r.error}")
        
        total_time += r.time_ms
        print()
    
    print("\n--- METRICHE AGGREGATE ---\n")
    
    det = metrics["detection"]
    print(f"DETECTION RETRO TESSERA:")
    print(f"   Precision: {det['precision']:.1f}%")
    print(f"   Recall:    {det['recall']:.1f}%")
    print(f"   F1 Score:  {det['f1']:.1f}%")
    print(f"   (TP={det['true_positives']}, FP={det['false_positives']}, TN={det['true_negatives']}, FN={det['false_negatives']})")
    
    mrz = metrics["mrz_extraction"]
    print(f"\nESTRAZIONE MRZ:")
    print(f"   Recall: {mrz['recall']:.1f}%")
    print(f"   (Valide={mrz['valid_extractions']}, Fallite={mrz['failed_extractions']})")
    
    print(f"\nPERFORMANCE:")
    print(f"   Tempo totale:  {total_time} ms")
    print(f"   Tempo medio:   {total_time // len(results)} ms/documento")
    print(f"   Target:        3000-5000 ms/documento")
    
    if total_time // len(results) <= 5000:
        print(f"   ✓ TARGET RAGGIUNTO!")
    else:
        print(f"   ✗ TARGET NON RAGGIUNTO")
    
    print("\n" + "=" * 70)


def discover_test_cases(base_path: Path) -> List[BenchmarkCase]:
    """
    Scopre automaticamente i casi di test dai documenti disponibili.
    """
    cases = []
    
    retro_patterns = ["retro", "back", "_2", "Pagina_2"]
    fronte_patterns = ["fronte", "front", "_1", "Pagina_1"]
    
    for f in base_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".pdf"):
            continue
        if "Zone.Identifier" in f.name or ":Zone" in str(f):
            continue
        
        name_lower = f.name.lower()
        
        if any(p.lower() in name_lower for p in retro_patterns):
            if "c.id" in name_lower or "carta" in name_lower or "identit" in name_lower:
                cases.append(BenchmarkCase(
                    file_path=str(f),
                    expected_is_back=True,
                    expected_has_mrz=True,
                    description=f"Retro tessera - {f.parent.name}"
                ))
        elif any(p.lower() in name_lower for p in fronte_patterns):
            if "c.id" in name_lower or "carta" in name_lower or "identit" in name_lower:
                cases.append(BenchmarkCase(
                    file_path=str(f),
                    expected_is_back=False,
                    expected_has_mrz=False,
                    description=f"Fronte tessera (skip) - {f.parent.name}"
                ))
    
    return cases


def main():
    base_path = Path("/workspace/esempi documenti")
    
    test_cases = [
        BenchmarkCase(
            file_path="/workspace/esempi documenti/Training documenti/A.L.A. Srl/C.ID Bisi retro.JPG",
            expected_is_back=True,
            expected_has_mrz=True,
            description="Retro tessera Bisi - immagine JPG"
        ),
        BenchmarkCase(
            file_path="/workspace/esempi documenti/Training documenti/Briani/C.Id Briani retro.jpeg",
            expected_is_back=True,
            expected_has_mrz=True,
            description="Retro tessera Briani - immagine JPEG"
        ),
        BenchmarkCase(
            file_path="/workspace/esempi documenti/Training documenti/Martinelli/MARTINELLI_CI_sc_27_01_30_Pagina_2.jpg",
            expected_is_back=True,
            expected_has_mrz=True,
            description="Retro tessera Martinelli - scan pagina 2"
        ),
        BenchmarkCase(
            file_path="/workspace/esempi documenti/Training documenti/A.L.A. Srl/C.ID Bisi fronte.JPG",
            expected_is_back=False,
            expected_has_mrz=False,
            description="Fronte tessera Bisi - deve essere ignorato"
        ),
        BenchmarkCase(
            file_path="/workspace/esempi documenti/Training documenti/Briani/C.Id Briani Fronte.jpeg",
            expected_is_back=False,
            expected_has_mrz=False,
            description="Fronte tessera Briani - deve essere ignorato"
        ),
        BenchmarkCase(
            file_path="/workspace/esempi documenti/Training documenti/Martinelli/MARTINELLI_CI_sc_27_01_30_Pagina_1.jpg",
            expected_is_back=False,
            expected_has_mrz=False,
            description="Fronte tessera Martinelli - deve essere ignorato"
        ),
    ]
    
    existing_cases = [c for c in test_cases if Path(c.file_path).exists()]
    
    if len(existing_cases) < 3:
        print("Cercando altri documenti di test...")
        discovered = discover_test_cases(base_path)
        for dc in discovered:
            if dc.file_path not in [c.file_path for c in existing_cases]:
                existing_cases.append(dc)
                if len(existing_cases) >= 6:
                    break
    
    print(f"Trovati {len(existing_cases)} casi di test")
    
    config = FastConfig(
        timeouts=TimeoutConfig(
            file_total_sec=5.0,
            page_render_sec=2.0,
            ocr_call_sec=1.5,
            mrz_extraction_sec=3.0,
            address_extraction_sec=1.0
        ),
        quick_detect_dpi=100,
        full_extract_dpi=200,
        debug_mode=True
    )
    
    results = []
    for case in existing_cases:
        print(f"Testing: {Path(case.file_path).name}...", end=" ", flush=True)
        result = run_single_benchmark(case, config)
        print(f"{result.time_ms}ms")
        results.append(result)
    
    metrics = calculate_metrics(results)
    
    print_report(results, metrics)
    
    output_path = Path("/workspace/output/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "results": [
            {
                "file": r.case.file_path,
                "description": r.case.description,
                "expected_back": r.case.expected_is_back,
                "detected_back": r.detected_back,
                "mrz_valid": r.mrz_valid,
                "time_ms": r.time_ms,
                "mrz_data": {
                    "surname": r.mrz_surname,
                    "birth_date": r.mrz_birth,
                    "codice_fiscale": r.mrz_cf
                } if r.mrz_surname else None
            }
            for r in results
        ],
        "metrics": metrics
    }
    
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    main()
