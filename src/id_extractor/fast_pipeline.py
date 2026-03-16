"""
Fast pipeline per riconoscimento retro carta d'identità elettronica (tessera).
Target: max 3-5 secondi per documento su CPU.

Strategia fail-fast:
1. Quick detection con timeout hard
2. MRZ detection prioritaria (senza MRZ → review)
3. Solo dopo address + codice fiscale
"""
from __future__ import annotations

import json
import re
import signal
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

import cv2
import fitz
import numpy as np
import pytesseract

fitz.TOOLS.mupdf_display_errors(False)


T = TypeVar("T")


@dataclass
class TimeoutConfig:
    """Hard timeouts per ogni fase della pipeline."""
    file_total_sec: float = 5.0
    page_render_sec: float = 1.5
    ocr_call_sec: float = 1.0
    mrz_extraction_sec: float = 1.5
    address_extraction_sec: float = 1.0


@dataclass
class FastConfig:
    """Configurazione pipeline veloce."""
    root_input: Path = field(default_factory=lambda: Path("."))
    output_json: Path = field(default_factory=lambda: Path("output/fast_extraction.json"))
    review_json: Path = field(default_factory=lambda: Path("output/review_queue.json"))
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    
    quick_detect_dpi: int = 100
    full_extract_dpi: int = 200
    min_mrz_confidence: float = 0.7
    debug_mode: bool = False


@dataclass 
class MrzResult:
    """Risultato estrazione MRZ TD1 completo."""
    raw_lines: List[str]
    
    document_type: Optional[str] = None
    issuing_country: Optional[str] = None
    document_number: Optional[str] = None
    
    surname: Optional[str] = None
    given_name: Optional[str] = None
    
    birth_date: Optional[str] = None
    sex: Optional[str] = None
    expiry_date: Optional[str] = None
    nationality: Optional[str] = None
    
    codice_fiscale: Optional[str] = None
    
    birth_date_check: Optional[str] = None
    expiry_date_check: Optional[str] = None
    doc_number_check: Optional[str] = None
    overall_check: Optional[str] = None
    
    is_valid_td1: bool = False
    confidence: float = 0.0


@dataclass
class AddressResult:
    """Risultato estrazione indirizzo."""
    raw_text: str = ""
    via: Optional[str] = None
    civico: Optional[str] = None
    cap: Optional[str] = None
    comune: Optional[str] = None
    provincia: Optional[str] = None


@dataclass
class ExtractionResult:
    """Risultato completo estrazione."""
    source_file: str
    page_index: int
    status: str
    processing_time_ms: int
    mrz: Optional[MrzResult] = None
    address: Optional[AddressResult] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("mrz") is None:
            del d["mrz"]
        if d.get("address") is None:
            del d["address"]
        if d.get("error") is None:
            del d["error"]
        return d


class TimeoutException(Exception):
    """Eccezione per timeout hard."""
    pass


def run_with_timeout(func: Callable[[], T], timeout_sec: float, default: T) -> T:
    """Esegue funzione con timeout usando ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            return default
        except Exception:
            return default


def load_image_fast(path: Path, dpi: int = 100, timeout_sec: float = 1.5) -> List[np.ndarray]:
    """Carica immagini da file con timeout."""
    def _load():
        ext = path.suffix.lower()
        if ext == ".pdf":
            return _render_pdf_fast(path, dpi)
        img = cv2.imread(str(path))
        if img is None:
            return []
        return [img]
    
    return run_with_timeout(_load, timeout_sec, [])


def _render_pdf_fast(path: Path, dpi: int) -> List[np.ndarray]:
    """Render PDF veloce con DPI ridotto."""
    images: List[np.ndarray] = []
    try:
        doc = fitz.open(path)
    except Exception:
        return images
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    try:
        for page_num, page in enumerate(doc):
            if page_num >= 5:
                break
            try:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            except Exception:
                continue
    finally:
        doc.close()
    
    return images


def quick_detect_back_tessera(image: np.ndarray, timeout_sec: float = 1.0) -> tuple[bool, float, Optional[tuple[float, float]]]:
    """
    Detection veloce del retro tessera.
    Cerca pattern MRZ in multiple regioni per gestire scan verticali/orizzontali.
    Returns: (is_back, confidence, mrz_region_bounds)
    """
    def _detect():
        h, w = image.shape[:2]
        
        is_portrait = h > w * 1.2
        
        if is_portrait:
            regions = [
                (0.15, 0.50),
                (0.20, 0.55),
                (0.25, 0.60),
                (0.30, 0.60),
                (0.30, 0.65),
                (0.35, 0.70),
                (0.40, 0.75),
                (0.45, 0.80),
                (0.50, 0.85),
                (0.55, 0.90),
            ]
        else:
            regions = [
                (0.55, 1.0),
                (0.50, 0.95),
                (0.45, 0.90),
                (0.60, 1.0),
            ]
        
        best_result = (False, 0.0, None)
        
        for top_pct, bottom_pct in regions:
            mrz_region = image[int(h * top_pct):int(h * bottom_pct), :]
            
            gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            text = pytesseract.image_to_string(
                bw,
                lang="eng",
                config="--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
            )
            
            lines = _extract_mrz_candidate_lines(text)
            
            if len(lines) >= 3:
                l1 = lines[0]
                if (l1.startswith("I<") or l1.startswith("ID") or 
                    l1.startswith("C<") or l1.startswith("A<") or
                    l1.startswith("CS") or l1.startswith("IC")):
                    return (True, 0.9, (top_pct, bottom_pct))
                if "ITA" in l1[:10] or l1[:5].count("<") >= 1:
                    return (True, 0.85, (top_pct, bottom_pct))
            
            if len(lines) >= 2:
                total_chevrons = sum(l.count("<") for l in lines)
                if total_chevrons >= 8:
                    return (True, 0.7, (top_pct, bottom_pct))
            
            if len(lines) >= 1:
                l1 = lines[0]
                if "<<" in l1 and len(l1) >= 25:
                    if best_result[1] < 0.5:
                        best_result = (True, 0.5, (top_pct, bottom_pct))
        
        return best_result
    
    return run_with_timeout(_detect, timeout_sec, (False, 0.0, None))


def _extract_mrz_candidate_lines(text: str) -> List[str]:
    """Estrae linee candidate MRZ dal testo OCR, mantenendo ordine verticale."""
    MRZ_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    candidates = []
    
    for line_idx, line in enumerate(text.splitlines()):
        cleaned = _clean_mrz_line(line)
        if len(cleaned) < 24 or len(cleaned) > 36:
            continue
        
        mrz_ratio = sum(ch in MRZ_CHARS for ch in cleaned) / max(1, len(cleaned))
        if mrz_ratio < 0.85:
            continue
        
        chevron_count = cleaned.count("<")
        if chevron_count < 2:
            continue
        
        score = chevron_count + (30 - abs(len(cleaned) - 30))
        candidates.append((line_idx, cleaned, score))
    
    best_3 = sorted(candidates, key=lambda x: x[2], reverse=True)[:3]
    best_3_ordered = sorted(best_3, key=lambda x: x[0])
    
    return [c[1] for c in best_3_ordered]


def _clean_mrz_line(line: str) -> str:
    """Pulisce linea MRZ da errori OCR comuni."""
    text = line.strip().upper().replace(" ", "")
    text = text.replace("«", "<")
    text = text.replace("く", "<")
    text = text.replace("〈", "<")
    
    if text.count("K") >= 2 and text.count("<") <= 3:
        text = text.replace("K", "<")
    
    if text.count("E") >= 2 and "<<" not in text:
        text = re.sub(r"E{2,}", lambda m: "<" * len(m.group()), text)
    
    if "R<" in text or "<R<" in text:
        text = text.replace("R<", "<<").replace("<R<", "<<<")
    
    text = text.replace("J<", "I<")
    text = text.replace("JC", "IC")
    
    if text.count("S") >= 3 and "<" in text:
        text = text.replace("S<", "<<")
    
    text = re.sub(r"[^A-Z0-9<]", "", text)
    return text


def extract_mrz_full(image: np.ndarray, timeout_sec: float = 1.5, 
                     region_bounds: Optional[tuple[float, float]] = None) -> MrzResult:
    """
    Estrazione MRZ completa con parsing TD1.
    region_bounds: (top_pct, bottom_pct) se noto dalla detection
    """
    def _try_region(top_pct: float, bottom_pct: float) -> tuple[List[str], int]:
        h, w = image.shape[:2]
        mrz_region = image[int(h * top_pct):int(h * bottom_pct), :]
        rh, rw = mrz_region.shape[:2]
        
        if rw < 800:
            scale = 800 / rw
            mrz_region = cv2.resize(mrz_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
        
        variants = []
        
        _, bw1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(bw1)
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(bw2)
        
        bw3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        variants.append(bw3)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, bw4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(bw4)
        
        best_lines: List[str] = []
        best_score = 0
        
        for variant in variants:
            for psm in [6, 4]:
                text = pytesseract.image_to_string(
                    variant,
                    lang="eng",
                    config=f"--oem 1 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
                )
                lines = _extract_mrz_candidate_lines(text)
                score = len(lines) * 10 + sum(l.count("<") for l in lines)
                if score > best_score:
                    best_score = score
                    best_lines = lines
                
                if len(best_lines) >= 3 and best_score >= 50:
                    return best_lines, best_score
        
        return best_lines, best_score
    
    def _extract():
        h, w = image.shape[:2]
        is_portrait = h > w * 1.2
        
        if region_bounds:
            rb = region_bounds
            regions_to_try = [
                rb,
                (rb[0] + 0.05, rb[1] + 0.05),
                (rb[0] + 0.10, rb[1] + 0.05),
                (rb[0], rb[1] + 0.10),
                (rb[0] + 0.05, rb[1]),
            ]
            if is_portrait:
                regions_to_try.extend([
                    (0.25, 0.60),
                    (0.30, 0.60),
                    (0.30, 0.65),
                    (0.35, 0.70),
                ])
            else:
                regions_to_try.extend([
                    (0.55, 1.0),
                    (0.50, 0.95),
                ])
        else:
            if is_portrait:
                regions_to_try = [
                    (0.15, 0.50), (0.20, 0.55), (0.25, 0.60), (0.30, 0.60),
                    (0.30, 0.65), (0.35, 0.70), (0.40, 0.75), (0.45, 0.80),
                    (0.50, 0.85),
                ]
            else:
                regions_to_try = [(0.55, 1.0), (0.50, 0.95), (0.45, 0.90)]
        
        best_result = MrzResult(raw_lines=[], confidence=0.0)
        
        for top_pct, bottom_pct in regions_to_try:
            lines, score = _try_region(top_pct, bottom_pct)
            
            if len(lines) >= 3:
                mrz = _parse_td1_mrz(lines[:3])
                if mrz.confidence > best_result.confidence:
                    best_result = mrz
                
                if mrz.is_valid_td1:
                    return mrz
        
        return best_result
    
    return run_with_timeout(_extract, timeout_sec, MrzResult(raw_lines=[], confidence=0.0))


def _parse_td1_mrz(lines: List[str]) -> MrzResult:
    """
    Parsing MRZ formato TD1 (carta identità italiana).
    
    TD1 Layout (3 righe x 30 caratteri):
    
    Linea 1: [TT][CCC][NNNNNNNNN][C][OOOOOOOOOOOOOOOO]
      - TT: Tipo documento (ID, I<, C<)
      - CCC: Paese emittente (ITA)
      - NNNNNNNNN: Numero documento (9 char)
      - C: Check digit numero documento
      - O...: Campo opzionale (può contenere CF in Italia)
    
    Linea 2: [DDDDDD][C][S][EEEEEE][C][NNN][OOOOOOOOOOO][C]
      - DDDDDD: Data nascita (YYMMDD)
      - C: Check digit data nascita
      - S: Sesso (M/F)
      - EEEEEE: Data scadenza (YYMMDD)
      - C: Check digit scadenza
      - NNN: Nazionalità
      - O...: Campo opzionale
      - C: Check digit complessivo
    
    Linea 3: [COGNOME]<<[NOME]<<<<<...
    """
    MRZ_LEN = 30
    
    l1 = lines[0][:MRZ_LEN].ljust(MRZ_LEN, "<")
    l2 = lines[1][:MRZ_LEN].ljust(MRZ_LEN, "<")
    l3 = lines[2][:MRZ_LEN].ljust(MRZ_LEN, "<")
    
    document_type = l1[0:2].replace("<", "").strip() or None
    issuing_country = l1[2:5].replace("<", "").strip() or None
    document_number = l1[5:14].replace("<", "").strip() or None
    doc_number_check = l1[14:15] if l1[14:15].isdigit() else None
    
    optional_l1 = l1[15:30].replace("<", "").strip()
    
    birth_date_raw = l2[0:6]
    birth_date_check = l2[6:7] if l2[6:7].isdigit() else None
    birth_date = _parse_mrz_date(birth_date_raw, is_expiry=False)
    
    sex = l2[7:8].replace("<", "") or None
    if sex not in ("M", "F"):
        sex = None
    
    expiry_date_raw = l2[8:14]
    expiry_date_check = l2[14:15] if l2[14:15].isdigit() else None
    expiry_date = _parse_mrz_date(expiry_date_raw, is_expiry=True)
    
    nationality = l2[15:18].replace("<", "").strip() or None
    
    overall_check = l2[29:30] if l2[29:30].isdigit() else None
    
    name_parts = l3.split("<<", 1)
    surname = name_parts[0].replace("<", " ").strip() or None
    given_name = name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else None
    
    codice_fiscale = _extract_codice_fiscale_from_mrz(l1, l2)
    
    confidence = _calculate_mrz_confidence(l1, l2, l3)
    is_valid = confidence >= 0.7
    
    return MrzResult(
        raw_lines=[l1, l2, l3],
        document_type=document_type,
        issuing_country=issuing_country,
        document_number=document_number,
        surname=surname,
        given_name=given_name,
        birth_date=birth_date,
        sex=sex,
        expiry_date=expiry_date,
        nationality=nationality,
        codice_fiscale=codice_fiscale,
        birth_date_check=birth_date_check,
        expiry_date_check=expiry_date_check,
        doc_number_check=doc_number_check,
        overall_check=overall_check,
        is_valid_td1=is_valid,
        confidence=confidence
    )


def _parse_mrz_date(raw: str, is_expiry: bool) -> Optional[str]:
    """Parsing data MRZ formato YYMMDD."""
    if not re.fullmatch(r"\d{6}", raw):
        return None
    
    yy, mm, dd = int(raw[0:2]), int(raw[2:4]), int(raw[4:6])
    
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    
    from datetime import date
    current_year = date.today().year
    century = (current_year // 100) * 100
    year = century + yy
    
    if not is_expiry and year > current_year:
        year -= 100
    if is_expiry and year < current_year - 20:
        year += 100
    
    return f"{year:04d}-{mm:02d}-{dd:02d}"


def _extract_codice_fiscale_from_mrz(l1: str, l2: str) -> Optional[str]:
    """
    Estrae codice fiscale dalla MRZ italiana.
    Il CF è in posizione 15-30 della linea 1 (campo opzionale).
    
    Formato CF: AAABBB00A00A000A (16 caratteri)
    - 3 lettere cognome + 3 lettere nome + 2 cifre anno
    - 1 lettera mese + 2 cifre giorno + 1 lettera comune
    - 3 cifre/lettere codice comune + 1 lettera controllo
    """
    cf_region = l1[15:30].replace("<", "").strip()
    
    if len(cf_region) == 16:
        if _validate_codice_fiscale_format(cf_region):
            return cf_region
    
    if len(cf_region) >= 16:
        candidate = cf_region[:16]
        if _validate_codice_fiscale_format(candidate):
            return candidate
    
    return None


def _validate_codice_fiscale_format(cf: str) -> bool:
    """Valida formato codice fiscale italiano."""
    if len(cf) != 16:
        return False
    
    pattern = r"^[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]$"
    return bool(re.match(pattern, cf))


def _calculate_mrz_confidence(l1: str, l2: str, l3: str) -> float:
    """Calcola confidence MRZ basata su euristiche."""
    score = 0.0
    
    if l1.startswith("I<") or l1.startswith("ID") or l1.startswith("C<"):
        score += 0.25
    elif l1.startswith("IC") or l1.startswith("CH"):
        score += 0.15
    
    if "ITA" in l1[2:6] or "ITA" in l2[15:19]:
        score += 0.25
    elif "TA" in l1[3:6] or "TA" in l2[16:19]:
        score += 0.15
    
    if "<<" in l3 and l3.count("<") >= 4:
        score += 0.2
    elif l3.count("<") >= 3:
        score += 0.1
    
    date1_ok = bool(re.search(r"\d{6}", l2[0:7]))
    date2_ok = bool(re.search(r"\d{6}", l2[7:15]))
    if date1_ok and date2_ok:
        score += 0.2
    elif date1_ok or date2_ok:
        score += 0.1
    
    if l2[7:8] in ("M", "F", "<"):
        score += 0.1
    
    return min(score, 1.0)


def extract_address(image: np.ndarray, timeout_sec: float = 1.0) -> AddressResult:
    """
    Estrazione indirizzo dalla zona superiore dell'immagine.
    """
    def _extract():
        h, w = image.shape[:2]
        addr_region = image[int(h * 0.15):int(h * 0.55), :]
        
        gray = cv2.cvtColor(addr_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 35, 35)
        
        upscaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        
        text = pytesseract.image_to_string(upscaled, lang="ita", config="--oem 1 --psm 6")
        
        return _parse_address_text(text)
    
    return run_with_timeout(_extract, timeout_sec, AddressResult())


def _parse_address_text(text: str) -> AddressResult:
    """Parsing testo indirizzo."""
    result = AddressResult(raw_text=text)
    
    via_match = re.search(r"(?:VIA|V\.|VIALE|PIAZZA|P\.ZZA|CORSO|C\.SO|LARGO)\s+[A-Z\s]+(?:\s*,?\s*\d+)?", text.upper())
    if via_match:
        parts = via_match.group().split(",")
        result.via = parts[0].strip()
        if len(parts) > 1:
            civico = re.search(r"\d+", parts[1])
            if civico:
                result.civico = civico.group()
    
    civico_match = re.search(r"\b(N\.?\s*\d+[/A-Z]*|\d+[/A-Z]*)\s*$", text.upper())
    if civico_match and not result.civico:
        result.civico = civico_match.group(1).strip()
    
    cap_match = re.search(r"\b(\d{5})\b", text)
    if cap_match:
        result.cap = cap_match.group(1)
    
    prov_match = re.search(r"\(([A-Z]{2})\)", text.upper())
    if prov_match:
        result.provincia = prov_match.group(1)
    
    return result


def process_file_fast(file_path: Path, config: FastConfig) -> List[ExtractionResult]:
    """
    Processa file con pipeline fail-fast.
    """
    start_time = time.time()
    results: List[ExtractionResult] = []
    
    images = load_image_fast(file_path, config.quick_detect_dpi, config.timeouts.page_render_sec)
    
    if not images:
        return [ExtractionResult(
            source_file=str(file_path),
            page_index=0,
            status="unreadable",
            processing_time_ms=int((time.time() - start_time) * 1000),
            error="Cannot load file"
        )]
    
    for page_idx, image in enumerate(images):
        page_start = time.time()
        
        elapsed = time.time() - start_time
        if elapsed > config.timeouts.file_total_sec:
            results.append(ExtractionResult(
                source_file=str(file_path),
                page_index=page_idx,
                status="timeout",
                processing_time_ms=int((time.time() - page_start) * 1000),
                error=f"File timeout exceeded ({config.timeouts.file_total_sec}s)"
            ))
            break
        
        is_back, detect_confidence, mrz_bounds = quick_detect_back_tessera(
            image, config.timeouts.ocr_call_sec
        )
        
        if not is_back:
            if config.debug_mode:
                results.append(ExtractionResult(
                    source_file=str(file_path),
                    page_index=page_idx,
                    status="skipped_not_back",
                    processing_time_ms=int((time.time() - page_start) * 1000)
                ))
            continue
        
        if detect_confidence < 0.5:
            results.append(ExtractionResult(
                source_file=str(file_path),
                page_index=page_idx,
                status="review",
                processing_time_ms=int((time.time() - page_start) * 1000),
                error=f"Low detection confidence: {detect_confidence:.2f}"
            ))
            continue
        
        full_images = load_image_fast(file_path, config.full_extract_dpi, config.timeouts.page_render_sec)
        if page_idx < len(full_images):
            full_image = full_images[page_idx]
        else:
            full_image = image
        
        mrz = extract_mrz_full(full_image, config.timeouts.mrz_extraction_sec, mrz_bounds)
        
        if not mrz.is_valid_td1:
            results.append(ExtractionResult(
                source_file=str(file_path),
                page_index=page_idx,
                status="review",
                processing_time_ms=int((time.time() - page_start) * 1000),
                mrz=mrz,
                error=f"Invalid MRZ (confidence: {mrz.confidence:.2f})"
            ))
            continue
        
        address = extract_address(full_image, config.timeouts.address_extraction_sec)
        
        results.append(ExtractionResult(
            source_file=str(file_path),
            page_index=page_idx,
            status="confirmed",
            processing_time_ms=int((time.time() - page_start) * 1000),
            mrz=mrz,
            address=address
        ))
    
    return results


def run_fast_pipeline(config: FastConfig) -> dict:
    """
    Esegue pipeline veloce su tutti i file nella directory.
    """
    from .discovery import iter_candidate_files
    
    confirmed: List[dict] = []
    review: List[dict] = []
    stats = {"total_files": 0, "total_pages": 0, "confirmed": 0, "review": 0, "skipped": 0, "errors": 0}
    
    for file_path in iter_candidate_files(config.root_input):
        stats["total_files"] += 1
        
        results = process_file_fast(file_path, config)
        
        for result in results:
            stats["total_pages"] += 1
            result_dict = result.to_dict()
            
            if result.status == "confirmed":
                stats["confirmed"] += 1
                confirmed.append(result_dict)
            elif result.status == "review":
                stats["review"] += 1
                review.append(result_dict)
            elif result.status in ("skipped_not_back", "timeout"):
                stats["skipped"] += 1
            else:
                stats["errors"] += 1
    
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.review_json.parent.mkdir(parents=True, exist_ok=True)
    
    config.output_json.write_text(json.dumps(confirmed, ensure_ascii=False, indent=2))
    config.review_json.write_text(json.dumps(review, ensure_ascii=False, indent=2))
    
    return {
        "stats": stats,
        "confirmed_output": str(config.output_json),
        "review_output": str(config.review_json)
    }
