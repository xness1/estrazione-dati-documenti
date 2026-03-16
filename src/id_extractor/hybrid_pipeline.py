"""
Pipeline ibrida veloce: Tesseract detect + EasyOCR extract.
Target: < 5 secondi per documento su CPU.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import fitz
import numpy as np
import pytesseract

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    from mrz.checker.td1 import TD1CodeChecker
    HAS_MRZ_LIB = True
except ImportError:
    HAS_MRZ_LIB = False

fitz.TOOLS.mupdf_display_errors(False)

_easyocr_reader = None


def _get_reader():
    """Lazy load EasyOCR (slow init, reuse)."""
    global _easyocr_reader
    if _easyocr_reader is None and HAS_EASYOCR:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


@dataclass
class MrzResult:
    """Risultato estrazione MRZ."""
    source_file: str
    page_index: int
    status: str
    
    document_number: Optional[str] = None
    surname: Optional[str] = None
    given_name: Optional[str] = None
    birth_date: Optional[str] = None
    sex: Optional[str] = None
    expiry_date: Optional[str] = None
    nationality: Optional[str] = None
    
    raw_lines: Optional[List[str]] = None
    icao_valid: bool = False
    processing_time_ms: int = 0
    error: Optional[str] = None


def _load_pages(path: Path, dpi: int = 150) -> List[np.ndarray]:
    """Carica pagine da PDF o immagine."""
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        images = []
        try:
            doc = fitz.open(path)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            for page_num, page in enumerate(doc):
                if page_num >= 5:
                    break
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            doc.close()
        except Exception:
            pass
        return images
    else:
        img = cv2.imread(str(path))
        return [img] if img is not None else []


def _quick_detect_tesseract(image: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """
    Detection veloce con Tesseract.
    Returns: (is_back, mrz_bounds)
    """
    h, w = image.shape[:2]
    
    if w > 600:
        scale = 600 / w
        small = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        small = image
    
    sh, sw = small.shape[:2]
    is_portrait = sh > sw * 1.2
    
    if is_portrait:
        regions = [(0.20, 0.55), (0.25, 0.50), (0.30, 0.60), (0.35, 0.65)]
    else:
        regions = [(0.55, 1.0), (0.50, 0.90), (0.60, 1.0)]
    
    for top, bottom in regions:
        mrz = small[int(sh * top):int(sh * bottom), :]
        gray = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        text = pytesseract.image_to_string(
            bw, lang="eng",
            config="--oem 1 --psm 6 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
        )
        
        lines = []
        for line in text.split('\n'):
            cleaned = line.strip().upper().replace(" ", "")
            if "<" in cleaned and len(cleaned) > 20:
                lines.append(cleaned)
        
        if len(lines) >= 2:
            return True, (top, bottom)
    
    return False, None


def _extract_mrz_easyocr(image: np.ndarray, bounds: Tuple[float, float]) -> List[str]:
    """Estrazione accurata MRZ con EasyOCR."""
    reader = _get_reader()
    if not reader:
        return []
    
    h, w = image.shape[:2]
    top, bottom = bounds
    
    mrz_region = image[int(h * top):int(h * bottom), :]
    
    rh, rw = mrz_region.shape[:2]
    if rw < 1000:
        scale = 1000 / rw
        mrz_region = cv2.resize(mrz_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    results = reader.readtext(mrz_region, detail=0, paragraph=False)
    
    mrz_lines = []
    for line in results:
        cleaned = line.upper().replace(" ", "")
        cleaned = cleaned.replace("«", "<").replace("K", "<")
        cleaned = re.sub(r"[^A-Z0-9<]", "", cleaned)
        
        if len(cleaned) >= 28 and cleaned.count("<") >= 2:
            mrz_lines.append(cleaned[:30].ljust(30, "<"))
    
    return mrz_lines[:3] if len(mrz_lines) >= 3 else mrz_lines


def _parse_mrz_icao(lines: List[str]) -> Optional[MrzResult]:
    """Parsing MRZ con validazione ICAO."""
    if not HAS_MRZ_LIB or len(lines) < 3:
        return None
    
    l1 = lines[0]
    l2 = lines[1]
    l3 = lines[2]
    
    l2_chars = list(l2)
    for i in range(min(7, len(l2_chars))):
        if l2_chars[i] == 'O':
            l2_chars[i] = '0'
    for i in range(8, min(15, len(l2_chars))):
        if l2_chars[i] == 'O':
            l2_chars[i] = '0'
    l2 = "".join(l2_chars)
    
    l3 = l3.replace("0", "O")
    
    mrz_string = f"{l1}\n{l2}\n{l3}"
    
    try:
        checker = TD1CodeChecker(mrz_string, check_expiry=False)
        fields = checker.fields()
        
        birth_date = _format_date(fields.birth_date, is_expiry=False)
        expiry_date = _format_date(fields.expiry_date, is_expiry=True)
        
        return MrzResult(
            source_file="",
            page_index=0,
            status="confirmed",
            document_number=fields.document_number,
            surname=fields.surname,
            given_name=fields.name,
            birth_date=birth_date,
            sex=fields.sex,
            expiry_date=expiry_date,
            nationality=fields.nationality,
            raw_lines=[l1, l2, l3],
            icao_valid=True
        )
    except Exception:
        return None


def _format_date(raw: str, is_expiry: bool) -> Optional[str]:
    """Formatta data YYMMDD → YYYY-MM-DD."""
    if not raw or len(raw) != 6:
        return None
    try:
        yy = int(raw[:2])
        mm = int(raw[2:4])
        dd = int(raw[4:6])
        
        if is_expiry:
            year = 2000 + yy
        else:
            year = 1900 + yy if yy > 30 else 2000 + yy
        
        return f"{year:04d}-{mm:02d}-{dd:02d}"
    except ValueError:
        return None


def process_document(path: Path, dpi: int = 150) -> List[MrzResult]:
    """
    Processa documento con pipeline ibrida veloce.
    Target: < 5 secondi su CPU.
    """
    results = []
    start = time.time()
    
    pages = _load_pages(path, dpi)
    if not pages:
        return [MrzResult(
            source_file=str(path),
            page_index=0,
            status="unreadable",
            error="Cannot load file",
            processing_time_ms=int((time.time() - start) * 1000)
        )]
    
    for page_idx, image in enumerate(pages):
        page_start = time.time()
        
        is_back, bounds = _quick_detect_tesseract(image)
        
        if not is_back:
            continue
        
        mrz_lines = _extract_mrz_easyocr(image, bounds)
        
        if len(mrz_lines) >= 3:
            mrz = _parse_mrz_icao(mrz_lines)
            
            if mrz and mrz.icao_valid:
                mrz.source_file = str(path)
                mrz.page_index = page_idx
                mrz.processing_time_ms = int((time.time() - page_start) * 1000)
                results.append(mrz)
                break
            else:
                results.append(MrzResult(
                    source_file=str(path),
                    page_index=page_idx,
                    status="review",
                    raw_lines=mrz_lines,
                    processing_time_ms=int((time.time() - page_start) * 1000),
                    error="ICAO validation failed"
                ))
    
    return results if results else [MrzResult(
        source_file=str(path),
        page_index=0,
        status="not_found",
        processing_time_ms=int((time.time() - start) * 1000)
    )]


def extract_id_card_back(path: Path) -> Optional[MrzResult]:
    """
    Estrae dati dal retro carta d'identità elettronica.
    Returns None se non trovato o non valido.
    """
    results = process_document(path)
    
    for r in results:
        if r.status == "confirmed" and r.icao_valid:
            return r
    
    return None
