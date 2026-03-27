"""
Pipeline MRZ con EasyOCR - molto più accurato di Tesseract.
Target: max 5-10 secondi per documento su CPU.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import fitz
import numpy as np

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

def get_easyocr_reader():
    """Lazy load EasyOCR reader (slow init)."""
    global _easyocr_reader
    if _easyocr_reader is None and HAS_EASYOCR:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


@dataclass
class MrzExtraction:
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
    
    raw_lines: List[str] = None
    icao_valid: bool = False
    processing_time_ms: int = 0
    error: Optional[str] = None


def load_pages(path: Path, dpi: int = 200) -> List[np.ndarray]:
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


def extract_mrz_easyocr(image: np.ndarray) -> tuple[List[str], bool]:
    """
    Estrae MRZ usando EasyOCR.
    Returns: (mrz_lines, is_back_detected)
    """
    reader = get_easyocr_reader()
    if not reader:
        return [], False
    
    h, w = image.shape[:2]
    is_portrait = h > w * 1.2
    
    if is_portrait:
        regions = [(0.20, 0.55), (0.25, 0.50), (0.30, 0.60), (0.35, 0.65)]
    else:
        regions = [(0.55, 1.0), (0.50, 0.90), (0.60, 1.0)]
    
    for top, bottom in regions:
        mrz_region = image[int(h * top):int(h * bottom), :]
        
        rh, rw = mrz_region.shape[:2]
        if rw < 800:
            scale = 800 / rw
            mrz_region = cv2.resize(mrz_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        results = reader.readtext(mrz_region, detail=0, paragraph=False)
        
        mrz_lines = []
        for line in results:
            cleaned = _clean_mrz_line_easyocr(line)
            if len(cleaned) >= 28 and cleaned.count("<") >= 2:
                mrz_lines.append(cleaned[:30].ljust(30, "<"))
        
        if len(mrz_lines) >= 3:
            return mrz_lines[:3], True
    
    return [], False


def _clean_mrz_line_easyocr(line: str) -> str:
    """Pulisce linea MRZ da EasyOCR."""
    text = line.upper().replace(" ", "")
    text = text.replace("«", "<").replace("〈", "<")
    
    if text.count("K") >= 3 and text.count("<") <= 2:
        text = text.replace("K", "<")
    
    text = re.sub(r"[^A-Z0-9<]", "", text)
    return text


def parse_mrz_icao(lines: List[str]) -> Optional[MrzExtraction]:
    """Parsing MRZ con libreria ICAO standard."""
    if not HAS_MRZ_LIB or len(lines) < 3:
        return None
    
    l1 = _fix_line1(lines[0])
    l2 = _fix_line2(lines[1])
    l3 = _fix_line3(lines[2])
    
    mrz_string = f"{l1}\n{l2}\n{l3}"
    
    try:
        checker = TD1CodeChecker(mrz_string, check_expiry=False)
        fields = checker.fields()
        
        birth_date = _format_date(fields.birth_date, is_expiry=False)
        expiry_date = _format_date(fields.expiry_date, is_expiry=True)
        
        return MrzExtraction(
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


def _fix_line1(line: str) -> str:
    """Fix linea 1: documento, paese, numero."""
    return line


def _fix_line2(line: str) -> str:
    """Fix linea 2: date (O→0), sesso, nazionalità."""
    chars = list(line)
    
    for i in range(min(7, len(chars))):
        if chars[i] == 'O':
            chars[i] = '0'
    
    for i in range(8, min(15, len(chars))):
        if chars[i] == 'O':
            chars[i] = '0'
    
    return "".join(chars)


def _fix_line3(line: str) -> str:
    """Fix linea 3: nomi (0→O)."""
    text = line.replace("0", "O")
    return text


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


def process_file_easyocr(path: Path, dpi: int = 250) -> List[MrzExtraction]:
    """Processa file con EasyOCR."""
    results = []
    
    pages = load_pages(path, dpi)
    if not pages:
        return [MrzExtraction(
            source_file=str(path),
            page_index=0,
            status="unreadable",
            error="Cannot load file"
        )]
    
    for page_idx, image in enumerate(pages):
        start = time.time()
        
        mrz_lines, is_back = extract_mrz_easyocr(image)
        
        if not is_back:
            continue
        
        mrz = parse_mrz_icao(mrz_lines)
        
        elapsed = int((time.time() - start) * 1000)
        
        if mrz and mrz.icao_valid:
            mrz.source_file = str(path)
            mrz.page_index = page_idx
            mrz.processing_time_ms = elapsed
            results.append(mrz)
        else:
            results.append(MrzExtraction(
                source_file=str(path),
                page_index=page_idx,
                status="review",
                raw_lines=mrz_lines,
                processing_time_ms=elapsed,
                error="ICAO validation failed"
            ))
    
    return results


def test_all_documents(base_path: Path):
    """Testa tutti i documenti."""
    all_files = []
    
    for f in base_path.rglob("*"):
        if not f.is_file():
            continue
        if "Zone.Identifier" in f.name:
            continue
        if f.suffix.lower() not in (".pdf", ".jpg", ".jpeg", ".png"):
            continue
        
        name_lower = f.name.lower()
        if any(k in name_lower for k in ["c.id", "carta", "identit", "c.i.", "doc.", "documenti", "grasso", "benedetti", "briani", "martinelli", "bisi", "aquironi", "prati"]):
            if "cf" not in name_lower and "visura" not in name_lower and "patente" not in name_lower:
                all_files.append(f)
    
    return sorted(set(all_files))
