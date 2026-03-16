from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import cv2

from .mrz import extract_mrz_lines, parse_td1
from .ocr import preprocess_for_mrz, preprocess_for_text, run_tesseract


@dataclass
class PageEvidence:
    page_index: int
    mrz_back_detected: bool
    front_keyword_hits: int
    text_excerpt: str


@dataclass
class DocumentEvidence:
    is_id_card: bool
    confidence: float
    reason: str
    pages: List[PageEvidence]

    def to_dict(self) -> dict:
        return asdict(self)


ID_FRONT_KEYWORDS = (
    "CARTA",
    "IDENTITA",
    "IDENTIT",
    "REPUBBLICA",
    "COGNOME",
    "NOME",
    "SESSO",
    "DATA DI NASCITA",
    "LUOGO DI NASCITA",
    "SCADENZA",
)


def analyze_document_pages(pages: List) -> DocumentEvidence:
    page_evidence: List[PageEvidence] = []
    has_back = False
    max_front_hits = 0

    for idx, img in enumerate(pages):
        evidence = _analyze_page(idx, img)
        page_evidence.append(evidence)
        has_back = has_back or evidence.mrz_back_detected
        max_front_hits = max(max_front_hits, evidence.front_keyword_hits)

    if has_back and max_front_hits >= 2:
        return DocumentEvidence(
            is_id_card=True,
            confidence=0.97,
            reason="MRZ back detected and front keywords found",
            pages=page_evidence,
        )
    if has_back:
        return DocumentEvidence(
            is_id_card=True,
            confidence=0.90,
            reason="MRZ back detected",
            pages=page_evidence,
        )
    if max_front_hits >= 4:
        return DocumentEvidence(
            is_id_card=True,
            confidence=0.65,
            reason="Strong front-side keywords without MRZ",
            pages=page_evidence,
        )
    return DocumentEvidence(
        is_id_card=False,
        confidence=0.05,
        reason="No ID evidence from content",
        pages=page_evidence,
    )


def _analyze_page(page_index: int, image) -> PageEvidence:
    h, w = image.shape[:2]

    mrz_crop = image[int(h * 0.62) : h, 0:w]
    mrz_img = preprocess_for_mrz(mrz_crop)
    mrz_text = run_tesseract(
        mrz_img,
        lang="eng",
        psm=6,
        whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
    )
    mrz_lines = extract_mrz_lines(mrz_text)
    mrz_data = parse_td1(mrz_lines)

    # OCR full-page (ridotto) per segnali fronte carta.
    small = cv2.resize(image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    text_img = preprocess_for_text(small)
    page_text = run_tesseract(text_img, lang="ita+eng", psm=6).upper()
    front_hits = sum(1 for k in ID_FRONT_KEYWORDS if k in page_text)

    excerpt = " ".join(page_text.split())[:180]
    return PageEvidence(
        page_index=page_index,
        mrz_back_detected=mrz_data.probable_back_side,
        front_keyword_hits=front_hits,
        text_excerpt=excerpt,
    )

