from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import cv2

from .back_detection import detect_back_side
from .ocr import preprocess_for_text, run_tesseract


@dataclass
class PageEvidence:
    page_index: int
    mrz_back_detected: bool
    back_signature_score: int
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
    max_back_signature = 0

    for idx, img in enumerate(pages):
        evidence = _analyze_page(idx, img)
        page_evidence.append(evidence)
        has_back = has_back or evidence.mrz_back_detected
        max_front_hits = max(max_front_hits, evidence.front_keyword_hits)
        max_back_signature = max(max_back_signature, evidence.back_signature_score)

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
    if max_back_signature >= 5:
        return DocumentEvidence(
            is_id_card=True,
            confidence=0.70,
            reason="MRZ-like signature found without full parse",
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
    back = detect_back_side(image)
    if back.probable_back_side:
        return PageEvidence(
            page_index=page_index,
            mrz_back_detected=True,
            back_signature_score=back.signature_score,
            front_keyword_hits=0,
            text_excerpt=back.signature_excerpt[:180],
        )

    # OCR full-page (ridotto) per segnali fronte carta.
    small = cv2.resize(image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    text_img = preprocess_for_text(small)
    page_text = run_tesseract(text_img, lang="ita+eng", psm=6).upper()
    front_hits = sum(1 for k in ID_FRONT_KEYWORDS if k in page_text)

    excerpt = " ".join(page_text.split())[:180]
    return PageEvidence(
        page_index=page_index,
        mrz_back_detected=False,
        back_signature_score=back.signature_score,
        front_keyword_hits=front_hits,
        text_excerpt=excerpt,
    )

