from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .mrz import MrzData, extract_mrz_lines, parse_td1
from .ocr import preprocess_for_mrz, preprocess_for_text, run_tesseract


@dataclass
class BackDetection:
    probable_back_side: bool
    mrz_data: MrzData
    mrz_bbox: Optional[Tuple[int, int, int, int]]
    signature_score: int
    signature_excerpt: str


def detect_back_side(image) -> BackDetection:
    h, w = image.shape[:2]
    quick_score, quick_excerpt = _quick_signature_scan(image)
    if quick_score < 2:
        return BackDetection(
            probable_back_side=False,
            mrz_data=parse_td1([]),
            mrz_bbox=None,
            signature_score=quick_score,
            signature_excerpt=quick_excerpt,
        )

    windows = _build_windows(h, w)

    best_data: Optional[MrzData] = None
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_data_score = -1
    best_signature_score = -1
    best_signature_excerpt = ""

    for x0, y0, x1, y1 in windows:
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        candidates = []
        text_variant = preprocess_for_text(crop)
        candidates.append((text_variant, 6))
        candidates.append((text_variant, 11))

        mrz_variant = preprocess_for_mrz(crop)
        candidates.append((mrz_variant, 6))

        for img, psm in candidates:
            ocr_text = run_tesseract(
                img,
                lang="eng",
                psm=psm,
                whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
            )
            mrz_lines = extract_mrz_lines(ocr_text)
            mrz_data = parse_td1(mrz_lines)
            recovered_data = _recover_td1_from_text(ocr_text)
            sig_score, sig_excerpt = _signature_score(ocr_text)

            if sig_score > best_signature_score:
                best_signature_score = sig_score
                best_signature_excerpt = sig_excerpt

            if mrz_data.probable_back_side:
                # Early return: quando la MRZ viene parse-ata bene, la detection è robusta.
                return BackDetection(
                    probable_back_side=True,
                    mrz_data=mrz_data,
                    mrz_bbox=(x0, y0, x1, y1),
                    signature_score=max(sig_score, 6),
                    signature_excerpt=sig_excerpt,
                )
            if recovered_data.probable_back_side:
                return BackDetection(
                    probable_back_side=True,
                    mrz_data=recovered_data,
                    mrz_bbox=(x0, y0, x1, y1),
                    signature_score=max(sig_score, 5),
                    signature_excerpt=sig_excerpt,
                )

            if best_data is None or sig_score > best_data_score:
                best_data = mrz_data
                best_bbox = (x0, y0, x1, y1)
                best_data_score = sig_score

    fallback_data = best_data or parse_td1([])
    return BackDetection(
        probable_back_side=False,
        mrz_data=fallback_data,
        mrz_bbox=best_bbox,
        signature_score=max(0, best_signature_score),
        signature_excerpt=best_signature_excerpt,
    )


def estimate_address_region(image_height: int, mrz_bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    if mrz_bbox is None:
        top = int(image_height * 0.40)
        bottom = int(image_height * 0.78)
        return top, bottom

    _, y0, _, y1 = mrz_bbox
    mrz_h = max(1, y1 - y0)
    top = max(0, int(y0 - (mrz_h * 1.20)))
    bottom = max(top + 20, min(image_height, int(y0 + (mrz_h * 0.10))))
    return top, bottom


def _build_windows(h: int, w: int) -> List[Tuple[int, int, int, int]]:
    windows = []
    ratios = (
        (0.15, 0.68),
        (0.32, 0.86),
        (0.00, 0.48),
        (0.00, 1.00),
        (0.50, 1.00),
    )
    for y0_r, y1_r in ratios:
        y0 = int(h * y0_r)
        y1 = int(h * y1_r)
        windows.append((0, y0, w, y1))
    return windows


def _signature_score(text: str) -> Tuple[int, str]:
    normalized = _normalize(text)
    score = 0

    if re.search(r"[CIA][<K]{1,2}ITA", normalized) or re.search(r"ITA[<K]{1,2}", normalized):
        score += 2
    if re.search(r"\d{6}[<K]?[MFX][<K]?\d{6}", normalized):
        score += 2
    if "<<" in normalized or "KK" in normalized:
        score += 1
    if re.search(r"[A-Z0-9<]{24,}", normalized):
        score += 1
    if re.search(r"<<[A-Z]{3,}", normalized):
        score += 1

    excerpt = " ".join(normalized.split())[:220]
    return score, excerpt


def _normalize(text: str) -> str:
    upper = text.upper().replace(" ", "").replace("«", "<")
    upper = re.sub(r"[^A-Z0-9<]", "", upper)
    # OCR confusion comuni in MRZ.
    upper = upper.replace("KK", "<<")
    upper = upper.replace("K<", "<<")
    upper = upper.replace("<K", "<<")
    return upper


def _quick_signature_scan(image) -> Tuple[int, str]:
    h, w = image.shape[:2]
    quick_windows = (
        (0, int(h * 0.15), w, int(h * 0.68)),
        (0, 0, w, h),
    )
    best_score = 0
    best_excerpt = ""
    for x0, y0, x1, y1 in quick_windows:
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        txt = run_tesseract(
            preprocess_for_text(crop),
            lang="eng",
            psm=11,
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
        )
        score, excerpt = _signature_score(txt)
        if score > best_score:
            best_score = score
            best_excerpt = excerpt
    return best_score, best_excerpt


def _recover_td1_from_text(text: str) -> MrzData:
    normalized = _normalize(text)
    relaxed = normalized
    if relaxed.count("K") >= 3 and relaxed.count("<") <= 6:
        relaxed = relaxed.replace("K", "<")

    l1_match = re.search(r"([CIA][<][I1]TA[A-Z0-9<]{8,25}?)(?=\d{7}[<]?[MFX])", relaxed)
    if l1_match is None:
        l1_match = re.search(r"([CIA][<][I1]TA[A-Z0-9<]{12,28})", relaxed)

    l2_match = re.search(r"(\d{6}[0-9<][<]?[MFX]\d{6}[0-9<][A-Z0-9<]{5,22}?)(?=[A-Z]{2,}<<)", relaxed)
    if l2_match is None:
        l2_match = re.search(r"(\d{6}[0-9<][<]?[MFX]\d{6}[0-9<][A-Z0-9<]{8,24})", relaxed)

    l3_line = None
    l3_candidates = list(re.finditer(r"([A-Z]{3,})<<([A-Z<]{3,})", relaxed))
    for m in reversed(l3_candidates):
        surname = m.group(1)
        if surname not in {"ITA", "UTO"}:
            l3_line = f"{surname}<<{m.group(2)}"
            break

    lines = []
    if l1_match:
        l1 = l1_match.group(1).replace("<1TA", "<ITA")
        lines.append(l1[:30].ljust(30, "<"))
    if l2_match:
        l2 = l2_match.group(1).replace("1TA", "ITA")
        lines.append(l2[:30].ljust(30, "<"))
    if l3_line:
        lines.append(l3_line[:30].ljust(30, "<"))

    return parse_td1(lines)
