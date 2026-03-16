from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import cv2

from .address import AddressData, extract_address
from .back_detection import detect_back_side, estimate_address_region
from .config import PipelineConfig
from .discovery import iter_candidate_files
from .mrz import MrzData
from .ocr import preprocess_for_text, run_tesseract
from .render import load_document_images


@dataclass
class ExtractionResult:
    source_file: str
    page_index: int
    is_probable_id_back: bool
    mrz: MrzData
    address: AddressData


def run_pipeline(config: PipelineConfig) -> List[ExtractionResult]:
    results: List[ExtractionResult] = []
    for file_path in iter_candidate_files(config.root_input):
        pages = load_document_images(file_path)
        for page_idx, page_image in enumerate(pages):
            result = _process_page(file_path, page_idx, page_image, config)
            if result is not None:
                results.append(result)
    _save_results(results, config.output_json)
    return results


def _process_page(path: Path, page_idx: int, image, config: PipelineConfig) -> ExtractionResult | None:
    h, w = image.shape[:2]
    back = detect_back_side(image)
    mrz_data = back.mrz_data

    # Se MRZ non riconosciuta, salta per ridurre falsi positivi.
    if not back.probable_back_side:
        return None

    addr_top, addr_bottom = estimate_address_region(h, back.mrz_bbox)
    addr_crop = image[addr_top:addr_bottom, 0:w]
    addr_text = _read_address_text(addr_crop, config.tesseract_lang)
    address = extract_address(addr_text)

    if config.debug_save_images:
        _save_debug(config.debug_dir, path, page_idx, mrz_img, preprocess_for_text(addr_crop))

    return ExtractionResult(
        source_file=str(path),
        page_index=page_idx,
        is_probable_id_back=True,
        mrz=mrz_data,
        address=address,
    )


def _read_address_text(addr_crop, lang: str) -> str:
    variants = []

    gray = preprocess_for_text(addr_crop)
    variants.append(gray)

    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(upscaled)

    adaptive = cv2.adaptiveThreshold(
        upscaled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    variants.append(adaptive)

    chunks = []
    for img in variants:
        chunks.append(run_tesseract(img, lang=lang, psm=6))
        chunks.append(run_tesseract(img, lang=lang, psm=11))
    return "\n".join(chunks)


def _save_results(results: List[ExtractionResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [asdict(item) for item in results]
    output_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_debug(debug_root: Path, src_path: Path, page_idx: int, mrz_img, addr_img) -> None:
    debug_root.mkdir(parents=True, exist_ok=True)
    stem = src_path.stem.replace(" ", "_")
    mrz_path = debug_root / f"{stem}_p{page_idx}_mrz.png"
    addr_path = debug_root / f"{stem}_p{page_idx}_addr.png"
    cv2.imwrite(str(mrz_path), mrz_img)
    cv2.imwrite(str(addr_path), addr_img)

