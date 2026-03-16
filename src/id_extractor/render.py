from pathlib import Path
from typing import List

import cv2
import fitz
import numpy as np

fitz.TOOLS.mupdf_display_errors(False)


def load_document_images(path: Path, pdf_dpi: int = 250) -> List[np.ndarray]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _render_pdf(path, pdf_dpi)
    image = cv2.imread(str(path))
    if image is None:
        return []
    return [image]


def _render_pdf(path: Path, dpi: int) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    try:
        doc = fitz.open(path)
    except Exception:
        return images
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page in doc:
            try:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                # PyMuPDF emette RGB, OpenCV usa BGR.
                images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            except Exception:
                # Pagina corrotta/non renderizzabile: la saltiamo.
                continue
    finally:
        doc.close()
    return images

