from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SKIP_SUFFIXES = (":zone.identifier", ":com.dropbox.attrs")


def iter_candidate_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip(path):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _should_skip(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SKIP_SUFFIXES)

