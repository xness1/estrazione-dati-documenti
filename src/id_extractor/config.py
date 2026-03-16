from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    root_input: Path
    output_json: Path
    tesseract_lang: str = "ita+eng"
    debug_save_images: bool = False
    debug_dir: Path = Path("output/debug")

    # Regioni verticali, espresse in percentuale altezza immagine.
    # y in [0,1]: 0 alto, 1 basso
    mrz_region_top: float = 0.62
    mrz_region_bottom: float = 1.00
    address_region_top: float = 0.42
    address_region_bottom: float = 0.78

