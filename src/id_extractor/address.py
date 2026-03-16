import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class AddressData:
    raw_text: str
    address_line: Optional[str]
    city: Optional[str]
    province: Optional[str]
    cap: Optional[str]


STREET_HINTS = ("VIA", "VIALE", "PIAZZA", "CORSO", "LARGO", "VICOLO", "RESIDENZA", "INDIRIZZO")


def extract_address(text: str) -> AddressData:
    lines = [normalize_line(x) for x in text.splitlines()]
    lines = [x for x in lines if x]
    merged = " ".join(lines)

    address_line = None
    for line in lines:
        if any(hint in line for hint in STREET_HINTS):
            address_line = line
            break

    cap_match = re.search(r"\b(\d{5})\b", merged)
    cap = cap_match.group(1) if cap_match else None

    city = None
    province = None
    city_match = re.search(r"\b([A-Z][A-Z' ]{2,})\s*\(([A-Z]{2})\)\b", merged)
    if city_match:
        city = city_match.group(1).strip()
        province = city_match.group(2).strip()

    return AddressData(
        raw_text="\n".join(lines),
        address_line=address_line,
        city=city,
        province=province,
        cap=cap,
    )


def normalize_line(line: str) -> str:
    text = line.upper().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Z0-9'()/,\-. ]", "", text)
    return text

