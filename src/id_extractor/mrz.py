import re
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

from dateutil.relativedelta import relativedelta


MRZ_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
MRZ_LEN = 30


@dataclass
class MrzData:
    raw_lines: List[str]
    document_type: Optional[str]
    issuing_country: Optional[str]
    document_number: Optional[str]
    surname: Optional[str]
    name: Optional[str]
    nationality: Optional[str]
    birth_date: Optional[str]
    expiry_date: Optional[str]
    sex: Optional[str]
    probable_back_side: bool


def extract_mrz_lines(text: str) -> List[str]:
    candidates: List[tuple[int, str, int]] = []
    for idx, line in enumerate(text.splitlines()):
        cleaned = _clean_line(line)
        if len(cleaned) < 24:
            continue
        if len(cleaned) > 36:
            continue
        mrz_ratio = sum(ch in MRZ_ALLOWED for ch in cleaned) / max(1, len(cleaned))
        if mrz_ratio < 0.85:
            continue
        if cleaned.count("<") < 2:
            continue
        score = (10 - abs(len(cleaned) - MRZ_LEN)) + cleaned.count("<")
        candidates.append((idx, cleaned, score))
    # Seleziona le 3 migliori ma mantieni ordine verticale originale.
    best = sorted(candidates, key=lambda x: x[2], reverse=True)[:3]
    best_in_order = sorted(best, key=lambda x: x[0])
    return [line[:MRZ_LEN].ljust(MRZ_LEN, "<") for _, line, _ in best_in_order if len(line) >= 24]


def parse_td1(lines: List[str]) -> MrzData:
    if len(lines) != 3:
        return MrzData(
            raw_lines=lines,
            document_type=None,
            issuing_country=None,
            document_number=None,
            surname=None,
            name=None,
            nationality=None,
            birth_date=None,
            expiry_date=None,
            sex=None,
            probable_back_side=False,
        )

    l1, l2, l3 = [ln[:MRZ_LEN].ljust(MRZ_LEN, "<") for ln in lines]

    document_type = l1[0:2].replace("<", "") or None
    issuing_country = l1[2:5].replace("<", "") or None
    document_number = l1[5:14].replace("<", "") or None

    birth_date = _parse_yy_mm_dd(l2[0:6])
    sex = l2[7:8].replace("<", "") or None
    expiry_date = _parse_yy_mm_dd(l2[8:14], is_expiry=True)
    nationality = l2[15:18].replace("<", "") or None

    surname, name = _parse_name(l3)
    probable_back = _is_probable_italian_back(l1, l2, l3)

    return MrzData(
        raw_lines=[l1, l2, l3],
        document_type=document_type,
        issuing_country=issuing_country,
        document_number=document_number,
        surname=surname,
        name=name,
        nationality=nationality,
        birth_date=birth_date,
        expiry_date=expiry_date,
        sex=sex,
        probable_back_side=probable_back,
    )


def _clean_line(line: str) -> str:
    text = line.strip().upper().replace(" ", "")
    text = text.replace("«", "<")
    # OCR confonde spesso '<' con 'K' nelle MRZ.
    if text.count("K") >= 3 and text.count("<") <= 2:
        text = text.replace("K", "<")
    text = text.replace("く", "<")
    text = re.sub(r"[^A-Z0-9<]", "", text)
    return text


def _parse_name(raw: str) -> tuple[Optional[str], Optional[str]]:
    normalized = raw.replace("0", "O").replace("K", "<")
    chunks = normalized.split("<<", maxsplit=1)
    surname = chunks[0].replace("<", " ").strip() or None
    name = None
    if len(chunks) > 1:
        name = chunks[1].replace("<", " ").strip() or None
    return surname, name


def _parse_yy_mm_dd(raw: str, is_expiry: bool = False) -> Optional[str]:
    if not re.fullmatch(r"\d{6}", raw):
        return None
    yy, mm, dd = int(raw[0:2]), int(raw[2:4]), int(raw[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    current_year = date.today().year
    century = (current_year // 100) * 100
    year = century + yy
    candidate = date(year, mm, dd)
    # su date di nascita tende ad essere nel passato; su scadenza nel presente/futuro.
    if not is_expiry and candidate > date.today():
        candidate = candidate - relativedelta(years=100)
    if is_expiry and candidate < date.today() - relativedelta(years=20):
        candidate = candidate + relativedelta(years=100)
    return candidate.isoformat()


def _is_probable_italian_back(l1: str, l2: str, l3: str) -> bool:
    if len(l1) < MRZ_LEN or len(l2) < MRZ_LEN or len(l3) < MRZ_LEN:
        return False
    starts_like_td1 = l1.startswith("I<") or l1.startswith("C<") or l1.startswith("A<")
    country_hint = l1[2:5] in {"ITA", "UTO", "XXX"} or l2[15:18] in {"ITA", "UTO"}
    has_name_separator = "<<" in l3
    has_many_fillers = l3.count("<") >= 5
    has_date_chunks = bool(re.fullmatch(r"\d{6}", l2[0:6]) and re.fullmatch(r"\d{6}", l2[8:14]))
    return bool(starts_like_td1 and country_hint and has_name_separator and has_many_fillers and has_date_chunks)

