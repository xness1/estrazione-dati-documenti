# Agent Operating Guide

## Goal

Process mixed document folders and extract data from Italian electronic ID card (CIE) back sides.
Never trust filename semantics. Classification is content-based only.

## Quick Start (Recommended)

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install Tesseract (if not present)
sudo apt-get install -y tesseract-ocr tesseract-ocr-ita tesseract-ocr-eng

# Extract single document
python scripts/extract_id.py documento.pdf

# Extract folder
python scripts/extract_id.py "esempi documenti" -o output/risultati.json
```

## Full Workflow

### 1. Reorganize raw input into clean buckets
```bash
python scripts/reorganize_dataset.py \
  --input "esempi documenti" \
  --output-root "dataset_pulito" \
  --manifest "output/reorganization_manifest.json"
```

Output structure:
- `id_cards/confirmed` - CIE back sides confirmed
- `id_cards/review` - Needs manual review
- `other_documents` - Licenses, visuras, etc.
- `unreadable` - Cannot process

### 2. Run extraction on confirmed CIE
```bash
python scripts/extract_id.py "dataset_pulito/id_cards/confirmed" -o "output/id_back_extraction.json"
```

### 3. Prepare cloud jobs (optional)
```bash
python scripts/prepare_cloud_jobs.py \
  --manifest "output/reorganization_manifest.json" \
  --jobs-out "output/cloud_jobs.jsonl"
```

## Key Files

| File | Purpose |
|------|---------|
| `src/id_extractor/hybrid_pipeline.py` | Main extraction pipeline (Tesseract+EasyOCR) |
| `scripts/extract_id.py` | CLI tool for extraction |
| `scripts/reorganize_dataset.py` | Dataset organization |
| `scripts/prepare_cloud_jobs.py` | Cloud job preparation |

## Non-negotiable Constraints

- `assume_from_filename=false` for every cloud job
- Keep originals untouched; write to new target roots
- Use manifest + sha256 as source of truth
- Keep evidence for audit (`content_evidence` in manifest)
- Only process **electronic ID cards** (CIE with MRZ)

## Extracted Fields

From MRZ (ICAO validated):
- Document number, expiry date
- Surname, given name, birth date, sex, nationality

From OCR (best effort):
- Codice fiscale (with checksum validation)
- Residence address

## Performance Target

- Max 5 seconds per document on CPU
- Fail-fast with review queue for uncertain cases
