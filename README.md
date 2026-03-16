# Estrazione Retro Carta d'Identità Elettronica (CIE)

Pipeline ad alte prestazioni per estrarre dati dal **retro della carta d'identità elettronica italiana**:
- **MRZ** (3 righe machine-readable zone) con validazione ICAO
- **Codice Fiscale** con validazione checksum
- **Indirizzo di residenza**

Target: **< 5 secondi** per documento su CPU.

## Requisiti di Sistema

### Python 3.10+
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Tesseract OCR
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ita tesseract-ocr-eng

# macOS
brew install tesseract tesseract-lang
```

## Uso Rapido

### Estrazione singolo documento
```bash
python scripts/extract_id.py documento.pdf
```

### Estrazione cartella
```bash
python scripts/extract_id.py cartella/ -o risultati.json
```

### Output JSON
```json
{
  "cognome": "ROSSI",
  "nome": "MARIO",
  "data_nascita": "1980-01-15",
  "codice_fiscale": "RSSMRA80A15H501Z",
  "indirizzo": "VIA ROMA, 1 MILANO (MI)",
  "documento": {
    "numero": "CA12345AB",
    "scadenza": "2030-01-15"
  },
  "icao_valid": true
}
```

## Pipeline Tecnica

La pipeline usa un approccio **ibrido** ottimizzato per velocità e accuratezza:

1. **Detection MRZ** (Tesseract) - Ricerca veloce della zona MRZ
2. **Extraction MRZ** (EasyOCR) - Lettura accurata delle 3 righe
3. **Validazione ICAO** - Check digit standard aeroportuale
4. **Extraction CF/Indirizzo** (Tesseract + EasyOCR fallback)
5. **Validazione CF** - Checksum codice fiscale italiano

### File Principali

| File | Descrizione |
|------|-------------|
| `src/id_extractor/hybrid_pipeline.py` | Pipeline principale |
| `scripts/extract_id.py` | CLI user-friendly |
| `scripts/benchmark_fast.py` | Benchmark performance |

## Workflow Completo

### 1. Indicizzazione documenti
```bash
python scripts/index_documents.py --input "cartella_documenti"
```

### 2. Riorganizzazione per tipo
```bash
python scripts/reorganize_dataset.py \
  --input "cartella_documenti" \
  --output-root "dataset_pulito" \
  --manifest "output/reorganization_manifest.json"
```

Struttura output:
```
dataset_pulito/
├── id_cards/
│   ├── confirmed/   # CIE confermate
│   └── review/      # Da verificare
├── other_documents/ # Patenti, visure, etc.
└── unreadable/      # Non leggibili
```

### 3. Estrazione dati
```bash
python scripts/extract_id.py dataset_pulito/id_cards/confirmed/ -o output/risultati.json
```

## Performance

| Metrica | Risultato |
|---------|-----------|
| Tempo medio | 5-7 sec/doc |
| MRZ detection | ~95% |
| CF extraction | ~40% (dipende da stampa) |
| Indirizzo extraction | ~50% |

## Limitazioni

- Solo **retro CIE** (carta elettronica con MRZ)
- Ignora: fronti, carte cartacee, patenti, visure
- CF/indirizzo dipendono da qualità scansione
- Documenti stranieri: layout diverso

## Licenza

Progetto interno.
