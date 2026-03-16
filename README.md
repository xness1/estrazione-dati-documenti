# Estrazione retro carta d'identita (MRZ + indirizzo)

Progetto base per lavorare su cartelle e sottocartelle non pulite con documenti misti (PDF/JPG/PNG), riconoscere in modo rapido il **retro della carta di identita** e leggere:

1. le **3 righe MRZ** in basso;
2. l'**indirizzo/residenza** nell'area poco sopra.

## Cosa fa la pipeline

- Scansiona ricorsivamente la cartella input.
- Ignora file rumorosi noti (`:Zone.Identifier`, `:com.dropbox.attrs`).
- Converte i PDF in immagini.
- Per ogni pagina, fa crop dell'area bassa e cerca pattern MRZ (TD1 3x30).
- Se il retro e probabile, estrae dati MRZ e poi OCR dell'area indirizzo.
- Scrive un JSON strutturato in output.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Nota OCR

Questo progetto usa `pytesseract`, quindi serve Tesseract installato nel sistema:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ita
```

## Esecuzione

Prima indicizza il materiale disordinato:

```bash
python3 scripts/index_documents.py --input "esempi documenti"
```

Questo crea:

- `output/document_manifest.json`
- `output/document_manifest.csv`

con una prima classificazione euristica dei file (carta identita, CF, visura, patente, altro).

Poi crea dataset pulito con classificazione **solo da contenuto** (non dal nome file):

```bash
python3 scripts/reorganize_dataset.py --input "esempi documenti" --output-root "dataset_pulito"
```

Struttura prodotta:

- `dataset_pulito/id_cards/confirmed` (carte identita confermate)
- `dataset_pulito/id_cards/review` (sospette da revisione)
- `dataset_pulito/other_documents`
- `dataset_pulito/unreadable`

Ogni file viene copiato con nome hash (`sha256`) per eliminare dipendenze dai nomi originali.

Prepara i job per cloud agent (JSONL):

```bash
python3 scripts/prepare_cloud_jobs.py --manifest "output/reorganization_manifest.json" --jobs-out "output/cloud_jobs.jsonl"
```

Poi esegui estrazione retro carta:

```bash
python3 scripts/run_extraction.py --input "dataset_pulito/id_cards/confirmed" --output "output/id_back_extraction.json"
```

Con debug immagini crop:

```bash
python3 scripts/run_extraction.py --input "esempi documenti" --debug-images
```

## Output

File: `output/id_back_extraction.json`

Per ogni retro carta trovato:

- file sorgente
- indice pagina
- linee MRZ raw
- campi MRZ principali (numero documento, nascita, scadenza, nome/cognome)
- indirizzo estratto (riga, CAP, citta, provincia)

## Prossimi step (cloud agent)

Quando sei pronto a passare al cloud agent:

1. separa `input/` e `output/` su storage cloud;
2. containerizza questo script (Docker);
3. aggiungi coda job (uno per file/cartella);
4. salva confidence e log per validazione manuale;
5. aggiungi fallback LLM vision solo sui casi dubbi.

