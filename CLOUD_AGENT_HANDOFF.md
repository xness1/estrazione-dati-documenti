# Cloud Agent Handoff

## Stato attuale

- Pipeline locale pronta in `scripts/run_extraction.py`
- Indicizzazione dataset sporco in `scripts/index_documents.py`
- Riorganizzazione content-based in `scripts/reorganize_dataset.py`
- Output locale:
  - `output/document_manifest.json`
  - `output/reorganization_manifest.json`
  - `output/cloud_jobs.jsonl`
  - `output/id_back_extraction.json`

## Contratto input/output consigliato

- **Input job**
  - `job_id`
  - `bucket/path` o cartella sorgente
  - opzioni (`debug_images`, lingua OCR)
  - `assume_from_filename=false` (sempre)
- **Output job**
  - JSON con array risultati per file/pagina
  - flag `is_probable_id_back`
  - campi MRZ estratti
  - blocco indirizzo (se trovato)
  - diagnostica (`errors`, `warnings`, `processing_ms`)

## Pipeline cloud consigliata

1. Worker prende job da coda (SQS/PubSub/Redis queue).
2. Scarica file da storage.
3. Esegue `reorganize_dataset` per separare `id_cards/confirmed|review` da altri documenti.
4. Esegue pipeline OCR retro carta solo su `id_cards/confirmed`.
5. Salva JSON risultati su storage + DB.
6. Per casi dubbi (`id_cards/review`) invia a fallback Vision/LLM o revisione umana.
7. Genera jobs di coda tramite `prepare_cloud_jobs` (jsonl) senza trust sui filename.

## Priorita immediate

1. Aggiungere endpoint API per orchestrare step `reorganize -> extract`.
2. Separare output cloud in bucket distinti (`confirmed`, `review`, `errors`).
3. Salvare sempre evidenze di classificazione per audit.
4. Dockerizzare worker per deploy.

