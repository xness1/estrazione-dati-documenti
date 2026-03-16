# Agent Operating Guide

## Goal

Process mixed document folders and extract data from Italian ID card back sides.
Never trust filename semantics. Classification is content-based only.

## Canonical workflow

1. Reorganize raw input into clean buckets:
   - `id_cards/confirmed`
   - `id_cards/review`
   - `other_documents`
   - `unreadable`
2. Run extraction only on `id_cards/confirmed`.
3. Route `id_cards/review` to review/fallback vision flow.
4. Persist outputs and evidence manifests.

## Commands

```bash
source .venv/bin/activate
python scripts/reorganize_dataset.py --input "esempi documenti" --output-root "dataset_pulito" --manifest "output/reorganization_manifest.json"
python scripts/run_extraction.py --input "dataset_pulito/id_cards/confirmed" --output "output/id_back_extraction_cleaned.json"
python scripts/prepare_cloud_jobs.py --manifest "output/reorganization_manifest.json" --jobs-out "output/cloud_jobs.jsonl"
```

## Non-negotiable constraints

- `assume_from_filename=false` for every cloud job.
- Keep originals untouched; write to new target roots.
- Use manifest + sha256 as source of truth.
- Keep evidence for audit (`content_evidence` in manifest).

## Key files

- `scripts/reorganize_dataset.py`
- `scripts/run_extraction.py`
- `scripts/prepare_cloud_jobs.py`
- `CLOUD_AGENT_HANDOFF.md`

