# Phase 3 — Data Pipeline

Phase 3 turns raw healthcare documents into reproducible, leakage-resistant training data.

## Completed pipeline

1. Inventory supported sources recursively.
2. Ingest PDF, TXT, and Markdown sources without modifying raw files.
3. Normalize Unicode, line endings, whitespace, and control characters.
4. Apply conservative quality filters.
5. Remove exact file duplicates using SHA-256.
6. Remove duplicate normalized documents using content hashes.
7. Remove highly similar documents with deterministic SimHash fingerprints.
8. Record source hashes, fingerprints, sizes, and dataset split in `manifest.jsonl`.
9. Deterministically split accepted documents into 90% train, 5% validation, and 5% test buckets.
10. Train a 10,000-token BPE tokenizer from the training split only.
11. Convert each corpus split into token IDs with the trained tokenizer.
12. Pack complete fixed-length language-model sequences with configurable stride.
13. Train the model from the explicit training split and evaluate on the validation split.
14. Keep the held-out test split available for final evaluation rather than training.

## Generated artifacts

- `data/processed/corpus.txt` — complete accepted corpus.
- `data/processed/train.txt` — training documents.
- `data/processed/validation.txt` — validation documents.
- `data/processed/test.txt` — held-out test documents.
- `data/processed/manifest.jsonl` — provenance and split metadata.
- `data/processed/tokenizer.json` — trained BPE tokenizer.

The raw dataset under `data/raw` is never modified by the pipeline. Dataset splits are assigned from content hashes, so rebuilding the same source set produces the same split assignments.

## Rebuild order

```text
python -m data.build_corpus
python -m data.train_tokenizer
python -m data.prepare_training_data
pytest
```

The tokenizer must be trained after the corpus split is built, and model training must use the train/validation split rather than randomly splitting already-packed sequences.
