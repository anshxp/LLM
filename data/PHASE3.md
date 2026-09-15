# Phase 3 — Data Pipeline

Phase 3 turns raw healthcare documents into reproducible training data.

## Pipeline

1. Inventory supported sources recursively.
2. Ingest PDF and text sources without modifying raw files.
3. Normalize Unicode, line endings, whitespace, and control characters.
4. Apply conservative quality filters.
5. Remove exact file duplicates using SHA-256.
6. Remove duplicate normalized documents using content hashes.
7. Record accepted-source provenance in `manifest.jsonl`.
8. Train a 10,000-token BPE tokenizer from the cleaned corpus.
9. Convert corpus text into token IDs for model training.
10. Build fixed-length language-model training sequences.

The corpus and manifest are intermediate artifacts. The tokenizer is stored as `data/processed/tokenizer.json` and is the tokenizer used by training-data preparation.

The raw dataset under `data/raw` is never modified by the pipeline.
