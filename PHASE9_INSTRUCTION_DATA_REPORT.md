# Phase 9 — Instruction Data Pipeline Report

## Objective

Improve the model's instruction-following behavior without replacing the existing pretraining corpus or pretending that the corpus contains supervised question-answer pairs that it does not contain.

## Repository state reviewed

The model repository already contains `data/instruction_dataset.py` and `train_instruction.py`. The dataset class formats records as `### Instruction / ### Input / ### Response` and applies loss only to response targets. The fine-tuning entry point loads a Phase 7 checkpoint and overrides its optimizer learning rate with the requested fine-tuning rate.

The companion `LLM-Data` repository contains the corpus-building and healthcare-data utilities, including corpus ingestion/cleaning/deduplication code and a small `healthcare_examples.jsonl` file.

## What was added

`data/build_instruction_data.py` is a deterministic, source-grounded builder. It:

1. Reads `.txt` and `.md` files from a supplied `LLM-Data` checkout.
2. Extracts explicit question/answer structures when a question line is immediately followed by an answer line.
3. Extracts sufficiently long source paragraphs.
4. Creates three conservative instruction styles around those passages: grounded extraction, grounded explanation, and grounded response.
5. Copies the source passage into the response rather than generating medical claims.
6. Stores source-file provenance and a deterministic record ID.
7. Sorts by ID and creates deterministic 90/5/5 train/validation/test splits.
8. Refuses to emit a tiny dataset when fewer than 100 examples are produced.
9. Writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`.

`tests/test_build_instruction_data.py` verifies deterministic IDs, source provenance, non-empty records, and disjoint deterministic splits.

## Important limitation

This pipeline does **not** create high-quality semantic question-answer pairs from arbitrary medical prose. It deliberately avoids hallucinating answers. Consequently, it should be treated as an instruction-formatting and provenance layer, not as the final source of supervised medical QA.

The existing eight-example instruction set is too small for meaningful fine-tuning. Expanding it by copying arbitrary passages into responses can improve response-format conditioning, but it is unlikely by itself to solve the model's current semantic-generation problem. The model's observed outputs show domain mixing, repetition, and weak question answering; those problems require better supervised targets, not merely more copies of the pretraining corpus.

## Recommended next step

Use this pipeline to establish an auditable baseline, then add a reviewed teacher-generated or human-authored QA layer derived from the same source documents. Each generated answer should retain its source document and passage IDs, undergo schema/length/deduplication checks, and be held out by source document where possible to reduce leakage.

Only after that should `train_instruction.py` be run against the Phase 7 `best_model.pt` checkpoint. Compare base versus instruction-tuned checkpoints on the existing Phase 8 generation suite and held-out source-grounded QA tests.

## Safety boundary

The generated records are not a substitute for clinical validation and should not be treated as clinical advice. The builder intentionally does not invent diagnoses, treatment recommendations, dosages, or other medical facts.
