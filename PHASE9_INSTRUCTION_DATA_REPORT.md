# Phase 9 — Instruction Data Pipeline Report

## Objective

Improve the model's instruction-following behavior without replacing the existing pretraining corpus or pretending that arbitrary source prose is supervised question-answer data.

## Current design

`data/build_instruction_data.py` remains deterministic and auditable. It reads source text, extracts explicit question/answer structures, loads the curated healthcare examples, preserves provenance, deduplicates examples, and creates deterministic 90/5/5 splits.

The builder still retains conservative passage-copy records (`grounded_extraction`, `grounded_explanation`, and `grounded_response`) in the generated artifact. These records are useful for provenance and future experiments, but they are not treated as default SFT supervision.

Explicit source Q/A records are now represented as:

- instruction: a request to answer using the source
- input: the question only
- response: the source answer

The previous format placed `Source answer: ...` in the input, which made the task partly self-copying and weakened the question-to-answer training signal.

The curated `data/healthcare_examples.jsonl` records are also loaded into the generated instruction artifact. These contain distinct instruction/input/response targets rather than copying the entire input into the response.

## SFT filtering

`data/instruction_dataset.py` now defines a supervised category allowlist:

- `health_information`
- `simplification`
- `source_qa`
- `summarization`
- `terminology`

Passage-copy categories are excluded by default during instruction training.

`train.py` exposes this through `--instruction-categories`, while retaining the supervised-only default. This makes the training objective explicit and allows controlled experiments with other categories without changing the generated artifact.

## Why this change matters

The earlier SFT dataset contained many examples where `input == response`. Training on large numbers of these records rewards the model for reproducing source passages rather than learning the behavior needed for a user question followed by a concise answer. The change separates provenance/audit data from the examples used to optimize the instruction-following objective.

This does not claim that the resulting dataset is sufficient for high-quality medical QA. The curated set is intentionally small and the explicit source-QA extraction is conservative. The change is an objective correction, not a substitute for a larger reviewed QA corpus.

## Validation requirements

Before a full SFT run:

1. Rebuild `data/instruction` from the source corpus.
2. Verify generated JSONL loads without duplicates.
3. Run the full pytest suite.
4. Run a short CPU SFT smoke test.
5. Compare the resulting checkpoint against the previous SFT checkpoint using held-out generation prompts.

A successful training loss alone is not evidence of medical answer quality. The real-checkpoint inference test should remain part of the validation process.

## Safety boundary

The generated records are not a substitute for clinical validation and should not be treated as clinical advice. The builder intentionally does not invent diagnoses, treatment recommendations, dosages, or other medical facts.
