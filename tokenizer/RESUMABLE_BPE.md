# Experimental resumable BPE trainer

File: \`tokenizer/train_tokenizer_resumable.py\`

This is a custom, SQLite-backed BPE implementation intended to support durable restart after process death. It does **not** resume the internal state of Hugging Face \`tokenizers.BpeTrainer\`; it maintains its own training state.

## What is persisted

The \`--state-db\` SQLite database stores:

- Dataset identity, split, text column, normalizer and pre-tokenizer configuration.
- Raw dataset row position, valid document count, and character count during corpus ingestion.
- Unique pre-tokenized piece strings and their corpus frequencies.
- Per-piece adjacent-pair occurrences and global frequency-weighted pair counts.
- Ordered BPE merges and the token vocabulary.
- The current phase: corpus ingestion, pair-count initialization, or merge learning.

Corpus ingestion batches and each merge update are committed transactionally. If the process or VM is interrupted, rerun the same command with the same state database and settings. The script resumes from the stored phase. If the input configuration differs, it refuses to reuse the checkpoint. Increasing the target vocabulary size is allowed.

## Important limitations

- This is an experimental custom BPE implementation, not a drop-in continuation of the previous \`tokenizers.BpeTrainer\` run. The old failed run cannot be resumed from its log; this trainer must build its own checkpoint database from the beginning.
- It has not yet been validated against the project's model-training tokenizer expectations. Run the smoke test below and compare encoding behavior before using its output to train the LLM.
- SQLite makes state durable, not fast. Pair-count initialization and updating affected pieces for every merge can be expensive, and the state database may become large. Do not launch the complete PubMed corpus until the smoke test and a larger benchmark have been inspected.
- Keep the database, its \`-wal\` / \`-shm\` companions when the process is live, and output artifacts on the persistent \`/data\` disk. Do not copy only the main database file while the process is writing.
- Do not run two processes against the same state database.
- The tokenizer is exported after the current run reaches the requested vocabulary target or no pair meets \`--min-frequency\`. The SQLite database remains the authoritative resumable checkpoint.

## Smoke test on Azure

Use a new state DB and output directory for each independent test. This command intentionally uses a small dataset slice; it is a correctness/recovery test, not a final tokenizer.

\`\`\`bash
cd /data/LLM
source .venv/bin/activate
mkdir -p /data/LLM/artifacts/resumable_bpe_smoke

python -u tokenizer/train_tokenizer_resumable.py \\
  --dataset common-pile/pubmed \\
  --split train \\
  --output-dir /data/LLM/artifacts/resumable_bpe_smoke/output \\
  --state-db /data/LLM/artifacts/resumable_bpe_smoke/state.sqlite \\
  --vocab-size 1000 \\
  --min-frequency 2 \\
  --ingest-batch-rows 10 \\
  2>&1 | tee /data/LLM/artifacts/resumable_bpe_smoke/run.log
\`\`\`

To test recovery, interrupt a smoke run after it has logged progress, then rerun the exact same command. It should log the saved phase and resume rather than double-counting committed input rows. Verify that the output directory contains \`tokenizer.json\` and \`tokenizer_report.json\`.

## Do not use the old command

Do not run \`tokenizer/train_tokenizer_hf.py\` expecting it to use this checkpoint database. That original script remains non-resumable. Use this new script explicitly when testing the custom trainer.
