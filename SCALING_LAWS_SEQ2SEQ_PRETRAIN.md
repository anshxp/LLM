# Encoder-decoder pretraining experiment

This branch adds an isolated encoder-decoder architecture and span-denoising pretraining entry point. The existing decoder-only/legacy model APIs are left unchanged.

## Scaling-law basis
Hoffmann et al. (2022), Training Compute-Optimal Large Language Models, https://arxiv.org/abs/2203.15556, reported a compute-optimal allocation near 20 training tokens per parameter for large language models. This is a heuristic, not a universal law for small corpora or encoder-decoder denoising. File bytes are not token count.

## Configuration
- Context length: 512, doubled from 256.
- Shared token embeddings and output head.
- Width 128; 2 encoder layers; 2 decoder layers; 4 heads.
- Token-level span corruption masks approximately 15% of contiguous source positions; decoder reconstructs the original sequence.
- Batch size 1, gradient accumulation 8, AdamW learning rate 1e-4.
- The trainer computes a target budget of parameters multiplied by 20, counts target tokens consumed, and stops once that budget is reached or the epoch cap is hit.
- Use validation loss and coherence evaluations to determine whether this heuristic is useful for this dataset.

## Required first step: count usable tokens
This count matches the training dataset's current behavior (skip records with fewer than 8 tokens and truncate each record to the configured context).

```powershell
python -m data.audit_pretrain_scale --parquet "data/raw/stage 1/train.parquet" --tokenizer data/processed/tokenizer.json --context-length 512
```

The model has an embedding-table parameter floor, so the theoretical token/parameter ratio may not be achievable at a smaller width. Use the count as a constraint, not as proof of an optimum.

## Commands

Run the smoke test first:
```powershell
python -m training.pretrain_seq2seq --parquet "data/raw/stage 1/train.parquet" --device cuda --epochs 1 --max-train-batches 2 --checkpoint-dir checkpoints/seq2seq_smoke
```

After the smoke test and token audit:
```powershell
python -m training.pretrain_seq2seq --parquet "data/raw/stage 1/train.parquet" --tokenizer data/processed/tokenizer.json --device cuda --epochs 10 --checkpoint-dir checkpoints/seq2seq_pretrain --context-length 512
```

Resume:
```powershell
python -m training.pretrain_seq2seq --parquet "data/raw/stage 1/train.parquet" --tokenizer data/processed/tokenizer.json --device cuda --epochs 10 --checkpoint-dir checkpoints/seq2seq_pretrain --context-length 512 --resume
```

Old decoder-only checkpoints are not compatible and are not loaded. These commits prepare the pipeline but do not execute training or create trained checkpoints.
