# Encoder-decoder pretraining experiment

This branch adds a fresh encoder-decoder architecture and span-denoising pretraining entry point.

Scaling reference: Hoffmann et al. (2022), Training Compute-Optimal Large Language Models, https://arxiv.org/abs/2203.15556. Their approximate 20 training tokens per parameter is a heuristic, not a universal law for small-data encoder-decoder denoising. Measure actual usable tokens with the project tokenizer first.

Configuration defaults: context 512 (2x 256), vocab loaded from tokenizer, width 128, two encoder layers, two decoder layers, four heads, shared token embeddings/output head, batch 1, gradient accumulation 8, AdamW LR 1e-4.

Pretraining objective: 15% of source token positions are replaced by <unk>; the bidirectional encoder processes the corrupted source, and the causal decoder reconstructs the original sequence through cross-attention. This is denoising pretraining, not ordinary decoder-only next-token training. The encoder-decoder architecture is chosen because the project's desired conditional answering requires that interface, not because cross-attention alone guarantees better output.

Commands (run from repository root):
- Smoke test: python -m training.pretrain_seq2seq --parquet "data/raw/stage 1/train.parquet" --device cuda --epochs 1 --max-train-batches 2 --checkpoint-dir checkpoints/seq2seq_smoke
- Train: python -m training.pretrain_seq2seq --parquet "data/raw/stage 1/train.parquet" --device cuda --epochs 10 --checkpoint-dir checkpoints/seq2seq_pretrain --context-length 512
- Resume: python -m training.pretrain_seq2seq --parquet "data/raw/stage 1/train.parquet" --device cuda --epochs 10 --checkpoint-dir checkpoints/seq2seq_pretrain --resume

Old decoder-only checkpoints are not compatible. This change does not start training or create checkpoints. Run the smoke test and token audit before a long run.
