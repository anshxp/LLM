# 512-context encoder-decoder pretraining experiment

This is an isolated experiment branch. It does not overwrite or reuse decoder-only checkpoints.

## Objective

Train a small encoder-decoder Transformer with causal decoder self-attention and cross-attention on the existing Stage 1 Parquet corpus. For each stream of corpus tokens, the encoder receives a 512-token source prefix and the decoder predicts the following 512 tokens. The encoder has bidirectional self-attention; the decoder has causal self-attention and attends to encoder states through cross-attention. The decoder input is `<bos> + target[:-1]`, while the loss is against the unshifted target sequence.

This is source-to-future-continuation pretraining, not instruction fine-tuning. It does not make the base model a good assistant by itself.

## Model budget

- Vocabulary: read from `data/processed/tokenizer.json` (expected 10,000)
- Source context: 512 tokens
- Target horizon: 512 tokens
- Embedding width: 256
- Encoder layers: 4
- Decoder layers: 4
- Attention heads: 4
- Feed-forward width: 4x embedding width
- Dropout: 0.1
- Token embedding/output projection: tied
- Estimated parameter count: roughly 9–10 million, confirmed at startup

Context length is doubled from 256 to 512; width and layer counts are held rather than blindly doubled, because widening and doubling both stacks would increase parameters and VRAM substantially. The chosen parameter budget is suited to a small experiment, not a production-strength foundation model.

## Corpus and splits

The default corpus path is `data/raw/stage 1/train.parquet`. Dataset rows are deterministically separated into train/validation/test before building examples. The loader streams Parquet batches rather than loading the entire file into RAM. It reuses the current tokenizer and does not retrain it.

## Smoke test (PowerShell)

From the repository root, with the virtual environment active:

```powershell
python train_seq2seq_pretraining.py --parquet "data/raw/stage 1/train.parquet" --tokenizer data/processed/tokenizer.json --checkpoint-dir checkpoints/seq2seq_318mb_smoke --device cuda --epochs 1 --max-train-batches 5 --batch-size 1 --gradient-accumulation-steps 1
```

Run tests before a full run:

```powershell
pytest -q
```

## Full run

```powershell
python train_seq2seq_pretraining.py --parquet "data/raw/stage 1/train.parquet" --tokenizer data/processed/tokenizer.json --checkpoint-dir checkpoints/seq2seq_318mb --device cuda --epochs 10 --batch-size 1 --gradient-accumulation-steps 4 --learning-rate 1e-4 --checkpoint-every-steps 100
```

Resume:

```powershell
python train_seq2seq_pretraining.py --parquet "data/raw/stage 1/train.parquet" --tokenizer data/processed/tokenizer.json --checkpoint-dir checkpoints/seq2seq_318mb --device cuda --epochs 10 --resume checkpoints/seq2seq_318mb/latest.pt
```

For Colab, upload/clone the repository and mount persistent storage (for example Google Drive) before training. Checkpoints written only to a temporary runtime are not durable.

## Evaluation

Compare best validation loss and perplexity, inspect generated continuation examples, and evaluate on the untouched test split after choosing a checkpoint. Because the objective predicts a future segment conditioned on the preceding segment, do not compare its perplexity directly against the earlier decoder-only run without keeping the token-level loss protocol in mind.
