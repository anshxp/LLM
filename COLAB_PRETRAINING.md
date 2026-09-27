# RXLM 2 Colab pretraining

This branch is intended for fresh RXLM 2 pretraining in Google Colab. The model is initialized from scratch; the RXLM 1 checkpoint is not used for this phase.

## Runtime assumptions

Use a Colab runtime with GPU enabled. A 12 GiB system-RAM runtime is sufficient for the bounded preprocessing path because the corpus is never loaded into one Python string or one Python token list during token-store construction.

The token store is written as a disk-backed array and consumed through NumPy memory mapping. Only individual training windows are copied into PyTorch tensors.

## Bootstrap

Run from the repository root:

```bash
cd /content/LLM
python -m pip install -r requirements.txt
python -m scripts.colab_preflight
```

If the repository is cloned somewhere other than `/content/LLM`, replace the `cd` path with the actual repository directory. Always run module commands from the repository root so Python can resolve `config`, `data`, `model`, and `training`.

## Base RXLM 2 corpus

The current base corpus is the processed book corpus under `data/processed/`.

First verify the repository:

```bash
python -m pytest -q
```

Train the tokenizer once if `data/processed/tokenizer.json` is not already present:

```bash
python -m data.train_tokenizer --dataset base --vocab-size 10000
```

Build the bounded token store:

```bash
python -m data.prepare_training_data --dataset base --chunk-chars 1048576
```

The preprocessing code deliberately does **not** tokenize arbitrary character slices. The project tokenizer uses Hugging Face's `Whitespace` pre-tokenizer, so chunks are ended at whitespace boundaries. This avoids the tokenization drift previously observed with independent 1024-character chunks while keeping the memory bound proportional to the configured chunk size.

The resulting files are:

```text
data/processed/token_store/base/train.tokens
data/processed/token_store/base/validation.tokens
data/processed/token_store/base/test.tokens
```

## Training

Run the normal training entry point after the token stores have been built:

```bash
python train.py --dataset base --device cuda --batch-size 1 --gradient-accumulation-steps 4
```

For a first smoke test, limit the number of batches:

```bash
python train.py --dataset base --device cuda --batch-size 1 --gradient-accumulation-steps 4 --epochs 1 --max-train-batches 20
```

The token store is disk-backed, so increasing the corpus size does not cause the entire token array to be copied into RAM.

## Continued pretraining corpus

The same preparation mechanism supports:

```text
data/processed/continued_pretraining/train.txt
data/processed/continued_pretraining/validation.txt
data/processed/continued_pretraining/test.txt
```

Build those token stores with:

```bash
python -m data.prepare_training_data --dataset continued_pretraining --chunk-chars 1048576
```

Do not run the continued-pretraining phase until the base RXLM 2 model has been trained and its checkpoint has been verified.

## Memory-safety rules

Do not use `Path.read_text()` on the full corpus for large preprocessing jobs. `load_token_ids_from_file()` remains only as a legacy/testing API; large training must use the token-store path.

Do not use `tokenizer.encode()` independently on arbitrary fixed-size character slices. That was demonstrated to change the token stream: the 5 MiB test produced 1,207,974 whole-file tokens versus 1,213,012 tokens with 1024-character independent chunks.

Do not upload the raw 2.5 GiB corpus into a Python variable. Keep the corpus on Google Drive or the Colab filesystem and stream it into the token store.

Keep checkpoints outside transient preprocessing directories and periodically copy important checkpoints to persistent Drive storage.
