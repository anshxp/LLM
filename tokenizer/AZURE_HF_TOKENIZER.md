# Azure: train tokenizer from Common Pile PubMed

This workflow streams `common-pile/pubmed` from Hugging Face directly on the Azure VM. It does not download the full dataset to `/data`. The Hugging Face repository's `main/data` folder is approximately 45.3 GB of compressed JSONL shards; its converted Parquet view is partial, so this workflow deliberately uses the dataset's streaming loader instead of the existing local-Parquet trainer.

## 1. Get this branch on the VM

Run in the Azure SSH terminal:

```bash
cd /data/LLM
git fetch origin azure/hf-common-pile-pubmed-tokenizer
git checkout -b azure/hf-common-pile-pubmed-tokenizer origin/azure/hf-common-pile-pubmed-tokenizer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install datasets tokenizers
```

If `.venv` already exists, the venv creation command may report that it exists; continue with `source .venv/bin/activate`.

## 2. Run a smoke test first

```bash
cd /data/LLM
source .venv/bin/activate
python -u tokenizer/train_tokenizer_hf.py \
  --dataset common-pile/pubmed \
  --split train \
  --output-dir /data/LLM/artifacts/tokenizer_hf_smoke \
  --vocab-size 8000 \
  --max-rows 10000 \
  --log-every 1000 2>&1 | tee /data/LLM/artifacts/tokenizer_hf_smoke.log
```

The smoke test must finish and produce both `tokenizer.json` and `tokenizer_report.json` before the full run. It tests the streaming connection, text column, training, and output writing. Its tokenizer is only a smoke-test artifact.

## 3. Start the full tokenizer run so it survives SSH disconnects

Install tmux if needed:

```bash
sudo apt-get update
sudo apt-get install -y tmux
mkdir -p /data/LLM/artifacts
tmux new -s pubmed-tokenizer
```

Inside the tmux session, run:

```bash
cd /data/LLM
source .venv/bin/activate
set -o pipefail
python -u tokenizer/train_tokenizer_hf.py \
  --dataset common-pile/pubmed \
  --split train \
  --output-dir /data/LLM/artifacts/tokenizer_common_pile_pubmed_bpe32k \
  --vocab-size 32000 \
  --min-frequency 2 \
  --log-every 10000 2>&1 | tee /data/LLM/artifacts/tokenizer_common_pile_pubmed_bpe32k.log
```

Detach from tmux with `Ctrl+B`, then `D`. Do not stop or deallocate the Azure VM while the run is active. The run uses CPU and can take many hours on a 2-vCPU VM. Tokenizer training is not checkpointed mid-run; if interrupted before save, it must start over.

## 4. Reconnect and check progress

```bash
tmux attach -t pubmed-tokenizer
```

Or inspect the log without attaching:

```bash
tail -n 40 /data/LLM/artifacts/tokenizer_common_pile_pubmed_bpe32k.log
```

Check whether the process is active:

```bash
pgrep -af train_tokenizer_hf.py
```

Expected final files:

- `/data/LLM/artifacts/tokenizer_common_pile_pubmed_bpe32k/tokenizer.json`
- `/data/LLM/artifacts/tokenizer_common_pile_pubmed_bpe32k/tokenizer_report.json`
- `/data/LLM/artifacts/tokenizer_common_pile_pubmed_bpe32k.log`

The report records the number of documents and characters consumed, actual vocabulary size, training duration, and clinical-term probe tokenizations.
