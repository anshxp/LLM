# LLM v2 training

v2 is trained from a fresh initialization. The old ~8.35M decoder-only checkpoint is not loaded into the encoder-decoder model.

## Corpus layout

The repository uses fixed source locations:

```text
data/
├── raw/                 # ALL pretraining data
│   ├── books / PDFs / text files
│   └── *.parquet        # e.g. TheBlueScrubs-v1-fixed shards
│
└── Fine tuning 2/       # ALL supervised fine-tuning data
```

Do not commit either corpus to Git. The `.gitignore` already excludes `data/`.

The current pretraining corpus includes `openmed-community/TheBlueScrubs-v1-fixed`. Its Hugging Face dataset card documents a single `text` field, 11,080,331 training documents and about 57 GB of Parquet data. The local pipeline streams the Parquet `text` column in batches instead of loading the 57 GB corpus into RAM.

## 1. Mount Google Drive in Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

Keep the large corpora in Drive. The working repository must expose them at exactly `data/raw` and `data/Fine tuning 2` before the pipeline is run.

For example, after cloning the repository into `/content/LLM`, the expected paths are:

```text
/content/LLM/data/raw/
/content/LLM/data/Fine tuning 2/
```

Do not pass alternate corpus paths to the pipeline; the source paths are intentionally fixed so the training code remains reproducible.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

`pyarrow` is required because Parquet is now a supported pretraining source.

## 3. Build the compact corpus artifacts

Run from the repository root:

```bash
python -m data.corpus_pipeline \
  --output-root data/processed/v2 \
  --tokenizer data/processed/tokenizer.json \
  --context-length 256
```

The pipeline automatically reads:

```text
data/raw              -> book/pretraining corpus
data/Fine tuning 2     -> supervised SFT corpus
```

For pretraining, text/PDF sources are cleaned and exact-deduplicated. Parquet sources are streamed through the `text` column and exact-deduplicated using a disk-backed SQLite index, so the 57 GB dataset is not loaded into RAM. The processed output is written under `data/processed/v2/books` as train/validation/test text splits.

For SFT, the existing disk-backed Fine tuning 2 ingestion pipeline creates deterministic JSONL shards and does not load the complete supervised corpus into RAM.

## 4. Train v2 from scratch on pretraining data

```bash
python train_v2.py \
  --stage pretrain \
  --data-root data/processed/v2 \
  --tokenizer data/processed/tokenizer.json \
  --device cuda \
  --max-steps 10000 \
  --gradient-accumulation-steps 8 \
  --eval-every 500
```

Book/pretraining training uses a prefix-to-continuation objective: the encoder receives the first half of a text window and the decoder predicts the continuation. This gives the encoder-decoder architecture a language-modeling objective without turning pretraining into a copy task.

## 5. SFT from the best pretraining checkpoint

```bash
python train_v2.py \
  --stage sft \
  --data-root data/processed/v2 \
  --tokenizer data/processed/tokenizer.json \
  --pretrained-checkpoint checkpoints/v2/best_model.pt \
  --device cuda \
  --max-steps 5000 \
  --gradient-accumulation-steps 8 \
  --eval-every 250
```

SFT uses the instruction/input as encoder context and the response as the decoder target. The loss is therefore directly on the generated response.

## Important resource note

The default model is ~10.8M parameters. Colab system RAM is not the same as GPU VRAM, so the practical batch size depends on the assigned GPU. v2 intentionally uses batch size 1 with gradient accumulation until padding masks are added.

DeepSeek MLA is not part of v2. It is a separate future attention-efficiency experiment; MLA primarily targets KV-cache/inference efficiency.
