# LLM v2 training

v2 is trained from a fresh initialization. The old ~8.35M decoder-only checkpoint is not loaded into the encoder-decoder model.

## 1. Mount Google Drive in Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

The repository and the corpus can live in Drive. Do not copy the entire corpus into the Git repository.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Build the compact corpus artifacts

```bash
python -m data.corpus_pipeline \
  --books-root /content/drive/MyDrive/LLM/books \
  --sft-root /content/drive/MyDrive/LLM/Fine\ tuning\ 2 \
  --output-root data/processed/v2 \
  --tokenizer data/processed/tokenizer.json \
  --context-length 256
```

The books pipeline creates cleaned/deduplicated text splits. The SFT side uses the existing disk-backed Fine tuning 2 ingestion pipeline and writes sharded JSONL, so the large supervised corpus is not loaded into RAM at once.

## 4. Train v2 from scratch on books

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

Book pretraining uses a prefix-to-continuation objective: the encoder receives the first half of a text window and the decoder predicts the continuation. This gives the encoder-decoder architecture a language-modeling objective without turning book pretraining into a trivial copy task.

## 5. SFT from the best book checkpoint

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

DeepSeek MLA is not part of v2. It is a separate future attention-efficiency experiment; DeepSeek describes MLA primarily as KV-cache compression for efficient inference.
