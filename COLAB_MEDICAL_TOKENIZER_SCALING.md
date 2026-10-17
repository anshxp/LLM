# Medical tokenizer + model scaling pipeline (Colab)

Use this branch: `experiment/medical-tokenizer-model-scaling`.

The pipeline deliberately trains tokenizer candidates from the downloaded medical corpus before initializing the model. Keep the chosen tokenizer and checkpoints together; never replace the tokenizer after training has started.

## 0. Mount Drive and clone the branch

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!pip install -q datasets pyarrow tokenizers
!git clone --branch experiment/medical-tokenizer-model-scaling https://github.com/anshxp/LLM.git
%cd /content/LLM
!mkdir -p /content/drive/MyDrive/LLM
```

If `/content/LLM` already exists from an earlier session, do not clone over it. Run `%cd /content/LLM`, then `!git fetch origin` and `!git checkout experiment/medical-tokenizer-model-scaling`.

## 1. Verify Drive read/write

```python
from pathlib import Path
root = Path("/content/drive/MyDrive/LLM")
root.mkdir(parents=True, exist_ok=True)
p = root / "write_test.txt"
p.write_text("ok", encoding="utf-8")
assert p.read_text(encoding="utf-8") == "ok"
p.unlink()
print("Drive read/write OK:", root)
```

Do not continue if this fails.

## 2. Download the complete source corpora (resumable)

```bash
!python -m data.prepare_medical_corpora_full \
  --output-dir /content/drive/MyDrive/LLM/medical_corpus_full \
  --shard-rows 1000
```

If Colab disconnects, remount Drive, return to `/content/LLM`, and rerun the exact same command. The script writes a manifest after each shard. This traverses both full train splits and may require many sessions.

Check progress:

```python
import json
from pathlib import Path
manifest = Path("/content/drive/MyDrive/LLM/medical_corpus_full/manifest.json")
print(json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else "Manifest not created")
```

Wait until both source entries show `complete: true` and overall status is `download_complete`. If manifest creation fails with an I/O error, fix the Drive mount/storage problem first.

## 3. Train tokenizer candidates on this medical corpus

This reads Parquet shards directly, so you do not need an existing tokenizer file. The default candidates are 10k, 16k, and 32k vocabularies.

```bash
!python -m data.train_medical_tokenizer \
  --corpus-dir /content/drive/MyDrive/LLM/medical_corpus_full \
  --output-dir /content/drive/MyDrive/LLM/tokenizers \
  --vocab-sizes 10000 16000 32000 \
  --max-documents 100000 \
  --seed 42
```

Review the report:

```python
import json
from pathlib import Path
report = Path("/content/drive/MyDrive/LLM/tokenizers/tokenizer_comparison.json")
print(json.dumps(json.loads(report.read_text(encoding="utf-8")), indent=2))
```

Choose the vocabulary after comparing characters per token, unknown-token counts, model parameter budget, and VRAM. Do not choose solely because a vocabulary is larger. The 32k embedding table is 3.2 times the size of a 10k table at the same width.

For the example below, select 16k. If your comparison supports another size, change the source filename accordingly:

```bash
!cp /content/drive/MyDrive/LLM/tokenizers/medical_bpe_16000.json /content/drive/MyDrive/LLM/tokenizer.json
```

Confirm the selected tokenizer:

```python
from data.tokenizer import Tokenizer
tok = Tokenizer.from_file("/content/drive/MyDrive/LLM/tokenizer.json")
print("Vocab:", len(tok), "Special IDs:", tok.token_to_id)
assert all(v is not None for v in tok.token_to_id.values())
```

## 4. Merge shards into the Parquet file used by the current trainer

The current trainer and token audit expect one Parquet file. Keep the shards as the durable source of truth; merging requires additional Drive space.

```bash
!python -m data.prepare_medical_corpora_full \
  --output-dir /content/drive/MyDrive/LLM/medical_corpus_full \
  --shard-rows 1000 \
  --merge
```

Inspect status and sample records:

```python
import json
from pathlib import Path
import pyarrow.parquet as pq
root = Path("/content/drive/MyDrive/LLM/medical_corpus_full")
print(json.loads((root / "manifest.json").read_text(encoding="utf-8"))["status"])
p = root / "medical_pretrain.parquet"
print("Parquet exists:", p.exists(), "bytes:", p.stat().st_size if p.exists() else None)
if p.exists():
    print(pq.read_table(p).slice(0, 2).to_pylist())
```

## 5. Audit tokens and parameter budget

```bash
!python -m data.audit_pretrain_scale \
  --parquet /content/drive/MyDrive/LLM/medical_corpus_full/medical_pretrain.parquet \
  --tokenizer /content/drive/MyDrive/LLM/tokenizer.json \
  --context-length 512 \
  --target-tokens-per-parameter 20
```

This scans the corpus to estimate usable tokens. The 20:1 ratio is a rough reference, not a law for a small encoder-decoder denoising model.

## 6. Smoke test before a long run

```bash
!python -m training.pretrain_seq2seq \
  --parquet /content/drive/MyDrive/LLM/medical_corpus_full/medical_pretrain.parquet \
  --tokenizer /content/drive/MyDrive/LLM/tokenizer.json \
  --device cuda \
  --epochs 1 \
  --max-train-batches 2 \
  --checkpoint-dir /content/drive/MyDrive/LLM/checkpoints/seq2seq_smoke \
  --context-length 512 \
  --auto-architecture
```

Confirm it reports the expected vocabulary, parameter count, architecture and successfully saves a checkpoint. The smoke-test checkpoint is separate from the full run.

## 7. Full pretraining

```bash
!python -m training.pretrain_seq2seq \
  --parquet /content/drive/MyDrive/LLM/medical_corpus_full/medical_pretrain.parquet \
  --tokenizer /content/drive/MyDrive/LLM/tokenizer.json \
  --device cuda \
  --epochs 10 \
  --target-data-passes 2 \
  --target-tokens-per-parameter 20 \
  --checkpoint-dir /content/drive/MyDrive/LLM/checkpoints/seq2seq_medical_full \
  --context-length 512 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 0.0001 \
  --auto-architecture
```

Resume the same run after a disconnect by rerunning the same command and adding `--resume`. Keep the tokenizer, data, and architecture arguments unchanged. The script resumes from `latest.pt`; all checkpoints and logs are on Drive.

## Important notes

- The automatic architecture search chooses a width, head count, and encoder/decoder depths from a candidate set that fits the token-derived parameter budget. The instantiated parameter count is printed and checked against the heuristic.
- Larger vocabulary increases the tied token embedding/output table and may force a larger parameter floor even when the corpus token budget is small.
- The current denoising objective masks source positions and reconstructs the original sequence. This is encoder-decoder denoising pretraining, not decoder-only next-token training.
- The complete dataset may be too large for a single free Colab session; data download and model training are separate jobs.
- Verify dataset provenance/licensing before use or redistribution, especially the `armanc/scientific_papers` source, whose license metadata may be unknown.
