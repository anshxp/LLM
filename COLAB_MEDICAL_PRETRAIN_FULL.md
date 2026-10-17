# Full medical-corpus pretraining on free Google Colab

This experiment uses the **entire available train split** from both selected sources:
1. [MedRAG PubMed](https://huggingface.co/datasets/MedRAG/pubmed) — first source.
2. [armanc/scientific_papers, pubmed config](https://huggingface.co/datasets/armanc/scientific_papers) — second source.

The downloader uses Hugging Face streaming and writes small resumable Parquet shards directly to Google Drive. It does not keep the whole dataset in RAM or Colab's temporary disk. Full-corpus preparation and model training are separate jobs.

## Important constraints

- “Full data” means iterating to natural exhaustion of each repository's train split. The exact number of usable documents/tokens is discovered by the run and recorded in `manifest.json`; it is not assumed from repository marketing numbers.
- Free Colab sessions can disconnect or impose usage limits. The downloader saves its manifest after every shard. Re-run the same command and output path to resume; Hugging Face streaming may need to skip rows already consumed, so resume time can be significant.
- The full merged Parquet may be large and needs extra Drive space. Shards are the durable source of truth; merging is optional for preparation but required by the current single-Parquet trainer.
- The current trainer scans the corpus to count tokens before training, scans it again for training, and uses a deterministic row-based split. A full training run over both entire sources may take many Colab sessions or be impractical under free-tier quotas. Do not interpret data download completion as training completion.
- The second repository reports `license: unknown`. Review the current dataset card and licenses/provenance of underlying articles before use. Do not assume redistribution or commercial-use rights.
- Both corpora are medical/scientific text for span-corruption denoising pretraining only. Keep medical QA for a separate SFT stage and isolate evaluation examples.

## Colab steps

Mount Google Drive and clone the experiment branch. Copy the project tokenizer into persistent Drive storage so it survives a Colab reset:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!pip install -q datasets pyarrow tokenizers
!git clone --branch experiment/encoder-decoder-scaling-pretrain https://github.com/anshxp/LLM.git
%cd LLM
!mkdir -p /content/drive/MyDrive/LLM
!cp data/processed/tokenizer.json /content/drive/MyDrive/LLM/tokenizer.json
```

Download **both full train splits**. This may require repeated sessions:

```bash
!python -m data.prepare_medical_corpora_full --output-dir /content/drive/MyDrive/LLM/medical_corpus_full --shard-rows 100
```

Re-run that exact command after a disconnect. When both sources are complete, merge the shards into one Parquet file (requires sufficient free Drive space):

```bash
!python -m data.prepare_medical_corpora_full --output-dir /content/drive/MyDrive/LLM/medical_corpus_full --shard-rows 1000 --merge
```

Inspect the manifest and sample rows before training:

```python
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/content/drive/MyDrive/LLM/medical_corpus_full")
print(json.loads((root / "manifest.json").read_text()))
p = next(root.glob("medrag_pubmed_*.parquet"))
print(pq.read_table(p).slice(0, 2).to_pylist())
```

Run the tokenizer token audit and smoke test on the merged file:

```bash
!python -m data.audit_pretrain_scale --parquet /content/drive/MyDrive/LLM/medical_corpus_full/medical_pretrain.parquet --tokenizer /content/drive/MyDrive/LLM/tokenizer.json --context-length 512
!python -m training.pretrain_seq2seq --parquet /content/drive/MyDrive/LLM/medical_corpus_full/medical_pretrain.parquet --tokenizer /content/drive/MyDrive/LLM/tokenizer.json --device cuda --epochs 1 --max-train-batches 2 --checkpoint-dir /content/drive/MyDrive/LLM/checkpoints/seq2seq_medical_smoke
```

Only after smoke-test success, start full-corpus training. The trainer defaults to two passes over the training split and uses the token count from the project's tokenizer to choose model width. Save all checkpoints/logs on Drive:

```bash
!python -m training.pretrain_seq2seq --parquet /content/drive/MyDrive/LLM/medical_corpus_full/medical_pretrain.parquet --tokenizer /content/drive/MyDrive/LLM/tokenizer.json --device cuda --epochs 10 --target-data-passes 2 --target-tokens-per-parameter 20 --checkpoint-dir /content/drive/MyDrive/LLM/checkpoints/seq2seq_medical_full --context-length 512
```

Resume with the same command and add `--resume`. Preserve identical model/training arguments when resuming.

## Model specification carried forward

- Encoder–decoder Transformer with encoder self-attention, decoder causal self-attention, cross-attention, residual connections and layer normalization.
- Context length 512 (the previous 256-token setting doubled).
- 2 encoder layers, 2 decoder layers, 4 attention heads; shared embeddings/output head.
- Batch size 1, gradient accumulation 8, AdamW learning rate 1e-4, weight decay 0.01, gradient clipping 1.0.
- Width auto-selected from measured usable tokens with a 20 tokens/parameter heuristic; this is only a reference, not a universal optimum, and vocabulary embeddings may set a parameter floor.
- Span-corruption noise density 15%; validation denoising loss and generated reconstructions must be inspected.

The code update prepares the full-data workflow and commands. It does **not** execute a Google Colab GPU job, download data to your Drive, or create trained checkpoints. You must run the notebook cells in your Colab account.
