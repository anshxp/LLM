# Medical corpus pretraining on free Google Colab

This experiment uses **MedRAG PubMed first**, then a bounded subset of **armanc/scientific_papers (pubmed)**. Both are raw document text for encoder-decoder denoising pretraining. Medical QA datasets are deliberately not mixed into this pretraining corpus.

## Data sources and caution

- MedRAG PubMed: https://huggingface.co/datasets/MedRAG/pubmed. The loader expects a `train` split and tries `title`, `content`, `contents`, or `text`. The script validates by producing only non-empty text rows; inspect a few samples before a full run.
- Scientific Papers PubMed: https://huggingface.co/datasets/armanc/scientific_papers, configuration `pubmed`. The loader combines abstract, section names, and article text. The repository's license metadata has been reported as unknown; review the dataset card and underlying article licenses before using or redistributing this data. This branch does not assume commercial-use permission.

The default is a **bounded pilot**, not a full dataset download: 20,000 MedRAG source rows and 10,000 scientific-papers source rows. These are source rows consumed; short/empty rows can be discarded, so actual usable documents will be fewer. Streaming avoids downloading the full datasets into Colab's temporary disk. Output shards are written to Drive, and a manifest supports resuming after disconnection. Resuming a Hugging Face stream skips previously consumed rows, which may take time.

## Colab setup

Mount Drive first:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Install dependencies:

```python
!pip install -q datasets pyarrow tokenizers
```

Clone the branch:

```bash
!git clone --branch experiment/encoder-decoder-scaling-pretrain https://github.com/anshxp/LLM.git
%cd LLM
```

Download a small streaming pilot into persistent Drive storage:

```bash
!python -m data.prepare_medical_corpora --output-dir /content/drive/MyDrive/LLM/medical_corpus --medrag-records 20000 --scientific-papers-records 10000 --shard-rows 1000
```

If Colab disconnects, rerun the same command with the same output directory and limits. After both sources reach their requested limits, merge shards:

```bash
!python -m data.prepare_medical_corpora --output-dir /content/drive/MyDrive/LLM/medical_corpus --medrag-records 20000 --scientific-papers-records 10000 --shard-rows 1000 --merge
```

The merged corpus is `/content/drive/MyDrive/LLM/medical_corpus/medical_pretrain.parquet`. Inspect rows and source balance before training. Never commit downloaded datasets or checkpoints to Git.

## Model/training decisions carried forward

- Encoder-decoder Transformer with cross-attention; old decoder-only checkpoints are not compatible.
- Context length: 512 tokens.
- Initial depth: 2 encoder layers, 2 decoder layers; 4 attention heads.
- Batch size 1 and gradient accumulation 8 as conservative starting values; lower context/batch if CUDA OOM occurs.
- Width is selected using the measured tokenizer token budget, with the approximate 20 tokens/parameter heuristic treated only as a reference, not a guarantee.
- Span-denoising pretraining uses document text. Medical QA instruction datasets are reserved for a separate SFT stage; held-out evaluation data must not enter training.
- Save all shards, checkpoints, logs, and tokenizer to Drive. Free Colab runtimes are ephemeral and can disconnect.

## Before a long run

1. Verify the source schemas by printing a sample from each source.
2. Check the licenses and provenance of both corpora.
3. Run the tokenizer-based token audit and a 2-batch smoke test.
4. Confirm the train/validation split is deterministic and has no near-duplicate leakage.
5. Monitor validation denoising loss, generated reconstructions, CUDA memory, and tokens processed.
6. Only increase record limits after the pilot runs end-to-end.

The data-preparation script downloads documents; it does not train the model and does not produce trained weights.
