# Common Pile PubMed ingestion on Google Colab

This branch uses **only** [Common Pile PubMed](https://huggingface.co/datasets/common-pile/pubmed) as the text source. No chest/topic filter is applied. The script preserves the source text and available document metadata in resumable Parquet shards.

## 1. Mount Drive and clone this branch

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone --branch cleanup/common-pile-pubmed-ingestion https://github.com/anshxp/LLM.git
%cd /content/LLM
!pip install -q datasets pyarrow
```

If the repository is already cloned in the current runtime, use `%cd /content/LLM` and `!git pull origin cleanup/common-pile-pubmed-ingestion` instead.

## 2. Test ingestion first

This reads at most 1,000 source rows and writes test shards to a separate Drive folder:

```bash
!python -u data/prepare_common_pile_pubmed.py \
  --output-dir /content/drive/MyDrive/LLM/common_pile_pubmed_test \
  --shard-rows 100 \
  --max-source-rows 1000 \
  --log-every-rows 100
```

Inspect `manifest.json` and the resulting Parquet files before starting the full run.

## 3. Full dataset ingestion

Use a new output directory for the full corpus:

```bash
!python -u data/prepare_common_pile_pubmed.py \
  --output-dir /content/drive/MyDrive/LLM/common_pile_pubmed \
  --shard-rows 1000 \
  --log-every-rows 1000
```

If Colab disconnects, rerun the exact same command with the same output path. The script resumes from the last committed shard using the manifest. Resuming a streaming Hugging Face dataset may take time because already committed rows must be skipped.

## Output and safety notes

- `part-XXXXXX.parquet`: text plus document ID, source fields, and license/provenance metadata.
- `manifest.json`: committed source-row cursor, shard inventory, row counts, status, and timestamps.
- The script skips only records with missing or empty text; it does not filter by specialty or rewrite text.
- Do not commit downloaded data, manifests containing corpus contents, checkpoints, or tokenizer artifacts to Git.
- The source dataset is large. Confirm Google Drive quota before the full run. The manifest reports actual downloaded/usable rows; estimates from the dataset card are not a substitute for measured output.
- Review each source record's license metadata and dataset terms before redistributing data or trained artifacts.

This is a data-ingestion command only. It does not train a tokenizer or model.
