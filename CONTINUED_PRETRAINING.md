# Continued pretraining: streaming shard workflow

The continued-pretraining stage is now designed for the large corpus workflow:

```text
local seed (~318 MB)
        ↓
checkpoint
        ↓
Hugging Face Parquet shard 1
        ↓
checkpoint
        ↓
Hugging Face Parquet shard 2
        ↓
...
        ↓
Hugging Face Parquet shard N
```

The trainer never combines the complete corpus into one in-memory object and
never requires all remote Parquet files to be present locally at the same time.

## What is reused

The Phase 7 foundation checkpoint remains the starting point:

```text
checkpoints/phase7_run/best_model.pt
```

The already-trained tokenizer remains fixed:

```text
data/processed/tokenizer.json
```

The tokenizer is not retrained for continued pretraining. The model vocabulary
must remain the same 10,000-token vocabulary used by the foundation checkpoint.

## Streaming pipeline

`data/continued_pretraining_stream.py` provides the data path:

```text
local file / one downloaded Parquet shard
        ↓
pyarrow batch reader
        ↓
record → text normalization
        ↓
clean_text + basic filters
        ↓
deterministic train/validation routing
        ↓
disk-backed exact deduplication (SQLite)
        ↓
fixed tokenizer
        ↓
context-length token windows
        ↓
PyTorch DataLoader
```

Parquet is read with bounded Arrow batches. Tokenization also remains bounded:
the streaming dataset keeps only the token window needed to produce the next
training example rather than creating a list containing all tokens in a 2.5 GB
file.

Exact duplicate detection is persisted in SQLite so the deduplication index does
not grow as a Python `set` in RAM. Training and validation use separate persistent
indexes.

## Hugging Face files

The trainer uses `huggingface_hub` to list the repository files and download one
Parquet file at a time. The repository type can be `dataset`, `space`, or `model`.
The default is `dataset`.

The default pattern is:

```text
*.parquet
```

Files are sorted deterministically before training. If the Hub repository contains
23 Parquet files, the trainer will therefore process them sequentially in that
order unless `--start-shard` or `--max-shards` is supplied.

## First run: train the local ~318 MB seed

If the current local corpus is one file, for example:

```powershell
python train_continued_pretraining.py `
  --local-shard "data/your_318mb_file.parquet"
```

For a local directory, pass the directory instead. Supported files inside it are
processed in stable path order.

This starts from `checkpoints/phase7_run/best_model.pt` and writes:

```text
checkpoints/continued_pretraining/latest.pt
checkpoints/continued_pretraining/model_shard_00000.pt
```

## Continue directly into the Hugging Face corpus

Use the same local seed argument together with the Hub repository so the shard
sequence is explicit and resume metadata remains stable:

```powershell
python train_continued_pretraining.py `
  --local-shard "data/your_318mb_file.parquet" `
  --hf-repo-id "YOUR_USER/YOUR_REPO" `
  --hf-repo-type dataset
```

For a Hugging Face Space, use:

```powershell
python train_continued_pretraining.py `
  --local-shard "data/your_318mb_file.parquet" `
  --hf-repo-id "YOUR_USER/YOUR_SPACE" `
  --hf-repo-type space
```

The local shard is always processed first. After it completes, the next model
update starts from its checkpoint and continues on the first Parquet shard. The
same pattern repeats for every subsequent Parquet file.

If the Hub repository contains approximately 23 files of approximately 2.5 GB
each, only the current file is downloaded for training. After the shard is
finished and checkpointed, its temporary download directory is deleted before the
next file is downloaded.

## Resume behavior

`train_continued_pretraining.py` automatically resumes from:

```text
checkpoints/continued_pretraining/latest.pt
```

The checkpoint contains model weights, optimizer state, RNG state, global step,
current shard name/index, completed-shard count, and batch position.

The shard name is used to locate the resume point in the currently supplied local
+ Hub shard list. This prevents changing from a local-only first run to a local+
Hub run from accidentally skipping the first remote file.

If an interruption occurs in the middle of a Parquet shard, the trainer resumes
from the last saved batch checkpoint instead of restarting that shard from its
beginning. `shuffle=False` is intentional so the streamed sequence order is
stable.

To intentionally discard the existing continued-pretraining run and reload the
Phase 7 foundation checkpoint, use:

```powershell
python train_continued_pretraining.py `
  --local-shard "data/your_318mb_file.parquet" `
  --hf-repo-id "YOUR_USER/YOUR_REPO" `
  --reset-run
```

## RAM and storage controls

The important controls are:

```text
--parquet-batch-size 4096
--batch-size 1
--gradient-accumulation-steps 4
--num-workers 0
```

`--parquet-batch-size` controls how many Parquet rows PyArrow converts to Python
objects at once. Lower it if individual rows are large. `--batch-size` and
gradient accumulation control GPU training memory independently of the Parquet
file size.

The default download directory is:

```text
checkpoints/continued_pretraining/downloads/
```

Only one remote shard is kept there at a time.

## Checkpoints

A progress checkpoint is written every 500 optimizer steps by default:

```text
checkpoints/continued_pretraining/latest.pt
```

At the end of every shard, an immutable shard checkpoint is also written:

```text
checkpoints/continued_pretraining/model_shard_00000.pt
checkpoints/continued_pretraining/model_shard_00001.pt
...
```

This gives us a recovery point after every corpus file while preserving the
optimizer state needed to continue training rather than merely loading model
weights.

## Validation

Each shard is deterministically divided into training and validation records.
The default validation fraction is 1/20 (5%). Validation is streamed in a second
pass over the same shard after training completes.

The validation split is not added to the training stream. Validation is evaluated
only after the current shard has finished.

## Important distinction from the old corpus builder

`data/build_continued_corpus.py` remains in the repository for the older
materialized-corpus workflow and its existing tests. It is not used by the new
large-shard trainer.

For the 57 GB-style workflow, do not first build a single
`data/processed/continued_pretraining/train.txt`. Use:

```text
train_continued_pretraining.py
        +
 data/continued_pretraining_stream.py
```

This avoids creating a second 57 GB processed corpus and avoids loading a giant
text file into RAM before tokenization.
