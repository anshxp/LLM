# Continued pretraining workflow

This project now has a separate continued-pretraining path. It does not use the
instruction SFT pipeline and it never scans `data/raw`.

## Data used

The corpus builder reads only:

- `data/processed/train.txt` — the already-processed original foundation corpus;
- `data/MedQuad` — the new MedQuAD data;
- `data/Fine tuning 2` — the new second data collection.

`data/raw` is explicitly rejected even if someone passes it as a source.

The generated corpus is written to:

```text
data/processed/continued_pretraining/
├── train.txt
├── validation.txt
├── test.txt
└── manifest.jsonl
```

The existing tokenizer at `data/processed/tokenizer.json` is reused. Do not
retrain the tokenizer for this run because the Phase 7 model has a fixed
10,000-token vocabulary.

## Build the corpus

From the repository root:

```powershell
python -m data.build_continued_corpus
```

The default source directories are exactly `data/MedQuad` and `data/Fine tuning 2`.
You can override them explicitly if needed:

```powershell
python -m data.build_continued_corpus `
  --source "data/MedQuad" `
  --source "data/Fine tuning 2"
```

The builder combines the existing processed foundation training corpus with the
new sources. The old corpus is assigned to the continued-training split only;
the new sources receive a deterministic 90/5/5 train/validation/test split.

## Start continued pretraining

The foundation checkpoint is:

```text
checkpoints/phase7_run/best_model.pt
```

Start the new run with:

```powershell
python train_continued_pretraining.py
```

The default output is:

```text
checkpoints/continued_pretraining/
├── latest.pt
├── best_model.pt
├── model_epoch_1.pt
└── ...
```

## Resume after interruption

The trainer automatically loads `checkpoints/continued_pretraining/latest.pt`
when it exists. It restores:

- model weights;
- optimizer state;
- scheduler state;
- global optimizer step;
- epoch;
- next batch index;
- validation-loss state;
- CPU/CUDA RNG state.

The continued-pretraining DataLoader deliberately uses deterministic sequential
ordering (`shuffle=False`). This makes the stored batch index an exact resume
cursor instead of depending on a newly generated shuffle order. Checkpoints are
written atomically, so an interrupted write cannot replace a valid checkpoint
with a partial file.

Progress is checkpointed every 500 optimizer steps by default, and again at the
end of every epoch. Therefore a failure can replay at most the work since the
last progress checkpoint rather than returning to the beginning of the run.

If you intentionally want to ignore the previous continued-pretraining run and
start again from the Phase 7 weights, use:

```powershell
python train_continued_pretraining.py --no-auto-resume
```

## Future SFT

Do not use the SFT scripts for this stage. After continued pretraining is
complete, the resulting `best_model.pt` can become the pretrained checkpoint
for a later high-quality instruction/SFT run.
