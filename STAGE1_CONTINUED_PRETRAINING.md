# Stage 1 continued pretraining

Stage 1 is continued pretraining on the Parquet corpus at `data/raw/stage 1/train.parquet`.

The Stage 0 tokenizer is reused unchanged. The Parquet file is streamed directly with PyArrow, so the full multi-GB file is not loaded into RAM and no new tokenizer is trained.

## Start Stage 1 from Stage 0

```powershell
python train.py --dataset stage1 --stage1-parquet "data/raw/stage 1/train.parquet" --device cuda --pretrained-checkpoint checkpoints\best_model.pt --checkpoint-dir checkpoints\stage1
```

Use `--epochs` or `--max-train-batches` for a smoke test before a full run, for example:

```powershell
python train.py --dataset stage1 --stage1-parquet "data/raw/stage 1/train.parquet" --device cuda --pretrained-checkpoint checkpoints\best_model.pt --checkpoint-dir checkpoints\stage1 --epochs 1 --max-train-batches 5
```

## Resume Stage 1

Stage 1 checkpoints contain model weights, optimizer state, scheduler state, validation history, the completed epoch count, and the next batch position. `latest.pt` is written atomically after each periodic checkpoint and completed epoch. The training code also keeps the 5 newest immutable `checkpoint_step_*.pt` files by default, so an interrupted or corrupted `latest.pt` can fall back to the newest valid history checkpoint. Pressing Ctrl+C also creates a history checkpoint before exiting.

Resume with:

```powershell
python train.py --dataset stage1 --stage1-parquet "data/raw/stage 1/train.parquet" --device cuda --resume checkpoints\stage1\latest.pt --checkpoint-dir checkpoints\stage1
```

A mid-epoch resume skips the already-consumed batches recorded in the checkpoint instead of restarting the epoch from the beginning.

## Checkpoints

Stage 1 writes:

- `checkpoints/stage1/latest.pt` — atomic pointer to the newest checkpoint.
- `checkpoints/stage1/checkpoint_step_N.pt` — the 5 newest immutable resume checkpoints by default.
- `checkpoints/stage1/model_epoch_N.pt` — epoch checkpoints.
- `checkpoints/stage1/best_model.pt` — best validation-loss checkpoint.

The retention count can be changed with `--checkpoint-history N`, but it must be at least 1. The default is 5.

`--pretrained-checkpoint` is for starting a new stage from existing weights. `--resume` is for continuing an interrupted run with its optimizer/scheduler/training position.
