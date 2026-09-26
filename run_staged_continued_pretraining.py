"""Run the exact staged workflow: build one corpus shard, train it, repeat.

For every shard the runner performs:
1. obtain one local seed or one Hugging Face Parquet file;
2. build its processed train/validation corpus on disk;
3. train from the previous checkpoint;
4. save a shard checkpoint;
5. delete temporary remote data and processed corpus after successful training;
6. continue with the next shard.

The persistent dedup database is shared across all shards.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from data.build_continued_shard import main as build_shard
from data.continued_pretraining_stream import download_hf_shard, hf_parquet_files
from train_prepared_continued_pretraining import main as train_shard

DEFAULT_CHECKPOINT_DIR = Path("checkpoints/continued_pretraining")
DEFAULT_WORK_DIR = Path("checkpoints/continued_pretraining/work")


def parse_args(args=None):
    p = argparse.ArgumentParser(description="Build then train each continued-pretraining shard.")
    p.add_argument("--local-shard", action="append", type=Path)
    p.add_argument("--hf-repo-id")
    p.add_argument("--hf-repo-type", choices=("dataset", "space", "model"), default="dataset")
    p.add_argument("--hf-revision", default=None)
    p.add_argument("--hf-pattern", default="*.parquet")
    p.add_argument("--start-shard", type=int, default=0)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    p.add_argument("--pretrained-checkpoint", type=Path, default=Path("checkpoints/phase7_run/best_model.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    p.add_argument("--parquet-batch-size", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--checkpoint-every-steps", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--reset-run", action="store_true")
    return p.parse_args(args)


def _discover(options):
    sources = []
    for path in options.local_shard or []:
        sources.append(("local", path.as_posix(), path))

    if options.hf_repo_id:
        names = hf_parquet_files(
            options.hf_repo_id,
            repo_type=options.hf_repo_type,
            revision=options.hf_revision,
            pattern=options.hf_pattern,
        )
        names = names[options.start_shard:]
        if options.max_shards is not None:
            names = names[: options.max_shards]
        sources.extend(("hf", name, name) for name in names)

    if not sources:
        raise ValueError("Provide at least one --local-shard or --hf-repo-id.")
    return sources


def _completed(checkpoint: Path):
    if not checkpoint.exists():
        return 0
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(payload.get("completed_shards", 0))


def main(args=None):
    options = parse_args(args)
    if options.start_shard < 0:
        raise ValueError("start-shard must be non-negative")
    if options.parquet_batch_size <= 0:
        raise ValueError("parquet-batch-size must be positive")

    sources = _discover(options)
    options.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    options.work_dir.mkdir(parents=True, exist_ok=True)
    dedup_db = options.checkpoint_dir / "dedup.sqlite3"
    checkpoint = options.checkpoint_dir / "latest.pt"
    completed = 0 if options.reset_run else _completed(checkpoint)

    print("=" * 72)
    print("STAGED CONTINUED PRETRAINING")
    print("=" * 72)
    print(f"Total shards: {len(sources)}")
    print(f"Already completed: {completed}")

    for index, (kind, name, payload) in enumerate(sources):
        if index < completed:
            print(f"Skipping completed shard {index}: {name}")
            continue

        source_path = None
        prepared_dir = options.work_dir / f"shard_{index:05d}"
        success = False
        try:
            if kind == "hf":
                download_dir = options.work_dir / f"download_{index:05d}"
                source_path = download_hf_shard(
                    options.hf_repo_id,
                    str(payload),
                    download_dir,
                    repo_type=options.hf_repo_type,
                    revision=options.hf_revision,
                )
                print(f"Downloaded: {name}")
            else:
                source_path = Path(payload)

            if prepared_dir.exists():
                print(f"Prepared corpus already exists; reusing: {prepared_dir}")
            else:
                print(f"Building corpus for shard {index}: {name}")
                build_shard(
                    [
                        "--input", str(source_path),
                        "--output-dir", str(prepared_dir),
                        "--dedup-db", str(dedup_db),
                        "--parquet-batch-size", str(options.parquet_batch_size),
                    ]
                )

            train_args = [
                "--prepared-dir", str(prepared_dir),
                "--pretrained-checkpoint", str(options.pretrained_checkpoint),
                "--tokenizer", str(options.tokenizer),
                "--checkpoint-dir", str(options.checkpoint_dir),
                "--shard-index", str(index),
                "--shard-name", name,
                "--batch-size", str(options.batch_size),
                "--gradient-accumulation-steps", str(options.gradient_accumulation_steps),
                "--learning-rate", str(options.learning_rate),
                "--weight-decay", str(options.weight_decay),
                "--checkpoint-every-steps", str(options.checkpoint_every_steps),
                "--log-every", str(options.log_every),
                "--max-grad-norm", str(options.max_grad_norm),
                "--device", options.device,
                "--num-workers", str(options.num_workers),
            ]
            if options.reset_run and index == 0:
                train_args.append("--reset-run")

            print(f"Training shard {index}: {name}")
            train_shard(train_args)
            success = True
        finally:
            if success and prepared_dir.exists():
                shutil.rmtree(prepared_dir, ignore_errors=True)
            if source_path is not None and kind == "hf":
                download_dir = source_path.parent
                shutil.rmtree(download_dir, ignore_errors=True)

        completed = index + 1
        print(f"Shard {index} finished. Continuing to shard {completed}.")

    print("All requested shards completed.")


if __name__ == "__main__":
    main()
