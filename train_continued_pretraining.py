"""RAM-efficient, shard-by-shard continued pretraining.

Workflow:
1. Train on the local seed shard(s) first.
2. Discover Parquet files on Hugging Face.
3. Download exactly one remote Parquet shard.
4. Stream rows -> clean/filter/dedup -> tokenize -> context windows.
5. Continue from the previous checkpoint.
6. Save a checkpoint at shard completion and before moving to the next shard.
7. Delete the downloaded shard before downloading the next one.

No 2.5GB Parquet file is loaded into RAM as a whole, and the complete 57GB
corpus is never materialized as one processed text file.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from data.continued_pretraining_stream import (
    ExactDedupStore,
    StreamingTokenDataset,
    download_hf_shard,
    hf_parquet_files,
    iter_local_shards,
    iter_local_texts,
    iter_training_texts,
)
from data.tokenizer import Tokenizer
from evaluation.evaluate import evaluate
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint
from training.loss import language_model_loss
from training.optimizer import create_optimizer

DEFAULT_PRETRAIN_CHECKPOINT = Path("checkpoints/phase7_run/best_model.pt")
DEFAULT_TOKENIZER = Path("data/processed/tokenizer.json")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/continued_pretraining")
DEFAULT_DEDUP_DB = Path("checkpoints/continued_pretraining/dedup.sqlite3")
DEFAULT_DOWNLOAD_DIR = Path("checkpoints/continued_pretraining/downloads")
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION = 4
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_LOG_EVERY = 50
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_VALIDATION_MOD = 20
DEFAULT_PARQUET_BATCH_SIZE = 4096


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Stream local/Hugging Face shards through continued pretraining."
    )
    parser.add_argument(
        "--local-shard",
        action="append",
        type=Path,
        help="Local seed file or directory. Multiple values are processed first, in order.",
    )
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument(
        "--hf-repo-type",
        choices=("dataset", "space", "model"),
        default="dataset",
    )
    parser.add_argument("--hf-revision", type=str, default=None)
    parser.add_argument("--hf-pattern", type=str, default="*.parquet")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAIN_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dedup-db", type=Path, default=DEFAULT_DEDUP_DB)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--checkpoint-every-steps", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--validation-mod", type=int, default=DEFAULT_VALIDATION_MOD)
    parser.add_argument("--parquet-batch-size", type=int, default=DEFAULT_PARQUET_BATCH_SIZE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reset-run",
        action="store_true",
        help="Ignore an existing continued-pretraining checkpoint and start from foundation weights.",
    )
    parsed = parser.parse_args(args)
    validate_args(parsed)
    return parsed


def validate_args(args):
    if not args.local_shard and not args.hf_repo_id:
        raise ValueError("Provide at least one --local-shard or --hf-repo-id.")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch-size and gradient-accumulation-steps must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative.")
    if args.checkpoint_every_steps <= 0 or args.log_every <= 0:
        raise ValueError("checkpoint/log intervals must be positive.")
    if args.max_grad_norm <= 0 or args.validation_mod < 2:
        raise ValueError("max-grad-norm must be positive and validation-mod must be >= 2.")
    if args.parquet_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("parquet-batch-size must be positive and num-workers non-negative.")
    if args.start_shard < 0:
        raise ValueError("start-shard must be non-negative.")
    if args.max_shards is not None and args.max_shards <= 0:
        raise ValueError("max-shards must be positive when supplied.")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but CUDA is not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_shards(args) -> list[tuple[str, str, object]]:
    shards: list[tuple[str, str, object]] = []

    for local in args.local_shard or []:
        for source_name, path in iter_local_shards(local):
            shards.append(("local", source_name, path))

    if args.hf_repo_id:
        files = hf_parquet_files(
            args.hf_repo_id,
            repo_type=args.hf_repo_type,
            revision=args.hf_revision,
            pattern=args.hf_pattern,
        )
        remote_files = files[args.start_shard:]
        if args.max_shards is not None:
            remote_files = remote_files[: args.max_shards]
        for filename in remote_files:
            shards.append(("hf", filename, filename))

    if not shards:
        raise ValueError("No training shards were discovered.")
    return shards


def _source_text_factory(path: Path, args):
    def factory(split: str):
        suffix = Path(args.dedup_db).suffix or ".sqlite3"
        dedup_path = Path(args.dedup_db).with_name(
            f"{Path(args.dedup_db).stem}_{split}{suffix}"
        )
        store = ExactDedupStore(dedup_path)
        try:
            for text in iter_training_texts(
                iter_local_texts(path),
                dedup=store,
                split=split,
                validation_mod=args.validation_mod,
            ):
                yield text
        finally:
            store.commit()
            store.close()

    return factory


def _build_loader(factory, tokenizer, context_length, split, args):
    dataset = StreamingTokenDataset(
        factory,
        tokenizer,
        context_length=context_length,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _load_or_initialize(model, optimizer, checkpoint_path, foundation_path, device, reset):
    if checkpoint_path.exists() and not reset:
        state = load_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            map_location=device,
            restore_rng=True,
        )
        print(
            f"Resumed checkpoint: step={int(state)} "
            f"shard_index={state.get('shard_index', 0)} "
            f"batch_index={state.get('batch_index', 0)}"
        )
        return state

    if not foundation_path.exists():
        raise FileNotFoundError(f"Foundation checkpoint not found: {foundation_path}")

    checkpoint = torch.load(foundation_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state_dict is None:
        raise ValueError("Foundation checkpoint does not contain model weights.")
    model.load_state_dict(state_dict)
    print(f"Loaded foundation weights from {foundation_path}")
    return None


def train_one_shard(
    model,
    optimizer,
    train_loader,
    device,
    args,
    checkpoint_path,
    global_step,
    shard_index,
    shard_name,
    resume_batch,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    accumulation_count = 0
    start = time.time()

    for batch_index, (input_ids, target_ids) in enumerate(train_loader):
        if batch_index < resume_batch:
            continue

        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        logits = model(input_ids)
        loss = language_model_loss(logits, target_ids)
        (loss / args.gradient_accumulation_steps).backward()
        running_loss += loss.item()
        accumulation_count += 1

        if accumulation_count != args.gradient_accumulation_steps:
            continue

        current_accumulation = accumulation_count
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        accumulation_count = 0

        if global_step == 1 or global_step % args.log_every == 0:
            elapsed = max(time.time() - start, 0.001)
            print(
                f"Shard {shard_index} | Step {global_step} | "
                f"Batch {batch_index + 1} | Loss {running_loss / current_accumulation:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e} | "
                f"{(batch_index + 1) / elapsed:.2f} batches/s"
            )
            running_loss = 0.0

        if global_step % args.checkpoint_every_steps == 0:
            save_checkpoint(
                model,
                optimizer,
                global_step,
                checkpoint_path,
                batch_index=batch_index + 1,
                extra_state={"shard_index": shard_index, "shard_name": shard_name},
            )
            print(f"Progress checkpoint saved: {checkpoint_path}")

    if accumulation_count:
        scale = args.gradient_accumulation_steps / accumulation_count
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    return global_step


def main(args=None):
    args = parse_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    config = ModelConfig()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match model vocabulary "
            f"({config.vocab_size})."
        )

    shards = discover_shards(args)
    print("=" * 72)
    print("STREAMING CONTINUED PRETRAINING")
    print("=" * 72)
    print(f"Shards discovered: {len(shards)}")
    for index, (kind, name, _) in enumerate(shards):
        print(f"  [{index:02d}] {kind}: {name}")
    print(f"Device: {device}")
    print("=" * 72)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "latest.pt"

    model = LLM(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    state = _load_or_initialize(
        model,
        optimizer,
        checkpoint_path,
        args.pretrained_checkpoint,
        device,
        args.reset_run,
    )

    completed_shards = state.get("completed_shards", 0) if state is not None else 0
    resume_batch = state.get("batch_index", 0) if state is not None else 0
    global_step = int(state) if state is not None else 0

    if state is None:
        resume_shard = 0
    else:
        saved_name = state.get("shard_name")
        if saved_name:
            matching = [index for index, (_, name, _) in enumerate(shards) if name == saved_name]
            resume_shard = matching[0] + (1 if completed_shards else 0) if matching else 0
        else:
            resume_shard = state.get("shard_index", 0)
        if completed_shards:
            resume_batch = 0

    for shard_index in range(resume_shard, len(shards)):
        kind, name, payload = shards[shard_index]
        print("\n" + "=" * 72)
        print(f"SHARD {shard_index + 1}/{len(shards)}: {name}")
        print("=" * 72)

        downloaded_path: Path | None = None
        try:
            if kind == "hf":
                download_dir = args.download_dir / f"shard_{shard_index:05d}"
                downloaded_path = download_hf_shard(
                    args.hf_repo_id,
                    str(payload),
                    download_dir,
                    repo_type=args.hf_repo_type,
                    revision=args.hf_revision,
                )
                shard_path = downloaded_path
                print(f"Downloaded one shard: {shard_path}")
            else:
                shard_path = Path(payload)

            factory = _source_text_factory(shard_path, args)
            train_loader = _build_loader(
                factory, tokenizer, config.context_length, "train", args
            )
            validation_loader = _build_loader(
                factory, tokenizer, config.context_length, "validation", args
            )

            shard_resume_batch = resume_batch if shard_index == resume_shard else 0
            global_step = train_one_shard(
                model,
                optimizer,
                train_loader,
                device,
                args,
                checkpoint_path,
                global_step,
                shard_index,
                name,
                shard_resume_batch,
            )

            validation_metrics = evaluate(model, validation_loader, device=device)
            print(
                f"Shard validation | loss={validation_metrics['loss']:.4f} "
                f"perplexity={validation_metrics['perplexity']:.2f}"
            )

            epoch_checkpoint = args.checkpoint_dir / f"model_shard_{shard_index:05d}.pt"
            save_checkpoint(
                model,
                optimizer,
                global_step,
                epoch_checkpoint,
                batch_index=0,
                extra_state={
                    "shard_index": shard_index,
                    "completed_shards": shard_index + 1,
                    "shard_name": name,
                    "validation_loss": validation_metrics["loss"],
                    "validation_perplexity": validation_metrics["perplexity"],
                },
            )
            save_checkpoint(
                model,
                optimizer,
                global_step,
                checkpoint_path,
                batch_index=0,
                extra_state={
                    "shard_index": shard_index + 1,
                    "completed_shards": shard_index + 1,
                    "shard_name": name,
                    "validation_loss": validation_metrics["loss"],
                    "validation_perplexity": validation_metrics["perplexity"],
                },
            )
            print(f"Completed shard {shard_index + 1}/{len(shards)}.")
        finally:
            if downloaded_path is not None:
                download_root = args.download_dir / f"shard_{shard_index:05d}"
                if download_root.exists():
                    shutil.rmtree(download_root, ignore_errors=True)
                    print(f"Deleted downloaded shard: {download_root}")

        resume_batch = 0

    print("\nContinued pretraining complete.")
    print(f"Final checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
