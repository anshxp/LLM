"""Pretrain RXLM encoder-decoder on the configured Parquet text corpus."""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from data.seq2seq_parquet_dataset import ParquetSeq2SeqDataset
from data.tokenizer import Tokenizer
from model.seq2seq_lm import Seq2SeqLanguageModel

DEFAULT_CORPUS = Path("data/raw/stage 1/train.parquet")
DEFAULT_TOKENIZER = Path("data/processed/tokenizer.json")
DEFAULT_OUT = Path("checkpoints/seq2seq_318mb")
DEFAULT_CONTEXT = 512


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Pretrain 512-context encoder-decoder LM")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args(args)


def validate_args(args):
    if args.context_length < 2 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError("context-length and epochs must be positive; context must be >= 2")
    if args.gradient_accumulation_steps < 1 or args.checkpoint_every_steps < 1 or args.log_every < 1:
        raise ValueError("accumulation/checkpoint/log intervals must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        raise ValueError("Invalid learning rate, weight decay, or worker count")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("max-train-batches must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_state(path, model, optimizer, scheduler, epoch, batch_index, step, best_val, config, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "batch_index": batch_index,
        "step": step,
        "best_validation_loss": best_val,
        "model_config": config.__dict__,
        "seed": seed,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def evaluate(model, loader, device, max_batches=None):
    was_training = model.training
    model.eval()
    loss_sum, n, target_tokens = 0.0, 0, 0
    with torch.no_grad():
        for i, (source, decoder_input, target) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            source, decoder_input, target = source.to(device), decoder_input.to(device), target.to(device)
            logits = model(source, decoder_input)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss_sum += loss.item()
            target_tokens += target.numel()
            n += 1
    model.train(was_training)
    if n == 0:
        raise RuntimeError("Validation split produced no pairs; check the corpus and split distribution.")
    mean = loss_sum / n
    return {"loss": mean, "perplexity": math.exp(min(mean, 20.0)), "batches": n, "target_tokens": target_tokens}


def main(args=None):
    args = parse_args(args)
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    if not args.parquet.is_file():
        raise FileNotFoundError(f"Parquet corpus not found: {args.parquet}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    config = ModelConfig(vocab_size=len(tokenizer), context_length=args.context_length)
    model = Seq2SeqLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_set = ParquetSeq2SeqDataset(args.parquet, tokenizer, args.context_length, "train")
    val_set = ParquetSeq2SeqDataset(args.parquet, tokenizer, args.context_length, "validation")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    epoch_start, resume_batch, step = 0, 0, 0
    best_val = float("inf")
    latest = args.checkpoint_dir / "latest.pt"
    best_path = args.checkpoint_dir / "best_model.pt"
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("model_config") != config.__dict__:
            raise ValueError("Resume checkpoint model configuration differs from requested settings.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        epoch_start = int(state.get("epoch", 0))
        resume_batch = int(state.get("batch_index", 0))
        step = int(state.get("step", 0))
        best_val = float(state.get("best_validation_loss", float("inf")))
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    print("=" * 72)
    print("RXLM encoder-decoder pretraining")
    print(f"Corpus: {args.parquet} ({args.parquet.stat().st_size / (1024**2):.2f} MiB)")
    print(f"Tokenizer: {args.tokenizer} | vocab={len(tokenizer):,}")
    print("Architecture: bidirectional encoder + causal decoder + cross-attention")
    print(f"Context: {args.context_length} source tokens + {args.context_length} prediction tokens")
    print(f"Parameters: {model.parameter_count():,} ({model.parameter_count()/1e6:.2f}M)")
    print(f"Device={device}; batch={args.batch_size}; accumulation={args.gradient_accumulation_steps}")
    print("=" * 72)

    for epoch in range(epoch_start + 1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum, epoch_batches, accumulation = 0.0, 0, 0
        skipped = resume_batch if epoch == epoch_start + 1 else 0
        last_batch = skipped
        started = time.monotonic()
        for batch_i, (source, decoder_input, target) in enumerate(train_loader):
            if batch_i < skipped:
                continue
            if args.max_train_batches is not None and batch_i - skipped >= args.max_train_batches:
                break
            source, decoder_input, target = source.to(device), decoder_input.to(device), target.to(device)
            logits = model(source, decoder_input)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            (loss / args.gradient_accumulation_steps).backward()
            epoch_loss_sum += loss.item()
            epoch_batches += 1
            accumulation += 1
            last_batch = batch_i + 1
            if accumulation >= args.gradient_accumulation_steps:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0
                step += 1
            if step > 0 and step % args.log_every == 0 and accumulation == 0:
                snapshot = train_set.progress_snapshot()
                print(f"epoch={epoch}/{args.epochs} step={step} loss={loss.item():.4f} "
                      f"data={snapshot['data_consumed_mb']:.1f}/{train_set.total_mb:.1f}MiB "
                      f"sequences={snapshot['sequences_yielded']:,} elapsed={time.monotonic()-started:.1f}s")
            if step > 0 and step % args.checkpoint_every_steps == 0 and accumulation == 0:
                save_state(latest, model, optimizer, scheduler, epoch - 1, last_batch, step, best_val, config, args.seed)
            if args.max_train_batches is not None and batch_i - skipped + 1 >= args.max_train_batches:
                break

        if accumulation:
            # Correctly normalize the partial final accumulation window.
            scale = args.gradient_accumulation_steps / accumulation
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

        val = evaluate(model, val_loader, device, max_batches=200)
        scheduler.step()
        train_loss = epoch_loss_sum / max(1, epoch_batches)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
              f"val_ppl={val['perplexity']:.2f} val_batches={val['batches']} "
              f"target_tokens={val['target_tokens']:,}")
        save_state(args.checkpoint_dir / f"model_epoch_{epoch}.pt", model, optimizer, scheduler, epoch, 0, step, best_val, config, args.seed)
        save_state(latest, model, optimizer, scheduler, epoch, 0, step, best_val, config, args.seed)
        if val["loss"] < best_val:
            best_val = val["loss"]
            save_state(best_path, model, optimizer, scheduler, epoch, 0, step, best_val, config, args.seed)
            print(f"New best checkpoint: {best_path}")
        if args.max_train_batches is not None:
            print("Smoke-test batch limit reached; this was not a full epoch.")
            break

    print(f"Done. Latest: {latest} | Best: {best_path}")


if __name__ == "__main__":
    main()
