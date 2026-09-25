"""Continued pretraining from the Phase 7 foundation checkpoint.

This trainer is intentionally separate from instruction SFT. It uses the same
next-token objective as base pretraining, reuses the existing 10k BPE tokenizer,
and automatically resumes from checkpoints/continued_pretraining/latest.pt.
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from data.dataset import LanguageModelDataset
from data.prepare_training_data import load_token_ids_from_file
from evaluation.evaluate import evaluate
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint
from training.loss import language_model_loss
from training.optimizer import create_optimizer

DEFAULT_PRETRAIN_CHECKPOINT = Path("checkpoints/phase7_run/best_model.pt")
DEFAULT_CORPUS_DIR = Path("data/processed/continued_pretraining")
DEFAULT_OLD_VALIDATION = Path("data/processed/validation.txt")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/continued_pretraining")
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION = 4
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EPOCHS = 1
DEFAULT_LOG_EVERY = 50
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_TRAIN_STRIDE = 128
DEFAULT_EVAL_STRIDE = 256
DEFAULT_LR_MIN = 3e-5


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Continue pretraining from the Phase 7 foundation checkpoint."
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAIN_CHECKPOINT)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD_VALIDATION)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--checkpoint-every-steps", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--train-stride", type=int, default=DEFAULT_TRAIN_STRIDE)
    parser.add_argument("--eval-stride", type=int, default=DEFAULT_EVAL_STRIDE)
    parser.add_argument("--lr-min", type=float, default=DEFAULT_LR_MIN)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Ignore latest.pt and start from the Phase 7 pretrained weights.",
    )
    parsed = parser.parse_args(args)
    validate_args(parsed)
    return parsed


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.log_every <= 0 or args.checkpoint_every_steps <= 0:
        raise ValueError("log_every and checkpoint_every_steps must be positive")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if args.train_stride <= 0 or args.eval_stride <= 0:
        raise ValueError("strides must be positive")
    if args.lr_min <= 0 or args.lr_min > args.learning_rate:
        raise ValueError("lr_min must be positive and no greater than learning_rate")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_dataset_from_file(path: Path, context_length: int, stride: int):
    token_ids = load_token_ids_from_file(path)
    return LanguageModelDataset(token_ids, context_length=context_length, stride=stride)


def _load_dataset(corpus_dir: Path, split: str, context_length: int, stride: int):
    return _load_dataset_from_file(corpus_dir / f"{split}.txt", context_length, stride)


def _load_pretrained_weights(model, path: Path, device):
    if not path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state_dict is None:
        raise ValueError("Pretrained checkpoint does not contain model weights")
    model.load_state_dict(state_dict)
    print(f"Loaded foundation weights from {path}")


def main(args=None):
    args = parse_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    config = ModelConfig()

    corpus_dir = args.corpus_dir
    train_dataset = _load_dataset(corpus_dir, "train", config.context_length, args.train_stride)
    validation_dataset = _load_dataset(corpus_dir, "validation", config.context_length, args.eval_stride)
    old_validation_dataset = _load_dataset_from_file(
        args.old_validation, config.context_length, args.eval_stride
    )
    if len(train_dataset) == 0 or len(validation_dataset) == 0 or len(old_validation_dataset) == 0:
        raise ValueError("All training and validation splits must contain complete sequences")

    # Sequential ordering is deliberate. It makes batch_index in the checkpoint
    # an exact resume cursor instead of depending on DataLoader shuffle state.
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, **loader_kwargs)
    old_validation_loader = DataLoader(old_validation_dataset, **loader_kwargs)

    model = LLM(config).to(device)
    optimizer = create_optimizer(
        model, learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, updates_per_epoch * args.epochs),
        eta_min=args.lr_min,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.checkpoint_dir / "latest.pt"
    best_path = args.checkpoint_dir / "best_model.pt"

    current_epoch = 1
    next_batch = 0
    global_step = 0
    best_validation_loss = math.inf

    if latest_path.exists() and not args.no_auto_resume:
        state = load_checkpoint(
            model,
            optimizer,
            latest_path,
            map_location=device,
            scheduler=scheduler,
            restore_rng=True,
        )
        global_step = int(state)
        current_epoch = int(state.get("epoch", 1))
        next_batch = int(state.get("batch_index", 0))
        if state.get("best_validation_loss") is not None:
            best_validation_loss = float(state["best_validation_loss"])
        print(
            f"Auto-resumed from {latest_path}: epoch={current_epoch}, "
            f"next_batch={next_batch}, step={global_step}"
        )
    else:
        _load_pretrained_weights(model, args.pretrained_checkpoint, device)
        print("Starting a new continued-pretraining run from the Phase 7 weights.")

    if current_epoch > args.epochs:
        print(f"Checkpoint already reached epoch {current_epoch}; target is {args.epochs}.")
        return

    print(
        f"Dataset: continued_pretraining | train={len(train_dataset):,} | "
        f"new_validation={len(validation_dataset):,} | "
        f"old_validation={len(old_validation_dataset):,}"
    )
    print(f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    for epoch in range(current_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        accumulation_count = 0
        resume_batch = next_batch if epoch == current_epoch else 0

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

            is_update = accumulation_count == args.gradient_accumulation_steps
            is_last_batch = batch_index + 1 == len(train_loader)
            if is_update or is_last_batch:
                current_accumulation = accumulation_count
                if current_accumulation < args.gradient_accumulation_steps:
                    scale = args.gradient_accumulation_steps / current_accumulation
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accumulation_count = 0

                if global_step == 1 or global_step % args.log_every == 0:
                    average_loss = running_loss / current_accumulation
                    print(
                        f"Epoch {epoch}/{args.epochs} | Step {global_step} | "
                        f"Batch {batch_index + 1}/{len(train_loader)} | "
                        f"Loss {average_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}"
                    )
                    running_loss = 0.0

                if global_step % args.checkpoint_every_steps == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        global_step,
                        latest_path,
                        epoch=epoch,
                        batch_index=batch_index + 1,
                        scheduler=scheduler,
                        best_validation_loss=best_validation_loss,
                    )
                    print(f"Progress checkpoint saved: {latest_path}")

        metrics = evaluate(model, validation_loader, device=device)
        old_metrics = evaluate(model, old_validation_loader, device=device)
        validation_loss = metrics["loss"]
        print(
            f"New validation loss: {validation_loss:.4f} | "
            f"Perplexity: {metrics['perplexity']:.2f}"
        )
        print(
            f"Original validation loss: {old_metrics['loss']:.4f} | "
            f"Perplexity: {old_metrics['perplexity']:.2f}"
        )
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            save_checkpoint(
                model,
                optimizer,
                global_step,
                best_path,
                epoch=epoch,
                batch_index=len(train_loader),
                scheduler=scheduler,
                best_validation_loss=best_validation_loss,
            )
            print(f"New best model saved: {best_path}")

        scheduler.step()
        epoch_path = args.checkpoint_dir / f"model_epoch_{epoch}.pt"
        save_checkpoint(
            model,
            optimizer,
            global_step,
            epoch_path,
            epoch=epoch,
            batch_index=len(train_loader),
            scheduler=scheduler,
            best_validation_loss=best_validation_loss,
        )
        save_checkpoint(
            model,
            optimizer,
            global_step,
            latest_path,
            epoch=epoch + 1,
            batch_index=0,
            scheduler=scheduler,
            best_validation_loss=best_validation_loss,
        )
        print(f"Epoch checkpoint saved: {epoch_path}")

        next_batch = 0

    print(f"Continued pretraining complete. Best validation loss: {best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
