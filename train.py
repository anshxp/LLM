import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from data.dataset import LanguageModelDataset
from data.prepare_training_data import load_token_ids
from evaluation.evaluate import evaluate
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint
from training.loss import language_model_loss
from training.optimizer import create_optimizer


DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EPOCHS = 10
DEFAULT_LOG_EVERY = 10
DEFAULT_MAX_TRAIN_BATCHES = None
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_TRAIN_STRIDE = 128
DEFAULT_EVAL_STRIDE = 256
DEFAULT_LR_MIN = 3e-5
DEFAULT_EARLY_STOPPING_PATIENCE = 2


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train the healthcare-focused language model."
    )
    parser.add_argument("--dataset", choices=("base", "healthcare"), default="base")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Total target epochs, including epochs already completed by a resumed checkpoint.",
    )
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument(
        "--max-train-batches", type=int, default=DEFAULT_MAX_TRAIN_BATCHES
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-stride", type=int, default=DEFAULT_TRAIN_STRIDE)
    parser.add_argument("--eval-stride", type=int, default=DEFAULT_EVAL_STRIDE)
    parser.add_argument("--lr-min", type=float, default=DEFAULT_LR_MIN)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
    )
    return parser.parse_args(args)


def resolve_device(requested):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device choice: {requested}")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    if args.log_every <= 0:
        raise ValueError("log_every must be positive")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive when provided")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive when provided")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.train_stride <= 0 or args.eval_stride <= 0:
        raise ValueError("strides must be positive")
    if args.lr_min <= 0 or args.lr_min > args.learning_rate:
        raise ValueError("lr_min must be positive and no greater than learning_rate")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")


def build_dataset(split, dataset_name, context_length, stride):
    token_ids = load_token_ids(split, dataset=dataset_name)
    return LanguageModelDataset(
        token_ids=token_ids,
        context_length=context_length,
        stride=stride,
    )


def main(args=None):
    args = parse_args(args)
    validate_args(args)

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = ModelConfig()

    print(f"Dataset: {args.dataset}")
    print("Creating datasets...")

    train_dataset = build_dataset(
        "train",
        args.dataset,
        config.context_length,
        args.train_stride,
    )
    validation_dataset = build_dataset(
        "validation",
        args.dataset,
        config.context_length,
        args.eval_stride,
    )

    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError(
            "Training and validation splits must contain complete sequences"
        )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs
    )

    print(
        f"Training sequences: {len(train_dataset):,} "
        f"(stride={args.train_stride})"
    )
    print(
        f"Validation sequences: {len(validation_dataset):,} "
        f"(stride={args.eval_stride})"
    )
    print(f"Device: {device}")
    print(
        f"Batch size: {args.batch_size} | "
        f"Gradient accumulation: {args.gradient_accumulation_steps}"
    )

    model = LLM(config).to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_parameters:,}")

    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr_min,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    epochs_without_improvement = 0

    if args.resume is not None:
        resume_state = load_checkpoint(
            model,
            optimizer,
            args.resume,
            map_location=device,
            scheduler=scheduler,
        )
        global_step = resume_state["step"]
        start_epoch = resume_state["epoch"]

        restored_best = resume_state.get("best_validation_loss")
        if restored_best is not None:
            best_validation_loss = float(restored_best)
        epochs_without_improvement = int(
            resume_state.get("epochs_without_improvement", 0)
        )

        if start_epoch >= args.epochs:
            print(
                f"Checkpoint already completed {start_epoch} epoch(s); "
                f"target is {args.epochs}. Nothing to train."
            )
            return

        print(
            f"Resumed from {args.resume} at optimizer step {global_step} "
            f"(completed epoch {start_epoch})"
        )

    best_path = args.checkpoint_dir / "best_model.pt"

    for display_epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        accumulation_count = 0

        for batch_index, (input_ids, target_ids) in enumerate(train_loader):
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = language_model_loss(logits, target_ids)

            (loss / args.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            accumulation_count += 1

            is_update = accumulation_count == args.gradient_accumulation_steps
            reached_limit = (
                args.max_train_batches is not None
                and batch_index + 1 >= args.max_train_batches
            )
            is_last_batch = batch_index + 1 == len(train_loader) or reached_limit

            if is_update or is_last_batch:
                current_accumulation = accumulation_count

                if current_accumulation < args.gradient_accumulation_steps:
                    scale = args.gradient_accumulation_steps / current_accumulation
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                accumulation_count = 0

                if global_step == 1 or global_step % args.log_every == 0:
                    average_loss = running_loss / current_accumulation
                    print(
                        f"Epoch {display_epoch}/{args.epochs} | "
                        f"Step {global_step} | Loss {average_loss:.4f} | "
                        f"LR {optimizer.param_groups[0]['lr']:.2e}"
                    )
                    running_loss = 0.0

        metrics = evaluate(model, validation_loader, device=device)
        validation_loss = metrics["loss"]

        print(
            f"Validation loss: {validation_loss:.4f} | "
            f"Perplexity: {metrics['perplexity']:.2f}"
        )

        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Advance the scheduler before writing the checkpoint so the saved
        # scheduler state is ready for the next epoch.
        scheduler.step()

        checkpoint_path = args.checkpoint_dir / f"model_epoch_{display_epoch}.pt"
        save_checkpoint(
            model,
            optimizer,
            global_step,
            checkpoint_path,
            epoch=display_epoch,
            scheduler=scheduler,
            best_validation_loss=best_validation_loss,
            epochs_without_improvement=epochs_without_improvement,
        )
        print(f"Checkpoint saved: {checkpoint_path}")

        if improved:
            save_checkpoint(
                model,
                optimizer,
                global_step,
                best_path,
                epoch=display_epoch,
                scheduler=scheduler,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
            )
            print(
                f"New best model: {best_path} "
                f"(validation loss={best_validation_loss:.4f})"
            )
        else:
            print(
                f"No validation improvement for "
                f"{epochs_without_improvement}/"
                f"{args.early_stopping_patience} epoch(s)"
            )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best validation loss: {best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
