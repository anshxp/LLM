"""Training entry point for base, healthcare, Stage 1 continued pretraining, and instruction SFT."""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from config.model_config import ModelConfig
from data.instruction_dataset import (
    DEFAULT_SFT_CATEGORIES,
    InstructionDataset,
    collate_instruction_batch,
    load_jsonl,
)
from data.instruction_v2_dataset import ShardedInstructionDataset
from data.prepare_training_data import STAGE1_PARQUET, create_dataset
from evaluation.evaluate import evaluate
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint, save_checkpoint_with_history
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
DEFAULT_INSTRUCTION_DIR = Path("data/instruction")
DEFAULT_SFT_LEARNING_RATE = 5e-5
DEFAULT_SFT_LR_MIN = 5e-6
DEFAULT_PRETRAIN_CHECKPOINT = Path("checkpoints/phase7_run/best_model.pt")
DEFAULT_INSTRUCTION_CATEGORIES = ",".join(sorted(DEFAULT_SFT_CATEGORIES))
DEFAULT_STAGE1_PARQUET = STAGE1_PARQUET
DEFAULT_CHECKPOINT_EVERY_STEPS = 100
DEFAULT_CHECKPOINT_HISTORY = 5


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train the RXLM language model.")
    parser.add_argument("--dataset", choices=("base", "healthcare", "stage1", "instruction"), default="base")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--max-train-batches", type=int, default=DEFAULT_MAX_TRAIN_BATCHES)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-every-steps", type=int, default=DEFAULT_CHECKPOINT_EVERY_STEPS)
    parser.add_argument("--checkpoint-history", type=int, default=DEFAULT_CHECKPOINT_HISTORY)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-stride", type=int, default=DEFAULT_TRAIN_STRIDE)
    parser.add_argument("--eval-stride", type=int, default=DEFAULT_EVAL_STRIDE)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--stage1-parquet", type=Path, default=DEFAULT_STAGE1_PARQUET)
    parser.add_argument("--instruction-dir", type=Path, default=DEFAULT_INSTRUCTION_DIR)
    parser.add_argument("--instruction-categories", default=DEFAULT_INSTRUCTION_CATEGORIES)

    parsed = parser.parse_args(args)
    if parsed.learning_rate is None:
        parsed.learning_rate = DEFAULT_SFT_LEARNING_RATE if parsed.dataset == "instruction" else DEFAULT_LEARNING_RATE
    if parsed.lr_min is None:
        parsed.lr_min = DEFAULT_SFT_LR_MIN if parsed.dataset == "instruction" else DEFAULT_LR_MIN
    if parsed.dataset == "instruction" and parsed.pretrained_checkpoint is None:
        parsed.pretrained_checkpoint = DEFAULT_PRETRAIN_CHECKPOINT
    parsed.instruction_categories = tuple(
        category.strip() for category in parsed.instruction_categories.split(",") if category.strip()
    )
    return parsed


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
    if args.checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive")
    if args.checkpoint_history < 1:
        raise ValueError("checkpoint_history must be at least 1")
    if args.dataset == "instruction" and not args.instruction_categories:
        raise ValueError("instruction_categories must not be empty")


def build_dataset(split, dataset_name, context_length, stride, instruction_dir=None, instruction_categories=None, stage1_parquet=None):
    if dataset_name == "instruction":
        instruction_dir = Path(instruction_dir or DEFAULT_INSTRUCTION_DIR)
        if list(instruction_dir.glob(f"{split}-*.jsonl")):
            return ShardedInstructionDataset(instruction_dir, split, context_length, categories=instruction_categories)
        records = load_jsonl(
            instruction_dir / f"{split}.jsonl",
            categories=instruction_categories or DEFAULT_SFT_CATEGORIES,
        )
        return InstructionDataset(records, context_length=context_length)
    return create_dataset(
        split,
        dataset=dataset_name,
        stride=stride,
        stage1_parquet=Path(stage1_parquet or DEFAULT_STAGE1_PARQUET),
    )


def make_loader(dataset, dataset_name, batch_size, shuffle, num_workers):
    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers}
    if isinstance(dataset, IterableDataset):
        kwargs["shuffle"] = False
    if dataset_name == "instruction":
        kwargs["collate_fn"] = collate_instruction_batch
    return DataLoader(dataset, **kwargs)


def _progress_text(dataset, epoch_sequences, epoch_tokens, epoch_start_time):
    if not isinstance(dataset, IterableDataset) or not hasattr(dataset, "progress_snapshot"):
        return ""
    snapshot = dataset.progress_snapshot()
    total_mb = snapshot["total_bytes"] / (1024 * 1024)
    consumed_mb = snapshot["data_consumed_mb"]
    remaining_mb = snapshot["data_remaining_mb"]
    percent = snapshot["progress_fraction"] * 100.0
    elapsed = max(0.001, time.monotonic() - epoch_start_time)
    if consumed_mb > 0 and percent > 0:
        eta_seconds = elapsed * (1.0 / snapshot["progress_fraction"] - 1.0)
        eta_minutes = max(0.0, eta_seconds / 60.0)
        eta = f"{eta_minutes:.1f}m"
    else:
        eta = "calculating"
    return (
        f" | Data {consumed_mb:.2f}/{total_mb:.2f} MB "
        f"({percent:5.1f}%) | Remaining {remaining_mb:.2f} MB "
        f"| Sequences {epoch_sequences:,} | Tokens {epoch_tokens:,} | ETA {eta}"
    )


def main(args=None):
    args = parse_args(args)
    validate_args(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = ModelConfig()

    train_dataset = build_dataset("train", args.dataset, config.context_length, args.train_stride, args.instruction_dir, args.instruction_categories, args.stage1_parquet)
    validation_dataset = build_dataset("validation", args.dataset, config.context_length, args.eval_stride, args.instruction_dir, args.instruction_categories, args.stage1_parquet)
    train_loader = make_loader(train_dataset, args.dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_loader = make_loader(validation_dataset, args.dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    if isinstance(train_dataset, IterableDataset):
        print(f"Dataset: disk-backed streaming corpus | {train_dataset.total_mb:.2f} MB")
    else:
        print(f"Dataset: {args.dataset} | train={len(train_dataset):,} | validation={len(validation_dataset):,}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size} | Gradient accumulation: {args.gradient_accumulation_steps}")

    model = LLM(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = create_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    resume_batch_index = 0
    global_step = 0
    best_validation_loss = math.inf
    epochs_without_improvement = 0

    if args.pretrained_checkpoint is not None and args.resume is not None:
        raise ValueError("Use either --pretrained-checkpoint or --resume, not both")
    if args.pretrained_checkpoint is not None:
        if not args.pretrained_checkpoint.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
        if state_dict is None:
            raise ValueError("Pretrained checkpoint does not contain model_state_dict or model")
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained model weights from {args.pretrained_checkpoint}")

    if args.resume is not None:
        resume_state = load_checkpoint(model, optimizer, args.resume, map_location=device, scheduler=scheduler)
        global_step = resume_state["step"]
        completed_epoch = int(resume_state.get("epoch", 0))
        resume_batch_index = int(resume_state.get("batch_index", 0))
        start_epoch = completed_epoch
        restored_best = resume_state.get("best_validation_loss")
        if restored_best is not None:
            best_validation_loss = float(restored_best)
        epochs_without_improvement = int(resume_state.get("epochs_without_improvement", 0))
        if resume_batch_index == 0 and start_epoch >= args.epochs:
            print(f"Checkpoint already completed {start_epoch} epoch(s); target is {args.epochs}. Nothing to train.")
            return
        resumed_from = resume_state.get("checkpoint_path", str(args.resume))
        fallback_note = " (fallback history checkpoint)" if resume_state.get("used_fallback") else ""
        print(
            f"Resumed from {resumed_from}{fallback_note} at optimizer step {global_step}; "
            f"completed epochs={start_epoch}, next batch in current epoch={resume_batch_index}"
        )

    best_path = args.checkpoint_dir / "best_model.pt"
    latest_path = args.checkpoint_dir / "latest.pt"
    current_epoch = start_epoch + 1
    last_completed_batch = resume_batch_index

    try:
        for display_epoch in range(start_epoch + 1, args.epochs + 1):
            current_epoch = display_epoch
            skip_batches = resume_batch_index if display_epoch == start_epoch + 1 else 0
            resume_batch_index = 0

            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            accumulation_count = 0
            batches_seen = 0
            epoch_sequences = 0
            epoch_tokens = 0
            epoch_start_time = time.monotonic()
            last_completed_batch = skip_batches

            for batch_index, (input_ids, target_ids) in enumerate(train_loader):
                if batch_index < skip_batches:
                    continue
                if args.max_train_batches is not None and (batch_index - skip_batches) >= args.max_train_batches:
                    break

                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                logits = model(input_ids)
                loss = language_model_loss(logits, target_ids)
                (loss / args.gradient_accumulation_steps).backward()
                running_loss += loss.item()
                accumulation_count += 1
                batches_seen += 1
                epoch_sequences += input_ids.shape[0]
                epoch_tokens += input_ids.numel()

                if accumulation_count == args.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    average_loss = running_loss / accumulation_count
                    if global_step == 1 or global_step % args.log_every == 0:
                        progress = _progress_text(train_dataset, epoch_sequences, epoch_tokens, epoch_start_time)
                        print(
                            f"Epoch {display_epoch}/{args.epochs} | Step {global_step} | "
                            f"Loss {average_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}"
                            f"{progress}"
                        )
                    running_loss = 0.0
                    accumulation_count = 0

                    last_completed_batch = batch_index + 1
                    if global_step % args.checkpoint_every_steps == 0:
                        history_path = save_checkpoint_with_history(
                            model,
                            optimizer,
                            global_step,
                            latest_path,
                            epoch=display_epoch - 1,
                            batch_index=last_completed_batch,
                            scheduler=scheduler,
                            best_validation_loss=best_validation_loss,
                            epochs_without_improvement=epochs_without_improvement,
                            keep_last=args.checkpoint_history,
                        )
                        print(
                            f"Latest checkpoint saved: {latest_path} "
                            f"(history={history_path.name}, keep_last={args.checkpoint_history}, "
                            f"epoch={display_epoch}, next_batch={last_completed_batch})"
                        )

            if accumulation_count > 0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(args.gradient_accumulation_steps / accumulation_count)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                last_completed_batch = max(last_completed_batch, skip_batches + batches_seen)
                progress = _progress_text(train_dataset, epoch_sequences, epoch_tokens, epoch_start_time)
                print(
                    f"Epoch {display_epoch}/{args.epochs} | Step {global_step} | "
                    f"Loss {running_loss / accumulation_count:.4f} | "
                    f"LR {optimizer.param_groups[0]['lr']:.2e}{progress}"
                )

            if batches_seen == 0:
                raise ValueError("Training dataset produced no complete sequences")

            metrics = evaluate(model, validation_loader, device=device)
            validation_loss = metrics["loss"]
            print(f"Validation loss: {validation_loss:.4f} | Perplexity: {metrics['perplexity']:.2f}")

            improved = validation_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler.step()
            checkpoint_path = args.checkpoint_dir / f"model_epoch_{display_epoch}.pt"
            save_checkpoint(
                model,
                optimizer,
                global_step,
                checkpoint_path,
                epoch=display_epoch,
                batch_index=0,
                scheduler=scheduler,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
            )
            history_path = save_checkpoint_with_history(
                model,
                optimizer,
                global_step,
                latest_path,
                epoch=display_epoch,
                batch_index=0,
                scheduler=scheduler,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
                keep_last=args.checkpoint_history,
            )
            print(f"Checkpoint saved: {checkpoint_path}")
            print(f"Latest checkpoint updated: {latest_path} (history={history_path.name})")

            if improved:
                save_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    best_path,
                    epoch=display_epoch,
                    batch_index=0,
                    scheduler=scheduler,
                    best_validation_loss=best_validation_loss,
                    epochs_without_improvement=epochs_without_improvement,
                )
                print(f"New best model: {best_path} (validation loss={best_validation_loss:.4f})")
            else:
                print(f"No validation improvement for {epochs_without_improvement}/{args.early_stopping_patience} epoch(s)")

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best validation loss: {best_validation_loss:.4f}")

        if args.dataset == "instruction":
            test_dataset = build_dataset("test", args.dataset, config.context_length, args.eval_stride, args.instruction_dir, args.instruction_categories, args.stage1_parquet)
            test_loader = make_loader(test_dataset, args.dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
            test_metrics = evaluate(model, test_loader, device=device)
            print(f"Instruction test loss: {test_metrics['loss']:.4f} | Perplexity: {test_metrics['perplexity']:.2f}")

    except KeyboardInterrupt:
        save_checkpoint(
            model,
            optimizer,
            global_step,
            latest_path,
            epoch=current_epoch - 1,
            batch_index=last_completed_batch,
            scheduler=scheduler,
            best_validation_loss=best_validation_loss,
            epochs_without_improvement=epochs_without_improvement,
        )
        print(
            f"\nTraining interrupted. Latest checkpoint saved: {latest_path} "
            f"(resume at epoch={current_epoch}, batch={last_completed_batch}, step={global_step})."
        )
        print(f"Resume with: python train.py --resume {latest_path}")


if __name__ == "__main__":
    main()
