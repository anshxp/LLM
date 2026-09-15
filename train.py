import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from data.prepare_training_data import create_dataset
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


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train the healthcare-focused language model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--max-train-batches", type=int, default=DEFAULT_MAX_TRAIN_BATCHES)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(args)


def resolve_device(requested):
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


def main(args=None):
    args = parse_args(args)
    validate_args(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = ModelConfig()

    print("Creating datasets...")
    train_dataset = create_dataset("train")
    validation_dataset = create_dataset("validation")
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation splits must contain complete sequences")

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(validation_dataset):,}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size} | Gradient accumulation: {args.gradient_accumulation_steps}")

    model = LLM(config).to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_parameters:,}")

    optimizer = create_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    if args.resume is not None:
        global_step = load_checkpoint(model, optimizer, args.resume, map_location=device)
        print(f"Resumed from {args.resume} at optimizer step {global_step}")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        accumulation_count = 0

        for batch_index, (input_ids, target_ids) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = language_model_loss(logits, target_ids)
            (loss / args.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            accumulation_count += 1

            is_update = accumulation_count == args.gradient_accumulation_steps
            is_last_batch = batch_index + 1 == len(train_loader)
            if is_update or is_last_batch:
                if accumulation_count < args.gradient_accumulation_steps:
                    scale = args.gradient_accumulation_steps / accumulation_count
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accumulation_count = 0

                if global_step == 1 or global_step % args.log_every == 0:
                    average_loss = running_loss / args.gradient_accumulation_steps
                    print(f"Epoch {epoch + 1}/{args.epochs} | Step {global_step} | Loss {average_loss:.4f}")
                    running_loss = 0.0

        metrics = evaluate(model, validation_loader, device=device)
        print(f"Validation loss: {metrics['loss']:.4f} | Perplexity: {metrics['perplexity']:.2f}")

        checkpoint_path = args.checkpoint_dir / f"model_epoch_{epoch + 1}.pt"
        save_checkpoint(model, optimizer, global_step, checkpoint_path, epoch=epoch + 1)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
