import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from data.instruction_dataset import InstructionDataset, load_jsonl
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint
from training.optimizer import create_optimizer


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Supervised instruction fine-tuning.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/instruction"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(args)


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            logits = model(input_ids.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.to(device).reshape(-1),
                ignore_index=-100,
            )
            total += loss.item()
            batches += 1
    if not batches:
        raise ValueError("Validation dataset is empty")
    loss = total / batches
    return loss, math.exp(min(loss, 20.0))


def main(args=None):
    args = parse_args(args)
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = ModelConfig()

    train_records = load_jsonl(args.train)
    validation_records = load_jsonl(args.validation)
    train_dataset = InstructionDataset(train_records, config.context_length)
    validation_dataset = InstructionDataset(validation_records, config.context_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    model = LLM(config).to(device)
    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Load model weights and resume metadata, but deliberately use the fine-tuning
    # learning rate instead of inheriting the base-model optimizer LR.
    state = load_checkpoint(model, optimizer, args.resume, map_location=device)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf

    print(f"Instruction examples: train={len(train_dataset)}, validation={len(validation_dataset)}")
    print(f"Device: {device} | Starting from: {args.resume}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for input_ids, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.to(device).reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            running += loss.item()

        val_loss, val_ppl = evaluate(model, validation_loader, device)
        train_loss = running / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} | train loss={train_loss:.4f} | validation loss={val_loss:.4f} | perplexity={val_ppl:.2f}")

        checkpoint = args.output_dir / f"model_epoch_{epoch}.pt"
        save_checkpoint(model, optimizer, state["step"] + epoch, checkpoint, epoch=epoch, best_validation_loss=min(best_loss, val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, state["step"] + epoch, args.output_dir / "best_model.pt", epoch=epoch, best_validation_loss=best_loss)
            print(f"New best model: {args.output_dir / 'best_model.pt'}")

    print(f"Instruction fine-tuning complete. Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
