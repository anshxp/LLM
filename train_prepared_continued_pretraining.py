"""Train exactly one already-built continued-pretraining shard.

This entrypoint deliberately separates corpus construction from model training.
The input directory must contain train.jsonl and validation.jsonl produced by
:data:`data.build_continued_shard`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from data.continued_pretraining_stream import StreamingTokenDataset, iter_prepared_texts
from data.tokenizer import Tokenizer
from evaluation.evaluate import evaluate
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint
from training.loss import language_model_loss
from training.optimizer import create_optimizer

DEFAULT_FOUNDATION = Path("checkpoints/phase7_run/best_model.pt")
DEFAULT_TOKENIZER = Path("data/processed/tokenizer.json")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/continued_pretraining")


def parse_args(args=None):
    p = argparse.ArgumentParser(description="Train one prepared continued-pretraining shard.")
    p.add_argument("--prepared-dir", required=True, type=Path)
    p.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_FOUNDATION)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-name", required=True)
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


def _device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but CUDA is not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(path: Path, tokenizer: Tokenizer, context_length: int, batch_size: int, workers: int):
    dataset = StreamingTokenDataset(
        lambda _split: iter_prepared_texts(path),
        tokenizer,
        context_length=context_length,
        split="train",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _load_model(model, optimizer, checkpoint_path, foundation_path, device, reset):
    if checkpoint_path.exists() and not reset:
        return load_checkpoint(
            model, optimizer, checkpoint_path, map_location=device, restore_rng=True
        )
    if not foundation_path.exists():
        raise FileNotFoundError(f"Foundation checkpoint not found: {foundation_path}")
    payload = torch.load(foundation_path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload.get("model"))
    if state_dict is None:
        raise ValueError("Foundation checkpoint does not contain model weights.")
    model.load_state_dict(state_dict)
    return None


def main(args=None):
    options = parse_args(args)
    prepared = options.prepared_dir
    train_path = prepared / "train.jsonl"
    validation_path = prepared / "validation.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            f"Prepared shard must contain {train_path.name} and {validation_path.name}: {prepared}"
        )
    if options.shard_index < 0:
        raise ValueError("shard-index must be non-negative")

    device = _device(options.device)
    config = ModelConfig()
    tokenizer = Tokenizer.from_file(options.tokenizer)
    if len(tokenizer) != config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match model vocabulary.")

    options.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = options.checkpoint_dir / "latest.pt"
    model = LLM(config).to(device)
    optimizer = create_optimizer(
        model, learning_rate=options.learning_rate, weight_decay=options.weight_decay
    )
    state = _load_model(
        model,
        optimizer,
        checkpoint_path,
        options.pretrained_checkpoint,
        device,
        options.reset_run,
    )

    saved_shard = state.get("shard_index", -1) if state is not None else -1
    saved_name = state.get("shard_name") if state is not None else None
    completed = state.get("completed_shards", 0) if state is not None else 0
    if state is not None and completed > options.shard_index:
        print(f"Shard {options.shard_index} is already complete; nothing to do.")
        return
    if state is not None and saved_shard == options.shard_index and saved_name not in (None, options.shard_name):
        raise RuntimeError(
            f"Checkpoint belongs to shard '{saved_name}', not '{options.shard_name}'."
        )

    global_step = int(state) if state is not None else 0
    resume_batch = state.get("batch_index", 0) if state is not None and saved_shard == options.shard_index else 0

    train_loader = _loader(
        train_path, tokenizer, config.context_length, options.batch_size, options.num_workers
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    running_loss = 0.0

    for batch_index, (input_ids, target_ids) in enumerate(train_loader):
        if batch_index < resume_batch:
            continue
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        logits = model(input_ids)
        loss = language_model_loss(logits, target_ids)
        (loss / options.gradient_accumulation_steps).backward()
        running_loss += loss.item()
        accumulation += 1

        if accumulation < options.gradient_accumulation_steps:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), options.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        accumulation = 0

        if global_step == 1 or global_step % options.log_every == 0:
            print(
                f"Shard {options.shard_index} | step={global_step} | "
                f"batch={batch_index + 1} | loss={running_loss / options.gradient_accumulation_steps:.4f}"
            )
            running_loss = 0.0

        if global_step % options.checkpoint_every_steps == 0:
            save_checkpoint(
                model,
                optimizer,
                global_step,
                checkpoint_path,
                batch_index=batch_index + 1,
                extra_state={
                    "shard_index": options.shard_index,
                    "shard_name": options.shard_name,
                    "completed_shards": options.shard_index,
                },
            )

    if accumulation:
        scale = options.gradient_accumulation_steps / accumulation
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    validation_loader = _loader(
        validation_path, tokenizer, config.context_length, options.batch_size, options.num_workers
    )
    metrics = evaluate(model, validation_loader, device=device)
    print(
        f"Shard validation | loss={metrics['loss']:.4f} "
        f"perplexity={metrics['perplexity']:.2f}"
    )

    shard_checkpoint = options.checkpoint_dir / f"model_shard_{options.shard_index:05d}.pt"
    extra = {
        "shard_index": options.shard_index,
        "shard_name": options.shard_name,
        "completed_shards": options.shard_index + 1,
        "validation_loss": metrics["loss"],
        "validation_perplexity": metrics["perplexity"],
    }
    save_checkpoint(model, optimizer, global_step, shard_checkpoint, extra_state=extra)
    save_checkpoint(model, optimizer, global_step, checkpoint_path, extra_state=extra)
    print(f"Completed shard {options.shard_index}: {options.shard_name}")


if __name__ == "__main__":
    main()
