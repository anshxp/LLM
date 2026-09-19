import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from data.prepare_training_data import load_token_ids
from data.dataset import LanguageModelDataset
from evaluation.evaluate import evaluate
from model.llm import LLM


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LLM checkpoint on a validation or held-out test split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("base", "healthcare"), default="base")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.eval_stride <= 0:
        raise ValueError("eval-stride must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = resolve_device(args.device)
    config = ModelConfig()

    token_ids = load_token_ids(args.split, dataset=args.dataset)
    dataset = LanguageModelDataset(
        token_ids=token_ids,
        context_length=config.context_length,
        stride=args.eval_stride,
    )
    if len(dataset) == 0:
        raise ValueError("Evaluation split does not contain complete sequences")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = LLM(config).to(device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, loader, device=device)

    epoch = checkpoint.get("epoch", "unknown")
    step = checkpoint.get("step", "unknown")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}/{args.split}")
    print(f"Checkpoint epoch: {epoch} | optimizer step: {step}")
    print(f"Sequences: {len(dataset):,} (stride={args.eval_stride})")
    print(f"Device: {device}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")


if __name__ == "__main__":
    main()
