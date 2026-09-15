from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from data.prepare_training_data import create_dataset
from evaluation.evaluate import evaluate
from model.llm import LLM
from training.checkpoint import save_checkpoint
from training.optimizer import create_optimizer
from training.trainer import train_step


BATCH_SIZE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EPOCHS = 10
LOG_EVERY = 50
MAX_TRAIN_BATCHES = None
CHECKPOINT_DIR = Path("checkpoints")


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()

    print("Creating datasets...")
    train_dataset = create_dataset("train")
    validation_dataset = create_dataset("validation")
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation splits must contain complete sequences")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(validation_dataset):,}")
    print(f"Device: {device}")

    model = LLM(config).to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_parameters:,}")

    optimizer = create_optimizer(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch_index, (input_ids, target_ids) in enumerate(train_loader):
            if MAX_TRAIN_BATCHES is not None and batch_index >= MAX_TRAIN_BATCHES:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss = train_step(model, optimizer, input_ids, target_ids)
            global_step += 1

            if global_step == 1 or global_step % LOG_EVERY == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} | Step {global_step} | Loss {loss:.4f}")

        metrics = evaluate(model, validation_loader, device=device)
        print(f"Validation loss: {metrics['loss']:.4f} | Perplexity: {metrics['perplexity']:.2f}")

        checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch + 1}.pt"
        save_checkpoint(model, optimizer, global_step, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
