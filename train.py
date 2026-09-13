from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from config.model_config import ModelConfig
from data.prepare_training_data import create_dataset
from model.llm import LLM
from training.checkpoint import save_checkpoint
from training.optimizer import create_optimizer
from training.trainer import train_step


BATCH_SIZE = 4
LEARNING_RATE = 3e-10
WEIGHT_DECAY = 0.01

EPOCHS = 500
LOG_EVERY = 50

MAX_TRAIN_BATCHES = 20

CHECKPOINT_DIR = Path("checkpoints")


def main():
    config = ModelConfig()

    print("Creating dataset...")
    dataset = create_dataset()

    # 90/10 train-validation split.
    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Total examples: {len(dataset):,}")
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(validation_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("\nCreating model...")

    model = LLM(config)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {total_parameters:,}")

    optimizer = create_optimizer(
        model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    global_step = 0

    for epoch in range(EPOCHS):
        model.train()

        for batch_index, (input_ids, target_ids) in enumerate(train_loader):
            if batch_index >= MAX_TRAIN_BATCHES:
                break

            loss = train_step(
                model,
                optimizer,
                input_ids,
                target_ids,
            )

            global_step += 1

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Step {global_step}/{MAX_TRAIN_BATCHES} | "
                f"Loss {loss:.4f}"
            )

        checkpoint_path = (
            CHECKPOINT_DIR / f"model_epoch_{epoch + 1}.pt"
        )

        save_checkpoint(
            model,
            optimizer,
            global_step,
            checkpoint_path,
        )

        print(
            f"\nCheckpoint saved: {checkpoint_path} \n"
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()