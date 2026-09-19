from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    step,
    path,
    epoch=0,
    batch_index=0,
    scheduler=None,
    best_validation_loss=None,
    epochs_without_improvement=0,
):
    """Save model and complete training state for reliable resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "batch_index": batch_index,
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    model,
    optimizer,
    path,
    map_location=None,
    scheduler=None,
):
    """Restore a checkpoint and return its stored training state."""
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "step": checkpoint["step"],
        "epoch": checkpoint.get("epoch", 0),
        "batch_index": checkpoint.get("batch_index", 0),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "epochs_without_improvement": checkpoint.get(
            "epochs_without_improvement", 0
        ),
    }
