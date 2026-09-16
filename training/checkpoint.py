from pathlib import Path

import torch


def save_checkpoint(model, optimizer, step, path, epoch=0, batch_index=0):
    """Save model, optimizer, and training progress for reliable resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "batch_index": batch_index,
        },
        path,
    )


def load_checkpoint(model, optimizer, path, map_location=None):
    """Restore a checkpoint and return its optimizer step."""
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["step"]
