from pathlib import Path
import os
import re
import tempfile

import torch


_CHECKPOINT_HISTORY_PATTERN = re.compile(r"^checkpoint_step_(\d+)\.pt$")


class CheckpointState(int):
    """Integer-compatible checkpoint state with resume metadata."""

    def __new__(cls, step, **metadata):
        instance = int.__new__(cls, step)
        instance._metadata = {"step": step, **metadata}
        return instance

    def __getitem__(self, key):
        return self._metadata[key]

    def get(self, key, default=None):
        return self._metadata.get(key, default)


def _build_payload(
    model,
    optimizer,
    step,
    epoch=0,
    batch_index=0,
    scheduler=None,
    best_validation_loss=None,
    epochs_without_improvement=0,
):
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
    return payload


def _atomic_torch_save(payload, path):
    """Write a checkpoint completely before replacing the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        torch.save(payload, temp_path)

        with temp_path.open("rb") as temp_file:
            os.fsync(temp_file.fileno())

        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _history_files(directory):
    directory = Path(directory)
    candidates = []
    for path in directory.glob("checkpoint_step_*.pt"):
        match = _CHECKPOINT_HISTORY_PATTERN.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return sorted(candidates, key=lambda item: item[0], reverse=True)


def prune_checkpoint_history(directory, keep_last=5):
    """Keep the newest checkpoint_step_*.pt files and remove older history."""
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")

    history = _history_files(directory)
    for _, path in history[keep_last:]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _update_latest_link(history_path, latest_path):
    """Make latest.pt refer to the fully written immutable checkpoint."""
    history_path = Path(history_path)
    latest_path = Path(latest_path)
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    temp_latest = latest_path.with_name(f".{latest_path.name}.tmp")
    try:
        try:
            temp_latest.unlink()
        except FileNotFoundError:
            pass

        try:
            os.link(history_path, temp_latest)
        except (OSError, NotImplementedError):
            import shutil
            shutil.copyfile(history_path, temp_latest)

        os.replace(temp_latest, latest_path)
    finally:
        try:
            temp_latest.unlink()
        except FileNotFoundError:
            pass


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
    """Save model and complete training state with an atomic file replacement."""
    path = Path(path)
    payload = _build_payload(
        model,
        optimizer,
        step,
        epoch=epoch,
        batch_index=batch_index,
        scheduler=scheduler,
        best_validation_loss=best_validation_loss,
        epochs_without_improvement=epochs_without_improvement,
    )
    _atomic_torch_save(payload, path)


def save_checkpoint_with_history(
    model,
    optimizer,
    step,
    latest_path,
    epoch=0,
    batch_index=0,
    scheduler=None,
    best_validation_loss=None,
    epochs_without_improvement=0,
    keep_last=5,
):
    """Save an immutable recent checkpoint and update latest.pt atomically."""
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")

    latest_path = Path(latest_path)
    history_path = latest_path.parent / f"checkpoint_step_{int(step):012d}.pt"
    payload = _build_payload(
        model,
        optimizer,
        step,
        epoch=epoch,
        batch_index=batch_index,
        scheduler=scheduler,
        best_validation_loss=best_validation_loss,
        epochs_without_improvement=epochs_without_improvement,
    )

    _atomic_torch_save(payload, history_path)
    _update_latest_link(history_path, latest_path)
    prune_checkpoint_history(latest_path.parent, keep_last=keep_last)
    return history_path


def _find_valid_history_checkpoint(path):
    path = Path(path)
    for _, candidate in _history_files(path.parent):
        try:
            torch.load(candidate, map_location="cpu", weights_only=False)
            return candidate
        except Exception:
            continue
    return None


def load_checkpoint(
    model,
    optimizer,
    path,
    map_location=None,
    scheduler=None,
):
    """Restore a checkpoint and return its stored training state.

    The returned object remains integer-compatible for legacy callers that
    compared the return value directly with the optimizer step, while also
    exposing the full resume metadata through mapping-style access.

    When latest.pt is unreadable, the newest valid checkpoint_step_*.pt in
    the same directory is used automatically.
    """
    requested_path = Path(path)
    checkpoint_path = requested_path

    try:
        checkpoint = torch.load(
            requested_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception:
        if requested_path.name != "latest.pt":
            raise
        fallback_path = _find_valid_history_checkpoint(requested_path)
        if fallback_path is None:
            raise
        checkpoint_path = fallback_path
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return CheckpointState(
        checkpoint["step"],
        epoch=checkpoint.get("epoch", 0),
        batch_index=checkpoint.get("batch_index", 0),
        best_validation_loss=checkpoint.get("best_validation_loss"),
        epochs_without_improvement=checkpoint.get(
            "epochs_without_improvement", 0
        ),
        checkpoint_path=str(checkpoint_path),
        used_fallback=checkpoint_path != requested_path,
    )
