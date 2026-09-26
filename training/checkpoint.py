from pathlib import Path

import torch


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


def _atomic_torch_save(payload, path: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _capture_rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    cpu_state = state.get("cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


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
    extra_state=None,
):
    """Save model and complete training state for reliable resume.

    ``extra_state`` lets shard-based trainers persist the current shard and
    other run metadata without changing the contract for older callers.
    """
    path = Path(path)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "batch_index": batch_index,
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "rng_state": _capture_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state:
        payload.update(dict(extra_state))

    _atomic_torch_save(payload, path)


def load_checkpoint(
    model,
    optimizer,
    path,
    map_location=None,
    scheduler=None,
    restore_rng=True,
):
    """Restore a checkpoint and return its stored training state.

    The returned object remains integer-compatible for legacy callers that
    compared the return value directly with the optimizer step, while also
    exposing full resume metadata through mapping-style access.
    """
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if restore_rng:
        _restore_rng_state(checkpoint.get("rng_state"))

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {
            "model_state_dict",
            "optimizer_state_dict",
            "rng_state",
            "scheduler_state_dict",
        }
    }
    return CheckpointState(checkpoint["step"], **metadata)
