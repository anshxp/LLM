"""Validate a Colab runtime before expensive RXLM 2 preprocessing/training."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "requirements.txt",
    "config/model_config.py",
    "data/prepare_training_data.py",
    "data/memmap_dataset.py",
    "data/train_tokenizer.py",
    "train.py",
)


def _memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024**3)
    except (AttributeError, OSError, ValueError):
        return None


def main() -> int:
    print(f"Repository root: {REPO_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    memory = _memory_gb()
    if memory is not None:
        print(f"System RAM: {memory:.2f} GiB")
        if memory < 8:
            print("WARNING: less than 8 GiB RAM detected.")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB"
            )
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        return 1

    missing = [path for path in REQUIRED_FILES if not (REPO_ROOT / path).exists()]
    if missing:
        print("ERROR: missing repository files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    for package in ("numpy", "tokenizers", "pytest"):
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"{package}: {version}")
        except ImportError:
            print(f"ERROR: missing Python package: {package}")
            return 1

    git = shutil.which("git")
    print(f"git: {git or 'not found'}")
    print("Colab preflight: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
