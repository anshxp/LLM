from pathlib import Path
import struct

import numpy as np
import pytest

from data.memmap_dataset import HEADER_SIZE, MAGIC, MemmapLanguageModelDataset, MemmapTokenStore

def _write_store(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.Struct("<8sQ").pack(MAGIC, len(values)))
        np.asarray(values, dtype="<u4").tofile(handle)

def test_memmap_store_reads_tokens(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_store(path, range(10))
    with MemmapTokenStore(path) as store:
        assert store.length == 10
        assert store.slice(2, 5).tolist() == [2, 3, 4]

def test_memmap_dataset_matches_legacy_windows(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_store(path, range(10))
    dataset = MemmapLanguageModelDataset(path, context_length=4)
    assert len(dataset) == 2
    inputs, targets = dataset[1]
    assert inputs.tolist() == [4, 5, 6, 7]
    assert targets.tolist() == [5, 6, 7, 8]

def test_memmap_dataset_supports_stride(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_store(path, range(8))
    dataset = MemmapLanguageModelDataset(path, context_length=4, stride=2)
    assert len(dataset) == 2
    inputs, targets = dataset[1]
    assert inputs.tolist() == [2, 3, 4, 5]
    assert targets.tolist() == [3, 4, 5, 6]

@pytest.mark.parametrize("bad_value", [-1, 0])
def test_memmap_dataset_rejects_bad_context(tmp_path, bad_value):
    path = tmp_path / "tokens.bin"
    _write_store(path, range(10))
    with pytest.raises(ValueError, match="context_length"):
        MemmapLanguageModelDataset(path, context_length=bad_value)

def test_corrupt_store_is_rejected(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError):
        MemmapTokenStore(path).open()
