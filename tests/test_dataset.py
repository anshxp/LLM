import pytest

from data.dataset import LanguageModelDataset


def test_dataset_packs_complete_non_overlapping_sequences():
    dataset = LanguageModelDataset(list(range(10)), context_length=4)
    assert len(dataset) == 2
    inputs, targets = dataset[1]
    assert inputs.tolist() == [4, 5, 6, 7]
    assert targets.tolist() == [5, 6, 7, 8]


def test_dataset_supports_overlapping_stride():
    dataset = LanguageModelDataset(list(range(8)), context_length=4, stride=2)
    assert len(dataset) == 2
    inputs, targets = dataset[1]
    assert inputs.tolist() == [2, 3, 4, 5]
    assert targets.tolist() == [3, 4, 5, 6]


def test_dataset_rejects_invalid_stride():
    with pytest.raises(ValueError, match="stride"):
        LanguageModelDataset(list(range(8)), context_length=4, stride=0)
