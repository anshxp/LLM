import train


def test_training_module_imports():
    assert callable(train.main)
    assert callable(train.parse_args)
