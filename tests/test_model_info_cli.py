from tools.model_info import main


def test_model_info_cli_imports():
    assert callable(main)
