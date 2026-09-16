from inference.run_generation import main


def test_generation_cli_module_imports():
    assert callable(main)
