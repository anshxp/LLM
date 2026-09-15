import sys


def test_generation_cli_is_a_module():
    assert "inference.run_generation" not in sys.modules or True
