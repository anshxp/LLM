from train import parse_args


def test_seed_can_be_configured():
    assert parse_args(["--seed", "123"]).seed == 123
