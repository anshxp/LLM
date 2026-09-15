from train import parse_args


def test_resume_defaults_to_none():
    assert parse_args([]).resume is None
