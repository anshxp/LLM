from train import parse_args


def test_training_parser_exposes_resume_and_device():
    assert parse_args(["--resume", "x.pt"]).resume.name == "x.pt"
    assert parse_args(["--device", "cpu"]).device == "cpu"
