from train import parse_args


def test_parser_defaults_are_stable():
    args = parse_args([])
    assert args.batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.device == "auto"
    assert args.num_workers == 0
