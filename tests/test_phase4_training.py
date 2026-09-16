from train import parse_args, resolve_device, validate_args


def test_default_training_config_targets_low_memory_hardware():
    args = parse_args([])

    assert args.batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.num_workers == 0
    assert args.max_grad_norm == 1.0


def test_training_arguments_are_configurable():
    args = parse_args([
        "--batch-size", "2",
        "--gradient-accumulation-steps", "8",
        "--epochs", "3",
        "--max-train-batches", "5",
    ])

    validate_args(args)
    assert args.batch_size == 2
    assert args.gradient_accumulation_steps == 8
    assert args.epochs == 3
    assert args.max_train_batches == 5


def test_invalid_batch_size_is_rejected():
    args = parse_args(["--batch-size", "0"])

    try:
        validate_args(args)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("Expected invalid batch size to be rejected")


def test_cpu_device_can_always_be_selected():
    assert str(resolve_device("cpu")) == "cpu"
