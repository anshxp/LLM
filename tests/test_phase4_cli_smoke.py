from train import parse_args


def test_low_memory_smoke_arguments():
    args = parse_args(["--batch-size", "1", "--gradient-accumulation-steps", "4", "--epochs", "1", "--max-train-batches", "1"])
    assert (args.batch_size, args.gradient_accumulation_steps, args.epochs, args.max_train_batches) == (1, 4, 1, 1)
