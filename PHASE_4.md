# Phase 4 — Hardware-Aware Training

The training pipeline is configured for the 8 GB RAM Lenovo IdeaPad Slim 1 target.

## Defaults

- Batch size: 1
- Gradient accumulation: 4
- DataLoader workers: 0
- Gradient clipping: max norm 1.0
- Device: automatic CUDA detection, otherwise CPU
- Checkpoint after every epoch

## Training

Run `python train.py` for the default configuration.

For a short smoke run, use `python train.py --epochs 1 --max-train-batches 5`.

To resume weights and optimizer state, use `python train.py --resume checkpoints/model_epoch_1.pt`.

## Generation

Generate text from a trained checkpoint with:

`python -m inference.run_generation --checkpoint checkpoints/model_epoch_1.pt --prompt "The patient" --max-new-tokens 50 --top-k 20`

Generation truncates the prompt context to the model's configured context length, preventing position-embedding overflow.

## Model resource inspection

Run `python -m tools.model_info` to print parameter count and approximate parameter/AdamW memory requirements.
