# Phase 5: Healthcare Data Pipeline

Phase 5 connects structured healthcare examples to the existing language-model training stack.

## Pipeline

```text
healthcare_examples.jsonl
        |
        v
healthcare_dataset.py
(validate, deduplicate, split)
        |
        v
healthcare_format.py
(structured examples -> training text)
        |
        v
prepare_healthcare_data.py
        |
        +--> healthcare_train.txt
        +--> healthcare_validation.txt
        +--> healthcare_test.txt
        |
        v
train_tokenizer.py --dataset healthcare
        |
        v
prepare_training_data.py -- healthcare dataset
        |
        v
train.py --dataset healthcare
```

## Commands

Prepare the healthcare corpus:

```bash
python -m data.prepare_healthcare_data
```

Train a tokenizer on the healthcare training split only:

```bash
python data/train_tokenizer.py --dataset healthcare
```

Train the model on the healthcare corpus:

```bash
python train.py --dataset healthcare
```

For an 8 GB laptop, start with a small smoke test before a longer run:

```bash
python train.py --dataset healthcare --epochs 1 --max-train-batches 5
```

## Important limitation

The checked-in `data/healthcare_examples.jsonl` file contains only seed examples for validating the pipeline. It is not large enough to produce a useful healthcare language model. Real training requires a substantially larger, legally usable, high-quality healthcare corpus with documented provenance and licensing.

The model is intended for health-information assistance such as terminology explanation, simplification, summarization, and general educational questions. It should not be presented as a diagnostic or treatment authority.
