"""Build deterministic supervised and audit instruction datasets.

The supervised dataset contains only curated healthcare examples and explicit source Q/A.
Passage-copy examples are returned separately as an audit corpus and are never written into
the default SFT dataset.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

SFT_CATEGORIES = frozenset(
    {"health_information", "simplification", "source_qa", "summarization", "terminology"}
)
PASSAGE_CATEGORIES = (
    ("grounded_extraction", "Extract the key information from the following medical passage."),
    ("grounded_explanation", "Explain the following passage without adding information not present in it."),
    ("grounded_response", "Provide the relevant source text for this request without inventing facts."),
)
FIELDS = ("instruction", "input", "response", "category", "source")
CURATED_FILENAME = "healthcare_examples.jsonl"
GENERATED_FILENAMES = {"train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"}


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def stable_id(record):
    return hashlib.sha256("\n".join(record[field] for field in FIELDS).encode("utf-8")).hexdigest()[:16]


def record_key(record):
    return tuple(record[field] for field in ("instruction", "input", "response"))


def make_record(instruction, input_text, response, category, source):
    values = [normalize(instruction), normalize(input_text), normalize(response)]
    if not all(values):
        return None
    record = {
        "instruction": values[0],
        "input": values[1],
        "response": values[2],
        "category": normalize(category),
        "source": normalize(source),
    }
    record["id"] = stable_id(record)
    return record


def deduplicate_records(records):
    unique = []
    seen_examples = set()
    seen_ids = set()
    for record in records:
        key = record_key(record)
        if key in seen_examples or record["id"] in seen_ids:
            continue
        unique.append(record)
        seen_examples.add(key)
        seen_ids.add(record["id"])
    return unique


def validate_unique_records(records):
    example_keys = [record_key(record) for record in records]
    ids = [record["id"] for record in records]
    if len(example_keys) != len(set(example_keys)):
        raise ValueError("Instruction builder produced duplicate examples")
    if len(ids) != len(set(ids)):
        raise ValueError("Instruction builder produced duplicate record ids")


def read_text_files(root):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if text.strip():
            yield path, text


def paragraphs(text, min_chars=120, max_chars=1800):
    blocks = re.split(r"\n\s*\n+", text)
    for block in blocks:
        block = normalize(block)
        if len(block) < min_chars:
            continue
        if len(block) <= max_chars:
            yield block
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            if current and len(current) + len(sentence) + 1 > max_chars:
                if len(current) >= min_chars:
                    yield current
                current = ""
            current = f"{current} {sentence}".strip()
        if len(current) >= min_chars:
            yield current


def explicit_qa(text):
    lines = [normalize(line) for line in text.splitlines() if normalize(line)]
    for index, question in enumerate(lines[:-1]):
        answer = lines[index + 1]
        if question.endswith("?") and len(question) >= 10 and len(answer) >= 40:
            yield question, answer


def load_curated_examples(source_root):
    path = Path(source_root) / CURATED_FILENAME
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid curated example at line {line_number}: {exc}") from exc
        for field in ("instruction", "input", "response", "category"):
            if not normalize(raw.get(field, "")):
                raise ValueError(f"Invalid curated example at line {line_number}: missing {field}")
        category = normalize(raw["category"])
        if category not in SFT_CATEGORIES:
            raise ValueError(
                f"Invalid curated example at line {line_number}: unsupported category {category!r}"
            )
        record = make_record(
            raw["instruction"],
            raw["input"],
            raw["response"],
            category,
            raw.get("source", CURATED_FILENAME),
        )
        if record:
            records.append(record)
    records = deduplicate_records(records)
    validate_unique_records(records)
    return records


def build_records(source_root):
    source_root = Path(source_root)
    supervised = load_curated_examples(source_root)
    supervised_keys = {record_key(record) for record in supervised}
    supervised_ids = {record["id"] for record in supervised}
    audit = list(supervised)

    def add_supervised(record):
        if record is None:
            return
        key = record_key(record)
        if key in supervised_keys or record["id"] in supervised_ids:
            return
        supervised.append(record)
        supervised_keys.add(key)
        supervised_ids.add(record["id"])

    for path, text in read_text_files(source_root):
        if path.name == CURATED_FILENAME:
            continue
        source = str(path.relative_to(source_root)).replace("\\", "/")
        for question, answer in explicit_qa(text):
            add_supervised(
                make_record(
                    "Answer the question using only the provided source text.",
                    question,
                    answer,
                    "source_qa",
                    source,
                )
            )

    # Build the audit corpus independently. It contains supervised examples plus passage
    # copies from source files. The curated JSONL is deliberately excluded from raw text
    # scanning so it cannot be transformed into passage-copy records.
    for path, text in read_text_files(source_root):
        if path.name == CURATED_FILENAME:
            continue
        source = str(path.relative_to(source_root)).replace("\\", "/")
        for passage in paragraphs(text):
            for category, instruction in PASSAGE_CATEGORIES:
                record = make_record(instruction, passage, passage, category, source)
                if record:
                    audit.append(record)

    supervised = deduplicate_records(supervised)
    audit = deduplicate_records(audit)
    validate_unique_records(supervised)
    validate_unique_records(audit)
    if not set(record_key(record) for record in supervised).issubset(
        {record_key(record) for record in audit}
    ):
        raise ValueError("Audit corpus must contain every supervised example")
    return supervised, audit


def split(records, train_ratio=0.9, validation_ratio=0.05):
    records = sorted(deduplicate_records(records), key=lambda record: record["id"])
    validate_unique_records(records)
    if len(records) < 3:
        raise ValueError("At least 3 records are required to create train/validation/test splits.")
    train_count = max(1, int(len(records) * train_ratio))
    validation_count = max(1, int(len(records) * validation_ratio))
    if train_count + validation_count >= len(records):
        train_count = len(records) - 2
        validation_count = 1
    return (
        records[:train_count],
        records[train_count:train_count + validation_count],
        records[train_count + validation_count:],
    )


def write_jsonl(records, path):
    validate_unique_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in GENERATED_FILENAMES:
        (output_dir / filename).unlink(missing_ok=True)


def verify_written_dataset(output_dir):
    output_dir = Path(output_dir)
    splits = {}
    all_records = []
    for split_name in ("train", "validation", "test"):
        path = output_dir / f"{split_name}.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_unique_records(records)
        splits[split_name] = records
        all_records.extend(records)
    validate_unique_records(all_records)
    ids = {name: {record["id"] for record in records} for name, records in splits.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise RuntimeError("Generated splits overlap")
    return len(all_records)


def main():
    parser = argparse.ArgumentParser(description="Build deterministic instruction-tuning data.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/instruction"))
    parser.add_argument("--min-supervised-examples", type=int, default=100)
    args = parser.parse_args()

    supervised, audit = build_records(args.source_root)
    if len(supervised) < args.min_supervised_examples:
        raise RuntimeError(
            f"Only {len(supervised)} supervised examples were produced; "
            f"need at least {args.min_supervised_examples}. Add reviewed/generated QA before SFT."
        )

    clean_output_dir(args.output_dir)
    train, validation, test = split(supervised)
    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(validation, args.output_dir / "validation.jsonl")
    write_jsonl(test, args.output_dir / "test.jsonl")
    written_total = verify_written_dataset(args.output_dir)
    if written_total != len(supervised):
        raise RuntimeError(
            f"Generated supervised dataset count mismatch: built {len(supervised)}, wrote {written_total}"
        )

    categories = {category: sum(record["category"] == category for record in supervised)
                  for category in sorted({record["category"] for record in supervised})}
    manifest = {
        "source_root": str(args.source_root),
        "supervised_total": len(supervised),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "supervised_categories": categories,
        "passage_copy_total": sum(record["category"] not in SFT_CATEGORIES for record in audit),
        "sft_categories": sorted(SFT_CATEGORIES),
        "duplicate_examples": 0,
        "provenance": "SFT records come only from curated healthcare examples or explicit source Q/A. Passage-copy records are retained only in the in-memory audit corpus.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
