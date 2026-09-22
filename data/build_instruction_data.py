"""Build auditable instruction-tuning JSONL from the existing LLM-Data corpus.

The builder separates genuinely supervised examples from passage-copy examples. Explicit
source Q/A records and curated healthcare examples are suitable for SFT because the response
is a distinct answer. Passage-copy transformations remain available for auditing, but are
excluded from the default supervised splits.

Supervised splits are built independently from passage-copy artifacts so tiny QA sets are not
diluted or accidentally excluded by a global split.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

FIELDS = ("instruction", "input", "response", "category", "source")
EXAMPLE_FIELDS = ("instruction", "input", "response")
OUTPUT_SPLITS = ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")
CURATED_CATEGORIES = {"terminology", "simplification", "summarization", "health_information"}
SFT_CATEGORIES = frozenset((*CURATED_CATEGORIES, "source_qa"))


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def stable_id(record):
    payload = "\n".join(record[k] for k in FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def record_key(record):
    return tuple(record[k] for k in EXAMPLE_FIELDS)


def make_record(instruction, source_text, response, category, source):
    instruction = normalize(instruction)
    source_text = normalize(source_text)
    response = normalize(response)
    if not instruction or not source_text or not response:
        return None
    record = {
        "instruction": instruction,
        "input": source_text,
        "response": response,
        "category": category,
        "source": source,
    }
    record["id"] = stable_id(record)
    return record


def read_text_files(root):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if text.strip():
                yield path, text


def paragraphs(text, min_chars=120, max_chars=1800):
    raw = re.split(r"\n\s*\n+", text)
    for block in raw:
        block = normalize(block)
        if len(block) < min_chars:
            continue
        if len(block) <= max_chars:
            yield block
            continue
        sentences = re.split(r"(?<=[.!?])\s+", block)
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                yield current
                current = ""
            current = f"{current} {sentence}".strip()
        if len(current) >= min_chars:
            yield current


def explicit_qa(text):
    lines = [normalize(x) for x in text.splitlines() if normalize(x)]
    for index, question in enumerate(lines[:-1]):
        if question.endswith("?") and len(question) >= 10:
            answer = lines[index + 1]
            if len(answer) >= 40:
                yield question, answer


def curated_examples(source_root):
    path = Path(source_root) / "healthcare_examples.jsonl"
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
        category = raw.get("category")
        if category not in CURATED_CATEGORIES:
            raise ValueError(f"Unknown curated category at line {line_number}: {category}")
        record = make_record(
            raw.get("instruction", ""),
            raw.get("input", ""),
            raw.get("response", ""),
            category,
            path.name,
        )
        if record is not None:
            records.append(record)
    return records


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
    keys = [record_key(record) for record in records]
    ids = [record["id"] for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Instruction builder produced duplicate examples")
    if len(ids) != len(set(ids)):
        raise ValueError("Instruction builder produced duplicate record ids")


def build_records(source_root):
    records = curated_examples(source_root)
    seen_ids = {record["id"] for record in records}
    seen_examples = {record_key(record) for record in records}

    def add_record(record):
        if record is None:
            return
        key = record_key(record)
        if key in seen_examples or record["id"] in seen_ids:
            return
        records.append(record)
        seen_examples.add(key)
        seen_ids.add(record["id"])

    for path, text in read_text_files(source_root):
        source = str(path.relative_to(source_root)).replace("\\", "/")
        for question, answer in explicit_qa(text):
            add_record(
                make_record(
                    "Answer the question using only the provided source text.",
                    question,
                    answer,
                    "source_qa",
                    source,
                )
            )

    passage_records = list(records)
    for path, text in read_text_files(source_root):
        source = str(path.relative_to(source_root)).replace("\\", "/")
        templates = (
            ("Extract the key information from the following medical passage.", "grounded_extraction"),
            ("Explain the following passage without adding information not present in it.", "grounded_explanation"),
            ("Provide the relevant source text for this request without inventing facts.", "grounded_response"),
        )
        for passage in paragraphs(text):
            for instruction, category in templates:
                record = make_record(instruction, passage, passage, category, source)
                if record is not None:
                    passage_records.append(record)

    return supervised_records, deduplicate_records(passage_records)


def split(records, train_ratio=0.9, validation_ratio=0.05):
    records = deduplicate_records(records)
    validate_unique_records(records)
    records = sorted(records, key=lambda record: record["id"])
    n = len(records)
    if n < 3:
        raise ValueError("At least 3 records are required to create train/validation/test splits.")
    train_count = max(1, int(n * train_ratio))
    validation_count = max(1, int(n * validation_ratio))
    if train_count + validation_count >= n:
        validation_count = 1
        train_count = n - 2
    return (
        records[:train_count],
        records[train_count:train_count + validation_count],
        records[train_count + validation_count:],
    )


def write_jsonl(records, path):
    validate_unique_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_SPLITS:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def verify_written_dataset(output_dir):
    output_dir = Path(output_dir)
    all_records = []
    split_ids = {}
    for split_name in ("train", "validation", "test"):
        path = output_dir / f"{split_name}.jsonl"
        if not path.exists():
            raise RuntimeError(f"Missing generated split: {path}")
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            records.append(record)
        validate_unique_records(records)
        split_ids[split_name] = {record["id"] for record in records}
        all_records.extend(records)
    validate_unique_records(all_records)
    names = tuple(split_ids)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if split_ids[left] & split_ids[right]:
                raise RuntimeError(f"Generated splits overlap: {left} and {right}")
    return len(all_records)


def main():
    parser = argparse.ArgumentParser(description="Build auditable instruction JSONL from LLM-Data.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/instruction"))
    parser.add_argument("--min-supervised-examples", type=int, default=100)
    args = parser.parse_args()

    supervised_records, passage_records = build_records(args.source_root)
    if len(supervised_records) < args.min_supervised_examples:
        raise RuntimeError(
            f"Only {len(supervised_records)} supervised examples were produced; "
            f"need at least {args.min_supervised_examples}. Add reviewed/generated QA before SFT."
        )

    clean_output_dir(args.output_dir)
    train, validation, test = split(supervised_records)
    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(validation, args.output_dir / "validation.jsonl")
    write_jsonl(test, args.output_dir / "test.jsonl")

    written_total = verify_written_dataset(args.output_dir)
    if written_total != len(supervised_records):
        raise RuntimeError(
            f"Generated supervised dataset count mismatch: built {len(supervised_records)}, wrote {written_total}"
        )

    manifest = {
        "source_root": str(args.source_root),
        "supervised_total": len(supervised_records),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "supervised_categories": {
            category: sum(record["category"] == category for record in supervised_records)
            for category in sorted({record["category"] for record in supervised_records})
        },
        "passage_copy_total": max(0, len(passage_records) - len(supervised_records)),
        "sft_categories": sorted(SFT_CATEGORIES),
        "duplicate_examples": 0,
        "provenance": (
            "SFT records come only from curated healthcare examples or explicit source Q/A. "
            "Passage-copy records are excluded from the supervised artifact."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
