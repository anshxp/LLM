"""Build auditable instruction-tuning JSONL from the existing LLM-Data corpus."""

import argparse
import hashlib
import json
import re
from pathlib import Path

SFT_CATEGORIES = {"health_information", "simplification", "source_qa", "summarization", "terminology"}
FIELDS = ("instruction", "input", "response", "category", "source")
CURATED_PATH = Path("healthcare_examples.jsonl")


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def stable_id(record):
    return hashlib.sha256("\n".join(record[k] for k in FIELDS).encode("utf-8")).hexdigest()[:16]


def make_record(instruction, source_text, response, category, source):
    instruction, source_text, response = normalize(instruction), normalize(source_text), normalize(response)
    if not instruction or not source_text or not response:
        return None
    record = {"instruction": instruction, "input": source_text, "response": response,
              "category": category, "source": source}
    record["id"] = stable_id(record)
    return record


def record_key(record):
    return tuple(record[k] for k in ("instruction", "input", "response"))


def deduplicate_records(records):
    unique, seen = [], set()
    for record in records:
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def validate_unique_records(records):
    keys = [record_key(r) for r in records]
    ids = [r["id"] for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Instruction builder produced duplicate examples")
    if len(ids) != len(set(ids)):
        raise ValueError("Instruction builder produced duplicate record ids")


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


def paragraphs(text, min_chars=160, max_chars=1800):
    for block in re.split(r"\n\s*\n+", text):
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
    lines = [normalize(x) for x in text.splitlines() if normalize(x)]
    for i, line in enumerate(lines[:-1]):
        if line.endswith("?") and len(line) >= 10 and len(lines[i + 1]) >= 40:
            yield line, lines[i + 1]


def curated_examples(source_root):
    path = Path(source_root) / CURATED_PATH
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
        required = ("instruction", "input", "response", "category")
        if any(not normalize(raw.get(field, "")) for field in required):
            raise ValueError(f"Invalid curated example at line {line_number}: missing required field")
        if raw["category"] not in SFT_CATEGORIES:
            raise ValueError(f"Invalid curated example at line {line_number}: unsupported category {raw['category']!r}")
        record = make_record(raw["instruction"], raw["input"], raw["response"], raw["category"], raw.get("source", str(CURATED_PATH)))
        if record:
            records.append(record)
    return deduplicate_records(records)


def build_records(source_root):
    source_root = Path(source_root)
    supervised_records = curated_examples(source_root)
    seen_ids = {r["id"] for r in supervised_records}
    seen_examples = {record_key(r) for r in supervised_records}

    def add_supervised(record):
        if record is None:
            return
        key = record_key(record)
        if key in seen_examples or record["id"] in seen_ids:
            return
        supervised_records.append(record)
        seen_examples.add(key)
        seen_ids.add(record["id"])

    for path, text in read_text_files(source_root):
        if path.name == CURATED_PATH.name:
            continue
        source = str(path.relative_to(source_root)).replace("\\", "/")
        for question, answer in explicit_qa(text):
            add_supervised(make_record("Answer the question using only the provided source text.", question, answer, "source_qa", source))

    all_records = list(supervised_records)
    for path, text in read_text_files(source_root):
        if path.name == CURATED_PATH.name:
            continue
        source = str(path.relative_to(source_root)).replace("\\", "/")
        templates = (("Extract the key information from the following medical passage.", "grounded_extraction"),
                     ("Explain the following passage without adding information not present in it.", "grounded_explanation"),
                     ("Provide the relevant source text for this request without inventing facts.", "grounded_response"))
        for passage in paragraphs(text):
            for instruction, category in templates:
                record = make_record(instruction, passage, passage, category, source)
                if record is not None:
                    all_records.append(record)

    return deduplicate_records(supervised_records), deduplicate_records(all_records)


def split(records, train_ratio=0.9, validation_ratio=0.05):
    records = sorted(deduplicate_records(records), key=lambda r: r["id"])
    validate_unique_records(records)
    n = len(records)
    if n < 3:
        raise ValueError("At least 3 records are required to create train/validation/test splits.")
    train_count = max(1, int(n * train_ratio))
    validation_count = max(1, int(n * validation_ratio))
    if train_count + validation_count >= n:
        validation_count = 1
        train_count = n - 2
    return records[:train_count], records[train_count:train_count + validation_count], records[train_count + validation_count:]


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        (path / name).unlink(missing_ok=True)


def verify_written_dataset(path):
    records = []
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        file = path / name
        if file.exists():
            for line in file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
    records = deduplicate_records(records)
    validate_unique_records(records)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/instruction"))
    parser.add_argument("--min-supervised-examples", type=int, default=100)
    args = parser.parse_args()

    supervised_records, all_records = build_records(args.source_root)
    if len(supervised_records) < args.min_supervised_examples:
        raise RuntimeError(f"Only {len(supervised_records)} supervised examples were produced; need at least {args.min_supervised_examples}.")

    clean_output_dir(args.output_dir)
    train, validation, test = split(supervised_records)
    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(validation, args.output_dir / "validation.jsonl")
    write_jsonl(test, args.output_dir / "test.jsonl")
    verify_written_dataset(args.output_dir)

    manifest = {"source_root": str(args.source_root), "supervised_total": len(supervised_records),
                "train": len(train), "validation": len(validation), "test": len(test),
                "supervised_categories": {c: sum(r["category"] == c for r in supervised_records) for c in sorted({r["category"] for r in supervised_records})},
                "passage_copy_total": max(0, len(all_records) - len(supervised_records)),
                "sft_categories": sorted(SFT_CATEGORIES), "duplicate_examples": 0,
                "provenance": "SFT records come only from curated healthcare examples or explicit source Q/A. Passage-copy records are excluded from the supervised artifact."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
