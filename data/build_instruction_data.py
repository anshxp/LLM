"""Build auditable instruction-tuning JSONL from the existing LLM-Data corpus.

This builder is intentionally extractive. It never invents medical facts: responses are
copied from source passages, or from explicit question/answer structures already present
in the source. Use a teacher model or human review later if genuinely generated QA is
required.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

FIELDS = ("instruction", "input", "response", "category", "source")
EXAMPLE_FIELDS = ("instruction", "input", "response")
OUTPUT_SPLITS = ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def stable_id(record):
    payload = "\n".join(record[k] for k in FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def record_key(record):
    """Return the semantic example key used for duplicate detection.

    Source/category are deliberately excluded: the same instruction/input/response
    must never appear twice, even if it was discovered in multiple source files or
    assigned to different categories.
    """
    return tuple(record[k] for k in EXAMPLE_FIELDS)


def make_record(instruction, source_text, response, category, source):
    source_text = normalize(source_text)
    response = normalize(response)
    instruction = normalize(instruction)
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


def paragraphs(text, min_chars=160, max_chars=1800):
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
    for i, line in enumerate(lines[:-1]):
        if line.endswith("?") and len(line) >= 10:
            answer = lines[i + 1]
            if len(answer) >= 40:
                yield line, answer


def deduplicate_records(records):
    """Canonicalize a record collection by semantic example content.

    This second deduplication pass is intentional. It protects the final artifact
    even if a future builder path bypasses ``add_record`` or introduces another
    record-producing source. The first occurrence is retained deterministically.
    """
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
    """Raise if records are not unique by training-example content or id."""
    example_keys = [record_key(record) for record in records]
    ids = [record["id"] for record in records]
    if len(example_keys) != len(set(example_keys)):
        raise ValueError("Instruction builder produced duplicate examples")
    if len(ids) != len(set(ids)):
        raise ValueError("Instruction builder produced duplicate record ids")


def build_records(source_root):
    records = []
    seen_ids = set()
    seen_examples = set()

    def add_record(record):
        if record is None:
            return
        example_key = record_key(record)
        if example_key in seen_examples or record["id"] in seen_ids:
            return
        records.append(record)
        seen_examples.add(example_key)
        seen_ids.add(record["id"])

    for path, text in read_text_files(source_root):
        source = str(path.relative_to(source_root)).replace("\\", "/")
        for question, answer in explicit_qa(text):
            add_record(
                make_record(
                    "Answer the question using only the provided source text.",
                    f"Question: {question}\nSource answer: {answer}",
                    answer,
                    "source_qa",
                    source,
                )
            )

        templates = [
            (
                "Extract the key information from the following medical passage.",
                "grounded_extraction",
            ),
            (
                "Explain the following passage without adding information not present in it.",
                "grounded_explanation",
            ),
            (
                "Provide the relevant source text for this request without inventing facts.",
                "grounded_response",
            ),
        ]
        for passage in paragraphs(text):
            for instruction, category in templates:
                add_record(make_record(instruction, passage, passage, category, source))

    records = deduplicate_records(records)
    validate_unique_records(records)
    return records


def split(records, train_ratio=0.9, validation_ratio=0.05):
    records = deduplicate_records(records)
    validate_unique_records(records)
    records = sorted(records, key=lambda r: r["id"])
    n = len(records)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * validation_ratio)
    return records[:train_end], records[train_end:val_end], records[val_end:]


def write_jsonl(records, path):
    validate_unique_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_output_dir(output_dir):
    """Remove only artifacts owned by this builder before writing a new dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_SPLITS:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def verify_written_dataset(output_dir):
    """Read all generated splits back and verify global uniqueness/disjointness."""
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
        for right in names[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise RuntimeError(f"Generated splits overlap: {left} and {right}")
    return len(all_records)


def main():
    parser = argparse.ArgumentParser(
        description="Build source-grounded instruction JSONL from LLM-Data."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/instruction"))
    args = parser.parse_args()

    records = build_records(args.source_root)
    if len(records) < 100:
        raise RuntimeError(
            f"Only {len(records)} examples were produced; refusing to create a tiny fine-tuning set."
        )

    clean_output_dir(args.output_dir)
    train, validation, test = split(records)
    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(validation, args.output_dir / "validation.jsonl")
    write_jsonl(test, args.output_dir / "test.jsonl")

    written_total = verify_written_dataset(args.output_dir)
    if written_total != len(records):
        raise RuntimeError(
            f"Generated dataset count mismatch: built {len(records)}, wrote {written_total}"
        )

    manifest = {
        "source_root": str(args.source_root),
        "total": len(records),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "categories": {
            category: sum(r["category"] == category for r in records)
            for category in sorted({r["category"] for r in records})
        },
        "duplicate_examples": 0,
        "provenance": "Every response is copied from the source passage or an explicit Q/A answer; no generated medical facts are introduced.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
