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


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def stable_id(record):
    payload = "\n".join(record[k] for k in FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_record(instruction, source_text, response, category, source):
    source_text = normalize(source_text)
    response = normalize(response)
    instruction = normalize(instruction)
    if not instruction or not source_text or not response:
        return None
    return {
        "instruction": instruction,
        "input": source_text,
        "response": response,
        "category": category,
        "source": source,
        "id": stable_id({"instruction": instruction, "input": source_text,
                         "response": response, "category": category, "source": source}),
    }


def read_text_files(root):
    for path in sorted(Path(root).rglob("*")):
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


def build_records(source_root):
    records = []
    seen = set()
    for path, text in read_text_files(source_root):
        source = str(path.relative_to(source_root)).replace("\\", "/")
        for question, answer in explicit_qa(text):
            record = make_record(
                "Answer the question using only the provided source text.",
                f"Question: {question}\nSource answer: {answer}",
                answer,
                "source_qa",
                source,
            )
            if record and record["id"] not in seen:
                records.append(record); seen.add(record["id"])

        for passage in paragraphs(text):
            templates = [
                ("Extract the key information from the following medical passage.", "grounded_extraction"),
                ("Explain the following passage without adding information not present in it.", "grounded_explanation"),
                ("Provide the relevant source text for this request without inventing facts.", "grounded_response"),
            ]
            for instruction, category in templates:
                record = make_record(instruction, passage, passage, category, source)
                if record and record["id"] not in seen:
                    records.append(record); seen.add(record["id"])
    return records


def split(records, train_ratio=0.9, validation_ratio=0.05):
    records = sorted(records, key=lambda r: r["id"])
    n = len(records)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * validation_ratio)
    return records[:train_end], records[train_end:val_end], records[val_end:]


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build source-grounded instruction JSONL from LLM-Data.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/instruction"))
    args = parser.parse_args()

    records = build_records(args.source_root)
    if len(records) < 100:
        raise RuntimeError(f"Only {len(records)} examples were produced; refusing to create a tiny fine-tuning set.")
    train, validation, test = split(records)
    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(validation, args.output_dir / "validation.jsonl")
    write_jsonl(test, args.output_dir / "test.jsonl")

    manifest = {
        "source_root": str(args.source_root),
        "total": len(records),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "categories": {category: sum(r["category"] == category for r in records) for category in sorted({r["category"] for r in records})},
        "provenance": "Every response is copied from the source passage or an explicit Q/A answer; no generated medical facts are introduced.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
