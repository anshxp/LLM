"""Ingest and normalize the complete local Fine tuning 2 corpus for SFT.

The raw corpus is intentionally kept outside the repository. This module reads it
locally, normalizes supervised records, audits malformed/overlength examples,
performs global exact deduplication, and writes deterministic train/validation/test
JSONL shards. It never modifies the source corpus and does not train the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

from data.tokenizer import Tokenizer

SUPPORTED = {".json", ".jsonl", ".csv", ".tsv", ".xml", ".txt", ".md", ".zip"}
TEXT_FIELDS = ("instruction", "input", "response")
QUESTION_FIELDS = ("question", "prompt", "query", "user", "instruction")
ANSWER_FIELDS = ("response", "answer", "output", "completion", "assistant")
ROLE_USER = {"user", "human", "customer", "question"}
ROLE_ASSISTANT = {"assistant", "bot", "model", "answer"}


def clean(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def source_name(path: str) -> str:
    return path.replace("\\", "/")


def make_record(instruction, input_text, response, category, source, source_id=""):
    instruction, input_text, response = map(clean, (instruction, input_text, response))
    category = clean(category) or "fine_tuning_2"
    if not instruction or not response:
        return None
    record = {
        "instruction": instruction,
        "input": input_text,
        "response": response,
        "category": category,
        "source": source_name(source),
        "source_id": clean(source_id),
    }
    record["id"] = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return record


def first_value(obj, fields):
    if not isinstance(obj, dict):
        return ""
    lowered = {str(k).lower(): v for k, v in obj.items()}
    for field in fields:
        value = lowered.get(field.lower())
        if value is not None and clean(value):
            return value
    return ""


def conversation_record(obj, source, source_id):
    messages = obj if isinstance(obj, list) else obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(messages, list):
        return None
    user_parts, assistant = [], ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = clean(message.get("role") or message.get("from") or message.get("speaker")).lower()
        text = clean(message.get("content") or message.get("text") or message.get("value"))
        if not text:
            continue
        if role in ROLE_USER:
            user_parts.append(text)
        elif role in ROLE_ASSISTANT and user_parts:
            assistant = text
    if not user_parts or not assistant:
        return None
    return make_record("Answer the user's request.", "\n\n".join(user_parts), assistant, "conversation", source, source_id)


def records_from_object(obj, source, source_id="") -> Iterable[dict]:
    if isinstance(obj, dict):
        conv = conversation_record(obj, source, source_id)
        if conv:
            yield conv
            return

        instruction = first_value(obj, ("instruction",))
        input_text = first_value(obj, ("input", "context"))
        response = first_value(obj, ANSWER_FIELDS)
        if instruction and response:
            record = make_record(instruction, input_text, response, first_value(obj, ("category", "type")), source, source_id)
            if record:
                yield record
            return

        question = first_value(obj, QUESTION_FIELDS)
        answer = first_value(obj, ANSWER_FIELDS)
        if question and answer:
            record = make_record("Answer the question accurately.", question, answer, "question_answer", source, source_id)
            if record:
                yield record
            return

        for key, value in obj.items():
            child_id = f"{source_id}.{key}" if source_id else str(key)
            yield from records_from_object(value, source, child_id)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            yield from records_from_object(item, source, f"{source_id}[{index}]")


def parse_json(text, source):
    obj = json.loads(text)
    yield from records_from_object(obj, source)


def parse_jsonl(text, source):
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        yield from records_from_object(obj, source, str(number))


def parse_delimited(text, source, delimiter):
    rows = csv.DictReader(text.splitlines(), delimiter=delimiter)
    for number, row in enumerate(rows, 2):
        yield from records_from_object(dict(row), source, str(number))


def xml_text(element):
    return clean(" ".join(element.itertext())) if element is not None else ""


def parse_xml(text, source):
    root = ElementTree.fromstring(text)
    for node in root.iter():
        children = {child.tag.split("}")[-1].lower(): xml_text(child) for child in list(node)}
        question = first_value(children, QUESTION_FIELDS)
        answer = first_value(children, ANSWER_FIELDS)
        if question and answer:
            yield make_record("Answer the question accurately.", question, answer, "question_answer", source, node.tag)


def parse_text(text, source):
    lines = [clean(x) for x in text.splitlines() if clean(x)]
    for i, line in enumerate(lines[:-1]):
        if line.endswith("?") and len(lines[i + 1]) >= 20:
            yield make_record("Answer the question accurately.", line, lines[i + 1], "question_answer", source, str(i + 1))


def iter_archive(path: Path):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED - {".zip"}:
                continue
            raw = archive.read(name)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="ignore")
            yield source_name(f"{path}::{name}"), suffix, text


def parse_source(suffix, text, source):
    if suffix == ".json":
        yield from parse_json(text, source)
    elif suffix == ".jsonl":
        yield from parse_jsonl(text, source)
    elif suffix in {".csv", ".tsv"}:
        yield from parse_delimited(text, source, "\t" if suffix == ".tsv" else ",")
    elif suffix == ".xml":
        yield from parse_xml(text, source)
    elif suffix in {".txt", ".md"}:
        yield from parse_text(text, source)


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def split_name(record_id: str) -> str:
    value = int(record_id[:8], 16) % 100
    if value < 90:
        return "train"
    if value < 95:
        return "validation"
    return "test"


def token_stats(record, tokenizer, context_length):
    prompt = f"### Instruction:\n{record['instruction']}\n\n### Input:\n{record['input']}\n\n### Response:\n"
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    response_ids = tokenizer.encode(record["response"], add_eos=True)
    total = len(prompt_ids) + len(response_ids)
    unk_id = tokenizer.token_to_id["<unk>"]
    return {
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "total_tokens": total,
        "unk_tokens": sum(x == unk_id for x in prompt_ids + response_ids),
        "fits_context": total <= context_length + 1,
    }


def write_jsonl(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(source_root: Path, output_dir: Path, tokenizer_path: Path, context_length: int, shard_size: int = 50000):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.jsonl"):
        old.unlink()
    for old in (output_dir / "manifest.json", output_dir / "rejections.jsonl"):
        old.unlink(missing_ok=True)

    seen = set()
    counts = Counter()
    source_counts = Counter()
    rejection_counts = Counter()
    length_counts = Counter()
    total_tokens = 0
    unk_tokens = 0
    output_handles = {}
    shard_counts = Counter()
    rejection_handle = (output_dir / "rejections.jsonl").open("w", encoding="utf-8")

    def handle_for(split):
        index = shard_counts[split] // shard_size
        key = (split, index)
        if key not in output_handles:
            path = output_dir / f"{split}-{index:05d}.jsonl"
            output_handles[key] = path.open("w", encoding="utf-8")
        return output_handles[key]

    try:
        for path in iter_files(source_root):
            relative = source_name(str(path.relative_to(source_root)))
            try:
                sources = [(relative, path.suffix.lower(), path.read_text(encoding="utf-8", errors="ignore"))]
                if path.suffix.lower() == ".zip":
                    sources = list(iter_archive(path))
                for source, suffix, text in sources:
                    counts["files_or_members"] += 1
                    try:
                        records = parse_source(suffix, text, source)
                        for record in records:
                            if record is None:
                                rejection_counts["empty_or_invalid"] += 1
                                continue
                            key = (clean(record["instruction"]).lower(), clean(record["input"]).lower(), clean(record["response"]).lower())
                            if key in seen:
                                rejection_counts["duplicate"] += 1
                                continue
                            seen.add(key)
                            stats = token_stats(record, tokenizer, context_length)
                            total_tokens += stats["total_tokens"]
                            unk_tokens += stats["unk_tokens"]
                            length_counts["within_context" if stats["fits_context"] else "over_context"] += 1
                            if not stats["fits_context"]:
                                rejection_counts["over_context"] += 1
                                rejection_handle.write(json.dumps({"reason": "over_context", "record": record, "stats": stats}, ensure_ascii=False) + "\n")
                                continue
                            split = split_name(record["id"])
                            record.update(stats)
                            write_jsonl(handle_for(split), record)
                            shard_counts[split] += 1
                            source_counts[source] += 1
                    except Exception as exc:
                        rejection_counts[f"parse_error:{type(exc).__name__}"] += 1
                        rejection_handle.write(json.dumps({"reason": f"parse_error:{type(exc).__name__}", "source": source, "error": str(exc)}, ensure_ascii=False) + "\n")
            except Exception as exc:
                rejection_counts[f"file_error:{type(exc).__name__}"] += 1
                rejection_handle.write(json.dumps({"reason": f"file_error:{type(exc).__name__}", "source": relative, "error": str(exc)}, ensure_ascii=False) + "\n")
    finally:
        rejection_handle.close()
        for handle in output_handles.values():
            handle.close()

    accepted = sum(shard_counts.values())
    manifest = {
        "source_root": str(source_root),
        "tokenizer": str(tokenizer_path),
        "context_length": context_length,
        "files_or_archive_members_seen": counts["files_or_members"],
        "unique_records_seen": len(seen),
        "accepted_records": accepted,
        "train_records": shard_counts["train"],
        "validation_records": shard_counts["validation"],
        "test_records": shard_counts["test"],
        "rejections": dict(rejection_counts),
        "length_distribution": dict(length_counts),
        "average_tokens": total_tokens / max(1, len(seen)),
        "unk_token_rate": unk_tokens / max(1, total_tokens),
        "source_counts": dict(source_counts),
        "split_policy": "stable hash of record id: 90% train, 5% validation, 5% test",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare the complete Fine tuning 2 corpus for second SFT.")
    parser.add_argument("--source-root", type=Path, default=Path("data/Fine tuning 2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/instruction_v2"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=50000)
    args = parser.parse_args()
    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Fine tuning 2 directory not found: {args.source_root}")
    if args.context_length < 2 or args.shard_size <= 0:
        raise ValueError("context-length must be >= 2 and shard-size must be positive")
    print(json.dumps(build(args.source_root, args.output_dir, args.tokenizer, args.context_length, args.shard_size), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
