"""Build the continued-pretraining corpus without reading data/raw.

The builder intentionally uses only:
- the already-processed original training corpus at data/processed/train.txt;
- data/MedQuad;
- data/Fine tuning 2.

It never scans data/ or data/raw recursively, so the original raw pretraining
sources cannot accidentally be re-ingested.

The builder is deliberately verbose: it logs discovery, extraction, filtering,
deduplication, output progress, and final statistics so a long build can be
inspected from the terminal without guessing what happened.
"""

import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from data.cleaner import clean_text
from data.dedup import is_near_duplicate, simhash
from data.filters import passes_basic_filters
from data.pdf_extractor import extract_pdf_text

DEFAULT_OLD_TRAIN = Path("data/processed/train.txt")
DEFAULT_SOURCES = (Path("data/MedQuad"), Path("data/Fine tuning 2"))
DEFAULT_OUTPUT = Path("data/processed/continued_pretraining")
SUPPORTED_TEXT = {".txt", ".md", ".markdown", ".text"}
SUPPORTED_STRUCTURED = {".json", ".jsonl", ".csv", ".tsv"}
SUPPORTED = SUPPORTED_TEXT | SUPPORTED_STRUCTURED | {".xml", ".pdf", ".xlsx"}

QUESTION_KEYS = {"question", "prompt", "instruction", "query", "user"}
ANSWER_KEYS = {"answer", "response", "output", "completion", "assistant"}
TEXT_KEYS = {"text", "content", "body"}
SKIP_KEYS = {"id", "source", "url", "metadata", "index", "uuid"}

VERBOSE_EVERY_DOCUMENTS = 100
PROGRESS_EVERY_BYTES = 64 * 1024 * 1024


def _log(message: str) -> None:
    print(f"[corpus] {message}", flush=True)


def _content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _split_for_hash(content_hash: str) -> str:
    bucket = int(content_hash[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def _assert_allowed_path(path: Path) -> None:
    resolved = path.resolve()
    raw_dir = Path("data/raw").resolve()
    if resolved == raw_dir or raw_dir in resolved.parents:
        raise ValueError(f"data/raw is explicitly excluded: {path}")


def _normalise_record(record: object) -> str:
    if not isinstance(record, dict):
        return str(record) if record is not None else ""

    question_parts: list[str] = []
    answer_parts: list[str] = []
    text_parts: list[str] = []
    other_parts: list[str] = []

    for key, value in record.items():
        if value is None or str(key).lower() in SKIP_KEYS:
            continue
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, ensure_ascii=False)
        else:
            value_text = str(value).strip()
        if not value_text:
            continue

        key_lower = str(key).lower()
        if key_lower in QUESTION_KEYS:
            question_parts.append(value_text)
        elif key_lower in ANSWER_KEYS:
            answer_parts.append(value_text)
        elif key_lower in TEXT_KEYS:
            text_parts.append(value_text)
        else:
            other_parts.append(value_text)

    if question_parts or answer_parts:
        parts = []
        if question_parts:
            parts.append("Question: " + "\n".join(question_parts))
        if answer_parts:
            parts.append("Answer: " + "\n".join(answer_parts))
        return "\n".join(parts)
    if text_parts:
        return "\n".join(text_parts)
    return "\n".join(other_parts)


def _records_from_xml(path: Path):
    root = ET.parse(path).getroot()
    pairs = list(root.iter("QAPair"))
    if pairs:
        for pair in pairs:
            question = (pair.findtext("Question") or "").strip()
            answer = (pair.findtext("Answer") or "").strip()
            if question and answer:
                yield f"Question: {question}\nAnswer: {answer}"
        return

    text = " ".join(t.strip() for t in root.itertext() if t.strip())
    if text:
        yield text


def _records_from_json(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield _normalise_record(json.loads(line))
        return

    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, list):
        for item in payload:
            yield _normalise_record(item)
    elif isinstance(payload, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(payload.get(key), list):
                for item in payload[key]:
                    yield _normalise_record(item)
                return
        yield _normalise_record(payload)
    else:
        yield str(payload)


def _records_from_delimited(path: Path):
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=delimiter):
            yield _normalise_record(row)


def _records_from_xlsx(path: Path):
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(values_only=True)
            try:
                raw_headers = next(rows)
            except StopIteration:
                continue
            headers = [str(value).strip() if value is not None else "" for value in raw_headers]
            for values in rows:
                record = {
                    headers[i]: values[i]
                    for i in range(min(len(headers), len(values)))
                    if headers[i]
                }
                yield _normalise_record(record)
    finally:
        workbook.close()


def _iter_text_documents(path: Path):
    """Yield bounded text chunks instead of loading a giant file into RAM."""
    buffer: list[str] = []
    buffer_chars = 0
    max_chars = 2_000_000

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                if buffer:
                    yield "".join(buffer)
                    buffer.clear()
                    buffer_chars = 0
                continue

            buffer.append(line)
            buffer_chars += len(line)
            if buffer_chars >= max_chars:
                yield "".join(buffer)
                buffer.clear()
                buffer_chars = 0

    if buffer:
        yield "".join(buffer)


def _extract_records(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".xml":
        yield from _records_from_xml(path)
        return

    if suffix in {".json", ".jsonl"}:
        yield from _records_from_json(path)
        return

    if suffix in {".csv", ".tsv"}:
        yield from _records_from_delimited(path)
        return

    if suffix == ".xlsx":
        yield from _records_from_xlsx(path)
        return

    if suffix == ".pdf":
        text = extract_pdf_text(path)
        if text.strip():
            for part in re.split(r"\n\s*\n+", text):
                if part.strip():
                    yield part
        return

    if suffix in SUPPORTED_TEXT:
        yield from _iter_text_documents(path)
        return


def _iter_source_files(root: Path):
    _assert_allowed_path(root)
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")

    if root.is_file():
        if root.suffix.lower() in SUPPORTED:
            yield root
        else:
            _log(f"SKIP unsupported source file: {root}")
        return

    files = sorted(root.rglob("*"))
    total_files = sum(path.is_file() for path in files)
    supported_files = sum(path.is_file() and path.suffix.lower() in SUPPORTED for path in files)
    unsupported_files = total_files - supported_files
    total_bytes = sum(path.stat().st_size for path in files if path.is_file())
    supported_bytes = sum(path.stat().st_size for path in files if path.is_file() and path.suffix.lower() in SUPPORTED)

    _log(f"SOURCE SCAN: {root}")
    _log(f"  files found       : {total_files:,}")
    _log(f"  supported files   : {supported_files:,}")
    _log(f"  unsupported files : {unsupported_files:,}")
    _log(f"  total disk size   : {total_bytes / (1024**3):.3f} GiB")
    _log(f"  supported size    : {supported_bytes / (1024**3):.3f} GiB")

    if unsupported_files:
        _log("  unsupported extensions:")
        counts: dict[str, tuple[int, int]] = {}
        for path in files:
            if path.is_file() and path.suffix.lower() not in SUPPORTED:
                ext = path.suffix.lower() or "<no extension>"
                old_count, old_bytes = counts.get(ext, (0, 0))
                counts[ext] = (old_count + 1, old_bytes + path.stat().st_size)
        for ext, (count, size) in sorted(counts.items(), key=lambda item: item[1][1], reverse=True):
            _log(f"    {ext}: {count:,} files, {size / (1024**3):.3f} GiB")

    for path in files:
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            _assert_allowed_path(path)
            yield path


def _clean_text_with_repair(text: str) -> str:
    """Clean text and repair common UTF-8-as-Windows-1252 mojibake when detected."""
    if "â" in text or "Ã" in text or "Â" in text:
        try:
            repaired = text.encode("latin1").decode("utf-8")
            if sum(ord(c) > 127 for c in repaired) >= sum(ord(c) > 127 for c in text):
                text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return clean_text(text)


def _iter_clean_documents(sources: list[Path], old_train: Path):
    _assert_allowed_path(old_train)
    if not old_train.exists():
        raise FileNotFoundError(f"Original processed training corpus not found: {old_train}")

    _log(f"OLD CORPUS: {old_train}")
    old_size = old_train.stat().st_size
    _log(f"  disk size: {old_size / (1024**2):.2f} MiB")
    _log("  processing already-prepared train.txt...")
    for index, text in enumerate(_iter_text_documents(old_train)):
        yield f"old-train:{index}", _clean_text_with_repair(text)

    for source in sources:
        _log(f"BEGIN SOURCE: {source}")
        file_number = 0
        for path in _iter_source_files(source):
            file_number += 1
            file_size = path.stat().st_size
            _log(f"FILE {file_number}: {path} | {file_size / (1024**2):.2f} MiB | {path.suffix.lower()}")
            try:
                record_count = 0
                for index, text in enumerate(_extract_records(path)):
                    if text:
                        record_count += 1
                        yield f"new:{path}:{index}", _clean_text_with_repair(text)
                _log(f"  extracted records/chunks: {record_count:,}")
            except Exception as exc:
                _log(f"  EXTRACTION FAILURE: {type(exc).__name__}: {exc}")
                yield f"__EXTRACTION_FAILURE__:{path}", ""
        _log(f"END SOURCE: {source} | processed files: {file_number:,}")


def build_corpus(sources: list[Path], old_train: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _log("=" * 72)
    _log("CONTINUED PRETRAINING CORPUS BUILD")
    _log("=" * 72)
    _log(f"Old corpus       : {old_train}")
    _log(f"New sources      : {', '.join(map(str, sources))}")
    _log(f"Output directory : {output_dir}")
    _log("EXPLICIT EXCLUSION: data/raw will NOT be read")
    _log(f"Supported formats: {', '.join(sorted(SUPPORTED))}")

    for source in sources:
        _assert_allowed_path(source)
    _assert_allowed_path(old_train)

    # Remove previous generated outputs so a failed/interrupted build cannot leave
    # a misleading mixture of old and new data.
    for name in ("train.txt", "validation.txt", "test.txt", "manifest.jsonl"):
        target = output_dir / name
        if target.exists():
            _log(f"REMOVE OLD OUTPUT: {target} ({target.stat().st_size / (1024**2):.2f} MiB)")
            target.unlink()

    output_files = {
        split: (output_dir / f"{split}.txt").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    manifest_path = output_dir / "manifest.jsonl"

    stats = {
        "old_train_documents": 0,
        "source_files": 0,
        "source_records": 0,
        "accepted": 0,
        "rejected": 0,
        "duplicates": 0,
        "near_duplicates": 0,
        "extraction_failures": 0,
        "train_documents": 0,
        "validation_documents": 0,
        "test_documents": 0,
        "accepted_chars": 0,
    }

    seen_hashes: set[str] = set()
    seen_simhashes: list[int] = []
    start = time.time()
    last_progress_bytes = 0
    processed_documents = 0

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            current_source = None

            for source_id, raw_text in _iter_clean_documents(sources, old_train):
                processed_documents += 1
                if source_id.startswith("__EXTRACTION_FAILURE__:"):
                    stats["extraction_failures"] += 1
                    continue

                if source_id.startswith("old-train:"):
                    stats["old_train_documents"] += 1
                else:
                    stats["source_records"] += 1
                    source_name = source_id.rsplit(":", 1)[0]
                    if source_name != current_source:
                        current_source = source_name
                        stats["source_files"] += 1

                try:
                    cleaned = raw_text
                    if not passes_basic_filters(cleaned):
                        stats["rejected"] += 1
                        if processed_documents <= 20:
                            _log(f"REJECT FILTER: {source_id}")
                        continue

                    content_hash = _content_hash(cleaned)
                    if content_hash in seen_hashes:
                        stats["duplicates"] += 1
                        if processed_documents <= 20:
                            _log(f"REJECT DUPLICATE: {source_id}")
                        continue
                    seen_hashes.add(content_hash)

                    fingerprint = simhash(cleaned)
                    if is_near_duplicate(fingerprint, seen_simhashes):
                        stats["near_duplicates"] += 1
                        if processed_documents <= 20:
                            _log(f"REJECT NEAR-DUPLICATE: {source_id}")
                        continue
                    seen_simhashes.append(fingerprint)
                except Exception as exc:
                    stats["extraction_failures"] += 1
                    _log(f"PROCESSING FAILURE: {source_id}: {type(exc).__name__}: {exc}")
                    continue

                split = "train" if source_id.startswith("old-train:") else _split_for_hash(content_hash)
                output_files[split].write(cleaned + "\n\n")
                stats[f"{split}_documents"] += 1
                stats["accepted"] += 1
                stats["accepted_chars"] += len(cleaned)
                manifest.write(
                    json.dumps(
                        {"source": source_id, "content_sha256": content_hash, "split": split},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                if processed_documents <= 20 or processed_documents % VERBOSE_EVERY_DOCUMENTS == 0:
                    elapsed = max(time.time() - start, 0.001)
                    _log(
                        f"PROGRESS docs={processed_documents:,} accepted={stats['accepted']:,} "
                        f"rejected={stats['rejected']:,} dup={stats['duplicates']:,} "
                        f"near_dup={stats['near_duplicates']:,} "
                        f"accepted_text={stats['accepted_chars'] / (1024**2):.2f} MiB "
                        f"rate={processed_documents / elapsed:.1f} docs/s"
                    )

                output_size = sum(
                    (output_dir / f"{split}.txt").stat().st_size
                    for split in ("train", "validation", "test")
                )
                if output_size - last_progress_bytes >= PROGRESS_EVERY_BYTES:
                    last_progress_bytes = output_size
                    _log(f"OUTPUT SIZE: {output_size / (1024**2):.2f} MiB")
    finally:
        for handle in output_files.values():
            handle.close()

    elapsed = max(time.time() - start, 0.001)
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["output_bytes"] = sum(
        (output_dir / f"{split}.txt").stat().st_size
        for split in ("train", "validation", "test")
    )

    _log("=" * 72)
    _log("BUILD COMPLETE")
    _log("=" * 72)
    for key, value in stats.items():
        if key.endswith("_seconds"):
            _log(f"{key:24s}: {value}")
        elif key.endswith("_bytes"):
            _log(f"{key:24s}: {value / (1024**2):.2f} MiB")
        elif key == "accepted_chars":
            _log(f"{key:24s}: {value:,} chars ({value / (1024**2):.2f} MiB UTF-8 approx.)")
        else:
            _log(f"{key:24s}: {value:,}")
    _log(f"Output directory: {output_dir.resolve()}")
    _log("=" * 72)
    return stats


def main(args=None):
    parser = argparse.ArgumentParser(description="Build the continued-pretraining corpus with verbose logging.")
    parser.add_argument("--source", dest="sources", action="append", type=Path)
    parser.add_argument("--old-train", type=Path, default=DEFAULT_OLD_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parsed = parser.parse_args(args)
    sources = parsed.sources or list(DEFAULT_SOURCES)
    build_corpus(sources, parsed.old_train, parsed.output_dir)


if __name__ == "__main__":
    main()
