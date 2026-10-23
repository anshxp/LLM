"""Download Common Pile PubMed as resumable Parquet shards; no topic filtering."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

DATASET_ID = "common-pile/pubmed"
SPLIT = "train"
SCHEMA = pa.schema([
    ("id", pa.string()), ("text", pa.string()), ("source", pa.string()),
    ("added", pa.string()), ("created", pa.string()), ("license", pa.string()),
    ("source_url", pa.string()), ("journal", pa.string()),
    ("authors_json", pa.string()), ("metadata_json", pa.string()),
])


def as_text(value):
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def convert(row):
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"original_metadata": metadata}
    return {
        "id": as_text(row.get("id")),
        "text": text,
        "source": as_text(row.get("source")),
        "added": as_text(row.get("added")),
        "created": as_text(row.get("created")),
        "license": as_text(metadata.get("license")),
        "source_url": as_text(metadata.get("url")),
        "journal": as_text(metadata.get("journal")),
        "authors_json": json.dumps(metadata.get("authors"), ensure_ascii=False) if metadata.get("authors") is not None else None,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_shard(rows, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression="zstd")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/common_pile_pubmed"))
    parser.add_argument("--shard-rows", type=int, default=1000)
    parser.add_argument("--max-source-rows", type=int, default=None, help="Test limit; omit for full ingestion")
    parser.add_argument("--log-every-rows", type=int, default=1000)
    args = parser.parse_args()
    if args.shard_rows < 1 or args.log_every_rows < 1:
        parser.error("--shard-rows and --log-every-rows must be positive")
    if args.max_source_rows is not None and args.max_source_rows < 1:
        parser.error("--max-source-rows must be positive")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != DATASET_ID or manifest.get("split") != SPLIT:
            raise RuntimeError("Manifest dataset/split mismatch; choose a new output directory.")
        if manifest.get("status") == "complete":
            print("Dataset stream already complete.", flush=True)
            return
    else:
        manifest = {
            "format_version": 1, "dataset_id": DATASET_ID, "split": SPLIT,
            "source_rows_consumed": 0, "usable_rows_written": 0,
            "next_shard_index": 0, "shards": [], "status": "in_progress",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    consumed_before = int(manifest["source_rows_consumed"])
    shard_index = int(manifest["next_shard_index"])
    usable_total = int(manifest["usable_rows_written"])
    print(f"Dataset: {DATASET_ID} split={SPLIT}", flush=True)
    print(f"Output: {out}; resuming after {consumed_before:,} source rows", flush=True)

    stream = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    if consumed_before:
        stream = stream.skip(consumed_before)

    buffer, rows_this_run, last_log = [], 0, 0
    start = time.monotonic()
    naturally_exhausted = True
    for row in stream:
        rows_this_run += 1
        record = convert(row)
        if record is not None:
            buffer.append(record)

        hit_limit = args.max_source_rows is not None and rows_this_run >= args.max_source_rows
        flush = len(buffer) >= args.shard_rows or hit_limit
        if flush and buffer:
            name = f"part-{shard_index:06d}.parquet"
            path = out / name
            count = len(buffer)
            write_shard(buffer, path)
            manifest["shards"].append({"file": name, "rows": count, "size_bytes": path.stat().st_size})
            shard_index += 1
            usable_total += count
            buffer = []

        if rows_this_run - last_log >= args.log_every_rows or hit_limit:
            elapsed = max(time.monotonic() - start, 0.001)
            print(f"source_rows={consumed_before + rows_this_run:,} | usable_rows={usable_total:,} | "
                  f"shards={shard_index:,} | rows_per_sec={rows_this_run / elapsed:.2f}", flush=True)
            last_log = rows_this_run

        manifest["source_rows_consumed"] = consumed_before + rows_this_run
        manifest["usable_rows_written"] = usable_total
        manifest["next_shard_index"] = shard_index
        manifest["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "in_progress"
        # Persist progress periodically even if a shard has not filled yet.
        if rows_this_run % args.log_every_rows == 0 or hit_limit:
            atomic_json(manifest_path, manifest)

        if hit_limit:
            naturally_exhausted = False
            break

    if buffer:
        name = f"part-{shard_index:06d}.parquet"
        path = out / name
        count = len(buffer)
        write_shard(buffer, path)
        manifest["shards"].append({"file": name, "rows": count, "size_bytes": path.stat().st_size})
        shard_index += 1
        usable_total += count

    manifest["source_rows_consumed"] = consumed_before + rows_this_run
    manifest["usable_rows_written"] = usable_total
    manifest["next_shard_index"] = shard_index
    manifest["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    if naturally_exhausted:
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    print(f"Status: {manifest['status']}; source rows: {manifest['source_rows_consumed']:,}; "
          f"usable rows: {usable_total:,}; shards: {shard_index:,}; "
          f"elapsed: {time.monotonic() - start:.1f}s", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
