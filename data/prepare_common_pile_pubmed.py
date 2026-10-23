"""Download Common Pile PubMed into resumable Parquet shards; no topic filtering."""
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


def to_text(value):
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
        "id": to_text(row.get("id")), "text": text,
        "source": to_text(row.get("source")), "added": to_text(row.get("added")),
        "created": to_text(row.get("created")),
        "license": to_text(metadata.get("license")),
        "source_url": to_text(metadata.get("url")),
        "journal": to_text(metadata.get("journal")),
        "authors_json": json.dumps(metadata.get("authors"), ensure_ascii=False) if metadata.get("authors") is not None else None,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def save_manifest(path, manifest):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def write_shard(rows, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temp, compression="zstd")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/common_pile_pubmed"))
    parser.add_argument("--shard-rows", type=int, default=1000)
    parser.add_argument("--max-source-rows", type=int, default=None, help="Limit source rows for a test; omit for full stream")
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
            raise RuntimeError("Manifest dataset/split mismatch; use a new output directory.")
        if manifest.get("status") == "complete":
            print("Dataset stream is already complete.", flush=True)
            return
    else:
        manifest = {
            "format_version": 1, "dataset_id": DATASET_ID, "split": SPLIT,
            "source_rows_consumed": 0, "usable_rows_written": 0,
            "next_shard_index": 0, "shards": [], "status": "in_progress",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    committed = int(manifest["source_rows_consumed"])
    usable_total = int(manifest["usable_rows_written"])
    shard_index = int(manifest["next_shard_index"])
    print(f"Dataset: {DATASET_ID}, split={SPLIT}", flush=True)
    print(f"Output: {out}; resuming from committed source row {committed:,}", flush=True)

    stream = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    if committed:
        stream = stream.skip(committed)

    buffer = []
    run_rows = 0
    pending_source_rows = 0
    start = time.monotonic()
    last_log = 0
    stopped_at_limit = False

    for row in stream:
        run_rows += 1
        pending_source_rows += 1
        record = convert(row)
        if record is not None:
            buffer.append(record)

        hit_limit = args.max_source_rows is not None and run_rows >= args.max_source_rows
        if len(buffer) >= args.shard_rows or hit_limit:
            if buffer:
                filename = f"part-{shard_index:06d}.parquet"
                target = out / filename
                row_count = len(buffer)
                write_shard(buffer, target)
                shard_index += 1
                usable_total += row_count
                manifest["shards"].append({
                    "file": filename, "rows": row_count, "size_bytes": target.stat().st_size,
                })
                buffer = []

            # Persist the source cursor only after all text from that range is safely on disk.
            committed += pending_source_rows
            pending_source_rows = 0
            manifest["source_rows_consumed"] = committed
            manifest["usable_rows_written"] = usable_total
            manifest["next_shard_index"] = shard_index
            manifest["status"] = "in_progress"
            manifest["last_updated_at"] = datetime.now(timezone.utc).isoformat()
            save_manifest(manifest_path, manifest)

        total_seen = int(manifest["source_rows_consumed"]) + pending_source_rows
        if total_seen - last_log >= args.log_every_rows or hit_limit:
            elapsed = max(time.monotonic() - start, 0.001)
            print(
                f"source_rows_seen={committed + pending_source_rows:,} | committed_source_rows={committed:,} | "
                f"usable_rows={usable_total:,} | shards={shard_index:,} | "
                f"elapsed={elapsed:.1f}s | rows_per_sec={run_rows / elapsed:.2f}",
                flush=True,
            )
            last_log = total_seen

        if hit_limit:
            stopped_at_limit = True
            break

    if buffer:
        filename = f"part-{shard_index:06d}.parquet"
        target = out / filename
        row_count = len(buffer)
        write_shard(buffer, target)
        shard_index += 1
        usable_total += row_count
        manifest["shards"].append({
            "file": filename, "rows": row_count, "size_bytes": target.stat().st_size,
        })
        buffer = []

    # Commit any trailing source rows after stream completion or an intentional test cap.
    if pending_source_rows:
        committed += pending_source_rows

    manifest["source_rows_consumed"] = committed
    manifest["usable_rows_written"] = usable_total
    manifest["next_shard_index"] = shard_index
    manifest["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    if not stopped_at_limit:
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest_path, manifest)

    print(
        f"Status={manifest['status']} | source_rows_committed={committed:,} | "
        f"usable_rows={usable_total:,} | shards={shard_index:,} | "
        f"elapsed={time.monotonic() - start:.1f}s",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
