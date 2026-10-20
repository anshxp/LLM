"""Download the complete MedRAG PubMed and scientific_papers PubMed streams to Drive.

This is an exhaustive data-preparation job, not a training command. It writes resumable
Parquet shards and never intentionally accumulates the corpus in RAM. Free Colab sessions
may need many reconnects; resuming a stream can take time because rows are skipped again.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

SCHEMA = pa.schema([("text", pa.string()), ("source", pa.string())])


def normalize(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value if x is not None)
    return re.sub(r"\s+", " ", str(value)).strip()


def record_text(row: dict, source: str) -> str:
    if source == "medrag_pubmed":
        title = normalize(row.get("title"))
        content = normalize(row.get("content") or row.get("contents") or row.get("text"))
        return "\n".join(part for part in (title, content) if part)
    abstract = normalize(row.get("abstract"))
    article = normalize(row.get("article"))
    sections = normalize(row.get("section_names"))
    # Avoid appending an article field that already contains the abstract verbatim.
    body = article if article and article not in abstract else ""
    return "\n".join(part for part in (abstract, sections, body) if part)


def open_stream(source: str):
    if source == "medrag_pubmed":
        return load_dataset("MedRAG/pubmed", split="train", streaming=True)
    # The legacy dataset script is unsupported by recent versions of `datasets`.
    # Use the Hugging Face-generated Parquet revision instead.
    return load_dataset(
        "armanc/scientific_papers",
        "pubmed",
        split="train",
        streaming=True,
        revision="refs/convert/parquet",
    )


def write_shard(rows, out: Path):
    temp = out.with_suffix(out.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temp, compression="zstd")
    os.replace(temp, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/medical_corpus_full"))
    parser.add_argument("--shard-rows", type=int, default=100)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--merge", action="store_true", help="After both sources finish, merge shards into one Parquet file (requires additional Drive space)")
    args = parser.parse_args()
    if args.shard_rows <= 0 or args.min_chars <= 0:
        parser.error("--shard-rows and --min-chars must be positive")

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "version": 2, "mode": "full", "sources": {}, "status": "in_progress"
    }
    if manifest.get("mode") != "full":
        raise RuntimeError("Output directory contains a non-full manifest; use a fresh output directory.")

    for source in ("medrag_pubmed", "scientific_papers_pubmed"):
        state = manifest["sources"].get(source, {"consumed": 0, "shards": 0, "complete": False, "usable_rows": 0})
        if state.get("complete"):
            print(f"{source}: already complete; consumed={state['consumed']:,}, usable={state['usable_rows']:,}")
            continue
        stream = open_stream(source)
        consumed = int(state["consumed"])
        if consumed:
            print(f"{source}: resuming after {consumed:,} source rows; stream skipping may take a while")
            stream = stream.skip(consumed)
        shard_idx = int(state["shards"])
        usable = int(state.get("usable_rows", 0))
        buffer = []
        started = time.time()
        for row in stream:
            consumed += 1
            text = record_text(row, source)
            if len(text) >= args.min_chars:
                buffer.append({"text": text, "source": source})
                usable += 1
            if len(buffer) >= args.shard_rows:
                write_shard(buffer, outdir / f"{source}_{shard_idx:06d}.parquet")
                shard_idx += 1
                buffer.clear()
                state.update({"consumed": consumed, "shards": shard_idx, "usable_rows": usable, "complete": False})
                manifest["sources"][source] = state
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"{source}: source_rows={consumed:,}; usable_rows={usable:,}; shards={shard_idx:,}")
        # Reaching natural stream exhaustion means the complete split was traversed.
        if buffer:
            write_shard(buffer, outdir / f"{source}_{shard_idx:06d}.parquet")
            shard_idx += 1
        state.update({"consumed": consumed, "shards": shard_idx, "usable_rows": usable, "complete": True})
        manifest["sources"][source] = state
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{source}: COMPLETE; source_rows={consumed:,}; usable_rows={usable:,}; shards={shard_idx:,}; elapsed={time.time()-started:.1f}s")

    manifest["status"] = "download_complete" if all(s.get("complete") for s in manifest["sources"].values()) and len(manifest["sources"]) == 2 else "in_progress"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.merge:
        print(f"All-source status: {manifest['status']}. Shards are in {outdir}. Re-run with --merge after both streams complete.")
        return
    if manifest["status"] != "download_complete":
        raise RuntimeError("Cannot merge until both full streams have reached natural exhaustion.")
    output = outdir / "medical_pretrain.parquet"
    temp = output.with_suffix(".parquet.tmp")
    writer = None
    shard_paths = sorted(p for p in outdir.glob("*.parquet") if p.name != output.name)
    try:
        for path in shard_paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=4096):
                if writer is None:
                    writer = pq.ParquetWriter(temp, parquet.schema_arrow, compression="zstd")
                writer.write_batch(batch)
        if writer is None:
            raise RuntimeError("No data shards found to merge.")
        writer.close()
        writer = None
        os.replace(temp, output)
    finally:
        if writer is not None:
            writer.close()
    manifest["status"] = "merged"
    manifest["merged_file"] = str(output)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Merged {len(shard_paths):,} shards to {output}; size={output.stat().st_size / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
