"""Stream MedRAG PubMed and scientific_papers/PubMed into resumable Parquet shards.

Run in Colab after mounting Google Drive. The script never loads the full corpus into RAM.
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


def normalize(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value if x is not None)
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text


def record_text(row: dict, source: str) -> str:
    if source == "medrag_pubmed":
        title = normalize(row.get("title"))
        content = normalize(row.get("content") or row.get("contents") or row.get("text"))
        return "\n".join(part for part in (title, content) if part)
    abstract = normalize(row.get("abstract"))
    article = normalize(row.get("article"))
    sections = normalize(row.get("section_names"))
    # Abstract first; article text adds context. Avoid duplicating the abstract when
    # the source already embeds it in the article string.
    body = article if article and article not in abstract else ""
    return "\n".join(part for part in (abstract, sections, body) if part)


def source_stream(source: str):
    if source == "medrag_pubmed":
        return load_dataset("MedRAG/pubmed", split="train", streaming=True)
    return load_dataset("armanc/scientific_papers", "pubmed", split="train", streaming=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/medical_corpus"))
    p.add_argument("--medrag-records", type=int, default=20000)
    p.add_argument("--scientific-papers-records", type=int, default=10000)
    p.add_argument("--shard-rows", type=int, default=1000)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--merge", action="store_true", help="Merge completed source shards into medical_pretrain.parquet")
    args = p.parse_args()
    if min(args.medrag_records, args.scientific_papers_records, args.shard_rows, args.min_chars) <= 0:
        p.error("record limits, shard-rows, and min-chars must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "version": 1, "sources": {}, "status": "in_progress"
    }
    limits = {"medrag_pubmed": args.medrag_records, "scientific_papers_pubmed": args.scientific_papers_records}

    for source, limit in limits.items():
        state = manifest["sources"].get(source, {"consumed": 0, "shards": 0, "complete": False})
        if state.get("complete"):
            print(f"{source}: already complete ({state['consumed']:,} records); skipping")
            continue
        stream_name = "medrag_pubmed" if source == "medrag_pubmed" else "scientific_papers_pubmed"
        stream = source_stream(stream_name)
        consumed = int(state["consumed"])
        if consumed:
            print(f"{source}: resuming after {consumed:,} source rows (stream skip may take time)")
            stream = stream.skip(consumed)
        buffer = []
        shard_index = int(state["shards"])
        started = time.time()
        for row in stream:
            consumed += 1
            text = record_text(row, stream_name)
            if len(text) >= args.min_chars:
                buffer.append({"text": text, "source": source})
            if len(buffer) >= args.shard_rows:
                out = args.output_dir / f"{source}_{shard_index:05d}.parquet"
                temp = out.with_suffix(".parquet.tmp")
                table = pa.Table.from_pylist(buffer, schema=pa.schema([
                    ("text", pa.string()), ("source", pa.string())
                ]))
                pq.write_table(table, temp, compression="zstd")
                os.replace(temp, out)
                shard_index += 1
                buffer.clear()
                state.update({"consumed": consumed, "shards": shard_index, "complete": False})
                manifest["sources"][source] = state
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f"{source}: consumed={consumed:,}/{limit:,} rows, shards={shard_index}")
            if consumed >= limit:
                break
        if buffer:
            out = args.output_dir / f"{source}_{shard_index:05d}.parquet"
            temp = out.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist(buffer, schema=pa.schema([
                ("text", pa.string()), ("source", pa.string())
            ])), temp, compression="zstd")
            os.replace(temp, out)
            shard_index += 1
        state.update({"consumed": consumed, "shards": shard_index, "complete": consumed >= limit})
        manifest["sources"][source] = state
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"{source}: finished in {time.time() - started:.1f}s; source rows consumed={consumed:,}")

    if args.merge:
        paths = sorted(args.output_dir.glob("*.parquet"))
        paths = [path for path in paths if path.name != "medical_pretrain.parquet"]
        if not paths:
            raise RuntimeError("No Parquet shards found to merge")
        output = args.output_dir / "medical_pretrain.parquet"
        temp = output.with_suffix(".parquet.tmp")
        writer = None
        try:
            for path in paths:
                table = pq.read_table(path)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                writer.write_table(table)
            if writer is not None:
                writer.close()
                writer = None
            os.replace(temp, output)
        finally:
            if writer is not None:
                writer.close()
        manifest["status"] = "merged"
        manifest["merged_file"] = str(output)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Merged {len(paths)} shards -> {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    else:
        print("Shard download complete/in progress. Re-run with --merge after both source limits are reached.")


if __name__ == "__main__":
    main()
